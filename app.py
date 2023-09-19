import gradio as gr

from diffusion_webui import (
    StableDiffusionControlNetGenerator,
    StableDiffusionControlNetInpaintGenerator,
    StableDiffusionImage2ImageGenerator,
    StableDiffusionInpaintGenerator,
    StableDiffusionText2ImageGenerator,
)


def diffusion_app():
    app = gr.Blocks()
    with app:
        gr.HTML(
        """
        <h1 style='text-align: center'>
       Stable Diffusion + ControlNet + Inpaint
        </h1>
        """
    )
        gr.HTML(
            """
            <h3 style='text-align: center'>
            Follow me for more! 
            <a href='https://twitter.com/kadirnar_ai' target='_blank'>Twitter</a> | <a href='https://github.com/kadirnar' target='_blank'>Github</a> | <a href='https://www.linkedin.com/in/kadir-nar/' target='_blank'>Linkedin</a>
            </h3>
            """
    )
        with gr.Row():
            with gr.Column():
                with gr.Tab(label="Text2Image"):
                    StableDiffusionText2ImageGenerator.app()
                with gr.Tab(label="Image2Image"):
                    StableDiffusionImage2ImageGenerator.app()
                with gr.Tab(label="Inpaint"):
                    StableDiffusionInpaintGenerator.app()
                with gr.Tab(label="Controlnet"):
                    StableDiffusionControlNetGenerator.app()
                with gr.Tab(label="Controlnet Inpaint"):
                    StableDiffusionControlNetInpaintGenerator.app()

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True)


if __name__ == "__main__":
    diffusion_app()
