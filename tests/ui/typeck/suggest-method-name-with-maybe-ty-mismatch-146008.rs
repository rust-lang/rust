struct LlamaModel;

impl LlamaModel {
    fn chat_template(&self) -> Result<&str, ()> {
        todo!()
    }
}

fn template_from_str(_x: &str) {}

fn main() {
    let model = LlamaModel;
    template_from_str(&model.chat_template); //~ ERROR attempted to take value of method `chat_template` on type `LlamaModel`
}
