use std::env;

pub struct Foo {
    text: String
}

pub fn foo() -> Foo {
    let args: Vec<String> = env::args().collect();
    let text = args[1].clone();

    pub Foo { text }
}
//~^^ ERROR missing `struct` for struct definition

fn main() {}
