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

pub fn bar() -> Foo {
    fn
    Foo { text: "".to_string() }
}
//~^^ ERROR expected one of `(` or `<`, found `{`

fn main() {}
