#![feature(tool_attributes)]
enum Foo {
    A(usize),
    B
}

fn main() {
    let x = Foo::A;
    let y = x as i32;
}
