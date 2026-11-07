#![warn(clippy::manual_let_else)]

enum Foo {
    A(u8),
    B,
}

fn main() {
    let x = match Foo::A(1) {
        //~^ ERROR: this could be rewritten as `let...else`
        Foo::A(x) => x,
        Foo::B => return,
    };
}
