#[derive(PartialEq)]
struct Foo {
    x: u32
}

const FOO: Foo = Foo { x: 0 };

fn main() {
    let y = Foo { x: 1 };
    match y {
        FOO => { }
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        _ => { }
    }
}
