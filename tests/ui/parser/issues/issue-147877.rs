// A malformed struct literal in an `if` condition should mention the ambiguity.

fn main() {
    let foo = Foo { x: 3 };
    let x = 3;
    let _ = if foo == Foo { x } { x } else { 0 };
    //~^ ERROR expected one of
}

#[derive(Eq, PartialEq)]
struct Foo {
    x: u32,
}
