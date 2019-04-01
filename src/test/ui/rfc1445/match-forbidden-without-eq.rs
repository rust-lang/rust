use std::f32;

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

    let x = 0.0;
    match x {
        f32::INFINITY => { }
        //~^ WARN floating-point types cannot be used in patterns
        //~| WARN this was previously accepted
        //~| WARN will become a hard error in a future release
        _ => { }
    }
}
