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
        //~^ WARNING floating-point types cannot be used in patterns
        //~| WARNING this was previously accepted by the compiler but is being phased out
        _ => { }
    }
}
