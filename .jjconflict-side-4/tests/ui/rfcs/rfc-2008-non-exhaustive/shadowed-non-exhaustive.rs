struct Shadowed {}

fn main() {
    let v = Shadowed::Foo;

    match v {
       //~^ ERROR: non-exhaustive patterns: `Shadowed::Foo` not covered [E0004]
    }

    enum Shadowed {
        Foo,
    }
}
