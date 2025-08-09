struct Shadowed {}

fn main() {
    let v = Shadowed::Foo;

    match v {
        //~^ non-exhaustive patterns: `main::Shadowed::Foo` not covered [E0004]
    }

    enum Shadowed {
        Foo,
    }
}
