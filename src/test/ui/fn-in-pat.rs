struct A {}

impl A {
    fn new() {}
}

fn hof<F>(_: F) where F: FnMut(()) {}

fn ice() {
    hof(|c| match c {
        A::new() => (), //~ ERROR expected tuple struct/variant, found method
        _ => ()
    })
}

fn main() {}
