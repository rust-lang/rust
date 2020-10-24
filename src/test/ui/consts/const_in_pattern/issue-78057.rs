#![deny(unreachable_patterns)]

#[derive(PartialEq)]
struct Opaque(i32);

impl Eq for Opaque {}

const FOO: Opaque = Opaque(42);

fn main() {
    match FOO {
        FOO => {},
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        _ => {}
        //~^ ERROR unreachable pattern
    }
}
