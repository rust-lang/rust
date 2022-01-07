#![feature(more_qualified_paths)]

struct Struct {}

trait Trait {
    type Type;
}

impl Trait for Struct {
    type Type = Self;
}

fn main() {
    // keep the qualified path details
    let _ = <Struct as Trait>::Type {};
}
