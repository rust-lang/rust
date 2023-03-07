// Enormous types are allowed if they are never actually instantiated.
// run-pass
trait Foo {
    type Assoc;
}

impl Foo for [u16; usize::MAX] {
    type Assoc = u32;
}

fn main() {
    let _a: Option<<[u16; usize::MAX] as Foo>::Assoc> = None;
}
