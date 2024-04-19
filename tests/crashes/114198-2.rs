//@ known-bug: #114198
//@ compile-flags: -Zprint-mono-items=eager

impl Trait for <Ty as Owner>::Struct {}
trait Trait {
    fn test(&self) {}
}

enum Ty {}
trait Owner { type Struct: ?Sized; }
impl Owner for Ty {
    type Struct = dyn Trait + Send;
}

fn main() {}
