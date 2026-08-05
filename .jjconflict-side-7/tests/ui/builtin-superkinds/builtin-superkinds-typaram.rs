//@ check-pass
// Tests correct implementation of traits with super-builtin-kinds
// using a bounded type parameter.


trait Foo : Send { }

impl <T: Send> Foo for T { }

pub fn main() { }
