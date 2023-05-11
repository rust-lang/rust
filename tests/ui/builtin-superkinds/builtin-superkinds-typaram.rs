// run-pass
// Tests correct implementation of traits with super-builtin-kinds
// using a bounded type parameter.

// pretty-expanded FIXME #23616

trait Foo : Send { }

impl <T: Send> Foo for T { }

pub fn main() { }
