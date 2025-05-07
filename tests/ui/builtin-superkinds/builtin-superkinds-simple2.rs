//@ check-pass
// Simple test case of implementing a trait with super-builtin-kinds.


trait Foo : Send { }

impl Foo for isize { }

pub fn main() { }
