// Simple test case of implementing a trait with super-builtin-kinds.

// pretty-expanded FIXME #23616

trait Foo : Send { }

impl Foo for isize { }

pub fn main() { }
