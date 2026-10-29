// Test that disallow lifetime parameters that are unused.

use std::marker;

struct Bivariant<'a>; //~ ERROR parameter `'a` is never used

struct Struct<'a, 'd> { //~ ERROR parameter `'d` is never used
    field: &'a [i32]
}

trait Trait<'a, 'd> { // OK on traits
    fn method(&'a self);
}

fn main() {}
