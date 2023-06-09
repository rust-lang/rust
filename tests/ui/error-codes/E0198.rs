#![feature(negative_impls)]

struct Foo;

unsafe impl !Send for Foo { } //~ ERROR E0198

fn main() {
}
