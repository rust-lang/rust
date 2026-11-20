#![feature(negative_impls)]

use std::marker::Sync;

struct Foo;
impl !Sync for Foo {}

static FOO: usize = 3;
static BAR: Foo = Foo;
//~^ ERROR: `Foo` cannot be shared between threads safely [E0277]

fn main() {}
