//@ check-pass

#![feature(final_associated_functions)]

trait Foo {
    final fn bar(&self) {}
}

impl Foo for () {}

fn main() {
    ().bar();
}
