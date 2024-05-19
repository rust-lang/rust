//@ build-pass
//@ compile-flags: -Znext-solver

// Tests that the specializing impl `<() as Foo>` holds during codegen.

#![feature(min_specialization)]

trait Foo {
    fn bar();
}

impl<T> Foo for T {
    default fn bar() {}
}

impl Foo for () {
    fn bar() {}
}

fn main() {
    <() as Foo>::bar();
}
