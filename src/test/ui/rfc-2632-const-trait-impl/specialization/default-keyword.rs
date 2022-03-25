// check-pass

#![feature(const_trait_impl)]
#![feature(min_specialization)]

trait Foo {
    fn foo();
}

impl const Foo for u32 {
    default fn foo() {}
}

fn main() {}
