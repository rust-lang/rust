#![feature(const_trait_impl)]
#![feature(effects)]
#![feature(inline_const)]

#[const_trait]
trait Foo {
    fn bar(&self);
}

fn foo<T: Foo<()>>() {}
//~^ ERROR this trait takes 0 generic arguments but 1 generic argument was supplied

fn bar(x: &dyn Foo) {
    x.bar();
}

fn main() {
}
