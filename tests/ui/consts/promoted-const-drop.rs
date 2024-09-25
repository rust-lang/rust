#![feature(const_trait_impl)]

struct A();

impl const Drop for A {
    //~^ ERROR const `impl` for trait `Drop` which is not marked with `#[const_trait]`
    fn drop(&mut self) {}
}

const C: A = A();

fn main() {
    let _: &'static A = &A(); //~ ERROR temporary value dropped while borrowed
    let _: &'static [A] = &[C]; //~ ERROR temporary value dropped while borrowed
}
