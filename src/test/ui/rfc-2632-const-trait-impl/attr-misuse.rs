#![feature(const_trait_impl)]

#[const_trait]
trait A {
    #[const_trait] //~ ERROR attribute should be applied
    fn foo(self);
}

#[const_trait] //~ ERROR attribute should be applied
fn main() {}
//~^ ERROR `main` function is not allowed to have generic parameters
