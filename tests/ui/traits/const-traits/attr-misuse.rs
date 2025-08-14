#![feature(const_trait_impl)]

#[const_trait]
trait A {
    #[const_trait] //~ ERROR attribute cannot be used on
    fn foo(self);
}

#[const_trait] //~ ERROR attribute cannot be used on
fn main() {}
