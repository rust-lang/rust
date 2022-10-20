#![feature(const_trait_impl)]
#![feature(effects)]

#[const_trait]
trait A {
    #[const_trait] //~ ERROR attribute should be applied
    fn foo(self);
}

#[const_trait] //~ ERROR attribute should be applied
fn main() {}
