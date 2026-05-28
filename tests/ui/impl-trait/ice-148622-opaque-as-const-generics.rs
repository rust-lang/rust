#![feature(type_alias_impl_trait)]

pub type Opaque = impl std::future::Future;
//~^ ERROR: unconstrained opaque type

trait Foo<const N: Opaque> {
    //~^ ERROR: `Opaque` is forbidden as the type of a const generic parameter
    fn bar(&self) -> [u8; N];
    //~^ ERROR: the constant `N` is not of type `usize`
}

fn main() {}
