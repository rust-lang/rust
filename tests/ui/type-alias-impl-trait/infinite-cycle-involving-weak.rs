#![feature(type_alias_impl_trait)]

type T = impl Copy;

#[define_opaque(T)]
fn foo() -> T {
    //~^ ERROR cannot resolve opaque type
    None::<&'static T>
}

fn main() {}
