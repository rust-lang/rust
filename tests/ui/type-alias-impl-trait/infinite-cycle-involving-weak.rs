#![feature(type_alias_impl_trait)]

type T = impl Copy;
//~^ ERROR cannot resolve opaque type

#[define_opaque(T)]
fn foo() -> T {
    None::<&'static T>
}

fn main() {}
