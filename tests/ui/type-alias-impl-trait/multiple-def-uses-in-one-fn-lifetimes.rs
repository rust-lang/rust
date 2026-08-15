#![feature(type_alias_impl_trait)]

type Foo<'a, 'b> = impl std::fmt::Debug;

#[define_opaque(Foo)]
fn foo<'x, 'y>(i: &'x i32, j: &'y i32) -> (Foo<'x, 'y>, Foo<'y, 'x>) {
    (i, j)
    //~^ ERROR opaque type used twice with different lifetimes
}

fn main() {}
