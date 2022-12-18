#![feature(type_alias_impl_trait)]

pub trait Captures<'a> {}

impl<'a, T: ?Sized> Captures<'a> for T {}

type Foo<'a, 'b> = impl std::fmt::Debug + Captures<'a> + Captures<'b>;

fn foo<'x, 'y>(i: &'x i32, j: &'y i32) -> (Foo<'x, 'y>, Foo<'y, 'x>) {
    (i, i) //~ ERROR concrete type differs from previous
}

fn main() {}
