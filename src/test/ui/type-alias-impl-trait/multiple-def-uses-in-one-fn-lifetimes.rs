#![feature(type_alias_impl_trait)]

type Foo<'a, 'b> = impl std::fmt::Debug;

fn foo<'x, 'y>(i: &'x i32, j: &'y i32) -> (Foo<'x, 'y>, Foo<'y, 'x>) {
    (i, i) //~ ERROR concrete type differs from previous
}

fn main() {}
