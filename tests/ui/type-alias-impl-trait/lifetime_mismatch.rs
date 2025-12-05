#![feature(type_alias_impl_trait)]

type Foo<'a> = impl Sized;

#[define_opaque(Foo)]
fn foo<'a, 'b>(x: &'a u32, y: &'b u32) -> (Foo<'a>, Foo<'b>) {
    (x, y)
    //~^ ERROR opaque type used twice with different lifetimes
}

type Bar<'a, 'b> = impl std::fmt::Debug;

#[define_opaque(Bar)]
fn bar<'x, 'y>(i: &'x i32, j: &'y i32) -> (Bar<'x, 'y>, Bar<'y, 'x>) {
    (i, j)
    //~^ ERROR opaque type used twice with different lifetimes
}

fn main() {
    let meh = 42;
    let muh = 69;
    println!("{:?}", bar(&meh, &muh));
}
