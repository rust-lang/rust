//@ check-pass
#![feature(type_alias_impl_trait)]

type X<A: ToString + Clone, B: ToString + Clone> = impl ToString;

fn f<A: ToString + Clone, B: ToString + Clone>(a: A, b: B) -> (X<A, B>, X<A, B>) {
    (a.clone(), a)
}

type Foo<'a, 'b> = impl std::fmt::Debug;

fn foo<'x, 'y>(i: &'x i32, j: &'y i32) -> (Foo<'x, 'y>, Foo<'y, 'x>) {
    (i, j)
}

fn main() {
    println!("{}", <X<_, _> as ToString>::to_string(&f(42_i32, String::new()).1));
    let meh = 42;
    let muh = 69;
    println!("{:?}", foo(&meh, &muh));
}
