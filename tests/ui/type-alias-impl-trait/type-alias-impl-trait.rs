//@ check-pass

#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![feature(type_alias_impl_trait)]

fn main() {
    assert_eq!(foo().to_string(), "foo");
    assert_eq!(bar1().to_string(), "bar1");
    assert_eq!(bar2().to_string(), "bar2");
    let mut x = bar1();
    x = bar2();
    assert_eq!(my_iter(42u8).collect::<Vec<u8>>(), vec![42u8]);
}

// single definition
type Foo = impl std::fmt::Display;

#[define_opaque(Foo)]
fn foo() -> Foo {
    "foo"
}

// two definitions
type Bar = impl std::fmt::Display;

#[define_opaque(Bar)]
fn bar1() -> Bar {
    "bar1"
}

#[define_opaque(Bar)]
fn bar2() -> Bar {
    "bar2"
}

type MyIter<T> = impl Iterator<Item = T>;

#[define_opaque(MyIter)]
fn my_iter<T>(t: T) -> MyIter<T> {
    std::iter::once(t)
}

#[define_opaque(MyIter)]
fn my_iter2<T>(t: T) -> MyIter<T> {
    std::iter::once(t)
}

// param names should not have an effect!
#[define_opaque(MyIter)]
fn my_iter3<U>(u: U) -> MyIter<U> {
    std::iter::once(u)
}

// param position should not have an effect!
#[define_opaque(MyIter)]
fn my_iter4<U, V>(_: U, v: V) -> MyIter<V> {
    std::iter::once(v)
}

// param names should not have an effect!
type MyOtherIter<T> = impl Iterator<Item = T>;

#[define_opaque(MyOtherIter)]
fn my_other_iter<U>(u: U) -> MyOtherIter<U> {
    std::iter::once(u)
}

trait Trait {}
type GenericBound<'a, T: Trait + 'a> = impl Sized + 'a;

#[define_opaque(GenericBound)]
fn generic_bound<'a, T: Trait + 'a>(t: T) -> GenericBound<'a, T> {
    t
}

pub type Passthrough<T: 'static> = impl Sized + 'static;

#[define_opaque(Passthrough)]
fn define_passthrough<T: 'static>(t: T) -> Passthrough<T> {
    t
}

fn use_passthrough(x: Passthrough<u32>) -> Passthrough<u32> {
    x
}
