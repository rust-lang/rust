#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![feature(existential_type)]

fn main() {
    assert_eq!(foo().to_string(), "foo");
    assert_eq!(bar1().to_string(), "bar1");
    assert_eq!(bar2().to_string(), "bar2");
    let mut x = bar1();
    x = bar2();
    assert_eq!(boo::boo().to_string(), "boo");
    assert_eq!(my_iter(42u8).collect::<Vec<u8>>(), vec![42u8]);
}

// single definition
existential type Foo: std::fmt::Display;

fn foo() -> Foo {
    "foo"
}

// two definitions
existential type Bar: std::fmt::Display;

fn bar1() -> Bar {
    "bar1"
}

fn bar2() -> Bar {
    "bar2"
}

// definition in submodule
existential type Boo: std::fmt::Display;

mod boo {
    pub fn boo() -> super::Boo {
        "boo"
    }
}

existential type MyIter<T>: Iterator<Item = T>;

fn my_iter<T>(t: T) -> MyIter<T> {
    std::iter::once(t)
}

fn my_iter2<T>(t: T) -> MyIter<T> {
    std::iter::once(t)
}

// param names should not have an effect!
fn my_iter3<U>(u: U) -> MyIter<U> {
    std::iter::once(u)
}

// param position should not have an effect!
fn my_iter4<U, V>(_: U, v: V) -> MyIter<V> {
    std::iter::once(v)
}

// param names should not have an effect!
existential type MyOtherIter<T>: Iterator<Item = T>;

fn my_other_iter<U>(u: U) -> MyOtherIter<U> {
    std::iter::once(u)
}

trait Trait {}
existential type GenericBound<'a, T: Trait>: Sized + 'a;

fn generic_bound<'a, T: Trait + 'a>(t: T) -> GenericBound<'a, T> {
    t
}

mod pass_through {
    pub existential type Passthrough<T>: Sized + 'static;

    fn define_passthrough<T: 'static>(t: T) -> Passthrough<T> {
        t
    }
}

fn use_passthrough(x: pass_through::Passthrough<u32>) -> pass_through::Passthrough<u32> {
    x
}
