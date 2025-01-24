#![crate_name = "foo"]

//@ has 'foo/trait.DynIncompatible.html'
//@ has - '//*[@class="dyn-compatibility-info"]' 'This trait is not dyn compatible.'
//@ has - '//*[@id="dyn-compatibility"]' 'Dyn Compatibility'
pub trait DynIncompatible {
    fn foo() -> Self;
}

//@ has 'foo/trait.DynIncompatible2.html'
//@ has - '//*[@class="dyn-compatibility-info"]' 'This trait is not dyn compatible.'
//@ has - '//*[@id="dyn-compatibility"]' 'Dyn Compatibility'
pub trait DynIncompatible2<T> {
    fn foo(i: T);
}

//@ has 'foo/trait.DynCompatible.html'
//@ !has - '//*[@class="dyn-compatibility-info"]' ''
//@ !has - '//*[@id="dyn-compatibility"]' ''
pub trait DynCompatible {
    fn foo(&self);
}

//@ has 'foo/struct.Foo.html'
//@ count - '//*[@class="dyn-compatibility-info"]' 0
//@ count - '//*[@id="dyn-compatibility"]' 0
pub struct Foo;
