// test for trait methods with `doc(hidden)`.
#![crate_name = "foo"]

//@ has foo/trait.Trait.html
//@ !has - '//*[@id="associatedtype.Foo"]' 'type Foo'
//@ has - '//*[@id="associatedtype.Bar"]' 'type Bar'
//@ !has - '//*[@id="tymethod.f"]' 'fn f()'
//@ has - '//*[@id="tymethod.g"]' 'fn g()'
pub trait Trait {
    #[doc(hidden)]
    type Foo;
    type Bar;
    #[doc(hidden)]
    fn f();
    fn g();
}

//@ has foo/struct.S.html
//@ !has - '//*[@id="associatedtype.Foo"]' 'type Foo'
//@ has - '//*[@id="associatedtype.Bar"]' 'type Bar'
//@ !has - '//*[@id="method.f"]' 'fn f()'
//@ has - '//*[@id="method.g"]' 'fn g()'
pub struct S;
impl Trait for S {
    type Foo = ();
    type Bar = ();
    fn f() {}
    fn g() {}
}
