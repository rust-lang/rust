// Regression test for <https://github.com/rust-lang/rust/issues/94183>.
// This test ensures that a publicly re-exported private trait will
// appear in the blanket impl list.

// https://github.com/rust-lang/rust/issues/94183
#![crate_name = "foo"]

//@ has 'foo/struct.S.html'

mod actual_sub {
    pub trait Actual {}
    pub trait Another {}

    // `Another` is publicly re-exported so it should appear in the blanket impl list.
    //@ has - '//*[@id="blanket-implementations-list"]//*[@class="code-header"]' 'impl<T> Another for T'
    impl<T> Another for T {}

    trait Foo {}

    // `Foo` is not publicly re-exported nor reachable so it shouldn't appear in the
    // blanket impl list.
    //@ !has - '//*[@id="blanket-implementations-list"]//*[@class="code-header"]' 'impl<T> Foo for T'
    impl<T> Foo for T {}
}

pub use actual_sub::{Actual, Another};

// `Actual` is publicly re-exported so it should appear in the blanket impl list.
//@ has - '//*[@id="blanket-implementations-list"]//*[@class="code-header"]' 'impl<T> Actual for T'
impl<T> Actual for T {}

pub struct S;
