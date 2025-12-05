//@ revisions: pass fail
//@[pass] check-pass

#![allow(internal_features)]
#![feature(staged_api)]
#![stable(feature = "a", since = "1.1.1" )]

/// Test the behaviour of marking a trait with #[unstable_feature_bound].
/// In this testcase, even though the trait method `bar` and the `struct Foo` are
/// both stable, #[unstable_feature_bound] is still needed at the call site of Foo::bar().

#[stable(feature = "a", since = "1.1.1" )]
struct Foo;

#[unstable(feature = "foo", issue = "none" )]
#[unstable_feature_bound(foo)]
trait Bar {
    #[stable(feature = "a", since = "1.1.1" )]
    fn bar() {}
}

#[unstable_feature_bound(foo)]
impl Bar for Foo {
}

#[cfg_attr(pass, unstable_feature_bound(foo))]
fn moo() {
    Foo::bar();
    //[fail]~^ ERROR: unstable feature `foo` is used without being enabled.
}


fn main() {}
