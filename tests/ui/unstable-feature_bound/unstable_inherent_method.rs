#![allow(internal_features)]
#![feature(staged_api)]
#![stable(feature = "a", since = "1.1.1" )]

/// FIXME(tiif): we haven't allowed marking trait and impl method as
/// unstable yet, but it should be possible.

#[stable(feature = "a", since = "1.1.1" )]
pub trait Trait {
    #[unstable(feature = "feat", issue = "none" )]
    #[unstable_feature_bound(foo)]
    //~^ ERROR: attribute should be applied to `impl` or free function outside of any `impl` or trait
    fn foo();
}

#[stable(feature = "a", since = "1.1.1" )]
impl Trait for u8 {
    #[unstable_feature_bound(foo)]
    //~^ ERROR: attribute should be applied to `impl` or free function outside of any `impl` or trait
    fn foo() {}
}

fn main() {}
