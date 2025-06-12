#![allow(internal_features)]
#![feature(staged_api)]
#![feature(impl_stability)]
#![stable(feature = "a", since = "1.1.1" )]


/// FIXME(tiif): we haven't allowed marking impl method as 
/// unstable yet, but it should be possible.

#[stable(feature = "a", since = "1.1.1" )]
pub trait Trait {
    #[unstable(feature = "feat", issue = "none" )]
    #[unstable_feature_bound(foo)]
    fn foo();
}

#[stable(feature = "a", since = "1.1.1" )]
impl Trait for u8 {
    #[unstable_feature_bound(foo)]
    fn foo() {}
}

fn main() {}