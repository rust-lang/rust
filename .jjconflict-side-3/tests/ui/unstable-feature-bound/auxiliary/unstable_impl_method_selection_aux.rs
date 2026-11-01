#![allow(internal_features)]
#![feature(staged_api)]
#![stable(feature = "a", since = "1.1.1" )]

#[stable(feature = "a", since = "1.1.1" )]
pub trait Trait {
    #[stable(feature = "a", since = "1.1.1" )]
    fn foo(&self) {}
}

#[stable(feature = "a", since = "1.1.1" )]
impl Trait for Vec<u32> {
    fn foo(&self) {}
}

#[unstable_feature_bound(bar)]
#[unstable(feature = "bar", issue = "none" )]
impl Trait for Vec<u64> {
    fn foo(&self) {}
}
