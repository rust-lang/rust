// run-pass
#![feature(staged_api)]

#![feature(const_generics)]
#![feature(const_generics_defaults)]
#![allow(incomplete_features)]

#![stable(feature = "const_default_test", since="none")]


#[unstable(feature = "const_default_stable", issue="none")]
pub struct ConstDefaultUnstable<const N: usize = 3>;

#[stable(feature = "const_default_unstable", since="none")]
pub struct ConstDefaultStable<const N: usize = {
    #[stable(feature = "const_default_unstable_val", since="none")]
    3
}>;

fn main() {}
