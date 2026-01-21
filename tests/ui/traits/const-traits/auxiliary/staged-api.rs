#![feature(const_trait_impl)]
#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "unstable", issue = "none")]
pub const trait MyTrait {
    #[stable(feature = "rust1", since = "1.0.0")]
    fn func();
}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Unstable;

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "unstable", issue = "none")]
impl const MyTrait for Unstable {
    fn func() {}
}

// tested in inherent-impl-stability.rs instead to avoid clutter
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "unstable", issue = "none")]
const impl Unstable {
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn inherent_func() {}
}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Unstable2;

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "unstable2", issue = "none")]
impl const MyTrait for Unstable2 {
    fn func() {}
}
