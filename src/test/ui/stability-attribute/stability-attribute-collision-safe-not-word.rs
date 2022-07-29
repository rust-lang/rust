#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "old_feature", since = "1.0.0")]

#[stable(feature = "old_feature", since = "1.0.0")]
pub struct Foo;

impl Foo {
    #[unstable(feature = "new_feature", issue = "none", collision_safe = "foo")]
    //~^ ERROR `collision_safe` should not have any arguments
    pub fn bad(&self) -> u32 {
        2
    }

    #[unstable(feature = "new_feature", issue = "none", collision_safe("foo"))]
    //~^ ERROR `collision_safe` should not have any arguments
    pub fn also_bad(&self) -> u32 {
        2
    }
}
