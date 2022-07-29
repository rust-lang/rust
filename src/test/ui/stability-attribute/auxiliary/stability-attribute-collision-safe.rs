#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "old_feature", since = "1.0.0")]

#[stable(feature = "old_feature", since = "1.0.0")]
pub struct Foo;

#[stable(feature = "old_feature", since = "1.0.0")]
pub struct Bar;

#[stable(feature = "old_feature", since = "1.0.0")]
impl std::ops::Deref for Foo {
    type Target = Bar;

    fn deref(&self) -> &Self::Target {
        &Bar
    }
}

impl Foo {
    #[unstable(feature = "new_feature", issue = "none")]
    pub fn example(&self) -> u32 {
        4
    }

    #[unstable(feature = "new_feature", issue = "none", collision_safe)]
    pub fn example_safe(&self) -> u32 {
        2
    }
}

impl Bar {
    #[stable(feature = "old_feature", since = "1.0.0")]
    pub fn example(&self) -> u32 {
        3
    }

    #[stable(feature = "old_feature", since = "1.0.0")]
    pub fn example_safe(&self) -> u32 {
        2
    }
}
