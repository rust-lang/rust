// Checks that exported items without stability attributes cause an error

#![crate_type="lib"]
#![feature(staged_api)]

#![stable(feature = "stable_test_feature", since = "1.0.0")]

pub fn unmarked() {
    //~^ ERROR function has missing stability attribute
    ()
}

#[unstable(feature = "unstable_test_feature", issue = "0")]
pub mod foo {
    // #[unstable] is inherited
    pub fn unmarked() {}
}

#[stable(feature = "stable_test_feature", since="1.0.0")]
pub mod bar {
    // #[stable] is not inherited
    pub fn unmarked() {}
    //~^ ERROR function has missing stability attribute
}
