// https://github.com/rust-lang/rust/issues/29503
#![crate_name="issue_29503"]

use std::fmt;

//@ has issue_29503/trait.MyTrait.html
pub trait MyTrait {
    fn my_string(&self) -> String;
}

//@ has - "//div[@id='implementors-list']//*[@id='impl-MyTrait-for-T']//h3[@class='code-header']" "impl<T> MyTrait for Twhere T: Debug"
impl<T> MyTrait for T
where
    T: fmt::Debug,
{
    fn my_string(&self) -> String {
        format!("{:?}", self)
    }
}

pub fn main() {}
