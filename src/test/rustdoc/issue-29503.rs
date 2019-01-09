// ignore-tidy-linelength

use std::fmt;

// @has issue_29503/trait.MyTrait.html
pub trait MyTrait {
    fn my_string(&self) -> String;
}

// @has - "//div[@id='implementors-list']/h3[@id='impl-MyTrait']//code" "impl<T> MyTrait for T where T: Debug"
impl<T> MyTrait for T where T: fmt::Debug {
    fn my_string(&self) -> String {
        format!("{:?}", self)
    }
}

pub fn main() {
}
