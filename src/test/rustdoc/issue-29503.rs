use std::fmt;

// @has issue_29503/trait.MyTrait.html
pub trait MyTrait {
    fn my_string(&self) -> String;
}

// @has - "//div[@id='implementors-list']//*[@id='impl-MyTrait-for-T']//h3[@class='code-header in-band']" "impl<T> MyTrait for Twhere T: Debug"
impl<T> MyTrait for T
where
    T: fmt::Debug,
{
    fn my_string(&self) -> String {
        format!("{:?}", self)
    }
}

pub fn main() {}
