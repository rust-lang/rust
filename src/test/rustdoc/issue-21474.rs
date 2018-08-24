pub use inner::*;

mod inner {
    impl super::Blah for super::What { }
}

pub trait Blah { }

// @count issue_21474/struct.What.html \
//        '//*[@id="implementations-list"]/*[@class="impl"]' 1
pub struct What;
