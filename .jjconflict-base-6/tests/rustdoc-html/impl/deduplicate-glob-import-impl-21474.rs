// https://github.com/rust-lang/rust/issues/21474
#![crate_name="issue_21474"]

pub use inner::*;

mod inner {
    impl super::Blah for super::What { }
}

pub trait Blah { }

//@ count issue_21474/struct.What.html \
//        '//*[@id="trait-implementations-list"]//*[@class="impl"]' 1
pub struct What;
