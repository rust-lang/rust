// known-bug: #110395

#![crate_type = "lib"]
#![feature(staged_api)]
#![feature(const_trait_impl)]
#![stable(feature = "foo", since = "1.0.0")]

#[stable(feature = "potato", since = "1.27.0")]
pub struct Data {
    _data: u128,
}

#[stable(feature = "potato", since = "1.27.0")]
#[rustc_const_unstable(feature = "data_foo", issue = "none")]
impl const Default for Data {
    fn default() -> Data {
        Data { _data: 42 }
    }
}
