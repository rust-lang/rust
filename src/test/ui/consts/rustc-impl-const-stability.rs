// build-pass

#![crate_type = "lib"]
#![feature(staged_api)]
#![feature(const_trait_impl)]
#![stable(feature = "foo", since = "1.0.0")]


#[stable(feature = "potato", since = "1.27.0")]
pub struct Data {
    _data: u128
}

#[stable(feature = "potato", since = "1.27.0")]
impl const Default for Data {
    #[rustc_const_unstable(feature = "data_foo", issue = "none")]
    fn default() -> Data {
        Data { _data: 42 }
    }
}
