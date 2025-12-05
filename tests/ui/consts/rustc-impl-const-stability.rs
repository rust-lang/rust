//@ compile-flags: -Znext-solver
//@ known-bug: #110395

#![crate_type = "lib"]
#![feature(staged_api, const_trait_impl, const_default)]
#![stable(feature = "foo", since = "1.0.0")]

#[stable(feature = "potato", since = "1.27.0")]
pub struct Data {
    _data: u128,
}

#[stable(feature = "potato", since = "1.27.0")]
#[rustc_const_unstable(feature = "data_foo", issue = "none")]
impl const std::fmt::Debug for Data {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        Ok(())
    }
}
