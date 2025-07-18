#![no_std]

//@ is "$.index[?(@.name=='example')].attrs[*].must_use.reason" null
#[must_use]
pub fn example() -> impl Iterator<Item = i64> {}

//@ is "$.index[?(@.name=='explicit_message')].attrs[*].must_use.reason" '"does nothing if you do not use it"'
#[must_use = "does nothing if you do not use it"]
pub fn explicit_message() -> impl Iterator<Item = i64> {}
