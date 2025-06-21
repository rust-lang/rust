#![no_std]

//@ is "$.index[?(@.name=='example')].attrs" '["#[attr = MustUse]"]'
#[must_use]
pub fn example() -> impl Iterator<Item = i64> {}

//@ is "$.index[?(@.name=='explicit_message')].attrs" '["#[attr = MustUse {reason: \"does nothing if you do not use it\"}]"]'
#[must_use = "does nothing if you do not use it"]
pub fn explicit_message() -> impl Iterator<Item = i64> {}
