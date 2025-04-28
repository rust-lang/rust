#![no_std]

//@ is "$.index[?(@.name=='example')].attrs" '[{"content": "#[must_use]", "is_inner": false}]'
#[must_use]
pub fn example() -> impl Iterator<Item = i64> {}

//@ is "$.index[?(@.name=='explicit_message')].attrs" '[{"content": "#[must_use = \"does nothing if you do not use it\"]", "is_inner": false}]'
#[must_use = "does nothing if you do not use it"]
pub fn explicit_message() -> impl Iterator<Item = i64> {}
