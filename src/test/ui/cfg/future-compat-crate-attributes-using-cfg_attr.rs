// check-fail
// compile-flags:--cfg foo

#![deny(warnings)]
#![cfg_attr(foo, crate_type="bin")]
//~^ERROR `crate_type` within
//~| WARN this was previously accepted
//~|ERROR `crate_type` within
//~| WARN this was previously accepted
#![cfg_attr(foo, crate_name="bar")]
//~^ERROR `crate_name` within
//~| WARN this was previously accepted
//~|ERROR `crate_name` within
//~| WARN this was previously accepted

fn main() {}
