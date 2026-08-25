//@ no-prefer-dynamic

//! Since this is `no-prefer-dynamic` we expect compiletest to _not_ look for
//! this create as `libno_prefer_dynamic_lib.so`.

#![crate_type = "rlib"]

pub fn return_42() -> i32 {
    42
}
