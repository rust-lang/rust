//! Since we and our aux-crate is `no-prefer-dynamic` we expect compiletest to
//! _not_ look for `libno_prefer_dynamic_lib.so`.

//@ check-pass
//@ no-prefer-dynamic
//@ aux-crate: no_prefer_dynamic_lib=no_prefer_dynamic_lib.rs

fn main() {
    no_prefer_dynamic_lib::return_42();
}
