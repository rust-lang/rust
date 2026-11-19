//! Regression test for <https://github.com/rust-lang/rust/issues/150749>


#![feature(gca_min_const_items)]

use std::gca;

trait CollectArray {
    fn inner_array();
}

impl CollectArray for () {
    fn inner_array() {
        let temp_ptr: [(); gca!(Self)];
        //~^ ERROR the `Self` constructor cannot be used as a const argument
    }
}

fn main() {}
