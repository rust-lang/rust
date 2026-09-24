//@ known-bug: #150749
#![feature(min_generic_const_args)]

use std::gca;

trait CollectArray {
    fn inner_array();
}

impl CollectArray for () {
    fn inner_array() {
        let temp_ptr: [(); gca!(Self)];
    }
}

fn main() {}
