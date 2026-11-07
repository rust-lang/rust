//@ known-bug: #150749
#![feature(gca_min_const_items)]

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
