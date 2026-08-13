//@ known-bug: #150749
#![feature(min_generic_const_args)]

trait CollectArray {
    fn inner_array();
}
impl CollectArray for () {
    fn inner_array() {
        let temp_ptr: [(); core::direct_const_arg!(Self)];
    }
}
fn main() {}
