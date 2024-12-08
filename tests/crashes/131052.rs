//@ known-bug: #131052
#![feature(adt_const_params)]

struct ConstBytes<const T: &'static [*mut u8; 3]>;

pub fn main() {
    let _: ConstBytes<b"AAA"> = ConstBytes::<b"BBB">;
}
