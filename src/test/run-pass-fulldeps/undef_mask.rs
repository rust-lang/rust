// ignore-cross-compile
// ignore-stage1

#![feature(rustc_private)]

extern crate rustc;

use rustc::mir::interpret::UndefMask;
use rustc::ty::layout::Size;

fn main() {
    let mut mask = UndefMask::new(Size::from_bytes(500), false);
    assert!(!mask.get(Size::from_bytes(499)));
    mask.set(Size::from_bytes(499), true);
    assert!(mask.get(Size::from_bytes(499)));
    mask.set_range_inbounds(Size::from_bytes(100), Size::from_bytes(256), true);
    for i in 0..100 {
        assert!(!mask.get(Size::from_bytes(i)));
    }
    for i in 100..256 {
        assert!(mask.get(Size::from_bytes(i)));
    }
    for i in 256..499 {
        assert!(!mask.get(Size::from_bytes(i)));
    }
}
