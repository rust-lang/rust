// run-pass

#![allow(incomplete_features)]
#![feature(inline_const)]
macro_rules! do_const_block{
    ($val:block) => { const $val }
}

fn main() {
    let s = do_const_block!({ 22u64 });
    assert_eq!(s, 22);
}
