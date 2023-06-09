// run-pass

#![feature(inline_const)]

macro_rules! do_const_block{
    ($val:block) => { const $val }
}

fn main() {
    let s = do_const_block!({ 22 });
    assert_eq!(s, 22);
}
