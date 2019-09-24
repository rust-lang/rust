// build-pass (FIXME(62277): could be check-pass?)

#![feature(const_fn, rustc_attrs)]

#[rustc_args_required_const(0)]
pub const fn a(value: u8) -> u8 {
    value
}

#[rustc_args_required_const(0)]
pub fn b(_: u8) {
    unimplemented!()
}

fn main() {
    let _ = b(a(0));
}
