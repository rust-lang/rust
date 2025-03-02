const UNINIT: core::mem::MaybeUninit<core::cell::Cell<&'static ()>> = core::mem::MaybeUninit::uninit();
//~^ declare_interior_mutable_const

fn main() {}
