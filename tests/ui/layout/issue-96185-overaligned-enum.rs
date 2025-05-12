//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$PREF_ALIGN"
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"
#![crate_type = "lib"]
#![feature(rustc_attrs)]

// This cannot use `Scalar` abi since there is padding.
#[rustc_layout(debug)]
#[repr(align(8))]
pub enum Aligned1 { //~ ERROR: layout_of
    Zero = 0,
    One = 1,
}

// This should use `Scalar` abi.
#[rustc_layout(debug)]
#[repr(align(1))]
pub enum Aligned2 { //~ ERROR: layout_of
    Zero = 0,
    One = 1,
}
