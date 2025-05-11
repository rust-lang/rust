//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$PREF_ALIGN"
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"
#![crate_type = "lib"]
#![feature(rustc_attrs)]

use std::mem::MaybeUninit;

enum HasNiche {
    A,
    B,
    C,
}

// This should result in ScalarPair(Initialized, Union),
// since the u8 payload will be uninit for `None`.
#[rustc_layout(debug)]
pub enum MissingPayloadField { //~ ERROR: layout_of
    Some(u8),
    None
}

// This should result in ScalarPair(Initialized, Initialized),
// since the u8 field is present in all variants,
// and hence will always be initialized.
#[rustc_layout(debug)]
pub enum CommonPayloadField { //~ ERROR: layout_of
    A(u8),
    B(u8),
}

// This should result in ScalarPair(Initialized, Union),
// since, though a u8-sized field is present in all variants, it might be uninit.
#[rustc_layout(debug)]
pub enum CommonPayloadFieldIsMaybeUninit { //~ ERROR: layout_of
    A(u8),
    B(MaybeUninit<u8>),
}

// This should result in ScalarPair(Initialized, Union),
// since only the niche field (used for the tag) is guaranteed to be initialized.
#[rustc_layout(debug)]
pub enum NicheFirst { //~ ERROR: layout_of
    A(HasNiche, u8),
    B,
    C
}

// This should result in ScalarPair(Union, Initialized),
// since only the niche field (used for the tag) is guaranteed to be initialized.
#[rustc_layout(debug)]
pub enum NicheSecond { //~ ERROR: layout_of
    A(u8, HasNiche),
    B,
    C,
}
