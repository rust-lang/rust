//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"

#![feature(rustc_attrs)]

#[rustc_dump_layout(debug)]
pub union Union {
    Float: f32,
    Int: u32,
}
