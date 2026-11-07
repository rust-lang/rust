//@ run-pass
//@ compile-flags: -C relocation-model=pic
//@ needs-relocation-model-pic

#![feature(cfg_relocation_model)]

#[cfg(relocation_model = "pic")]
fn main() {}
