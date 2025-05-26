//@check-pass
//@ compile-flags: --print=file-names

#[crate_name = concat!("wrapped")]

macro_rules! inline {
    () => {};
}
