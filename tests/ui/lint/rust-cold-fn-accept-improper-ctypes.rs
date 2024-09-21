//@ check-pass
#![feature(rust_cold_cc)]

// extern "rust-cold" is a "Rust" ABI so we accept `repr(Rust)` types as arg/ret without warnings.

pub extern "rust-cold" fn f(_: ()) -> Result<(), ()> {
    Ok(())
}

extern "rust-cold" {
    pub fn g(_: ()) -> Result<(), ()>;
}

fn main() {}
