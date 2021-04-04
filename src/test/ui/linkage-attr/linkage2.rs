// FIXME https://github.com/rust-lang/rust/issues/59774

// build-fail
// normalize-stderr-test "thread.*panicked.*Metadata module not compiled.*\n" -> ""
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""
// ignore-sgx no weak linkages permitted

#![feature(linkage)]

extern "C" {
    #[linkage = "extern_weak"]
    static foo: i32;
//~^ ERROR: must have type `*const T` or `*mut T` due to `#[linkage]` attribute
}

fn main() {
    println!("{}", unsafe { foo });
}
