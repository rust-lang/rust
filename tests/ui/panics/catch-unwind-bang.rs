//! Check that the unwind machinery handles uninhabited types correctly.
//! It used to call `std::mem::uninitialized::<!>();` at some point...
//!
//! See <https://github.com/rust-lang/rust/issues/39432>

//@ run-pass
//@ needs-unwind

fn worker() -> ! {
    panic!()
}

fn main() {
    std::panic::catch_unwind(worker).unwrap_err();
}
