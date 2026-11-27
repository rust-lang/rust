#![feature(rustc_attrs)]

#[unsafe(no_mangle)]
extern "C-unwind" fn does_not_unwind_but_could() {}

fn main() {
    // Calling a maybe-unwinding function with a non-unwinding ABI is okay if the function
    // does not actually unwind. See `tests/fail/panic/bad_unwind.rs` for the dual test that is UB.
    let f: extern "C-unwind" fn() = does_not_unwind_but_could;
    let f: extern "C" fn() = unsafe { std::mem::transmute(f) };
    f();

    // The same applies when we call such a function via an extern import.
    // This is the dual to `tests/fail/function_calls/exported_symbol_bad_unwind1.rs`.
    extern "C" {
        #[link_name = "does_not_unwind_but_could"]
        fn imported();
    }
    unsafe { imported() };

    // The same does for shims: we can invoke maybe-unwinding shims via a non-unwinding declaration.
    // We don't have "C-unwind" shims that we could import with "C" so the closest thing we can test
    // are "Rust" shims imported via `#[rustc_nounwind]`.
    extern "Rust" {
        #[rustc_nounwind]
        pub fn miri_spin_loop();
    }
    unsafe { miri_spin_loop() };
}
