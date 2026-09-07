//@ run-pass
//@ needs-unwind
//@ ignore-backends: gcc
//@ compile-flags: -Copt-level=3

struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        core::hint::black_box(());
    }
}

#[inline(never)]
fn unwind() {
    if core::hint::black_box(true) {
        std::panic::resume_unwind(Box::new(()));
    }
}

fn main() {
    // The `catch_unwind` will generate `landingpad catch` and the destructor will generate
    // `landingpad cleanup`; after LLVM inlining it will become `landingpad cleanup catch`, and this
    // is translated to action record chains in LSDA.
    let _ = std::panic::catch_unwind(|| {
        let _guard = Guard;
        unwind();
    });
}
