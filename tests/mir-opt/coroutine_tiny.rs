// skip-filecheck
//! Tests that coroutines that cannot return or unwind don't have unnecessary
//! panic branches.

// compile-flags: -C panic=abort
// no-prefer-dynamic

#![feature(coroutines, coroutine_trait)]

struct HasDrop;

impl Drop for HasDrop {
    fn drop(&mut self) {}
}

fn callee() {}

// EMIT_MIR coroutine_tiny.main-{closure#0}.coroutine_resume.0.mir
fn main() {
    let _gen = |_x: u8| {
        let _d = HasDrop;
        loop {
            yield;
            callee();
        }
    };
}
