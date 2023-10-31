// only-aarch64
// run-pass
// needs-asm-support
// needs-unwind

#![feature(asm_unwind)]

use std::arch::asm;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

struct Foo<'a>(&'a mut bool);

impl Drop for Foo<'_> {
    fn drop(&mut self) {
        *self.0 = false;
    }
}

extern "C" fn panicky() {
    resume_unwind(Box::new(()));
}

fn main() {
    let flag = &mut true;
    catch_unwind(AssertUnwindSafe(|| {
        let _foo = Foo(flag);
        unsafe {
            asm!(
                "bl {}",
                sym panicky,
                clobber_abi("C"),
                options(may_unwind)
            );
        }
    }))
    .expect_err("expected a panic");
    assert_eq!(*flag, false);
}
