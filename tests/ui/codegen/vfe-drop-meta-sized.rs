//@ run-pass
//@ compile-flags: -Zvirtual-function-elimination -Clto=fat -Copt-level=3 -Zverify-llvm-ir
//@ no-prefer-dynamic
//@ ignore-backends: gcc
#![feature(sized_hierarchy)]
#![no_main]

// Check that the Drop implementation actually runs.
//
// Before, vtables were not annotated with `!type` annotations for supertraits. Hence, when only
// methods from the supertrait were used, VFE considered the vtable unused and hence the drop
// implementation would be seen as dead and removed.
//
// Checking this on MetaSized is interesting because it is a supertrait to most dyn-compatible
// traits, and it has no methods itself so its vtable is always a prefix of the subtrait's vtable.
//
// This test uses #[no_main] because the standard library is compiled without
// -Zvirtual-function-elimination and therefore virtual calls can miscompile.
//
// Usually this shows up first in `lang_start`, because in `lang_start_internal` we coerce main
// to a `&dyn Fn() -> i32`. But the slot for the function in the vtable is considered dead
// and nulled. The call to the NULL function pointer traps.

use std::hint::black_box;
use std::marker::MetaSized;
use std::sync::atomic::{AtomicBool, Ordering};

static DROPPED: AtomicBool = AtomicBool::new(false);

#[allow(dead_code)]
trait T {
    fn method(&self) -> i32;
}

struct S;

impl T for S {
    fn method(&self) -> i32 {
        4
    }
}

impl Drop for S {
    fn drop(&mut self) {
        DROPPED.store(true, Ordering::Relaxed);
    }
}

#[unsafe(no_mangle)]
extern "C" fn main() -> i32 {
    let t: Box<dyn T> = Box::new(S);
    let m: Box<dyn MetaSized> = black_box(t);
    drop(m);

    assert!(DROPPED.load(Ordering::Relaxed));

    0
}
