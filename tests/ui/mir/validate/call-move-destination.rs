//@ revisions: bare field zst generic built indirect
//@[built] compile-flags: -Zvalidate-mir
//@[bare] build-fail
//@[field] build-fail
//@[zst] build-fail
//@[generic] build-fail
//@[built] check-fail
//@[bare] failure-status: 101
//@[bare] dont-check-compiler-stderr
//@[field] failure-status: 101
//@[field] dont-check-compiler-stderr
//@[zst] failure-status: 101
//@[zst] dont-check-compiler-stderr
//@[generic] failure-status: 101
//@[generic] dont-check-compiler-stderr
//@[built] failure-status: 101
//@[built] dont-check-compiler-stderr
//@[indirect] run-pass
//@[indirect] compile-flags: -Zvalidate-mir -Zmir-opt-level=0

// Direct call destinations cannot be rooted in a whole-local move argument.
// This is a structural invariant, including ZSTs and types with unknown layout.

#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

#[cfg(any(bare, built))]
#[cfg_attr(built, custom_mir(dialect = "built"))]
#[cfg_attr(bare, custom_mir(dialect = "runtime", phase = "optimized"))]
fn main() {
    mir! {
        {
            let value = 42u32;
            Call(value = identity(Move(value)), ReturnTo(done), UnwindContinue())
            //[bare,built]~^ ERROR broken MIR
            //[bare,built]~| ERROR return destination refers to a moved local
        }
        done = { Return() }
    }
}

#[cfg(field)]
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        {
            let value = (1u32, 2u32);
            Call(value.0 = first(Move(value)), ReturnTo(done), UnwindContinue())
            //[field]~^ ERROR broken MIR
            //[field]~| ERROR return destination refers to a moved local
        }
        done = { Return() }
    }
}
#[cfg(field)]
fn first(value: (u32, u32)) -> u32 { value.0 }

#[cfg(zst)]
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        {
            let value = ();
            Call(value = identity(Move(value)), ReturnTo(done), UnwindContinue())
            //[zst]~^ ERROR broken MIR
            //[zst]~| ERROR return destination refers to a moved local
        }
        done = { Return() }
    }
}

#[cfg(generic)]
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn generic<T>(value: T) {
    mir! {
        {
            Call(value = identity(Move(value)), ReturnTo(done), UnwindContinue())
            //[generic]~^ ERROR broken MIR
            //[generic]~| ERROR return destination refers to a moved local
        }
        done = { Return() }
    }
}
#[cfg(generic)]
fn main() { generic(()); }

#[cfg(any(bare, built, zst, generic))]
fn identity<T>(value: T) -> T { value }

// Before analysis cleanup, a dereference need not be the first projection.
// Moving the pointer-containing local does not overlap its pointee destination.

#[cfg(indirect)]
#[custom_mir(dialect = "built")]
fn call() -> u32 {
    mir! {
        let raw: *mut u32;
        let pointer: (*mut u32,);
        {
            RET = 0;
            raw = &raw mut RET;
            pointer = (raw,);
            Call(*pointer.0 = consume(Move(pointer)), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}
#[cfg(indirect)]
fn consume(_: (*mut u32,)) -> u32 { 42 }

#[cfg(indirect)]
fn main() { assert_eq!(call(), 42); }
