//@ skip-filecheck
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

fn f(x: i32) -> i32 {
    x
}

#[custom_mir(dialect = "built")]
fn reify_fn_ptr() -> fn(i32) -> i32 {
    mir! {
        {
            RET = Cast(
                f,
                CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer(Safety::Safe)),
            );
            Return()
        }
    }
}

#[custom_mir(dialect = "built")]
fn fn_ptr_to_unsafe(f: fn()) -> unsafe fn() {
    mir! {
        {
            RET = Cast(
                f,
                CastKind::PointerCoercion(PointerCoercion::UnsafeFnPointer),
            );
            Return()
        }
    }
}

#[custom_mir(dialect = "runtime")]
fn subtype_fn_ptr(f: fn(&i32)) -> fn(&'static i32) {
    mir! {
        {
            RET = Cast::<fn(&i32), fn(&'static i32)>(f, CastKind::Subtype);
            Return()
        }
    }
}

#[custom_mir(dialect = "built")]
fn expose_ptr(p: *const i32) -> usize {
    mir! {
        {
            RET = Cast(p, CastKind::PointerExposeProvenance);
            Return()
        }
    }
}

#[custom_mir(dialect = "built")]
fn ptr_from_exposed(p: usize) -> *const i32 {
    mir! {
        {
            RET = Cast(p, CastKind::PointerWithExposedProvenance);
            Return()
        }
    }
}

fn main() {
    assert_eq!(reify_fn_ptr(), f as fn(i32) -> i32);

    let fn_ptr: fn() = || {};
    assert_eq!(fn_ptr as unsafe fn(), fn_ptr_to_unsafe(fn_ptr));

    let fn_ptr: fn(&i32) = |_| {};
    assert_eq!(fn_ptr as fn(&'static i32), subtype_fn_ptr(fn_ptr));

    let p = &1;
    assert_eq!(p as *const i32 as usize, expose_ptr(p));

    assert_eq!(ptr_from_exposed(1), 1 as *const i32);
}
