//@ compile-flags: -Zmir-opt-level=0

#![allow(internal_features)]
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

// Source: adapted from rustc's MIR-opt custom debuginfo fixture:
// `rust/tests/mir-opt/building/custom/debuginfo.rs`.
//
// Priroda uses this as a CLI locals corpus for projected debug-info storage:
// tuple fields, struct fields, enum variant fields, variant-field derefs, and
// constants.
#[custom_mir(dialect = "built")]
fn pointee(opt: &mut Option<i32>) {
    mir! {
        debug foo => Field::<i32>(Variant(*opt, 1), 0);
        {
            Return()
        }
    }
}

#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn numbered(i: (u32, i32)) {
    mir! {
        debug first => i.0;
        debug second => i.1;
        {
            Return()
        }
    }
}

struct S {
    x: f32,
}

#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn structured(i: S) {
    mir! {
        debug x => i.x;
        {
            Return()
        }
    }
}

#[custom_mir(dialect = "built")]
fn variant(opt: Option<i32>) {
    mir! {
        debug inner => Field::<i32>(Variant(opt, 1), 0);
        {
            Return()
        }
    }
}

#[custom_mir(dialect = "built")]
fn variant_deref(opt: Option<&i32>) {
    mir! {
        debug pointer => Field::<&i32>(Variant(opt, 1), 0);
        debug deref => *Field::<&i32>(Variant(opt, 1), 0);
        {
            Return()
        }
    }
}

#[custom_mir(dialect = "built")]
fn constant() {
    mir!(
        debug scalar => 5_usize;
        {
            Return()
        }
    )
}

fn main() {
    numbered((5, 6));
    structured(S { x: 5. });
    variant(Some(5));
    variant_deref(Some(&5));
    pointee(&mut Some(5));
    constant();
}
