//@ test-mir-pass: MatchBranchSimplification

#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Foo {
    A,
    B,
    // This variant is not used, but makes the enum BackendRepr::Memory. Without it, the enum is a
    // scalar and overlapping copies of it are permitted.
    C(u32),
}

// EMIT_MIR match_branch_simplification_aliasing.aliasing_locals.MatchBranchSimplification.diff
#[inline(never)]
#[custom_mir(dialect = "runtime")]
fn aliasing_locals(init: Foo) -> Foo {
    // CHECK-LABEL: fn aliasing_locals(_1
    // CHECK: _2 = copy _1;
    // CHECK: _3 = &raw const _2;
    // CHECK: _4 = discriminant((*_3));
    // CHECK-NOT: copy (*_3);
    // CHECK: switchInt
    // CHECK: _2 = Foo::A;
    // CHECK: _2 = Foo::B;
    // CHECK: _0 = copy _2;
    mir! {
        let x: Foo;
        let p: *const Foo;
        let d: u8;
        {
            x = init;
            p = core::ptr::addr_of!(x);
            d = Discriminant(*p);
            match d {
                0 => bb_a,
                1 => bb_b,
                _ => bb_unreachable,
            }
        }
        bb_unreachable = {
            Unreachable()
        }
        bb_a = {
            x = Foo::A;
            Goto(bb_join)
        }
        bb_b = {
            x = Foo::B;
            Goto(bb_join)
        }
        bb_join = {
            RET = x;
            Return()
        }
    }
}

union U {
    a: Foo,
    b: Foo,
}

// EMIT_MIR match_branch_simplification_aliasing.union_fields.MatchBranchSimplification.diff
#[inline(never)]
#[custom_mir(dialect = "runtime")]
fn union_fields(init: Foo) -> Foo {
    // CHECK-LABEL: fn union_fields(_1
    // CHECK: (_2.1: Foo) = copy _1;
    // CHECK: _3 = discriminant((_2.1: Foo));
    // CHECK-NOT: copy(_2.1: Foo);
    // CHECK: switchInt
    // CHECK: (_2.0: Foo) = Foo::A;
    // CHECK: (_2.0: Foo) = Foo::B;
    // CHECK: _0 = copy (_2.0: Foo);
    mir! {
        let u: U;
        let d: u8;
        {
            u.b = init;
            d = Discriminant(u.b);
            match d {
                0 => bb_a,
                1 => bb_b,
                _ => bb_unreachable,
            }
        }
        bb_unreachable = {
            Unreachable()
        }
        bb_a = {
            u.a = Foo::A;
            Goto(bb_join)
        }
        bb_b = {
            u.a = Foo::B;
            Goto(bb_join)
        }
        bb_join = {
            RET = u.a;
            Return()
        }
    }
}

fn main() {
    let r = aliasing_locals(std::hint::black_box(Foo::B));
    assert!(r == Foo::B);

    let r = union_fields(std::hint::black_box(Foo::B));
    assert!(r == Foo::B);
}
