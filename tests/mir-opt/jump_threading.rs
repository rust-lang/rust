//@ test-mir-pass: JumpThreading
//@ compile-flags: -Zmir-enable-passes=+Inline
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(try_trait_v2)]
#![feature(custom_mir, core_intrinsics, rustc_attrs)]

use std::intrinsics::mir::*;
use std::ops::ControlFlow;

fn too_complex(x: Result<i32, usize>) -> Option<i32> {
    // CHECK-LABEL: fn too_complex(
    // CHECK: bb0: {
    // CHECK:     switchInt(move {{_.*}}) -> [0: bb3, 1: bb2, otherwise: bb1];
    // CHECK: bb1: {
    // CHECK:     unreachable;
    // CHECK: bb2: {
    // CHECK:     [[controlflow:_.*]] = ControlFlow::<usize, i32>::Break(
    // CHECK:     goto -> bb8;
    // CHECK: bb3: {
    // CHECK:     [[controlflow]] = ControlFlow::<usize, i32>::Continue(
    // CHECK:     goto -> bb4;
    // CHECK: bb4: {
    // CHECK:     goto -> bb6;
    // CHECK: bb5: {
    // CHECK:     {{_.*}} = copy (([[controlflow]] as Break).0: usize);
    // CHECK:     _0 = Option::<i32>::None;
    // CHECK:     goto -> bb7;
    // CHECK: bb6: {
    // CHECK:     {{_.*}} = copy (([[controlflow]] as Continue).0: i32);
    // CHECK:     _0 = Option::<i32>::Some(
    // CHECK:     goto -> bb7;
    // CHECK: bb7: {
    // CHECK:     return;
    // CHECK: bb8: {
    // CHECK:     goto -> bb5;
    match {
        match x {
            Ok(v) => ControlFlow::Continue(v),
            Err(r) => ControlFlow::Break(r),
        }
    } {
        ControlFlow::Continue(v) => Some(v),
        ControlFlow::Break(r) => None,
    }
}

fn identity(x: Result<i32, i32>) -> Result<i32, i32> {
    // CHECK-LABEL: fn identity(
    // CHECK: bb0: {
    // CHECK:     [[x:_.*]] = copy _1;
    // CHECK:     switchInt(move {{_.*}}) -> [0: bb7, 1: bb6, otherwise: bb1];
    // CHECK: bb1: {
    // CHECK:     unreachable;
    // CHECK: bb2: {
    // CHECK:     {{_.*}} = copy (([[controlflow:_.*]] as Continue).0: i32);
    // CHECK:     _0 = Result::<i32, i32>::Ok(
    // CHECK:     goto -> bb4;
    // CHECK: bb3: {
    // CHECK:     {{_.*}} = copy (([[controlflow]] as Break).0: std::result::Result<std::convert::Infallible, i32>);
    // CHECK:     _0 = Result::<i32, i32>::Err(
    // CHECK:     goto -> bb4;
    // CHECK: bb4: {
    // CHECK:     return;
    // CHECK: bb5: {
    // CHECK:     goto -> bb2;
    // CHECK: bb6: {
    // CHECK:     {{_.*}} = move (([[x]] as Err).0: i32);
    // CHECK:     [[controlflow]] = ControlFlow::<Result<Infallible, i32>, i32>::Break(
    // CHECK:     goto -> bb8;
    // CHECK: bb7: {
    // CHECK:     {{_.*}} = move (([[x]] as Ok).0: i32);
    // CHECK:     [[controlflow]] = ControlFlow::<Result<Infallible, i32>, i32>::Continue(
    // CHECK:     goto -> bb5;
    // CHECK: bb8: {
    // CHECK:     goto -> bb3;
    Ok(x?)
}

enum DFA {
    A,
    B,
    C,
    D,
}

/// Check that we do not thread through a loop header,
/// to avoid creating an irreducible CFG.
fn dfa() {
    // CHECK-LABEL: fn dfa(
    // CHECK: bb0: {
    // CHECK:     {{_.*}} = DFA::A;
    // CHECK:     goto -> bb1;
    // CHECK: bb1: {
    // CHECK:     switchInt({{.*}}) -> [0: bb6, 1: bb5, 2: bb4, 3: bb3, otherwise: bb2];
    // CHECK: bb2: {
    // CHECK:     unreachable;
    // CHECK: bb3: {
    // CHECK:     return;
    // CHECK: bb4: {
    // CHECK:     {{_.*}} = DFA::D;
    // CHECK:     goto -> bb1;
    // CHECK: bb5: {
    // CHECK:     {{_.*}} = DFA::C;
    // CHECK:     goto -> bb1;
    // CHECK: bb6: {
    // CHECK:     {{_.*}} = DFA::B;
    // CHECK:     goto -> bb1;
    let mut state = DFA::A;
    loop {
        match state {
            DFA::A => state = DFA::B,
            DFA::B => state = DFA::C,
            DFA::C => state = DFA::D,
            DFA::D => return,
        }
    }
}

#[repr(u8)]
enum CustomDiscr {
    A = 35,
    B = 73,
    C = 99,
}

/// Verify that we correctly match the discriminant value, and not its index.
fn custom_discr(x: bool) -> u8 {
    // CHECK-LABEL: fn custom_discr(
    // CHECK: bb0: {
    // CHECK:     switchInt({{.*}}) -> [0: bb2, otherwise: bb1];
    // CHECK: bb1: {
    // CHECK:     {{_.*}} = CustomDiscr::A;
    // CHECK:     goto -> bb7;
    // CHECK: bb2: {
    // CHECK:     {{_.*}} = CustomDiscr::B;
    // CHECK:     goto -> bb3;
    // CHECK: bb3: {
    // CHECK:     goto -> bb4;
    // CHECK: bb4: {
    // CHECK:     _0 = const 13_u8;
    // CHECK:     goto -> bb6;
    // CHECK: bb5: {
    // CHECK:     _0 = const 5_u8;
    // CHECK:     goto -> bb6;
    // CHECK: bb6: {
    // CHECK:     return;
    // CHECK: bb7: {
    // CHECK:     goto -> bb5;
    match if x { CustomDiscr::A } else { CustomDiscr::B } {
        CustomDiscr::A => 5,
        _ => 13,
    }
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn multiple_match(x: u8) -> u8 {
    // CHECK-LABEL: fn multiple_match(
    mir! {
        {
            // CHECK: bb0: {
            // CHECK:     switchInt(copy [[x:_.*]]) -> [3: bb1, otherwise: bb2];
            match x { 3 => bb1, _ => bb2 }
        }
        bb1 = {
            // We know `x == 3`, so we can take `bb3`.
            // CHECK: bb1: {
            // CHECK:     {{_.*}} = copy [[x]];
            // CHECK:     goto -> bb3;
            let y = x;
            match y { 3 => bb3, _ => bb4 }
        }
        bb2 = {
            // We know `x != 3`, so we can take `bb6`.
            // CHECK: bb2: {
            // CHECK:     [[z:_.*]] = copy [[x]];
            // CHECK:     goto -> bb6;
            let z = x;
            match z { 3 => bb5, _ => bb6 }
        }
        bb3 = {
            // CHECK: bb3: {
            // CHECK:     _0 = const 5_u8;
            // CHECK:     return;
            RET = 5;
            Return()
        }
        bb4 = {
            // CHECK: bb4: {
            // CHECK:     _0 = const 7_u8;
            // CHECK:     return;
            RET = 7;
            Return()
        }
        bb5 = {
            // CHECK: bb5: {
            // CHECK:     _0 = const 9_u8;
            // CHECK:     return;
            RET = 9;
            Return()
        }
        bb6 = {
            // We know `z != 3`, so we CANNOT take `bb7`.
            // CHECK: bb6: {
            // CHECK:     switchInt(copy [[z]]) -> [1: bb7, otherwise: bb8];
            match z { 1 => bb7, _ => bb8 }
        }
        bb7 = {
            // CHECK: bb7: {
            // CHECK:     _0 = const 9_u8;
            // CHECK:     return;
            RET = 9;
            Return()
        }
        bb8 = {
            // CHECK: bb8: {
            // CHECK:     _0 = const 11_u8;
            // CHECK:     return;
            RET = 11;
            Return()
        }
    }
}

/// Both 1-3-4 and 2-3-4 are threadable. As 1 and 2 are the only predecessors of 3,
/// verify that we only thread the 3-4 part.
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn duplicate_chain(x: bool) -> u8 {
    // CHECK-LABEL: fn duplicate_chain(
    mir! {
        let a: u8;
        {
            // CHECK: bb0: {
            // CHECK:     switchInt({{.*}}) -> [1: bb1, otherwise: bb2];
            match x { true => bb1, _ => bb2 }
        }
        bb1 = {
            // CHECK: bb1: {
            // CHECK:     [[a:_.*]] = const 5_u8;
            // CHECK:     goto -> bb3;
            a = 5;
            Goto(bb3)
        }
        bb2 = {
            // CHECK: bb2: {
            // CHECK:     [[a]] = const 5_u8;
            // CHECK:     goto -> bb3;
            a = 5;
            Goto(bb3)
        }
        bb3 = {
            // CHECK: bb3: {
            // CHECK:     {{_.*}} = const 13_i32;
            // CHECK:     goto -> bb4;
            let b = 13;
            Goto(bb4)
        }
        bb4 = {
            // CHECK: bb4: {
            // CHECK:     {{_.*}} = const 15_i32;
            // CHECK-NOT: switchInt(
            // CHECK:     goto -> bb5;
            let c = 15;
            match a { 5 => bb5, _ => bb6 }
        }
        bb5 = {
            // CHECK: bb5: {
            // CHECK:     _0 = const 7_u8;
            // CHECK:     return;
            RET = 7;
            Return()
        }
        bb6 = {
            // CHECK: bb6: {
            // CHECK:     _0 = const 9_u8;
            // CHECK:     return;
            RET = 9;
            Return()
        }
    }
}

#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_nonnull_optimization_guaranteed]
struct NonZeroUsize(usize);

/// Verify that we correctly discard threads that may mutate a discriminant by aliasing.
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn mutate_discriminant() -> u8 {
    // CHECK-LABEL: fn mutate_discriminant(
    // CHECK-NOT: goto -> {{bb.*}};
    // CHECK: switchInt(
    // CHECK-NOT: goto -> {{bb.*}};
    mir! {
        let x: Option<NonZeroUsize>;
        {
            SetDiscriminant(x, 1);
            // This assignment overwrites the niche in which the discriminant is stored.
            place!(Field(Field(Variant(x, 1), 0), 0)) = 0_usize;
            // So we cannot know the value of this discriminant.
            let a = Discriminant(x);
            match a {
                0 => bb1,
                _ => bad,
            }
        }
        bb1 = {
            RET = 1;
            Return()
        }
        bad = {
            RET = 2;
            Unreachable()
        }
    }
}

/// Verify that we do not try to reason when there are mutable pointers involved.
fn mutable_ref() -> bool {
    // CHECK-LABEL: fn mutable_ref(
    // CHECK-NOT: goto -> {{bb.*}};
    // CHECK: switchInt(
    // CHECK: goto -> [[bbret:bb.*]];
    // CHECK: goto -> [[bbret]];
    // CHECK: [[bbret]]: {
    // CHECK-NOT: {{bb.*}}: {
    // CHECK: return;
    let mut x = 5;
    let a = std::ptr::addr_of_mut!(x);
    x = 7;
    unsafe { *a = 8 };
    if x == 7 { true } else { false }
}

/// This function has 2 TOs: 1-3-4 and 0-1-3-4-6.
/// We verify that the second TO does not modify 3 once the first has been applied.
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn renumbered_bb(x: bool) -> u8 {
    // CHECK-LABEL: fn renumbered_bb(
    mir! {
        let a: bool;
        let b: bool;
        {
            // CHECK: bb0: {
            // CHECK:     switchInt({{.*}}) -> [1: bb1, otherwise: bb2];
            b = false;
            match x { true => bb1, _ => bb2 }
        }
        bb1 = {
            // CHECK: bb1: {
            // CHECK:     goto -> bb8;
            a = false;
            Goto(bb3)
        }
        bb2 = {
            // CHECK: bb2: {
            // CHECK:     goto -> bb3;
            a = x;
            b = x;
            Goto(bb3)
        }
        bb3 = {
            // CHECK: bb3: {
            // CHECK:     switchInt({{.*}}) -> [0: bb4, otherwise: bb5];
            match a { false => bb4, _ => bb5 }
        }
        bb4 = {
            // CHECK: bb4: {
            // CHECK:     switchInt({{.*}}) -> [0: bb6, otherwise: bb7];
            match b { false => bb6, _ => bb7 }
        }
        bb5 = {
            // CHECK: bb5: {
            // CHECK:     _0 = const 7_u8;
            RET = 7;
            Return()
        }
        bb6 = {
            // CHECK: bb6: {
            // CHECK:     _0 = const 9_u8;
            RET = 9;
            Return()
        }
        bb7 = {
            // CHECK: bb7: {
            // CHECK:     _0 = const 11_u8;
            RET = 11;
            Return()
        }
        // Duplicate of bb3.
        // CHECK: bb8: {
        // CHECK-NEXT: goto -> bb9;
        // Duplicate of bb4.
        // CHECK: bb9: {
        // CHECK-NEXT: goto -> bb6;
    }
}

/// This function has 3 TOs: 1-4-5, 0-1-4-7-5-8 and 3-4-7-5-6
/// After applying the first TO, we create bb9 to replace 4, and rename 1-4 edge by 1-9. The
/// second TO may try to thread non-existing edge 9-4.
/// This test verifies that we preserve semantics by bailing out of this second TO.
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn disappearing_bb(x: u8) -> u8 {
    // CHECK-LABEL: fn disappearing_bb(
    mir! {
        let a: bool;
        let b: bool;
        {
            a = true;
            b = true;
            match x { 0 => bb3, 1 => bb3, 2 => bb1, _ => bb2 }
        }
        bb1 = {
            // CHECK: bb1: {
            // CHECK: goto -> bb9;
            b = false;
            Goto(bb4)
        }
        bb2 = {
            Unreachable()
        }
        bb3 = {
            // CHECK: bb3: {
            // CHECK: goto -> bb10;
            a = false;
            Goto(bb4)
        }
        bb4 = {
            match b { false => bb5, _ => bb7 }
        }
        bb5 = {
            match a { false => bb6, _ => bb8 }
        }
        bb6 = {
            Return()
        }
        bb7 = {
            Goto(bb5)
        }
        bb8 = {
            Goto(bb6)
        }
        // CHECK: bb9: {
        // CHECK: goto -> bb5;
        // CHECK: bb10: {
        // CHECK: goto -> bb6;
    }
}

/// Verify that we can thread jumps when we assign from an aggregate constant.
fn aggregate(x: u8) -> u8 {
    // CHECK-LABEL: fn aggregate(
    // CHECK-NOT: switchInt(

    const FOO: (u8, u8) = (5, 13);

    let (a, b) = FOO;
    if a == 7 { b } else { a }
}

/// Verify that we can leverage the existence of an `Assume` terminator.
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn assume(a: u8, b: bool) -> u8 {
    // CHECK-LABEL: fn assume(
    mir! {
        {
            // CHECK: bb0: {
            // CHECK-NEXT: switchInt(copy _1) -> [7: bb1, otherwise: bb2]
            match a { 7 => bb1, _ => bb2 }
        }
        bb1 = {
            // CHECK: bb1: {
            // CHECK-NEXT: assume(copy _2);
            // CHECK-NEXT: goto -> bb6;
            Assume(b);
            Goto(bb3)
        }
        bb2 = {
            // CHECK: bb2: {
            // CHECK-NEXT: goto -> bb3;
            Goto(bb3)
        }
        bb3 = {
            // CHECK: bb3: {
            // CHECK-NEXT: switchInt(copy _2) -> [0: bb4, otherwise: bb5];
            match b { false => bb4, _ => bb5 }
        }
        bb4 = {
            // CHECK: bb4: {
            // CHECK-NEXT: _0 = const 4_u8;
            // CHECK-NEXT: return;
            RET = 4;
            Return()
        }
        bb5 = {
            // CHECK: bb5: {
            // CHECK-NEXT: _0 = const 5_u8;
            // CHECK-NEXT: return;
            RET = 5;
            Return()
        }
        // CHECK: bb6: {
        // CHECK-NEXT: goto -> bb5;
    }
}

/// Verify that jump threading succeeds seeing through copies of aggregates.
fn aggregate_copy() -> u32 {
    // CHECK-LABEL: fn aggregate_copy(
    // CHECK-NOT: switchInt(

    const Foo: (u32, u32) = (5, 3);

    let a = Foo;
    // This copies a tuple, we want to ensure that the threading condition on `b.1` propagates to a
    // condition on `a.1`.
    let b = a;
    let c = b.1;
    if c == 2 { b.0 } else { 13 }
}

fn floats() -> u32 {
    // CHECK-LABEL: fn floats(
    // CHECK: switchInt(

    // Test for issue #128243, where float equality was assumed to be bitwise.
    // When adding float support, it must be ensured that this continues working properly.
    let x = if true { -0.0 } else { 1.0 };
    if x == 0.0 { 0 } else { 1 }
}

pub fn bitwise_not() -> i32 {
    // CHECK-LABEL: fn bitwise_not(

    // Test for #131195, which was optimizing `!a == b` into `a != b`.
    let a = 1;
    if !a == 0 { 1 } else { 0 }
}

pub fn logical_not() -> i32 {
    // CHECK-LABEL: fn logical_not(

    let a = false;
    if !a == true { 1 } else { 0 }
}

fn main() {
    // CHECK-LABEL: fn main(
    too_complex(Ok(0));
    identity(Ok(0));
    custom_discr(false);
    dfa();
    multiple_match(5);
    duplicate_chain(false);
    mutate_discriminant();
    mutable_ref();
    renumbered_bb(true);
    disappearing_bb(7);
    aggregate(7);
    assume(7, false);
    floats();
    bitwise_not();
    logical_not();
}

// EMIT_MIR jump_threading.too_complex.JumpThreading.diff
// EMIT_MIR jump_threading.identity.JumpThreading.diff
// EMIT_MIR jump_threading.custom_discr.JumpThreading.diff
// EMIT_MIR jump_threading.dfa.JumpThreading.diff
// EMIT_MIR jump_threading.multiple_match.JumpThreading.diff
// EMIT_MIR jump_threading.duplicate_chain.JumpThreading.diff
// EMIT_MIR jump_threading.mutate_discriminant.JumpThreading.diff
// EMIT_MIR jump_threading.mutable_ref.JumpThreading.diff
// EMIT_MIR jump_threading.renumbered_bb.JumpThreading.diff
// EMIT_MIR jump_threading.disappearing_bb.JumpThreading.diff
// EMIT_MIR jump_threading.aggregate.JumpThreading.diff
// EMIT_MIR jump_threading.assume.JumpThreading.diff
// EMIT_MIR jump_threading.aggregate_copy.JumpThreading.diff
// EMIT_MIR jump_threading.floats.JumpThreading.diff
// EMIT_MIR jump_threading.bitwise_not.JumpThreading.diff
// EMIT_MIR jump_threading.logical_not.JumpThreading.diff
