// unit-test: JumpThreading
// compile-flags: -Zmir-enable-passes=+Inline
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(control_flow_enum)]
#![feature(try_trait_v2)]
#![feature(custom_mir, core_intrinsics, rustc_attrs)]

use std::intrinsics::mir::*;
use std::ops::ControlFlow;

fn too_complex(x: Result<i32, usize>) -> Option<i32> {
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
    Ok(x?)
}

enum DFA {
    A,
    B,
    C,
    D,
}

fn dfa() {
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

fn custom_discr(x: bool) -> u8 {
    match if x { CustomDiscr::A } else { CustomDiscr::B } {
        CustomDiscr::A => 5,
        _ => 13,
    }
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn multiple_match(x: u8) -> u8 {
    mir!(
        {
            match x { 3 => bb1, _ => bb2 }
        }
        bb1 = {
            // We know `x == 3`, so we can take `bb3`.
            let y = x;
            match y { 3 => bb3, _ => bb4 }
        }
        bb2 = {
            // We know `x != 3`, so we can take `bb6`.
            let z = x;
            match z { 3 => bb5, _ => bb6 }
        }
        bb3 = {
            RET = 5;
            Return()
        }
        bb4 = {
            RET = 7;
            Return()
        }
        bb5 = {
            RET = 9;
            Return()
        }
        bb6 = {
            // We know `z != 3`, so we CANNOT take `bb7`.
            match z { 1 => bb7, _ => bb8 }
        }
        bb7 = {
            RET = 9;
            Return()
        }
        bb8 = {
            RET = 11;
            Return()
        }
    )
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn duplicate_chain(x: bool) -> u8 {
    mir!(
        let a: u8;
        {
            match x { true => bb1, _ => bb2 }
        }
        bb1 = {
            a = 5;
            Goto(bb3)
        }
        bb2 = {
            a = 5;
            Goto(bb3)
        }
        // Verify that we do not create multiple copied of `bb3`.
        bb3 = {
            let b = 13;
            Goto(bb4)
        }
        bb4 = {
            let c = 15;
            match a { 5 => bb5, _ => bb6 }
        }
        bb5 = {
            RET = 7;
            Return()
        }
        bb6 = {
            RET = 9;
            Return()
        }
    )
}

#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_nonnull_optimization_guaranteed]
struct NonZeroUsize(usize);

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn mutate_discriminant() -> u8 {
    mir!(
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
    )
}

// Verify that we do not try to reason when there are mutable pointers involved.
fn mutable_ref() -> bool {
    let mut x = 5;
    let a = std::ptr::addr_of_mut!(x);
    x = 7;
    unsafe { *a = 8 };
    if x == 7 {
        true
    } else {
        false
    }
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn renumbered_bb(x: bool) -> u8 {
    // This function has 2 TOs: 1-3-4 and 0-1-3-4-6.
    // We verify that the second TO does not modify 3 once the first has been applied.
    mir!(
        let a: bool;
        let b: bool;
        {
            b = false;
            match x { true => bb1, _ => bb2 }
        }
        bb1 = {
            a = false;
            Goto(bb3)
        }
        bb2 = {
            a = x;
            b = x;
            Goto(bb3)
        }
        bb3 = {
            match a { false => bb4, _ => bb5 }
        }
        bb4 = {
            match b { false => bb6, _ => bb7 }
        }
        bb5 = {
            RET = 7;
            Return()
        }
        bb6 = {
            RET = 9;
            Return()
        }
        bb7 = {
            RET = 11;
            Return()
        }
    )
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn disappearing_bb(x: u8) -> u8 {
    // This function has 3 TOs: 1-4-5, 0-1-4-7-5-8 and 3-4-7-5-6
    // After applying the first TO, we create bb9 to replace 4, and rename 1-4 edge by 1-9. The
    // second TO may try to thread non-existing edge 9-4.
    // This test verifies that we preserve semantics by bailing out of this second TO.
    mir!(
        let _11: i8;
        let _12: bool;
        let _13: bool;
        {
            _13 = false;
            _12 = false;
            _13 = true;
            _12 = true;
            match x { 0 => bb3, 1 => bb3, 2 => bb1, _ => bb2 }
        }
        bb1 = {
            _12 = false;
            Goto(bb4)
        }
        bb2 = {
            Unreachable()
        }
        bb3 = {
            _13 = false;
            Goto(bb4)
        }
        bb4 = {
            match _12 { false => bb5, _ => bb7 }
        }
        bb5 = {
            match _13 { false => bb6, _ => bb8 }
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
    )
}

fn main() {
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
