//@ test-mir-pass: DataflowConstProp
//@ compile-flags: -Zdump-mir-exclude-alloc-bytes
// EMIT_MIR_FOR_EACH_BIT_WIDTH

#![feature(custom_mir, core_intrinsics, rustc_attrs)]

use std::intrinsics::mir::*;

#[derive(Copy, Clone)]
enum E {
    V1(i32),
    V2(i32),
}

// EMIT_MIR enum.simple.DataflowConstProp.diff

// CHECK-LABEL: fn simple(
fn simple() {
    // CHECK: debug e => [[e:_.*]];
    // CHECK: debug x => [[x:_.*]];
    // CHECK: [[e]] = const E::V1(0_i32);
    let e = E::V1(0);

    // CHECK: switchInt(const 0_isize) -> [0: [[target_bb:bb.*]], 1: bb2, otherwise: bb1];
    // CHECK: [[target_bb]]: {
    // CHECK:     [[x]] = const 0_i32;
    let x = match e {
        E::V1(x1) => x1,
        E::V2(x2) => x2,
    };
}

// EMIT_MIR enum.constant.DataflowConstProp.diff

// CHECK-LABEL: fn constant(
fn constant() {
    // CHECK: debug e => [[e:_.*]];
    // CHECK: debug x => [[x:_.*]];
    const C: E = E::V1(0);

    // CHECK: [[e]] = const constant::C;
    let e = C;
    // CHECK: switchInt(const 0_isize) -> [0: [[target_bb:bb.*]], 1: bb2, otherwise: bb1];
    // CHECK: [[target_bb]]: {
    // CHECK:     [[x]] = const 0_i32;
    let x = match e {
        E::V1(x1) => x1,
        E::V2(x2) => x2,
    };
}

// EMIT_MIR enum.statics.DataflowConstProp.diff

// CHECK-LABEL: fn statics(
fn statics() {
    // CHECK: debug e1 => [[e1:_.*]];
    // CHECK: debug x1 => [[x1:_.*]];
    // CHECK: debug e2 => [[e2:_.*]];
    // CHECK: debug x2 => [[x2:_.*]];

    static C: E = E::V1(0);

    // CHECK: [[e1]] = const E::V1(0_i32);
    let e1 = C;
    // CHECK: switchInt(const 0_isize) -> [0: [[target_bb:bb.*]], 1: bb2, otherwise: bb1];
    // CHECK: [[target_bb]]: {
    // CHECK:     [[x1]] = const 0_i32;
    let x1 = match e1 {
        E::V1(x11) => x11,
        E::V2(x12) => x12,
    };

    static RC: &E = &E::V2(4);

    // CHECK: [[t:_.*]] = const {alloc5: &&E};
    // CHECK: [[e2]] = copy (*[[t]]);
    let e2 = RC;

    // CHECK: switchInt({{move _.*}}) -> {{.*}}
    // FIXME: add checks for x2. Currently, their MIRs are not symmetric in the two
    // switch branches.
    // One is `_9 = &(*_12) and another is `_9 = _11`. It is different from what we can
    // get by printing MIR directly. It is better to check if there are any bugs in the
    // MIR passes around this stage.
    let x2 = match e2 {
        E::V1(x21) => x21,
        E::V2(x22) => x22,
    };
}

#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_nonnull_optimization_guaranteed]
struct NonZeroUsize(usize);

// EMIT_MIR enum.mutate_discriminant.DataflowConstProp.diff

// CHECK-LABEL: fn mutate_discriminant(
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn mutate_discriminant() -> u8 {
    mir! {
        let x: Option<NonZeroUsize>;
        {
            SetDiscriminant(x, 1);
            // This assignment overwrites the niche in which the discriminant is stored.
            place!(Field(Field(Variant(x, 1), 0), 0)) = 0_usize;
            // So we cannot know the value of this discriminant.

            // CHECK: [[a:_.*]] = discriminant({{_.*}});
            let a = Discriminant(x);

            // CHECK: switchInt(copy [[a]]) -> [0: {{bb.*}}, otherwise: {{bb.*}}];
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

// EMIT_MIR enum.multiple.DataflowConstProp.diff
// CHECK-LABEL: fn multiple(
fn multiple(x: bool, i: u8) {
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug e => [[e:_.*]];
    // CHECK: debug x2 => [[x2:_.*]];
    // CHECK: debug y => [[y:_.*]];
    let e = if x {
        // CHECK: [[e]] = Option::<u8>::Some(move {{_.*}});
        Some(i)
    } else {
        // CHECK: [[e]] = Option::<u8>::None;
        None
    };
    // The dataflow state must have:
    //   discriminant(e) => Top
    //   (e as Some).0 => Top
    // CHECK: [[x2]] = const 0_u8;
    // CHECK: [[some:_.*]] = copy (({{_.*}} as Some).0: u8)
    // CHECK: [[x2]] = copy [[some]];
    let x2 = match e {
        Some(i) => i,
        None => 0,
    };

    // Therefore, `x2` should be `Top` here, and no replacement shall happen.

    // CHECK-NOT: [[y]] = const
    // CHECK: [[y]] = copy [[x2]];
    // CHECK-NOT: [[y]] = const
    let y = x2;
}

fn main() {
    simple();
    constant();
    statics();
    mutate_discriminant();
    multiple(false, 5);
}
