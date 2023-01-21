// unit-test: DataflowConstProp

#![feature(custom_mir, core_intrinsics, rustc_attrs)]

use std::intrinsics::mir::*;

enum E {
    V1(i32),
    V2(i32)
}

// EMIT_MIR enum.simple.DataflowConstProp.diff
fn simple() {
    let e = E::V1(0);
    let x = match e { E::V1(x) => x, E::V2(x) => x };
}

#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_nonnull_optimization_guaranteed]
struct NonZeroUsize(usize);

// EMIT_MIR enum.mutate_discriminant.DataflowConstProp.diff
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

fn main() {
    simple();
    mutate_discriminant();
}
