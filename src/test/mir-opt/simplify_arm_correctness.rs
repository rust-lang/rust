// Cases that should not be optimized by `SimplifyArmIdentity`

// It may be possible to optimize these first two cases in the future, with improvements to other
// optimizations or changes in MIR semantics. However, at the moment, it would not be correct given
// the input to the pass and the extent of the analysis
enum MixedCopyA {
    A(u32),
    B,
}

// EMIT_MIR simplify_arm_correctness.mixed_copy_a_id.SimplifyArmIdentity.diff
fn mixed_copy_a_id(x: MixedCopyA) -> MixedCopyA {
    match x {
        MixedCopyA::A(a) => MixedCopyA::A(a),
        MixedCopyA::B => MixedCopyA::B,
    }
}

enum MixedCopyB {
    A(u32, String),
    B,
}

// EMIT_MIR simplify_arm_correctness.mixed_copy_b_id.SimplifyArmIdentity.diff
fn mixed_copy_b_id(x: MixedCopyB) -> MixedCopyB {
    match x {
        MixedCopyB::A(a, b) => MixedCopyB::A(a, b),
        MixedCopyB::B => MixedCopyB::B,
    }
}

// EMIT_MIR simplify_arm_correctness.mismatched_variant.SimplifyArmIdentity.diff
fn mismatched_variant(x: Result<u32, u32>) -> Result<u32, u32> {
    match x {
        Ok(a) => Err(a),
        Err(a) => Ok(a),
    }
}

enum MultiField {
    A(u32, u32),
    B,
}

// EMIT_MIR simplify_arm_correctness.partial_copy.SimplifyArmIdentity.diff
fn partial_copy(x: MultiField) -> MultiField {
    match x {
        MultiField::A(a, _b) => MultiField::A(a, 5),
        MultiField::B => MultiField::B,
    }
}

fn main() {
    mixed_copy_a_id(MixedCopyA::A(0));
    mixed_copy_b_id(MixedCopyB::A(0, String::new()));
    mismatched_variant(Ok(0));
    partial_copy(MultiField::B);
}
