use int::{Int, CastInto};
use float::Float;

#[derive(Clone, Copy)]
enum Result {
    Less,
    Equal,
    Greater,
    Unordered
}

impl Result {
    fn to_le_abi(self) -> i32 {
        match self {
            Result::Less      => -1,
            Result::Equal     => 0,
            Result::Greater   => 1,
            Result::Unordered => 1
        }
    }

    fn to_ge_abi(self) -> i32 {
        match self {
            Result::Less      => -1,
            Result::Equal     => 0,
            Result::Greater   => 1,
            Result::Unordered => -1
        }
    }
}

fn cmp<F: Float>(a: F, b: F) -> Result where
    u32: CastInto<F::Int>,
    F::Int: CastInto<u32>,
    i32: CastInto<F::Int>,
    F::Int: CastInto<i32>,
{
    let one = F::Int::ONE;
    let zero = F::Int::ZERO;

    let sign_bit =      F::SIGN_MASK as F::Int;
    let abs_mask =      sign_bit - one;
    let exponent_mask = F::EXPONENT_MASK;
    let inf_rep =       exponent_mask;

    let a_rep = a.repr();
    let b_rep = b.repr();
    let a_abs = a_rep & abs_mask;
    let b_abs = b_rep & abs_mask;

    // If either a or b is NaN, they are unordered.
    if a_abs > inf_rep || b_abs > inf_rep {
        return Result::Unordered
    }

    // If a and b are both zeros, they are equal.
    if a_abs | b_abs == zero {
        return Result::Equal
    }

    // If at least one of a and b is positive, we get the same result comparing
    // a and b as signed integers as we would with a fp_ting-point compare.
    if a_rep & b_rep >= zero {
        if a_rep < b_rep {
            return Result::Less
        } else if a_rep == b_rep {
            return Result::Equal
        } else {
            return Result::Greater
        }
    }

    // Otherwise, both are negative, so we need to flip the sense of the
    // comparison to get the correct result.  (This assumes a twos- or ones-
    // complement integer representation; if integers are represented in a
    // sign-magnitude representation, then this flip is incorrect).
    else {
        if a_rep > b_rep {
            return Result::Less
        } else if a_rep == b_rep {
            return Result::Equal
        } else {
            return Result::Greater
        }
    }
}
fn unord<F: Float>(a: F, b: F) -> bool where
    u32: CastInto<F::Int>,
    F::Int: CastInto<u32>,
    i32: CastInto<F::Int>,
    F::Int: CastInto<i32>,
{
    let one = F::Int::ONE;

    let sign_bit =      F::SIGN_MASK as F::Int;
    let abs_mask =      sign_bit - one;
    let exponent_mask = F::EXPONENT_MASK;
    let inf_rep =       exponent_mask;

    let a_rep = a.repr();
    let b_rep = b.repr();
    let a_abs = a_rep & abs_mask;
    let b_abs = b_rep & abs_mask;

    a_abs > inf_rep || b_abs > inf_rep
}

intrinsics! {
    pub extern "C" fn __lesf2(a: f32, b: f32) -> i32 {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __gesf2(a: f32, b: f32) -> i32 {
        cmp(a, b).to_ge_abi()
    }

    #[arm_aeabi_alias = fcmpun]
    pub extern "C" fn __unordsf2(a: f32, b: f32) -> bool {
        unord(a, b)
    }

    pub extern "C" fn __eqsf2(a: f32, b: f32) -> bool {
        cmp(a, b).to_le_abi() != 0
    }

    pub extern "C" fn __ltsf2(a: f32, b: f32) -> bool {
        cmp(a, b).to_le_abi() != 0
    }

    pub extern "C" fn __nesf2(a: f32, b: f32) -> bool {
        cmp(a, b).to_le_abi() != 0
    }

    pub extern "C" fn __gtsf2(a: f32, b: f32) -> bool {
        cmp(a, b).to_ge_abi() != 0
    }

    pub extern "C" fn __ledf2(a: f64, b: f64) -> i32 {
        cmp(a, b).to_le_abi()
    }

    pub extern "C" fn __gedf2(a: f64, b: f64) -> i32 {
        cmp(a, b).to_ge_abi()
    }

    #[arm_aeabi_alias = dcmpun]
    pub extern "C" fn __unorddf2(a: f64, b: f64) -> bool {
        unord(a, b)
    }

    pub extern "C" fn __eqdf2(a: f64, b: f64) -> bool {
        cmp(a, b).to_le_abi() != 0
    }

    pub extern "C" fn __ltdf2(a: f64, b: f64) -> bool {
        cmp(a, b).to_le_abi() != 0
    }

    pub extern "C" fn __nedf2(a: f64, b: f64) -> bool {
        cmp(a, b).to_le_abi() != 0
    }

    pub extern "C" fn __gtdf2(a: f32, b: f32) -> bool {
        cmp(a, b).to_ge_abi() != 0
    }
}
