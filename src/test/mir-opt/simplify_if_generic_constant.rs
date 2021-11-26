#![crate_type = "lib"]

pub trait HasBoolConst {
    const B: bool;
}

// EMIT_MIR simplify_if_generic_constant.use_associated_const.SimplifyIfConst.diff
pub fn use_associated_const<T: HasBoolConst>() -> u8 {
    if <T as HasBoolConst>::B {
        13
    } else {
        42
    }
}
