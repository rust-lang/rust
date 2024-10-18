// skip-filecheck
//@ compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=2

#![crate_type = "lib"]

pub trait TraitWithBool {
    const FLAG: bool;
}

// EMIT_MIR if_associated_const.check_bool.PreCodegen.after.mir
pub fn check_bool<T: TraitWithBool>() -> u32 {
    if T::FLAG { 123 } else { 456 }
}

pub trait TraitWithInt {
    const VALUE: i32;
}

// EMIT_MIR if_associated_const.check_int.PreCodegen.after.mir
pub fn check_int<T: TraitWithInt>() -> u32 {
    match T::VALUE {
        1 => 123,
        2 => 456,
        3 => 789,
        _ => 0,
    }
}
