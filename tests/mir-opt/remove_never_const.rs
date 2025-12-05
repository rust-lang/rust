// skip-filecheck
// This was originally a regression test for #66975 - ensure that we do not generate never typed
// consts in codegen. We also have tests for this that catches the error, see
// tests/ui/consts/const-eval/index-out-of-bounds-never-type.rs.

#![feature(never_type)]

struct PrintName<T>(T);

impl<T> PrintName<T> {
    const VOID: ! = panic!();
}

// EMIT_MIR remove_never_const.no_codegen.PreCodegen.after.mir
fn no_codegen<T>() {
    let _ = PrintName::<T>::VOID;
}

fn main() {}
