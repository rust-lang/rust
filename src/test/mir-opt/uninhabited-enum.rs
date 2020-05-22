#![feature(never_type)]

pub enum Void {}

// EMIT_MIR rustc.process_never.SimplifyLocals.after.mir
#[no_mangle]
pub fn process_never(input: *const !) {
   let _input = unsafe { &*input };
}

// EMIT_MIR rustc.process_void.SimplifyLocals.after.mir
#[no_mangle]
pub fn process_void(input: *const Void) {
   let _input = unsafe { &*input };
   // In the future, this should end with `unreachable`, but we currently only do
   // unreachability analysis for `!`.
}

fn main() {}
