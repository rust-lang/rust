#![feature(never_type)]

pub enum Void {}

#[no_mangle]
pub fn process_never(input: *const !) {
   let _input = unsafe { &*input };
}

#[no_mangle]
pub fn process_void(input: *const Void) {
   let _input = unsafe { &*input };
   // In the future, this should end with `unreachable`, but we currently only do
   // unreachability analysis for `!`.
}

fn main() {}

// END RUST SOURCE
//
// START rustc.process_never.SimplifyLocals.after.mir
// bb0: {
//     StorageLive(_2);
//     _2 = &(*_1);
//     StorageDead(_2);
//     unreachable;
// }
// END rustc.process_never.SimplifyLocals.after.mir
//
// START rustc.process_void.SimplifyLocals.after.mir
// bb0: {
//     StorageLive(_2);
//     _2 = &(*_1);
//     StorageDead(_2);
//     return;
// }
// END rustc.process_void.SimplifyLocals.after.mir
