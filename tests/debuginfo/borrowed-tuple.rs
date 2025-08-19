//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *stack_val_ref
// gdb-check:$1 = (-14, -19)

// gdb-command:print *ref_to_unnamed
// gdb-check:$2 = (-15, -20)

// gdb-command:print *unique_val_ref
// gdb-check:$3 = (-17, -22)


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v *stack_val_ref
// lldb-check:[...] { 0 = -14 1 = -19 }

// lldb-command:v *ref_to_unnamed
// lldb-check:[...] { 0 = -15 1 = -20 }

// lldb-command:v *unique_val_ref
// lldb-check:[...] { 0 = -17 1 = -22 }


#![allow(unused_variables)]

fn main() {
    let stack_val: (i16, f32) = (-14, -19f32);
    let stack_val_ref: &(i16, f32) = &stack_val;
    let ref_to_unnamed: &(i16, f32) = &(-15, -20f32);

    let unique_val: Box<(i16, f32)> = Box::new((-17, -22f32));
    let unique_val_ref: &(i16, f32) = &*unique_val;

    zzz(); // #break
}

fn zzz() {()}
