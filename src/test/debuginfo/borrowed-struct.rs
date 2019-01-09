// compile-flags:-g
// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *stack_val_ref
// gdbg-check:$1 = {x = 10, y = 23.5}
// gdbr-check:$1 = borrowed_struct::SomeStruct {x: 10, y: 23.5}

// gdb-command:print *stack_val_interior_ref_1
// gdb-check:$2 = 10

// gdb-command:print *stack_val_interior_ref_2
// gdb-check:$3 = 23.5

// gdb-command:print *ref_to_unnamed
// gdbg-check:$4 = {x = 11, y = 24.5}
// gdbr-check:$4 = borrowed_struct::SomeStruct {x: 11, y: 24.5}

// gdb-command:print *unique_val_ref
// gdbg-check:$5 = {x = 13, y = 26.5}
// gdbr-check:$5 = borrowed_struct::SomeStruct {x: 13, y: 26.5}

// gdb-command:print *unique_val_interior_ref_1
// gdb-check:$6 = 13

// gdb-command:print *unique_val_interior_ref_2
// gdb-check:$7 = 26.5


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print *stack_val_ref
// lldbg-check:[...]$0 = SomeStruct { x: 10, y: 23.5 }
// lldbr-check:(borrowed_struct::SomeStruct) *stack_val_ref = SomeStruct { x: 10, y: 23.5 }

// lldb-command:print *stack_val_interior_ref_1
// lldbg-check:[...]$1 = 10
// lldbr-check:(isize) *stack_val_interior_ref_1 = 10

// lldb-command:print *stack_val_interior_ref_2
// lldbg-check:[...]$2 = 23.5
// lldbr-check:(f64) *stack_val_interior_ref_2 = 23.5

// lldb-command:print *ref_to_unnamed
// lldbg-check:[...]$3 = SomeStruct { x: 11, y: 24.5 }
// lldbr-check:(borrowed_struct::SomeStruct) *ref_to_unnamed = SomeStruct { x: 11, y: 24.5 }

// lldb-command:print *unique_val_ref
// lldbg-check:[...]$4 = SomeStruct { x: 13, y: 26.5 }
// lldbr-check:(borrowed_struct::SomeStruct) *unique_val_ref = SomeStruct { x: 13, y: 26.5 }

// lldb-command:print *unique_val_interior_ref_1
// lldbg-check:[...]$5 = 13
// lldbr-check:(isize) *unique_val_interior_ref_1 = 13

// lldb-command:print *unique_val_interior_ref_2
// lldbg-check:[...]$6 = 26.5
// lldbr-check:(f64) *unique_val_interior_ref_2 = 26.5

#![allow(unused_variables)]
#![feature(box_syntax)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct SomeStruct {
    x: isize,
    y: f64
}

fn main() {
    let stack_val: SomeStruct = SomeStruct { x: 10, y: 23.5 };
    let stack_val_ref: &SomeStruct = &stack_val;
    let stack_val_interior_ref_1: &isize = &stack_val.x;
    let stack_val_interior_ref_2: &f64 = &stack_val.y;
    let ref_to_unnamed: &SomeStruct = &SomeStruct { x: 11, y: 24.5 };

    let unique_val: Box<_> = box SomeStruct { x: 13, y: 26.5 };
    let unique_val_ref: &SomeStruct = &*unique_val;
    let unique_val_interior_ref_1: &isize = &unique_val.x;
    let unique_val_interior_ref_2: &f64 = &unique_val.y;

    zzz(); // #break
}

fn zzz() {()}
