//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *stack_val_ref
// gdb-check:$1 = borrowed_struct::SomeStruct {x: 10, y: 23.5}

// gdb-command:print *stack_val_interior_ref_1
// gdb-check:$2 = 10

// gdb-command:print *stack_val_interior_ref_2
// gdb-check:$3 = 23.5

// gdb-command:print *ref_to_unnamed
// gdb-check:$4 = borrowed_struct::SomeStruct {x: 11, y: 24.5}

// gdb-command:print *unique_val_ref
// gdb-check:$5 = borrowed_struct::SomeStruct {x: 13, y: 26.5}

// gdb-command:print *unique_val_interior_ref_1
// gdb-check:$6 = 13

// gdb-command:print *unique_val_interior_ref_2
// gdb-check:$7 = 26.5


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v *stack_val_ref
// lldb-check:[...] { x = 10 y = 23.5 }

// lldb-command:v *stack_val_interior_ref_1
// lldb-check:[...] 10

// lldb-command:v *stack_val_interior_ref_2
// lldb-check:[...] 23.5

// lldb-command:v *ref_to_unnamed
// lldb-check:[...] { x = 11 y = 24.5 }

// lldb-command:v *unique_val_ref
// lldb-check:[...] { x = 13 y = 26.5 }

// lldb-command:v *unique_val_interior_ref_1
// lldb-check:[...] 13

// lldb-command:v *unique_val_interior_ref_2
// lldb-check:[...] 26.5

#![allow(unused_variables)]

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

    let unique_val: Box<_> = Box::new(SomeStruct { x: 13, y: 26.5 });
    let unique_val_ref: &SomeStruct = &*unique_val;
    let unique_val_interior_ref_1: &isize = &unique_val.x;
    let unique_val_interior_ref_2: &f64 = &unique_val.y;

    zzz(); // #break
}

fn zzz() {()}
