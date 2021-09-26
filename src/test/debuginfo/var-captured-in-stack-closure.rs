// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print variable
// gdb-check:$1 = 1
// gdb-command:print constant
// gdb-check:$2 = 2
// gdb-command:print a_struct
// gdbg-check:$3 = {a = -3, b = 4.5, c = 5}
// gdbr-check:$3 = var_captured_in_stack_closure::Struct {a: -3, b: 4.5, c: 5}
// gdb-command:print *struct_ref
// gdbg-check:$4 = {a = -3, b = 4.5, c = 5}
// gdbr-check:$4 = var_captured_in_stack_closure::Struct {a: -3, b: 4.5, c: 5}
// gdb-command:print *owned
// gdb-check:$5 = 6

// gdb-command:continue

// gdb-command:print variable
// gdb-check:$6 = 2
// gdb-command:print constant
// gdb-check:$7 = 2
// gdb-command:print a_struct
// gdbg-check:$8 = {a = -3, b = 4.5, c = 5}
// gdbr-check:$8 = var_captured_in_stack_closure::Struct {a: -3, b: 4.5, c: 5}
// gdb-command:print *struct_ref
// gdbg-check:$9 = {a = -3, b = 4.5, c = 5}
// gdbr-check:$9 = var_captured_in_stack_closure::Struct {a: -3, b: 4.5, c: 5}
// gdb-command:print *owned
// gdb-check:$10 = 6


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print variable
// lldbg-check:[...]$0 = 1
// lldbr-check:(isize) variable = 1
// lldb-command:print constant
// lldbg-check:[...]$1 = 2
// lldbr-check:(isize) constant = 2
// lldb-command:print a_struct
// lldbg-check:[...]$2 = { a = -3 b = 4.5 c = 5 }
// lldbr-check:(var_captured_in_stack_closure::Struct) a_struct = { a = -3 b = 4.5 c = 5 }
// lldb-command:print *struct_ref
// lldbg-check:[...]$3 = { a = -3 b = 4.5 c = 5 }
// lldbr-check:(var_captured_in_stack_closure::Struct) *struct_ref = { a = -3 b = 4.5 c = 5 }
// lldb-command:print *owned
// lldbg-check:[...]$4 = 6
// lldbr-check:(isize) *owned = 6

// lldb-command:continue

// lldb-command:print variable
// lldbg-check:[...]$5 = 2
// lldbr-check:(isize) variable = 2
// lldb-command:print constant
// lldbg-check:[...]$6 = 2
// lldbr-check:(isize) constant = 2
// lldb-command:print a_struct
// lldbg-check:[...]$7 = { a = -3 b = 4.5 c = 5 }
// lldbr-check:(var_captured_in_stack_closure::Struct) a_struct = { a = -3 b = 4.5 c = 5 }
// lldb-command:print *struct_ref
// lldbg-check:[...]$8 = { a = -3 b = 4.5 c = 5 }
// lldbr-check:(var_captured_in_stack_closure::Struct) *struct_ref = { a = -3 b = 4.5 c = 5 }
// lldb-command:print *owned
// lldbg-check:[...]$9 = 6
// lldbr-check:(isize) *owned = 6


// === CDB TESTS ===================================================================================

// cdb-command: g

// cdb-command: dx variable
// cdb-check:variable         : 1 [Type: [...]]
// cdb-command: dx constant
// cdb-check:constant         : 2 [Type: [...]]
// cdb-command: dx a_struct
// cdb-check:a_struct         [Type: var_captured_in_stack_closure::Struct]
// cdb-check:    [+0x[...]] a                : -3 [Type: [...]]
// cdb-check:    [+0x[...]] b                : 4.500000 [Type: [...]]
// cdb-check:    [+0x[...]] c                : 0x5 [Type: unsigned [...]]
// cdb-command: dx struct_ref
// cdb-check:struct_ref       : 0x[...] [Type: var_captured_in_stack_closure::Struct *]
// cdb-check:    [+0x[...]] a                : -3 [Type: [...]]
// cdb-check:    [+0x[...]] b                : 4.500000 [Type: [...]]
// cdb-check:    [+0x[...]] c                : 0x5 [Type: unsigned [...]]
// cdb-command: dx owned
// cdb-check:owned            : 0x[...] : 6 [Type: [...] *]


// cdb-command: g

// cdb-command: dx variable
// cdb-check:variable         : 2 [Type: [...]]
// cdb-command: dx constant
// cdb-check:constant         : 2 [Type: [...]]
// cdb-command: dx a_struct
// cdb-check:a_struct         [Type: var_captured_in_stack_closure::Struct]
// cdb-check:    [+0x[...]] a                : -3 [Type: [...]]
// cdb-check:    [+0x[...]] b                : 4.500000 [Type: [...]]
// cdb-check:    [+0x[...]] c                : 0x5 [Type: unsigned [...]]
// cdb-command: dx struct_ref
// cdb-check:struct_ref       : 0x[...] [Type: var_captured_in_stack_closure::Struct *]
// cdb-check:    [+0x[...]] a                : -3 [Type: [...]]
// cdb-check:    [+0x[...]] b                : 4.500000 [Type: [...]]
// cdb-check:    [+0x[...]] c                : 0x5 [Type: unsigned [...]]
// cdb-command: dx owned
// cdb-check:owned            : 0x[...] : 6 [Type: [...] *]

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct Struct {
    a: isize,
    b: f64,
    c: usize
}

fn main() {
    let mut variable = 1;
    let constant = 2;

    let a_struct = Struct {
        a: -3,
        b: 4.5,
        c: 5
    };

    let struct_ref = &a_struct;
    let owned: Box<_> = Box::new(6);

    {
        let mut first_closure = || {
            zzz(); // #break
            variable = constant + a_struct.a + struct_ref.a + *owned;
        };

        first_closure();
    }

    {
        let mut second_closure = || {
            zzz(); // #break
            variable = constant + a_struct.a + struct_ref.a + *owned;
        };
        second_closure();
    }
}

fn zzz() {()}
