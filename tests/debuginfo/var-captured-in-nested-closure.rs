//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print variable
// gdb-check:$1 = 1
// gdb-command:print constant
// gdb-check:$2 = 2
// gdb-command:print a_struct
// gdb-check:$3 = var_captured_in_nested_closure::Struct {a: -3, b: 4.5, c: 5}
// gdb-command:print *struct_ref
// gdb-check:$4 = var_captured_in_nested_closure::Struct {a: -3, b: 4.5, c: 5}
// gdb-command:print *owned
// gdb-check:$5 = 6
// gdb-command:print closure_local
// gdb-check:$6 = 8
// gdb-command:continue

// gdb-command:print variable
// gdb-check:$7 = 1
// gdb-command:print constant
// gdb-check:$8 = 2
// gdb-command:print a_struct
// gdb-check:$9 = var_captured_in_nested_closure::Struct {a: -3, b: 4.5, c: 5}
// gdb-command:print *struct_ref
// gdb-check:$10 = var_captured_in_nested_closure::Struct {a: -3, b: 4.5, c: 5}
// gdb-command:print *owned
// gdb-check:$11 = 6
// gdb-command:print closure_local
// gdb-check:$12 = 8
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v variable
// lldb-check:[...] 1
// lldb-command:v constant
// lldb-check:[...] 2
// lldb-command:v a_struct
// lldb-check:[...] { a = -3 b = 4.5 c = 5 }
// lldb-command:v *struct_ref
// lldb-check:[...] { a = -3 b = 4.5 c = 5 }
// lldb-command:v *owned
// lldb-check:[...] 6
// lldb-command:v closure_local
// lldb-check:[...] 8
// lldb-command:continue

// lldb-command:v variable
// lldb-check:[...] 1
// lldb-command:v constant
// lldb-check:[...] 2
// lldb-command:v a_struct
// lldb-check:[...] { a = -3 b = 4.5 c = 5 }
// lldb-command:v *struct_ref
// lldb-check:[...] { a = -3 b = 4.5 c = 5 }
// lldb-command:v *owned
// lldb-check:[...] 6
// lldb-command:v closure_local
// lldb-check:[...] 8
// lldb-command:continue


// === CDB TESTS ===================================================================================

// cdb-command: g

// cdb-command: dx variable
// cdb-check:variable         : 1 [Type: [...]]
// cdb-command: dx constant
// cdb-check:constant         : 2 [Type: [...]]
// cdb-command: dx a_struct
// cdb-check:a_struct         [Type: var_captured_in_nested_closure::Struct]
// cdb-check:    [+0x[...]] a                : -3 [Type: [...]]
// cdb-check:    [+0x[...]] b                : 4.500000 [Type: [...]]
// cdb-check:    [+0x[...]] c                : 0x5 [Type: unsigned [...]]
// cdb-command: dx struct_ref
// cdb-check:struct_ref       : 0x[...] [Type: var_captured_in_nested_closure::Struct *]
// cdb-check:    [+0x[...]] a                : -3 [Type: [...]]
// cdb-check:    [+0x[...]] b                : 4.500000 [Type: [...]]
// cdb-check:    [+0x[...]] c                : 0x5 [Type: unsigned [...]]
// cdb-command: dx owned
// cdb-check:owned            : 0x[...] : 6 [Type: [...] *]
// cdb-check:    6 [Type: [...]]
// cdb-command: dx closure_local
// cdb-check:closure_local    : 8 [Type: [...]]
// cdb-command: dx nested_closure
// cdb-check:nested_closure   [Type: var_captured_in_nested_closure::main::closure$0::closure_env$0]

// cdb-command: g

// cdb-command: dx variable
// cdb-check:variable         : 1 [Type: [...]]
// cdb-command: dx constant
// cdb-check:constant         : 2 [Type: [...]]
// cdb-command: dx a_struct
// cdb-check:a_struct         [Type: var_captured_in_nested_closure::Struct]
// cdb-check:    [+0x[...]] a                : -3 [Type: [...]]
// cdb-check:    [+0x[...]] b                : 4.500000 [Type: [...]]
// cdb-check:    [+0x[...]] c                : 0x5 [Type: unsigned [...]]
// cdb-command: dx struct_ref
// cdb-check:struct_ref       : 0x[...] [Type: var_captured_in_nested_closure::Struct *]
// cdb-check:    [+0x[...]] a                : -3 [Type: [...]]
// cdb-check:    [+0x[...]] b                : 4.500000 [Type: [...]]
// cdb-check:    [+0x[...]] c                : 0x5 [Type: unsigned [...]]
// cdb-command: dx owned
// cdb-check:owned            : 0x[...] : 6 [Type: [...] *]
// cdb-check:    6 [Type: [...]]
// cdb-command: dx closure_local
// cdb-check:closure_local    : 8 [Type: [...]]

#![allow(unused_variables)]

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

    let mut closure = || {
        let closure_local = 8;

        let mut nested_closure = || {
            zzz(); // #break
            variable = constant + a_struct.a + struct_ref.a + *owned + closure_local;
        };

        zzz(); // #break

        nested_closure();
    };

    closure();
}

fn zzz() {()}
