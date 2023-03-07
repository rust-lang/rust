// compile-flags:-g
// edition:2021
// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print test
// gdbr-check:$1 = captured_fields_1::main::{closure_env#0} {_ref__my_ref__my_field1: 0x[...]}
// gdb-command:continue
// gdb-command:print test
// gdbr-check:$2 = captured_fields_1::main::{closure_env#1} {_ref__my_ref__my_field2: 0x[...]}
// gdb-command:continue
// gdb-command:print test
// gdbr-check:$3 = captured_fields_1::main::{closure_env#2} {_ref__my_ref: 0x[...]}
// gdb-command:continue
// gdb-command:print test
// gdbr-check:$4 = captured_fields_1::main::{closure_env#3} {my_ref: 0x[...]}
// gdb-command:continue
// gdb-command:print test
// gdbr-check:$5 = captured_fields_1::main::{closure_env#4} {my_var__my_field2: 22}
// gdb-command:continue
// gdb-command:print test
// gdbr-check:$6 = captured_fields_1::main::{closure_env#5} {my_var: captured_fields_1::MyStruct {my_field1: 11, my_field2: 22}}
// gdb-command:continue

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print test
// lldbg-check:(captured_fields_1::main::{closure_env#0}) $0 = { _ref__my_ref__my_field1 = 0x[...] }
// lldb-command:continue
// lldb-command:print test
// lldbg-check:(captured_fields_1::main::{closure_env#1}) $1 = { _ref__my_ref__my_field2 = 0x[...] }
// lldb-command:continue
// lldb-command:print test
// lldbg-check:(captured_fields_1::main::{closure_env#2}) $2 = { _ref__my_ref = 0x[...] }
// lldb-command:continue
// lldb-command:print test
// lldbg-check:(captured_fields_1::main::{closure_env#3}) $3 = { my_ref = 0x[...] }
// lldb-command:continue
// lldb-command:print test
// lldbg-check:(captured_fields_1::main::{closure_env#4}) $4 = { my_var__my_field2 = 22 }
// lldb-command:continue
// lldb-command:print test
// lldbg-check:(captured_fields_1::main::{closure_env#5}) $5 = { my_var = { my_field1 = 11 my_field2 = 22 } }
// lldb-command:continue

#![allow(unused)]

struct MyStruct {
    my_field1: u32,
    my_field2: u32,
}

fn main() {
    let mut my_var = MyStruct { my_field1: 11, my_field2: 22 };
    let my_ref = &mut my_var;

    let test = || {
        let a = &mut my_ref.my_field1;
    };

    _zzz(); // #break

    let test = || {
        let a = &my_ref.my_field2;
    };

    _zzz(); // #break

    let test = || {
        let a = &my_ref;
    };

    _zzz(); // #break

    let test = || {
        let a = my_ref;
    };

    _zzz(); // #break

    let test = move || {
        let a = my_var.my_field2;
    };

    _zzz(); // #break

    let test = || {
        let a = my_var;
    };

    _zzz(); // #break
}

fn _zzz() {}
