// compile-flags:-g
// edition:2021
// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print my_ref__my_field1
// gdbr-check:$1 = 11
// gdb-command:continue
// gdb-command:print my_var__my_field2
// gdbr-check:$2 = 22
// gdb-command:continue

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:print my_ref__my_field1
// lldbg-check:(unsigned int) $0 = 11
// lldb-command:continue
// lldb-command:print my_var__my_field2
// lldbg-check:(unsigned int) $1 = 22
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
        let a = my_ref.my_field1;

        _zzz(); // #break
    };

    test();

    let test = move || {
        let a = my_var.my_field2;

        _zzz(); // #break
    };

    test();
}

fn _zzz() {}
