// This test case makes sure that we get proper break points for binaries
// compiled with multiple codegen units. (see #39160)


// min-lldb-version: 310

// compile-flags:-g -Ccodegen-units=2

// === GDB TESTS ===============================================================

// gdb-command:run

// gdb-command:print xxx
// gdb-check:$1 = 12345
// gdb-command:continue

// gdb-command:print yyy
// gdb-check:$2 = 67890
// gdb-command:continue


// === LLDB TESTS ==============================================================

// lldb-command:run

// lldb-command:print xxx
// lldbg-check:[...]$0 = 12345
// lldbr-check:(u32) xxx = 12345
// lldb-command:continue

// lldb-command:print yyy
// lldbg-check:[...]$1 = 67890
// lldbr-check:(u64) yyy = 67890
// lldb-command:continue


#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

mod a {
    pub fn foo(xxx: u32) {
        super::_zzz(); // #break
    }
}

mod b {
    pub fn bar(yyy: u64) {
        super::_zzz(); // #break
    }
}

fn main() {
    a::foo(12345);
    b::bar(67890);
}

#[inline(never)]
fn _zzz() {}
