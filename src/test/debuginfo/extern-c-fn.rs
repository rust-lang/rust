// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================
// gdb-command:run

// gdb-command:printf "s = \"%s\"\n", s
// gdb-check:s = "abcd"
// gdb-command:print len
// gdb-check:$1 = 20
// gdb-command:print local0
// gdb-check:$2 = 19
// gdb-command:print local1
// gdb-check:$3 = true
// gdb-command:print local2
// gdb-check:$4 = 20.5

// gdb-command:continue

// === LLDB TESTS ==================================================================================
// lldb-command:run

// lldb-command:print len
// lldbg-check:[...]$0 = 20
// lldbr-check:(i32) len = 20
// lldb-command:print local0
// lldbg-check:[...]$1 = 19
// lldbr-check:(i32) local0 = 19
// lldb-command:print local1
// lldbg-check:[...]$2 = true
// lldbr-check:(bool) local1 = true
// lldb-command:print local2
// lldbg-check:[...]$3 = 20.5
// lldbr-check:(f64) local2 = 20.5

// lldb-command:continue

#![allow(unused_variables)]
#![allow(dead_code)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]


#[no_mangle]
pub unsafe extern "C" fn fn_with_c_abi(s: *const u8, len: i32) -> i32 {
    let local0 = len - 1;
    let local1 = len > 2;
    let local2 = (len as f64) + 0.5;

    zzz(); // #break

    return 0;
}

fn main() {
    unsafe {
        fn_with_c_abi(b"abcd\0".as_ptr(), 20);
    }
}

#[inline(never)]
fn zzz() {()}
