//@ compile-flags:-g
//@ disable-gdb-pretty-printers

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

// lldb-command:v len
// lldb-check:[...] 20
// lldb-command:v local0
// lldb-check:[...] 19
// lldb-command:v local1
// lldb-check:[...] true
// lldb-command:v local2
// lldb-check:[...] 20.5

// lldb-command:continue

#![allow(unused_variables)]
#![allow(dead_code)]

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
