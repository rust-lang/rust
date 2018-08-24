// ignore-windows failing on win32 bot
// ignore-freebsd: gdb package too new
// ignore-android: FIXME(#10381)
// compile-flags:-g
// min-gdb-version 7.7
// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print vec
// gdb-check:$1 = Vec<u8>(len: 1000000000, cap: 1000000000) = {[...]...}

// gdb-command: print slice
// gdb-check:$2 = &[u8](len: 1000000000) = {[...]...}


#![allow(unused_variables)]

fn main() {

    // Vec
    let mut vec: Vec<u8> = Vec::with_capacity(1_000_000_000);
    unsafe{ vec.set_len(1_000_000_000) }
    let slice = &vec[..];

    zzz(); // #break
}

fn zzz() { () }
