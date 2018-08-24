// ignore-windows failing on win32 bot
// ignore-freebsd: gdb package too new
// ignore-android: FIXME(#10381)
// compile-flags:-g
// min-gdb-version 7.7
// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print vec
// gdb-check:$1 = Vec<i32>(len: [...], cap: [...])[...]


#![allow(unused_variables)]

fn main() {

    let vec;
    zzz(); // #break
    vec = vec![0];

}

fn zzz() { () }
