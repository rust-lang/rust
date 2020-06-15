// ignore-windows failing on win32 bot
// ignore-freebsd: gdb package too new
// ignore-android: FIXME(#10381)
// compile-flags:-g
// min-gdb-version 8.1
// min-lldb-version: 310

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print vec
// gdb-check:$1 = Vec(size=[...])[...]


#![allow(unused_variables)]

fn main() {

    let vec;
    zzz(); // #break
    vec = vec![0];

}

fn zzz() { () }
