//@ ignore-windows-gnu: #128981
//@ ignore-android: FIXME(#10381)
//@ ignore-aix: FIXME(#137965)
//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print vec
// gdb-check:$1 = Vec(size=1000000000) = {[...]...}

// gdb-command: print slice
// gdb-check:$2 = &[u8](size=1000000000) = {[...]...}

#![allow(unused_variables)]

fn main() {

    // Vec
    let mut vec: Vec<u8> = Vec::with_capacity(1_000_000_000);
    unsafe{ vec.set_len(1_000_000_000) }
    let slice = &vec[..];

    zzz(); // #break
}

fn zzz() { () }
