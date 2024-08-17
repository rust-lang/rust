//@ ignore-windows-gnu: #128981
//@ ignore-android: FIXME(#10381)
//@ compile-flags:-g

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
