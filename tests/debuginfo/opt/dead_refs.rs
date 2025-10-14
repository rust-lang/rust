//@ min-lldb-version: 1800
//@ min-gdb-version: 13.0
//@ compile-flags: -g -Copt-level=3
//@ disable-gdb-pretty-printers

// Checks that we still can access dead variables from debuginfos.

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print *ref_v0
// gdb-check:$1 = 0

// gdb-command:print *ref_v1
// gdb-check:$2 = 1

// gdb-command:print *ref_v2
// gdb-check:$3 = 2

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:v *ref_v0
// lldb-check:[...] 0

// lldb-command:v *ref_v1
// lldb-check:[...] 1

// lldb-command:v *ref_v2
// lldb-check:[...] 2

#![allow(unused_variables)]

use std::hint::black_box;

pub struct Foo(i32, i64, i32);

#[inline(never)]
#[no_mangle]
fn test_ref(ref_foo: &Foo) -> i32 {
    let ref_v0 = &ref_foo.0;
    let ref_v1 = &ref_foo.1;
    let ref_v2 = &ref_foo.2;
    ref_foo.0 // #break
}

fn main() {
    let foo = black_box(Foo(0, 1, 2));
    black_box(test_ref(&foo));
}
