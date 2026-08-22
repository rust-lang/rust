// This test does not passed with gdb < 8.0. See #53497.
//
// This test seems to have become very flaky with "Couldn't write extended state status: Bad
// address." since around June 2026, where CI typically uses GDB 12.1 on Ubuntu 22.04. I tried
// running this locally with GDB 15.1 and could not reproduce the flakiness. See #159073.
//@ min-gdb-version: 15.1

//@ compile-flags:-g
//@ ignore-backends: gcc

// === GDB TESTS ===================================================================================

//@ gdb-command:run

//@ gdb-command:print fun(45, true)
//@ gdb-check:$1 = true
//@ gdb-command:print fun(444, false)
//@ gdb-check:$2 = false

//@ gdb-command: print function_call::RegularStruct::get_x(&r)
//@ gdb-check:$3 = 4

#![allow(dead_code, unused_variables)]

struct RegularStruct {
    x: i32
}

impl RegularStruct {
    fn get_x(&self) -> i32 {
        self.x
    }
}

fn main() {
    let _ = fun(4, true);
    let r = RegularStruct{x: 4};
    let _ = r.get_x();

    zzz(); // #break
}

fn fun(x: isize, y: bool) -> bool {
    y
}

fn zzz() { () }
