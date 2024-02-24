// This test does not passed with gdb < 8.0. See #53497.
//@ min-gdb-version: 10.1

//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print fun(45, true)
// gdb-check:$1 = true
// gdb-command:print fun(444, false)
// gdb-check:$2 = false

// gdb-command:print r.get_x()
// gdb-check:$3 = 4

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
