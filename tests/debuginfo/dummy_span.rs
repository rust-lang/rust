//@ min-lldb-version: 310

//@ compile-flags:-g
//@ ignore-backends: gcc

// === GDB TESTS ===================================================================================

//@ gdb-command:run

//@ gdb-command:next
//@ gdb-command:next
//@ gdb-check:[...]#loc1[...]
//@ gdb-command:next
//@ gdb-check:[...]#loc2[...]

// === LLDB TESTS ==================================================================================

//@ lldb-command:run

//@ lldb-command:next
//@ lldb-command:next
//@ lldb-command:frame select
//@ lldb-check:[...]#loc1[...]
//@ lldb-command:next
//@ lldb-command:frame select
//@ lldb-check:[...]#loc2[...]

use std::env;
use std::num::ParseIntError;

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn main() -> Result<(), ParseIntError> {
    let foo = Foo;
    let number = Ok(7)?;
    zzz(); // #break
    if number % 7 == 0 {
        // This generates code with a dummy span for
        // some reason. If that ever changes this
        // test will not test what it wants to test.
        return Ok(()); // #loc1
    }
    Ok(())
} // #loc2

fn zzz() { () }
