//@ min-lldb-version: 310

//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run 7

// gdb-command:next
// gdb-command:next
// gdb-check:[...]#loc1[...]
// gdb-command:next
// gdb-check:[...]#loc2[...]

// === LLDB TESTS ==================================================================================

// lldb-command:run 7

// lldb-command:next
// lldb-command:next
// lldb-command:frame select
// lldb-check:[...]#loc1[...]
// lldb-command:next
// lldb-command:frame select
// lldb-check:[...]#loc2[...]

use std::env;
use std::num::ParseIntError;

fn main() -> Result<(), ParseIntError> {
    let args = env::args();
    let number_str = args.skip(1).next().unwrap();
    let number = number_str.parse::<i32>()?;
    zzz(); // #break
    if number % 7 == 0 {
        // This generates code with a dummy span for
        // some reason. If that ever changes this
        // test will not test what it wants to test.
        return Ok(()); // #loc1
    }
    println!("{}", number);
    Ok(())
} // #loc2

fn zzz() { () }
