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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Collect command line arguments
    let mut args = env::args().skip(1);

    // Ensure an argument is provided
    let number_str = match args.next() {
        Some(arg) => arg,
        None => {
            eprintln!("Error: No number provided.");
            return Err("No number provided.".into());
        }
    };

    // Attempt to parse the argument as an integer
    let number = match number_str.parse::<i32>() {
        Ok(num) => num,
        Err(e) => {
            eprintln!("Error: Failed to parse '{}'. Details: {}", number_str, e);
            return Err(Box::new(e));
        }
    };

    zzz(); // #break

    // Check if the number is divisible by 7
    if number % 7 == 0 {
        // This generates code with a dummy span for some reason.
        // If that ever changes, this test will not test what it intends to.
        return Ok(()); // #loc1
    }

    println!("{}", number);
    Ok(())
} // #loc2

fn zzz() {}
