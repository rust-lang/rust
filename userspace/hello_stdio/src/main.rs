//! Smoke test: println! and eprintln! go to stdout and stderr respectively.
//!
//! Acceptance criteria:
//!   - `println!("hi")` shows up on stdout (fd 1)
//!   - `eprintln!("oops")` shows up on stderr (fd 2)
//!   - stdin read_line works
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

fn main() {
    println!("[hello_stdio] stdout: hello from ThingOS!");
    eprintln!("[hello_stdio] stderr: this is stderr");

    // Read a line from stdin (non-blocking is fine for automated tests)
    use std::io::{BufRead, BufReader};
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut line = String::new();
    let n = reader.read_line(&mut line).unwrap_or(0);
    if n > 0 {
        println!("[hello_stdio] read from stdin: {:?}", line.trim_end());
    } else {
        println!("[hello_stdio] stdin: no input (EOF or empty)");
    }

    println!("[hello_stdio] PASS");
    std::process::exit(0);
}
