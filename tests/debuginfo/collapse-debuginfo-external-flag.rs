//@ ignore-lldb
#![feature(collapse_debuginfo)]

// Test that println macro debug info is collapsed with "collapse_macro_debuginfo=external" flag

//@ compile-flags:-g -Z collapse_macro_debuginfo=external

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#println_callsite[...]
// gdb-command:continue

macro_rules! outer {
    () => {
        println!("one"); // #println_callsite
    };
}

fn main() {
    let ret = 0; // #break
    outer!();
    std::process::exit(ret);
}
