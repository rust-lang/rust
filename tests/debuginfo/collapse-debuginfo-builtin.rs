// ignore-lldb
#![feature(collapse_debuginfo)]

use std::fmt;

// Test that builtin macro debug info is collapsed.
// Debug info for format_args must be #format_arg_external, not #format_arg_internal.
// Because format_args is a builtin macro.
// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:next 2
// gdb-command:frame
// gdb-check:[...]#format_arg_external[...]
// gdb-command:continue

fn main() {
    let ret = 0; // #break
    let w = "world".to_string();
    let s = fmt::format(
        format_args!( // #format_arg_external
            "hello {}", w // #format_arg_internal
        )
    ); // #format_callsite
    println!(
        "{}",
        s
    ); // #println_callsite
    std::process::exit(ret);
}
