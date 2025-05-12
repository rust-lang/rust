//@ ignore-lldb

// Test that macro attribute #[collapse_debuginfo(no)]
// overrides "collapse_macro_debuginfo=external" flag

//@ compile-flags:-g -C collapse_macro_debuginfo=external

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#one_callsite[...]
// gdb-command:next
// gdb-command:frame
// gdb-command:continue

fn one() {
    println!("one");
}

#[collapse_debuginfo(no)]
macro_rules! outer {
    () => {
        one(); // #one_callsite
    };
}

fn main() {
    let ret = 0; // #break
    outer!();
    std::process::exit(ret);
}
