//@ ignore-lldb

// Test that local macro debug info is not collapsed with #[collapse_debuginfo(external)]

//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#one_callsite[...]
// gdb-command:continue

fn one() {
    println!("one");
}

#[collapse_debuginfo(external)]
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
