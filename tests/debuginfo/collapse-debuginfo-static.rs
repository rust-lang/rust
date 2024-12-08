//@ ignore-lldb

// Test that static debug info is collapsed with #[collapse_debuginfo(yes)]

//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:info line collapse_debuginfo_static::FOO
// gdb-check:[...]Line 19[...]

#[collapse_debuginfo(yes)]
macro_rules! decl_foo {
    () => {
        static FOO: u32 = 0;
    };
}

decl_foo!();

fn main() {
    // prevent FOO from getting optimized out
    std::hint::black_box(&FOO);
}
