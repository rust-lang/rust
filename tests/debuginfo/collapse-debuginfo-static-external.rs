//@ ignore-lldb

// Test that static debug info is not collapsed with #[collapse_debuginfo(external)]

//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:info line collapse_debuginfo_static_external::FOO
// gdb-check:[...]Line 15[...]

#[collapse_debuginfo(external)]
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
