//@ min-lldb-version: 310

//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:break constant_ordering_prologue::binding
// gdb-command:run

// gdb-command:print a
// gdb-check: = 19
// gdb-command:print b
// gdb-check: = 20
// gdb-command:print c
// gdb-check: = 21.5

// === LLDB TESTS ==================================================================================

// lldb-command:b "constant_ordering_prologue::binding"
// lldb-command:run

// lldb-command:print a
// lldb-check: 19
// lldb-command:print b
// lldb-check: 20
// lldb-command:print c
// lldb-check: 21.5

fn binding(a: i64, b: u64, c: f64) {
    let x = 0;
}

fn main() {
    binding(19, 20, 21.5);
}
