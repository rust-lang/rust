//@ aux-build:cross_crate_spans.rs
extern crate cross_crate_spans;

//@ compile-flags:-g
//@ disable-gdb-pretty-printers


// === GDB TESTS ===================================================================================

// gdb-command:break cross_crate_spans.rs:12
// gdb-command:run

// gdb-command:print result
// gdb-check:$1 = (17, 17)
// gdb-command:print a_variable
// gdb-check:$2 = 123456789
// gdb-command:print another_variable
// gdb-check:$3 = 123456789.5
// gdb-command:continue

// gdb-command:print result
// gdb-check:$4 = (1212, 1212)
// gdb-command:print a_variable
// gdb-check:$5 = 123456789
// gdb-command:print another_variable
// gdb-check:$6 = 123456789.5
// gdb-command:continue



// === LLDB TESTS ==================================================================================

// lldb-command:b cross_crate_spans.rs:12
// lldb-command:run

// lldb-command:v result
// lldb-check:[...] { 0 = 17 1 = 17 }
// lldb-command:v a_variable
// lldb-check:[...] 123456789
// lldb-command:v another_variable
// lldb-check:[...] 123456789.5
// lldb-command:continue

// lldb-command:v result
// lldb-check:[...] { 0 = 1212 1 = 1212 }
// lldb-command:v a_variable
// lldb-check:[...] 123456789
// lldb-command:v another_variable
// lldb-check:[...] 123456789.5
// lldb-command:continue


// This test makes sure that we can break in functions inlined from other crates.

fn main() {

    let _ = cross_crate_spans::generic_function(17u32);
    let _ = cross_crate_spans::generic_function(1212i16);

}
