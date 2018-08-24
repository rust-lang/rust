// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print x
// gdb-check:$1 = false
// gdb-command:print y
// gdb-check:$2 = true
// gdb-command:continue

// gdb-command:print x
// gdb-check:$3 = 10
// gdb-command:print y
// gdb-check:$4 = true
// gdb-command:continue

// gdb-command:print x
// gdb-check:$5 = 10.5
// gdb-command:print y
// gdb-check:$6 = 20
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print x
// lldb-check:[...]$0 = false
// lldb-command:print y
// lldb-check:[...]$1 = true
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$2 = 10
// lldb-command:print y
// lldb-check:[...]$3 = true
// lldb-command:continue

// lldb-command:print x
// lldb-check:[...]$4 = 10.5
// lldb-command:print y
// lldb-check:[...]$5 = 20
// lldb-command:continue


#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn a_function(x: bool, y: bool) {
    zzz(); // #break
    sentinel();

    let x = 10;

    zzz(); // #break
    sentinel();

    let x = 10.5f64;
    let y = 20;

    zzz(); // #break
    sentinel();
}

fn main() {
    a_function(false, true);
}

fn zzz() {()}
fn sentinel() {()}
