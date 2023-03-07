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
// lldbg-check:[...]$0 = false
// lldbr-check:(bool) x = false
// lldb-command:print y
// lldbg-check:[...]$1 = true
// lldbr-check:(bool) y = true
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$2 = 10
// lldbr-check:(i32) x = 10
// lldb-command:print y
// lldbg-check:[...]$3 = true
// lldbr-check:(bool) y = true
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$4 = 10.5
// lldbr-check:(f64) x = 10.5
// lldb-command:print y
// lldbg-check:[...]$5 = 20
// lldbr-check:(i32) y = 20
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
