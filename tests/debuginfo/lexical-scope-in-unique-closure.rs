//@ min-lldb-version: 310

//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print x
// gdb-check:$1 = false
// gdb-command:continue

// gdb-command:print x
// gdb-check:$2 = false
// gdb-command:continue

// gdb-command:print x
// gdb-check:$3 = 1000
// gdb-command:continue

// gdb-command:print x
// gdb-check:$4 = 2.5
// gdb-command:continue

// gdb-command:print x
// gdb-check:$5 = true
// gdb-command:continue

// gdb-command:print x
// gdb-check:$6 = false
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v x
// lldbg-check:[...] false
// lldbr-check:(bool) x = false
// lldb-command:continue

// lldb-command:v x
// lldbg-check:[...] false
// lldbr-check:(bool) x = false
// lldb-command:continue

// lldb-command:v x
// lldbg-check:[...] 1000
// lldbr-check:(isize) x = 1000
// lldb-command:continue

// lldb-command:v x
// lldbg-check:[...] 2.5
// lldbr-check:(f64) x = 2.5
// lldb-command:continue

// lldb-command:v x
// lldbg-check:[...] true
// lldbr-check:(bool) x = true
// lldb-command:continue

// lldb-command:v x
// lldbg-check:[...] false
// lldbr-check:(bool) x = false
// lldb-command:continue


#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn main() {

    let x = false;

    zzz(); // #break
    sentinel();

    let unique_closure = |x:isize| {
        zzz(); // #break
        sentinel();

        let x = 2.5f64;

        zzz(); // #break
        sentinel();

        let x = true;

        zzz(); // #break
        sentinel();
    };

    zzz(); // #break
    sentinel();

    unique_closure(1000);

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
