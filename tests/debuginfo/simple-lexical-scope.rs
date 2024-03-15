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
// gdb-check:$3 = 10
// gdb-command:continue

// gdb-command:print x
// gdb-check:$4 = 10
// gdb-command:continue

// gdb-command:print x
// gdb-check:$5 = 10.5
// gdb-command:continue

// gdb-command:print x
// gdb-check:$6 = 10
// gdb-command:continue

// gdb-command:print x
// gdb-check:$7 = false
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
// lldbg-check:[...] 10
// lldbr-check:(i32) x = 10
// lldb-command:continue

// lldb-command:v x
// lldbg-check:[...] 10
// lldbr-check:(i32) x = 10
// lldb-command:continue

// lldb-command:v x
// lldbg-check:[...] 10.5
// lldbr-check:(f64) x = 10.5
// lldb-command:continue

// lldb-command:v x
// lldbg-check:[...] 10
// lldbr-check:(i32) x = 10
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

    {
        zzz(); // #break
        sentinel();

        let x = 10;

        zzz(); // #break
        sentinel();

        {
            zzz(); // #break
            sentinel();

            let x = 10.5f64;

            zzz(); // #break
            sentinel();
        }

        zzz(); // #break
        sentinel();
    }

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
