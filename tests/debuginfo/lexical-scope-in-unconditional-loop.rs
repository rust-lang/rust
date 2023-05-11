// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// FIRST ITERATION
// gdb-command:print x
// gdb-check:$1 = 0
// gdb-command:continue

// gdb-command:print x
// gdb-check:$2 = 1
// gdb-command:continue

// gdb-command:print x
// gdb-check:$3 = 101
// gdb-command:continue

// gdb-command:print x
// gdb-check:$4 = 101
// gdb-command:continue

// gdb-command:print x
// gdb-check:$5 = -987
// gdb-command:continue

// gdb-command:print x
// gdb-check:$6 = 101
// gdb-command:continue


// SECOND ITERATION
// gdb-command:print x
// gdb-check:$7 = 1
// gdb-command:continue

// gdb-command:print x
// gdb-check:$8 = 2
// gdb-command:continue

// gdb-command:print x
// gdb-check:$9 = 102
// gdb-command:continue

// gdb-command:print x
// gdb-check:$10 = 102
// gdb-command:continue

// gdb-command:print x
// gdb-check:$11 = -987
// gdb-command:continue

// gdb-command:print x
// gdb-check:$12 = 102
// gdb-command:continue

// gdb-command:print x
// gdb-check:$13 = 2
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// FIRST ITERATION
// lldb-command:print x
// lldbg-check:[...]$0 = 0
// lldbr-check:(i32) x = 0
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$1 = 1
// lldbr-check:(i32) x = 1
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$2 = 101
// lldbr-check:(i32) x = 101
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$3 = 101
// lldbr-check:(i32) x = 101
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$4 = -987
// lldbr-check:(i32) x = -987
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$5 = 101
// lldbr-check:(i32) x = 101
// lldb-command:continue


// SECOND ITERATION
// lldb-command:print x
// lldbg-check:[...]$6 = 1
// lldbr-check:(i32) x = 1
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$7 = 2
// lldbr-check:(i32) x = 2
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$8 = 102
// lldbr-check:(i32) x = 102
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$9 = 102
// lldbr-check:(i32) x = 102
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$10 = -987
// lldbr-check:(i32) x = -987
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$11 = 102
// lldbr-check:(i32) x = 102
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$12 = 2
// lldbr-check:(i32) x = 2
// lldb-command:continue

#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn main() {

    let mut x = 0;

    loop {
        if x >= 2 {
            break;
        }

        zzz(); // #break
        sentinel();

        x += 1;
        zzz(); // #break
        sentinel();

        // Shadow x
        let x = x + 100;
        zzz(); // #break
        sentinel();

        // open scope within loop's top level scope
        {
            zzz(); // #break
            sentinel();

            let x = -987;

            zzz(); // #break
            sentinel();
        }

        // Check that we get the x before the inner scope again
        zzz(); // #break
        sentinel();
    }

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
