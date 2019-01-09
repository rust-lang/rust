// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// BEFORE if
// gdb-command:print x
// gdb-check:$1 = 999
// gdb-command:print y
// gdb-check:$2 = -1
// gdb-command:continue

// AT BEGINNING of 'then' block
// gdb-command:print x
// gdb-check:$3 = 999
// gdb-command:print y
// gdb-check:$4 = -1
// gdb-command:continue

// AFTER 1st redeclaration of 'x'
// gdb-command:print x
// gdb-check:$5 = 1001
// gdb-command:print y
// gdb-check:$6 = -1
// gdb-command:continue

// AFTER 2st redeclaration of 'x'
// gdb-command:print x
// gdb-check:$7 = 1002
// gdb-command:print y
// gdb-check:$8 = 1003
// gdb-command:continue

// AFTER 1st if expression
// gdb-command:print x
// gdb-check:$9 = 999
// gdb-command:print y
// gdb-check:$10 = -1
// gdb-command:continue

// BEGINNING of else branch
// gdb-command:print x
// gdb-check:$11 = 999
// gdb-command:print y
// gdb-check:$12 = -1
// gdb-command:continue

// BEGINNING of else branch
// gdb-command:print x
// gdb-check:$13 = 1004
// gdb-command:print y
// gdb-check:$14 = 1005
// gdb-command:continue

// BEGINNING of else branch
// gdb-command:print x
// gdb-check:$15 = 999
// gdb-command:print y
// gdb-check:$16 = -1
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// BEFORE if
// lldb-command:print x
// lldbg-check:[...]$0 = 999
// lldbr-check:(i32) x = 999
// lldb-command:print y
// lldbg-check:[...]$1 = -1
// lldbr-check:(i32) y = -1
// lldb-command:continue

// AT BEGINNING of 'then' block
// lldb-command:print x
// lldbg-check:[...]$2 = 999
// lldbr-check:(i32) x = 999
// lldb-command:print y
// lldbg-check:[...]$3 = -1
// lldbr-check:(i32) y = -1
// lldb-command:continue

// AFTER 1st redeclaration of 'x'
// lldb-command:print x
// lldbg-check:[...]$4 = 1001
// lldbr-check:(i32) x = 1001
// lldb-command:print y
// lldbg-check:[...]$5 = -1
// lldbr-check:(i32) y = -1
// lldb-command:continue

// AFTER 2st redeclaration of 'x'
// lldb-command:print x
// lldbg-check:[...]$6 = 1002
// lldbr-check:(i32) x = 1002
// lldb-command:print y
// lldbg-check:[...]$7 = 1003
// lldbr-check:(i32) y = 1003
// lldb-command:continue

// AFTER 1st if expression
// lldb-command:print x
// lldbg-check:[...]$8 = 999
// lldbr-check:(i32) x = 999
// lldb-command:print y
// lldbg-check:[...]$9 = -1
// lldbr-check:(i32) y = -1
// lldb-command:continue

// BEGINNING of else branch
// lldb-command:print x
// lldbg-check:[...]$10 = 999
// lldbr-check:(i32) x = 999
// lldb-command:print y
// lldbg-check:[...]$11 = -1
// lldbr-check:(i32) y = -1
// lldb-command:continue

// BEGINNING of else branch
// lldb-command:print x
// lldbg-check:[...]$12 = 1004
// lldbr-check:(i32) x = 1004
// lldb-command:print y
// lldbg-check:[...]$13 = 1005
// lldbr-check:(i32) y = 1005
// lldb-command:continue

// BEGINNING of else branch
// lldb-command:print x
// lldbg-check:[...]$14 = 999
// lldbr-check:(i32) x = 999
// lldb-command:print y
// lldbg-check:[...]$15 = -1
// lldbr-check:(i32) y = -1
// lldb-command:continue

#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn main() {

    let x = 999;
    let y = -1;

    zzz(); // #break
    sentinel();

    if x < 1000 {
        zzz(); // #break
        sentinel();

        let x = 1001;

        zzz(); // #break
        sentinel();

        let x = 1002;
        let y = 1003;
        zzz(); // #break
        sentinel();
    } else {
        unreachable!();
    }

    zzz(); // #break
    sentinel();

    if x > 1000 {
        unreachable!();
    } else {
        zzz(); // #break
        sentinel();

        let x = 1004;
        let y = 1005;
        zzz(); // #break
        sentinel();
    }

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
