// min-lldb-version: 310
// ignore-lldb FIXME #48807

// compile-flags:-g -Zdebug-macros

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print a
// gdb-check:$1 = 10
// gdb-command:print b
// gdb-check:$2 = 34
// gdb-command:continue

// gdb-command:print a
// gdb-check:$3 = 890242
// gdb-command:print b
// gdb-check:$4 = 34
// gdb-command:continue

// gdb-command:print a
// gdb-check:$5 = 10
// gdb-command:print b
// gdb-check:$6 = 34
// gdb-command:continue

// gdb-command:print a
// gdb-check:$7 = 102
// gdb-command:print b
// gdb-check:$8 = 34
// gdb-command:continue

// gdb-command:print a
// gdb-check:$9 = 110
// gdb-command:print b
// gdb-check:$10 = 34
// gdb-command:continue

// gdb-command:print a
// gdb-check:$11 = 10
// gdb-command:print b
// gdb-check:$12 = 34
// gdb-command:continue

// gdb-command:print a
// gdb-check:$13 = 10
// gdb-command:print b
// gdb-check:$14 = 34
// gdb-command:print c
// gdb-check:$15 = 400
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print a
// lldbg-check:[...]$0 = 10
// lldbr-check:(i32) a = 10
// lldb-command:print b
// lldbg-check:[...]$1 = 34
// lldbr-check:(i32) b = 34
// lldb-command:continue

// lldb-command:print a
// lldbg-check:[...]$2 = 890242
// lldbr-check:(i32) a = 10
// lldb-command:print b
// lldbg-check:[...]$3 = 34
// lldbr-check:(i32) b = 34
// lldb-command:continue

// lldb-command:print a
// lldbg-check:[...]$4 = 10
// lldbr-check:(i32) a = 10
// lldb-command:print b
// lldbg-check:[...]$5 = 34
// lldbr-check:(i32) b = 34
// lldb-command:continue

// lldb-command:print a
// lldbg-check:[...]$6 = 102
// lldbr-check:(i32) a = 10
// lldb-command:print b
// lldbg-check:[...]$7 = 34
// lldbr-check:(i32) b = 34
// lldb-command:continue

// Don't test this with rust-enabled lldb for now; see issue #48807
// lldbg-command:print a
// lldbg-check:[...]$8 = 110
// lldbg-command:print b
// lldbg-check:[...]$9 = 34
// lldbg-command:continue

// lldbg-command:print a
// lldbg-check:[...]$10 = 10
// lldbg-command:print b
// lldbg-check:[...]$11 = 34
// lldbg-command:continue

// lldbg-command:print a
// lldbg-check:[...]$12 = 10
// lldbg-command:print b
// lldbg-check:[...]$13 = 34
// lldbg-command:print c
// lldbg-check:[...]$14 = 400
// lldbg-command:continue


#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

macro_rules! trivial {
    ($e1:expr) => ($e1)
}

macro_rules! no_new_scope {
    ($e1:expr) => (($e1 + 2) - 1)
}

macro_rules! new_scope {
    () => ({
        let a = 890242;
        zzz(); // #break
        sentinel();
    })
}

macro_rules! shadow_within_macro {
    ($e1:expr) => ({
        let a = $e1 + 2;

        zzz(); // #break
        sentinel();

        let a = $e1 + 10;

        zzz(); // #break
        sentinel();
    })
}


macro_rules! dup_expr {
    ($e1:expr) => (($e1) + ($e1))
}


fn main() {

    let a = trivial!(10);
    let b = no_new_scope!(33);

    zzz(); // #break
    sentinel();

    new_scope!();

    zzz(); // #break
    sentinel();

    shadow_within_macro!(100);

    zzz(); // #break
    sentinel();

    let c = dup_expr!(10 * 20);

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
