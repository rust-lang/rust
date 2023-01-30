// min-lldb-version: 310

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print x
// gdb-check:$1 = -1
// gdb-command:print y
// gdb-check:$2 = 1
// gdb-command:continue

// gdb-command:print x
// gdb-check:$3 = -1
// gdb-command:print y
// gdb-check:$4 = 2.5
// gdb-command:continue

// gdb-command:print x
// gdb-check:$5 = -2.5
// gdb-command:print y
// gdb-check:$6 = 1
// gdb-command:continue

// gdb-command:print x
// gdb-check:$7 = -2.5
// gdb-command:print y
// gdb-check:$8 = 2.5
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print x
// lldbg-check:[...]$0 = -1
// lldbr-check:(i32) x = -1
// lldb-command:print y
// lldbg-check:[...]$1 = 1
// lldbr-check:(i32) y = 1
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$2 = -1
// lldbr-check:(i32) x = -1
// lldb-command:print y
// lldbg-check:[...]$3 = 2.5
// lldbr-check:(f64) y = 2.5
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$4 = -2.5
// lldbr-check:(f64) x = -2.5
// lldb-command:print y
// lldbg-check:[...]$5 = 1
// lldbr-check:(i32) y = 1
// lldb-command:continue

// lldb-command:print x
// lldbg-check:[...]$6 = -2.5
// lldbr-check:(f64) x = -2.5
// lldb-command:print y
// lldbg-check:[...]$7 = 2.5
// lldbr-check:(f64) y = 2.5
// lldb-command:continue


#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

fn outer<TA: Clone>(a: TA) {
    inner(a.clone(), 1);
    inner(a.clone(), 2.5f64);

    fn inner<TX, TY>(x: TX, y: TY) {
        zzz(); // #break
    }
}

fn main() {
    outer(-1);
    outer(-2.5f64);
}

fn zzz() { () }
