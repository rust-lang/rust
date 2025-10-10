//@ compile-flags:-g
//@ disable-gdb-pretty-printers

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

// lldb-command:v x
// lldb-check:[...] -1
// lldb-command:v y
// lldb-check:[...] 1
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] -1
// lldb-command:v y
// lldb-check:[...] 2.5
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] -2.5
// lldb-command:v y
// lldb-check:[...] 1
// lldb-command:continue

// lldb-command:v x
// lldb-check:[...] -2.5
// lldb-command:v y
// lldb-check:[...] 2.5
// lldb-command:continue

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
