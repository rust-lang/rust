//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print x
// gdb-check:$1 = 111102
// gdb-command:print y
// gdb-check:$2 = true
// gdb-command:continue

// gdb-command:print a
// gdb-check:$3 = 2000
// gdb-command:print b
// gdb-check:$4 = 3000
// gdb-command:continue

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v x
// lldb-check:[...] 111102
// lldb-command:v y
// lldb-check:[...] true
// lldb-command:continue

// lldb-command:v a
// lldb-check:[...] 2000
// lldb-command:v b
// lldb-check:[...] 3000
// lldb-command:continue

fn main() {

    fun(111102, true);
    nested(2000, 3000);

    fn nested(a: i32, b: i64) -> (i32, i64) {
        zzz(); // #break
        (a, b)
    }
}

fn fun(x: isize, y: bool) -> (isize, bool) {
    zzz(); // #break

    (x, y)
}

fn zzz() { () }
