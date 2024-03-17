//@ min-lldb-version: 310

//@ compile-flags:-g

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
// lldbg-check:[...] 111102
// lldbr-check:(isize) x = 111102
// lldb-command:v y
// lldbg-check:[...] true
// lldbr-check:(bool) y = true
// lldb-command:continue

// lldb-command:v a
// lldbg-check:[...] 2000
// lldbr-check:(i32) a = 2000
// lldb-command:v b
// lldbg-check:[...] 3000
// lldbr-check:(i64) b = 3000
// lldb-command:continue


#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

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
