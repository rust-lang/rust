//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:print a
// gdb-check:$1 = 5
// gdb-command:print c
// gdb-check:$2 = 6
// gdb-command:print d
// gdb-check:$3 = 7
// gdb-command:continue
// gdb-command:print a
// gdb-check:$4 = 7
// gdb-command:print c
// gdb-check:$5 = 6
// gdb-command:print e
// gdb-check:$6 = 8
// gdb-command:continue
// gdb-command:print a
// gdb-check:$7 = 8
// gdb-command:print c
// gdb-check:$8 = 6

// === LLDB TESTS ==================================================================================

// lldb-command:run
// lldb-command:v a
// lldb-check:(int) a = 5
// lldb-command:v c
// lldb-check:(int) c = 6
// lldb-command:v d
// lldb-check:(int) d = 7
// lldb-command:continue
// lldb-command:v a
// lldb-check:(int) a = 7
// lldb-command:v c
// lldb-check:(int) c = 6
// lldb-command:v e
// lldb-check:(int) e = 8
// lldb-command:continue
// lldb-command:v a
// lldb-check:(int) a = 8
// lldb-command:v c
// lldb-check:(int) c = 6

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    let mut a = 5;
    let mut b = #[coroutine]
    || {
        let c = 6; // Live across multiple yield points

        let d = 7; // Live across only one yield point
        yield;
        _zzz(); // #break
        a = d;

        let e = 8; // Live across zero yield points
        _zzz(); // #break
        a = e;

        yield;
        _zzz(); // #break
        a = c;
    };
    Pin::new(&mut b).resume(());
    Pin::new(&mut b).resume(());
    Pin::new(&mut b).resume(());
    _zzz(); // #break
}

fn _zzz() {
    ()
}
