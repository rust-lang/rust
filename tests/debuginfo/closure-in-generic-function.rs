//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print x
// gdb-check:$1 = 0.5
// gdb-command:print y
// gdb-check:$2 = 10
// gdb-command:continue

// gdb-command:print *x
// gdb-check:$3 = 29
// gdb-command:print *y
// gdb-check:$4 = 110
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v x
// lldb-check:[...] 0.5
// lldb-command:v y
// lldb-check:[...] 10
// lldb-command:continue

// lldb-command:v *x
// lldb-check:[...] 29
// lldb-command:v *y
// lldb-check:[...] 110
// lldb-command:continue

fn some_generic_fun<T1, T2>(a: T1, b: T2) -> (T2, T1) {

    let closure = |x, y| {
        zzz(); // #break
        (y, x)
    };

    closure(a, b)
}

fn main() {
    some_generic_fun(0.5f64, 10);
    some_generic_fun(&29, Box::new(110));
}

fn zzz() { () }
