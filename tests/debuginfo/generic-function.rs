//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print *t0
// gdb-check:$1 = 1
// gdb-command:print *t1
// gdb-check:$2 = 2.5
// gdb-command:continue

// gdb-command:print *t0
// gdb-check:$3 = 3.5
// gdb-command:print *t1
// gdb-check:$4 = 4
// gdb-command:continue

// gdb-command:print *t0
// gdb-check:$5 = 5
// gdb-command:print *t1
// gdb-check:$6 = generic_function::Struct {a: 6, b: 7.5}
// gdb-command:continue

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v *t0
// lldb-check:[...] 1
// lldb-command:v *t1
// lldb-check:[...] 2.5
// lldb-command:continue

// lldb-command:v *t0
// lldb-check:[...] 3.5
// lldb-command:v *t1
// lldb-check:[...] 4
// lldb-command:continue

// lldb-command:v *t0
// lldb-check:[...] 5
// lldb-command:v *t1
// lldb-check:[...] { a = 6 b = 7.5 }
// lldb-command:continue

#[derive(Clone)]
struct Struct {
    a: isize,
    b: f64
}

fn dup_tup<T0: Clone, T1: Clone>(t0: &T0, t1: &T1) -> ((T0, T1), (T1, T0)) {
    let ret = ((t0.clone(), t1.clone()), (t1.clone(), t0.clone()));
    zzz(); // #break
    ret
}

fn main() {

    let _ = dup_tup(&1, &2.5f64);
    let _ = dup_tup(&3.5f64, &4_u16);
    let _ = dup_tup(&5, &Struct { a: 6, b: 7.5 });
}

fn zzz() {()}
