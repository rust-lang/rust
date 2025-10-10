//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print constant
// gdb-check:$1 = 1
// gdb-command:print a_struct
// gdb-check:$2 = var_captured_in_sendable_closure::Struct {a: -2, b: 3.5, c: 4}
// gdb-command:print *owned
// gdb-check:$3 = 5
// gdb-command:continue

// gdb-command:print constant2
// gdb-check:$4 = 6
// gdb-command:continue

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v constant
// lldb-check:[...] 1
// lldb-command:v a_struct
// lldb-check:[...] { a = -2 b = 3.5 c = 4 }
// lldb-command:v *owned
// lldb-check:[...] 5

#![allow(unused_variables)]

struct Struct {
    a: isize,
    b: f64,
    c: usize
}

fn main() {
    let constant = 1;

    let a_struct = Struct {
        a: -2,
        b: 3.5,
        c: 4
    };

    let owned: Box<_> = Box::new(5);

    let closure = move || {
        zzz(); // #break
        do_something(&constant, &a_struct.a, &*owned);
    };

    closure();

    let constant2 = 6_usize;

    // The `self` argument of the following closure should be passed by value
    // to FnOnce::call_once(self, args), which gets codegened a bit differently
    // than the regular case. Let's make sure this is supported too.
    let immedate_env = move || {
        zzz(); // #break
        return constant2;
    };

    immedate_env();
}

fn do_something(_: &isize, _:&isize, _:&isize) {

}

fn zzz() {()}
