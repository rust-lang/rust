//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print self
// gdb-check:$1 = 1111
// gdb-command:continue

// gdb-command:print self
// gdb-check:$2 = by_value_self_argument_in_trait_impl::Struct {x: 2222, y: 3333}
// gdb-command:continue

// gdb-command:print self
// gdb-check:$3 = (4444.5, 5555, 6666, 7777.5)
// gdb-command:continue


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v self
// lldb-check:[...] 1111
// lldb-command:continue

// lldb-command:v self
// lldb-check:[...] { x = 2222 y = 3333 }
// lldb-command:continue

// lldb-command:v self
// lldb-check:[...] { 0 = 4444.5 1 = 5555 2 = 6666 3 = 7777.5 }
// lldb-command:continue

trait Trait {
    fn method(self) -> Self;
}

impl Trait for isize {
    fn method(self) -> isize {
        zzz(); // #break
        self
    }
}

struct Struct {
    x: usize,
    y: usize,
}

impl Trait for Struct {
    fn method(self) -> Struct {
        zzz(); // #break
        self
    }
}

impl Trait for (f64, isize, isize, f64) {
    fn method(self) -> (f64, isize, isize, f64) {
        zzz(); // #break
        self
    }
}

fn main() {
    let _ = (1111 as isize).method();
    let _ = Struct { x: 2222, y: 3333 }.method();
    let _ = (4444.5, 5555, 6666, 7777.5).method();
}

fn zzz() { () }
