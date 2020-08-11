// ignore-windows pretty-printers are not loaded
// compile-flags:-g

// min-gdb-version: 8.1

// === GDB TESTS ==================================================================================

// gdb-command:run

// gdb-command:print r
// gdb-check:[...]$1 = Rc(strong=2, weak=1) = {value = 42, strong = 2, weak = 1}
// gdb-command:print a
// gdb-check:[...]$2 = Arc(strong=2, weak=1) = {value = 42, strong = 2, weak = 1}


// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print r
// lldb-check:[...]$0 = strong=2, weak=1 { value = 42 }
// lldb-command:print a
// lldb-check:[...]$1 = strong=2, weak=1 { data = 42 }

use std::rc::Rc;
use std::sync::Arc;

fn main() {
    let r = Rc::new(42);
    let r1 = Rc::clone(&r);
    let w1 = Rc::downgrade(&r);

    let a = Arc::new(42);
    let a1 = Arc::clone(&a);
    let w2 = Arc::downgrade(&a);

    print!(""); // #break
}
