// ignore-windows-gnu: pretty-printers are not loaded
// compile-flags:-g

// min-gdb-version: 8.1
// min-cdb-version: 10.0.18317.1001

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

// === CDB TESTS ==================================================================================

// cdb-command:g

// cdb-command:dx r,d
// cdb-check:r,d              : 42 [Type: alloc::rc::Rc<i32>]
// cdb-check:    [<Raw View>]     [Type: alloc::rc::Rc<i32>]
// cdb-check:    [Reference count] : 2 [Type: core::cell::Cell<usize>]
// cdb-check:    [Weak reference count] : 2 [Type: core::cell::Cell<usize>]

// cdb-command:dx r1,d
// cdb-check:r1,d             : 42 [Type: alloc::rc::Rc<i32>]
// cdb-check:    [<Raw View>]     [Type: alloc::rc::Rc<i32>]
// cdb-check:    [Reference count] : 2 [Type: core::cell::Cell<usize>]
// cdb-check:    [Weak reference count] : 2 [Type: core::cell::Cell<usize>]

// cdb-command:dx w1,d
// cdb-check:w1,d             : 42 [Type: alloc::rc::Weak<i32>]
// cdb-check:    [<Raw View>]     [Type: alloc::rc::Weak<i32>]
// cdb-check:    [Reference count] : 2 [Type: core::cell::Cell<usize>]
// cdb-check:    [Weak reference count] : 2 [Type: core::cell::Cell<usize>]

// cdb-command:dx a,d
// cdb-check:a,d              : 42 [Type: alloc::sync::Arc<i32>]
// cdb-check:    [<Raw View>]     [Type: alloc::sync::Arc<i32>]
// cdb-check:    [Reference count] : 2 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [Weak reference count] : 2 [Type: core::sync::atomic::AtomicUsize]

// cdb-command:dx a1,d
// cdb-check:a1,d             : 42 [Type: alloc::sync::Arc<i32>]
// cdb-check:    [<Raw View>]     [Type: alloc::sync::Arc<i32>]
// cdb-check:    [Reference count] : 2 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [Weak reference count] : 2 [Type: core::sync::atomic::AtomicUsize]

// cdb-command:dx w2,d
// cdb-check:w2,d             : 42 [Type: alloc::sync::Weak<i32>]
// cdb-check:    [<Raw View>]     [Type: alloc::sync::Weak<i32>]
// cdb-check:    [Reference count] : 2 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [Weak reference count] : 2 [Type: core::sync::atomic::AtomicUsize]

use std::rc::Rc;
use std::sync::Arc;

fn main() {
    let r = Rc::new(42);
    let r1 = Rc::clone(&r);
    let w1 = Rc::downgrade(&r);

    let a = Arc::new(42);
    let a1 = Arc::clone(&a);
    let w2 = Arc::downgrade(&a);

    zzz(); // #break
}

fn zzz() { () }
