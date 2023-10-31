// ignore-windows-gnu: pretty-printers are not loaded
// compile-flags:-g

// min-gdb-version: 8.1
// min-cdb-version: 10.0.18317.1001

// === GDB TESTS ==================================================================================

// gdb-command:run

// gdb-command:print rc
// gdb-check:[...]$1 = Rc(strong=11, weak=1) = {value = 111, strong = 11, weak = 1}
// gdb-command:print arc
// gdb-check:[...]$2 = Arc(strong=21, weak=1) = {value = 222, strong = 21, weak = 1}

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print rc
// lldb-check:[...]$0 = strong=11, weak=1 { value = 111 }
// lldb-command:print arc
// lldb-check:[...]$1 = strong=21, weak=1 { data = 222 }

// === CDB TESTS ==================================================================================

// cdb-command:g

// cdb-command:dx rc,d
// cdb-check:rc,d             : 111 [Type: alloc::rc::Rc<i32,alloc::alloc::Global>]
// cdb-check:    [Reference count] : 11 [Type: core::cell::Cell<usize>]
// cdb-check:    [Weak reference count] : 2 [Type: core::cell::Cell<usize>]

// cdb-command:dx weak_rc,d
// cdb-check:weak_rc,d        : 111 [Type: alloc::rc::Weak<i32,alloc::alloc::Global>]
// cdb-check:    [Reference count] : 11 [Type: core::cell::Cell<usize>]
// cdb-check:    [Weak reference count] : 2 [Type: core::cell::Cell<usize>]

// cdb-command:dx arc,d
// cdb-check:arc,d            : 222 [Type: alloc::sync::Arc<i32,alloc::alloc::Global>]
// cdb-check:    [Reference count] : 21 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [Weak reference count] : 2 [Type: core::sync::atomic::AtomicUsize]

// cdb-command:dx weak_arc,d
// cdb-check:weak_arc,d       : 222 [Type: alloc::sync::Weak<i32,alloc::alloc::Global>]
// cdb-check:    [Reference count] : 21 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [Weak reference count] : 2 [Type: core::sync::atomic::AtomicUsize]

// cdb-command:dx dyn_rc,d
// cdb-check:dyn_rc,d         [Type: alloc::rc::Rc<dyn$<core::fmt::Debug>,alloc::alloc::Global>]
// cdb-check:    [Reference count] : 31 [Type: core::cell::Cell<usize>]
// cdb-check:    [Weak reference count] : 2 [Type: core::cell::Cell<usize>]

// cdb-command:dx dyn_rc_weak,d
// cdb-check:dyn_rc_weak,d    [Type: alloc::rc::Weak<dyn$<core::fmt::Debug>,alloc::alloc::Global>]
// cdb-check:    [Reference count] : 31 [Type: core::cell::Cell<usize>]
// cdb-check:    [Weak reference count] : 2 [Type: core::cell::Cell<usize>]

// cdb-command:dx slice_rc,d
// cdb-check:slice_rc,d       : { len=3 } [Type: alloc::rc::Rc<slice2$<u32>,alloc::alloc::Global>]
// cdb-check:    [Length]         : 3 [Type: [...]]
// cdb-check:    [Reference count] : 41 [Type: core::cell::Cell<usize>]
// cdb-check:    [Weak reference count] : 2 [Type: core::cell::Cell<usize>]
// cdb-check:    [0]              : 1 [Type: u32]
// cdb-check:    [1]              : 2 [Type: u32]
// cdb-check:    [2]              : 3 [Type: u32]

// cdb-command:dx slice_rc_weak,d
// cdb-check:slice_rc_weak,d  : { len=3 } [Type: alloc::rc::Weak<slice2$<u32>,alloc::alloc::Global>]
// cdb-check:    [Length]         : 3 [Type: [...]]
// cdb-check:    [Reference count] : 41 [Type: core::cell::Cell<usize>]
// cdb-check:    [Weak reference count] : 2 [Type: core::cell::Cell<usize>]
// cdb-check:    [0]              : 1 [Type: u32]
// cdb-check:    [1]              : 2 [Type: u32]
// cdb-check:    [2]              : 3 [Type: u32]

// cdb-command:dx dyn_arc,d
// cdb-check:dyn_arc,d        [Type: alloc::sync::Arc<dyn$<core::fmt::Debug>,alloc::alloc::Global>]
// cdb-check:    [Reference count] : 51 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [Weak reference count] : 2 [Type: core::sync::atomic::AtomicUsize]

// cdb-command:dx dyn_arc_weak,d
// cdb-check:dyn_arc_weak,d   [Type: alloc::sync::Weak<dyn$<core::fmt::Debug>,alloc::alloc::Global>]
// cdb-check:    [Reference count] : 51 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [Weak reference count] : 2 [Type: core::sync::atomic::AtomicUsize]

// cdb-command:dx slice_arc,d
// cdb-check:slice_arc,d      : { len=3 } [Type: alloc::sync::Arc<slice2$<u32>,alloc::alloc::Global>]
// cdb-check:    [Length]         : 3 [Type: [...]]
// cdb-check:    [Reference count] : 61 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [Weak reference count] : 2 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [0]              : 4 [Type: u32]
// cdb-check:    [1]              : 5 [Type: u32]
// cdb-check:    [2]              : 6 [Type: u32]

// cdb-command:dx slice_arc_weak,d
// cdb-check:slice_arc_weak,d : { len=3 } [Type: alloc::sync::Weak<slice2$<u32>,alloc::alloc::Global>]
// cdb-check:    [Length]         : 3 [Type: [...]]
// cdb-check:    [Reference count] : 61 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [Weak reference count] : 2 [Type: core::sync::atomic::AtomicUsize]
// cdb-check:    [0]              : 4 [Type: u32]
// cdb-check:    [1]              : 5 [Type: u32]
// cdb-check:    [2]              : 6 [Type: u32]

use std::fmt::Debug;
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    let rc = Rc::new(111);
    inc_ref_count(&rc, 10);
    let weak_rc = Rc::downgrade(&rc);

    let arc = Arc::new(222);
    inc_ref_count(&arc, 20);
    let weak_arc = Arc::downgrade(&arc);

    let dyn_rc: Rc<dyn Debug> = Rc::new(333);
    inc_ref_count(&dyn_rc, 30);
    let dyn_rc_weak = Rc::downgrade(&dyn_rc);

    let slice_rc: Rc<[u32]> = Rc::from(vec![1, 2, 3]);
    inc_ref_count(&slice_rc, 40);
    let slice_rc_weak = Rc::downgrade(&slice_rc);

    let dyn_arc: Arc<dyn Debug> = Arc::new(444);
    inc_ref_count(&dyn_arc, 50);
    let dyn_arc_weak = Arc::downgrade(&dyn_arc);

    let slice_arc: Arc<[u32]> = Arc::from(vec![4, 5, 6]);
    inc_ref_count(&slice_arc, 60);
    let slice_arc_weak = Arc::downgrade(&slice_arc);

    zzz(); // #break
}

fn inc_ref_count<T: Clone>(rc: &T, count: usize) {
    for _ in 0..count {
        std::mem::forget(rc.clone());
    }
}

fn zzz() {
    ()
}
