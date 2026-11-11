use super::*;
use crate::cell::Cell;
use crate::pin::Pin;

/// All lists are constructed by `WaitVariable::new`; this test
/// stand-in likewise initializes the list before pinning it.
fn new_list<T>() -> Pin<Box<UnsafeList<T>>> {
    // SAFETY: `init` is called below, before the list is otherwise used or
    // dropped.
    let mut list = Box::new(unsafe { UnsafeList::new() });
    list.init();
    Box::into_pin(list)
}

/// # Safety
/// List must be valid.
unsafe fn assert_empty<T>(list: Pin<&mut UnsafeList<T>>) {
    assert!(unsafe { list.pop() }.is_none(), "assertion failed: list is not empty");
}

#[test]
fn init_empty() {
    unsafe {
        assert_empty(new_list::<i32>().as_mut());
    }
}

#[test]
fn push_pop() {
    unsafe {
        let mut node = UnsafeListEntry::new(1234);
        let mut list = new_list();
        assert_eq!(list.as_mut().push(&mut node), &1234);
        assert_eq!(list.as_mut().pop().unwrap(), &1234);
        assert_empty(list.as_mut());
    }
}

#[test]
fn push_remove() {
    unsafe {
        let mut node = UnsafeListEntry::new(1234);
        let mut list = new_list();
        assert_eq!(list.as_mut().push(&mut node), &1234);
        list.as_mut().remove(&mut node);
        assert_empty(list.as_mut());
    }
}

#[test]
fn push_remove_pop() {
    unsafe {
        let mut node1 = UnsafeListEntry::new(11);
        let mut node2 = UnsafeListEntry::new(12);
        let mut node3 = UnsafeListEntry::new(13);
        let mut node4 = UnsafeListEntry::new(14);
        let mut node5 = UnsafeListEntry::new(15);
        let mut list = new_list();
        assert_eq!(list.as_mut().push(&mut node1), &11);
        assert_eq!(list.as_mut().push(&mut node2), &12);
        assert_eq!(list.as_mut().push(&mut node3), &13);
        assert_eq!(list.as_mut().push(&mut node4), &14);
        assert_eq!(list.as_mut().push(&mut node5), &15);

        list.as_mut().remove(&mut node1);
        assert_eq!(list.as_mut().pop().unwrap(), &12);
        list.as_mut().remove(&mut node3);
        assert_eq!(list.as_mut().pop().unwrap(), &14);
        list.as_mut().remove(&mut node5);
        assert_empty(list.as_mut());

        assert_eq!(list.as_mut().push(&mut node1), &11);
        assert_eq!(list.as_mut().pop().unwrap(), &11);
        assert_empty(list.as_mut());

        assert_eq!(list.as_mut().push(&mut node3), &13);
        assert_eq!(list.as_mut().push(&mut node4), &14);
        list.as_mut().remove(&mut node3);
        list.as_mut().remove(&mut node4);
        assert_empty(list.as_mut());
    }
}

#[test]
fn complex_pushes_pops() {
    unsafe {
        let mut node1 = UnsafeListEntry::new(1234);
        let mut node2 = UnsafeListEntry::new(4567);
        let mut node3 = UnsafeListEntry::new(9999);
        let mut node4 = UnsafeListEntry::new(8642);
        let mut list = new_list();
        list.as_mut().push(&mut node1);
        list.as_mut().push(&mut node2);
        assert_eq!(list.as_mut().pop().unwrap(), &1234);
        list.as_mut().push(&mut node3);
        assert_eq!(list.as_mut().pop().unwrap(), &4567);
        assert_eq!(list.as_mut().pop().unwrap(), &9999);
        assert_empty(list.as_mut());
        list.as_mut().push(&mut node4);
        assert_eq!(list.as_mut().pop().unwrap(), &8642);
        assert_empty(list.as_mut());
    }
}

#[test]
fn cell() {
    unsafe {
        let mut node = UnsafeListEntry::new(Cell::new(0));
        let mut list = new_list();
        let noderef = list.as_mut().push(&mut node);
        assert_eq!(noderef.get(), 0);
        list.as_mut().pop().unwrap().set(1);
        assert_empty(list.as_mut());
        assert_eq!(noderef.get(), 1);
    }
}

// Regression tests for the aliasing issues in rust-lang/rust#160603,
// exercising the usage patterns of the SGX `WaitQueue`. `hostile_reborrow`
// mirrors safe code reborrowing the structure containing the list between
// list operations (as `WaitVariable::lock_var_mut` and the pin projections
// do).

struct Wrapper<T> {
    list: UnsafeList<T>,
    other: u32,
}

impl<T> Wrapper<T> {
    fn new() -> Pin<Box<Wrapper<T>>> {
        // SAFETY: `init` is called below, before the list is otherwise used
        // or dropped.
        let mut wrapper = Box::new(Wrapper { list: unsafe { UnsafeList::new() }, other: 0 });
        wrapper.list.init();
        Box::into_pin(wrapper)
    }

    fn list(self: Pin<&mut Self>) -> Pin<&mut UnsafeList<T>> {
        // SAFETY: `list` is structurally pinned: a pinned `Wrapper` pins it,
        // and it is never moved out of it.
        unsafe { self.map_unchecked_mut(|this| &mut this.list) }
    }

    fn hostile_reborrow(self: Pin<&mut Self>) {
        // SAFETY: nothing is moved; `other` is not structurally pinned.
        let this = unsafe { self.get_unchecked_mut() };
        this.other = this.other.wrapping_add(1);
    }
}

// The `wait_timeout` fallback path: push an entry, use the returned
// reference, then remove the entry.
#[test]
fn wait_timeout_fallback() {
    unsafe {
        let mut w = Wrapper::new();
        let mut entry = UnsafeListEntry::new(1234);
        let value = w.as_mut().list().push(&mut entry);
        assert_eq!(*value, 1234);

        w.as_mut().hostile_reborrow();

        // Not woken up: remove our own entry, as `wait_timeout` does.
        w.as_mut().list().remove(&mut entry);
        assert_empty(w.as_mut().list());
    }
}

// Removing the first entry while others are present.
#[test]
fn remove_first_of_many() {
    unsafe {
        let mut w = Wrapper::new();
        let mut e1 = UnsafeListEntry::new(1);
        let mut e2 = UnsafeListEntry::new(2);
        let mut e3 = UnsafeListEntry::new(3);
        w.as_mut().list().push(&mut e1);
        w.as_mut().list().push(&mut e2);
        w.as_mut().list().push(&mut e3);
        w.as_mut().list().remove(&mut e1);
        assert_eq!(w.as_mut().list().pop().unwrap(), &2);
        assert_eq!(w.as_mut().list().pop().unwrap(), &3);
        assert_empty(w.as_mut().list());
    }
}

// Entries pushed from different "stack frames" and popped by a "notifier"
// (like `notify_all`), with hostile reborrows between every operation.
#[test]
fn notify_all_pattern() {
    unsafe {
        let mut w = Wrapper::new();
        let mut e1 = UnsafeListEntry::new(1);
        let mut e2 = UnsafeListEntry::new(2);
        w.as_mut().list().push(&mut e1);
        w.as_mut().hostile_reborrow();
        w.as_mut().list().push(&mut e2);
        w.as_mut().hostile_reborrow();

        let mut count = 0;
        while let Some(v) = w.as_mut().list().pop() {
            count += *v;
            w.as_mut().hostile_reborrow();
        }
        assert_eq!(count, 3);
    }
}

// Empty-list churn: repeated push/pop cycles with reborrows in between.
#[test]
fn empty_churn() {
    unsafe {
        let mut w = Wrapper::new();
        for i in 0..4 {
            let mut e = UnsafeListEntry::new(i);
            w.as_mut().list().push(&mut e);
            w.as_mut().hostile_reborrow();
            assert_eq!(w.as_mut().list().pop().unwrap(), &i);
            w.as_mut().hostile_reborrow();
            assert!(w.list.is_empty());
        }
    }
}

// Cross-thread `wait`/`notify_one` pattern: the waiting thread pushes a
// stack-allocated entry and keeps reading through the reference returned by
// `push` while the notifying thread pops the entry and stores through the
// reference returned by `pop`.
#[test]
fn cross_thread_wait_notify() {
    use crate::sync::atomic::{AtomicBool, Ordering};
    use crate::sync::{Arc, Mutex};
    use crate::thread;

    struct Queue {
        list: UnsafeList<AtomicBool>,
    }
    // SAFETY: like the real `WaitQueue`, the list is only accessed while
    // holding the mutex.
    unsafe impl Send for Queue {}

    let queue = Arc::new(Mutex::new(Queue {
        // SAFETY: `init` is called below, before the list is otherwise used
        // or dropped.
        list: unsafe { UnsafeList::new() },
    }));
    queue.lock().unwrap().list.init();

    for _ in 0..3 {
        let waiter = {
            let queue = Arc::clone(&queue);
            thread::spawn(move || {
                let mut entry = UnsafeListEntry::new(AtomicBool::new(false));
                let mut guard = queue.lock().unwrap();
                // SAFETY: the list lives in the heap allocation behind the
                // `Arc` and is never moved.
                let list = unsafe { Pin::new_unchecked(&mut guard.list) };
                // SAFETY: `entry` is only dropped after the notifier popped
                // it and set the flag, and is not otherwise accessed while it
                // is in the list.
                let wake = unsafe { list.push(&mut entry) };
                drop(guard);
                while !wake.load(Ordering::Acquire) {
                    thread::yield_now();
                }
            })
        };
        loop {
            let mut guard = queue.lock().unwrap();
            // SAFETY: the list lives in the heap allocation behind the `Arc`
            // and is never moved.
            let list = unsafe { Pin::new_unchecked(&mut guard.list) };
            // SAFETY: the entry is not deallocated until the waiting thread
            // observes the flag, which is only set below.
            if let Some(wake) = unsafe { list.pop() } {
                // Set under the queue lock, like `notify_one`.
                wake.store(true, Ordering::Release);
                break;
            }
            drop(guard);
            thread::yield_now();
        }
        waiter.join().unwrap();
    }
}
