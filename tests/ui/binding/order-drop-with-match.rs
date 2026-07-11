//@ run-pass

// Test to make sure the destructors run in the right order.
// Each destructor sets it's tag in the corresponding entry
// in ORDER matching up to when it ran.
// Correct order is: matched, inner, outer

use std::sync::atomic::{AtomicUsize, Ordering};

static ORDER: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
static INDEX: AtomicUsize = AtomicUsize::new(0);

fn push_order(value: usize) {
    let index = INDEX.fetch_add(1, Ordering::Relaxed);
    ORDER[index].store(value, Ordering::Relaxed);
}

fn order() -> [usize; 3] {
    [
        ORDER[0].load(Ordering::Relaxed),
        ORDER[1].load(Ordering::Relaxed),
        ORDER[2].load(Ordering::Relaxed),
    ]
}

struct A;
impl Drop for A {
    fn drop(&mut self) {
        push_order(1);
    }
}

struct B;
impl Drop for B {
    fn drop(&mut self) {
        push_order(2);
    }
}

struct C;
impl Drop for C {
    fn drop(&mut self) {
        push_order(3);
    }
}

fn main() {
    {
        let matched = A;
        let _outer = C;
        {
            match matched {
                _s => {}
            }
            let _inner = B;
        }
    }
    assert_eq!(order(), [1, 2, 3]);
}
