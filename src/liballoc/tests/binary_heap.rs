use std::cmp;
use std::collections::BinaryHeap;
use std::collections::binary_heap::{Drain, PeekMut};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};

use rand::{thread_rng, seq::SliceRandom};

#[test]
fn test_iterator() {
    let data = vec![5, 9, 3];
    let iterout = [9, 5, 3];
    let heap = BinaryHeap::from(data);
    let mut i = 0;
    for el in &heap {
        assert_eq!(*el, iterout[i]);
        i += 1;
    }
}

#[test]
fn test_iterator_reverse() {
    let data = vec![5, 9, 3];
    let iterout = vec![3, 5, 9];
    let pq = BinaryHeap::from(data);

    let v: Vec<_> = pq.iter().rev().cloned().collect();
    assert_eq!(v, iterout);
}

#[test]
fn test_move_iter() {
    let data = vec![5, 9, 3];
    let iterout = vec![9, 5, 3];
    let pq = BinaryHeap::from(data);

    let v: Vec<_> = pq.into_iter().collect();
    assert_eq!(v, iterout);
}

#[test]
fn test_move_iter_size_hint() {
    let data = vec![5, 9];
    let pq = BinaryHeap::from(data);

    let mut it = pq.into_iter();

    assert_eq!(it.size_hint(), (2, Some(2)));
    assert_eq!(it.next(), Some(9));

    assert_eq!(it.size_hint(), (1, Some(1)));
    assert_eq!(it.next(), Some(5));

    assert_eq!(it.size_hint(), (0, Some(0)));
    assert_eq!(it.next(), None);
}

#[test]
fn test_move_iter_reverse() {
    let data = vec![5, 9, 3];
    let iterout = vec![3, 5, 9];
    let pq = BinaryHeap::from(data);

    let v: Vec<_> = pq.into_iter().rev().collect();
    assert_eq!(v, iterout);
}

#[test]
fn test_peek_and_pop() {
    let data = vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
    let mut sorted = data.clone();
    sorted.sort();
    let mut heap = BinaryHeap::from(data);
    while !heap.is_empty() {
        assert_eq!(heap.peek().unwrap(), sorted.last().unwrap());
        assert_eq!(heap.pop().unwrap(), sorted.pop().unwrap());
    }
}

#[test]
fn test_peek_mut() {
    let data = vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
    let mut heap = BinaryHeap::from(data);
    assert_eq!(heap.peek(), Some(&10));
    {
        let mut top = heap.peek_mut().unwrap();
        *top -= 2;
    }
    assert_eq!(heap.peek(), Some(&9));
}

#[test]
fn test_peek_mut_pop() {
    let data = vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
    let mut heap = BinaryHeap::from(data);
    assert_eq!(heap.peek(), Some(&10));
    {
        let mut top = heap.peek_mut().unwrap();
        *top -= 2;
        assert_eq!(PeekMut::pop(top), 8);
    }
    assert_eq!(heap.peek(), Some(&9));
}

#[test]
fn test_push() {
    let mut heap = BinaryHeap::from(vec![2, 4, 9]);
    assert_eq!(heap.len(), 3);
    assert!(*heap.peek().unwrap() == 9);
    heap.push(11);
    assert_eq!(heap.len(), 4);
    assert!(*heap.peek().unwrap() == 11);
    heap.push(5);
    assert_eq!(heap.len(), 5);
    assert!(*heap.peek().unwrap() == 11);
    heap.push(27);
    assert_eq!(heap.len(), 6);
    assert!(*heap.peek().unwrap() == 27);
    heap.push(3);
    assert_eq!(heap.len(), 7);
    assert!(*heap.peek().unwrap() == 27);
    heap.push(103);
    assert_eq!(heap.len(), 8);
    assert!(*heap.peek().unwrap() == 103);
}

#[test]
fn test_push_unique() {
    let mut heap = BinaryHeap::<Box<_>>::from(vec![box 2, box 4, box 9]);
    assert_eq!(heap.len(), 3);
    assert!(**heap.peek().unwrap() == 9);
    heap.push(box 11);
    assert_eq!(heap.len(), 4);
    assert!(**heap.peek().unwrap() == 11);
    heap.push(box 5);
    assert_eq!(heap.len(), 5);
    assert!(**heap.peek().unwrap() == 11);
    heap.push(box 27);
    assert_eq!(heap.len(), 6);
    assert!(**heap.peek().unwrap() == 27);
    heap.push(box 3);
    assert_eq!(heap.len(), 7);
    assert!(**heap.peek().unwrap() == 27);
    heap.push(box 103);
    assert_eq!(heap.len(), 8);
    assert!(**heap.peek().unwrap() == 103);
}

fn check_to_vec(mut data: Vec<i32>) {
    let heap = BinaryHeap::from(data.clone());
    let mut v = heap.clone().into_vec();
    v.sort();
    data.sort();

    assert_eq!(v, data);
    assert_eq!(heap.into_sorted_vec(), data);
}

#[test]
fn test_to_vec() {
    check_to_vec(vec![]);
    check_to_vec(vec![5]);
    check_to_vec(vec![3, 2]);
    check_to_vec(vec![2, 3]);
    check_to_vec(vec![5, 1, 2]);
    check_to_vec(vec![1, 100, 2, 3]);
    check_to_vec(vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 0]);
    check_to_vec(vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1]);
    check_to_vec(vec![9, 11, 9, 9, 9, 9, 11, 2, 3, 4, 11, 9, 0, 0, 0, 0]);
    check_to_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    check_to_vec(vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    check_to_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 2]);
    check_to_vec(vec![5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1]);
}

#[test]
fn test_empty_pop() {
    let mut heap = BinaryHeap::<i32>::new();
    assert!(heap.pop().is_none());
}

#[test]
fn test_empty_peek() {
    let empty = BinaryHeap::<i32>::new();
    assert!(empty.peek().is_none());
}

#[test]
fn test_empty_peek_mut() {
    let mut empty = BinaryHeap::<i32>::new();
    assert!(empty.peek_mut().is_none());
}

#[test]
fn test_from_iter() {
    let xs = vec![9, 8, 7, 6, 5, 4, 3, 2, 1];

    let mut q: BinaryHeap<_> = xs.iter().rev().cloned().collect();

    for &x in &xs {
        assert_eq!(q.pop().unwrap(), x);
    }
}

#[test]
fn test_drain() {
    let mut q: BinaryHeap<_> = [9, 8, 7, 6, 5, 4, 3, 2, 1].iter().cloned().collect();

    assert_eq!(q.drain().take(5).count(), 5);

    assert!(q.is_empty());
}

#[test]
fn test_extend_ref() {
    let mut a = BinaryHeap::new();
    a.push(1);
    a.push(2);

    a.extend(&[3, 4, 5]);

    assert_eq!(a.len(), 5);
    assert_eq!(a.into_sorted_vec(), [1, 2, 3, 4, 5]);

    let mut a = BinaryHeap::new();
    a.push(1);
    a.push(2);
    let mut b = BinaryHeap::new();
    b.push(3);
    b.push(4);
    b.push(5);

    a.extend(&b);

    assert_eq!(a.len(), 5);
    assert_eq!(a.into_sorted_vec(), [1, 2, 3, 4, 5]);
}

#[test]
fn test_append() {
    let mut a = BinaryHeap::from(vec![-10, 1, 2, 3, 3]);
    let mut b = BinaryHeap::from(vec![-20, 5, 43]);

    a.append(&mut b);

    assert_eq!(a.into_sorted_vec(), [-20, -10, 1, 2, 3, 3, 5, 43]);
    assert!(b.is_empty());
}

#[test]
fn test_append_to_empty() {
    let mut a = BinaryHeap::new();
    let mut b = BinaryHeap::from(vec![-20, 5, 43]);

    a.append(&mut b);

    assert_eq!(a.into_sorted_vec(), [-20, 5, 43]);
    assert!(b.is_empty());
}

#[test]
fn test_extend_specialization() {
    let mut a = BinaryHeap::from(vec![-10, 1, 2, 3, 3]);
    let b = BinaryHeap::from(vec![-20, 5, 43]);

    a.extend(b);

    assert_eq!(a.into_sorted_vec(), [-20, -10, 1, 2, 3, 3, 5, 43]);
}

#[allow(dead_code)]
fn assert_covariance() {
    fn drain<'new>(d: Drain<'static, &'static str>) -> Drain<'new, &'new str> {
        d
    }
}

// old binaryheap failed this test
//
// Integrity means that all elements are present after a comparison panics,
// even if the order may not be correct.
//
// Destructors must be called exactly once per element.
#[test]
#[cfg(not(miri))] // Miri does not support catching panics
fn panic_safe() {
    static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);

    #[derive(Eq, PartialEq, Ord, Clone, Debug)]
    struct PanicOrd<T>(T, bool);

    impl<T> Drop for PanicOrd<T> {
        fn drop(&mut self) {
            // update global drop count
            DROP_COUNTER.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl<T: PartialOrd> PartialOrd for PanicOrd<T> {
        fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
            if self.1 || other.1 {
                panic!("Panicking comparison");
            }
            self.0.partial_cmp(&other.0)
        }
    }
    let mut rng = thread_rng();
    const DATASZ: usize = 32;
    const NTEST: usize = 10;

    // don't use 0 in the data -- we want to catch the zeroed-out case.
    let data = (1..=DATASZ).collect::<Vec<_>>();

    // since it's a fuzzy test, run several tries.
    for _ in 0..NTEST {
        for i in 1..=DATASZ {
            DROP_COUNTER.store(0, Ordering::SeqCst);

            let mut panic_ords: Vec<_> = data.iter()
                                             .filter(|&&x| x != i)
                                             .map(|&x| PanicOrd(x, false))
                                             .collect();
            let panic_item = PanicOrd(i, true);

            // heapify the sane items
            panic_ords.shuffle(&mut rng);
            let mut heap = BinaryHeap::from(panic_ords);
            let inner_data;

            {
                // push the panicking item to the heap and catch the panic
                let thread_result = {
                    let mut heap_ref = AssertUnwindSafe(&mut heap);
                    panic::catch_unwind(move || {
                        heap_ref.push(panic_item);
                    })
                };
                assert!(thread_result.is_err());

                // Assert no elements were dropped
                let drops = DROP_COUNTER.load(Ordering::SeqCst);
                assert!(drops == 0, "Must not drop items. drops={}", drops);
                inner_data = heap.clone().into_vec();
                drop(heap);
            }
            let drops = DROP_COUNTER.load(Ordering::SeqCst);
            assert_eq!(drops, DATASZ);

            let mut data_sorted = inner_data.into_iter().map(|p| p.0).collect::<Vec<_>>();
            data_sorted.sort();
            assert_eq!(data_sorted, data);
        }
    }
}
