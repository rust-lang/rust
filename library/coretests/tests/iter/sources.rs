use core::iter::*;

use super::*;

#[test]
fn test_repeat() {
    let mut it = repeat(42);
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), Some(42));
    assert_eq!(repeat(42).size_hint(), (usize::MAX, None));
}

#[test]
fn test_repeat_take() {
    let mut it = repeat(42).take(3);
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), None);
    is_trusted_len(repeat(42).take(3));
    assert_eq!(repeat(42).take(3).size_hint(), (3, Some(3)));
    assert_eq!(repeat(42).take(0).size_hint(), (0, Some(0)));
    assert_eq!(repeat(42).take(usize::MAX).size_hint(), (usize::MAX, Some(usize::MAX)));
}

#[test]
fn test_repeat_take_collect() {
    let v: Vec<_> = repeat(42).take(3).collect();
    assert_eq!(v, vec![42, 42, 42]);
}

#[test]
#[should_panic = "iterator is infinite"]
fn test_repeat_count() {
    repeat(42).count();
}

#[test]
fn test_repeat_last() {
    assert_eq!(repeat(42).last(), Some(42));
}

#[test]
fn test_repeat_with() {
    #[derive(PartialEq, Debug)]
    struct NotClone(usize);
    let mut it = repeat_with(|| NotClone(42));
    assert_eq!(it.next(), Some(NotClone(42)));
    assert_eq!(it.next(), Some(NotClone(42)));
    assert_eq!(it.next(), Some(NotClone(42)));
    assert_eq!(repeat_with(|| NotClone(42)).size_hint(), (usize::MAX, None));
}

#[test]
fn test_repeat_with_take() {
    let mut it = repeat_with(|| 42).take(3);
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), None);
    is_trusted_len(repeat_with(|| 42).take(3));
    assert_eq!(repeat_with(|| 42).take(3).size_hint(), (3, Some(3)));
    assert_eq!(repeat_with(|| 42).take(0).size_hint(), (0, Some(0)));
    assert_eq!(repeat_with(|| 42).take(usize::MAX).size_hint(), (usize::MAX, Some(usize::MAX)));
}

#[test]
fn test_repeat_with_take_collect() {
    let mut curr = 1;
    let v: Vec<_> = repeat_with(|| {
        let tmp = curr;
        curr *= 2;
        tmp
    })
    .take(5)
    .collect();
    assert_eq!(v, vec![1, 2, 4, 8, 16]);
}

#[test]
fn test_successors() {
    let mut powers_of_10 = successors(Some(1_u16), |n| n.checked_mul(10));
    assert_eq!(powers_of_10.by_ref().collect::<Vec<_>>(), &[1, 10, 100, 1_000, 10_000]);
    assert_eq!(powers_of_10.next(), None);

    let mut empty = successors(None::<u32>, |_| unimplemented!());
    assert_eq!(empty.next(), None);
    assert_eq!(empty.next(), None);
}

#[test]
fn test_once() {
    let mut it = once(42);
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), None);
}

#[test]
fn test_once_with() {
    let count = Cell::new(0);
    let mut it = once_with(|| {
        count.set(count.get() + 1);
        42
    });

    assert_eq!(count.get(), 0);
    assert_eq!(it.next(), Some(42));
    assert_eq!(count.get(), 1);
    assert_eq!(it.next(), None);
    assert_eq!(count.get(), 1);
    assert_eq!(it.next(), None);
    assert_eq!(count.get(), 1);
}

#[test]
fn test_empty() {
    let mut it = empty::<i32>();
    assert_eq!(it.next(), None);
}

#[test]
fn test_repeat_n_drop() {
    #[derive(Clone, Debug)]
    struct DropCounter<'a>(&'a Cell<usize>);
    impl Drop for DropCounter<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    // `repeat_n(x, 0)` drops `x` immediately
    let count = Cell::new(0);
    let item = DropCounter(&count);
    let mut it = repeat_n(item, 0);
    assert_eq!(count.get(), 1);
    assert!(it.next().is_none());
    assert_eq!(count.get(), 1);
    drop(it);
    assert_eq!(count.get(), 1);

    // Dropping the iterator needs to drop the item if it's non-empty
    let count = Cell::new(0);
    let item = DropCounter(&count);
    let it = repeat_n(item, 3);
    assert_eq!(count.get(), 0);
    drop(it);
    assert_eq!(count.get(), 1);

    // Dropping the iterator doesn't drop the item if it was exhausted
    let count = Cell::new(0);
    let item = DropCounter(&count);
    let mut it = repeat_n(item, 3);
    assert_eq!(count.get(), 0);
    let x0 = it.next().unwrap();
    assert_eq!(count.get(), 0);
    let x1 = it.next().unwrap();
    assert_eq!(count.get(), 0);
    let x2 = it.next().unwrap();
    assert_eq!(count.get(), 0);
    assert!(it.next().is_none());
    assert_eq!(count.get(), 0);
    assert!(it.next().is_none());
    assert_eq!(count.get(), 0);
    drop(it);
    assert_eq!(count.get(), 0);
    drop((x0, x1, x2));
    assert_eq!(count.get(), 3);
}

#[test]
fn test_repeat_n_soundness() {
    let x = std::iter::repeat_n(String::from("use after free"), 0);
    println!("{x:?}");

    pub struct PanicOnClone;

    impl Clone for PanicOnClone {
        fn clone(&self) -> Self {
            unreachable!()
        }
    }

    // `repeat_n` should drop the element immediately if `count` is zero.
    // `Clone` should then not try to clone the element.
    let x = std::iter::repeat_n(PanicOnClone, 0);
    let _ = x.clone();

    let mut y = std::iter::repeat_n(Box::new(0), 1);
    let x = y.next().unwrap();
    let _z = y;
    assert_eq!(0, *x);
}
