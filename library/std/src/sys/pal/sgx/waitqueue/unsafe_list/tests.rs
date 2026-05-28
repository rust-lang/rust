use super::*;
use crate::cell::Cell;

/// # Safety
/// List must be valid.
unsafe fn assert_empty<T>(list: &mut UnsafeList<T>) {
    assert!(unsafe { list.pop() }.is_none(), "assertion failed: list is not empty");
}

#[test]
fn init_empty() {
    unsafe {
        assert_empty(&mut UnsafeList::<i32>::new());
    }
}

#[test]
fn push_pop() {
    unsafe {
        let mut node = UnsafeListEntry::new(1234);
        let mut list = UnsafeList::new();
        assert_eq!(list.push(&mut node), &1234);
        assert_eq!(list.pop().unwrap(), &1234);
        assert_empty(&mut list);
    }
}

#[test]
fn push_remove() {
    unsafe {
        let mut node = UnsafeListEntry::new(1234);
        let mut list = UnsafeList::new();
        assert_eq!(list.push(&mut node), &1234);
        list.remove(&mut node);
        assert_empty(&mut list);
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
        let mut list = UnsafeList::new();
        assert_eq!(list.push(&mut node1), &11);
        assert_eq!(list.push(&mut node2), &12);
        assert_eq!(list.push(&mut node3), &13);
        assert_eq!(list.push(&mut node4), &14);
        assert_eq!(list.push(&mut node5), &15);

        list.remove(&mut node1);
        assert_eq!(list.pop().unwrap(), &12);
        list.remove(&mut node3);
        assert_eq!(list.pop().unwrap(), &14);
        list.remove(&mut node5);
        assert_empty(&mut list);

        assert_eq!(list.push(&mut node1), &11);
        assert_eq!(list.pop().unwrap(), &11);
        assert_empty(&mut list);

        assert_eq!(list.push(&mut node3), &13);
        assert_eq!(list.push(&mut node4), &14);
        list.remove(&mut node3);
        list.remove(&mut node4);
        assert_empty(&mut list);
    }
}

#[test]
fn complex_pushes_pops() {
    unsafe {
        let mut node1 = UnsafeListEntry::new(1234);
        let mut node2 = UnsafeListEntry::new(4567);
        let mut node3 = UnsafeListEntry::new(9999);
        let mut node4 = UnsafeListEntry::new(8642);
        let mut list = UnsafeList::new();
        list.push(&mut node1);
        list.push(&mut node2);
        assert_eq!(list.pop().unwrap(), &1234);
        list.push(&mut node3);
        assert_eq!(list.pop().unwrap(), &4567);
        assert_eq!(list.pop().unwrap(), &9999);
        assert_empty(&mut list);
        list.push(&mut node4);
        assert_eq!(list.pop().unwrap(), &8642);
        assert_empty(&mut list);
    }
}

#[test]
fn cell() {
    unsafe {
        let mut node = UnsafeListEntry::new(Cell::new(0));
        let mut list = UnsafeList::new();
        let noderef = list.push(&mut node);
        assert_eq!(noderef.get(), 0);
        list.pop().unwrap().set(1);
        assert_empty(&mut list);
        assert_eq!(noderef.get(), 1);
    }
}
