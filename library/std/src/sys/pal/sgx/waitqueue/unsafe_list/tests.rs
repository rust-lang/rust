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
