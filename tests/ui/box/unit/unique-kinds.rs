//@ run-pass

use std::fmt::Debug;

fn sendable() {

    fn f<T:Send + PartialEq + Debug>(i: T, j: T) {
        assert_eq!(i, j);
    }

    fn g<T:Send + PartialEq>(i: T, j: T) {
        assert!(i != j);
    }

    let i: Box<_> = Box::new(100);
    let j: Box<_> = Box::new(100);
    f(i, j);
    let i: Box<_> = Box::new(100);
    let j: Box<_> = Box::new(101);
    g(i, j);
}

fn copyable() {

    fn f<T:PartialEq + Debug>(i: T, j: T) {
        assert_eq!(i, j);
    }

    fn g<T:PartialEq>(i: T, j: T) {
        assert!(i != j);
    }

    let i: Box<_> = Box::new(100);
    let j: Box<_> = Box::new(100);
    f(i, j);
    let i: Box<_> = Box::new(100);
    let j: Box<_> = Box::new(101);
    g(i, j);
}

fn noncopyable() {

    fn f<T:PartialEq + Debug>(i: T, j: T) {
        assert_eq!(i, j);
    }

    fn g<T:PartialEq>(i: T, j: T) {
        assert!(i != j);
    }

    let i: Box<_> = Box::new(100);
    let j: Box<_> = Box::new(100);
    f(i, j);
    let i: Box<_> = Box::new(100);
    let j: Box<_> = Box::new(101);
    g(i, j);
}

pub fn main() {
    sendable();
    copyable();
    noncopyable();
}
