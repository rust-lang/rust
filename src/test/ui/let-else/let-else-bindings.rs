// run-pass
// adapted from src/test/ui/binding/if-let.rs
#![feature(let_else)]
#![allow(dead_code)]

fn none() -> bool {
    let None = Some("test") else {
        return true;
    };
    false
}

fn ok() -> bool {
    let Ok(()) = Err::<(),&'static str>("test") else {
        return true;
    };
    false
}

pub fn main() {
    let x = Some(3);
    let Some(y) = x else {
        panic!("let-else panicked");
    };
    assert_eq!(y, 3);
    let Some(_) = x else {
        panic!("bad match");
    };
    assert!(none());
    assert!(ok());

    assert!((|| {
        let 1 = 2 else {
            return true;
        };
        false
    })());

    enum Foo {
        One,
        Two(usize),
        Three(String, isize),
    }

    let foo = Foo::Three("three".to_string(), 42);
    let one = || {
        let Foo::One = foo else {
            return true;
        };
        false
    };
    assert!(one());
    let two = || {
        let Foo::Two(_x) = foo else {
            return true;
        };
        false
    };
    assert!(two());
    let three = || {
        let Foo::Three(s, _x) = foo else {
            return false;
        };
        s == "three"
    };
    assert!(three());

    let a@Foo::Two(_) = Foo::Two(42_usize) else {
        panic!("bad match")
    };
    let Foo::Two(b) = a else {
        panic!("panic in nested `if let`");
    };
    assert_eq!(b, 42_usize);
}
