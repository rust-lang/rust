// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Add;

#[test]
fn misc() {
    assert!(!false);
    assert!(-3 == 3 * (-1));
    assert!(!r"\b\w{13}\b".is_empty());

    {
        let x = &3;
        let y = &1;
        assert!(*x << *y == 6);
    }
    {
        use std::cell::RefCell;
        let x = RefCell::new(0);
        assert!(*(&0 as &PartialEq<i32>) == *x.borrow());
    }
    {
        fn x() {}
        let y: fn() = x;
        assert!(y == x);
    }
    {
        #[derive(PartialEq)]
        struct Foo;
        impl<'a> Add<&'a mut Foo> for Foo {
            type Output = Self;
            fn add(self, _: &mut Foo) -> Self {
                Foo
            }
        }
        assert!(Foo == Foo + &mut Foo);
    }
}

#[cfg(not(stage0))]
#[test]
#[should_panic(expected=
"1 == 1 && (y) + (NonDebug(1)) == (NonDebug(0)) && (unevaluated) == 1 && (unevaluated)")]
fn capture() {
    #[derive(Ord, Eq, PartialOrd, PartialEq)]
    struct NonDebug(i32);
    impl Add for NonDebug {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            NonDebug(self.0 + rhs.0)
        }
    }

    let x = 1;
    let y = NonDebug(0);
    assert!(x == 1 && y + NonDebug(1) == NonDebug(0) && x == 1 && { 1 == 0 });
}

#[cfg(not(stage0))]
#[test]
#[should_panic(expected="(Foo) != (Foo) && (unevaluated) == (unevaluated)")]
fn debug_unevaluated() {
    #[derive(PartialEq)]
    struct Foo;
    assert!(Foo != Foo && Foo == Foo);
}

#[cfg(not(stage0))]
#[test]
#[should_panic(expected=r#"assertion failed: "☃\n" == "☀\n""#)]
fn escape_expr() {
    assert!("☃\n" == "☀\n");
}

#[cfg(not(stage0))]
#[test]
#[should_panic(expected=r#"with expansion: "☃\n" == "☀\n""#)]
fn escape_expn() {
    assert!("☃\n" == "☀\n");
}

#[test]
fn evaluation_order() {
    let mut it = vec![1, 2, 4, 8, 16].into_iter();
    assert!(
        it.next().unwrap() != 1
            || it.next().unwrap() == 2
                && (it.next().unwrap() + it.next().unwrap() == 12 && it.next().unwrap() == 16)
    );

    let mut it = vec![2, 3, 16].into_iter();
    assert!(it.next().unwrap() << it.next().unwrap() == it.next().unwrap());

    let mut b = false;
    assert!(
        true || {
            b = true;
            false
        }
    );
    assert!(!b);
    assert!(
        !(false && {
            b = true;
            false
        })
    );
    assert!(!b);

    let mut n = 0;
    assert!(
        true && (false || {
            n += 1;
            true
        } || {
            n += 2;
            true
        }) && (({
            n += 4;
            false
        } && {
            n += 8;
            true
        }) != ({
            n += 16;
            false
        } || {
            n += 32;
            true
        }))
    );
    assert!(n == 1 + 4 + 16 + 32);
}
