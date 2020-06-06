#![allow(unused)]
#![warn(clippy::let_and_return)]

fn test() -> i32 {
    let _y = 0; // no warning
    let x = 5;
    x
}

fn test_inner() -> i32 {
    if true {
        let x = 5;
        x
    } else {
        0
    }
}

fn test_nowarn_1() -> i32 {
    let mut x = 5;
    x += 1;
    x
}

fn test_nowarn_2() -> i32 {
    let x = 5;
    x + 1
}

fn test_nowarn_3() -> (i32, i32) {
    // this should technically warn, but we do not compare complex patterns
    let (x, y) = (5, 9);
    (x, y)
}

fn test_nowarn_4() -> i32 {
    // this should technically warn, but not b/c of clippy::let_and_return, but b/c of useless type
    let x: i32 = 5;
    x
}

fn test_nowarn_5(x: i16) -> u16 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let x = x as u16;
    x
}

// False positive example
trait Decode {
    fn decode<D: std::io::Read>(d: D) -> Result<Self, ()>
    where
        Self: Sized;
}

macro_rules! tuple_encode {
    ($($x:ident),*) => (
        impl<$($x: Decode),*> Decode for ($($x),*) {
            #[inline]
            #[allow(non_snake_case)]
            fn decode<D: std::io::Read>(mut d: D) -> Result<Self, ()> {
                // Shouldn't trigger lint
                Ok(($({let $x = Decode::decode(&mut d)?; $x }),*))
            }
        }
    );
}

tuple_encode!(T0, T1, T2, T3, T4, T5, T6, T7);

mod no_lint_if_stmt_borrows {
    mod issue_3792 {
        use std::io::{self, BufRead, Stdin};

        fn read_line() -> String {
            let stdin = io::stdin();
            let line = stdin.lock().lines().next().unwrap().unwrap();
            line
        }
    }

    mod issue_3324 {
        use std::cell::RefCell;
        use std::rc::{Rc, Weak};

        fn test(value: Weak<RefCell<Bar>>) -> u32 {
            let value = value.upgrade().unwrap();
            let ret = value.borrow().baz();
            ret
        }

        struct Bar {}

        impl Bar {
            fn new() -> Self {
                Bar {}
            }
            fn baz(&self) -> u32 {
                0
            }
        }

        fn main() {
            let a = Rc::new(RefCell::new(Bar::new()));
            let b = Rc::downgrade(&a);
            test(b);
        }
    }

    mod free_function {
        struct Inner;

        struct Foo<'a> {
            inner: &'a Inner,
        }

        impl Drop for Foo<'_> {
            fn drop(&mut self) {}
        }

        impl Foo<'_> {
            fn value(&self) -> i32 {
                42
            }
        }

        fn some_foo(inner: &Inner) -> Foo<'_> {
            Foo { inner }
        }

        fn test() -> i32 {
            let x = Inner {};
            let value = some_foo(&x).value();
            value
        }
    }
}

fn main() {}
