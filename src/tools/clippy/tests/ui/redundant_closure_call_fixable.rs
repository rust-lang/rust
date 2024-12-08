#![feature(async_closure)]
#![warn(clippy::redundant_closure_call)]
#![allow(clippy::redundant_async_block)]
#![allow(clippy::type_complexity)]
#![allow(unused)]

async fn something() -> u32 {
    21
}

async fn something_else() -> u32 {
    2
}

fn main() {
    let a = (|| 42)();
    let b = (async || {
        let x = something().await;
        let y = something_else().await;
        x * y
    })();
    let c = (|| {
        let x = 21;
        let y = 2;
        x * y
    })();
    let d = (async || something().await)();

    macro_rules! m {
        () => {
            (|| 0)()
        };
    }
    macro_rules! m2 {
        () => {
            (|| m!())()
        };
    }
    m2!();
    issue9956();
}

fn issue9956() {
    assert_eq!((|| || 43)()(), 42);

    // ... and some more interesting cases I've found while implementing the fix

    // not actually immediately calling the closure:
    let a = (|| 42);
    dbg!(a());

    // immediately calling it inside of a macro
    dbg!((|| 42)());

    // immediately calling only one closure, so we can't remove the other ones
    let a = (|| || || 123)();
    dbg!(a()());

    // nested async closures
    let a = (|| || || || async || 1)()()()()();
    let h = async { a.await };

    // macro expansion tests
    macro_rules! echo {
        ($e:expr) => {
            $e
        };
    }
    let a = (|| echo!(|| echo!(|| 1)))()()();
    assert_eq!(a, 1);
    let a = (|| echo!((|| 123)))()();
    assert_eq!(a, 123);

    // chaining calls, but not closures
    fn x() -> fn() -> fn() -> fn() -> i32 {
        || || || 42
    }
    let _ = x()()()();

    fn bar() -> fn(i32, i32) {
        foo
    }
    fn foo(_: i32, _: i32) {}
    bar()((|| || 42)()(), 5);
    foo((|| || 42)()(), 5);
}

async fn issue11357() {
    (|| async {})().await;
}

mod issue11707 {
    use core::future::Future;

    fn spawn_on(fut: impl Future<Output = ()>) {}

    fn demo() {
        spawn_on((|| async move {})());
    }
}

fn avoid_double_parens() {
    std::convert::identity((|| 13_i32 + 36_i32)()).leading_zeros();
}

fn fp_11274() {
    macro_rules! m {
        ($closure:expr) => {
            $closure(1)
        };
    }
    m!(|x| println!("{x}"));
}

// Issue #12358: When a macro expands into a closure, immediately calling the expanded closure
// triggers the lint.
fn issue_12358() {
    macro_rules! make_closure {
        () => {
            (|| || {})
        };
        (x) => {
            make_closure!()()
        };
    }

    // The lint would suggest to alter the line below to `make_closure!(x)`, which is semantically
    // different.
    make_closure!(x)();
}
