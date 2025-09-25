#![warn(clippy::if_then_some_else_none)]
#![allow(clippy::redundant_pattern_matching, clippy::unnecessary_lazy_evaluations)]

fn main() {
    // Should issue an error.
    let _ = if foo() {
        //~^ if_then_some_else_none

        println!("true!");
        Some("foo")
    } else {
        None
    };

    // Should issue an error when macros are used.
    let _ = if matches!(true, true) {
        //~^ if_then_some_else_none

        println!("true!");
        Some(matches!(true, false))
    } else {
        None
    };

    // Should issue an error. Binary expression `o < 32` should be parenthesized.
    let x = Some(5);
    let _ = x.and_then(|o| if o < 32 { Some(o) } else { None });
    //~^ if_then_some_else_none

    // Should issue an error. Unary expression `!x` should be parenthesized.
    let x = true;
    let _ = if !x { Some(0) } else { None };
    //~^ if_then_some_else_none

    // Should not issue an error since the `else` block has a statement besides `None`.
    let _ = if foo() {
        println!("true!");
        Some("foo")
    } else {
        eprintln!("false...");
        None
    };

    // Should not issue an error since there are more than 2 blocks in the if-else chain.
    let _ = if foo() {
        println!("foo true!");
        Some("foo")
    } else if bar() {
        println!("bar true!");
        Some("bar")
    } else {
        None
    };

    let _ = if foo() {
        println!("foo true!");
        Some("foo")
    } else {
        bar().then(|| {
            println!("bar true!");
            "bar"
        })
    };

    // Should not issue an error since the `then` block has `None`, not `Some`.
    let _ = if foo() { None } else { Some("foo is false") };

    // Should not issue an error since the `else` block doesn't use `None` directly.
    let _ = if foo() { Some("foo is true") } else { into_none() };

    // Should not issue an error since the `then` block doesn't use `Some` directly.
    let _ = if foo() { into_some("foo") } else { None };
}

#[clippy::msrv = "1.49"]
fn _msrv_1_49() {
    // `bool::then` was stabilized in 1.50. Do not lint this
    let _ = if foo() {
        println!("true!");
        Some(149)
    } else {
        None
    };
}

#[clippy::msrv = "1.50"]
fn _msrv_1_50() {
    let _ = if foo() {
        //~^ if_then_some_else_none

        println!("true!");
        Some(150)
    } else {
        None
    };
}

fn foo() -> bool {
    unimplemented!()
}

fn bar() -> bool {
    unimplemented!()
}

fn into_some<T>(v: T) -> Option<T> {
    Some(v)
}

fn into_none<T>() -> Option<T> {
    None
}

// Should not warn
fn f(b: bool, v: Option<()>) -> Option<()> {
    if b {
        v?; // This is a potential early return, is not equivalent with `bool::then`

        Some(())
    } else {
        None
    }
}

fn issue11394(b: bool, v: Result<(), ()>) -> Result<(), ()> {
    let x = if b {
        #[allow(clippy::let_unit_value)]
        let _ = v?;
        Some(())
    } else {
        None
    };

    Ok(())
}

fn issue13407(s: &str) -> Option<bool> {
    if s == "1" { Some(true) } else { None }
    //~^ if_then_some_else_none
}

const fn issue12103(x: u32) -> Option<u32> {
    // Should not issue an error in `const` context
    if x > 42 { Some(150) } else { None }
}

mod issue15257 {
    struct Range {
        start: u8,
        end: u8,
    }

    fn can_be_safely_rewrite(rs: &[&Range]) -> Option<Vec<u8>> {
        if rs.len() == 1 && rs[0].start == rs[0].end {
            //~^ if_then_some_else_none
            Some(vec![rs[0].start])
        } else {
            None
        }
    }

    fn reborrow_as_ptr(i: *mut i32) -> Option<*const i32> {
        let modulo = unsafe { *i % 2 };
        if modulo == 0 {
            //~^ if_then_some_else_none
            Some(i)
        } else {
            None
        }
    }

    fn reborrow_as_fn_ptr(i: i32) {
        fn do_something(fn_: Option<fn(i32)>) {
            todo!()
        }

        fn item_fn(i: i32) {
            todo!()
        }

        do_something(if i % 2 == 0 {
            //~^ if_then_some_else_none
            Some(item_fn)
        } else {
            None
        });
    }

    fn reborrow_as_fn_unsafe(i: i32) {
        fn do_something(fn_: Option<unsafe fn(i32)>) {
            todo!()
        }

        fn item_fn(i: i32) {
            todo!()
        }

        do_something(if i % 2 == 0 {
            //~^ if_then_some_else_none
            Some(item_fn)
        } else {
            None
        });

        let closure_fn = |i: i32| {};
        do_something(if i % 2 == 0 {
            //~^ if_then_some_else_none
            Some(closure_fn)
        } else {
            None
        });
    }
}

fn issue15005() {
    struct Counter {
        count: u32,
    }

    impl Counter {
        fn new() -> Counter {
            Counter { count: 0 }
        }
    }

    impl Iterator for Counter {
        type Item = u32;

        fn next(&mut self) -> Option<Self::Item> {
            //~v if_then_some_else_none
            if self.count < 5 {
                self.count += 1;
                Some(self.count)
            } else {
                None
            }
        }
    }
}

fn statements_from_macro() {
    macro_rules! mac {
        () => {
            println!("foo");
            println!("bar");
        };
    }
    //~v if_then_some_else_none
    let _ = if true {
        mac!();
        Some(42)
    } else {
        None
    };
}

fn dont_lint_inside_macros() {
    macro_rules! mac {
        ($cond:expr, $res:expr) => {
            if $cond { Some($res) } else { None }
        };
    }
    let _: Option<u32> = mac!(true, 42);
}
