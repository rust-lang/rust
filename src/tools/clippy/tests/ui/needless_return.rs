//@aux-build:proc_macros.rs
#![feature(yeet_expr)]
#![allow(unused)]
#![allow(
    clippy::if_same_then_else,
    clippy::single_match,
    clippy::needless_bool,
    clippy::equatable_if_let,
    clippy::needless_else,
    clippy::missing_safety_doc
)]
#![warn(clippy::needless_return)]

extern crate proc_macros;
use proc_macros::with_span;

use std::cell::RefCell;

macro_rules! the_answer {
    () => {
        42
    };
}

fn test_end_of_fn() -> bool {
    if true {
        // no error!
        return true;
    }
    return true;
    //~^ needless_return
}

fn test_no_semicolon() -> bool {
    return true;
    //~^ needless_return
}

#[rustfmt::skip]
fn test_multiple_semicolon() -> bool {
    return true;;;
    //~^ needless_return
}

#[rustfmt::skip]
fn test_multiple_semicolon_with_spaces() -> bool {
    return true;; ; ;
    //~^ needless_return
}

fn test_if_block() -> bool {
    if true {
        return true;
        //~^ needless_return
    } else {
        return false;
        //~^ needless_return
    }
}

fn test_match(x: bool) -> bool {
    match x {
        true => return false,
        //~^ needless_return
        false => {
            return true;
            //~^ needless_return
        },
    }
}

fn test_closure() {
    let _ = || {
        return true;
        //~^ needless_return
    };
    let _ = || return true;
    //~^ needless_return
}

fn test_macro_call() -> i32 {
    return the_answer!();
    //~^ needless_return
}

fn test_void_fun() {
    return;
    //~^ needless_return
}

fn test_void_if_fun(b: bool) {
    if b {
        return;
        //~^ needless_return
    } else {
        return;
        //~^ needless_return
    }
}

fn test_void_match(x: u32) {
    match x {
        0 => (),
        _ => return,
        //~^ needless_return
    }
}

fn test_nested_match(x: u32) {
    match x {
        0 => (),
        1 => {
            let _ = 42;
            return;
            //~^ needless_return
        },
        _ => return,
        //~^ needless_return
    }
}

fn temporary_outlives_local() -> String {
    let x = RefCell::<String>::default();
    return x.borrow().clone();
}

fn borrows_but_not_last(value: bool) -> String {
    if value {
        let x = RefCell::<String>::default();
        let _a = x.borrow().clone();
        return String::from("test");
        //~^ needless_return
    } else {
        return String::new();
        //~^ needless_return
    }
}

macro_rules! needed_return {
    ($e:expr) => {
        if $e > 3 {
            return;
        }
    };
}

fn test_return_in_macro() {
    // This will return and the macro below won't be executed. Removing the `return` from the macro
    // will change semantics.
    needed_return!(10);
    needed_return!(0);
}

mod issue6501 {
    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn foo(bar: Result<(), ()>) {
        bar.unwrap_or_else(|_| return)
        //~^ needless_return
    }

    fn test_closure() {
        let _ = || {
            return;
            //~^ needless_return
        };
        let _ = || return;
        //~^ needless_return
    }

    struct Foo;
    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn bar(res: Result<Foo, u8>) -> Foo {
        res.unwrap_or_else(|_| return Foo)
        //~^ needless_return
    }
}

async fn async_test_end_of_fn() -> bool {
    if true {
        // no error!
        return true;
    }
    return true;
    //~^ needless_return
}

async fn async_test_no_semicolon() -> bool {
    return true;
    //~^ needless_return
}

async fn async_test_if_block() -> bool {
    if true {
        return true;
        //~^ needless_return
    } else {
        return false;
        //~^ needless_return
    }
}

async fn async_test_match(x: bool) -> bool {
    match x {
        true => return false,
        //~^ needless_return
        false => {
            return true;
            //~^ needless_return
        },
    }
}

async fn async_test_closure() {
    let _ = || {
        return true;
        //~^ needless_return
    };
    let _ = || return true;
    //~^ needless_return
}

async fn async_test_macro_call() -> i32 {
    return the_answer!();
    //~^ needless_return
}

async fn async_test_void_fun() {
    return;
    //~^ needless_return
}

async fn async_test_void_if_fun(b: bool) {
    if b {
        return;
        //~^ needless_return
    } else {
        return;
        //~^ needless_return
    }
}

async fn async_test_void_match(x: u32) {
    match x {
        0 => (),
        _ => return,
        //~^ needless_return
    }
}

async fn async_temporary_outlives_local() -> String {
    let x = RefCell::<String>::default();
    return x.borrow().clone();
}

async fn async_borrows_but_not_last(value: bool) -> String {
    if value {
        let x = RefCell::<String>::default();
        let _a = x.borrow().clone();
        return String::from("test");
        //~^ needless_return
    } else {
        return String::new();
        //~^ needless_return
    }
}

async fn async_test_return_in_macro() {
    needed_return!(10);
    needed_return!(0);
}

fn let_else() {
    let Some(1) = Some(1) else { return };
}

fn needless_return_macro() -> String {
    let _ = "foo";
    let _ = "bar";
    return format!("Hello {}", "world!");
    //~^ needless_return
}

fn issue_9361(n: i32) -> i32 {
    #[expect(clippy::arithmetic_side_effects)]
    return n + n;
}

mod issue_12998 {
    fn expect_lint() -> i32 {
        let x = 1;

        #[expect(clippy::needless_return)]
        return x;
    }

    fn expect_group() -> i32 {
        let x = 1;

        #[expect(clippy::style)]
        return x;
    }

    fn expect_all() -> i32 {
        let x = 1;

        #[expect(clippy::all)]
        return x;
    }

    fn expect_warnings() -> i32 {
        let x = 1;

        #[expect(warnings)]
        return x;
    }
}

fn issue8336(x: i32) -> bool {
    if x > 0 {
        println!("something");
        return true;
        //~^ needless_return
    } else {
        return false;
        //~^ needless_return
    };
}

fn issue8156(x: u8) -> u64 {
    match x {
        80 => {
            return 10;
            //~^ needless_return
        },
        _ => {
            return 100;
            //~^ needless_return
        },
    };
}

// Ideally the compiler should throw `unused_braces` in this case
fn issue9192() -> i32 {
    {
        return 0;
        //~^ needless_return
    };
}

fn issue9503(x: usize) -> isize {
    unsafe {
        if x > 12 {
            return *(x as *const isize);
            //~^ needless_return
        } else {
            return !*(x as *const isize);
            //~^ needless_return
        };
    };
}

mod issue9416 {
    pub fn with_newline() {
        let _ = 42;
        return;
        //~^ needless_return
    }

    #[rustfmt::skip]
    pub fn oneline() {
        let _ = 42; return;
        //~^ needless_return
    }
}

fn issue9947() -> Result<(), String> {
    do yeet "hello";
}

// without anyhow, but triggers the same bug I believe
#[expect(clippy::useless_format)]
fn issue10051() -> Result<String, String> {
    if true {
        return Ok(format!("ok!"));
        //~^ needless_return
    } else {
        return Err(format!("err!"));
        //~^ needless_return
    }
}

mod issue10049 {
    fn single() -> u32 {
        return if true { 1 } else { 2 };
        //~^ needless_return
    }

    fn multiple(b1: bool, b2: bool, b3: bool) -> u32 {
        return if b1 { 0 } else { 1 } | if b2 { 2 } else { 3 } | if b3 { 4 } else { 5 };
        //~^ needless_return
    }
}

fn test_match_as_stmt() {
    let x = 9;
    match x {
        1 => 2,
        2 => return,
        _ => 0,
    };
}

fn allow_works() -> i32 {
    #[allow(clippy::needless_return, clippy::match_single_binding)]
    match () {
        () => return 42,
    }
}

fn conjunctive_blocks() -> String {
    return { "a".to_string() } + "b" + { "c" };
    //~^ needless_return
}

fn issue12907() -> String {
    return "".split("").next().unwrap().to_string();
    //~^ needless_return
}

fn issue13458() {
    with_span!(span return);
}

fn main() {}

fn a(x: Option<u8>) -> Option<u8> {
    match x {
        Some(_) => None,
        None => {
            #[expect(clippy::needless_return, reason = "Use early return for errors.")]
            return None;
        },
    }
}

fn b(x: Option<u8>) -> Option<u8> {
    match x {
        Some(_) => None,
        None => {
            #[expect(clippy::needless_return)]
            return None;
        },
    }
}

unsafe fn todo() -> *const u8 {
    todo!()
}

pub unsafe fn issue_12157() -> *const i32 {
    return unsafe { todo() } as *const i32;
    //~^ needless_return
}

mod else_ifs {
    fn test1(a: i32) -> u32 {
        if a == 0 {
            return 1;
        //~^ needless_return
        } else if a < 10 {
            return 2;
        //~^ needless_return
        } else {
            return 3;
            //~^ needless_return
        }
    }

    fn test2(a: i32) -> u32 {
        if a == 0 {
            return 1;
        //~^ needless_return
        } else if a < 10 {
            2
        } else {
            return 3;
            //~^ needless_return
        }
    }

    fn test3(a: i32) -> u32 {
        if a == 0 {
            return 1;
        //~^ needless_return
        } else if a < 10 {
            2
        } else {
            return 3;
            //~^ needless_return
        }
    }

    #[allow(clippy::match_single_binding, clippy::redundant_pattern)]
    fn test4(a: i32) -> u32 {
        if a == 0 {
            return 1;
            //~^ needless_return
        } else if if if a > 0x1_1 {
            return 2;
        } else {
            return 5;
        } {
            true
        } else {
            true
        } {
            0xDEADC0DE
        } else if match a {
            b @ _ => {
                return 1;
            },
        } {
            0xDEADBEEF
        } else {
            1
        }
    }
}
