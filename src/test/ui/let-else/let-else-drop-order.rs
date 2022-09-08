// run-pass
// edition:2021
// check-run-results
//
// Drop order tests for let else
//
// Mostly this ensures two things:
// 1. That let and let else temporary drop order is the same.
//    This is a specific design request: https://github.com/rust-lang/rust/pull/93628#issuecomment-1047140316
// 2. That the else block truly only runs after the
//    temporaries have dropped.
//
// We also print some nice tables for an overview by humans.
// Changes in those tables are considered breakages, but the
// important properties 1 and 2 are also enforced by the code.
// This is important as it's easy to update the stdout file
// with a --bless and miss the impact of that change.

#![feature(let_else)]
#![allow(irrefutable_let_patterns)]

use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
struct DropAccountant(Rc<RefCell<Vec<Vec<String>>>>);

impl DropAccountant {
    fn new() -> Self {
        Self(Default::default())
    }
    fn build_droppy(&self, v: u32) -> Droppy<u32> {
        Droppy(self.clone(), v)
    }
    fn build_droppy_enum_none(&self, _v: u32) -> ((), DroppyEnum<u32>) {
        ((), DroppyEnum::None(self.clone()))
    }
    fn new_list(&self, s: impl ToString) {
        self.0.borrow_mut().push(vec![s.to_string()]);
    }
    fn push(&self, s: impl ToString) {
        let s = s.to_string();
        let mut accounts = self.0.borrow_mut();
        accounts.last_mut().unwrap().push(s);
    }
    fn print_table(&self) {
        println!();

        let accounts = self.0.borrow();
        let before_last = &accounts[accounts.len() - 2];
        let last = &accounts[accounts.len() - 1];
        let before_last = get_comma_list(before_last);
        let last = get_comma_list(last);
        const LINES: &[&str] = &[
            "vanilla",
            "&",
            "&mut",
            "move",
            "fn(this)",
            "tuple",
            "array",
            "ref &",
            "ref mut &mut",
        ];
        let max_len = LINES.iter().map(|v| v.len()).max().unwrap();
        let max_len_before = before_last.iter().map(|v| v.len()).max().unwrap();
        let max_len_last = last.iter().map(|v| v.len()).max().unwrap();

        println!(
            "| {: <max_len$} | {: <max_len_before$} | {: <max_len_last$} |",
            "construct", before_last[0], last[0]
        );
        println!("| {:-<max_len$} | {:-<max_len_before$} | {:-<max_len_last$} |", "", "", "");

        for ((l, l_before), l_last) in
            LINES.iter().zip(before_last[1..].iter()).zip(last[1..].iter())
        {
            println!(
                "| {: <max_len$} | {: <max_len_before$} | {: <max_len_last$} |",
                l, l_before, l_last,
            );
        }
    }
    #[track_caller]
    fn assert_all_equal_to(&self, st: &str) {
        let accounts = self.0.borrow();
        let last = &accounts[accounts.len() - 1];
        let last = get_comma_list(last);
        for line in last[1..].iter() {
            assert_eq!(line.trim(), st.trim());
        }
    }
    #[track_caller]
    fn assert_equality_last_two_lists(&self) {
        let accounts = self.0.borrow();
        let last = &accounts[accounts.len() - 1];
        let before_last = &accounts[accounts.len() - 2];
        for (l, b) in last[1..].iter().zip(before_last[1..].iter()) {
            if !(l == b || l == "n/a" || b == "n/a") {
                panic!("not equal: '{last:?}' != '{before_last:?}'");
            }
        }
    }
}

fn get_comma_list(sl: &[String]) -> Vec<String> {
    std::iter::once(sl[0].clone())
        .chain(sl[1..].chunks(2).map(|c| c.join(",")))
        .collect::<Vec<String>>()
}

struct Droppy<T>(DropAccountant, T);

impl<T> Drop for Droppy<T> {
    fn drop(&mut self) {
        self.0.push("drop");
    }
}

#[allow(dead_code)]
enum DroppyEnum<T> {
    Some(DropAccountant, T),
    None(DropAccountant),
}

impl<T> Drop for DroppyEnum<T> {
    fn drop(&mut self) {
        match self {
            DroppyEnum::Some(acc, _inner) => acc,
            DroppyEnum::None(acc) => acc,
        }
        .push("drop");
    }
}

macro_rules! nestings_with {
    ($construct:ident, $binding:pat, $exp:expr) => {
        // vanilla:
        $construct!($binding, $exp.1);

        // &:
        $construct!(&$binding, &$exp.1);

        // &mut:
        $construct!(&mut $binding, &mut ($exp.1));

        {
            // move:
            let w = $exp;
            $construct!(
                $binding,
                {
                    let w = w;
                    w
                }
                .1
            );
        }

        // fn(this):
        $construct!($binding, std::convert::identity($exp).1);
    };
}

macro_rules! nestings {
    ($construct:ident, $binding:pat, $exp:expr) => {
        nestings_with!($construct, $binding, $exp);

        // tuple:
        $construct!(($binding, 77), ($exp.1, 77));

        // array:
        $construct!([$binding], [$exp.1]);
    };
}

macro_rules! let_else {
    ($acc:expr, $v:expr, $binding:pat, $build:ident) => {
        let acc = $acc;
        let v = $v;

        macro_rules! let_else_construct {
            ($arg:pat, $exp:expr) => {
                loop {
                    let $arg = $exp else {
                        acc.push("else");
                        break;
                    };
                    acc.push("body");
                    break;
                }
            };
        }
        nestings!(let_else_construct, $binding, acc.$build(v));
        // ref &:
        let_else_construct!($binding, &acc.$build(v).1);

        // ref mut &mut:
        let_else_construct!($binding, &mut acc.$build(v).1);
    };
}

macro_rules! let_ {
    ($acc:expr, $binding:tt) => {
        let acc = $acc;

        macro_rules! let_construct {
            ($arg:pat, $exp:expr) => {{
                let $arg = $exp;
                acc.push("body");
            }};
        }
        let v = 0;
        {
            nestings_with!(let_construct, $binding, acc.build_droppy(v));
        }
        acc.push("n/a");
        acc.push("n/a");
        acc.push("n/a");
        acc.push("n/a");

        // ref &:
        let_construct!($binding, &acc.build_droppy(v).1);

        // ref mut &mut:
        let_construct!($binding, &mut acc.build_droppy(v).1);
    };
}

fn main() {
    let acc = DropAccountant::new();

    println!(" --- matching cases ---");

    // Ensure that let and let else have the same behaviour
    acc.new_list("let _");
    let_!(&acc, _);
    acc.new_list("let else _");
    let_else!(&acc, 0, _, build_droppy);
    acc.assert_equality_last_two_lists();
    acc.print_table();

    // Ensure that let and let else have the same behaviour
    acc.new_list("let _v");
    let_!(&acc, _v);
    acc.new_list("let else _v");
    let_else!(&acc, 0, _v, build_droppy);
    acc.assert_equality_last_two_lists();
    acc.print_table();

    println!();

    println!(" --- mismatching cases ---");

    acc.new_list("let else _ mismatch");
    let_else!(&acc, 1, DroppyEnum::Some(_, _), build_droppy_enum_none);
    acc.new_list("let else _v mismatch");
    let_else!(&acc, 1, DroppyEnum::Some(_, _v), build_droppy_enum_none);
    acc.print_table();
    // This ensures that we always drop before visiting the else case
    acc.assert_all_equal_to("drop,else");

    acc.new_list("let else 0 mismatch");
    let_else!(&acc, 1, 0, build_droppy);
    acc.new_list("let else 0 mismatch");
    let_else!(&acc, 1, 0, build_droppy);
    acc.print_table();
    // This ensures that we always drop before visiting the else case
    acc.assert_all_equal_to("drop,else");
}
