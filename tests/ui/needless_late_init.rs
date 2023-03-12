// run-rustfix
// aux-build:proc_macros.rs
#![feature(let_chains)]
#![allow(unused)]
#![allow(
    clippy::assign_op_pattern,
    clippy::blocks_in_if_conditions,
    clippy::let_and_return,
    clippy::let_unit_value,
    clippy::nonminimal_bool,
    clippy::uninlined_format_args
)]

extern crate proc_macros;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;

struct SignificantDrop;
impl std::ops::Drop for SignificantDrop {
    fn drop(&mut self) {
        println!("dropped");
    }
}

fn simple() {
    let a;
    a = "zero";

    let b;
    let c;
    b = 1;
    c = 2;

    let d: usize;
    d = 1;

    let e;
    e = format!("{}", d);
}

fn main() {
    let a;
    let n = 1;
    match n {
        1 => a = "one",
        _ => {
            a = "two";
        },
    }

    let b;
    if n == 3 {
        b = "four";
    } else {
        b = "five"
    }

    let d;
    if true {
        let temp = 5;
        d = temp;
    } else {
        d = 15;
    }

    let e;
    if true {
        e = format!("{} {}", a, b);
    } else {
        e = format!("{}", n);
    }

    let f;
    match 1 {
        1 => f = "three",
        _ => return,
    }; // has semi

    let g: usize;
    if true {
        g = 5;
    } else {
        panic!();
    }

    // Drop order only matters if both are significant
    let x;
    let y = SignificantDrop;
    x = 1;

    let x;
    let y = 1;
    x = SignificantDrop;

    let x;
    // types that should be considered insignificant
    let y = 1;
    let y = "2";
    let y = String::new();
    let y = vec![3.0];
    let y = HashMap::<usize, usize>::new();
    let y = BTreeMap::<usize, usize>::new();
    let y = HashSet::<usize>::new();
    let y = BTreeSet::<usize>::new();
    let y = Box::new(4);
    x = SignificantDrop;
}

async fn in_async() -> &'static str {
    async fn f() -> &'static str {
        "one"
    }

    let a;
    let n = 1;
    match n {
        1 => a = f().await,
        _ => {
            a = "two";
        },
    }

    a
}

const fn in_const() -> &'static str {
    const fn f() -> &'static str {
        "one"
    }

    let a;
    let n = 1;
    match n {
        1 => a = f(),
        _ => {
            a = "two";
        },
    }

    a
}

#[proc_macros::inline_macros]
fn does_not_lint() {
    let z;
    if false {
        z = 1;
    }

    let x;
    let y;
    if true {
        x = 1;
    } else {
        y = 1;
    }

    let mut x;
    if true {
        x = 5;
        x = 10 / x;
    } else {
        x = 2;
    }

    let x;
    let _ = match 1 {
        1 => x = 10,
        _ => x = 20,
    };

    // using tuples would be possible, but not always preferable
    let x;
    let y;
    if true {
        x = 1;
        y = 2;
    } else {
        x = 3;
        y = 4;
    }

    // could match with a smarter heuristic to avoid multiple assignments
    let x;
    if true {
        let mut y = 5;
        y = 6;
        x = y;
    } else {
        x = 2;
    }

    let (x, y);
    if true {
        x = 1;
    } else {
        x = 2;
    }
    y = 3;

    let x;
    inline!($x = 1;);

    let x;
    if true {
        inline!($x = 1;);
    } else {
        x = 2;
    }

    inline!({
        let x;
        x = 1;

        let x;
        if true {
            x = 1;
        } else {
            x = 2;
        }
    });

    // ignore if-lets - https://github.com/rust-lang/rust-clippy/issues/8613
    let x;
    if let Some(n) = Some("v") {
        x = 1;
    } else {
        x = 2;
    }

    let x;
    if true && let Some(n) = Some("let chains too") {
        x = 1;
    } else {
        x = 2;
    }

    // ignore mut bindings
    // https://github.com/shepmaster/twox-hash/blob/b169c16d86eb8ea4a296b0acb9d00ca7e3c3005f/src/sixty_four.rs#L88-L93
    // https://github.com/dtolnay/thiserror/blob/21c26903e29cb92ba1a7ff11e82ae2001646b60d/tests/test_generics.rs#L91-L100
    let mut x: usize;
    x = 1;
    x = 2;
    x = 3;

    // should not move the declaration if `x` has a significant drop, and there
    // is another binding with a significant drop between it and the first usage
    let x;
    let y = SignificantDrop;
    x = SignificantDrop;
}

#[rustfmt::skip]
fn issue8911() -> u32 {
    let x;
    match 1 {
        _ if { x = 1; false } => return 1,
        _ => return 2,
    }

    let x;
    if { x = 1; true } {
        return 1;
    } else {
        return 2;
    }

    3
}
