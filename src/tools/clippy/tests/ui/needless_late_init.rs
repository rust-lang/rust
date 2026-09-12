//@aux-build:proc_macros.rs
#![warn(clippy::needless_late_init)]
#![allow(clippy::let_and_return)]
#![expect(clippy::blocks_in_conditions, clippy::let_unit_value, clippy::useless_vec)]

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
    //~^ needless_late_init
    a = "zero";

    let b;
    //~^ needless_late_init
    let c;
    //~^ needless_late_init
    b = 1;
    c = 2;

    let d: usize;
    //~^ needless_late_init
    d = 1;

    let e;
    //~^ needless_late_init
    e = format!("{}", d);
}

fn main() {
    let a;
    //~^ needless_late_init
    let n = 1;
    match n {
        1 => a = "one",
        _ => {
            a = "two";
        },
    }

    let b;
    //~^ needless_late_init
    if n == 3 {
        b = "four";
    } else {
        b = "five"
    }

    let d;
    //~^ needless_late_init
    if true {
        let temp = 5;
        d = temp;
    } else {
        d = 15;
    }

    let e;
    //~^ needless_late_init
    if true {
        e = format!("{} {}", a, b);
    } else {
        e = format!("{}", n);
    }

    let f;
    //~^ needless_late_init
    match 1 {
        1 => f = "three",
        _ => return,
    }; // has semi

    let g: usize;
    //~^ needless_late_init
    if true {
        g = 5;
    } else {
        panic!();
    }

    // Drop order only matters if both are significant
    let x;
    //~^ needless_late_init
    let y = SignificantDrop;
    x = 1;

    let x;
    //~^ needless_late_init
    let y = 1;
    x = SignificantDrop;

    let x;
    //~^ needless_late_init
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
    //~^ needless_late_init
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
    //~^ needless_late_init
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
        //~^ needless_late_init
        x = 1;
        y = 2;
    } else {
        x = 3;
        y = 4;
    }

    let x;
    //~^ needless_late_init
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

macro_rules! issue13776_mac {
    ($var:expr, $val:literal) => {
        $var = $val;
    };
}

fn issue13776() {
    let x;
    issue13776_mac!(x, 10); // should not lint
}

fn issue9895() {
    let r;
    //~^ needless_late_init
    (r = 5);
}

fn if_or_match_in_block_expr() {
    let z;
    //~^ needless_late_init
    if true {
        z = 1;
    } else {
        z = 2;
    }
}

fn issue16330() {
    // Late init in both if branches, should lint
    let a;
    let b;
    if true {
        //~^ needless_late_init
        a = 1;
        b = 2;
    } else {
        a = 3;
        b = 4;
    }

    // One of the variables is not late init, should not lint
    let a;
    let mut b = 1;
    let c;
    if true {
        b = 1;
        a = 2;
        c = 3;
    } else {
        b = 6;
        a = 4;
        c = 5;
    }

    // One of the variables is defined outside the block, should not lint
    let b;
    {
        let a;
        let c;
        if true {
            b = 1;
            a = 2;
            c = 3;
        } else {
            b = 6;
            a = 4;
            c = 5;
        }
    }

    // Late init in all match arms, should lint
    let a;
    let b;
    let c;
    match 1 {
        //~^ needless_late_init
        1 => {
            a = 1;
            b = 2;
            c = 3;
        },
        _ if false => {
            a = 4;
            b = 5;
            c = 6;
        },
        _ => {
            a = 7;
            b = 8;
            c = 9;
        },
    }

    // Late init in all if branches, should lint
    let a;
    let b;
    let c;
    if true {
        //~^ needless_late_init
        a = 1;
        b = 2;
        c = 3;
    } else if false {
        a = 4;
        b = 5;
        c = 6;
    } else {
        a = 7;
        b = 8;
        c = 9;
    }

    // One of the variables is not assigned in all branches, should not lint
    let a;
    let b;
    let c;
    if true {
        a = 1;
        b = 2;
        c = 3;
    } else if false {
        a = 4;
        c = 6;
    } else {
        a = 7;
        b = 8;
        c = 9;
    }

    // One of the variables is assigned multiple times, should not lint
    let mut a;
    let b;
    if true {
        a = 1;
        b = 2;
        a = 3;
    } else {
        a = 4;
        b = 5;
    }

    // One of the variables is assigned in a nested block, should not lint
    let a;
    let b;
    if true {
        a = 1;
        b = 2;
    } else {
        a = 4;
        {
            b = 5;
        }
    }

    // The order of the variables is different in different branches, should not lint
    let a;
    let b;
    if true {
        a = 1;
        b = 2;
    } else {
        b = 5;
        a = 4;
    }

    // Later assignments depend on the earlier ones, should only lint the last ones
    let a;
    let b;
    let c;
    //~^ needless_late_init
    if true {
        a = 1;
        b = a + 1;
        c = b + 1;
    } else {
        a = 4;
        b = a + 2;
        c = b + 2;
    }
    let a;
    let b;
    let c;
    let d;
    if true {
        //~^ needless_late_init
        a = 1;
        b = a + 1;
        c = b + 1;
        d = 1;
    } else {
        a = 4;
        b = a + 2;
        c = b + 2;
        d = 2;
    }
}
