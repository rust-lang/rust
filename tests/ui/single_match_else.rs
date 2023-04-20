//@run-rustfix
//@aux-build: proc_macros.rs:proc-macro
#![warn(clippy::single_match_else)]
#![allow(unused, clippy::needless_return, clippy::no_effect, clippy::uninlined_format_args)]
extern crate proc_macros;
use proc_macros::with_span;

enum ExprNode {
    ExprAddrOf,
    Butterflies,
    Unicorns,
}

static NODE: ExprNode = ExprNode::Unicorns;

fn unwrap_addr() -> Option<&'static ExprNode> {
    let _ = match ExprNode::Butterflies {
        ExprNode::ExprAddrOf => Some(&NODE),
        _ => {
            let x = 5;
            None
        },
    };

    // Don't lint
    with_span!(span match ExprNode::Butterflies {
        ExprNode::ExprAddrOf => Some(&NODE),
        _ => {
            let x = 5;
            None
        },
    })
}

macro_rules! unwrap_addr {
    ($expression:expr) => {
        match $expression {
            ExprNode::ExprAddrOf => Some(&NODE),
            _ => {
                let x = 5;
                None
            },
        }
    };
}

#[rustfmt::skip]
fn main() {
    unwrap_addr!(ExprNode::Unicorns);

    //
    // don't lint single exprs/statements
    //

    // don't lint here
    match Some(1) {
        Some(a) => println!("${:?}", a),
        None => return,
    }

    // don't lint here
    match Some(1) {
        Some(a) => println!("${:?}", a),
        None => {
            return
        },
    }

    // don't lint here
    match Some(1) {
        Some(a) => println!("${:?}", a),
        None => {
            return;
        },
    }

    //
    // lint multiple exprs/statements "else" blocks
    //

    // lint here
    match Some(1) {
        Some(a) => println!("${:?}", a),
        None => {
            println!("else block");
            return
        },
    }

    // lint here
    match Some(1) {
        Some(a) => println!("${:?}", a),
        None => {
            println!("else block");
            return;
        },
    }

    // lint here
    use std::convert::Infallible;
    match Result::<i32, Infallible>::Ok(1) {
        Ok(a) => println!("${:?}", a),
        Err(_) => {
            println!("else block");
            return;
        }
    }

    use std::borrow::Cow;
    match Cow::from("moo") {
        Cow::Owned(a) => println!("${:?}", a),
        Cow::Borrowed(_) => {
            println!("else block");
            return;
        }
    }
}

fn issue_10808(bar: Option<i32>) {
    match bar {
        Some(v) => unsafe {
            let r = &v as *const i32;
            println!("{}", *r);
        },
        None => {
            println!("None1");
            println!("None2");
        },
    }

    match bar {
        Some(v) => {
            println!("Some");
            println!("{v}");
        },
        None => unsafe {
            let v = 0;
            let r = &v as *const i32;
            println!("{}", *r);
        },
    }

    match bar {
        Some(v) => unsafe {
            let r = &v as *const i32;
            println!("{}", *r);
        },
        None => unsafe {
            let v = 0;
            let r = &v as *const i32;
            println!("{}", *r);
        },
    }

    match bar {
        #[rustfmt::skip]
        Some(v) => {
            unsafe {
                let r = &v as *const i32;
                println!("{}", *r);
            }
        },
        None => {
            println!("None");
            println!("None");
        },
    }

    match bar {
        Some(v) => {
            println!("Some");
            println!("{v}");
        },
        #[rustfmt::skip]
        None => {
            unsafe {
                let v = 0;
                let r = &v as *const i32;
                println!("{}", *r);
            }
        },
    }

    match bar {
        #[rustfmt::skip]
        Some(v) => {
            unsafe {
                let r = &v as *const i32;
                println!("{}", *r);
            }
        },
        #[rustfmt::skip]
        None => {
            unsafe {
                let v = 0;
                let r = &v as *const i32;
                println!("{}", *r);
            }
        },
    }
}
