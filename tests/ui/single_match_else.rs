// aux-build: proc_macros.rs
#![warn(clippy::single_match_else)]
#![allow(clippy::needless_return, clippy::no_effect, clippy::uninlined_format_args)]

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
