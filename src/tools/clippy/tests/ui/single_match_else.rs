#![warn(clippy::single_match_else)]
#![allow(clippy::needless_return)]
#![allow(clippy::no_effect)]

enum ExprNode {
    ExprAddrOf,
    Butterflies,
    Unicorns,
}

static NODE: ExprNode = ExprNode::Unicorns;

fn unwrap_addr() -> Option<&'static ExprNode> {
    match ExprNode::Butterflies {
        ExprNode::ExprAddrOf => Some(&NODE),
        _ => {
            let x = 5;
            None
        },
    }
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
}
