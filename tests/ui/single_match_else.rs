#![warn(clippy::single_match_else)]

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

fn main() {}
