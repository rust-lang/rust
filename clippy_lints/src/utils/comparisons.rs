use rustc::hir::{BinOp_, Expr};

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum Rel {
    Lt,
    Le,
    Eq,
    Ne,
}

/// Put the expression in the form  `lhs < rhs` or `lhs <= rhs`.
pub fn normalize_comparison<'a>(op: BinOp_, lhs: &'a Expr, rhs: &'a Expr)
                                -> Option<(Rel, &'a Expr, &'a Expr)> {
    match op {
        BinOp_::BiLt => Some((Rel::Lt, lhs, rhs)),
        BinOp_::BiLe => Some((Rel::Le, lhs, rhs)),
        BinOp_::BiGt => Some((Rel::Lt, rhs, lhs)),
        BinOp_::BiGe => Some((Rel::Le, rhs, lhs)),
        BinOp_::BiEq => Some((Rel::Eq, rhs, lhs)),
        BinOp_::BiNe => Some((Rel::Ne, rhs, lhs)),
        _ => None,
    }
}
