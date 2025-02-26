//@ check-pass
//@ revisions: gate nogate
#![cfg_attr(gate, feature(generic_arg_infer))]

fn main() {
    // AST Types preserve parens for pretty printing reasons. This means
    // that this is parsed as a `TyKind::Paren(TyKind::Infer)`. Generic
    // arg lowering therefore needs to take into account not just `TyKind::Infer`
    // but `TyKind::Infer` wrapped in arbitrarily many `TyKind::Paren`.
    let a: Vec<(_)> = vec![1_u8];
    let a: Vec<(((((_)))))> = vec![1_u8];
}
