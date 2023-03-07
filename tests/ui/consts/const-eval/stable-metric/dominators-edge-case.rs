// check-pass
//
// Exercising an edge case which was found during Stage 2 compilation.
// Compilation would fail for this code when running the `CtfeLimit`
// MirPass (specifically when looking up the dominators).
#![crate_type="lib"]

const DUMMY: Expr = Expr::Path(ExprPath {
    attrs: Vec::new(),
    path: Vec::new(),
});

pub enum Expr {
    Path(ExprPath),
}
pub struct ExprPath {
    pub attrs: Vec<()>,
    pub path: Vec<()>,
}
