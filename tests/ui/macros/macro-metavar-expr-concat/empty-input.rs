// Issue 50403
// Ensure that `concat` can't create empty identifiers
// FIXME(macro_metavar_expr_concat): this error message could be improved

macro_rules! empty {
    () => { ${concat()} } //~ ERROR expected identifier or string literal
                          //~^ERROR expected expression
}

fn main() {
    let x = empty!();
}
