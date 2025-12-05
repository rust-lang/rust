// rustfmt-single_line_let_else_max_width: 0

fn main() {
    let Some(a) = opt else {};

    let Some(b) = opt else { return };

    let Some(c) = opt else {
        return
    };

    let Some(c) = opt else {
        // a comment should always force the block to be multi-lined
        return
    };

    let Some(c) = opt else { /* a comment should always force the block to be multi-lined */ return };

    let Some(d) = some_very_very_very_very_long_name else { return };

    let Expr::Slice(ast::ExprSlice { lower, upper, step, range: _ }) = slice.as_ref() else {
        return
    };

    let Some((base_place, current)) = self.lower_expr_as_place(current, *base, true)? else {
        return Ok(None)
    };

    let Some(doc_attr) = variant.attrs.iter().find(|attr| attr.path().is_ident("doc")) else {
        return Err(Error::new(variant.span(), r#"expected a doc comment"#))
    };

    let Some((base_place, current)) = self.lower_expr_as_place(current, *base, true) else {
        return Ok(None)
    };

    let Stmt::Expr(Expr::Call(ExprCall { args: some_args, .. }), _) = last_stmt else {
        return Err(Error::new(last_stmt.span(), "expected last expression to be `Some(match (..) { .. })`"))
    };
}
