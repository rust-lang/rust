// rustfmt-match_arm_indent: false

fn multiple() {
    match body.kind {
    // We do not allow `if` to stay on the same line, since we could easily mistake
    // `pat => if cond { ... }` and `pat if cond => { ... }`.
    ast::ExprKind::If(..) => false,
    // We do not allow collapsing a block around expression with condition
    // to avoid it being cluttered with match arm.
    ast::ExprKind::ForLoop { .. } | ast::ExprKind::While(..) => false,
    ast::ExprKind::Loop(..)
    | ast::ExprKind::Match(..)
    | ast::ExprKind::Block(..)
    | ast::ExprKind::Closure(..)
    | ast::ExprKind::Array(..)
    | ast::ExprKind::Call(..)
    | ast::ExprKind::MethodCall(..)
    | ast::ExprKind::MacCall(..)
    | ast::ExprKind::Struct(..)
    | ast::ExprKind::Tup(..) => true,
    ast::ExprKind::AddrOf(_, _, ref expr)
    | ast::ExprKind::Try(ref expr)
    | ast::ExprKind::Unary(_, ref expr)
    | ast::ExprKind::Index(ref expr, _, _)
    | ast::ExprKind::Cast(ref expr, _) => can_flatten_block_around_this(expr),
    _ => false,
    }
}
