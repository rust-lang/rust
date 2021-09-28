//! Various helper functions to work with SyntaxNodes.
use syntax::{
    ast::{self, PathSegmentKind, VisibilityKind},
    AstNode, WalkEvent,
};

pub fn expr_as_name_ref(expr: &ast::Expr) -> Option<ast::NameRef> {
    if let ast::Expr::PathExpr(expr) = expr {
        let path = expr.path()?;
        path.as_single_name_ref()
    } else {
        None
    }
}

pub fn block_as_lone_tail(block: &ast::BlockExpr) -> Option<ast::Expr> {
    block.statements().next().is_none().then(|| block.tail_expr()).flatten()
}

/// Preorder walk all the expression's child expressions.
pub fn walk_expr(expr: &ast::Expr, cb: &mut dyn FnMut(ast::Expr)) {
    preorder_expr(expr, &mut |ev| {
        if let WalkEvent::Enter(expr) = ev {
            cb(expr);
        }
        false
    })
}

/// Preorder walk all the expression's child expressions preserving events.
/// If the callback returns true on an [`WalkEvent::Enter`], the subtree of the expression will be skipped.
/// Note that the subtree may already be skipped due to the context analysis this function does.
pub fn preorder_expr(start: &ast::Expr, cb: &mut dyn FnMut(WalkEvent<ast::Expr>) -> bool) {
    let mut preorder = start.syntax().preorder();
    while let Some(event) = preorder.next() {
        let node = match event {
            WalkEvent::Enter(node) => node,
            WalkEvent::Leave(node) => {
                if let Some(expr) = ast::Expr::cast(node) {
                    cb(WalkEvent::Leave(expr));
                }
                continue;
            }
        };
        if let Some(let_stmt) = node.parent().and_then(ast::LetStmt::cast) {
            if Some(node.clone()) != let_stmt.initializer().map(|it| it.syntax().clone()) {
                // skipping potential const pat expressions in  let statements
                preorder.skip_subtree();
                continue;
            }
        }

        match ast::Stmt::cast(node.clone()) {
            // Don't skip subtree since we want to process the expression child next
            Some(ast::Stmt::ExprStmt(_)) | Some(ast::Stmt::LetStmt(_)) => (),
            // This might be an expression
            Some(ast::Stmt::Item(ast::Item::MacroCall(mcall))) => {
                cb(WalkEvent::Enter(ast::Expr::MacroCall(mcall)));
                preorder.skip_subtree();
            }
            // skip inner items which might have their own expressions
            Some(ast::Stmt::Item(_)) => preorder.skip_subtree(),
            None => {
                // skip const args, those expressions are a different context
                if ast::GenericArg::can_cast(node.kind()) {
                    preorder.skip_subtree();
                } else if let Some(expr) = ast::Expr::cast(node) {
                    let is_different_context = match &expr {
                        ast::Expr::BlockExpr(block_expr) => {
                            matches!(
                                block_expr.modifier(),
                                Some(
                                    ast::BlockModifier::Async(_)
                                        | ast::BlockModifier::Try(_)
                                        | ast::BlockModifier::Const(_)
                                )
                            )
                        }
                        ast::Expr::ClosureExpr(_) => true,
                        _ => false,
                    } && expr.syntax() != start.syntax();
                    let skip = cb(WalkEvent::Enter(expr));
                    if skip || is_different_context {
                        preorder.skip_subtree();
                    }
                }
            }
        }
    }
}

/// Preorder walk all the expression's child patterns.
pub fn walk_patterns_in_expr(start: &ast::Expr, cb: &mut dyn FnMut(ast::Pat)) {
    let mut preorder = start.syntax().preorder();
    while let Some(event) = preorder.next() {
        let node = match event {
            WalkEvent::Enter(node) => node,
            WalkEvent::Leave(_) => continue,
        };
        match ast::Stmt::cast(node.clone()) {
            Some(ast::Stmt::LetStmt(l)) => {
                if let Some(pat) = l.pat() {
                    walk_pat(&pat, cb);
                }
                if let Some(expr) = l.initializer() {
                    walk_patterns_in_expr(&expr, cb);
                }
                preorder.skip_subtree();
            }
            // Don't skip subtree since we want to process the expression child next
            Some(ast::Stmt::ExprStmt(_)) => (),
            // skip inner items which might have their own patterns
            Some(ast::Stmt::Item(_)) => preorder.skip_subtree(),
            None => {
                // skip const args, those are a different context
                if ast::GenericArg::can_cast(node.kind()) {
                    preorder.skip_subtree();
                } else if let Some(expr) = ast::Expr::cast(node.clone()) {
                    let is_different_context = match &expr {
                        ast::Expr::BlockExpr(block_expr) => {
                            matches!(
                                block_expr.modifier(),
                                Some(
                                    ast::BlockModifier::Async(_)
                                        | ast::BlockModifier::Try(_)
                                        | ast::BlockModifier::Const(_)
                                )
                            )
                        }
                        ast::Expr::ClosureExpr(_) => true,
                        _ => false,
                    } && expr.syntax() != start.syntax();
                    if is_different_context {
                        preorder.skip_subtree();
                    }
                } else if let Some(pat) = ast::Pat::cast(node) {
                    preorder.skip_subtree();
                    walk_pat(&pat, cb);
                }
            }
        }
    }
}

/// Preorder walk all the pattern's sub patterns.
pub fn walk_pat(pat: &ast::Pat, cb: &mut dyn FnMut(ast::Pat)) {
    let mut preorder = pat.syntax().preorder();
    while let Some(event) = preorder.next() {
        let node = match event {
            WalkEvent::Enter(node) => node,
            WalkEvent::Leave(_) => continue,
        };
        let kind = node.kind();
        match ast::Pat::cast(node) {
            Some(pat @ ast::Pat::ConstBlockPat(_)) => {
                preorder.skip_subtree();
                cb(pat);
            }
            Some(pat) => {
                cb(pat);
            }
            // skip const args
            None if ast::GenericArg::can_cast(kind) => {
                preorder.skip_subtree();
            }
            None => (),
        }
    }
}

/// Preorder walk all the type's sub types.
pub fn walk_ty(ty: &ast::Type, cb: &mut dyn FnMut(ast::Type)) {
    let mut preorder = ty.syntax().preorder();
    while let Some(event) = preorder.next() {
        let node = match event {
            WalkEvent::Enter(node) => node,
            WalkEvent::Leave(_) => continue,
        };
        let kind = node.kind();
        match ast::Type::cast(node) {
            Some(ty @ ast::Type::MacroType(_)) => {
                preorder.skip_subtree();
                cb(ty)
            }
            Some(ty) => {
                cb(ty);
            }
            // skip const args
            None if ast::ConstArg::can_cast(kind) => {
                preorder.skip_subtree();
            }
            None => (),
        }
    }
}

pub fn vis_eq(this: &ast::Visibility, other: &ast::Visibility) -> bool {
    match (this.kind(), other.kind()) {
        (VisibilityKind::In(this), VisibilityKind::In(other)) => {
            stdx::iter_eq_by(this.segments(), other.segments(), |lhs, rhs| {
                lhs.kind().zip(rhs.kind()).map_or(false, |it| match it {
                    (PathSegmentKind::CrateKw, PathSegmentKind::CrateKw)
                    | (PathSegmentKind::SelfKw, PathSegmentKind::SelfKw)
                    | (PathSegmentKind::SuperKw, PathSegmentKind::SuperKw) => true,
                    (PathSegmentKind::Name(lhs), PathSegmentKind::Name(rhs)) => {
                        lhs.text() == rhs.text()
                    }
                    _ => false,
                })
            })
        }
        (VisibilityKind::PubSelf, VisibilityKind::PubSelf)
        | (VisibilityKind::PubSuper, VisibilityKind::PubSuper)
        | (VisibilityKind::PubCrate, VisibilityKind::PubCrate)
        | (VisibilityKind::Pub, VisibilityKind::Pub) => true,
        _ => false,
    }
}
