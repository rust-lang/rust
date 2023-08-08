//! Various helper functions to work with SyntaxNodes.
use itertools::Itertools;
use parser::T;
use syntax::{
    ast::{self, HasLoopBody, MacroCall, PathSegmentKind, VisibilityKind},
    AstNode, AstToken, Preorder, RustLanguage, WalkEvent,
};

pub fn expr_as_name_ref(expr: &ast::Expr) -> Option<ast::NameRef> {
    if let ast::Expr::PathExpr(expr) = expr {
        let path = expr.path()?;
        path.as_single_name_ref()
    } else {
        None
    }
}

pub fn full_path_of_name_ref(name_ref: &ast::NameRef) -> Option<ast::Path> {
    let mut ancestors = name_ref.syntax().ancestors();
    let _ = ancestors.next()?; // skip self
    let _ = ancestors.next().filter(|it| ast::PathSegment::can_cast(it.kind()))?; // skip self
    ancestors.take_while(|it| ast::Path::can_cast(it.kind())).last().and_then(ast::Path::cast)
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
            if let_stmt.initializer().map(|it| it.syntax() != &node).unwrap_or(true)
                && let_stmt.let_else().map(|it| it.syntax() != &node).unwrap_or(true)
            {
                // skipping potential const pat expressions in  let statements
                preorder.skip_subtree();
                continue;
            }
        }

        match ast::Stmt::cast(node.clone()) {
            // Don't skip subtree since we want to process the expression child next
            Some(ast::Stmt::ExprStmt(_)) | Some(ast::Stmt::LetStmt(_)) => (),
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
// FIXME: Make the control flow more proper
pub fn walk_ty(ty: &ast::Type, cb: &mut dyn FnMut(ast::Type) -> bool) {
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
                cb(ty);
            }
            Some(ty) => {
                if cb(ty) {
                    preorder.skip_subtree();
                }
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

/// Returns the `let` only if there is exactly one (that is, `let pat = expr`
/// or `((let pat = expr))`, but not `let pat = expr && expr` or `non_let_expr`).
pub fn single_let(expr: ast::Expr) -> Option<ast::LetExpr> {
    match expr {
        ast::Expr::ParenExpr(expr) => expr.expr().and_then(single_let),
        ast::Expr::LetExpr(expr) => Some(expr),
        _ => None,
    }
}

pub fn is_pattern_cond(expr: ast::Expr) -> bool {
    match expr {
        ast::Expr::BinExpr(expr)
            if expr.op_kind() == Some(ast::BinaryOp::LogicOp(ast::LogicOp::And)) =>
        {
            expr.lhs()
                .map(is_pattern_cond)
                .or_else(|| expr.rhs().map(is_pattern_cond))
                .unwrap_or(false)
        }
        ast::Expr::ParenExpr(expr) => expr.expr().map_or(false, is_pattern_cond),
        ast::Expr::LetExpr(_) => true,
        _ => false,
    }
}

/// Calls `cb` on each expression inside `expr` that is at "tail position".
/// Does not walk into `break` or `return` expressions.
/// Note that modifying the tree while iterating it will cause undefined iteration which might
/// potentially results in an out of bounds panic.
pub fn for_each_tail_expr(expr: &ast::Expr, cb: &mut dyn FnMut(&ast::Expr)) {
    let walk_loop = |cb: &mut dyn FnMut(&ast::Expr), label, body: Option<ast::BlockExpr>| {
        for_each_break_expr(label, body.and_then(|it| it.stmt_list()), &mut |b| {
            cb(&ast::Expr::BreakExpr(b))
        })
    };
    match expr {
        ast::Expr::BlockExpr(b) => {
            match b.modifier() {
                Some(
                    ast::BlockModifier::Async(_)
                    | ast::BlockModifier::Try(_)
                    | ast::BlockModifier::Const(_),
                ) => return cb(expr),

                Some(ast::BlockModifier::Label(label)) => {
                    for_each_break_expr(Some(label), b.stmt_list(), &mut |b| {
                        cb(&ast::Expr::BreakExpr(b))
                    });
                }
                Some(ast::BlockModifier::Unsafe(_)) => (),
                None => (),
            }
            if let Some(stmt_list) = b.stmt_list() {
                if let Some(e) = stmt_list.tail_expr() {
                    for_each_tail_expr(&e, cb);
                }
            }
        }
        ast::Expr::IfExpr(if_) => {
            let mut if_ = if_.clone();
            loop {
                if let Some(block) = if_.then_branch() {
                    for_each_tail_expr(&ast::Expr::BlockExpr(block), cb);
                }
                match if_.else_branch() {
                    Some(ast::ElseBranch::IfExpr(it)) => if_ = it,
                    Some(ast::ElseBranch::Block(block)) => {
                        for_each_tail_expr(&ast::Expr::BlockExpr(block), cb);
                        break;
                    }
                    None => break,
                }
            }
        }
        ast::Expr::LoopExpr(l) => walk_loop(cb, l.label(), l.loop_body()),
        ast::Expr::WhileExpr(w) => walk_loop(cb, w.label(), w.loop_body()),
        ast::Expr::ForExpr(f) => walk_loop(cb, f.label(), f.loop_body()),
        ast::Expr::MatchExpr(m) => {
            if let Some(arms) = m.match_arm_list() {
                arms.arms().filter_map(|arm| arm.expr()).for_each(|e| for_each_tail_expr(&e, cb));
            }
        }
        ast::Expr::ArrayExpr(_)
        | ast::Expr::AwaitExpr(_)
        | ast::Expr::BinExpr(_)
        | ast::Expr::BoxExpr(_)
        | ast::Expr::BreakExpr(_)
        | ast::Expr::CallExpr(_)
        | ast::Expr::CastExpr(_)
        | ast::Expr::ClosureExpr(_)
        | ast::Expr::ContinueExpr(_)
        | ast::Expr::FieldExpr(_)
        | ast::Expr::IndexExpr(_)
        | ast::Expr::Literal(_)
        | ast::Expr::MacroExpr(_)
        | ast::Expr::MethodCallExpr(_)
        | ast::Expr::ParenExpr(_)
        | ast::Expr::PathExpr(_)
        | ast::Expr::PrefixExpr(_)
        | ast::Expr::RangeExpr(_)
        | ast::Expr::RecordExpr(_)
        | ast::Expr::RefExpr(_)
        | ast::Expr::ReturnExpr(_)
        | ast::Expr::TryExpr(_)
        | ast::Expr::TupleExpr(_)
        | ast::Expr::LetExpr(_)
        | ast::Expr::UnderscoreExpr(_)
        | ast::Expr::YieldExpr(_)
        | ast::Expr::YeetExpr(_) => cb(expr),
    }
}

pub fn for_each_break_and_continue_expr(
    label: Option<ast::Label>,
    body: Option<ast::StmtList>,
    cb: &mut dyn FnMut(ast::Expr),
) {
    let label = label.and_then(|lbl| lbl.lifetime());
    if let Some(b) = body {
        let tree_depth_iterator = TreeWithDepthIterator::new(b);
        for (expr, depth) in tree_depth_iterator {
            match expr {
                ast::Expr::BreakExpr(b)
                    if (depth == 0 && b.lifetime().is_none())
                        || eq_label_lt(&label, &b.lifetime()) =>
                {
                    cb(ast::Expr::BreakExpr(b));
                }
                ast::Expr::ContinueExpr(c)
                    if (depth == 0 && c.lifetime().is_none())
                        || eq_label_lt(&label, &c.lifetime()) =>
                {
                    cb(ast::Expr::ContinueExpr(c));
                }
                _ => (),
            }
        }
    }
}

fn for_each_break_expr(
    label: Option<ast::Label>,
    body: Option<ast::StmtList>,
    cb: &mut dyn FnMut(ast::BreakExpr),
) {
    let label = label.and_then(|lbl| lbl.lifetime());
    if let Some(b) = body {
        let tree_depth_iterator = TreeWithDepthIterator::new(b);
        for (expr, depth) in tree_depth_iterator {
            match expr {
                ast::Expr::BreakExpr(b)
                    if (depth == 0 && b.lifetime().is_none())
                        || eq_label_lt(&label, &b.lifetime()) =>
                {
                    cb(b);
                }
                _ => (),
            }
        }
    }
}

fn eq_label_lt(lt1: &Option<ast::Lifetime>, lt2: &Option<ast::Lifetime>) -> bool {
    lt1.as_ref().zip(lt2.as_ref()).map_or(false, |(lt, lbl)| lt.text() == lbl.text())
}

struct TreeWithDepthIterator {
    preorder: Preorder<RustLanguage>,
    depth: u32,
}

impl TreeWithDepthIterator {
    fn new(body: ast::StmtList) -> Self {
        let preorder = body.syntax().preorder();
        Self { preorder, depth: 0 }
    }
}

impl Iterator for TreeWithDepthIterator {
    type Item = (ast::Expr, u32);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(event) = self.preorder.find_map(|ev| match ev {
            WalkEvent::Enter(it) => ast::Expr::cast(it).map(WalkEvent::Enter),
            WalkEvent::Leave(it) => ast::Expr::cast(it).map(WalkEvent::Leave),
        }) {
            match event {
                WalkEvent::Enter(
                    ast::Expr::LoopExpr(_) | ast::Expr::WhileExpr(_) | ast::Expr::ForExpr(_),
                ) => {
                    self.depth += 1;
                }
                WalkEvent::Leave(
                    ast::Expr::LoopExpr(_) | ast::Expr::WhileExpr(_) | ast::Expr::ForExpr(_),
                ) => {
                    self.depth -= 1;
                }
                WalkEvent::Enter(ast::Expr::BlockExpr(e)) if e.label().is_some() => {
                    self.depth += 1;
                }
                WalkEvent::Leave(ast::Expr::BlockExpr(e)) if e.label().is_some() => {
                    self.depth -= 1;
                }
                WalkEvent::Enter(expr) => return Some((expr, self.depth)),
                _ => (),
            }
        }
        None
    }
}

/// Parses the input token tree as comma separated plain paths.
pub fn parse_tt_as_comma_sep_paths(input: ast::TokenTree) -> Option<Vec<ast::Path>> {
    let r_paren = input.r_paren_token();
    let tokens =
        input.syntax().children_with_tokens().skip(1).map_while(|it| match it.into_token() {
            // seeing a keyword means the attribute is unclosed so stop parsing here
            Some(tok) if tok.kind().is_keyword() => None,
            // don't include the right token tree parenthesis if it exists
            tok @ Some(_) if tok == r_paren => None,
            // only nodes that we can find are other TokenTrees, those are unexpected in this parse though
            None => None,
            Some(tok) => Some(tok),
        });
    let input_expressions = tokens.group_by(|tok| tok.kind() == T![,]);
    let paths = input_expressions
        .into_iter()
        .filter_map(|(is_sep, group)| (!is_sep).then_some(group))
        .filter_map(|mut tokens| {
            syntax::hacks::parse_expr_from_str(&tokens.join("")).and_then(|expr| match expr {
                ast::Expr::PathExpr(it) => it.path(),
                _ => None,
            })
        })
        .collect();
    Some(paths)
}

pub fn macro_call_for_string_token(string: &ast::String) -> Option<MacroCall> {
    let macro_call = string.syntax().parent_ancestors().find_map(ast::MacroCall::cast)?;
    Some(macro_call)
}
