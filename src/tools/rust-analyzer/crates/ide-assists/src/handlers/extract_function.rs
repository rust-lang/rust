use std::{iter, ops::RangeInclusive};

use ast::make;
use either::Either;
use hir::{
    HasSource, HirDisplay, InFile, Local, LocalSource, ModuleDef, PathResolution, Semantics,
    TypeInfo, TypeParam,
};
use ide_db::{
    FxIndexSet, RootDatabase,
    assists::GroupLabel,
    defs::{Definition, NameRefClass},
    famous_defs::FamousDefs,
    helpers::mod_path_to_ast,
    imports::insert_use::{ImportScope, insert_use},
    search::{FileReference, ReferenceCategory, SearchScope},
    source_change::SourceChangeBuilder,
    syntax_helpers::node_ext::{
        for_each_tail_expr, preorder_expr, walk_expr, walk_pat, walk_patterns_in_expr,
    },
};
use itertools::Itertools;
use syntax::{
    Edition, SyntaxElement,
    SyntaxKind::{self, COMMENT},
    SyntaxNode, SyntaxToken, T, TextRange, TextSize, TokenAtOffset, WalkEvent,
    ast::{
        self, AstNode, AstToken, HasGenericParams, HasName, edit::IndentLevel,
        edit_in_place::Indent,
    },
    match_ast, ted,
};

use crate::{
    AssistId,
    assist_context::{AssistContext, Assists, TreeMutator},
    utils::generate_impl,
};

// Assist: extract_function
//
// Extracts selected statements and comments into new function.
//
// ```
// fn main() {
//     let n = 1;
//     $0let m = n + 2;
//     // calculate
//     let k = m + n;$0
//     let g = 3;
// }
// ```
// ->
// ```
// fn main() {
//     let n = 1;
//     fun_name(n);
//     let g = 3;
// }
//
// fn $0fun_name(n: i32) {
//     let m = n + 2;
//     // calculate
//     let k = m + n;
// }
// ```
pub(crate) fn extract_function(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let range = ctx.selection_trimmed();
    if range.is_empty() {
        return None;
    }

    let node = ctx.covering_element();
    if matches!(node.kind(), T!['{'] | T!['}'] | T!['('] | T![')'] | T!['['] | T![']']) {
        cov_mark::hit!(extract_function_in_braces_is_not_applicable);
        return None;
    }

    if node.kind() == COMMENT {
        cov_mark::hit!(extract_function_in_comment_is_not_applicable);
        return None;
    }

    let node = match node {
        syntax::NodeOrToken::Node(n) => n,
        syntax::NodeOrToken::Token(t) => t.parent()?,
    };

    let body = extraction_target(&node, range)?;

    let (locals_used, self_param) = body.analyze(&ctx.sema);

    let anchor = if self_param.is_some() { Anchor::Method } else { Anchor::Freestanding };
    let insert_after = node_to_insert_after(&body, anchor)?;
    let semantics_scope = ctx.sema.scope(&insert_after)?;
    let module = semantics_scope.module();
    let edition = semantics_scope.krate().edition(ctx.db());

    let (container_info, contains_tail_expr) = body.analyze_container(&ctx.sema, edition)?;

    let ret_ty = body.return_ty(ctx)?;
    let control_flow = body.external_control_flow(ctx, &container_info)?;
    let ret_values = body.ret_values(ctx, node.parent().as_ref().unwrap_or(&node));

    let target_range = body.text_range();

    let scope = ImportScope::find_insert_use_container(&node, &ctx.sema)?;

    acc.add_group(
        &GroupLabel("Extract into...".to_owned()),
        AssistId::refactor_extract("extract_function"),
        "Extract into function",
        target_range,
        move |builder| {
            let outliving_locals: Vec<_> = ret_values.collect();
            if stdx::never!(!outliving_locals.is_empty() && !ret_ty.is_unit()) {
                // We should not have variables that outlive body if we have expression block
                return;
            }

            let params = body.extracted_function_params(ctx, &container_info, locals_used);

            let name = make_function_name(&semantics_scope);

            let fun = Function {
                name,
                self_param,
                params,
                control_flow,
                ret_ty,
                body,
                outliving_locals,
                contains_tail_expr,
                mods: container_info,
            };

            let new_indent = IndentLevel::from_node(&insert_after);
            let old_indent = fun.body.indent_level();

            let insert_after = builder.make_syntax_mut(insert_after);

            let call_expr = make_call(ctx, &fun, old_indent);

            // Map the element range to replace into the mutable version
            let elements = match &fun.body {
                FunctionBody::Expr(expr) => {
                    // expr itself becomes the replacement target
                    let expr = &builder.make_mut(expr.clone());
                    let node = SyntaxElement::Node(expr.syntax().clone());

                    node.clone()..=node
                }
                FunctionBody::Span { parent, elements, .. } => {
                    // Map the element range into the mutable versions
                    let parent = builder.make_mut(parent.clone());

                    let start = parent
                        .syntax()
                        .children_with_tokens()
                        .nth(elements.start().index())
                        .expect("should be able to find mutable start element");

                    let end = parent
                        .syntax()
                        .children_with_tokens()
                        .nth(elements.end().index())
                        .expect("should be able to find mutable end element");

                    start..=end
                }
            };

            let has_impl_wrapper =
                insert_after.ancestors().any(|a| a.kind() == SyntaxKind::IMPL && a != insert_after);

            let fn_def = format_function(ctx, module, &fun, old_indent).clone_for_update();

            if let Some(cap) = ctx.config.snippet_cap {
                if let Some(name) = fn_def.name() {
                    builder.add_tabstop_before(cap, name);
                }
            }

            let fn_def = match fun.self_param_adt(ctx) {
                Some(adt) if anchor == Anchor::Method && !has_impl_wrapper => {
                    fn_def.indent(1.into());

                    let impl_ = generate_impl(&adt);
                    impl_.indent(new_indent);
                    impl_.get_or_create_assoc_item_list().add_item(fn_def.into());

                    impl_.syntax().clone()
                }
                _ => {
                    fn_def.indent(new_indent);

                    fn_def.syntax().clone()
                }
            };

            // There are external control flows
            if fun
                .control_flow
                .kind
                .is_some_and(|kind| matches!(kind, FlowKind::Break(_, _) | FlowKind::Continue(_)))
            {
                let scope = builder.make_import_scope_mut(scope);
                let control_flow_enum =
                    FamousDefs(&ctx.sema, module.krate()).core_ops_ControlFlow();

                if let Some(control_flow_enum) = control_flow_enum {
                    let mod_path = module.find_use_path(
                        ctx.sema.db,
                        ModuleDef::from(control_flow_enum),
                        ctx.config.insert_use.prefix_kind,
                        ctx.config.import_path_config(),
                    );

                    if let Some(mod_path) = mod_path {
                        insert_use(
                            &scope,
                            mod_path_to_ast(&mod_path, edition),
                            &ctx.config.insert_use,
                        );
                    }
                }
            }

            // Replace the call site with the call to the new function
            fixup_call_site(builder, &fun.body);
            ted::replace_all(elements, vec![call_expr.into()]);

            // Insert the newly extracted function (or impl)
            ted::insert_all_raw(
                ted::Position::after(insert_after),
                vec![make::tokens::whitespace(&format!("\n\n{new_indent}")).into(), fn_def.into()],
            );
        },
    )
}

fn make_function_name(semantics_scope: &hir::SemanticsScope<'_>) -> ast::NameRef {
    let mut names_in_scope = vec![];
    semantics_scope.process_all_names(&mut |name, _| {
        names_in_scope.push(
            name.display(semantics_scope.db, semantics_scope.krate().edition(semantics_scope.db))
                .to_string(),
        )
    });

    let default_name = "fun_name";

    let mut name = default_name.to_owned();
    let mut counter = 0;
    while names_in_scope.contains(&name) {
        counter += 1;
        name = format!("{default_name}{counter}")
    }
    make::name_ref(&name)
}

/// Try to guess what user wants to extract
///
/// We have basically have two cases:
/// * We want whole node, like `loop {}`, `2 + 2`, `{ let n = 1; }` exprs.
///   Then we can use `ast::Expr`
/// * We want a few statements for a block. E.g.
///   ```ignore
///   fn foo() -> i32 {
///     let m = 1;
///     $0
///     let n = 2;
///     let k = 3;
///     k + n
///     $0
///   }
///   ```
///
fn extraction_target(node: &SyntaxNode, selection_range: TextRange) -> Option<FunctionBody> {
    if let Some(stmt) = ast::Stmt::cast(node.clone()) {
        return match stmt {
            ast::Stmt::Item(_) => None,
            ast::Stmt::ExprStmt(_) | ast::Stmt::LetStmt(_) => FunctionBody::from_range(
                node.parent().and_then(ast::StmtList::cast)?,
                node.text_range(),
            ),
        };
    }

    // Covering element returned the parent block of one or multiple statements that have been selected
    if let Some(stmt_list) = ast::StmtList::cast(node.clone()) {
        if let Some(block_expr) = stmt_list.syntax().parent().and_then(ast::BlockExpr::cast) {
            if block_expr.syntax().text_range() == selection_range {
                return FunctionBody::from_expr(block_expr.into());
            }
        }

        // Extract the full statements.
        return FunctionBody::from_range(stmt_list, selection_range);
    }

    let expr = ast::Expr::cast(node.clone())?;
    // A node got selected fully
    if node.text_range() == selection_range {
        return FunctionBody::from_expr(expr);
    }

    node.ancestors().find_map(ast::Expr::cast).and_then(FunctionBody::from_expr)
}

#[derive(Debug)]
struct Function<'db> {
    name: ast::NameRef,
    self_param: Option<ast::SelfParam>,
    params: Vec<Param<'db>>,
    control_flow: ControlFlow<'db>,
    ret_ty: RetType<'db>,
    body: FunctionBody,
    outliving_locals: Vec<OutlivedLocal>,
    /// Whether at least one of the container's tail expr is contained in the range we're extracting.
    contains_tail_expr: bool,
    mods: ContainerInfo<'db>,
}

#[derive(Debug)]
struct Param<'db> {
    var: Local,
    ty: hir::Type<'db>,
    move_local: bool,
    requires_mut: bool,
    is_copy: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamKind {
    Value,
    MutValue,
    SharedRef,
    MutRef,
}

#[derive(Debug)]
enum FunType<'db> {
    Unit,
    Single(hir::Type<'db>),
    Tuple(Vec<hir::Type<'db>>),
}

/// Where to put extracted function definition
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
enum Anchor {
    /// Extract free function and put right after current top-level function
    Freestanding,
    /// Extract method and put right after current function in the impl-block
    Method,
}

// FIXME: ControlFlow and ContainerInfo both track some function modifiers, feels like these two should
// probably be merged somehow.
#[derive(Debug)]
struct ControlFlow<'db> {
    kind: Option<FlowKind<'db>>,
    is_async: bool,
    is_unsafe: bool,
}

/// The thing whose expression we are extracting from. Can be a function, const, static, const arg, ...
#[derive(Clone, Debug)]
struct ContainerInfo<'db> {
    is_const: bool,
    parent_loop: Option<SyntaxNode>,
    /// The function's return type, const's type etc.
    ret_type: Option<hir::Type<'db>>,
    generic_param_lists: Vec<ast::GenericParamList>,
    where_clauses: Vec<ast::WhereClause>,
    edition: Edition,
}

/// Control flow that is exported from extracted function
///
/// E.g.:
/// ```ignore
/// loop {
///     $0
///     if 42 == 42 {
///         break;
///     }
///     $0
/// }
/// ```
#[derive(Debug, Clone)]
enum FlowKind<'db> {
    /// Return with value (`return $expr;`)
    Return(Option<ast::Expr>),
    Try {
        kind: TryKind<'db>,
    },
    /// Break with label and value (`break 'label $expr;`)
    Break(Option<ast::Lifetime>, Option<ast::Expr>),
    /// Continue with label (`continue 'label;`)
    Continue(Option<ast::Lifetime>),
}

#[derive(Debug, Clone)]
enum TryKind<'db> {
    Option,
    Result { ty: hir::Type<'db> },
}

#[derive(Debug)]
enum RetType<'db> {
    Expr(hir::Type<'db>),
    Stmt,
}

impl RetType<'_> {
    fn is_unit(&self) -> bool {
        match self {
            RetType::Expr(ty) => ty.is_unit(),
            RetType::Stmt => true,
        }
    }
}

/// Semantically same as `ast::Expr`, but preserves identity when using only part of the Block
/// This is the future function body, the part that is being extracted.
#[derive(Debug)]
enum FunctionBody {
    Expr(ast::Expr),
    Span { parent: ast::StmtList, elements: RangeInclusive<SyntaxElement>, text_range: TextRange },
}

#[derive(Debug)]
struct OutlivedLocal {
    local: Local,
    mut_usage_outside_body: bool,
}

/// Container of local variable usages
///
/// Semantically same as `UsageSearchResult`, but provides more convenient interface
struct LocalUsages(ide_db::search::UsageSearchResult);

impl LocalUsages {
    fn find_local_usages(ctx: &AssistContext<'_>, var: Local) -> Self {
        Self(
            Definition::Local(var)
                .usages(&ctx.sema)
                .in_scope(&SearchScope::single_file(ctx.file_id()))
                .all(),
        )
    }

    fn iter(&self) -> impl Iterator<Item = &FileReference> + '_ {
        self.0.iter().flat_map(|(_, rs)| rs)
    }
}

impl<'db> Function<'db> {
    fn return_type(&self, ctx: &AssistContext<'db>) -> FunType<'db> {
        match &self.ret_ty {
            RetType::Expr(ty) if ty.is_unit() => FunType::Unit,
            RetType::Expr(ty) => FunType::Single(ty.clone()),
            RetType::Stmt => match self.outliving_locals.as_slice() {
                [] => FunType::Unit,
                [var] => FunType::Single(var.local.ty(ctx.db())),
                vars => {
                    let types = vars.iter().map(|v| v.local.ty(ctx.db())).collect();
                    FunType::Tuple(types)
                }
            },
        }
    }

    fn self_param_adt(&self, ctx: &AssistContext<'_>) -> Option<ast::Adt> {
        let self_param = self.self_param.as_ref()?;
        let def = ctx.sema.to_def(self_param)?;
        let adt = def.ty(ctx.db()).strip_references().as_adt()?;
        let InFile { file_id: _, value } = adt.source(ctx.db())?;
        Some(value)
    }
}

impl ParamKind {
    fn is_ref(&self) -> bool {
        matches!(self, ParamKind::SharedRef | ParamKind::MutRef)
    }
}

impl<'db> Param<'db> {
    fn kind(&self) -> ParamKind {
        match (self.move_local, self.requires_mut, self.is_copy) {
            (false, true, _) => ParamKind::MutRef,
            (false, false, false) => ParamKind::SharedRef,
            (true, true, _) => ParamKind::MutValue,
            (_, false, _) => ParamKind::Value,
        }
    }

    fn to_arg(&self, ctx: &AssistContext<'db>, edition: Edition) -> ast::Expr {
        let var = path_expr_from_local(ctx, self.var, edition);
        match self.kind() {
            ParamKind::Value | ParamKind::MutValue => var,
            ParamKind::SharedRef => make::expr_ref(var, false),
            ParamKind::MutRef => make::expr_ref(var, true),
        }
    }

    fn to_param(
        &self,
        ctx: &AssistContext<'_>,
        module: hir::Module,
        edition: Edition,
    ) -> ast::Param {
        let var = self.var.name(ctx.db()).display(ctx.db(), edition).to_string();
        let var_name = make::name(&var);
        let pat = match self.kind() {
            ParamKind::MutValue => make::ident_pat(false, true, var_name),
            ParamKind::Value | ParamKind::SharedRef | ParamKind::MutRef => {
                make::ext::simple_ident_pat(var_name)
            }
        };

        let ty = make_ty(&self.ty, ctx, module);
        let ty = match self.kind() {
            ParamKind::Value | ParamKind::MutValue => ty,
            ParamKind::SharedRef => make::ty_ref(ty, false),
            ParamKind::MutRef => make::ty_ref(ty, true),
        };

        make::param(pat.into(), ty)
    }
}

impl<'db> TryKind<'db> {
    fn of_ty(
        ty: hir::Type<'db>,
        ctx: &AssistContext<'db>,
        edition: Edition,
    ) -> Option<TryKind<'db>> {
        if ty.is_unknown() {
            // We favour Result for `expr?`
            return Some(TryKind::Result { ty });
        }
        let adt = ty.as_adt()?;
        let name = adt.name(ctx.db());
        // FIXME: use lang items to determine if it is std type or user defined
        //        E.g. if user happens to define type named `Option`, we would have false positive
        let name = &name.display(ctx.db(), edition).to_string();
        match name.as_str() {
            "Option" => Some(TryKind::Option),
            "Result" => Some(TryKind::Result { ty }),
            _ => None,
        }
    }
}

impl<'db> FlowKind<'db> {
    fn make_result_handler(&self, expr: Option<ast::Expr>) -> ast::Expr {
        match self {
            FlowKind::Return(_) => make::expr_return(expr),
            FlowKind::Break(label, _) => make::expr_break(label.clone(), expr),
            FlowKind::Try { .. } => {
                stdx::never!("cannot have result handler with try");
                expr.unwrap_or_else(|| make::expr_return(None))
            }
            FlowKind::Continue(label) => {
                stdx::always!(expr.is_none(), "continue with value is not possible");
                make::expr_continue(label.clone())
            }
        }
    }

    fn expr_ty(&self, ctx: &AssistContext<'db>) -> Option<hir::Type<'db>> {
        match self {
            FlowKind::Return(Some(expr)) | FlowKind::Break(_, Some(expr)) => {
                ctx.sema.type_of_expr(expr).map(TypeInfo::adjusted)
            }
            FlowKind::Try { .. } => {
                stdx::never!("try does not have defined expr_ty");
                None
            }
            _ => None,
        }
    }
}

impl FunctionBody {
    fn parent(&self) -> Option<SyntaxNode> {
        match self {
            FunctionBody::Expr(expr) => expr.syntax().parent(),
            FunctionBody::Span { parent, .. } => Some(parent.syntax().clone()),
        }
    }

    fn node(&self) -> &SyntaxNode {
        match self {
            FunctionBody::Expr(e) => e.syntax(),
            FunctionBody::Span { parent, .. } => parent.syntax(),
        }
    }

    fn extracted_from_trait_impl(&self) -> bool {
        match self.node().ancestors().find_map(ast::Impl::cast) {
            Some(c) => c.trait_().is_some(),
            None => false,
        }
    }

    fn descendants(&self) -> impl Iterator<Item = SyntaxNode> {
        match self {
            FunctionBody::Expr(expr) => expr.syntax().descendants(),
            FunctionBody::Span { parent, .. } => parent.syntax().descendants(),
        }
    }

    fn descendant_paths(&self) -> impl Iterator<Item = ast::Path> {
        self.descendants().filter_map(|node| {
            match_ast! {
                match node {
                    ast::Path(it) => Some(it),
                    _ => None
                }
            }
        })
    }

    fn from_expr(expr: ast::Expr) -> Option<Self> {
        match expr {
            ast::Expr::BreakExpr(it) => it.expr().map(Self::Expr),
            ast::Expr::ReturnExpr(it) => it.expr().map(Self::Expr),
            ast::Expr::BlockExpr(it) if !it.is_standalone() => None,
            expr => Some(Self::Expr(expr)),
        }
    }

    fn from_range(parent: ast::StmtList, selected: TextRange) -> Option<FunctionBody> {
        let full_body = parent.syntax().children_with_tokens();

        // Get all of the elements intersecting with the selection
        let mut stmts_in_selection = full_body
            .filter(|it| ast::Stmt::can_cast(it.kind()) || it.kind() == COMMENT)
            .filter(|it| selected.intersect(it.text_range()).filter(|it| !it.is_empty()).is_some());

        let first_element = stmts_in_selection.next();

        // If the tail expr is part of the selection too, make that the last element
        // Otherwise use the last stmt
        let last_element = if let Some(tail_expr) =
            parent.tail_expr().filter(|it| selected.intersect(it.syntax().text_range()).is_some())
        {
            Some(tail_expr.syntax().clone().into())
        } else {
            stmts_in_selection.last()
        };

        let elements = match (first_element, last_element) {
            (None, _) => {
                cov_mark::hit!(extract_function_empty_selection_is_not_applicable);
                return None;
            }
            (Some(first), None) => first.clone()..=first,
            (Some(first), Some(last)) => first..=last,
        };

        let text_range = elements.start().text_range().cover(elements.end().text_range());

        Some(Self::Span { parent, elements, text_range })
    }

    fn indent_level(&self) -> IndentLevel {
        match &self {
            FunctionBody::Expr(expr) => IndentLevel::from_node(expr.syntax()),
            FunctionBody::Span { parent, .. } => IndentLevel::from_node(parent.syntax()) + 1,
        }
    }

    fn tail_expr(&self) -> Option<ast::Expr> {
        match &self {
            FunctionBody::Expr(expr) => Some(expr.clone()),
            FunctionBody::Span { parent, text_range, .. } => {
                let tail_expr = parent.tail_expr()?;
                text_range.contains_range(tail_expr.syntax().text_range()).then_some(tail_expr)
            }
        }
    }

    fn walk_expr(&self, cb: &mut dyn FnMut(ast::Expr)) {
        match self {
            FunctionBody::Expr(expr) => walk_expr(expr, cb),
            FunctionBody::Span { parent, text_range, .. } => {
                parent
                    .statements()
                    .filter(|stmt| text_range.contains_range(stmt.syntax().text_range()))
                    .filter_map(|stmt| match stmt {
                        ast::Stmt::ExprStmt(expr_stmt) => expr_stmt.expr(),
                        ast::Stmt::Item(_) => None,
                        ast::Stmt::LetStmt(stmt) => stmt.initializer(),
                    })
                    .for_each(|expr| walk_expr(&expr, cb));
                if let Some(expr) = parent
                    .tail_expr()
                    .filter(|it| text_range.contains_range(it.syntax().text_range()))
                {
                    walk_expr(&expr, cb);
                }
            }
        }
    }

    fn preorder_expr(&self, cb: &mut dyn FnMut(WalkEvent<ast::Expr>) -> bool) {
        match self {
            FunctionBody::Expr(expr) => preorder_expr(expr, cb),
            FunctionBody::Span { parent, text_range, .. } => {
                parent
                    .statements()
                    .filter(|stmt| text_range.contains_range(stmt.syntax().text_range()))
                    .filter_map(|stmt| match stmt {
                        ast::Stmt::ExprStmt(expr_stmt) => expr_stmt.expr(),
                        ast::Stmt::Item(_) => None,
                        ast::Stmt::LetStmt(stmt) => stmt.initializer(),
                    })
                    .for_each(|expr| preorder_expr(&expr, cb));
                if let Some(expr) = parent
                    .tail_expr()
                    .filter(|it| text_range.contains_range(it.syntax().text_range()))
                {
                    preorder_expr(&expr, cb);
                }
            }
        }
    }

    fn walk_pat(&self, cb: &mut dyn FnMut(ast::Pat)) {
        match self {
            FunctionBody::Expr(expr) => walk_patterns_in_expr(expr, cb),
            FunctionBody::Span { parent, text_range, .. } => {
                parent
                    .statements()
                    .filter(|stmt| text_range.contains_range(stmt.syntax().text_range()))
                    .for_each(|stmt| match stmt {
                        ast::Stmt::ExprStmt(expr_stmt) => {
                            if let Some(expr) = expr_stmt.expr() {
                                walk_patterns_in_expr(&expr, cb)
                            }
                        }
                        ast::Stmt::Item(_) => (),
                        ast::Stmt::LetStmt(stmt) => {
                            if let Some(pat) = stmt.pat() {
                                _ = walk_pat(&pat, &mut |pat| {
                                    cb(pat);
                                    std::ops::ControlFlow::<(), ()>::Continue(())
                                });
                            }
                            if let Some(expr) = stmt.initializer() {
                                walk_patterns_in_expr(&expr, cb);
                            }
                        }
                    });
                if let Some(expr) = parent
                    .tail_expr()
                    .filter(|it| text_range.contains_range(it.syntax().text_range()))
                {
                    walk_patterns_in_expr(&expr, cb);
                }
            }
        }
    }

    fn text_range(&self) -> TextRange {
        match self {
            FunctionBody::Expr(expr) => expr.syntax().text_range(),
            &FunctionBody::Span { text_range, .. } => text_range,
        }
    }

    fn contains_range(&self, range: TextRange) -> bool {
        self.text_range().contains_range(range)
    }

    fn precedes_range(&self, range: TextRange) -> bool {
        self.text_range().end() <= range.start()
    }

    fn contains_node(&self, node: &SyntaxNode) -> bool {
        self.contains_range(node.text_range())
    }
}

impl FunctionBody {
    /// Analyzes a function body, returning the used local variables that are referenced in it as well as
    /// whether it contains an await expression.
    fn analyze(
        &self,
        sema: &Semantics<'_, RootDatabase>,
    ) -> (FxIndexSet<Local>, Option<ast::SelfParam>) {
        let mut self_param = None;
        let mut res = FxIndexSet::default();

        fn local_from_name_ref(
            sema: &Semantics<'_, RootDatabase>,
            name_ref: ast::NameRef,
        ) -> Option<hir::Local> {
            match NameRefClass::classify(sema, &name_ref) {
                Some(
                    NameRefClass::Definition(Definition::Local(local_ref), _)
                    | NameRefClass::FieldShorthand { local_ref, field_ref: _, adt_subst: _ },
                ) => Some(local_ref),
                _ => None,
            }
        }

        let mut add_name_if_local = |local_ref: Local| {
            let InFile { file_id, value } = local_ref.primary_source(sema.db).source;
            // locals defined inside macros are not relevant to us
            if !file_id.is_macro() {
                match value {
                    Either::Right(it) => {
                        self_param.replace(it);
                    }
                    Either::Left(_) => {
                        res.insert(local_ref);
                    }
                }
            }
        };
        self.walk_expr(&mut |expr| match expr {
            ast::Expr::PathExpr(path_expr) => {
                if let Some(local) = path_expr
                    .path()
                    .and_then(|it| it.as_single_name_ref())
                    .and_then(|name_ref| local_from_name_ref(sema, name_ref))
                {
                    add_name_if_local(local);
                }
            }
            ast::Expr::ClosureExpr(closure_expr) => {
                if let Some(body) = closure_expr.body() {
                    body.syntax()
                        .descendants()
                        .filter_map(ast::NameRef::cast)
                        .filter_map(|name_ref| local_from_name_ref(sema, name_ref))
                        .for_each(&mut add_name_if_local);
                }
            }
            ast::Expr::MacroExpr(expr) => {
                if let Some(tt) = expr.macro_call().and_then(|call| call.token_tree()) {
                    tt.syntax()
                        .descendants_with_tokens()
                        .filter_map(SyntaxElement::into_token)
                        .filter(|it| {
                            matches!(it.kind(), SyntaxKind::STRING | SyntaxKind::IDENT | T![self])
                        })
                        .for_each(|t| {
                            if ast::String::can_cast(t.kind()) {
                                if let Some(parts) =
                                    ast::String::cast(t).and_then(|s| sema.as_format_args_parts(&s))
                                {
                                    parts
                                        .into_iter()
                                        .filter_map(|(_, value)| value.and_then(|it| it.left()))
                                        .filter_map(|path| match path {
                                            PathResolution::Local(local) => Some(local),
                                            _ => None,
                                        })
                                        .for_each(&mut add_name_if_local);
                                }
                            } else {
                                sema.descend_into_macros_exact(t)
                                    .into_iter()
                                    .filter_map(|t| t.parent().and_then(ast::NameRef::cast))
                                    .filter_map(|name_ref| local_from_name_ref(sema, name_ref))
                                    .for_each(&mut add_name_if_local);
                            }
                        });
                }
            }
            _ => (),
        });
        (res, self_param)
    }

    fn analyze_container<'db>(
        &self,
        sema: &Semantics<'db, RootDatabase>,
        edition: Edition,
    ) -> Option<(ContainerInfo<'db>, bool)> {
        let mut ancestors = self.parent()?.ancestors();
        let infer_expr_opt = |expr| sema.type_of_expr(&expr?).map(TypeInfo::adjusted);
        let mut parent_loop = None;
        let mut set_parent_loop = |loop_: &dyn ast::HasLoopBody| {
            if loop_
                .loop_body()
                .is_some_and(|it| it.syntax().text_range().contains_range(self.text_range()))
            {
                parent_loop.get_or_insert(loop_.syntax().clone());
            }
        };

        let (is_const, expr, ty) = loop {
            let anc = ancestors.next()?;
            break match_ast! {
                match anc {
                    ast::ClosureExpr(closure) => (false, closure.body(), infer_expr_opt(closure.body())),
                    ast::BlockExpr(block_expr) => {
                        let (constness, block) = match block_expr.modifier() {
                            Some(ast::BlockModifier::Const(_)) => (true, block_expr),
                            Some(ast::BlockModifier::Try(_)) => (false, block_expr),
                            Some(ast::BlockModifier::Label(label)) if label.lifetime().is_some() => (false, block_expr),
                            _ => continue,
                        };
                        let expr = Some(ast::Expr::BlockExpr(block));
                        (constness, expr.clone(), infer_expr_opt(expr))
                    },
                    ast::Fn(fn_) => {
                        let func = sema.to_def(&fn_)?;
                        let mut ret_ty = func.ret_type(sema.db);
                        if func.is_async(sema.db) {
                            if let Some(async_ret) = func.async_ret_type(sema.db) {
                                ret_ty = async_ret;
                            }
                        }
                        (fn_.const_token().is_some(), fn_.body().map(ast::Expr::BlockExpr), Some(ret_ty))
                    },
                    ast::Static(statik) => {
                        (true, statik.body(), Some(sema.to_def(&statik)?.ty(sema.db)))
                    },
                    ast::ConstArg(ca) => {
                        (true, ca.expr(), infer_expr_opt(ca.expr()))
                    },
                    ast::Const(konst) => {
                        (true, konst.body(), Some(sema.to_def(&konst)?.ty(sema.db)))
                    },
                    ast::ConstParam(cp) => {
                        (true, cp.default_val()?.expr(), Some(sema.to_def(&cp)?.ty(sema.db)))
                    },
                    ast::ConstBlockPat(cbp) => {
                        let expr = cbp.block_expr().map(ast::Expr::BlockExpr);
                        (true, expr.clone(), infer_expr_opt(expr))
                    },
                    ast::Variant(__) => return None,
                    ast::Meta(__) => return None,
                    ast::LoopExpr(it) => {
                        set_parent_loop(&it);
                        continue;
                    },
                    ast::ForExpr(it) => {
                        set_parent_loop(&it);
                        continue;
                    },
                    ast::WhileExpr(it) => {
                        set_parent_loop(&it);
                        continue;
                    },
                    _ => continue,
                }
            };
        };

        let expr = expr?;
        let contains_tail_expr = if let Some(body_tail) = self.tail_expr() {
            let mut contains_tail_expr = false;
            let tail_expr_range = body_tail.syntax().text_range();
            for_each_tail_expr(&expr, &mut |e| {
                if tail_expr_range.contains_range(e.syntax().text_range()) {
                    contains_tail_expr = true;
                }
            });
            contains_tail_expr
        } else {
            false
        };

        let parent = self.parent()?;
        let parents = generic_parents(&parent);
        let generic_param_lists = parents.iter().filter_map(|it| it.generic_param_list()).collect();
        let where_clauses = parents.iter().filter_map(|it| it.where_clause()).collect();

        Some((
            ContainerInfo {
                is_const,
                parent_loop,
                ret_type: ty,
                generic_param_lists,
                where_clauses,
                edition,
            },
            contains_tail_expr,
        ))
    }

    fn return_ty<'db>(&self, ctx: &AssistContext<'db>) -> Option<RetType<'db>> {
        match self.tail_expr() {
            Some(expr) => ctx.sema.type_of_expr(&expr).map(TypeInfo::original).map(RetType::Expr),
            None => Some(RetType::Stmt),
        }
    }

    /// Local variables defined inside `body` that are accessed outside of it
    fn ret_values<'a>(
        &self,
        ctx: &'a AssistContext<'_>,
        parent: &SyntaxNode,
    ) -> impl Iterator<Item = OutlivedLocal> + 'a {
        let parent = parent.clone();
        let range = self.text_range();
        locals_defined_in_body(&ctx.sema, self)
            .into_iter()
            .filter_map(move |local| local_outlives_body(ctx, range, local, &parent))
    }

    /// Analyses the function body for external control flow.
    fn external_control_flow<'db>(
        &self,
        ctx: &AssistContext<'db>,
        container_info: &ContainerInfo<'db>,
    ) -> Option<ControlFlow<'db>> {
        let mut ret_expr = None;
        let mut try_expr = None;
        let mut break_expr = None;
        let mut continue_expr = None;
        let mut is_async = false;
        let mut _is_unsafe = false;

        let mut unsafe_depth = 0;
        let mut loop_depth = 0;

        self.preorder_expr(&mut |expr| {
            let expr = match expr {
                WalkEvent::Enter(e) => e,
                WalkEvent::Leave(expr) => {
                    match expr {
                        ast::Expr::LoopExpr(_)
                        | ast::Expr::ForExpr(_)
                        | ast::Expr::WhileExpr(_) => loop_depth -= 1,
                        ast::Expr::BlockExpr(block_expr) if block_expr.unsafe_token().is_some() => {
                            unsafe_depth -= 1
                        }
                        _ => (),
                    }
                    return false;
                }
            };
            match expr {
                ast::Expr::LoopExpr(_) | ast::Expr::ForExpr(_) | ast::Expr::WhileExpr(_) => {
                    loop_depth += 1;
                }
                ast::Expr::BlockExpr(block_expr) if block_expr.unsafe_token().is_some() => {
                    unsafe_depth += 1
                }
                ast::Expr::ReturnExpr(it) => {
                    ret_expr = Some(it);
                }
                ast::Expr::TryExpr(it) => {
                    try_expr = Some(it);
                }
                ast::Expr::BreakExpr(it) if loop_depth == 0 => {
                    break_expr = Some(it);
                }
                ast::Expr::ContinueExpr(it) if loop_depth == 0 => {
                    continue_expr = Some(it);
                }
                ast::Expr::AwaitExpr(_) => is_async = true,
                // FIXME: Do unsafe analysis on expression, sem highlighting knows this so we should be able
                // to just lift that out of there
                // expr if unsafe_depth ==0 && expr.is_unsafe => is_unsafe = true,
                _ => {}
            }
            false
        });

        let kind = match (try_expr, ret_expr, break_expr, continue_expr) {
            (Some(_), _, None, None) => {
                let ret_ty = container_info.ret_type.clone()?;
                let kind = TryKind::of_ty(ret_ty, ctx, container_info.edition)?;

                Some(FlowKind::Try { kind })
            }
            (Some(_), _, _, _) => {
                cov_mark::hit!(external_control_flow_try_and_bc);
                return None;
            }
            (None, Some(r), None, None) => Some(FlowKind::Return(r.expr())),
            (None, Some(_), _, _) => {
                cov_mark::hit!(external_control_flow_return_and_bc);
                return None;
            }
            (None, None, Some(_), Some(_)) => {
                cov_mark::hit!(external_control_flow_break_and_continue);
                return None;
            }
            (None, None, Some(b), None) => Some(FlowKind::Break(b.lifetime(), b.expr())),
            (None, None, None, Some(c)) => Some(FlowKind::Continue(c.lifetime())),
            (None, None, None, None) => None,
        };

        Some(ControlFlow { kind, is_async, is_unsafe: _is_unsafe })
    }

    /// find variables that should be extracted as params
    ///
    /// Computes additional info that affects param type and mutability
    fn extracted_function_params<'db>(
        &self,
        ctx: &AssistContext<'db>,
        container_info: &ContainerInfo<'db>,
        locals: FxIndexSet<Local>,
    ) -> Vec<Param<'db>> {
        locals
            .into_iter()
            .sorted()
            .map(|local| (local, local.primary_source(ctx.db())))
            .filter(|(_, src)| is_defined_outside_of_body(ctx, self, src))
            .filter_map(|(local, src)| match src.into_ident_pat() {
                Some(src) => Some((local, src)),
                None => {
                    stdx::never!(false, "Local::is_self returned false, but source is SelfParam");
                    None
                }
            })
            .map(|(var, src)| {
                let usages = LocalUsages::find_local_usages(ctx, var);
                let ty = var.ty(ctx.db());

                let defined_outside_parent_loop = container_info
                    .parent_loop
                    .as_ref()
                    .is_none_or(|it| it.text_range().contains_range(src.syntax().text_range()));

                let is_copy = ty.is_copy(ctx.db());
                let has_usages = self.has_usages_after_body(&usages);
                let requires_mut =
                    !ty.is_mutable_reference() && has_exclusive_usages(ctx, &usages, self);
                // We can move the value into the function call if it's not used after the call,
                // if the var is not used but defined outside a loop we are extracting from we can't move it either
                // as the function will reuse it in the next iteration.
                let move_local = (!has_usages && defined_outside_parent_loop) || ty.is_reference();
                Param { var, ty, move_local, requires_mut, is_copy }
            })
            .collect()
    }

    fn has_usages_after_body(&self, usages: &LocalUsages) -> bool {
        usages.iter().any(|reference| self.precedes_range(reference.range))
    }
}

enum GenericParent {
    Fn(ast::Fn),
    Impl(ast::Impl),
    Trait(ast::Trait),
}

impl GenericParent {
    fn generic_param_list(&self) -> Option<ast::GenericParamList> {
        match self {
            GenericParent::Fn(fn_) => fn_.generic_param_list(),
            GenericParent::Impl(impl_) => impl_.generic_param_list(),
            GenericParent::Trait(trait_) => trait_.generic_param_list(),
        }
    }

    fn where_clause(&self) -> Option<ast::WhereClause> {
        match self {
            GenericParent::Fn(fn_) => fn_.where_clause(),
            GenericParent::Impl(impl_) => impl_.where_clause(),
            GenericParent::Trait(trait_) => trait_.where_clause(),
        }
    }
}

/// Search `parent`'s ancestors for items with potentially applicable generic parameters
fn generic_parents(parent: &SyntaxNode) -> Vec<GenericParent> {
    let mut list = Vec::new();
    if let Some(parent_item) = parent.ancestors().find_map(ast::Item::cast) {
        if let ast::Item::Fn(ref fn_) = parent_item {
            if let Some(parent_parent) =
                parent_item.syntax().parent().and_then(|it| it.parent()).and_then(ast::Item::cast)
            {
                match parent_parent {
                    ast::Item::Impl(impl_) => list.push(GenericParent::Impl(impl_)),
                    ast::Item::Trait(trait_) => list.push(GenericParent::Trait(trait_)),
                    _ => (),
                }
            }
            list.push(GenericParent::Fn(fn_.clone()));
        }
    }
    list
}

/// checks if relevant var is used with `&mut` access inside body
fn has_exclusive_usages(
    ctx: &AssistContext<'_>,
    usages: &LocalUsages,
    body: &FunctionBody,
) -> bool {
    usages
        .iter()
        .filter(|reference| body.contains_range(reference.range))
        .any(|reference| reference_is_exclusive(reference, body, ctx))
}

/// checks if this reference requires `&mut` access inside node
fn reference_is_exclusive(
    reference: &FileReference,
    node: &dyn HasTokenAtOffset,
    ctx: &AssistContext<'_>,
) -> bool {
    // FIXME: this quite an incorrect way to go about doing this :-)
    // `FileReference` is an IDE-type --- it encapsulates data communicated to the human,
    // but doesn't necessary fully reflect all the intricacies of the underlying language semantics
    // The correct approach here would be to expose this entire analysis as a method on some hir
    // type. Something like `body.free_variables(statement_range)`.

    // we directly modify variable with set: `n = 0`, `n += 1`
    if reference.category.contains(ReferenceCategory::WRITE) {
        return true;
    }

    // we take `&mut` reference to variable: `&mut v`
    let path = match path_element_of_reference(node, reference) {
        Some(path) => path,
        None => return false,
    };

    expr_require_exclusive_access(ctx, &path).unwrap_or(false)
}

/// checks if this expr requires `&mut` access, recurses on field access
fn expr_require_exclusive_access(ctx: &AssistContext<'_>, expr: &ast::Expr) -> Option<bool> {
    if let ast::Expr::MacroExpr(_) = expr {
        // FIXME: expand macro and check output for mutable usages of the variable?
        return None;
    }

    let parent = expr.syntax().parent()?;

    if let Some(bin_expr) = ast::BinExpr::cast(parent.clone()) {
        if matches!(bin_expr.op_kind()?, ast::BinaryOp::Assignment { .. }) {
            return Some(bin_expr.lhs()?.syntax() == expr.syntax());
        }
        return Some(false);
    }

    if let Some(ref_expr) = ast::RefExpr::cast(parent.clone()) {
        return Some(ref_expr.mut_token().is_some());
    }

    if let Some(method_call) = ast::MethodCallExpr::cast(parent.clone()) {
        let func = ctx.sema.resolve_method_call(&method_call)?;
        let self_param = func.self_param(ctx.db())?;
        let access = self_param.access(ctx.db());

        return Some(matches!(access, hir::Access::Exclusive));
    }

    if let Some(field) = ast::FieldExpr::cast(parent) {
        return expr_require_exclusive_access(ctx, &field.into());
    }

    Some(false)
}

trait HasTokenAtOffset {
    fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken>;
}

impl HasTokenAtOffset for SyntaxNode {
    fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken> {
        SyntaxNode::token_at_offset(self, offset)
    }
}

impl HasTokenAtOffset for FunctionBody {
    fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken> {
        match self {
            FunctionBody::Expr(expr) => expr.syntax().token_at_offset(offset),
            FunctionBody::Span { parent, text_range, .. } => {
                match parent.syntax().token_at_offset(offset) {
                    TokenAtOffset::None => TokenAtOffset::None,
                    TokenAtOffset::Single(t) => {
                        if text_range.contains_range(t.text_range()) {
                            TokenAtOffset::Single(t)
                        } else {
                            TokenAtOffset::None
                        }
                    }
                    TokenAtOffset::Between(a, b) => {
                        match (
                            text_range.contains_range(a.text_range()),
                            text_range.contains_range(b.text_range()),
                        ) {
                            (true, true) => TokenAtOffset::Between(a, b),
                            (true, false) => TokenAtOffset::Single(a),
                            (false, true) => TokenAtOffset::Single(b),
                            (false, false) => TokenAtOffset::None,
                        }
                    }
                }
            }
        }
    }
}

/// find relevant `ast::Expr` for reference
///
/// # Preconditions
///
/// `node` must cover `reference`, that is `node.text_range().contains_range(reference.range)`
fn path_element_of_reference(
    node: &dyn HasTokenAtOffset,
    reference: &FileReference,
) -> Option<ast::Expr> {
    let token = node.token_at_offset(reference.range.start()).right_biased().or_else(|| {
        stdx::never!(false, "cannot find token at variable usage: {:?}", reference);
        None
    })?;
    let path = token.parent_ancestors().find_map(ast::Expr::cast).or_else(|| {
        stdx::never!(false, "cannot find path parent of variable usage: {:?}", token);
        None
    })?;
    stdx::always!(
        matches!(path, ast::Expr::PathExpr(_) | ast::Expr::MacroExpr(_)),
        "unexpected expression type for variable usage: {:?}",
        path
    );
    Some(path)
}

/// list local variables defined inside `body`
fn locals_defined_in_body(
    sema: &Semantics<'_, RootDatabase>,
    body: &FunctionBody,
) -> FxIndexSet<Local> {
    // FIXME: this doesn't work well with macros
    //        see https://github.com/rust-lang/rust-analyzer/pull/7535#discussion_r570048550
    let mut res = FxIndexSet::default();
    body.walk_pat(&mut |pat| {
        if let ast::Pat::IdentPat(pat) = pat {
            if let Some(local) = sema.to_def(&pat) {
                res.insert(local);
            }
        }
    });
    res
}

/// Returns usage details if local variable is used after(outside of) body
fn local_outlives_body(
    ctx: &AssistContext<'_>,
    body_range: TextRange,
    local: Local,
    parent: &SyntaxNode,
) -> Option<OutlivedLocal> {
    let usages = LocalUsages::find_local_usages(ctx, local);
    let mut has_mut_usages = false;
    let mut any_outlives = false;
    for usage in usages.iter() {
        if body_range.end() <= usage.range.start() {
            has_mut_usages |= reference_is_exclusive(usage, parent, ctx);
            any_outlives |= true;
            if has_mut_usages {
                break; // no need to check more elements we have all the info we wanted
            }
        }
    }
    if !any_outlives {
        return None;
    }
    Some(OutlivedLocal { local, mut_usage_outside_body: has_mut_usages })
}

/// checks if the relevant local was defined before(outside of) body
fn is_defined_outside_of_body(
    ctx: &AssistContext<'_>,
    body: &FunctionBody,
    src: &LocalSource,
) -> bool {
    src.original_file(ctx.db()) == ctx.file_id() && !body.contains_node(src.syntax())
}

/// find where to put extracted function definition
///
/// Function should be put right after returned node
fn node_to_insert_after(body: &FunctionBody, anchor: Anchor) -> Option<SyntaxNode> {
    let node = body.node();
    let mut ancestors = node.ancestors().peekable();
    let mut last_ancestor = None;
    while let Some(next_ancestor) = ancestors.next() {
        match next_ancestor.kind() {
            SyntaxKind::SOURCE_FILE => break,
            SyntaxKind::IMPL => {
                if body.extracted_from_trait_impl() && matches!(anchor, Anchor::Method) {
                    let impl_node = find_non_trait_impl(&next_ancestor);
                    if let target_node @ Some(_) = impl_node.as_ref().and_then(last_impl_member) {
                        return target_node;
                    }
                }
            }
            SyntaxKind::ITEM_LIST if !matches!(anchor, Anchor::Freestanding) => continue,
            SyntaxKind::ITEM_LIST => {
                if ancestors.peek().map(SyntaxNode::kind) == Some(SyntaxKind::MODULE) {
                    break;
                }
            }
            SyntaxKind::ASSOC_ITEM_LIST if !matches!(anchor, Anchor::Method) => continue,
            SyntaxKind::ASSOC_ITEM_LIST if body.extracted_from_trait_impl() => continue,
            SyntaxKind::ASSOC_ITEM_LIST => {
                if ancestors.peek().map(SyntaxNode::kind) == Some(SyntaxKind::IMPL) {
                    break;
                }
            }
            _ => (),
        }
        last_ancestor = Some(next_ancestor);
    }
    last_ancestor
}

fn find_non_trait_impl(trait_impl: &SyntaxNode) -> Option<ast::Impl> {
    let as_impl = ast::Impl::cast(trait_impl.clone())?;
    let impl_type = Some(impl_type_name(&as_impl)?);

    let siblings = trait_impl.parent()?.children();
    siblings
        .filter_map(ast::Impl::cast)
        .find(|s| impl_type_name(s) == impl_type && !is_trait_impl(s))
}

fn last_impl_member(impl_node: &ast::Impl) -> Option<SyntaxNode> {
    let last_child = impl_node.assoc_item_list()?.assoc_items().last()?;
    Some(last_child.syntax().clone())
}

fn is_trait_impl(node: &ast::Impl) -> bool {
    node.trait_().is_some()
}

fn impl_type_name(impl_node: &ast::Impl) -> Option<String> {
    Some(impl_node.self_ty()?.to_string())
}

/// Fixes up the call site before the target expressions are replaced with the call expression
fn fixup_call_site(builder: &mut SourceChangeBuilder, body: &FunctionBody) {
    let parent_match_arm = body.parent().and_then(ast::MatchArm::cast);

    if let Some(parent_match_arm) = parent_match_arm {
        if parent_match_arm.comma_token().is_none() {
            let parent_match_arm = builder.make_mut(parent_match_arm);
            ted::append_child_raw(parent_match_arm.syntax(), make::token(T![,]));
        }
    }
}

fn make_call(ctx: &AssistContext<'_>, fun: &Function<'_>, indent: IndentLevel) -> SyntaxNode {
    let ret_ty = fun.return_type(ctx);

    let args = make::arg_list(fun.params.iter().map(|param| param.to_arg(ctx, fun.mods.edition)));
    let name = fun.name.clone();
    let mut call_expr = if fun.self_param.is_some() {
        let self_arg = make::expr_path(make::ext::ident_path("self"));
        make::expr_method_call(self_arg, name, args).into()
    } else {
        let func = make::expr_path(make::path_unqualified(make::path_segment(name)));
        make::expr_call(func, args).into()
    };

    let handler = FlowHandler::from_ret_ty(fun, &ret_ty);

    if fun.control_flow.is_async {
        call_expr = make::expr_await(call_expr);
    }

    let expr = handler.make_call_expr(call_expr).clone_for_update();
    expr.indent(indent);

    let outliving_bindings = match fun.outliving_locals.as_slice() {
        [] => None,
        [var] => {
            let name = var.local.name(ctx.db());
            let name = make::name(&name.display(ctx.db(), fun.mods.edition).to_string());
            Some(ast::Pat::IdentPat(make::ident_pat(false, var.mut_usage_outside_body, name)))
        }
        vars => {
            let binding_pats = vars.iter().map(|var| {
                let name = var.local.name(ctx.db());
                let name = make::name(&name.display(ctx.db(), fun.mods.edition).to_string());
                make::ident_pat(false, var.mut_usage_outside_body, name).into()
            });
            Some(ast::Pat::TuplePat(make::tuple_pat(binding_pats)))
        }
    };

    let parent_match_arm = fun.body.parent().and_then(ast::MatchArm::cast);

    if let Some(bindings) = outliving_bindings {
        // with bindings that outlive it
        make::let_stmt(bindings, None, Some(expr)).syntax().clone_for_update()
    } else if parent_match_arm.as_ref().is_some() {
        // as a tail expr for a match arm
        expr.syntax().clone()
    } else if parent_match_arm.as_ref().is_none()
        && fun.ret_ty.is_unit()
        && (!fun.outliving_locals.is_empty() || !expr.is_block_like())
    {
        // as an expr stmt
        make::expr_stmt(expr).syntax().clone_for_update()
    } else {
        // as a tail expr, or a block
        expr.syntax().clone()
    }
}

enum FlowHandler<'db> {
    None,
    Try { kind: TryKind<'db> },
    If { action: FlowKind<'db> },
    IfOption { action: FlowKind<'db> },
    MatchOption { none: FlowKind<'db> },
    MatchResult { err: FlowKind<'db> },
}

impl<'db> FlowHandler<'db> {
    fn from_ret_ty(fun: &Function<'db>, ret_ty: &FunType<'db>) -> FlowHandler<'db> {
        if fun.contains_tail_expr {
            return FlowHandler::None;
        }
        let Some(action) = fun.control_flow.kind.clone() else {
            return FlowHandler::None;
        };

        if let FunType::Unit = ret_ty {
            match action {
                FlowKind::Return(None) | FlowKind::Break(_, None) | FlowKind::Continue(_) => {
                    FlowHandler::If { action }
                }
                FlowKind::Return(_) | FlowKind::Break(_, _) => FlowHandler::IfOption { action },
                FlowKind::Try { kind } => FlowHandler::Try { kind },
            }
        } else {
            match action {
                FlowKind::Return(None) | FlowKind::Break(_, None) | FlowKind::Continue(_) => {
                    FlowHandler::MatchOption { none: action }
                }
                FlowKind::Return(_) | FlowKind::Break(_, _) => {
                    FlowHandler::MatchResult { err: action }
                }
                FlowKind::Try { kind } => FlowHandler::Try { kind },
            }
        }
    }

    fn make_call_expr(&self, call_expr: ast::Expr) -> ast::Expr {
        match self {
            FlowHandler::None => call_expr,
            FlowHandler::Try { kind: _ } => make::expr_try(call_expr),
            FlowHandler::If { action } => {
                let action = action.make_result_handler(None);
                let stmt = make::expr_stmt(action);
                let block = make::block_expr(iter::once(stmt.into()), None);
                let controlflow_break_path = make::path_from_text("ControlFlow::Break");
                let condition = make::expr_let(
                    make::tuple_struct_pat(
                        controlflow_break_path,
                        iter::once(make::wildcard_pat().into()),
                    )
                    .into(),
                    call_expr,
                );
                make::expr_if(condition.into(), block, None).into()
            }
            FlowHandler::IfOption { action } => {
                let path = make::ext::ident_path("Some");
                let value_pat = make::ext::simple_ident_pat(make::name("value"));
                let pattern = make::tuple_struct_pat(path, iter::once(value_pat.into()));
                let cond = make::expr_let(pattern.into(), call_expr);
                let value = make::expr_path(make::ext::ident_path("value"));
                let action_expr = action.make_result_handler(Some(value));
                let action_stmt = make::expr_stmt(action_expr);
                let then = make::block_expr(iter::once(action_stmt.into()), None);
                make::expr_if(cond.into(), then, None).into()
            }
            FlowHandler::MatchOption { none } => {
                let some_name = "value";

                let some_arm = {
                    let path = make::ext::ident_path("Some");
                    let value_pat = make::ext::simple_ident_pat(make::name(some_name));
                    let pat = make::tuple_struct_pat(path, iter::once(value_pat.into()));
                    let value = make::expr_path(make::ext::ident_path(some_name));
                    make::match_arm(pat.into(), None, value)
                };
                let none_arm = {
                    let path = make::ext::ident_path("None");
                    let pat = make::path_pat(path);
                    make::match_arm(pat, None, none.make_result_handler(None))
                };
                let arms = make::match_arm_list(vec![some_arm, none_arm]);
                make::expr_match(call_expr, arms).into()
            }
            FlowHandler::MatchResult { err } => {
                let ok_name = "value";
                let err_name = "value";

                let ok_arm = {
                    let path = make::ext::ident_path("Ok");
                    let value_pat = make::ext::simple_ident_pat(make::name(ok_name));
                    let pat = make::tuple_struct_pat(path, iter::once(value_pat.into()));
                    let value = make::expr_path(make::ext::ident_path(ok_name));
                    make::match_arm(pat.into(), None, value)
                };
                let err_arm = {
                    let path = make::ext::ident_path("Err");
                    let value_pat = make::ext::simple_ident_pat(make::name(err_name));
                    let pat = make::tuple_struct_pat(path, iter::once(value_pat.into()));
                    let value = make::expr_path(make::ext::ident_path(err_name));
                    make::match_arm(pat.into(), None, err.make_result_handler(Some(value)))
                };
                let arms = make::match_arm_list(vec![ok_arm, err_arm]);
                make::expr_match(call_expr, arms).into()
            }
        }
    }
}

fn path_expr_from_local(ctx: &AssistContext<'_>, var: Local, edition: Edition) -> ast::Expr {
    let name = var.name(ctx.db()).display(ctx.db(), edition).to_string();
    make::expr_path(make::ext::ident_path(&name))
}

fn format_function(
    ctx: &AssistContext<'_>,
    module: hir::Module,
    fun: &Function<'_>,
    old_indent: IndentLevel,
) -> ast::Fn {
    let fun_name = make::name(&fun.name.text());
    let params = fun.make_param_list(ctx, module, fun.mods.edition);
    let ret_ty = fun.make_ret_ty(ctx, module);
    let body = make_body(ctx, old_indent, fun);
    let (generic_params, where_clause) = make_generic_params_and_where_clause(ctx, fun);

    make::fn_(
        None,
        fun_name,
        generic_params,
        where_clause,
        params,
        body,
        ret_ty,
        fun.control_flow.is_async,
        fun.mods.is_const,
        fun.control_flow.is_unsafe,
        false,
    )
}

fn make_generic_params_and_where_clause(
    ctx: &AssistContext<'_>,
    fun: &Function<'_>,
) -> (Option<ast::GenericParamList>, Option<ast::WhereClause>) {
    let used_type_params = fun.type_params(ctx);

    let generic_param_list = make_generic_param_list(ctx, fun, &used_type_params);
    let where_clause = make_where_clause(ctx, fun, &used_type_params);

    (generic_param_list, where_clause)
}

fn make_generic_param_list(
    ctx: &AssistContext<'_>,
    fun: &Function<'_>,
    used_type_params: &[TypeParam],
) -> Option<ast::GenericParamList> {
    let mut generic_params = fun
        .mods
        .generic_param_lists
        .iter()
        .flat_map(|parent_params| {
            parent_params
                .generic_params()
                .filter(|param| param_is_required(ctx, param, used_type_params))
        })
        .peekable();

    if generic_params.peek().is_some() {
        Some(make::generic_param_list(generic_params))
    } else {
        None
    }
}

fn param_is_required(
    ctx: &AssistContext<'_>,
    param: &ast::GenericParam,
    used_type_params: &[TypeParam],
) -> bool {
    match param {
        ast::GenericParam::ConstParam(_) | ast::GenericParam::LifetimeParam(_) => false,
        ast::GenericParam::TypeParam(type_param) => match &ctx.sema.to_def(type_param) {
            Some(def) => used_type_params.contains(def),
            _ => false,
        },
    }
}

fn make_where_clause(
    ctx: &AssistContext<'_>,
    fun: &Function<'_>,
    used_type_params: &[TypeParam],
) -> Option<ast::WhereClause> {
    let mut predicates = fun
        .mods
        .where_clauses
        .iter()
        .flat_map(|parent_where_clause| {
            parent_where_clause
                .predicates()
                .filter(|pred| pred_is_required(ctx, pred, used_type_params))
        })
        .peekable();

    if predicates.peek().is_some() { Some(make::where_clause(predicates)) } else { None }
}

fn pred_is_required(
    ctx: &AssistContext<'_>,
    pred: &ast::WherePred,
    used_type_params: &[TypeParam],
) -> bool {
    match resolved_type_param(ctx, pred) {
        Some(it) => used_type_params.contains(&it),
        None => false,
    }
}

fn resolved_type_param(ctx: &AssistContext<'_>, pred: &ast::WherePred) -> Option<TypeParam> {
    let path = match pred.ty()? {
        ast::Type::PathType(path_type) => path_type.path(),
        _ => None,
    }?;

    match ctx.sema.resolve_path(&path)? {
        PathResolution::TypeParam(type_param) => Some(type_param),
        _ => None,
    }
}

impl<'db> Function<'db> {
    /// Collect all the `TypeParam`s used in the `body` and `params`.
    fn type_params(&self, ctx: &AssistContext<'db>) -> Vec<TypeParam> {
        let type_params_in_descendant_paths =
            self.body.descendant_paths().filter_map(|it| match ctx.sema.resolve_path(&it) {
                Some(PathResolution::TypeParam(type_param)) => Some(type_param),
                _ => None,
            });
        let type_params_in_params = self.params.iter().filter_map(|p| p.ty.as_type_param(ctx.db()));
        type_params_in_descendant_paths.chain(type_params_in_params).collect()
    }

    fn make_param_list(
        &self,
        ctx: &AssistContext<'_>,
        module: hir::Module,
        edition: Edition,
    ) -> ast::ParamList {
        let self_param = self.self_param.clone();
        let params = self.params.iter().map(|param| param.to_param(ctx, module, edition));
        make::param_list(self_param, params)
    }

    fn make_ret_ty(&self, ctx: &AssistContext<'_>, module: hir::Module) -> Option<ast::RetType> {
        let fun_ty = self.return_type(ctx);
        let handler = FlowHandler::from_ret_ty(self, &fun_ty);
        let ret_ty = match &handler {
            FlowHandler::None => {
                if matches!(fun_ty, FunType::Unit) {
                    return None;
                }
                fun_ty.make_ty(ctx, module)
            }
            FlowHandler::Try { kind: TryKind::Option } => {
                make::ext::ty_option(fun_ty.make_ty(ctx, module))
            }
            FlowHandler::Try { kind: TryKind::Result { ty: parent_ret_ty } } => {
                let handler_ty = parent_ret_ty
                    .type_arguments()
                    .nth(1)
                    .map(|ty| make_ty(&ty, ctx, module))
                    .unwrap_or_else(make::ty_placeholder);
                make::ext::ty_result(fun_ty.make_ty(ctx, module), handler_ty)
            }
            FlowHandler::If { .. } => make::ty("ControlFlow<()>"),
            FlowHandler::IfOption { action } => {
                let handler_ty = action
                    .expr_ty(ctx)
                    .map(|ty| make_ty(&ty, ctx, module))
                    .unwrap_or_else(make::ty_placeholder);
                make::ext::ty_option(handler_ty)
            }
            FlowHandler::MatchOption { .. } => make::ext::ty_option(fun_ty.make_ty(ctx, module)),
            FlowHandler::MatchResult { err } => {
                let handler_ty = err
                    .expr_ty(ctx)
                    .map(|ty| make_ty(&ty, ctx, module))
                    .unwrap_or_else(make::ty_placeholder);
                make::ext::ty_result(fun_ty.make_ty(ctx, module), handler_ty)
            }
        };
        Some(make::ret_type(ret_ty))
    }
}

impl<'db> FunType<'db> {
    fn make_ty(&self, ctx: &AssistContext<'db>, module: hir::Module) -> ast::Type {
        match self {
            FunType::Unit => make::ty_unit(),
            FunType::Single(ty) => make_ty(ty, ctx, module),
            FunType::Tuple(types) => match types.as_slice() {
                [] => {
                    stdx::never!("tuple type with 0 elements");
                    make::ty_unit()
                }
                [ty] => {
                    stdx::never!("tuple type with 1 element");
                    make_ty(ty, ctx, module)
                }
                types => {
                    let types = types.iter().map(|ty| make_ty(ty, ctx, module));
                    make::ty_tuple(types)
                }
            },
        }
    }
}

fn make_body(
    ctx: &AssistContext<'_>,
    old_indent: IndentLevel,
    fun: &Function<'_>,
) -> ast::BlockExpr {
    let ret_ty = fun.return_type(ctx);
    let handler = FlowHandler::from_ret_ty(fun, &ret_ty);

    let block = match &fun.body {
        FunctionBody::Expr(expr) => {
            let expr = rewrite_body_segment(ctx, &fun.params, &handler, expr.syntax());
            let expr = ast::Expr::cast(expr).expect("Body segment should be an expr");
            match expr {
                ast::Expr::BlockExpr(block) => {
                    // If the extracted expression is itself a block, there is no need to wrap it inside another block.
                    block.dedent(old_indent);
                    let elements = block.stmt_list().map_or_else(
                        || Either::Left(iter::empty()),
                        |stmt_list| {
                            let elements = stmt_list.syntax().children_with_tokens().filter_map(
                                |node_or_token| match &node_or_token {
                                    syntax::NodeOrToken::Node(node) => {
                                        ast::Stmt::cast(node.clone()).map(|_| node_or_token)
                                    }
                                    syntax::NodeOrToken::Token(token) => {
                                        ast::Comment::cast(token.clone()).map(|_| node_or_token)
                                    }
                                },
                            );
                            Either::Right(elements)
                        },
                    );
                    make::hacky_block_expr(elements, block.tail_expr())
                }
                _ => {
                    expr.reindent_to(1.into());

                    make::block_expr(Vec::new(), Some(expr))
                }
            }
        }
        FunctionBody::Span { parent, text_range, .. } => {
            let mut elements: Vec<_> = parent
                .syntax()
                .children_with_tokens()
                .filter(|it| text_range.contains_range(it.text_range()))
                .map(|it| match &it {
                    syntax::NodeOrToken::Node(n) => syntax::NodeOrToken::Node(
                        rewrite_body_segment(ctx, &fun.params, &handler, n),
                    ),
                    _ => it,
                })
                .collect();

            let mut tail_expr = match &elements.last() {
                Some(syntax::NodeOrToken::Node(node)) if ast::Expr::can_cast(node.kind()) => {
                    ast::Expr::cast(node.clone())
                }
                _ => None,
            };

            match tail_expr {
                Some(_) => {
                    elements.pop();
                }
                None => match fun.outliving_locals.as_slice() {
                    [] => {}
                    [var] => {
                        tail_expr = Some(path_expr_from_local(ctx, var.local, fun.mods.edition));
                    }
                    vars => {
                        let exprs = vars
                            .iter()
                            .map(|var| path_expr_from_local(ctx, var.local, fun.mods.edition));
                        let expr = make::expr_tuple(exprs);
                        tail_expr = Some(expr.into());
                    }
                },
            };

            let body_indent = IndentLevel(1);
            let elements = elements
                .into_iter()
                .map(|node_or_token| match &node_or_token {
                    syntax::NodeOrToken::Node(node) => match ast::Stmt::cast(node.clone()) {
                        Some(stmt) => {
                            stmt.reindent_to(body_indent);
                            let ast_node = stmt.syntax().clone_subtree();
                            syntax::NodeOrToken::Node(ast_node)
                        }
                        _ => node_or_token,
                    },
                    _ => node_or_token,
                })
                .collect::<Vec<SyntaxElement>>();
            if let Some(tail_expr) = &mut tail_expr {
                tail_expr.reindent_to(body_indent);
            }

            make::hacky_block_expr(elements, tail_expr)
        }
    };

    match &handler {
        FlowHandler::None => block,
        FlowHandler::Try { kind } => {
            let block = with_default_tail_expr(block, make::ext::expr_unit());
            map_tail_expr(block, |tail_expr| {
                let constructor = match kind {
                    TryKind::Option => "Some",
                    TryKind::Result { .. } => "Ok",
                };
                let func = make::expr_path(make::ext::ident_path(constructor));
                let args = make::arg_list(iter::once(tail_expr));
                make::expr_call(func, args).into()
            })
        }
        FlowHandler::If { .. } => {
            let controlflow_continue = make::expr_call(
                make::expr_path(make::path_from_text("ControlFlow::Continue")),
                make::arg_list([make::ext::expr_unit()]),
            )
            .into();
            with_tail_expr(block, controlflow_continue)
        }
        FlowHandler::IfOption { .. } => {
            let none = make::expr_path(make::ext::ident_path("None"));
            with_tail_expr(block, none)
        }
        FlowHandler::MatchOption { .. } => map_tail_expr(block, |tail_expr| {
            let some = make::expr_path(make::ext::ident_path("Some"));
            let args = make::arg_list(iter::once(tail_expr));
            make::expr_call(some, args).into()
        }),
        FlowHandler::MatchResult { .. } => map_tail_expr(block, |tail_expr| {
            let ok = make::expr_path(make::ext::ident_path("Ok"));
            let args = make::arg_list(iter::once(tail_expr));
            make::expr_call(ok, args).into()
        }),
    }
}

fn map_tail_expr(block: ast::BlockExpr, f: impl FnOnce(ast::Expr) -> ast::Expr) -> ast::BlockExpr {
    let tail_expr = match block.tail_expr() {
        Some(tail_expr) => tail_expr,
        None => return block,
    };
    make::block_expr(block.statements(), Some(f(tail_expr)))
}

fn with_default_tail_expr(block: ast::BlockExpr, tail_expr: ast::Expr) -> ast::BlockExpr {
    match block.tail_expr() {
        Some(_) => block,
        None => make::block_expr(block.statements(), Some(tail_expr)),
    }
}

fn with_tail_expr(block: ast::BlockExpr, tail_expr: ast::Expr) -> ast::BlockExpr {
    let stmt_tail_opt: Option<ast::Stmt> =
        block.tail_expr().map(|expr| make::expr_stmt(expr).into());

    let mut elements: Vec<SyntaxElement> = vec![];

    block.statements().for_each(|stmt| {
        elements.push(syntax::NodeOrToken::Node(stmt.syntax().clone()));
    });

    if let Some(stmt_list) = block.stmt_list() {
        stmt_list.syntax().children_with_tokens().for_each(|node_or_token| {
            if let syntax::NodeOrToken::Token(_) = &node_or_token {
                elements.push(node_or_token)
            };
        });
    }

    if let Some(stmt_tail) = stmt_tail_opt {
        elements.push(syntax::NodeOrToken::Node(stmt_tail.syntax().clone()));
    }

    make::hacky_block_expr(elements, Some(tail_expr))
}

fn format_type(ty: &hir::Type<'_>, ctx: &AssistContext<'_>, module: hir::Module) -> String {
    ty.display_source_code(ctx.db(), module.into(), true).ok().unwrap_or_else(|| "_".to_owned())
}

fn make_ty(ty: &hir::Type<'_>, ctx: &AssistContext<'_>, module: hir::Module) -> ast::Type {
    let ty_str = format_type(ty, ctx, module);
    make::ty(&ty_str)
}

fn rewrite_body_segment(
    ctx: &AssistContext<'_>,
    params: &[Param<'_>],
    handler: &FlowHandler<'_>,
    syntax: &SyntaxNode,
) -> SyntaxNode {
    let syntax = fix_param_usages(ctx, params, syntax);
    update_external_control_flow(handler, &syntax);
    syntax
}

/// change all usages to account for added `&`/`&mut` for some params
fn fix_param_usages(
    ctx: &AssistContext<'_>,
    params: &[Param<'_>],
    syntax: &SyntaxNode,
) -> SyntaxNode {
    let mut usages_for_param: Vec<(&Param<'_>, Vec<ast::Expr>)> = Vec::new();

    let tm = TreeMutator::new(syntax);

    for param in params {
        if !param.kind().is_ref() {
            continue;
        }

        let usages = LocalUsages::find_local_usages(ctx, param.var);
        let usages = usages
            .iter()
            .filter(|reference| syntax.text_range().contains_range(reference.range))
            .filter_map(|reference| path_element_of_reference(syntax, reference))
            .map(|expr| tm.make_mut(&expr));

        usages_for_param.push((param, usages.collect()));
    }

    let res = tm.make_syntax_mut(syntax);

    for (param, usages) in usages_for_param {
        for usage in usages {
            match usage.syntax().ancestors().skip(1).find_map(ast::Expr::cast) {
                Some(ast::Expr::MethodCallExpr(_) | ast::Expr::FieldExpr(_)) => {
                    // do nothing
                }
                Some(ast::Expr::RefExpr(node))
                    if param.kind() == ParamKind::MutRef && node.mut_token().is_some() =>
                {
                    ted::replace(
                        node.syntax(),
                        node.expr().expect("RefExpr::expr() cannot be None").syntax(),
                    );
                }
                Some(ast::Expr::RefExpr(node))
                    if param.kind() == ParamKind::SharedRef && node.mut_token().is_none() =>
                {
                    ted::replace(
                        node.syntax(),
                        node.expr().expect("RefExpr::expr() cannot be None").syntax(),
                    );
                }
                Some(_) | None => {
                    let p = &make::expr_prefix(T![*], usage.clone()).clone_for_update();
                    ted::replace(usage.syntax(), p.syntax())
                }
            }
        }
    }

    res
}

fn update_external_control_flow(handler: &FlowHandler<'_>, syntax: &SyntaxNode) {
    let mut nested_loop = None;
    let mut nested_scope = None;
    for event in syntax.preorder() {
        match event {
            WalkEvent::Enter(e) => match e.kind() {
                SyntaxKind::LOOP_EXPR | SyntaxKind::WHILE_EXPR | SyntaxKind::FOR_EXPR => {
                    if nested_loop.is_none() {
                        nested_loop = Some(e.clone());
                    }
                }
                SyntaxKind::FN
                | SyntaxKind::CONST
                | SyntaxKind::STATIC
                | SyntaxKind::IMPL
                | SyntaxKind::MODULE => {
                    if nested_scope.is_none() {
                        nested_scope = Some(e.clone());
                    }
                }
                _ => {}
            },
            WalkEvent::Leave(e) => {
                if nested_scope.is_none() {
                    if let Some(expr) = ast::Expr::cast(e.clone()) {
                        match expr {
                            ast::Expr::ReturnExpr(return_expr) => {
                                let expr = return_expr.expr();
                                if let Some(replacement) = make_rewritten_flow(handler, expr) {
                                    ted::replace(return_expr.syntax(), replacement.syntax())
                                }
                            }
                            ast::Expr::BreakExpr(break_expr) if nested_loop.is_none() => {
                                let expr = break_expr.expr();
                                if let Some(replacement) = make_rewritten_flow(handler, expr) {
                                    ted::replace(break_expr.syntax(), replacement.syntax())
                                }
                            }
                            ast::Expr::ContinueExpr(continue_expr) if nested_loop.is_none() => {
                                if let Some(replacement) = make_rewritten_flow(handler, None) {
                                    ted::replace(continue_expr.syntax(), replacement.syntax())
                                }
                            }
                            _ => {
                                // do nothing
                            }
                        }
                    }
                }

                if nested_loop.as_ref() == Some(&e) {
                    nested_loop = None;
                }
                if nested_scope.as_ref() == Some(&e) {
                    nested_scope = None;
                }
            }
        };
    }
}

fn make_rewritten_flow(
    handler: &FlowHandler<'_>,
    arg_expr: Option<ast::Expr>,
) -> Option<ast::Expr> {
    let value = match handler {
        FlowHandler::None | FlowHandler::Try { .. } => return None,
        FlowHandler::If { .. } => make::expr_call(
            make::expr_path(make::path_from_text("ControlFlow::Break")),
            make::arg_list([make::ext::expr_unit()]),
        )
        .into(),
        FlowHandler::IfOption { .. } => {
            let expr = arg_expr.unwrap_or_else(make::ext::expr_unit);
            let args = make::arg_list([expr]);
            make::expr_call(make::expr_path(make::ext::ident_path("Some")), args).into()
        }
        FlowHandler::MatchOption { .. } => make::expr_path(make::ext::ident_path("None")),
        FlowHandler::MatchResult { .. } => {
            let expr = arg_expr.unwrap_or_else(make::ext::expr_unit);
            let args = make::arg_list([expr]);
            make::expr_call(make::expr_path(make::ext::ident_path("Err")), args).into()
        }
    };
    Some(make::expr_return(Some(value)).clone_for_update())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn no_args_from_binary_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    foo($01 + 1$0);
}
"#,
            r#"
fn foo() {
    foo(fun_name());
}

fn $0fun_name() -> i32 {
    1 + 1
}
"#,
        );
    }

    #[test]
    fn no_args_from_binary_expr_in_module() {
        check_assist(
            extract_function,
            r#"
mod bar {
    fn foo() {
        foo($01 + 1$0);
    }
}
"#,
            r#"
mod bar {
    fn foo() {
        foo(fun_name());
    }

    fn $0fun_name() -> i32 {
        1 + 1
    }
}
"#,
        );
    }

    #[test]
    fn no_args_from_binary_expr_indented() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0{ 1 + 1 }$0;
}
"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() -> i32 {
    1 + 1
}
"#,
        );
    }

    #[test]
    fn no_args_from_stmt_with_last_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i32 {
    let k = 1;
    $0let m = 1;
    m + 1$0
}
"#,
            r#"
fn foo() -> i32 {
    let k = 1;
    fun_name()
}

fn $0fun_name() -> i32 {
    let m = 1;
    m + 1
}
"#,
        );
    }

    #[test]
    fn no_args_from_stmt_unit() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let k = 3;
    $0let m = 1;
    let n = m + 1;$0
    let g = 5;
}
"#,
            r#"
fn foo() {
    let k = 3;
    fun_name();
    let g = 5;
}

fn $0fun_name() {
    let m = 1;
    let n = m + 1;
}
"#,
        );
    }

    #[test]
    fn no_args_if() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0if true { }$0
}
"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() {
    if true { }
}
"#,
        );
    }

    #[test]
    fn no_args_if_else() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i32 {
    $0if true { 1 } else { 2 }$0
}
"#,
            r#"
fn foo() -> i32 {
    fun_name()
}

fn $0fun_name() -> i32 {
    if true { 1 } else { 2 }
}
"#,
        );
    }

    #[test]
    fn no_args_if_let_else() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i32 {
    $0if let true = false { 1 } else { 2 }$0
}
"#,
            r#"
fn foo() -> i32 {
    fun_name()
}

fn $0fun_name() -> i32 {
    if let true = false { 1 } else { 2 }
}
"#,
        );
    }

    #[test]
    fn no_args_match() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i32 {
    $0match true {
        true => 1,
        false => 2,
    }$0
}
"#,
            r#"
fn foo() -> i32 {
    fun_name()
}

fn $0fun_name() -> i32 {
    match true {
        true => 1,
        false => 2,
    }
}
"#,
        );
    }

    #[test]
    fn no_args_while() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0while true { }$0
}
"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() {
    while true { }
}
"#,
        );
    }

    #[test]
    fn no_args_for() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0for v in &[0, 1] { }$0
}
"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() {
    for v in &[0, 1] { }
}
"#,
        );
    }

    #[test]
    fn no_args_from_loop_unit() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0loop {
        let m = 1;
    }$0
}
"#,
            r#"
fn foo() {
    fun_name()
}

fn $0fun_name() -> ! {
    loop {
        let m = 1;
    }
}
"#,
        );
    }

    #[test]
    fn no_args_from_loop_with_return() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let v = $0loop {
        let m = 1;
        break m;
    }$0;
}
"#,
            r#"
fn foo() {
    let v = fun_name();
}

fn $0fun_name() -> i32 {
    loop {
        let m = 1;
        break m;
    }
}
"#,
        );
    }

    #[test]
    fn no_args_from_match() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let v: i32 = $0match Some(1) {
        Some(x) => x,
        None => 0,
    }$0;
}
"#,
            r#"
fn foo() {
    let v: i32 = fun_name();
}

fn $0fun_name() -> i32 {
    match Some(1) {
        Some(x) => x,
        None => 0,
    }
}
"#,
        );
    }

    #[test]
    fn extract_partial_block_single_line() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let n = 1;
    let mut v = $0n * n;$0
    v += 1;
}
"#,
            r#"
fn foo() {
    let n = 1;
    let mut v = fun_name(n);
    v += 1;
}

fn $0fun_name(n: i32) -> i32 {
    let mut v = n * n;
    v
}
"#,
        );
    }

    #[test]
    fn extract_partial_block() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let m = 2;
    let n = 1;
    let mut v = m $0* n;
    let mut w = 3;$0
    v += 1;
    w += 1;
}
"#,
            r#"
fn foo() {
    let m = 2;
    let n = 1;
    let (mut v, mut w) = fun_name(m, n);
    v += 1;
    w += 1;
}

fn $0fun_name(m: i32, n: i32) -> (i32, i32) {
    let mut v = m * n;
    let mut w = 3;
    (v, w)
}
"#,
        );
    }

    #[test]
    fn argument_form_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() -> u32 {
    let n = 2;
    $0n+2$0
}
"#,
            r#"
fn foo() -> u32 {
    let n = 2;
    fun_name(n)
}

fn $0fun_name(n: u32) -> u32 {
    n+2
}
"#,
        )
    }

    #[test]
    fn argument_used_twice_form_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() -> u32 {
    let n = 2;
    $0n+n$0
}
"#,
            r#"
fn foo() -> u32 {
    let n = 2;
    fun_name(n)
}

fn $0fun_name(n: u32) -> u32 {
    n+n
}
"#,
        )
    }

    #[test]
    fn two_arguments_form_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() -> u32 {
    let n = 2;
    let m = 3;
    $0n+n*m$0
}
"#,
            r#"
fn foo() -> u32 {
    let n = 2;
    let m = 3;
    fun_name(n, m)
}

fn $0fun_name(n: u32, m: u32) -> u32 {
    n+n*m
}
"#,
        )
    }

    #[test]
    fn argument_and_locals() {
        check_assist(
            extract_function,
            r#"
fn foo() -> u32 {
    let n = 2;
    $0let m = 1;
    n + m$0
}
"#,
            r#"
fn foo() -> u32 {
    let n = 2;
    fun_name(n)
}

fn $0fun_name(n: u32) -> u32 {
    let m = 1;
    n + m
}
"#,
        )
    }

    #[test]
    fn in_comment_is_not_applicable() {
        cov_mark::check!(extract_function_in_comment_is_not_applicable);
        check_assist_not_applicable(extract_function, r"fn main() { 1 + /* $0comment$0 */ 1; }");
    }

    #[test]
    fn empty_selection_is_not_applicable() {
        cov_mark::check!(extract_function_empty_selection_is_not_applicable);
        check_assist_not_applicable(
            extract_function,
            r#"
fn main() {
    $0

    $0
}"#,
        );
    }

    #[test]
    fn part_of_expr_stmt() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $01$0 + 1;
}
"#,
            r#"
fn foo() {
    fun_name() + 1;
}

fn $0fun_name() -> i32 {
    1
}
"#,
        );
    }

    #[test]
    fn function_expr() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0bar(1 + 1)$0
}
"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() {
    bar(1 + 1)
}
"#,
        )
    }

    #[test]
    fn extract_from_nested() {
        check_assist(
            extract_function,
            r#"
fn main() {
    let x = true;
    let tuple = match x {
        true => ($02 + 2$0, true)
        _ => (0, false)
    };
}
"#,
            r#"
fn main() {
    let x = true;
    let tuple = match x {
        true => (fun_name(), true)
        _ => (0, false)
    };
}

fn $0fun_name() -> i32 {
    2 + 2
}
"#,
        );
    }

    #[test]
    fn param_from_closure() {
        check_assist(
            extract_function,
            r#"
fn main() {
    let lambda = |x: u32| $0x * 2$0;
}
"#,
            r#"
fn main() {
    let lambda = |x: u32| fun_name(x);
}

fn $0fun_name(x: u32) -> u32 {
    x * 2
}
"#,
        );
    }

    #[test]
    fn extract_return_stmt() {
        check_assist(
            extract_function,
            r#"
fn foo() -> u32 {
    $0return 2 + 2$0;
}
"#,
            r#"
fn foo() -> u32 {
    return fun_name();
}

fn $0fun_name() -> u32 {
    2 + 2
}
"#,
        );
    }

    #[test]
    fn does_not_add_extra_whitespace() {
        check_assist(
            extract_function,
            r#"
fn foo() -> u32 {


    $0return 2 + 2$0;
}
"#,
            r#"
fn foo() -> u32 {


    return fun_name();
}

fn $0fun_name() -> u32 {
    2 + 2
}
"#,
        );
    }

    #[test]
    fn break_stmt() {
        check_assist(
            extract_function,
            r#"
fn main() {
    let result = loop {
        $0break 2 + 2$0;
    };
}
"#,
            r#"
fn main() {
    let result = loop {
        break fun_name();
    };
}

fn $0fun_name() -> i32 {
    2 + 2
}
"#,
        );
    }

    #[test]
    fn extract_cast() {
        check_assist(
            extract_function,
            r#"
fn main() {
    let v = $00f32 as u32$0;
}
"#,
            r#"
fn main() {
    let v = fun_name();
}

fn $0fun_name() -> u32 {
    0f32 as u32
}
"#,
        );
    }

    #[test]
    fn return_not_applicable() {
        check_assist_not_applicable(extract_function, r"fn foo() { $0return$0; } ");
    }

    #[test]
    fn method_to_freestanding() {
        check_assist(
            extract_function,
            r#"
struct S;

impl S {
    fn foo(&self) -> i32 {
        $01+1$0
    }
}
"#,
            r#"
struct S;

impl S {
    fn foo(&self) -> i32 {
        fun_name()
    }
}

fn $0fun_name() -> i32 {
    1+1
}
"#,
        );
    }

    #[test]
    fn method_with_reference() {
        check_assist(
            extract_function,
            r#"
struct S { f: i32 };

impl S {
    fn foo(&self) -> i32 {
        $0self.f+self.f$0
    }
}
"#,
            r#"
struct S { f: i32 };

impl S {
    fn foo(&self) -> i32 {
        self.fun_name()
    }

    fn $0fun_name(&self) -> i32 {
        self.f+self.f
    }
}
"#,
        );
    }

    #[test]
    fn method_with_mut() {
        check_assist(
            extract_function,
            r#"
struct S { f: i32 };

impl S {
    fn foo(&mut self) {
        $0self.f += 1;$0
    }
}
"#,
            r#"
struct S { f: i32 };

impl S {
    fn foo(&mut self) {
        self.fun_name();
    }

    fn $0fun_name(&mut self) {
        self.f += 1;
    }
}
"#,
        );
    }

    #[test]
    fn variable_defined_inside_and_used_after_no_ret() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let n = 1;
    $0let k = n * n;$0
    let m = k + 1;
}
"#,
            r#"
fn foo() {
    let n = 1;
    let k = fun_name(n);
    let m = k + 1;
}

fn $0fun_name(n: i32) -> i32 {
    let k = n * n;
    k
}
"#,
        );
    }

    #[test]
    fn variable_defined_inside_and_used_after_mutably_no_ret() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let n = 1;
    $0let mut k = n * n;$0
    k += 1;
}
"#,
            r#"
fn foo() {
    let n = 1;
    let mut k = fun_name(n);
    k += 1;
}

fn $0fun_name(n: i32) -> i32 {
    let mut k = n * n;
    k
}
"#,
        );
    }

    #[test]
    fn two_variables_defined_inside_and_used_after_no_ret() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let n = 1;
    $0let k = n * n;
    let m = k + 2;$0
    let h = k + m;
}
"#,
            r#"
fn foo() {
    let n = 1;
    let (k, m) = fun_name(n);
    let h = k + m;
}

fn $0fun_name(n: i32) -> (i32, i32) {
    let k = n * n;
    let m = k + 2;
    (k, m)
}
"#,
        );
    }

    #[test]
    fn multi_variables_defined_inside_and_used_after_mutably_no_ret() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let n = 1;
    $0let mut k = n * n;
    let mut m = k + 2;
    let mut o = m + 3;
    o += 1;$0
    k += o;
    m = 1;
}
"#,
            r#"
fn foo() {
    let n = 1;
    let (mut k, mut m, o) = fun_name(n);
    k += o;
    m = 1;
}

fn $0fun_name(n: i32) -> (i32, i32, i32) {
    let mut k = n * n;
    let mut m = k + 2;
    let mut o = m + 3;
    o += 1;
    (k, m, o)
}
"#,
        );
    }

    #[test]
    fn nontrivial_patterns_define_variables() {
        check_assist(
            extract_function,
            r#"
struct Counter(i32);
fn foo() {
    $0let Counter(n) = Counter(0);$0
    let m = n;
}
"#,
            r#"
struct Counter(i32);
fn foo() {
    let n = fun_name();
    let m = n;
}

fn $0fun_name() -> i32 {
    let Counter(n) = Counter(0);
    n
}
"#,
        );
    }

    #[test]
    fn struct_with_two_fields_pattern_define_variables() {
        check_assist(
            extract_function,
            r#"
struct Counter { n: i32, m: i32 };
fn foo() {
    $0let Counter { n, m: k } = Counter { n: 1, m: 2 };$0
    let h = n + k;
}
"#,
            r#"
struct Counter { n: i32, m: i32 };
fn foo() {
    let (n, k) = fun_name();
    let h = n + k;
}

fn $0fun_name() -> (i32, i32) {
    let Counter { n, m: k } = Counter { n: 1, m: 2 };
    (n, k)
}
"#,
        );
    }

    #[test]
    fn mut_var_from_outer_scope() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let mut n = 1;
    $0n += 1;$0
    let m = n + 1;
}
"#,
            r#"
fn foo() {
    let mut n = 1;
    fun_name(&mut n);
    let m = n + 1;
}

fn $0fun_name(n: &mut i32) {
    *n += 1;
}
"#,
        );
    }

    #[test]
    fn mut_field_from_outer_scope() {
        check_assist(
            extract_function,
            r#"
struct C { n: i32 }
fn foo() {
    let mut c = C { n: 0 };
    $0c.n += 1;$0
    let m = c.n + 1;
}
"#,
            r#"
struct C { n: i32 }
fn foo() {
    let mut c = C { n: 0 };
    fun_name(&mut c);
    let m = c.n + 1;
}

fn $0fun_name(c: &mut C) {
    c.n += 1;
}
"#,
        );
    }

    #[test]
    fn mut_nested_field_from_outer_scope() {
        check_assist(
            extract_function,
            r#"
struct P { n: i32}
struct C { p: P }
fn foo() {
    let mut c = C { p: P { n: 0 } };
    let mut v = C { p: P { n: 0 } };
    let u = C { p: P { n: 0 } };
    $0c.p.n += u.p.n;
    let r = &mut v.p.n;$0
    let m = c.p.n + v.p.n + u.p.n;
}
"#,
            r#"
struct P { n: i32}
struct C { p: P }
fn foo() {
    let mut c = C { p: P { n: 0 } };
    let mut v = C { p: P { n: 0 } };
    let u = C { p: P { n: 0 } };
    fun_name(&mut c, &mut v, &u);
    let m = c.p.n + v.p.n + u.p.n;
}

fn $0fun_name(c: &mut C, v: &mut C, u: &C) {
    c.p.n += u.p.n;
    let r = &mut v.p.n;
}
"#,
        );
    }

    #[test]
    fn mut_param_many_usages_stmt() {
        check_assist(
            extract_function,
            r#"
fn bar(k: i32) {}
trait I: Copy {
    fn succ(&self) -> Self;
    fn inc(&mut self) -> Self { let v = self.succ(); *self = v; v }
}
impl I for i32 {
    fn succ(&self) -> Self { *self + 1 }
}
fn foo() {
    let mut n = 1;
    $0n += n;
    bar(n);
    bar(n+1);
    bar(n*n);
    bar(&n);
    n.inc();
    let v = &mut n;
    *v = v.succ();
    n.succ();$0
    let m = n + 1;
}
"#,
            r#"
fn bar(k: i32) {}
trait I: Copy {
    fn succ(&self) -> Self;
    fn inc(&mut self) -> Self { let v = self.succ(); *self = v; v }
}
impl I for i32 {
    fn succ(&self) -> Self { *self + 1 }
}
fn foo() {
    let mut n = 1;
    fun_name(&mut n);
    let m = n + 1;
}

fn $0fun_name(n: &mut i32) {
    *n += *n;
    bar(*n);
    bar(*n+1);
    bar(*n**n);
    bar(&*n);
    n.inc();
    let v = n;
    *v = v.succ();
    n.succ();
}
"#,
        );
    }

    #[test]
    fn mut_param_many_usages_expr() {
        check_assist(
            extract_function,
            r#"
fn bar(k: i32) {}
trait I: Copy {
    fn succ(&self) -> Self;
    fn inc(&mut self) -> Self { let v = self.succ(); *self = v; v }
}
impl I for i32 {
    fn succ(&self) -> Self { *self + 1 }
}
fn foo() {
    let mut n = 1;
    $0{
        n += n;
        bar(n);
        bar(n+1);
        bar(n*n);
        bar(&n);
        n.inc();
        let v = &mut n;
        *v = v.succ();
        n.succ();
    }$0
    let m = n + 1;
}
"#,
            r#"
fn bar(k: i32) {}
trait I: Copy {
    fn succ(&self) -> Self;
    fn inc(&mut self) -> Self { let v = self.succ(); *self = v; v }
}
impl I for i32 {
    fn succ(&self) -> Self { *self + 1 }
}
fn foo() {
    let mut n = 1;
    fun_name(&mut n);
    let m = n + 1;
}

fn $0fun_name(n: &mut i32) {
    *n += *n;
    bar(*n);
    bar(*n+1);
    bar(*n**n);
    bar(&*n);
    n.inc();
    let v = n;
    *v = v.succ();
    n.succ();
}
"#,
        );
    }

    #[test]
    fn mut_param_by_value() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let mut n = 1;
    $0n += 1;$0
}
"#,
            r"
fn foo() {
    let mut n = 1;
    fun_name(n);
}

fn $0fun_name(mut n: i32) {
    n += 1;
}
",
        );
    }

    #[test]
    fn mut_param_because_of_mut_ref() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let mut n = 1;
    $0let v = &mut n;
    *v += 1;$0
    let k = n;
}
"#,
            r#"
fn foo() {
    let mut n = 1;
    fun_name(&mut n);
    let k = n;
}

fn $0fun_name(n: &mut i32) {
    let v = n;
    *v += 1;
}
"#,
        );
    }

    #[test]
    fn mut_param_by_value_because_of_mut_ref() {
        check_assist(
            extract_function,
            r"
fn foo() {
    let mut n = 1;
    $0let v = &mut n;
    *v += 1;$0
}
",
            r#"
fn foo() {
    let mut n = 1;
    fun_name(n);
}

fn $0fun_name(mut n: i32) {
    let v = &mut n;
    *v += 1;
}
"#,
        );
    }

    #[test]
    fn mut_method_call() {
        check_assist(
            extract_function,
            r#"
trait I {
    fn inc(&mut self);
}
impl I for i32 {
    fn inc(&mut self) { *self += 1 }
}
fn foo() {
    let mut n = 1;
    $0n.inc();$0
}
"#,
            r#"
trait I {
    fn inc(&mut self);
}
impl I for i32 {
    fn inc(&mut self) { *self += 1 }
}
fn foo() {
    let mut n = 1;
    fun_name(n);
}

fn $0fun_name(mut n: i32) {
    n.inc();
}
"#,
        );
    }

    #[test]
    fn shared_method_call() {
        check_assist(
            extract_function,
            r#"
trait I {
    fn succ(&self);
}
impl I for i32 {
    fn succ(&self) { *self + 1 }
}
fn foo() {
    let mut n = 1;
    $0n.succ();$0
}
"#,
            r"
trait I {
    fn succ(&self);
}
impl I for i32 {
    fn succ(&self) { *self + 1 }
}
fn foo() {
    let mut n = 1;
    fun_name(n);
}

fn $0fun_name(n: i32) {
    n.succ();
}
",
        );
    }

    #[test]
    fn mut_method_call_with_other_receiver() {
        check_assist(
            extract_function,
            r#"
trait I {
    fn inc(&mut self, n: i32);
}
impl I for i32 {
    fn inc(&mut self, n: i32) { *self += n }
}
fn foo() {
    let mut n = 1;
    $0let mut m = 2;
    m.inc(n);$0
}
"#,
            r"
trait I {
    fn inc(&mut self, n: i32);
}
impl I for i32 {
    fn inc(&mut self, n: i32) { *self += n }
}
fn foo() {
    let mut n = 1;
    fun_name(n);
}

fn $0fun_name(n: i32) {
    let mut m = 2;
    m.inc(n);
}
",
        );
    }

    #[test]
    fn non_copy_without_usages_after() {
        check_assist(
            extract_function,
            r#"
struct Counter(i32);
fn foo() {
    let c = Counter(0);
    $0let n = c.0;$0
}
"#,
            r"
struct Counter(i32);
fn foo() {
    let c = Counter(0);
    fun_name(c);
}

fn $0fun_name(c: Counter) {
    let n = c.0;
}
",
        );
    }

    #[test]
    fn non_copy_used_after() {
        check_assist(
            extract_function,
            r"
struct Counter(i32);
fn foo() {
    let c = Counter(0);
    $0let n = c.0;$0
    let m = c.0;
}
",
            r#"
struct Counter(i32);
fn foo() {
    let c = Counter(0);
    fun_name(&c);
    let m = c.0;
}

fn $0fun_name(c: &Counter) {
    let n = c.0;
}
"#,
        );
    }

    #[test]
    fn copy_used_after() {
        check_assist(
            extract_function,
            r#"
//- minicore: copy
fn foo() {
    let n = 0;
    $0let m = n;$0
    let k = n;
}
"#,
            r#"
fn foo() {
    let n = 0;
    fun_name(n);
    let k = n;
}

fn $0fun_name(n: i32) {
    let m = n;
}
"#,
        )
    }

    #[test]
    fn copy_custom_used_after() {
        check_assist(
            extract_function,
            r#"
//- minicore: copy, derive
#[derive(Clone, Copy)]
struct Counter(i32);
fn foo() {
    let c = Counter(0);
    $0let n = c.0;$0
    let m = c.0;
}
"#,
            r#"
#[derive(Clone, Copy)]
struct Counter(i32);
fn foo() {
    let c = Counter(0);
    fun_name(c);
    let m = c.0;
}

fn $0fun_name(c: Counter) {
    let n = c.0;
}
"#,
        );
    }

    #[test]
    fn indented_stmts() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    if true {
        loop {
            $0let n = 1;
            let m = 2;$0
        }
    }
}
"#,
            r#"
fn foo() {
    if true {
        loop {
            fun_name();
        }
    }
}

fn $0fun_name() {
    let n = 1;
    let m = 2;
}
"#,
        );
    }

    #[test]
    fn indented_stmts_inside_mod() {
        check_assist(
            extract_function,
            r#"
mod bar {
    fn foo() {
        if true {
            loop {
                $0let n = 1;
                let m = 2;$0
            }
        }
    }
}
"#,
            r#"
mod bar {
    fn foo() {
        if true {
            loop {
                fun_name();
            }
        }
    }

    fn $0fun_name() {
        let n = 1;
        let m = 2;
    }
}
"#,
        );
    }

    #[test]
    fn break_loop() {
        check_assist(
            extract_function,
            r#"
//- minicore: option
fn foo() {
    loop {
        let n = 1;
        $0let m = n + 1;
        break;
        let k = 2;$0
        let h = 1 + k;
    }
}
"#,
            r#"
fn foo() {
    loop {
        let n = 1;
        let k = match fun_name(n) {
            Some(value) => value,
            None => break,
        };
        let h = 1 + k;
    }
}

fn $0fun_name(n: i32) -> Option<i32> {
    let m = n + 1;
    return None;
    let k = 2;
    Some(k)
}
"#,
        );
    }

    #[test]
    fn return_to_parent() {
        check_assist(
            extract_function,
            r#"
//- minicore: copy, result
fn foo() -> i64 {
    let n = 1;
    $0let m = n + 1;
    return 1;
    let k = 2;$0
    (n + k) as i64
}
"#,
            r#"
fn foo() -> i64 {
    let n = 1;
    let k = match fun_name(n) {
        Ok(value) => value,
        Err(value) => return value,
    };
    (n + k) as i64
}

fn $0fun_name(n: i32) -> Result<i32, i64> {
    let m = n + 1;
    return Err(1);
    let k = 2;
    Ok(k)
}
"#,
        );
    }

    #[test]
    fn break_and_continue() {
        cov_mark::check!(external_control_flow_break_and_continue);
        check_assist_not_applicable(
            extract_function,
            r#"
fn foo() {
    loop {
        let n = 1;
        $0let m = n + 1;
        break;
        let k = 2;
        continue;
        let k = k + 1;$0
        let r = n + k;
    }
}
"#,
        );
    }

    #[test]
    fn return_and_break() {
        cov_mark::check!(external_control_flow_return_and_bc);
        check_assist_not_applicable(
            extract_function,
            r#"
fn foo() {
    loop {
        let n = 1;
        $0let m = n + 1;
        break;
        let k = 2;
        return;
        let k = k + 1;$0
        let r = n + k;
    }
}
"#,
        );
    }

    #[test]
    fn break_loop_with_if() {
        check_assist(
            extract_function,
            r#"
//- minicore: try
fn foo() {
    loop {
        let mut n = 1;
        $0let m = n + 1;
        break;
        n += m;$0
        let h = 1 + n;
    }
}
"#,
            r#"
use core::ops::ControlFlow;

fn foo() {
    loop {
        let mut n = 1;
        if let ControlFlow::Break(_) = fun_name(&mut n) {
            break;
        }
        let h = 1 + n;
    }
}

fn $0fun_name(n: &mut i32) -> ControlFlow<()> {
    let m = *n + 1;
    return ControlFlow::Break(());
    *n += m;
    ControlFlow::Continue(())
}
"#,
        );
    }

    #[test]
    fn break_loop_nested() {
        check_assist(
            extract_function,
            r#"
//- minicore: try
fn foo() {
    loop {
        let mut n = 1;
        $0let m = n + 1;
        if m == 42 {
            break;
        }$0
        let h = 1;
    }
}
"#,
            r#"
use core::ops::ControlFlow;

fn foo() {
    loop {
        let mut n = 1;
        if let ControlFlow::Break(_) = fun_name(n) {
            break;
        }
        let h = 1;
    }
}

fn $0fun_name(n: i32) -> ControlFlow<()> {
    let m = n + 1;
    if m == 42 {
        return ControlFlow::Break(());
    }
    ControlFlow::Continue(())
}
"#,
        );
    }

    #[test]
    fn break_loop_nested_labeled() {
        check_assist(
            extract_function,
            r#"
//- minicore: try
fn foo() {
    'bar: loop {
        loop {
            $0break 'bar;$0
        }
    }
}
"#,
            r#"
use core::ops::ControlFlow;

fn foo() {
    'bar: loop {
        loop {
            if let ControlFlow::Break(_) = fun_name() {
                break 'bar;
            }
        }
    }
}

fn $0fun_name() -> ControlFlow<()> {
    return ControlFlow::Break(());
    ControlFlow::Continue(())
}
"#,
        );
    }

    #[test]
    fn continue_loop_nested_labeled() {
        check_assist(
            extract_function,
            r#"
//- minicore: try
fn foo() {
    'bar: loop {
        loop {
            $0continue 'bar;$0
        }
    }
}
"#,
            r#"
use core::ops::ControlFlow;

fn foo() {
    'bar: loop {
        loop {
            if let ControlFlow::Break(_) = fun_name() {
                continue 'bar;
            }
        }
    }
}

fn $0fun_name() -> ControlFlow<()> {
    return ControlFlow::Break(());
    ControlFlow::Continue(())
}
"#,
        );
    }

    #[test]
    fn return_from_nested_loop() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    loop {
        let n = 1;$0
        let k = 1;
        loop {
            return;
        }
        let m = k + 1;$0
        let h = 1 + m;
    }
}
"#,
            r#"
fn foo() {
    loop {
        let n = 1;
        let m = match fun_name() {
            Some(value) => value,
            None => return,
        };
        let h = 1 + m;
    }
}

fn $0fun_name() -> Option<i32> {
    let k = 1;
    loop {
        return None;
    }
    let m = k + 1;
    Some(m)
}
"#,
        );
    }

    #[test]
    fn break_from_nested_loop() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    loop {
        let n = 1;
        $0let k = 1;
        loop {
            break;
        }
        let m = k + 1;$0
        let h = 1 + m;
    }
}
"#,
            r#"
fn foo() {
    loop {
        let n = 1;
        let m = fun_name();
        let h = 1 + m;
    }
}

fn $0fun_name() -> i32 {
    let k = 1;
    loop {
        break;
    }
    let m = k + 1;
    m
}
"#,
        );
    }

    #[test]
    fn break_from_nested_and_outer_loops() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    loop {
        let n = 1;
        $0let k = 1;
        loop {
            break;
        }
        if k == 42 {
            break;
        }
        let m = k + 1;$0
        let h = 1 + m;
    }
}
"#,
            r#"
fn foo() {
    loop {
        let n = 1;
        let m = match fun_name() {
            Some(value) => value,
            None => break,
        };
        let h = 1 + m;
    }
}

fn $0fun_name() -> Option<i32> {
    let k = 1;
    loop {
        break;
    }
    if k == 42 {
        return None;
    }
    let m = k + 1;
    Some(m)
}
"#,
        );
    }

    #[test]
    fn return_from_nested_fn() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    loop {
        let n = 1;
        $0let k = 1;
        fn test() {
            return;
        }
        let m = k + 1;$0
        let h = 1 + m;
    }
}
"#,
            r#"
fn foo() {
    loop {
        let n = 1;
        let m = fun_name();
        let h = 1 + m;
    }
}

fn $0fun_name() -> i32 {
    let k = 1;
    fn test() {
        return;
    }
    let m = k + 1;
    m
}
"#,
        );
    }

    #[test]
    fn break_with_value() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i32 {
    loop {
        let n = 1;
        $0let k = 1;
        if k == 42 {
            break 3;
        }
        let m = k + 1;$0
        let h = 1;
    }
}
"#,
            r#"
fn foo() -> i32 {
    loop {
        let n = 1;
        if let Some(value) = fun_name() {
            break value;
        }
        let h = 1;
    }
}

fn $0fun_name() -> Option<i32> {
    let k = 1;
    if k == 42 {
        return Some(3);
    }
    let m = k + 1;
    None
}
"#,
        );
    }

    #[test]
    fn break_with_value_and_label() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i32 {
    'bar: loop {
        let n = 1;
        $0let k = 1;
        if k == 42 {
            break 'bar 4;
        }
        let m = k + 1;$0
        let h = 1;
    }
}
"#,
            r#"
fn foo() -> i32 {
    'bar: loop {
        let n = 1;
        if let Some(value) = fun_name() {
            break 'bar value;
        }
        let h = 1;
    }
}

fn $0fun_name() -> Option<i32> {
    let k = 1;
    if k == 42 {
        return Some(4);
    }
    let m = k + 1;
    None
}
"#,
        );
    }

    #[test]
    fn break_with_value_and_return() {
        check_assist(
            extract_function,
            r#"
fn foo() -> i64 {
    loop {
        let n = 1;$0
        let k = 1;
        if k == 42 {
            break 3;
        }
        let m = k + 1;$0
        let h = 1 + m;
    }
}
"#,
            r#"
fn foo() -> i64 {
    loop {
        let n = 1;
        let m = match fun_name() {
            Ok(value) => value,
            Err(value) => break value,
        };
        let h = 1 + m;
    }
}

fn $0fun_name() -> Result<i32, i64> {
    let k = 1;
    if k == 42 {
        return Err(3);
    }
    let m = k + 1;
    Ok(m)
}
"#,
        );
    }

    #[test]
    fn try_option() {
        check_assist(
            extract_function,
            r#"
//- minicore: option
fn bar() -> Option<i32> { None }
fn foo() -> Option<()> {
    let n = bar()?;
    $0let k = foo()?;
    let m = k + 1;$0
    let h = 1 + m;
    Some(())
}
"#,
            r#"
fn bar() -> Option<i32> { None }
fn foo() -> Option<()> {
    let n = bar()?;
    let m = fun_name()?;
    let h = 1 + m;
    Some(())
}

fn $0fun_name() -> Option<i32> {
    let k = foo()?;
    let m = k + 1;
    Some(m)
}
"#,
        );
    }

    #[test]
    fn try_option_unit() {
        check_assist(
            extract_function,
            r#"
//- minicore: option
fn foo() -> Option<()> {
    let n = 1;
    $0let k = foo()?;
    let m = k + 1;$0
    let h = 1 + n;
    Some(())
}
"#,
            r#"
fn foo() -> Option<()> {
    let n = 1;
    fun_name()?;
    let h = 1 + n;
    Some(())
}

fn $0fun_name() -> Option<()> {
    let k = foo()?;
    let m = k + 1;
    Some(())
}
"#,
        );
    }

    #[test]
    fn try_result() {
        check_assist(
            extract_function,
            r#"
//- minicore: result
fn foo() -> Result<(), i64> {
    let n = 1;
    $0let k = foo()?;
    let m = k + 1;$0
    let h = 1 + m;
    Ok(())
}
"#,
            r#"
fn foo() -> Result<(), i64> {
    let n = 1;
    let m = fun_name()?;
    let h = 1 + m;
    Ok(())
}

fn $0fun_name() -> Result<i32, i64> {
    let k = foo()?;
    let m = k + 1;
    Ok(m)
}
"#,
        );
    }

    #[test]
    fn try_option_with_return() {
        check_assist(
            extract_function,
            r#"
//- minicore: option
fn foo() -> Option<()> {
    let n = 1;
    $0let k = foo()?;
    if k == 42 {
        return None;
    }
    let m = k + 1;$0
    let h = 1 + m;
    Some(())
}
"#,
            r#"
fn foo() -> Option<()> {
    let n = 1;
    let m = fun_name()?;
    let h = 1 + m;
    Some(())
}

fn $0fun_name() -> Option<i32> {
    let k = foo()?;
    if k == 42 {
        return None;
    }
    let m = k + 1;
    Some(m)
}
"#,
        );
    }

    #[test]
    fn try_result_with_return() {
        check_assist(
            extract_function,
            r#"
//- minicore: result
fn foo() -> Result<(), i64> {
    let n = 1;
    $0let k = foo()?;
    if k == 42 {
        return Err(1);
    }
    let m = k + 1;$0
    let h = 1 + m;
    Ok(())
}
"#,
            r#"
fn foo() -> Result<(), i64> {
    let n = 1;
    let m = fun_name()?;
    let h = 1 + m;
    Ok(())
}

fn $0fun_name() -> Result<i32, i64> {
    let k = foo()?;
    if k == 42 {
        return Err(1);
    }
    let m = k + 1;
    Ok(m)
}
"#,
        );
    }

    #[test]
    fn try_and_break() {
        cov_mark::check!(external_control_flow_try_and_bc);
        check_assist_not_applicable(
            extract_function,
            r#"
//- minicore: option
fn foo() -> Option<()> {
    loop {
        let n = Some(1);
        $0let m = n? + 1;
        break;
        let k = 2;
        let k = k + 1;$0
        let r = n + k;
    }
    Some(())
}
"#,
        );
    }

    #[test]
    fn try_and_return_ok() {
        check_assist(
            extract_function,
            r#"
//- minicore: result
fn foo() -> Result<(), i64> {
    let n = 1;
    $0let k = foo()?;
    if k == 42 {
        return Ok(1);
    }
    let m = k + 1;$0
    let h = 1 + m;
    Ok(())
}
"#,
            r#"
fn foo() -> Result<(), i64> {
    let n = 1;
    let m = fun_name()?;
    let h = 1 + m;
    Ok(())
}

fn $0fun_name() -> Result<i32, i64> {
    let k = foo()?;
    if k == 42 {
        return Ok(1);
    }
    let m = k + 1;
    Ok(m)
}
"#,
        );
    }

    #[test]
    fn param_usage_in_macro() {
        check_assist(
            extract_function,
            r#"
macro_rules! m {
    ($val:expr) => { $val };
}

fn foo() {
    let n = 1;
    $0let k = n * m!(n);$0
    let m = k + 1;
}
"#,
            r#"
macro_rules! m {
    ($val:expr) => { $val };
}

fn foo() {
    let n = 1;
    let k = fun_name(n);
    let m = k + 1;
}

fn $0fun_name(n: i32) -> i32 {
    let k = n * m!(n);
    k
}
"#,
        );
    }

    #[test]
    fn param_usage_in_macro_with_nested_tt() {
        check_assist(
            extract_function,
            r#"
macro_rules! m {
    ($val:expr) => { $val };
}

fn foo() {
    let n = 1;
    let t = 1;
    $0let k = n * m!((n) + { t });$0
    let m = k + 1;
}
"#,
            r#"
macro_rules! m {
    ($val:expr) => { $val };
}

fn foo() {
    let n = 1;
    let t = 1;
    let k = fun_name(n, t);
    let m = k + 1;
}

fn $0fun_name(n: i32, t: i32) -> i32 {
    let k = n * m!((n) + { t });
    k
}
"#,
        )
    }

    #[test]
    fn param_usage_in_macro_with_nested_tt_2() {
        check_assist(
            extract_function,
            r#"
macro_rules! m {
    ($val:expr) => { $val };
}

struct S(i32);
impl S {
    fn foo(&self) {
        let n = 1;
        $0let k = n * m!((n) + { self.0 });$0
        let m = k + 1;
    }
}
"#,
            r#"
macro_rules! m {
    ($val:expr) => { $val };
}

struct S(i32);
impl S {
    fn foo(&self) {
        let n = 1;
        let k = self.fun_name(n);
        let m = k + 1;
    }

    fn $0fun_name(&self, n: i32) -> i32 {
        let k = n * m!((n) + { self.0 });
        k
    }
}
"#,
        )
    }

    #[test]
    fn extract_with_await() {
        check_assist(
            extract_function,
            r#"
//- minicore: future
fn main() {
    $0some_function().await;$0
}

async fn some_function() {

}
"#,
            r#"
fn main() {
    fun_name().await;
}

async fn $0fun_name() {
    some_function().await;
}

async fn some_function() {

}
"#,
        );
    }

    #[test]
    fn extract_with_await_and_result_not_producing_match_expr() {
        check_assist(
            extract_function,
            r#"
//- minicore: future, result
async fn foo() -> Result<(), ()> {
    $0async {}.await;
    Err(())?$0
}
"#,
            r#"
async fn foo() -> Result<(), ()> {
    fun_name().await
}

async fn $0fun_name() -> Result<(), ()> {
    async {}.await;
    Err(())?
}
"#,
        );
    }

    #[test]
    fn extract_with_await_and_result_producing_match_expr() {
        check_assist(
            extract_function,
            r#"
//- minicore: future
async fn foo() -> i32 {
    loop {
        let n = 1;$0
        let k = async { 1 }.await;
        if k == 42 {
            break 3;
        }
        let m = k + 1;$0
        let h = 1 + m;
    }
}
"#,
            r#"
async fn foo() -> i32 {
    loop {
        let n = 1;
        let m = match fun_name().await {
            Ok(value) => value,
            Err(value) => break value,
        };
        let h = 1 + m;
    }
}

async fn $0fun_name() -> Result<i32, i32> {
    let k = async { 1 }.await;
    if k == 42 {
        return Err(3);
    }
    let m = k + 1;
    Ok(m)
}
"#,
        );
    }

    #[test]
    fn extract_with_await_in_args() {
        check_assist(
            extract_function,
            r#"
//- minicore: future
fn main() {
    $0function_call("a", some_function().await);$0
}

async fn some_function() {

}
"#,
            r#"
fn main() {
    fun_name().await;
}

async fn $0fun_name() {
    function_call("a", some_function().await);
}

async fn some_function() {

}
"#,
        );
    }

    #[test]
    fn extract_does_not_extract_standalone_blocks() {
        check_assist_not_applicable(
            extract_function,
            r#"
fn main() $0{}$0
"#,
        );
    }

    #[test]
    fn extract_adds_comma_for_match_arm() {
        check_assist(
            extract_function,
            r#"
fn main() {
    match 6 {
        100 => $0{ 100 }$0
        _ => 0,
    };
}
"#,
            r#"
fn main() {
    match 6 {
        100 => fun_name(),
        _ => 0,
    };
}

fn $0fun_name() -> i32 {
    100
}
"#,
        );
        check_assist(
            extract_function,
            r#"
fn main() {
    match 6 {
        100 => $0{ 100 }$0,
        _ => 0,
    };
}
"#,
            r#"
fn main() {
    match 6 {
        100 => fun_name(),
        _ => 0,
    };
}

fn $0fun_name() -> i32 {
    100
}
"#,
        );

        // Makes sure no semicolon is added for unit-valued match arms
        check_assist(
            extract_function,
            r#"
fn main() {
    match () {
        _ => $0()$0,
    }
}
"#,
            r#"
fn main() {
    match () {
        _ => fun_name(),
    }
}

fn $0fun_name() {
    ()
}
"#,
        )
    }

    #[test]
    fn extract_does_not_tear_comments_apart() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    /*$0*/
    foo();
    foo();
    /*$0*/
}
"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() {
    /**/
    foo();
    foo();
    /**/
}
"#,
        );
    }

    #[test]
    fn extract_does_not_tear_body_apart() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    $0foo();
}$0
"#,
            r#"
fn foo() {
    fun_name();
}

fn $0fun_name() {
    foo();
}
"#,
        );
    }

    #[test]
    fn extract_does_not_wrap_res_in_res() {
        check_assist(
            extract_function,
            r#"
//- minicore: result, try
fn foo() -> Result<(), i64> {
    $0Result::<i32, i64>::Ok(0)?;
    Ok(())$0
}
"#,
            r#"
fn foo() -> Result<(), i64> {
    fun_name()
}

fn $0fun_name() -> Result<(), i64> {
    Result::<i32, i64>::Ok(0)?;
    Ok(())
}
"#,
        );
    }

    #[test]
    fn extract_knows_const() {
        check_assist(
            extract_function,
            r#"
const fn foo() {
    $0()$0
}
"#,
            r#"
const fn foo() {
    fun_name();
}

const fn $0fun_name() {
    ()
}
"#,
        );
        check_assist(
            extract_function,
            r#"
const FOO: () = {
    $0()$0
};
"#,
            r#"
const FOO: () = {
    fun_name();
};

const fn $0fun_name() {
    ()
}
"#,
        );
    }

    #[test]
    fn extract_does_not_move_outer_loop_vars() {
        check_assist(
            extract_function,
            r#"
//- minicore: iterator
fn foo() {
    let mut x = 5;
    for _ in 0..10 {
        $0x += 1;$0
    }
}
"#,
            r#"
fn foo() {
    let mut x = 5;
    for _ in 0..10 {
        fun_name(&mut x);
    }
}

fn $0fun_name(x: &mut i32) {
    *x += 1;
}
"#,
        );
        check_assist(
            extract_function,
            r#"
//- minicore: iterator
fn foo() {
    for _ in 0..10 {
        let mut x = 5;
        $0x += 1;$0
    }
}
"#,
            r#"
fn foo() {
    for _ in 0..10 {
        let mut x = 5;
        fun_name(x);
    }
}

fn $0fun_name(mut x: i32) {
    x += 1;
}
"#,
        );
        check_assist(
            extract_function,
            r#"
//- minicore: iterator
fn foo() {
    loop {
        let mut x = 5;
        for _ in 0..10 {
            $0x += 1;$0
        }
    }
}
"#,
            r#"
fn foo() {
    loop {
        let mut x = 5;
        for _ in 0..10 {
            fun_name(&mut x);
        }
    }
}

fn $0fun_name(x: &mut i32) {
    *x += 1;
}
"#,
        );
    }

    // regression test for #9822
    #[test]
    fn extract_mut_ref_param_has_no_mut_binding_in_loop() {
        check_assist(
            extract_function,
            r#"
struct Foo;
impl Foo {
    fn foo(&mut self) {}
}
fn foo() {
    let mut x = Foo;
    while false {
        let y = &mut x;
        $0y.foo();$0
    }
    let z = x;
}
"#,
            r#"
struct Foo;
impl Foo {
    fn foo(&mut self) {}
}
fn foo() {
    let mut x = Foo;
    while false {
        let y = &mut x;
        fun_name(y);
    }
    let z = x;
}

fn $0fun_name(y: &mut Foo) {
    y.foo();
}
"#,
        );
    }

    #[test]
    fn extract_with_macro_arg() {
        check_assist(
            extract_function,
            r#"
macro_rules! m {
    ($val:expr) => { $val };
}
fn main() {
    let bar = "bar";
    $0m!(bar);$0
}
"#,
            r#"
macro_rules! m {
    ($val:expr) => { $val };
}
fn main() {
    let bar = "bar";
    fun_name(bar);
}

fn $0fun_name(bar: &'static str) {
    m!(bar);
}
"#,
        );
    }

    #[test]
    fn unresolvable_types_default_to_placeholder() {
        check_assist(
            extract_function,
            r#"
fn foo() {
    let a = __unresolved;
    let _ = $0{a}$0;
}
"#,
            r#"
fn foo() {
    let a = __unresolved;
    let _ = fun_name(a);
}

fn $0fun_name(a: _) -> _ {
    a
}
"#,
        );
    }

    #[test]
    fn reference_mutable_param_with_further_usages() {
        check_assist(
            extract_function,
            r#"
pub struct Foo {
    field: u32,
}

pub fn testfn(arg: &mut Foo) {
    $0arg.field = 8;$0
    // Simulating access after the extracted portion
    arg.field = 16;
}
"#,
            r#"
pub struct Foo {
    field: u32,
}

pub fn testfn(arg: &mut Foo) {
    fun_name(arg);
    // Simulating access after the extracted portion
    arg.field = 16;
}

fn $0fun_name(arg: &mut Foo) {
    arg.field = 8;
}
"#,
        );
    }

    #[test]
    fn reference_mutable_param_without_further_usages() {
        check_assist(
            extract_function,
            r#"
pub struct Foo {
    field: u32,
}

pub fn testfn(arg: &mut Foo) {
    $0arg.field = 8;$0
}
"#,
            r#"
pub struct Foo {
    field: u32,
}

pub fn testfn(arg: &mut Foo) {
    fun_name(arg);
}

fn $0fun_name(arg: &mut Foo) {
    arg.field = 8;
}
"#,
        );
    }
    #[test]
    fn does_not_import_control_flow() {
        check_assist(
            extract_function,
            r#"
//- minicore: try
fn func() {
    $0let cf = "I'm ControlFlow";$0
}
"#,
            r#"
fn func() {
    fun_name();
}

fn $0fun_name() {
    let cf = "I'm ControlFlow";
}
"#,
        );
    }

    #[test]
    fn extract_function_copies_comment_at_start() {
        check_assist(
            extract_function,
            r#"
fn func() {
    let i = 0;
    $0// comment here!
    let x = 0;$0
}
"#,
            r#"
fn func() {
    let i = 0;
    fun_name();
}

fn $0fun_name() {
    // comment here!
    let x = 0;
}
"#,
        );
    }

    #[test]
    fn extract_function_copies_comment_in_between() {
        check_assist(
            extract_function,
            r#"
fn func() {
    let i = 0;$0
    let a = 0;
    // comment here!
    let x = 0;$0
}
"#,
            r#"
fn func() {
    let i = 0;
    fun_name();
}

fn $0fun_name() {
    let a = 0;
    // comment here!
    let x = 0;
}
"#,
        );
    }

    #[test]
    fn extract_function_copies_comment_at_end() {
        check_assist(
            extract_function,
            r#"
fn func() {
    let i = 0;
    $0let x = 0;
    // comment here!$0
}
"#,
            r#"
fn func() {
    let i = 0;
    fun_name();
}

fn $0fun_name() {
    let x = 0;
    // comment here!
}
"#,
        );
    }

    #[test]
    fn extract_function_copies_comment_indented() {
        check_assist(
            extract_function,
            r#"
fn func() {
    let i = 0;
    $0let x = 0;
    while(true) {
        // comment here!
    }$0
}
"#,
            r#"
fn func() {
    let i = 0;
    fun_name();
}

fn $0fun_name() {
    let x = 0;
    while(true) {
        // comment here!
    }
}
"#,
        );
    }

    #[test]
    fn extract_function_does_preserve_whitespace() {
        check_assist(
            extract_function,
            r#"
fn func() {
    let i = 0;
    $0let a = 0;

    let x = 0;$0
}
"#,
            r#"
fn func() {
    let i = 0;
    fun_name();
}

fn $0fun_name() {
    let a = 0;

    let x = 0;
}
"#,
        );
    }

    #[test]
    fn extract_function_long_form_comment() {
        check_assist(
            extract_function,
            r#"
fn func() {
    let i = 0;
    $0/* a comment */
    let x = 0;$0
}
"#,
            r#"
fn func() {
    let i = 0;
    fun_name();
}

fn $0fun_name() {
    /* a comment */
    let x = 0;
}
"#,
        );
    }

    #[test]
    fn it_should_not_generate_duplicate_function_names() {
        check_assist(
            extract_function,
            r#"
fn fun_name() {
    $0let x = 0;$0
}
"#,
            r#"
fn fun_name() {
    fun_name1();
}

fn $0fun_name1() {
    let x = 0;
}
"#,
        );
    }

    #[test]
    fn should_increment_suffix_until_it_finds_space() {
        check_assist(
            extract_function,
            r#"
fn fun_name1() {
    let y = 0;
}

fn fun_name() {
    $0let x = 0;$0
}
"#,
            r#"
fn fun_name1() {
    let y = 0;
}

fn fun_name() {
    fun_name2();
}

fn $0fun_name2() {
    let x = 0;
}
"#,
        );
    }

    #[test]
    fn extract_method_from_trait_impl() {
        check_assist(
            extract_function,
            r#"
struct Struct(i32);
trait Trait {
    fn bar(&self) -> i32;
}

impl Trait for Struct {
    fn bar(&self) -> i32 {
        $0self.0 + 2$0
    }
}
"#,
            r#"
struct Struct(i32);
trait Trait {
    fn bar(&self) -> i32;
}

impl Trait for Struct {
    fn bar(&self) -> i32 {
        self.fun_name()
    }
}

impl Struct {
    fn $0fun_name(&self) -> i32 {
        self.0 + 2
    }
}
"#,
        );
    }

    #[test]
    fn extract_method_from_trait_with_existing_non_empty_impl_block() {
        check_assist(
            extract_function,
            r#"
struct Struct(i32);
trait Trait {
    fn bar(&self) -> i32;
}

impl Struct {
    fn foo() {}
}

impl Trait for Struct {
    fn bar(&self) -> i32 {
        $0self.0 + 2$0
    }
}
"#,
            r#"
struct Struct(i32);
trait Trait {
    fn bar(&self) -> i32;
}

impl Struct {
    fn foo() {}

    fn $0fun_name(&self) -> i32 {
        self.0 + 2
    }
}

impl Trait for Struct {
    fn bar(&self) -> i32 {
        self.fun_name()
    }
}
"#,
        )
    }

    #[test]
    fn extract_function_from_trait_with_existing_non_empty_impl_block() {
        check_assist(
            extract_function,
            r#"
struct Struct(i32);
trait Trait {
    fn bar(&self) -> i32;
}

impl Struct {
    fn foo() {}
}

impl Trait for Struct {
    fn bar(&self) -> i32 {
        let three_squared = $03 * 3$0;
        self.0 + three_squared
    }
}
"#,
            r#"
struct Struct(i32);
trait Trait {
    fn bar(&self) -> i32;
}

impl Struct {
    fn foo() {}
}

impl Trait for Struct {
    fn bar(&self) -> i32 {
        let three_squared = fun_name();
        self.0 + three_squared
    }
}

fn $0fun_name() -> i32 {
    3 * 3
}
"#,
        )
    }

    #[test]
    fn extract_method_from_trait_with_multiple_existing_impl_blocks() {
        check_assist(
            extract_function,
            r#"
struct Struct(i32);
struct StructBefore(i32);
struct StructAfter(i32);
trait Trait {
    fn bar(&self) -> i32;
}

impl StructBefore {
    fn foo(){}
}

impl Struct {
    fn foo(){}
}

impl StructAfter {
    fn foo(){}
}

impl Trait for Struct {
    fn bar(&self) -> i32 {
        $0self.0 + 2$0
    }
}
"#,
            r#"
struct Struct(i32);
struct StructBefore(i32);
struct StructAfter(i32);
trait Trait {
    fn bar(&self) -> i32;
}

impl StructBefore {
    fn foo(){}
}

impl Struct {
    fn foo(){}

    fn $0fun_name(&self) -> i32 {
        self.0 + 2
    }
}

impl StructAfter {
    fn foo(){}
}

impl Trait for Struct {
    fn bar(&self) -> i32 {
        self.fun_name()
    }
}
"#,
        )
    }

    #[test]
    fn extract_method_from_trait_with_multiple_existing_trait_impl_blocks() {
        check_assist(
            extract_function,
            r#"
struct Struct(i32);
trait Trait {
    fn bar(&self) -> i32;
}
trait TraitBefore {
    fn before(&self) -> i32;
}
trait TraitAfter {
    fn after(&self) -> i32;
}

impl TraitBefore for Struct {
    fn before(&self) -> i32 {
        42
    }
}

impl Struct {
    fn foo(){}
}

impl TraitAfter for Struct {
    fn after(&self) -> i32 {
        42
    }
}

impl Trait for Struct {
    fn bar(&self) -> i32 {
        $0self.0 + 2$0
    }
}
"#,
            r#"
struct Struct(i32);
trait Trait {
    fn bar(&self) -> i32;
}
trait TraitBefore {
    fn before(&self) -> i32;
}
trait TraitAfter {
    fn after(&self) -> i32;
}

impl TraitBefore for Struct {
    fn before(&self) -> i32 {
        42
    }
}

impl Struct {
    fn foo(){}

    fn $0fun_name(&self) -> i32 {
        self.0 + 2
    }
}

impl TraitAfter for Struct {
    fn after(&self) -> i32 {
        42
    }
}

impl Trait for Struct {
    fn bar(&self) -> i32 {
        self.fun_name()
    }
}
"#,
        )
    }

    #[test]
    fn closure_arguments() {
        check_assist(
            extract_function,
            r#"
fn parent(factor: i32) {
    let v = &[1, 2, 3];

    $0v.iter().map(|it| it * factor);$0
}
"#,
            r#"
fn parent(factor: i32) {
    let v = &[1, 2, 3];

    fun_name(factor, v);
}

fn $0fun_name(factor: i32, v: &[i32; 3]) {
    v.iter().map(|it| it * factor);
}
"#,
        );
    }

    #[test]
    fn preserve_generics() {
        check_assist(
            extract_function,
            r#"
fn func<T: Debug>(i: T) {
    $0foo(i);$0
}
"#,
            r#"
fn func<T: Debug>(i: T) {
    fun_name(i);
}

fn $0fun_name<T: Debug>(i: T) {
    foo(i);
}
"#,
        );
    }

    #[test]
    fn dont_emit_type_with_hidden_lifetime_parameter() {
        // FIXME: We should emit a `<T: Debug>` generic argument for the generated function
        check_assist(
            extract_function,
            r#"
struct Struct<'a, T>(&'a T);
fn func<T: Debug>(i: Struct<'_, T>) {
    $0foo(i);$0
}
"#,
            r#"
struct Struct<'a, T>(&'a T);
fn func<T: Debug>(i: Struct<'_, T>) {
    fun_name(i);
}

fn $0fun_name(i: Struct<'_, T>) {
    foo(i);
}
"#,
        );
    }

    #[test]
    fn preserve_generics_from_body() {
        check_assist(
            extract_function,
            r#"
fn func<T: Default>() -> T {
    $0T::default()$0
}
"#,
            r#"
fn func<T: Default>() -> T {
    fun_name()
}

fn $0fun_name<T: Default>() -> T {
    T::default()
}
"#,
        );
    }

    #[test]
    fn filter_unused_generics() {
        check_assist(
            extract_function,
            r#"
fn func<T: Debug, U: Copy>(i: T, u: U) {
    bar(u);
    $0foo(i);$0
}
"#,
            r#"
fn func<T: Debug, U: Copy>(i: T, u: U) {
    bar(u);
    fun_name(i);
}

fn $0fun_name<T: Debug>(i: T) {
    foo(i);
}
"#,
        );
    }

    #[test]
    fn empty_generic_param_list() {
        check_assist(
            extract_function,
            r#"
fn func<T: Debug>(t: T, i: u32) {
    bar(t);
    $0foo(i);$0
}
"#,
            r#"
fn func<T: Debug>(t: T, i: u32) {
    bar(t);
    fun_name(i);
}

fn $0fun_name(i: u32) {
    foo(i);
}
"#,
        );
    }

    #[test]
    fn preserve_where_clause() {
        check_assist(
            extract_function,
            r#"
fn func<T>(i: T) where T: Debug {
    $0foo(i);$0
}
"#,
            r#"
fn func<T>(i: T) where T: Debug {
    fun_name(i);
}

fn $0fun_name<T>(i: T) where T: Debug {
    foo(i);
}
"#,
        );
    }

    #[test]
    fn filter_unused_where_clause() {
        check_assist(
            extract_function,
            r#"
fn func<T, U>(i: T, u: U) where T: Debug, U: Copy {
    bar(u);
    $0foo(i);$0
}
"#,
            r#"
fn func<T, U>(i: T, u: U) where T: Debug, U: Copy {
    bar(u);
    fun_name(i);
}

fn $0fun_name<T>(i: T) where T: Debug {
    foo(i);
}
"#,
        );
    }

    #[test]
    fn nested_generics() {
        check_assist(
            extract_function,
            r#"
struct Struct<T: Into<i32>>(T);
impl <T: Into<i32> + Copy> Struct<T> {
    fn func<V: Into<i32>>(&self, v: V) -> i32 {
        let t = self.0;
        $0t.into() + v.into()$0
    }
}
"#,
            r#"
struct Struct<T: Into<i32>>(T);
impl <T: Into<i32> + Copy> Struct<T> {
    fn func<V: Into<i32>>(&self, v: V) -> i32 {
        let t = self.0;
        fun_name(v, t)
    }
}

fn $0fun_name<T: Into<i32> + Copy, V: Into<i32>>(v: V, t: T) -> i32 {
    t.into() + v.into()
}
"#,
        );
    }

    #[test]
    fn filters_unused_nested_generics() {
        check_assist(
            extract_function,
            r#"
struct Struct<T: Into<i32>, U: Debug>(T, U);
impl <T: Into<i32> + Copy, U: Debug> Struct<T, U> {
    fn func<V: Into<i32>>(&self, v: V) -> i32 {
        let t = self.0;
        $0t.into() + v.into()$0
    }
}
"#,
            r#"
struct Struct<T: Into<i32>, U: Debug>(T, U);
impl <T: Into<i32> + Copy, U: Debug> Struct<T, U> {
    fn func<V: Into<i32>>(&self, v: V) -> i32 {
        let t = self.0;
        fun_name(v, t)
    }
}

fn $0fun_name<T: Into<i32> + Copy, V: Into<i32>>(v: V, t: T) -> i32 {
    t.into() + v.into()
}
"#,
        );
    }

    #[test]
    fn nested_where_clauses() {
        check_assist(
            extract_function,
            r#"
struct Struct<T>(T) where T: Into<i32>;
impl <T> Struct<T> where T: Into<i32> + Copy {
    fn func<V>(&self, v: V) -> i32 where V: Into<i32> {
        let t = self.0;
        $0t.into() + v.into()$0
    }
}
"#,
            r#"
struct Struct<T>(T) where T: Into<i32>;
impl <T> Struct<T> where T: Into<i32> + Copy {
    fn func<V>(&self, v: V) -> i32 where V: Into<i32> {
        let t = self.0;
        fun_name(v, t)
    }
}

fn $0fun_name<T, V>(v: V, t: T) -> i32 where T: Into<i32> + Copy, V: Into<i32> {
    t.into() + v.into()
}
"#,
        );
    }

    #[test]
    fn filters_unused_nested_where_clauses() {
        check_assist(
            extract_function,
            r#"
struct Struct<T, U>(T, U) where T: Into<i32>, U: Debug;
impl <T, U> Struct<T, U> where T: Into<i32> + Copy, U: Debug {
    fn func<V>(&self, v: V) -> i32 where V: Into<i32> {
        let t = self.0;
        $0t.into() + v.into()$0
    }
}
"#,
            r#"
struct Struct<T, U>(T, U) where T: Into<i32>, U: Debug;
impl <T, U> Struct<T, U> where T: Into<i32> + Copy, U: Debug {
    fn func<V>(&self, v: V) -> i32 where V: Into<i32> {
        let t = self.0;
        fun_name(v, t)
    }
}

fn $0fun_name<T, V>(v: V, t: T) -> i32 where T: Into<i32> + Copy, V: Into<i32> {
    t.into() + v.into()
}
"#,
        );
    }

    #[test]
    fn tail_expr_no_extra_control_flow() {
        check_assist(
            extract_function,
            r#"
//- minicore: result
fn fallible() -> Result<(), ()> {
    $0if true {
        return Err(());
    }
    Ok(())$0
}
"#,
            r#"
fn fallible() -> Result<(), ()> {
    fun_name()
}

fn $0fun_name() -> Result<(), ()> {
    if true {
        return Err(());
    }
    Ok(())
}
"#,
        );
    }

    #[test]
    fn non_tail_expr_of_tail_expr_loop() {
        check_assist(
            extract_function,
            r#"
pub fn f() {
    loop {
        $0if true {
            continue;
        }$0

        if false {
            break;
        }
    }
}
"#,
            r#"
pub fn f() {
    loop {
        if let ControlFlow::Break(_) = fun_name() {
            continue;
        }

        if false {
            break;
        }
    }
}

fn $0fun_name() -> ControlFlow<()> {
    if true {
        return ControlFlow::Break(());
    }
    ControlFlow::Continue(())
}
"#,
        );
    }

    #[test]
    fn non_tail_expr_of_tail_if_block() {
        // FIXME: double semicolon
        check_assist(
            extract_function,
            r#"
//- minicore: option, try
fn f() -> Option<()> {
    if true {
        let a = $0if true {
            Some(())?
        } else {
            ()
        }$0;
        Some(a)
    } else {
        None
    }
}
"#,
            r#"
fn f() -> Option<()> {
    if true {
        let a = fun_name()?;;
        Some(a)
    } else {
        None
    }
}

fn $0fun_name() -> Option<()> {
    Some(if true {
        Some(())?
    } else {
        ()
    })
}
"#,
        );
    }

    #[test]
    fn tail_expr_of_tail_block_nested() {
        check_assist(
            extract_function,
            r#"
//- minicore: option, try
fn f() -> Option<()> {
    if true {
        $0{
            let a = if true {
                Some(())?
            } else {
                ()
            };
            Some(a)
        }$0
    } else {
        None
    }
}
"#,
            r#"
fn f() -> Option<()> {
    if true {
        fun_name()
    } else {
        None
    }
}

fn $0fun_name() -> Option<()> {
    let a = if true {
        Some(())?
    } else {
        ()
    };
    Some(a)
}
"#,
        );
    }

    #[test]
    fn non_tail_expr_with_comment_of_tail_expr_loop() {
        check_assist(
            extract_function,
            r#"
pub fn f() {
    loop {
        $0// A comment
        if true {
            continue;
        }$0
        if false {
            break;
        }
    }
}
"#,
            r#"
pub fn f() {
    loop {
        if let ControlFlow::Break(_) = fun_name() {
            continue;
        }
        if false {
            break;
        }
    }
}

fn $0fun_name() -> ControlFlow<()> {
    // A comment
    if true {
        return ControlFlow::Break(());
    }
    ControlFlow::Continue(())
}
"#,
        );
    }

    #[test]
    fn comments_in_block_expr() {
        check_assist(
            extract_function,
            r#"
fn f() {
    let c = $0{
        // comment 1
        let a = 2 + 3;
        // comment 2
        let b = 5;
        a + b
    }$0;
}
"#,
            r#"
fn f() {
    let c = fun_name();
}

fn $0fun_name() -> i32 {
    // comment 1
    let a = 2 + 3;
    // comment 2
    let b = 5;
    a + b
}
"#,
        );
    }

    #[test]
    fn sort_params_in_order() {
        check_assist(
            extract_function,
            r#"
fn existing(a: i32, b: i32, c: i32) {
    let x = 32;

    let p = $0x + b + c + a$0;
}
"#,
            r#"
fn existing(a: i32, b: i32, c: i32) {
    let x = 32;

    let p = fun_name(a, b, c, x);
}

fn $0fun_name(a: i32, b: i32, c: i32, x: i32) -> i32 {
    x + b + c + a
}
"#,
        );
    }

    #[test]
    fn fmt_macro_argument() {
        check_assist(
            extract_function,
            r#"
//- minicore: fmt
fn existing(a: i32, b: i32, c: i32) {
    $0print!("{a}{}{}", b, "{c}");$0
}
"#,
            r#"
fn existing(a: i32, b: i32, c: i32) {
    fun_name(a, b);
}

fn $0fun_name(a: i32, b: i32) {
    print!("{a}{}{}", b, "{c}");
}
"#,
        );
    }

    #[test]
    fn in_left_curly_is_not_applicable() {
        cov_mark::check!(extract_function_in_braces_is_not_applicable);
        check_assist_not_applicable(extract_function, r"fn foo() { $0}$0");
    }

    #[test]
    fn in_right_curly_is_not_applicable() {
        cov_mark::check!(extract_function_in_braces_is_not_applicable);
        check_assist_not_applicable(extract_function, r"fn foo() $0{$0 }");
    }

    #[test]
    fn in_left_paren_is_not_applicable() {
        cov_mark::check!(extract_function_in_braces_is_not_applicable);
        check_assist_not_applicable(extract_function, r"fn foo( $0)$0 { }");
    }

    #[test]
    fn in_right_paren_is_not_applicable() {
        cov_mark::check!(extract_function_in_braces_is_not_applicable);
        check_assist_not_applicable(extract_function, r"fn foo $0($0 ) { }");
    }

    #[test]
    fn in_left_brack_is_not_applicable() {
        cov_mark::check!(extract_function_in_braces_is_not_applicable);
        check_assist_not_applicable(extract_function, r"fn foo(arr: &mut [i32$0]$0) {}");
    }

    #[test]
    fn in_right_brack_is_not_applicable() {
        cov_mark::check!(extract_function_in_braces_is_not_applicable);
        check_assist_not_applicable(extract_function, r"fn foo(arr: &mut $0[$0i32]) {}");
    }
}
