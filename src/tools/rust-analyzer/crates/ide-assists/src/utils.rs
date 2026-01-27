//! Assorted functions shared by several assists.

use std::slice;

pub(crate) use gen_trait_fn_body::gen_trait_fn_body;
use hir::{
    DisplayTarget, HasAttrs as HirHasAttrs, HirDisplay, InFile, ModuleDef, PathResolution,
    Semantics,
    db::{ExpandDatabase, HirDatabase},
};
use ide_db::{
    RootDatabase,
    assists::ExprFillDefaultMode,
    famous_defs::FamousDefs,
    path_transform::PathTransform,
    syntax_helpers::{node_ext::preorder_expr, prettify_macro_expansion},
};
use stdx::format_to;
use syntax::{
    AstNode, AstToken, Direction, NodeOrToken, SourceFile,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken, T, TextRange, TextSize, WalkEvent,
    ast::{
        self, HasArgList, HasAttrs, HasGenericParams, HasName, HasTypeBounds, Whitespace,
        edit::{AstNodeEdit, IndentLevel},
        edit_in_place::AttrsOwnerEdit,
        make,
        syntax_factory::SyntaxFactory,
    },
    syntax_editor::{Element, Removable, SyntaxEditor},
};

use crate::{
    AssistConfig,
    assist_context::{AssistContext, SourceChangeBuilder},
};

mod gen_trait_fn_body;
pub(crate) mod ref_field_expr;

pub(crate) fn unwrap_trivial_block(block_expr: ast::BlockExpr) -> ast::Expr {
    extract_trivial_expression(&block_expr)
        .filter(|expr| !expr.syntax().text().contains_char('\n'))
        .unwrap_or_else(|| block_expr.into())
}

pub fn extract_trivial_expression(block_expr: &ast::BlockExpr) -> Option<ast::Expr> {
    if block_expr.modifier().is_some() {
        return None;
    }
    let stmt_list = block_expr.stmt_list()?;
    let has_anything_else = |thing: &SyntaxNode| -> bool {
        let mut non_trivial_children =
            stmt_list.syntax().children_with_tokens().filter(|it| match it.kind() {
                WHITESPACE | T!['{'] | T!['}'] => false,
                _ => it.as_node() != Some(thing),
            });
        non_trivial_children.next().is_some()
    };
    if stmt_list
        .syntax()
        .children_with_tokens()
        .filter_map(NodeOrToken::into_token)
        .any(|token| token.kind() == syntax::SyntaxKind::COMMENT)
    {
        return None;
    }

    if let Some(expr) = stmt_list.tail_expr() {
        if has_anything_else(expr.syntax()) {
            return None;
        }
        return Some(expr);
    }
    // Unwrap `{ continue; }`
    let stmt = stmt_list.statements().next()?;
    if let ast::Stmt::ExprStmt(expr_stmt) = stmt {
        if has_anything_else(expr_stmt.syntax()) {
            return None;
        }
        let expr = expr_stmt.expr()?;
        if matches!(expr.syntax().kind(), CONTINUE_EXPR | BREAK_EXPR | RETURN_EXPR) {
            return Some(expr);
        }
    }
    None
}

pub(crate) fn wrap_block(expr: &ast::Expr) -> ast::BlockExpr {
    if let ast::Expr::BlockExpr(block) = expr
        && let Some(first) = block.syntax().first_token()
        && first.kind() == T!['{']
    {
        block.reset_indent()
    } else {
        make::block_expr(None, Some(expr.reset_indent().indent(1.into())))
    }
}

/// This is a method with a heuristics to support test methods annotated with custom test annotations, such as
/// `#[test_case(...)]`, `#[tokio::test]` and similar.
/// Also a regular `#[test]` annotation is supported.
///
/// It may produce false positives, for example, `#[wasm_bindgen_test]` requires a different command to run the test,
/// but it's better than not to have the runnables for the tests at all.
pub fn test_related_attribute_syn(fn_def: &ast::Fn) -> Option<ast::Attr> {
    fn_def.attrs().find_map(|attr| {
        let path = attr.path()?;
        let text = path.syntax().text().to_string();
        if text.starts_with("test") || text.ends_with("test") { Some(attr) } else { None }
    })
}

pub fn has_test_related_attribute(attrs: &hir::AttrsWithOwner) -> bool {
    attrs.is_test()
}

#[derive(Clone, Copy, PartialEq)]
pub enum IgnoreAssocItems {
    DocHiddenAttrPresent,
    No,
}

#[derive(Copy, Clone, PartialEq)]
pub enum DefaultMethods {
    Only,
    No,
}

pub fn filter_assoc_items(
    sema: &Semantics<'_, RootDatabase>,
    items: &[hir::AssocItem],
    default_methods: DefaultMethods,
    ignore_items: IgnoreAssocItems,
) -> Vec<InFile<ast::AssocItem>> {
    return items
        .iter()
        .copied()
        .filter(|assoc_item| {
            if ignore_items == IgnoreAssocItems::DocHiddenAttrPresent
                && assoc_item.attrs(sema.db).is_doc_hidden()
            {
                if let hir::AssocItem::Function(f) = assoc_item
                    && !f.has_body(sema.db)
                {
                    return true;
                }
                return false;
            }

            true
        })
        // Note: This throws away items with no source.
        .filter_map(|assoc_item| {
            let item = match assoc_item {
                hir::AssocItem::Function(it) => sema.source(it)?.map(ast::AssocItem::Fn),
                hir::AssocItem::TypeAlias(it) => sema.source(it)?.map(ast::AssocItem::TypeAlias),
                hir::AssocItem::Const(it) => sema.source(it)?.map(ast::AssocItem::Const),
            };
            Some(item)
        })
        .filter(has_def_name)
        .filter(|it| match &it.value {
            ast::AssocItem::Fn(def) => matches!(
                (default_methods, def.body()),
                (DefaultMethods::Only, Some(_)) | (DefaultMethods::No, None)
            ),
            ast::AssocItem::Const(def) => matches!(
                (default_methods, def.body()),
                (DefaultMethods::Only, Some(_)) | (DefaultMethods::No, None)
            ),
            _ => default_methods == DefaultMethods::No,
        })
        .collect();

    fn has_def_name(item: &InFile<ast::AssocItem>) -> bool {
        match &item.value {
            ast::AssocItem::Fn(def) => def.name(),
            ast::AssocItem::TypeAlias(def) => def.name(),
            ast::AssocItem::Const(def) => def.name(),
            ast::AssocItem::MacroCall(_) => None,
        }
        .is_some()
    }
}

/// Given `original_items` retrieved from the trait definition (usually by
/// [`filter_assoc_items()`]), clones each item for update and applies path transformation to it,
/// then inserts into `impl_`. Returns the modified `impl_` and the first associated item that got
/// inserted.
#[must_use]
pub fn add_trait_assoc_items_to_impl(
    sema: &Semantics<'_, RootDatabase>,
    config: &AssistConfig,
    original_items: &[InFile<ast::AssocItem>],
    trait_: hir::Trait,
    impl_: &ast::Impl,
    target_scope: &hir::SemanticsScope<'_>,
) -> Vec<ast::AssocItem> {
    let new_indent_level = IndentLevel::from_node(impl_.syntax()) + 1;
    original_items
        .iter()
        .map(|InFile { file_id, value: original_item }| {
            let mut cloned_item = {
                if let Some(macro_file) = file_id.macro_file() {
                    let span_map = sema.db.expansion_span_map(macro_file);
                    let item_prettified = prettify_macro_expansion(
                        sema.db,
                        original_item.syntax().clone(),
                        &span_map,
                        target_scope.krate().into(),
                    );
                    if let Some(formatted) = ast::AssocItem::cast(item_prettified) {
                        return formatted;
                    } else {
                        stdx::never!("formatted `AssocItem` could not be cast back to `AssocItem`");
                    }
                }
                original_item
            }
            .reset_indent();

            if let Some(source_scope) = sema.scope(original_item.syntax()) {
                // FIXME: Paths in nested macros are not handled well. See
                // `add_missing_impl_members::paths_in_nested_macro_should_get_transformed` test.
                let transform =
                    PathTransform::trait_impl(target_scope, &source_scope, trait_, impl_.clone());
                cloned_item = ast::AssocItem::cast(transform.apply(cloned_item.syntax())).unwrap();
            }
            cloned_item.remove_attrs_and_docs();
            cloned_item
        })
        .filter_map(|item| match item {
            ast::AssocItem::Fn(fn_) if fn_.body().is_none() => {
                let fn_ = fn_.clone_subtree();
                let new_body = &make::block_expr(
                    None,
                    Some(match config.expr_fill_default {
                        ExprFillDefaultMode::Todo => make::ext::expr_todo(),
                        ExprFillDefaultMode::Underscore => make::ext::expr_underscore(),
                        ExprFillDefaultMode::Default => make::ext::expr_todo(),
                    }),
                );
                let new_body = AstNodeEdit::indent(new_body, IndentLevel::single());
                let mut fn_editor = SyntaxEditor::new(fn_.syntax().clone());
                fn_.replace_or_insert_body(&mut fn_editor, new_body);
                let new_fn_ = fn_editor.finish().new_root().clone();
                ast::AssocItem::cast(new_fn_)
            }
            ast::AssocItem::TypeAlias(type_alias) => {
                let type_alias = type_alias.clone_subtree();
                if let Some(type_bound_list) = type_alias.type_bound_list() {
                    let mut type_alias_editor = SyntaxEditor::new(type_alias.syntax().clone());
                    type_bound_list.remove(&mut type_alias_editor);
                    let type_alias = type_alias_editor.finish().new_root().clone();
                    ast::AssocItem::cast(type_alias)
                } else {
                    Some(ast::AssocItem::TypeAlias(type_alias))
                }
            }
            item => Some(item),
        })
        .map(|item| AstNodeEdit::indent(&item, new_indent_level))
        .collect()
}

pub(crate) fn vis_offset(node: &SyntaxNode) -> TextSize {
    node.children_with_tokens()
        .find(|it| !matches!(it.kind(), WHITESPACE | COMMENT | ATTR))
        .map(|it| it.text_range().start())
        .unwrap_or_else(|| node.text_range().start())
}

pub(crate) fn invert_boolean_expression(make: &SyntaxFactory, expr: ast::Expr) -> ast::Expr {
    invert_special_case(make, &expr).unwrap_or_else(|| make.expr_prefix(T![!], expr).into())
}

// FIXME: Migrate usages of this function to the above function and remove this.
pub(crate) fn invert_boolean_expression_legacy(expr: ast::Expr) -> ast::Expr {
    invert_special_case_legacy(&expr).unwrap_or_else(|| make::expr_prefix(T![!], expr).into())
}

fn invert_special_case(make: &SyntaxFactory, expr: &ast::Expr) -> Option<ast::Expr> {
    match expr {
        ast::Expr::BinExpr(bin) => {
            let op_kind = bin.op_kind()?;
            let rev_kind = match op_kind {
                ast::BinaryOp::CmpOp(ast::CmpOp::Eq { negated }) => {
                    ast::BinaryOp::CmpOp(ast::CmpOp::Eq { negated: !negated })
                }
                ast::BinaryOp::CmpOp(ast::CmpOp::Ord { ordering: ast::Ordering::Less, strict }) => {
                    ast::BinaryOp::CmpOp(ast::CmpOp::Ord {
                        ordering: ast::Ordering::Greater,
                        strict: !strict,
                    })
                }
                ast::BinaryOp::CmpOp(ast::CmpOp::Ord {
                    ordering: ast::Ordering::Greater,
                    strict,
                }) => ast::BinaryOp::CmpOp(ast::CmpOp::Ord {
                    ordering: ast::Ordering::Less,
                    strict: !strict,
                }),
                // Parenthesize other expressions before prefixing `!`
                _ => {
                    return Some(
                        make.expr_prefix(T![!], make.expr_paren(expr.clone()).into()).into(),
                    );
                }
            };

            Some(make.expr_bin(bin.lhs()?, rev_kind, bin.rhs()?).into())
        }
        ast::Expr::MethodCallExpr(mce) => {
            let receiver = mce.receiver()?;
            let method = mce.name_ref()?;
            let arg_list = mce.arg_list()?;

            let method = match method.text().as_str() {
                "is_some" => "is_none",
                "is_none" => "is_some",
                "is_ok" => "is_err",
                "is_err" => "is_ok",
                _ => return None,
            };

            Some(make.expr_method_call(receiver, make.name_ref(method), arg_list).into())
        }
        ast::Expr::PrefixExpr(pe) if pe.op_kind()? == ast::UnaryOp::Not => match pe.expr()? {
            ast::Expr::ParenExpr(parexpr) => {
                parexpr.expr().map(|e| e.clone_subtree().clone_for_update())
            }
            _ => pe.expr().map(|e| e.clone_subtree().clone_for_update()),
        },
        ast::Expr::Literal(lit) => match lit.kind() {
            ast::LiteralKind::Bool(b) => match b {
                true => Some(ast::Expr::Literal(make.expr_literal("false"))),
                false => Some(ast::Expr::Literal(make.expr_literal("true"))),
            },
            _ => None,
        },
        _ => None,
    }
}

fn invert_special_case_legacy(expr: &ast::Expr) -> Option<ast::Expr> {
    match expr {
        ast::Expr::BinExpr(bin) => {
            let bin = bin.clone_subtree();
            let op_token = bin.op_token()?;
            let rev_token = match op_token.kind() {
                T![==] => T![!=],
                T![!=] => T![==],
                T![<] => T![>=],
                T![<=] => T![>],
                T![>] => T![<=],
                T![>=] => T![<],
                // Parenthesize other expressions before prefixing `!`
                _ => {
                    return Some(
                        make::expr_prefix(T![!], make::expr_paren(expr.clone()).into()).into(),
                    );
                }
            };
            let mut bin_editor = SyntaxEditor::new(bin.syntax().clone());
            bin_editor.replace(op_token, make::token(rev_token));
            ast::Expr::cast(bin_editor.finish().new_root().clone())
        }
        ast::Expr::MethodCallExpr(mce) => {
            let receiver = mce.receiver()?;
            let method = mce.name_ref()?;
            let arg_list = mce.arg_list()?;

            let method = match method.text().as_str() {
                "is_some" => "is_none",
                "is_none" => "is_some",
                "is_ok" => "is_err",
                "is_err" => "is_ok",
                _ => return None,
            };
            Some(make::expr_method_call(receiver, make::name_ref(method), arg_list).into())
        }
        ast::Expr::PrefixExpr(pe) if pe.op_kind()? == ast::UnaryOp::Not => match pe.expr()? {
            ast::Expr::ParenExpr(parexpr) => parexpr.expr(),
            _ => pe.expr(),
        },
        ast::Expr::Literal(lit) => match lit.kind() {
            ast::LiteralKind::Bool(b) => match b {
                true => Some(ast::Expr::Literal(make::expr_literal("false"))),
                false => Some(ast::Expr::Literal(make::expr_literal("true"))),
            },
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn insert_attributes(
    before: impl Element,
    edit: &mut SyntaxEditor,
    attrs: impl IntoIterator<Item = ast::Attr>,
) {
    let mut attrs = attrs.into_iter().peekable();
    if attrs.peek().is_none() {
        return;
    }
    let elem = before.syntax_element();
    let indent = IndentLevel::from_element(&elem);
    let whitespace = format!("\n{indent}");
    edit.insert_all(
        syntax::syntax_editor::Position::before(elem),
        attrs
            .flat_map(|attr| {
                [attr.syntax().clone().into(), make::tokens::whitespace(&whitespace).into()]
            })
            .collect(),
    );
}

pub(crate) fn next_prev() -> impl Iterator<Item = Direction> {
    [Direction::Next, Direction::Prev].into_iter()
}

pub(crate) fn does_pat_match_variant(pat: &ast::Pat, var: &ast::Pat) -> bool {
    let first_node_text = |pat: &ast::Pat| pat.syntax().first_child().map(|node| node.text());

    let pat_head = match pat {
        ast::Pat::IdentPat(bind_pat) => match bind_pat.pat() {
            Some(p) => first_node_text(&p),
            None => return pat.syntax().text() == var.syntax().text(),
        },
        pat => first_node_text(pat),
    };

    let var_head = first_node_text(var);

    pat_head == var_head
}

pub(crate) fn does_pat_variant_nested_or_literal(ctx: &AssistContext<'_>, pat: &ast::Pat) -> bool {
    check_pat_variant_nested_or_literal_with_depth(ctx, pat, 0)
}

fn check_pat_variant_from_enum(ctx: &AssistContext<'_>, pat: &ast::Pat) -> bool {
    ctx.sema.type_of_pat(pat).is_none_or(|ty: hir::TypeInfo<'_>| {
        ty.adjusted().as_adt().is_some_and(|adt| matches!(adt, hir::Adt::Enum(_)))
    })
}

fn check_pat_variant_nested_or_literal_with_depth(
    ctx: &AssistContext<'_>,
    pat: &ast::Pat,
    depth_after_refutable: usize,
) -> bool {
    if depth_after_refutable > 1 {
        return true;
    }

    match pat {
        ast::Pat::RestPat(_) | ast::Pat::WildcardPat(_) | ast::Pat::RefPat(_) => false,

        ast::Pat::LiteralPat(_)
        | ast::Pat::RangePat(_)
        | ast::Pat::MacroPat(_)
        | ast::Pat::PathPat(_)
        | ast::Pat::BoxPat(_)
        | ast::Pat::ConstBlockPat(_) => true,

        ast::Pat::IdentPat(ident_pat) => ident_pat.pat().is_some_and(|pat| {
            check_pat_variant_nested_or_literal_with_depth(ctx, &pat, depth_after_refutable)
        }),
        ast::Pat::ParenPat(paren_pat) => paren_pat.pat().is_none_or(|pat| {
            check_pat_variant_nested_or_literal_with_depth(ctx, &pat, depth_after_refutable)
        }),
        ast::Pat::TuplePat(tuple_pat) => tuple_pat.fields().any(|pat| {
            check_pat_variant_nested_or_literal_with_depth(ctx, &pat, depth_after_refutable)
        }),
        ast::Pat::RecordPat(record_pat) => {
            let adjusted_next_depth =
                depth_after_refutable + if check_pat_variant_from_enum(ctx, pat) { 1 } else { 0 };
            record_pat.record_pat_field_list().is_none_or(|pat| {
                pat.fields().any(|pat| {
                    pat.pat().is_none_or(|pat| {
                        check_pat_variant_nested_or_literal_with_depth(
                            ctx,
                            &pat,
                            adjusted_next_depth,
                        )
                    })
                })
            })
        }
        ast::Pat::OrPat(or_pat) => or_pat.pats().any(|pat| {
            check_pat_variant_nested_or_literal_with_depth(ctx, &pat, depth_after_refutable)
        }),
        ast::Pat::TupleStructPat(tuple_struct_pat) => {
            let adjusted_next_depth =
                depth_after_refutable + if check_pat_variant_from_enum(ctx, pat) { 1 } else { 0 };
            tuple_struct_pat.fields().any(|pat| {
                check_pat_variant_nested_or_literal_with_depth(ctx, &pat, adjusted_next_depth)
            })
        }
        ast::Pat::SlicePat(slice_pat) => {
            let mut pats = slice_pat.pats();
            pats.next()
                .is_none_or(|pat| !matches!(pat, ast::Pat::RestPat(_)) || pats.next().is_some())
        }
    }
}

// Uses a syntax-driven approach to find any impl blocks for the struct that
// exist within the module/file
//
// Returns `None` if we've found an existing fn
//
// FIXME: change the new fn checking to a more semantic approach when that's more
// viable (e.g. we process proc macros, etc)
// FIXME: this partially overlaps with `find_impl_block_*`

/// `find_struct_impl` looks for impl of a struct, but this also has additional feature
/// where it takes a list of function names and check if they exist inside impl_, if
/// even one match is found, it returns None.
///
/// That means this function can have 3 potential return values:
///  - `None`: an impl exists, but one of the function names within the impl matches one of the provided names.
///  - `Some(None)`: no impl exists.
///  - `Some(Some(_))`: an impl exists, with no matching function names.
pub(crate) fn find_struct_impl(
    ctx: &AssistContext<'_>,
    adt: &ast::Adt,
    names: &[String],
) -> Option<Option<ast::Impl>> {
    let db = ctx.db();
    let module = adt.syntax().parent()?;

    let struct_def = ctx.sema.to_def(adt)?;

    let block = module.descendants().filter_map(ast::Impl::cast).find_map(|impl_blk| {
        let blk = ctx.sema.to_def(&impl_blk)?;

        // FIXME: handle e.g. `struct S<T>; impl<U> S<U> {}`
        // (we currently use the wrong type parameter)
        // also we wouldn't want to use e.g. `impl S<u32>`

        let same_ty = match blk.self_ty(db).as_adt() {
            Some(def) => def == struct_def,
            None => false,
        };
        let not_trait_impl = blk.trait_(db).is_none();

        if !(same_ty && not_trait_impl) { None } else { Some(impl_blk) }
    });

    if let Some(ref impl_blk) = block
        && has_any_fn(impl_blk, names)
    {
        return None;
    }

    Some(block)
}

fn has_any_fn(imp: &ast::Impl, names: &[String]) -> bool {
    if let Some(il) = imp.assoc_item_list() {
        for item in il.assoc_items() {
            if let ast::AssocItem::Fn(f) = item
                && let Some(name) = f.name()
                && names.iter().any(|n| n.eq_ignore_ascii_case(&name.text()))
            {
                return true;
            }
        }
    }

    false
}

/// Find the end of the `impl` block for the given `ast::Impl`.
//
// FIXME: this partially overlaps with `find_struct_impl`
pub(crate) fn find_impl_block_end(impl_def: ast::Impl, buf: &mut String) -> Option<TextSize> {
    buf.push('\n');
    let end = impl_def
        .assoc_item_list()
        .and_then(|it| it.r_curly_token())?
        .prev_sibling_or_token()?
        .text_range()
        .end();
    Some(end)
}

/// Generates the surrounding `impl Type { <code> }` including type and lifetime
/// parameters.
// FIXME: migrate remaining uses to `generate_impl`
pub(crate) fn generate_impl_text(adt: &ast::Adt, code: &str) -> String {
    generate_impl_text_inner(adt, None, true, code)
}

/// Generates the surrounding `impl <trait> for Type { <code> }` including type
/// and lifetime parameters, with `<trait>` appended to `impl`'s generic parameters' bounds.
///
/// This is useful for traits like `PartialEq`, since `impl<T> PartialEq for U<T>` often requires `T: PartialEq`.
// FIXME: migrate remaining uses to `generate_trait_impl`
#[allow(dead_code)]
pub(crate) fn generate_trait_impl_text(adt: &ast::Adt, trait_text: &str, code: &str) -> String {
    generate_impl_text_inner(adt, Some(trait_text), true, code)
}

/// Generates the surrounding `impl <trait> for Type { <code> }` including type
/// and lifetime parameters, with `impl`'s generic parameters' bounds kept as-is.
///
/// This is useful for traits like `From<T>`, since `impl<T> From<T> for U<T>` doesn't require `T: From<T>`.
// FIXME: migrate remaining uses to `generate_trait_impl_intransitive`
pub(crate) fn generate_trait_impl_text_intransitive(
    adt: &ast::Adt,
    trait_text: &str,
    code: &str,
) -> String {
    generate_impl_text_inner(adt, Some(trait_text), false, code)
}

fn generate_impl_text_inner(
    adt: &ast::Adt,
    trait_text: Option<&str>,
    trait_is_transitive: bool,
    code: &str,
) -> String {
    // Ensure lifetime params are before type & const params
    let generic_params = adt.generic_param_list().map(|generic_params| {
        let lifetime_params =
            generic_params.lifetime_params().map(ast::GenericParam::LifetimeParam);
        let ty_or_const_params = generic_params.type_or_const_params().filter_map(|param| {
            let param = match param {
                ast::TypeOrConstParam::Type(param) => {
                    // remove defaults since they can't be specified in impls
                    let mut bounds =
                        param.type_bound_list().map_or_else(Vec::new, |it| it.bounds().collect());
                    if let Some(trait_) = trait_text {
                        // Add the current trait to `bounds` if the trait is transitive,
                        // meaning `impl<T> Trait for U<T>` requires `T: Trait`.
                        if trait_is_transitive {
                            bounds.push(make::type_bound_text(trait_));
                        }
                    };
                    // `{ty_param}: {bounds}`
                    let param = make::type_param(param.name()?, make::type_bound_list(bounds));
                    ast::GenericParam::TypeParam(param)
                }
                ast::TypeOrConstParam::Const(param) => {
                    // remove defaults since they can't be specified in impls
                    let param = make::const_param(param.name()?, param.ty()?);
                    ast::GenericParam::ConstParam(param)
                }
            };
            Some(param)
        });

        make::generic_param_list(itertools::chain(lifetime_params, ty_or_const_params))
    });

    // FIXME: use syntax::make & mutable AST apis instead
    // `trait_text` and `code` can't be opaque blobs of text
    let mut buf = String::with_capacity(code.len());

    // Copy any cfg attrs from the original adt
    buf.push_str("\n\n");
    let cfg_attrs = adt
        .attrs()
        .filter(|attr| attr.as_simple_call().map(|(name, _arg)| name == "cfg").unwrap_or(false));
    cfg_attrs.for_each(|attr| buf.push_str(&format!("{attr}\n")));

    // `impl{generic_params} {trait_text} for {name}{generic_params.to_generic_args()}`
    buf.push_str("impl");
    if let Some(generic_params) = &generic_params {
        format_to!(buf, "{generic_params}");
    }
    buf.push(' ');
    if let Some(trait_text) = trait_text {
        buf.push_str(trait_text);
        buf.push_str(" for ");
    }
    buf.push_str(&adt.name().unwrap().text());
    if let Some(generic_params) = generic_params {
        format_to!(buf, "{}", generic_params.to_generic_args());
    }

    match adt.where_clause() {
        Some(where_clause) => {
            format_to!(buf, "\n{where_clause}\n{{\n{code}\n}}");
        }
        None => {
            format_to!(buf, " {{\n{code}\n}}");
        }
    }

    buf
}

/// Generates the corresponding `impl Type {}` including type and lifetime
/// parameters.
pub(crate) fn generate_impl_with_item(
    adt: &ast::Adt,
    body: Option<ast::AssocItemList>,
) -> ast::Impl {
    generate_impl_inner(false, adt, None, true, body)
}

pub(crate) fn generate_impl(adt: &ast::Adt) -> ast::Impl {
    generate_impl_inner(false, adt, None, true, None)
}

/// Generates the corresponding `impl <trait> for Type {}` including type
/// and lifetime parameters, with `<trait>` appended to `impl`'s generic parameters' bounds.
///
/// This is useful for traits like `PartialEq`, since `impl<T> PartialEq for U<T>` often requires `T: PartialEq`.
pub(crate) fn generate_trait_impl(is_unsafe: bool, adt: &ast::Adt, trait_: ast::Type) -> ast::Impl {
    generate_impl_inner(is_unsafe, adt, Some(trait_), true, None)
}

/// Generates the corresponding `impl <trait> for Type {}` including type
/// and lifetime parameters, with `impl`'s generic parameters' bounds kept as-is.
///
/// This is useful for traits like `From<T>`, since `impl<T> From<T> for U<T>` doesn't require `T: From<T>`.
pub(crate) fn generate_trait_impl_intransitive(adt: &ast::Adt, trait_: ast::Type) -> ast::Impl {
    generate_impl_inner(false, adt, Some(trait_), false, None)
}

fn generate_impl_inner(
    is_unsafe: bool,
    adt: &ast::Adt,
    trait_: Option<ast::Type>,
    trait_is_transitive: bool,
    body: Option<ast::AssocItemList>,
) -> ast::Impl {
    // Ensure lifetime params are before type & const params
    let generic_params = adt.generic_param_list().map(|generic_params| {
        let lifetime_params =
            generic_params.lifetime_params().map(ast::GenericParam::LifetimeParam);
        let ty_or_const_params = generic_params.type_or_const_params().filter_map(|param| {
            let param = match param {
                ast::TypeOrConstParam::Type(param) => {
                    // remove defaults since they can't be specified in impls
                    let mut bounds =
                        param.type_bound_list().map_or_else(Vec::new, |it| it.bounds().collect());
                    if let Some(trait_) = &trait_ {
                        // Add the current trait to `bounds` if the trait is transitive,
                        // meaning `impl<T> Trait for U<T>` requires `T: Trait`.
                        if trait_is_transitive {
                            bounds.push(make::type_bound(trait_.clone()));
                        }
                    };
                    // `{ty_param}: {bounds}`
                    let param = make::type_param(param.name()?, make::type_bound_list(bounds));
                    ast::GenericParam::TypeParam(param)
                }
                ast::TypeOrConstParam::Const(param) => {
                    // remove defaults since they can't be specified in impls
                    let param = make::const_param(param.name()?, param.ty()?);
                    ast::GenericParam::ConstParam(param)
                }
            };
            Some(param)
        });

        make::generic_param_list(itertools::chain(lifetime_params, ty_or_const_params))
    });
    let generic_args =
        generic_params.as_ref().map(|params| params.to_generic_args().clone_for_update());
    let ty = make::ty_path(make::ext::ident_path(&adt.name().unwrap().text()));

    let cfg_attrs =
        adt.attrs().filter(|attr| attr.as_simple_call().is_some_and(|(name, _arg)| name == "cfg"));
    match trait_ {
        Some(trait_) => make::impl_trait(
            cfg_attrs,
            is_unsafe,
            None,
            None,
            generic_params,
            generic_args,
            false,
            trait_,
            ty,
            None,
            adt.where_clause(),
            body,
        ),
        None => make::impl_(cfg_attrs, generic_params, generic_args, ty, adt.where_clause(), body),
    }
    .clone_for_update()
}

pub(crate) fn add_method_to_adt(
    builder: &mut SourceChangeBuilder,
    adt: &ast::Adt,
    impl_def: Option<ast::Impl>,
    method: &str,
) {
    let mut buf = String::with_capacity(method.len() + 2);
    if impl_def.is_some() {
        buf.push('\n');
    }
    buf.push_str(method);

    let start_offset = impl_def
        .and_then(|impl_def| find_impl_block_end(impl_def, &mut buf))
        .unwrap_or_else(|| {
            buf = generate_impl_text(adt, &buf);
            adt.syntax().text_range().end()
        });

    builder.insert(start_offset, buf);
}

#[derive(Debug)]
pub(crate) struct ReferenceConversion<'db> {
    conversion: ReferenceConversionType,
    ty: hir::Type<'db>,
    impls_deref: bool,
}

#[derive(Debug)]
enum ReferenceConversionType {
    // reference can be stripped if the type is Copy
    Copy,
    // &String -> &str
    AsRefStr,
    // &Vec<T> -> &[T]
    AsRefSlice,
    // &Box<T> -> &T
    Dereferenced,
    // &Option<T> -> Option<&T>
    Option,
    // &Result<T, E> -> Result<&T, &E>
    Result,
}

impl<'db> ReferenceConversion<'db> {
    pub(crate) fn convert_type(
        &self,
        db: &'db dyn HirDatabase,
        display_target: DisplayTarget,
    ) -> ast::Type {
        let ty = match self.conversion {
            ReferenceConversionType::Copy => self.ty.display(db, display_target).to_string(),
            ReferenceConversionType::AsRefStr => "&str".to_owned(),
            ReferenceConversionType::AsRefSlice => {
                let type_argument_name = self
                    .ty
                    .type_arguments()
                    .next()
                    .unwrap()
                    .display(db, display_target)
                    .to_string();
                format!("&[{type_argument_name}]")
            }
            ReferenceConversionType::Dereferenced => {
                let type_argument_name = self
                    .ty
                    .type_arguments()
                    .next()
                    .unwrap()
                    .display(db, display_target)
                    .to_string();
                format!("&{type_argument_name}")
            }
            ReferenceConversionType::Option => {
                let type_argument_name = self
                    .ty
                    .type_arguments()
                    .next()
                    .unwrap()
                    .display(db, display_target)
                    .to_string();
                format!("Option<&{type_argument_name}>")
            }
            ReferenceConversionType::Result => {
                let mut type_arguments = self.ty.type_arguments();
                let first_type_argument_name =
                    type_arguments.next().unwrap().display(db, display_target).to_string();
                let second_type_argument_name =
                    type_arguments.next().unwrap().display(db, display_target).to_string();
                format!("Result<&{first_type_argument_name}, &{second_type_argument_name}>")
            }
        };

        make::ty(&ty)
    }

    pub(crate) fn getter(&self, field_name: String) -> ast::Expr {
        let expr = make::expr_field(make::ext::expr_self(), &field_name);

        match self.conversion {
            ReferenceConversionType::Copy => expr,
            ReferenceConversionType::AsRefStr
            | ReferenceConversionType::AsRefSlice
            | ReferenceConversionType::Dereferenced
            | ReferenceConversionType::Option
            | ReferenceConversionType::Result => {
                if self.impls_deref {
                    make::expr_ref(expr, false)
                } else {
                    make::expr_method_call(expr, make::name_ref("as_ref"), make::arg_list([]))
                        .into()
                }
            }
        }
    }
}

// FIXME: It should return a new hir::Type, but currently constructing new types is too cumbersome
//        and all users of this function operate on string type names, so they can do the conversion
//        itself themselves.
pub(crate) fn convert_reference_type<'db>(
    ty: hir::Type<'db>,
    db: &'db RootDatabase,
    famous_defs: &FamousDefs<'_, 'db>,
) -> Option<ReferenceConversion<'db>> {
    handle_copy(&ty, db)
        .or_else(|| handle_as_ref_str(&ty, db, famous_defs))
        .or_else(|| handle_as_ref_slice(&ty, db, famous_defs))
        .or_else(|| handle_dereferenced(&ty, db, famous_defs))
        .or_else(|| handle_option_as_ref(&ty, db, famous_defs))
        .or_else(|| handle_result_as_ref(&ty, db, famous_defs))
        .map(|(conversion, impls_deref)| ReferenceConversion { ty, conversion, impls_deref })
}

fn could_deref_to_target(ty: &hir::Type<'_>, target: &hir::Type<'_>, db: &dyn HirDatabase) -> bool {
    let ty_ref = ty.add_reference(hir::Mutability::Shared);
    let target_ref = target.add_reference(hir::Mutability::Shared);
    ty_ref.could_coerce_to(db, &target_ref)
}

fn handle_copy(
    ty: &hir::Type<'_>,
    db: &dyn HirDatabase,
) -> Option<(ReferenceConversionType, bool)> {
    ty.is_copy(db).then_some((ReferenceConversionType::Copy, true))
}

fn handle_as_ref_str(
    ty: &hir::Type<'_>,
    db: &dyn HirDatabase,
    famous_defs: &FamousDefs<'_, '_>,
) -> Option<(ReferenceConversionType, bool)> {
    let str_type = hir::BuiltinType::str().ty(db);

    ty.impls_trait(db, famous_defs.core_convert_AsRef()?, slice::from_ref(&str_type))
        .then_some((ReferenceConversionType::AsRefStr, could_deref_to_target(ty, &str_type, db)))
}

fn handle_as_ref_slice(
    ty: &hir::Type<'_>,
    db: &dyn HirDatabase,
    famous_defs: &FamousDefs<'_, '_>,
) -> Option<(ReferenceConversionType, bool)> {
    let type_argument = ty.type_arguments().next()?;
    let slice_type = hir::Type::new_slice(type_argument);

    ty.impls_trait(db, famous_defs.core_convert_AsRef()?, slice::from_ref(&slice_type)).then_some((
        ReferenceConversionType::AsRefSlice,
        could_deref_to_target(ty, &slice_type, db),
    ))
}

fn handle_dereferenced(
    ty: &hir::Type<'_>,
    db: &dyn HirDatabase,
    famous_defs: &FamousDefs<'_, '_>,
) -> Option<(ReferenceConversionType, bool)> {
    let type_argument = ty.type_arguments().next()?;

    ty.impls_trait(db, famous_defs.core_convert_AsRef()?, slice::from_ref(&type_argument))
        .then_some((
            ReferenceConversionType::Dereferenced,
            could_deref_to_target(ty, &type_argument, db),
        ))
}

fn handle_option_as_ref(
    ty: &hir::Type<'_>,
    db: &dyn HirDatabase,
    famous_defs: &FamousDefs<'_, '_>,
) -> Option<(ReferenceConversionType, bool)> {
    if ty.as_adt() == famous_defs.core_option_Option()?.ty(db).as_adt() {
        Some((ReferenceConversionType::Option, false))
    } else {
        None
    }
}

fn handle_result_as_ref(
    ty: &hir::Type<'_>,
    db: &dyn HirDatabase,
    famous_defs: &FamousDefs<'_, '_>,
) -> Option<(ReferenceConversionType, bool)> {
    if ty.as_adt() == famous_defs.core_result_Result()?.ty(db).as_adt() {
        Some((ReferenceConversionType::Result, false))
    } else {
        None
    }
}

pub(crate) fn get_methods(items: &ast::AssocItemList) -> Vec<ast::Fn> {
    items
        .assoc_items()
        .flat_map(|i| match i {
            ast::AssocItem::Fn(f) => Some(f),
            _ => None,
        })
        .filter(|f| f.name().is_some())
        .collect()
}

/// Trim(remove leading and trailing whitespace) `initial_range` in `source_file`, return the trimmed range.
pub(crate) fn trimmed_text_range(source_file: &SourceFile, initial_range: TextRange) -> TextRange {
    let mut trimmed_range = initial_range;
    while source_file
        .syntax()
        .token_at_offset(trimmed_range.start())
        .find_map(Whitespace::cast)
        .is_some()
        && trimmed_range.start() < trimmed_range.end()
    {
        let start = trimmed_range.start() + TextSize::from(1);
        trimmed_range = TextRange::new(start, trimmed_range.end());
    }
    while source_file
        .syntax()
        .token_at_offset(trimmed_range.end())
        .find_map(Whitespace::cast)
        .is_some()
        && trimmed_range.start() < trimmed_range.end()
    {
        let end = trimmed_range.end() - TextSize::from(1);
        trimmed_range = TextRange::new(trimmed_range.start(), end);
    }
    trimmed_range
}

/// Convert a list of function params to a list of arguments that can be passed
/// into a function call.
pub(crate) fn convert_param_list_to_arg_list(list: ast::ParamList) -> ast::ArgList {
    let mut args = vec![];
    for param in list.params() {
        if let Some(ast::Pat::IdentPat(pat)) = param.pat()
            && let Some(name) = pat.name()
        {
            let name = name.to_string();
            let expr = make::expr_path(make::ext::ident_path(&name));
            args.push(expr);
        }
    }
    make::arg_list(args)
}

/// Calculate the number of hashes required for a raw string containing `s`
pub(crate) fn required_hashes(s: &str) -> usize {
    let mut res = 0usize;
    for idx in s.match_indices('"').map(|(i, _)| i) {
        let (_, sub) = s.split_at(idx + 1);
        let n_hashes = sub.chars().take_while(|c| *c == '#').count();
        res = res.max(n_hashes + 1)
    }
    res
}
#[test]
fn test_required_hashes() {
    assert_eq!(0, required_hashes("abc"));
    assert_eq!(0, required_hashes("###"));
    assert_eq!(1, required_hashes("\""));
    assert_eq!(2, required_hashes("\"#abc"));
    assert_eq!(0, required_hashes("#abc"));
    assert_eq!(3, required_hashes("#ab\"##c"));
    assert_eq!(5, required_hashes("#ab\"##\"####c"));
}

/// Calculate the string literal suffix length
pub(crate) fn string_suffix(s: &str) -> Option<&str> {
    s.rfind(['"', '\'', '#']).map(|i| &s[i + 1..])
}
#[test]
fn test_string_suffix() {
    assert_eq!(Some(""), string_suffix(r#""abc""#));
    assert_eq!(Some(""), string_suffix(r#""""#));
    assert_eq!(Some("a"), string_suffix(r#"""a"#));
    assert_eq!(Some("i32"), string_suffix(r#"""i32"#));
    assert_eq!(Some("i32"), string_suffix(r#"r""i32"#));
    assert_eq!(Some("i32"), string_suffix(r##"r#""#i32"##));
}

/// Calculate the string literal prefix length
pub(crate) fn string_prefix(s: &str) -> Option<&str> {
    s.split_once(['"', '\'', '#']).map(|(prefix, _)| prefix)
}
#[test]
fn test_string_prefix() {
    assert_eq!(Some(""), string_prefix(r#""abc""#));
    assert_eq!(Some(""), string_prefix(r#""""#));
    assert_eq!(Some(""), string_prefix(r#"""suffix"#));
    assert_eq!(Some("c"), string_prefix(r#"c"""#));
    assert_eq!(Some("r"), string_prefix(r#"r"""#));
    assert_eq!(Some("cr"), string_prefix(r#"cr"""#));
    assert_eq!(Some("r"), string_prefix(r##"r#""#"##));
}

pub(crate) fn add_group_separators(s: &str, group_size: usize) -> String {
    let mut chars = Vec::new();
    for (i, ch) in s.chars().filter(|&ch| ch != '_').rev().enumerate() {
        if i > 0 && i % group_size == 0 && ch != '-' {
            chars.push('_');
        }
        chars.push(ch);
    }

    chars.into_iter().rev().collect()
}

/// Replaces the record expression, handling field shorthands including inside macros.
pub(crate) fn replace_record_field_expr(
    ctx: &AssistContext<'_>,
    edit: &mut SourceChangeBuilder,
    record_field: ast::RecordExprField,
    initializer: ast::Expr,
) {
    if let Some(ast::Expr::PathExpr(path_expr)) = record_field.expr() {
        // replace field shorthand
        let file_range = ctx.sema.original_range(path_expr.syntax());
        edit.insert(file_range.range.end(), format!(": {}", initializer.syntax().text()))
    } else if let Some(expr) = record_field.expr() {
        // just replace expr
        let file_range = ctx.sema.original_range(expr.syntax());
        edit.replace(file_range.range, initializer.syntax().text());
    }
}

/// Creates a token tree list from a syntax node, creating the needed delimited sub token trees.
/// Assumes that the input syntax node is a valid syntax tree.
pub(crate) fn tt_from_syntax(node: SyntaxNode) -> Vec<NodeOrToken<ast::TokenTree, SyntaxToken>> {
    let mut tt_stack = vec![(None, vec![])];

    for element in node.descendants_with_tokens() {
        let NodeOrToken::Token(token) = element else { continue };

        match token.kind() {
            T!['('] | T!['{'] | T!['['] => {
                // Found an opening delimiter, start a new sub token tree
                tt_stack.push((Some(token.kind()), vec![]));
            }
            T![')'] | T!['}'] | T![']'] => {
                // Closing a subtree
                let (delimiter, tt) = tt_stack.pop().expect("unbalanced delimiters");
                let (_, parent_tt) = tt_stack
                    .last_mut()
                    .expect("parent token tree was closed before it was completed");
                let closing_delimiter = delimiter.map(|it| match it {
                    T!['('] => T![')'],
                    T!['{'] => T!['}'],
                    T!['['] => T![']'],
                    _ => unreachable!(),
                });
                stdx::always!(
                    closing_delimiter == Some(token.kind()),
                    "mismatched opening and closing delimiters"
                );

                let sub_tt = make::token_tree(delimiter.expect("unbalanced delimiters"), tt);
                parent_tt.push(NodeOrToken::Node(sub_tt));
            }
            _ => {
                let (_, current_tt) = tt_stack.last_mut().expect("unmatched delimiters");
                current_tt.push(NodeOrToken::Token(token))
            }
        }
    }

    tt_stack.pop().expect("parent token tree was closed before it was completed").1
}

pub(crate) fn cover_let_chain(mut expr: ast::Expr, range: TextRange) -> Option<ast::Expr> {
    if !expr.syntax().text_range().contains_range(range) {
        return None;
    }
    loop {
        let (chain_expr, rest) = if let ast::Expr::BinExpr(bin_expr) = &expr
            && bin_expr.op_kind() == Some(ast::BinaryOp::LogicOp(ast::LogicOp::And))
        {
            (bin_expr.rhs(), bin_expr.lhs())
        } else {
            (Some(expr), None)
        };

        if let Some(chain_expr) = chain_expr
            && chain_expr.syntax().text_range().contains_range(range)
        {
            break Some(chain_expr);
        }
        expr = rest?;
    }
}

pub(crate) fn is_selected(
    it: &impl AstNode,
    selection: syntax::TextRange,
    allow_empty: bool,
) -> bool {
    selection.intersect(it.syntax().text_range()).is_some_and(|it| !it.is_empty())
        || allow_empty && it.syntax().text_range().contains_range(selection)
}

pub fn is_body_const(sema: &Semantics<'_, RootDatabase>, expr: &ast::Expr) -> bool {
    let mut is_const = true;
    preorder_expr(expr, &mut |ev| {
        let expr = match ev {
            WalkEvent::Enter(_) if !is_const => return true,
            WalkEvent::Enter(expr) => expr,
            WalkEvent::Leave(_) => return false,
        };
        match expr {
            ast::Expr::CallExpr(call) => {
                if let Some(ast::Expr::PathExpr(path_expr)) = call.expr()
                    && let Some(PathResolution::Def(ModuleDef::Function(func))) =
                        path_expr.path().and_then(|path| sema.resolve_path(&path))
                {
                    is_const &= func.is_const(sema.db);
                }
            }
            ast::Expr::MethodCallExpr(call) => {
                is_const &=
                    sema.resolve_method_call(&call).map(|it| it.is_const(sema.db)).unwrap_or(true)
            }
            ast::Expr::ForExpr(_)
            | ast::Expr::ReturnExpr(_)
            | ast::Expr::TryExpr(_)
            | ast::Expr::YieldExpr(_)
            | ast::Expr::AwaitExpr(_) => is_const = false,
            _ => (),
        }
        !is_const
    });
    is_const
}

// FIXME: #20460 When hir-ty can analyze the `never` statement at the end of block, remove it
pub(crate) fn is_never_block(
    sema: &Semantics<'_, RootDatabase>,
    block_expr: &ast::BlockExpr,
) -> bool {
    if let Some(tail_expr) = block_expr.tail_expr() {
        sema.type_of_expr(&tail_expr).is_some_and(|ty| ty.original.is_never())
    } else if let Some(ast::Stmt::ExprStmt(expr_stmt)) = block_expr.statements().last()
        && let Some(expr) = expr_stmt.expr()
    {
        sema.type_of_expr(&expr).is_some_and(|ty| ty.original.is_never())
    } else {
        false
    }
}
