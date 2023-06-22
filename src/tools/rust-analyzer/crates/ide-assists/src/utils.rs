//! Assorted functions shared by several assists.

use std::ops;

pub(crate) use gen_trait_fn_body::gen_trait_fn_body;
use hir::{db::HirDatabase, HirDisplay, InFile, Semantics};
use ide_db::{
    famous_defs::FamousDefs, path_transform::PathTransform,
    syntax_helpers::insert_whitespace_into_node::insert_ws_into, RootDatabase, SnippetCap,
};
use stdx::format_to;
use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        edit_in_place::{AttrsOwnerEdit, Indent, Removable},
        make, HasArgList, HasAttrs, HasGenericParams, HasName, HasTypeBounds, Whitespace,
    },
    ted, AstNode, AstToken, Direction, SourceFile,
    SyntaxKind::*,
    SyntaxNode, TextRange, TextSize, T,
};

use crate::assist_context::{AssistContext, SourceChangeBuilder};

pub(crate) mod suggest_name;
mod gen_trait_fn_body;

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

/// This is a method with a heuristics to support test methods annotated with custom test annotations, such as
/// `#[test_case(...)]`, `#[tokio::test]` and similar.
/// Also a regular `#[test]` annotation is supported.
///
/// It may produce false positives, for example, `#[wasm_bindgen_test]` requires a different command to run the test,
/// but it's better than not to have the runnables for the tests at all.
pub fn test_related_attribute(fn_def: &ast::Fn) -> Option<ast::Attr> {
    fn_def.attrs().find_map(|attr| {
        let path = attr.path()?;
        let text = path.syntax().text().to_string();
        if text.starts_with("test") || text.ends_with("test") {
            Some(attr)
        } else {
            None
        }
    })
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
) -> Vec<InFile<ast::AssocItem>> {
    return items
        .iter()
        // Note: This throws away items with no source.
        .copied()
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
pub fn add_trait_assoc_items_to_impl(
    sema: &Semantics<'_, RootDatabase>,
    original_items: &[InFile<ast::AssocItem>],
    trait_: hir::Trait,
    impl_: &ast::Impl,
    target_scope: hir::SemanticsScope<'_>,
) -> ast::AssocItem {
    let new_indent_level = IndentLevel::from_node(impl_.syntax()) + 1;
    let items = original_items.into_iter().map(|InFile { file_id, value: original_item }| {
        let cloned_item = {
            if file_id.is_macro() {
                if let Some(formatted) =
                    ast::AssocItem::cast(insert_ws_into(original_item.syntax().clone()))
                {
                    return formatted;
                } else {
                    stdx::never!("formatted `AssocItem` could not be cast back to `AssocItem`");
                }
            }
            original_item.clone_for_update()
        };

        if let Some(source_scope) = sema.scope(original_item.syntax()) {
            // FIXME: Paths in nested macros are not handled well. See
            // `add_missing_impl_members::paths_in_nested_macro_should_get_transformed` test.
            let transform =
                PathTransform::trait_impl(&target_scope, &source_scope, trait_, impl_.clone());
            transform.apply(cloned_item.syntax());
        }
        cloned_item.remove_attrs_and_docs();
        cloned_item.reindent_to(new_indent_level);
        cloned_item
    });

    let assoc_item_list = impl_.get_or_create_assoc_item_list();
    let mut first_item = None;
    for item in items {
        first_item.get_or_insert_with(|| item.clone());
        match &item {
            ast::AssocItem::Fn(fn_) if fn_.body().is_none() => {
                let body = AstNodeEdit::indent(
                    &make::block_expr(None, Some(make::ext::expr_todo())),
                    new_indent_level,
                );
                ted::replace(fn_.get_or_create_body().syntax(), body.clone_for_update().syntax())
            }
            ast::AssocItem::TypeAlias(type_alias) => {
                if let Some(type_bound_list) = type_alias.type_bound_list() {
                    type_bound_list.remove()
                }
            }
            _ => {}
        }

        assoc_item_list.add_item(item)
    }

    first_item.unwrap()
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Cursor<'a> {
    Replace(&'a SyntaxNode),
    Before(&'a SyntaxNode),
}

impl<'a> Cursor<'a> {
    fn node(self) -> &'a SyntaxNode {
        match self {
            Cursor::Replace(node) | Cursor::Before(node) => node,
        }
    }
}

pub(crate) fn render_snippet(_cap: SnippetCap, node: &SyntaxNode, cursor: Cursor<'_>) -> String {
    assert!(cursor.node().ancestors().any(|it| it == *node));
    let range = cursor.node().text_range() - node.text_range().start();
    let range: ops::Range<usize> = range.into();

    let mut placeholder = cursor.node().to_string();
    escape(&mut placeholder);
    let tab_stop = match cursor {
        Cursor::Replace(placeholder) => format!("${{0:{placeholder}}}"),
        Cursor::Before(placeholder) => format!("$0{placeholder}"),
    };

    let mut buf = node.to_string();
    buf.replace_range(range, &tab_stop);
    return buf;

    fn escape(buf: &mut String) {
        stdx::replace(buf, '{', r"\{");
        stdx::replace(buf, '}', r"\}");
        stdx::replace(buf, '$', r"\$");
    }
}

pub(crate) fn vis_offset(node: &SyntaxNode) -> TextSize {
    node.children_with_tokens()
        .find(|it| !matches!(it.kind(), WHITESPACE | COMMENT | ATTR))
        .map(|it| it.text_range().start())
        .unwrap_or_else(|| node.text_range().start())
}

pub(crate) fn invert_boolean_expression(expr: ast::Expr) -> ast::Expr {
    invert_special_case(&expr).unwrap_or_else(|| make::expr_prefix(T![!], expr))
}

fn invert_special_case(expr: &ast::Expr) -> Option<ast::Expr> {
    match expr {
        ast::Expr::BinExpr(bin) => {
            let bin = bin.clone_for_update();
            let op_token = bin.op_token()?;
            let rev_token = match op_token.kind() {
                T![==] => T![!=],
                T![!=] => T![==],
                T![<] => T![>=],
                T![<=] => T![>],
                T![>] => T![<=],
                T![>=] => T![<],
                // Parenthesize other expressions before prefixing `!`
                _ => return Some(make::expr_prefix(T![!], make::expr_paren(expr.clone()))),
            };
            ted::replace(op_token, make::token(rev_token));
            Some(bin.into())
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
            Some(make::expr_method_call(receiver, make::name_ref(method), arg_list))
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

pub(crate) fn does_nested_pattern(pat: &ast::Pat) -> bool {
    let depth = calc_depth(pat, 0);

    if 1 < depth {
        return true;
    }
    false
}

fn calc_depth(pat: &ast::Pat, depth: usize) -> usize {
    match pat {
        ast::Pat::IdentPat(_)
        | ast::Pat::BoxPat(_)
        | ast::Pat::RestPat(_)
        | ast::Pat::LiteralPat(_)
        | ast::Pat::MacroPat(_)
        | ast::Pat::OrPat(_)
        | ast::Pat::ParenPat(_)
        | ast::Pat::PathPat(_)
        | ast::Pat::WildcardPat(_)
        | ast::Pat::RangePat(_)
        | ast::Pat::RecordPat(_)
        | ast::Pat::RefPat(_)
        | ast::Pat::SlicePat(_)
        | ast::Pat::TuplePat(_)
        | ast::Pat::ConstBlockPat(_) => depth,

        // FIXME: Other patterns may also be nested. Currently it simply supports only `TupleStructPat`
        ast::Pat::TupleStructPat(pat) => {
            let mut max_depth = depth;
            for p in pat.fields() {
                let d = calc_depth(&p, depth + 1);
                if d > max_depth {
                    max_depth = d
                }
            }
            max_depth
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

        if !(same_ty && not_trait_impl) {
            None
        } else {
            Some(impl_blk)
        }
    });

    if let Some(ref impl_blk) = block {
        if has_any_fn(impl_blk, names) {
            return None;
        }
    }

    Some(block)
}

fn has_any_fn(imp: &ast::Impl, names: &[String]) -> bool {
    if let Some(il) = imp.assoc_item_list() {
        for item in il.assoc_items() {
            if let ast::AssocItem::Fn(f) = item {
                if let Some(name) = f.name() {
                    if names.iter().any(|n| n.eq_ignore_ascii_case(&name.text())) {
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Find the start of the `impl` block for the given `ast::Impl`.
//
// FIXME: this partially overlaps with `find_struct_impl`
pub(crate) fn find_impl_block_start(impl_def: ast::Impl, buf: &mut String) -> Option<TextSize> {
    buf.push('\n');
    let start = impl_def.assoc_item_list().and_then(|it| it.l_curly_token())?.text_range().end();
    Some(start)
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
pub(crate) fn generate_impl_text(adt: &ast::Adt, code: &str) -> String {
    generate_impl_text_inner(adt, None, true, code)
}

/// Generates the surrounding `impl <trait> for Type { <code> }` including type
/// and lifetime parameters, with `<trait>` appended to `impl`'s generic parameters' bounds.
///
/// This is useful for traits like `PartialEq`, since `impl<T> PartialEq for U<T>` often requires `T: PartialEq`.
pub(crate) fn generate_trait_impl_text(adt: &ast::Adt, trait_text: &str, code: &str) -> String {
    generate_impl_text_inner(adt, Some(trait_text), true, code)
}

/// Generates the surrounding `impl <trait> for Type { <code> }` including type
/// and lifetime parameters, with `impl`'s generic parameters' bounds kept as-is.
///
/// This is useful for traits like `From<T>`, since `impl<T> From<T> for U<T>` doesn't require `T: From<T>`.
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
        let ty_or_const_params = generic_params.type_or_const_params().map(|param| {
            match param {
                ast::TypeOrConstParam::Type(param) => {
                    let param = param.clone_for_update();
                    // remove defaults since they can't be specified in impls
                    param.remove_default();
                    let mut bounds =
                        param.type_bound_list().map_or_else(Vec::new, |it| it.bounds().collect());
                    if let Some(trait_) = trait_text {
                        // Add the current trait to `bounds` if the trait is transitive,
                        // meaning `impl<T> Trait for U<T>` requires `T: Trait`.
                        if trait_is_transitive {
                            bounds.push(make::type_bound(trait_));
                        }
                    };
                    // `{ty_param}: {bounds}`
                    let param =
                        make::type_param(param.name().unwrap(), make::type_bound_list(bounds));
                    ast::GenericParam::TypeParam(param)
                }
                ast::TypeOrConstParam::Const(param) => {
                    let param = param.clone_for_update();
                    // remove defaults since they can't be specified in impls
                    param.remove_default();
                    ast::GenericParam::ConstParam(param)
                }
            }
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
pub(crate) struct ReferenceConversion {
    conversion: ReferenceConversionType,
    ty: hir::Type,
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

impl ReferenceConversion {
    pub(crate) fn convert_type(&self, db: &dyn HirDatabase) -> String {
        match self.conversion {
            ReferenceConversionType::Copy => self.ty.display(db).to_string(),
            ReferenceConversionType::AsRefStr => "&str".to_string(),
            ReferenceConversionType::AsRefSlice => {
                let type_argument_name =
                    self.ty.type_arguments().next().unwrap().display(db).to_string();
                format!("&[{type_argument_name}]")
            }
            ReferenceConversionType::Dereferenced => {
                let type_argument_name =
                    self.ty.type_arguments().next().unwrap().display(db).to_string();
                format!("&{type_argument_name}")
            }
            ReferenceConversionType::Option => {
                let type_argument_name =
                    self.ty.type_arguments().next().unwrap().display(db).to_string();
                format!("Option<&{type_argument_name}>")
            }
            ReferenceConversionType::Result => {
                let mut type_arguments = self.ty.type_arguments();
                let first_type_argument_name =
                    type_arguments.next().unwrap().display(db).to_string();
                let second_type_argument_name =
                    type_arguments.next().unwrap().display(db).to_string();
                format!("Result<&{first_type_argument_name}, &{second_type_argument_name}>")
            }
        }
    }

    pub(crate) fn getter(&self, field_name: String) -> String {
        match self.conversion {
            ReferenceConversionType::Copy => format!("self.{field_name}"),
            ReferenceConversionType::AsRefStr
            | ReferenceConversionType::AsRefSlice
            | ReferenceConversionType::Dereferenced
            | ReferenceConversionType::Option
            | ReferenceConversionType::Result => format!("self.{field_name}.as_ref()"),
        }
    }
}

// FIXME: It should return a new hir::Type, but currently constructing new types is too cumbersome
//        and all users of this function operate on string type names, so they can do the conversion
//        itself themselves.
pub(crate) fn convert_reference_type(
    ty: hir::Type,
    db: &RootDatabase,
    famous_defs: &FamousDefs<'_, '_>,
) -> Option<ReferenceConversion> {
    handle_copy(&ty, db)
        .or_else(|| handle_as_ref_str(&ty, db, famous_defs))
        .or_else(|| handle_as_ref_slice(&ty, db, famous_defs))
        .or_else(|| handle_dereferenced(&ty, db, famous_defs))
        .or_else(|| handle_option_as_ref(&ty, db, famous_defs))
        .or_else(|| handle_result_as_ref(&ty, db, famous_defs))
        .map(|conversion| ReferenceConversion { ty, conversion })
}

fn handle_copy(ty: &hir::Type, db: &dyn HirDatabase) -> Option<ReferenceConversionType> {
    ty.is_copy(db).then_some(ReferenceConversionType::Copy)
}

fn handle_as_ref_str(
    ty: &hir::Type,
    db: &dyn HirDatabase,
    famous_defs: &FamousDefs<'_, '_>,
) -> Option<ReferenceConversionType> {
    let str_type = hir::BuiltinType::str().ty(db);

    ty.impls_trait(db, famous_defs.core_convert_AsRef()?, &[str_type])
        .then_some(ReferenceConversionType::AsRefStr)
}

fn handle_as_ref_slice(
    ty: &hir::Type,
    db: &dyn HirDatabase,
    famous_defs: &FamousDefs<'_, '_>,
) -> Option<ReferenceConversionType> {
    let type_argument = ty.type_arguments().next()?;
    let slice_type = hir::Type::new_slice(type_argument);

    ty.impls_trait(db, famous_defs.core_convert_AsRef()?, &[slice_type])
        .then_some(ReferenceConversionType::AsRefSlice)
}

fn handle_dereferenced(
    ty: &hir::Type,
    db: &dyn HirDatabase,
    famous_defs: &FamousDefs<'_, '_>,
) -> Option<ReferenceConversionType> {
    let type_argument = ty.type_arguments().next()?;

    ty.impls_trait(db, famous_defs.core_convert_AsRef()?, &[type_argument])
        .then_some(ReferenceConversionType::Dereferenced)
}

fn handle_option_as_ref(
    ty: &hir::Type,
    db: &dyn HirDatabase,
    famous_defs: &FamousDefs<'_, '_>,
) -> Option<ReferenceConversionType> {
    if ty.as_adt() == famous_defs.core_option_Option()?.ty(db).as_adt() {
        Some(ReferenceConversionType::Option)
    } else {
        None
    }
}

fn handle_result_as_ref(
    ty: &hir::Type,
    db: &dyn HirDatabase,
    famous_defs: &FamousDefs<'_, '_>,
) -> Option<ReferenceConversionType> {
    if ty.as_adt() == famous_defs.core_result_Result()?.ty(db).as_adt() {
        Some(ReferenceConversionType::Result)
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
        if let Some(ast::Pat::IdentPat(pat)) = param.pat() {
            if let Some(name) = pat.name() {
                let name = name.to_string();
                let expr = make::expr_path(make::ext::ident_path(&name));
                args.push(expr);
            }
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
