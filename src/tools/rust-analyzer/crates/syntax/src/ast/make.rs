//! This module contains free-standing functions for creating AST fragments out
//! of smaller pieces.
//!
//! Note that all functions here intended to be stupid constructors, which just
//! assemble a finish node from immediate children. If you want to do something
//! smarter than that, it belongs to the `ext` submodule.
//!
//! Keep in mind that `from_text` functions should be kept private. The public
//! API should require to assemble every node piecewise. The trick of
//! `parse(format!())` we use internally is an implementation detail -- long
//! term, it will be replaced with direct tree manipulation.
use itertools::Itertools;
use stdx::{format_to, never};

use crate::{ast, utils::is_raw_identifier, AstNode, SourceFile, SyntaxKind, SyntaxToken};

/// While the parent module defines basic atomic "constructors", the `ext`
/// module defines shortcuts for common things.
///
/// It's named `ext` rather than `shortcuts` just to keep it short.
pub mod ext {
    use super::*;

    pub fn simple_ident_pat(name: ast::Name) -> ast::IdentPat {
        return from_text(&name.text());

        fn from_text(text: &str) -> ast::IdentPat {
            ast_from_text(&format!("fn f({text}: ())"))
        }
    }
    pub fn ident_path(ident: &str) -> ast::Path {
        path_unqualified(path_segment(name_ref(ident)))
    }

    pub fn path_from_idents<'a>(
        parts: impl std::iter::IntoIterator<Item = &'a str>,
    ) -> Option<ast::Path> {
        let mut iter = parts.into_iter();
        let base = ext::ident_path(iter.next()?);
        let path = iter.fold(base, |base, s| {
            let path = ext::ident_path(s);
            path_concat(base, path)
        });
        Some(path)
    }

    pub fn field_from_idents<'a>(
        parts: impl std::iter::IntoIterator<Item = &'a str>,
    ) -> Option<ast::Expr> {
        let mut iter = parts.into_iter();
        let base = expr_path(ext::ident_path(iter.next()?));
        let expr = iter.fold(base, expr_field);
        Some(expr)
    }

    pub fn expr_unreachable() -> ast::Expr {
        expr_from_text("unreachable!()")
    }
    pub fn expr_todo() -> ast::Expr {
        expr_from_text("todo!()")
    }
    pub fn expr_ty_default(ty: &ast::Type) -> ast::Expr {
        expr_from_text(&format!("{ty}::default()"))
    }
    pub fn expr_ty_new(ty: &ast::Type) -> ast::Expr {
        expr_from_text(&format!("{ty}::new()"))
    }

    pub fn zero_number() -> ast::Expr {
        expr_from_text("0")
    }
    pub fn zero_float() -> ast::Expr {
        expr_from_text("0.0")
    }
    pub fn empty_str() -> ast::Expr {
        expr_from_text(r#""""#)
    }
    pub fn empty_char() -> ast::Expr {
        expr_from_text("'\x00'")
    }
    pub fn default_bool() -> ast::Expr {
        expr_from_text("false")
    }
    pub fn option_none() -> ast::Expr {
        expr_from_text("None")
    }
    pub fn empty_block_expr() -> ast::BlockExpr {
        block_expr(None, None)
    }

    pub fn ty_name(name: ast::Name) -> ast::Type {
        ty_path(ident_path(&name.to_string()))
    }
    pub fn ty_bool() -> ast::Type {
        ty_path(ident_path("bool"))
    }
    pub fn ty_option(t: ast::Type) -> ast::Type {
        ty_from_text(&format!("Option<{t}>"))
    }
    pub fn ty_result(t: ast::Type, e: ast::Type) -> ast::Type {
        ty_from_text(&format!("Result<{t}, {e}>"))
    }
}

pub fn name(name: &str) -> ast::Name {
    let raw_escape = raw_ident_esc(name);
    ast_from_text(&format!("mod {raw_escape}{name};"))
}
pub fn name_ref(name_ref: &str) -> ast::NameRef {
    let raw_escape = raw_ident_esc(name_ref);
    ast_from_text(&format!("fn f() {{ {raw_escape}{name_ref}; }}"))
}
fn raw_ident_esc(ident: &str) -> &'static str {
    if is_raw_identifier(ident) {
        "r#"
    } else {
        ""
    }
}

pub fn lifetime(text: &str) -> ast::Lifetime {
    let mut text = text;
    let tmp;
    if never!(!text.starts_with('\'')) {
        tmp = format!("'{text}");
        text = &tmp;
    }
    ast_from_text(&format!("fn f<{text}>() {{ }}"))
}

// FIXME: replace stringly-typed constructor with a family of typed ctors, a-la
// `expr_xxx`.
pub fn ty(text: &str) -> ast::Type {
    ty_from_text(text)
}
pub fn ty_placeholder() -> ast::Type {
    ty_from_text("_")
}
pub fn ty_unit() -> ast::Type {
    ty_from_text("()")
}
pub fn ty_tuple(types: impl IntoIterator<Item = ast::Type>) -> ast::Type {
    let mut count: usize = 0;
    let mut contents = types.into_iter().inspect(|_| count += 1).join(", ");
    if count == 1 {
        contents.push(',');
    }

    ty_from_text(&format!("({contents})"))
}
pub fn ty_ref(target: ast::Type, exclusive: bool) -> ast::Type {
    ty_from_text(&if exclusive { format!("&mut {target}") } else { format!("&{target}") })
}
pub fn ty_path(path: ast::Path) -> ast::Type {
    ty_from_text(&path.to_string())
}
fn ty_from_text(text: &str) -> ast::Type {
    ast_from_text(&format!("type _T = {text};"))
}

pub fn ty_alias(
    ident: &str,
    generic_param_list: Option<ast::GenericParamList>,
    type_param_bounds: Option<ast::TypeParam>,
    where_clause: Option<ast::WhereClause>,
    assignment: Option<(ast::Type, Option<ast::WhereClause>)>,
) -> ast::TypeAlias {
    let mut s = String::new();
    s.push_str(&format!("type {}", ident));

    if let Some(list) = generic_param_list {
        s.push_str(&list.to_string());
    }

    if let Some(list) = type_param_bounds {
        s.push_str(&format!(" : {}", &list));
    }

    if let Some(cl) = where_clause {
        s.push_str(&format!(" {}", &cl.to_string()));
    }

    if let Some(exp) = assignment {
        if let Some(cl) = exp.1 {
            s.push_str(&format!(" = {} {}", &exp.0.to_string(), &cl.to_string()));
        } else {
            s.push_str(&format!(" = {}", &exp.0.to_string()));
        }
    }

    s.push(';');
    ast_from_text(&s)
}

pub fn assoc_item_list() -> ast::AssocItemList {
    ast_from_text("impl C for D {}")
}

fn merge_gen_params(
    ps: Option<ast::GenericParamList>,
    bs: Option<ast::GenericParamList>,
) -> Option<ast::GenericParamList> {
    match (ps, bs) {
        (None, None) => None,
        (None, Some(bs)) => Some(bs),
        (Some(ps), None) => Some(ps),
        (Some(ps), Some(bs)) => {
            for b in bs.generic_params() {
                ps.add_generic_param(b);
            }
            Some(ps)
        }
    }
}

pub fn impl_(
    generic_params: Option<ast::GenericParamList>,
    generic_args: Option<ast::GenericParamList>,
    path_type: ast::Type,
    where_clause: Option<ast::WhereClause>,
    body: Option<Vec<either::Either<ast::Attr, ast::AssocItem>>>,
) -> ast::Impl {
    let (gen_params, tr_gen_args) = match (generic_params, generic_args) {
        (None, None) => (String::new(), String::new()),
        (None, Some(args)) => (String::new(), args.to_generic_args().to_string()),
        (Some(params), None) => (params.to_string(), params.to_generic_args().to_string()),
        (Some(params), Some(args)) => match merge_gen_params(Some(params.clone()), Some(args)) {
            Some(merged) => (params.to_string(), merged.to_generic_args().to_string()),
            None => (params.to_string(), String::new()),
        },
    };

    let where_clause = match where_clause {
        Some(pr) => pr.to_string(),
        None => " ".to_string(),
    };

    let body = match body {
        Some(bd) => bd.iter().map(|elem| elem.to_string()).join(""),
        None => String::new(),
    };

    ast_from_text(&format!("impl{gen_params} {path_type}{tr_gen_args}{where_clause}{{{}}}", body))
}

// FIXME : We must make *_gen_args' type ast::GenericArgList but in order to do so we must implement in `edit_in_place.rs`
// `add_generic_arg()` just like `add_generic_param()`
// is implemented for `ast::GenericParamList`
pub fn impl_trait(
    is_unsafe: bool,
    trait_gen_params: Option<ast::GenericParamList>,
    trait_gen_args: Option<ast::GenericParamList>,
    type_gen_params: Option<ast::GenericParamList>,
    type_gen_args: Option<ast::GenericParamList>,
    is_negative: bool,
    path_type: ast::Type,
    ty: ast::Type,
    trait_where_clause: Option<ast::WhereClause>,
    ty_where_clause: Option<ast::WhereClause>,
    body: Option<Vec<either::Either<ast::Attr, ast::AssocItem>>>,
) -> ast::Impl {
    let is_unsafe = if is_unsafe { "unsafe " } else { "" };
    let ty_gen_args = match merge_gen_params(type_gen_params.clone(), type_gen_args) {
        Some(pars) => pars.to_generic_args().to_string(),
        None => String::new(),
    };

    let tr_gen_args = match merge_gen_params(trait_gen_params.clone(), trait_gen_args) {
        Some(pars) => pars.to_generic_args().to_string(),
        None => String::new(),
    };

    let gen_params = match merge_gen_params(trait_gen_params, type_gen_params) {
        Some(pars) => pars.to_string(),
        None => String::new(),
    };

    let is_negative = if is_negative { "! " } else { "" };

    let where_clause = match (ty_where_clause, trait_where_clause) {
        (None, None) => " ".to_string(),
        (None, Some(tr)) => format!("\n{}\n", tr).to_string(),
        (Some(ty), None) => format!("\n{}\n", ty).to_string(),
        (Some(ty), Some(tr)) => {
            let updated = ty.clone_for_update();
            tr.predicates().for_each(|p| {
                ty.add_predicate(p);
            });
            format!("\n{}\n", updated).to_string()
        }
    };

    let body = match body {
        Some(bd) => bd.iter().map(|elem| elem.to_string()).join(""),
        None => String::new(),
    };

    ast_from_text(&format!("{is_unsafe}impl{gen_params} {is_negative}{path_type}{tr_gen_args} for {ty}{ty_gen_args}{where_clause}{{{}}}" , body))
}

pub fn impl_trait_type(bounds: ast::TypeBoundList) -> ast::ImplTraitType {
    ast_from_text(&format!("fn f(x: impl {bounds}) {{}}"))
}

pub fn path_segment(name_ref: ast::NameRef) -> ast::PathSegment {
    ast_from_text(&format!("type __ = {name_ref};"))
}

pub fn path_segment_ty(type_ref: ast::Type, trait_ref: Option<ast::PathType>) -> ast::PathSegment {
    let text = match trait_ref {
        Some(trait_ref) => format!("fn f(x: <{type_ref} as {trait_ref}>) {{}}"),
        None => format!("fn f(x: <{type_ref}>) {{}}"),
    };
    ast_from_text(&text)
}

pub fn path_segment_self() -> ast::PathSegment {
    ast_from_text("use self;")
}

pub fn path_segment_super() -> ast::PathSegment {
    ast_from_text("use super;")
}

pub fn path_segment_crate() -> ast::PathSegment {
    ast_from_text("use crate;")
}

pub fn path_unqualified(segment: ast::PathSegment) -> ast::Path {
    ast_from_text(&format!("type __ = {segment};"))
}

pub fn path_qualified(qual: ast::Path, segment: ast::PathSegment) -> ast::Path {
    ast_from_text(&format!("{qual}::{segment}"))
}
// FIXME: path concatenation operation doesn't make sense as AST op.
pub fn path_concat(first: ast::Path, second: ast::Path) -> ast::Path {
    ast_from_text(&format!("type __ = {first}::{second};"))
}

pub fn path_from_segments(
    segments: impl IntoIterator<Item = ast::PathSegment>,
    is_abs: bool,
) -> ast::Path {
    let segments = segments.into_iter().map(|it| it.syntax().clone()).join("::");
    ast_from_text(&if is_abs {
        format!("fn f(x: ::{segments}) {{}}")
    } else {
        format!("fn f(x: {segments}) {{}}")
    })
}

pub fn join_paths(paths: impl IntoIterator<Item = ast::Path>) -> ast::Path {
    let paths = paths.into_iter().map(|it| it.syntax().clone()).join("::");
    ast_from_text(&format!("type __ = {paths};"))
}

// FIXME: should not be pub
pub fn path_from_text(text: &str) -> ast::Path {
    ast_from_text(&format!("fn main() {{ let test = {text}; }}"))
}

pub fn use_tree_glob() -> ast::UseTree {
    ast_from_text("use *;")
}
pub fn use_tree(
    path: ast::Path,
    use_tree_list: Option<ast::UseTreeList>,
    alias: Option<ast::Rename>,
    add_star: bool,
) -> ast::UseTree {
    let mut buf = "use ".to_string();
    buf += &path.syntax().to_string();
    if let Some(use_tree_list) = use_tree_list {
        format_to!(buf, "::{use_tree_list}");
    }
    if add_star {
        buf += "::*";
    }

    if let Some(alias) = alias {
        format_to!(buf, " {alias}");
    }
    ast_from_text(&buf)
}

pub fn use_tree_list(use_trees: impl IntoIterator<Item = ast::UseTree>) -> ast::UseTreeList {
    let use_trees = use_trees.into_iter().map(|it| it.syntax().clone()).join(", ");
    ast_from_text(&format!("use {{{use_trees}}};"))
}

pub fn use_(visibility: Option<ast::Visibility>, use_tree: ast::UseTree) -> ast::Use {
    let visibility = match visibility {
        None => String::new(),
        Some(it) => format!("{it} "),
    };
    ast_from_text(&format!("{visibility}use {use_tree};"))
}

pub fn record_expr(path: ast::Path, fields: ast::RecordExprFieldList) -> ast::RecordExpr {
    ast_from_text(&format!("fn f() {{ {path} {fields} }}"))
}

pub fn record_expr_field_list(
    fields: impl IntoIterator<Item = ast::RecordExprField>,
) -> ast::RecordExprFieldList {
    let fields = fields.into_iter().join(", ");
    ast_from_text(&format!("fn f() {{ S {{ {fields} }} }}"))
}

pub fn record_expr_field(name: ast::NameRef, expr: Option<ast::Expr>) -> ast::RecordExprField {
    return match expr {
        Some(expr) => from_text(&format!("{name}: {expr}")),
        None => from_text(&name.to_string()),
    };

    fn from_text(text: &str) -> ast::RecordExprField {
        ast_from_text(&format!("fn f() {{ S {{ {text}, }} }}"))
    }
}

pub fn record_field(
    visibility: Option<ast::Visibility>,
    name: ast::Name,
    ty: ast::Type,
) -> ast::RecordField {
    let visibility = match visibility {
        None => String::new(),
        Some(it) => format!("{it} "),
    };
    ast_from_text(&format!("struct S {{ {visibility}{name}: {ty}, }}"))
}

// TODO
pub fn block_expr(
    stmts: impl IntoIterator<Item = ast::Stmt>,
    tail_expr: Option<ast::Expr>,
) -> ast::BlockExpr {
    let mut buf = "{\n".to_string();
    for stmt in stmts.into_iter() {
        format_to!(buf, "    {stmt}\n");
    }
    if let Some(tail_expr) = tail_expr {
        format_to!(buf, "    {tail_expr}\n");
    }
    buf += "}";
    ast_from_text(&format!("fn f() {buf}"))
}

pub fn tail_only_block_expr(tail_expr: ast::Expr) -> ast::BlockExpr {
    ast_from_text(&format!("fn f() {{ {tail_expr} }}"))
}

/// Ideally this function wouldn't exist since it involves manual indenting.
/// It differs from `make::block_expr` by also supporting comments and whitespace.
///
/// FIXME: replace usages of this with the mutable syntax tree API
pub fn hacky_block_expr(
    elements: impl IntoIterator<Item = crate::SyntaxElement>,
    tail_expr: Option<ast::Expr>,
) -> ast::BlockExpr {
    let mut buf = "{\n".to_string();
    for node_or_token in elements.into_iter() {
        match node_or_token {
            rowan::NodeOrToken::Node(n) => format_to!(buf, "    {n}\n"),
            rowan::NodeOrToken::Token(t) => {
                let kind = t.kind();
                if kind == SyntaxKind::COMMENT {
                    format_to!(buf, "    {t}\n")
                } else if kind == SyntaxKind::WHITESPACE {
                    let content = t.text().trim_matches(|c| c != '\n');
                    if !content.is_empty() {
                        format_to!(buf, "{}", &content[1..])
                    }
                }
            }
        }
    }
    if let Some(tail_expr) = tail_expr {
        format_to!(buf, "    {tail_expr}\n");
    }
    buf += "}";
    ast_from_text(&format!("fn f() {buf}"))
}

pub fn expr_unit() -> ast::Expr {
    expr_from_text("()")
}
pub fn expr_literal(text: &str) -> ast::Literal {
    assert_eq!(text.trim(), text);
    ast_from_text(&format!("fn f() {{ let _ = {text}; }}"))
}

pub fn expr_empty_block() -> ast::Expr {
    expr_from_text("{}")
}
pub fn expr_path(path: ast::Path) -> ast::Expr {
    expr_from_text(&path.to_string())
}
pub fn expr_continue(label: Option<ast::Lifetime>) -> ast::Expr {
    match label {
        Some(label) => expr_from_text(&format!("continue {label}")),
        None => expr_from_text("continue"),
    }
}
// Consider `op: SyntaxKind` instead for nicer syntax at the call-site?
pub fn expr_bin_op(lhs: ast::Expr, op: ast::BinaryOp, rhs: ast::Expr) -> ast::Expr {
    expr_from_text(&format!("{lhs} {op} {rhs}"))
}
pub fn expr_break(label: Option<ast::Lifetime>, expr: Option<ast::Expr>) -> ast::Expr {
    let mut s = String::from("break");

    if let Some(label) = label {
        format_to!(s, " {label}");
    }

    if let Some(expr) = expr {
        format_to!(s, " {expr}");
    }

    expr_from_text(&s)
}
pub fn expr_return(expr: Option<ast::Expr>) -> ast::Expr {
    match expr {
        Some(expr) => expr_from_text(&format!("return {expr}")),
        None => expr_from_text("return"),
    }
}
pub fn expr_try(expr: ast::Expr) -> ast::Expr {
    expr_from_text(&format!("{expr}?"))
}
pub fn expr_await(expr: ast::Expr) -> ast::Expr {
    expr_from_text(&format!("{expr}.await"))
}
pub fn expr_match(expr: ast::Expr, match_arm_list: ast::MatchArmList) -> ast::Expr {
    expr_from_text(&format!("match {expr} {match_arm_list}"))
}
pub fn expr_if(
    condition: ast::Expr,
    then_branch: ast::BlockExpr,
    else_branch: Option<ast::ElseBranch>,
) -> ast::Expr {
    let else_branch = match else_branch {
        Some(ast::ElseBranch::Block(block)) => format!("else {block}"),
        Some(ast::ElseBranch::IfExpr(if_expr)) => format!("else {if_expr}"),
        None => String::new(),
    };
    expr_from_text(&format!("if {condition} {then_branch} {else_branch}"))
}
pub fn expr_for_loop(pat: ast::Pat, expr: ast::Expr, block: ast::BlockExpr) -> ast::Expr {
    expr_from_text(&format!("for {pat} in {expr} {block}"))
}

pub fn expr_loop(block: ast::BlockExpr) -> ast::Expr {
    expr_from_text(&format!("loop {block}"))
}

pub fn expr_prefix(op: SyntaxKind, expr: ast::Expr) -> ast::Expr {
    let token = token(op);
    expr_from_text(&format!("{token}{expr}"))
}
pub fn expr_call(f: ast::Expr, arg_list: ast::ArgList) -> ast::Expr {
    expr_from_text(&format!("{f}{arg_list}"))
}
pub fn expr_method_call(
    receiver: ast::Expr,
    method: ast::NameRef,
    arg_list: ast::ArgList,
) -> ast::Expr {
    expr_from_text(&format!("{receiver}.{method}{arg_list}"))
}
pub fn expr_macro_call(f: ast::Expr, arg_list: ast::ArgList) -> ast::Expr {
    expr_from_text(&format!("{f}!{arg_list}"))
}
pub fn expr_ref(expr: ast::Expr, exclusive: bool) -> ast::Expr {
    expr_from_text(&if exclusive { format!("&mut {expr}") } else { format!("&{expr}") })
}
pub fn expr_closure(pats: impl IntoIterator<Item = ast::Param>, expr: ast::Expr) -> ast::Expr {
    let params = pats.into_iter().join(", ");
    expr_from_text(&format!("|{params}| {expr}"))
}
pub fn expr_field(receiver: ast::Expr, field: &str) -> ast::Expr {
    expr_from_text(&format!("{receiver}.{field}"))
}
pub fn expr_paren(expr: ast::Expr) -> ast::Expr {
    expr_from_text(&format!("({expr})"))
}
pub fn expr_tuple(elements: impl IntoIterator<Item = ast::Expr>) -> ast::Expr {
    let expr = elements.into_iter().format(", ");
    expr_from_text(&format!("({expr})"))
}
pub fn expr_assignment(lhs: ast::Expr, rhs: ast::Expr) -> ast::Expr {
    expr_from_text(&format!("{lhs} = {rhs}"))
}
fn expr_from_text(text: &str) -> ast::Expr {
    ast_from_text(&format!("const C: () = {text};"))
}
pub fn expr_let(pattern: ast::Pat, expr: ast::Expr) -> ast::LetExpr {
    ast_from_text(&format!("const _: () = while let {pattern} = {expr} {{}};"))
}

pub fn arg_list(args: impl IntoIterator<Item = ast::Expr>) -> ast::ArgList {
    let args = args.into_iter().format(", ");
    ast_from_text(&format!("fn main() {{ ()({args}) }}"))
}

pub fn ident_pat(ref_: bool, mut_: bool, name: ast::Name) -> ast::IdentPat {
    let mut s = String::from("fn f(");
    if ref_ {
        s.push_str("ref ");
    }
    if mut_ {
        s.push_str("mut ");
    }
    format_to!(s, "{name}");
    s.push_str(": ())");
    ast_from_text(&s)
}

pub fn wildcard_pat() -> ast::WildcardPat {
    return from_text("_");

    fn from_text(text: &str) -> ast::WildcardPat {
        ast_from_text(&format!("fn f({text}: ())"))
    }
}

pub fn literal_pat(lit: &str) -> ast::LiteralPat {
    return from_text(lit);

    fn from_text(text: &str) -> ast::LiteralPat {
        ast_from_text(&format!("fn f() {{ match x {{ {text} => {{}} }} }}"))
    }
}

pub fn slice_pat(pats: impl IntoIterator<Item = ast::Pat>) -> ast::SlicePat {
    let pats_str = pats.into_iter().join(", ");
    return from_text(&format!("[{pats_str}]"));

    fn from_text(text: &str) -> ast::SlicePat {
        ast_from_text(&format!("fn f() {{ match () {{{text} => ()}} }}"))
    }
}

/// Creates a tuple of patterns from an iterator of patterns.
///
/// Invariant: `pats` must be length > 0
pub fn tuple_pat(pats: impl IntoIterator<Item = ast::Pat>) -> ast::TuplePat {
    let mut count: usize = 0;
    let mut pats_str = pats.into_iter().inspect(|_| count += 1).join(", ");
    if count == 1 {
        pats_str.push(',');
    }
    return from_text(&format!("({pats_str})"));

    fn from_text(text: &str) -> ast::TuplePat {
        ast_from_text(&format!("fn f({text}: ())"))
    }
}

pub fn tuple_struct_pat(
    path: ast::Path,
    pats: impl IntoIterator<Item = ast::Pat>,
) -> ast::TupleStructPat {
    let pats_str = pats.into_iter().join(", ");
    return from_text(&format!("{path}({pats_str})"));

    fn from_text(text: &str) -> ast::TupleStructPat {
        ast_from_text(&format!("fn f({text}: ())"))
    }
}

pub fn record_pat(path: ast::Path, pats: impl IntoIterator<Item = ast::Pat>) -> ast::RecordPat {
    let pats_str = pats.into_iter().join(", ");
    return from_text(&format!("{path} {{ {pats_str} }}"));

    fn from_text(text: &str) -> ast::RecordPat {
        ast_from_text(&format!("fn f({text}: ())"))
    }
}

pub fn record_pat_with_fields(path: ast::Path, fields: ast::RecordPatFieldList) -> ast::RecordPat {
    ast_from_text(&format!("fn f({path} {fields}: ()))"))
}

pub fn record_pat_field_list(
    fields: impl IntoIterator<Item = ast::RecordPatField>,
) -> ast::RecordPatFieldList {
    let fields = fields.into_iter().join(", ");
    ast_from_text(&format!("fn f(S {{ {fields} }}: ()))"))
}

pub fn record_pat_field(name_ref: ast::NameRef, pat: ast::Pat) -> ast::RecordPatField {
    ast_from_text(&format!("fn f(S {{ {name_ref}: {pat} }}: ()))"))
}

pub fn record_pat_field_shorthand(name_ref: ast::NameRef) -> ast::RecordPatField {
    ast_from_text(&format!("fn f(S {{ {name_ref} }}: ()))"))
}

/// Returns a `BindPat` if the path has just one segment, a `PathPat` otherwise.
pub fn path_pat(path: ast::Path) -> ast::Pat {
    return from_text(&path.to_string());
    fn from_text(text: &str) -> ast::Pat {
        ast_from_text(&format!("fn f({text}: ())"))
    }
}

pub fn match_arm(
    pats: impl IntoIterator<Item = ast::Pat>,
    guard: Option<ast::Expr>,
    expr: ast::Expr,
) -> ast::MatchArm {
    let pats_str = pats.into_iter().join(" | ");
    return match guard {
        Some(guard) => from_text(&format!("{pats_str} if {guard} => {expr}")),
        None => from_text(&format!("{pats_str} => {expr}")),
    };

    fn from_text(text: &str) -> ast::MatchArm {
        ast_from_text(&format!("fn f() {{ match () {{{text}}} }}"))
    }
}

pub fn match_arm_with_guard(
    pats: impl IntoIterator<Item = ast::Pat>,
    guard: ast::Expr,
    expr: ast::Expr,
) -> ast::MatchArm {
    let pats_str = pats.into_iter().join(" | ");
    return from_text(&format!("{pats_str} if {guard} => {expr}"));

    fn from_text(text: &str) -> ast::MatchArm {
        ast_from_text(&format!("fn f() {{ match () {{{text}}} }}"))
    }
}

pub fn match_arm_list(arms: impl IntoIterator<Item = ast::MatchArm>) -> ast::MatchArmList {
    let arms_str = arms
        .into_iter()
        .map(|arm| {
            let needs_comma = arm.expr().map_or(true, |it| !it.is_block_like());
            let comma = if needs_comma { "," } else { "" };
            let arm = arm.syntax();
            format!("    {arm}{comma}\n")
        })
        .collect::<String>();
    return from_text(&arms_str);

    fn from_text(text: &str) -> ast::MatchArmList {
        ast_from_text(&format!("fn f() {{ match () {{\n{text}}} }}"))
    }
}

pub fn where_pred(
    path: ast::Path,
    bounds: impl IntoIterator<Item = ast::TypeBound>,
) -> ast::WherePred {
    let bounds = bounds.into_iter().join(" + ");
    return from_text(&format!("{path}: {bounds}"));

    fn from_text(text: &str) -> ast::WherePred {
        ast_from_text(&format!("fn f() where {text} {{ }}"))
    }
}

pub fn where_clause(preds: impl IntoIterator<Item = ast::WherePred>) -> ast::WhereClause {
    let preds = preds.into_iter().join(", ");
    return from_text(preds.as_str());

    fn from_text(text: &str) -> ast::WhereClause {
        ast_from_text(&format!("fn f() where {text} {{ }}"))
    }
}

pub fn let_stmt(
    pattern: ast::Pat,
    ty: Option<ast::Type>,
    initializer: Option<ast::Expr>,
) -> ast::LetStmt {
    let mut text = String::new();
    format_to!(text, "let {pattern}");
    if let Some(ty) = ty {
        format_to!(text, ": {ty}");
    }
    match initializer {
        Some(it) => format_to!(text, " = {it};"),
        None => format_to!(text, ";"),
    };
    ast_from_text(&format!("fn f() {{ {text} }}"))
}

pub fn let_else_stmt(
    pattern: ast::Pat,
    ty: Option<ast::Type>,
    expr: ast::Expr,
    diverging: ast::BlockExpr,
) -> ast::LetStmt {
    let mut text = String::new();
    format_to!(text, "let {pattern}");
    if let Some(ty) = ty {
        format_to!(text, ": {ty}");
    }
    format_to!(text, " = {expr} else {diverging};");
    ast_from_text(&format!("fn f() {{ {text} }}"))
}

pub fn expr_stmt(expr: ast::Expr) -> ast::ExprStmt {
    let semi = if expr.is_block_like() { "" } else { ";" };
    ast_from_text(&format!("fn f() {{ {expr}{semi} (); }}"))
}

pub fn item_const(
    visibility: Option<ast::Visibility>,
    name: ast::Name,
    ty: ast::Type,
    expr: ast::Expr,
) -> ast::Const {
    let visibility = match visibility {
        None => String::new(),
        Some(it) => format!("{it} "),
    };
    ast_from_text(&format!("{visibility} const {name}: {ty} = {expr};"))
}

pub fn param(pat: ast::Pat, ty: ast::Type) -> ast::Param {
    ast_from_text(&format!("fn f({pat}: {ty}) {{ }}"))
}

pub fn self_param() -> ast::SelfParam {
    ast_from_text("fn f(&self) { }")
}

pub fn ret_type(ty: ast::Type) -> ast::RetType {
    ast_from_text(&format!("fn f() -> {ty} {{ }}"))
}

pub fn param_list(
    self_param: Option<ast::SelfParam>,
    pats: impl IntoIterator<Item = ast::Param>,
) -> ast::ParamList {
    let args = pats.into_iter().join(", ");
    let list = match self_param {
        Some(self_param) if args.is_empty() => format!("fn f({self_param}) {{ }}"),
        Some(self_param) => format!("fn f({self_param}, {args}) {{ }}"),
        None => format!("fn f({args}) {{ }}"),
    };
    ast_from_text(&list)
}

pub fn type_bound(bound: &str) -> ast::TypeBound {
    ast_from_text(&format!("fn f<T: {bound}>() {{ }}"))
}

pub fn type_bound_list(
    bounds: impl IntoIterator<Item = ast::TypeBound>,
) -> Option<ast::TypeBoundList> {
    let bounds = bounds.into_iter().map(|it| it.to_string()).unique().join(" + ");
    if bounds.is_empty() {
        return None;
    }
    Some(ast_from_text(&format!("fn f<T: {bounds}>() {{ }}")))
}

pub fn type_param(name: ast::Name, bounds: Option<ast::TypeBoundList>) -> ast::TypeParam {
    let bounds = bounds.map_or_else(String::new, |it| format!(": {it}"));
    ast_from_text(&format!("fn f<{name}{bounds}>() {{ }}"))
}

pub fn lifetime_param(lifetime: ast::Lifetime) -> ast::LifetimeParam {
    ast_from_text(&format!("fn f<{lifetime}>() {{ }}"))
}

pub fn generic_param_list(
    pats: impl IntoIterator<Item = ast::GenericParam>,
) -> ast::GenericParamList {
    let args = pats.into_iter().join(", ");
    ast_from_text(&format!("fn f<{args}>() {{ }}"))
}

pub fn type_arg(ty: ast::Type) -> ast::TypeArg {
    ast_from_text(&format!("const S: T<{ty}> = ();"))
}

pub fn lifetime_arg(lifetime: ast::Lifetime) -> ast::LifetimeArg {
    ast_from_text(&format!("const S: T<{lifetime}> = ();"))
}

pub(crate) fn generic_arg_list(
    args: impl IntoIterator<Item = ast::GenericArg>,
) -> ast::GenericArgList {
    let args = args.into_iter().join(", ");
    ast_from_text(&format!("const S: T<{args}> = ();"))
}

pub fn visibility_pub_crate() -> ast::Visibility {
    ast_from_text("pub(crate) struct S")
}

pub fn visibility_pub() -> ast::Visibility {
    ast_from_text("pub struct S")
}

pub fn tuple_field_list(fields: impl IntoIterator<Item = ast::TupleField>) -> ast::TupleFieldList {
    let fields = fields.into_iter().join(", ");
    ast_from_text(&format!("struct f({fields});"))
}

pub fn record_field_list(
    fields: impl IntoIterator<Item = ast::RecordField>,
) -> ast::RecordFieldList {
    let fields = fields.into_iter().join(", ");
    ast_from_text(&format!("struct f {{ {fields} }}"))
}

pub fn tuple_field(visibility: Option<ast::Visibility>, ty: ast::Type) -> ast::TupleField {
    let visibility = match visibility {
        None => String::new(),
        Some(it) => format!("{it} "),
    };
    ast_from_text(&format!("struct f({visibility}{ty});"))
}

pub fn variant(name: ast::Name, field_list: Option<ast::FieldList>) -> ast::Variant {
    let field_list = match field_list {
        None => String::new(),
        Some(it) => match it {
            ast::FieldList::RecordFieldList(record) => format!(" {record}"),
            ast::FieldList::TupleFieldList(tuple) => format!("{tuple}"),
        },
    };
    ast_from_text(&format!("enum f {{ {name}{field_list} }}"))
}

pub fn fn_(
    visibility: Option<ast::Visibility>,
    fn_name: ast::Name,
    type_params: Option<ast::GenericParamList>,
    where_clause: Option<ast::WhereClause>,
    params: ast::ParamList,
    body: ast::BlockExpr,
    ret_type: Option<ast::RetType>,
    is_async: bool,
    is_const: bool,
    is_unsafe: bool,
) -> ast::Fn {
    let type_params = match type_params {
        Some(type_params) => format!("{type_params}"),
        None => "".into(),
    };
    let where_clause = match where_clause {
        Some(it) => format!("{it} "),
        None => "".into(),
    };
    let ret_type = match ret_type {
        Some(ret_type) => format!("{ret_type} "),
        None => "".into(),
    };
    let visibility = match visibility {
        None => String::new(),
        Some(it) => format!("{it} "),
    };

    let async_literal = if is_async { "async " } else { "" };
    let const_literal = if is_const { "const " } else { "" };
    let unsafe_literal = if is_unsafe { "unsafe " } else { "" };

    ast_from_text(&format!(
        "{visibility}{async_literal}{const_literal}{unsafe_literal}fn {fn_name}{type_params}{params} {ret_type}{where_clause}{body}",
    ))
}
pub fn struct_(
    visibility: Option<ast::Visibility>,
    strukt_name: ast::Name,
    generic_param_list: Option<ast::GenericParamList>,
    field_list: ast::FieldList,
) -> ast::Struct {
    let semicolon = if matches!(field_list, ast::FieldList::TupleFieldList(_)) { ";" } else { "" };
    let type_params = generic_param_list.map_or_else(String::new, |it| it.to_string());
    let visibility = match visibility {
        None => String::new(),
        Some(it) => format!("{it} "),
    };

    ast_from_text(&format!("{visibility}struct {strukt_name}{type_params}{field_list}{semicolon}",))
}

#[track_caller]
fn ast_from_text<N: AstNode>(text: &str) -> N {
    let parse = SourceFile::parse(text);
    let node = match parse.tree().syntax().descendants().find_map(N::cast) {
        Some(it) => it,
        None => {
            let node = std::any::type_name::<N>();
            panic!("Failed to make ast node `{node}` from text {text}")
        }
    };
    let node = node.clone_subtree();
    assert_eq!(node.syntax().text_range().start(), 0.into());
    node
}

pub fn token(kind: SyntaxKind) -> SyntaxToken {
    tokens::SOURCE_FILE
        .tree()
        .syntax()
        .clone_for_update()
        .descendants_with_tokens()
        .filter_map(|it| it.into_token())
        .find(|it| it.kind() == kind)
        .unwrap_or_else(|| panic!("unhandled token: {kind:?}"))
}

pub mod tokens {
    use once_cell::sync::Lazy;

    use crate::{ast, AstNode, Parse, SourceFile, SyntaxKind::*, SyntaxToken};

    pub(super) static SOURCE_FILE: Lazy<Parse<SourceFile>> = Lazy::new(|| {
        SourceFile::parse(
            "const C: <()>::Item = (1 != 1, 2 == 2, 3 < 3, 4 <= 4, 5 > 5, 6 >= 6, !true, *p, &p , &mut p)\n;\n\n",
        )
    });

    pub fn single_space() -> SyntaxToken {
        SOURCE_FILE
            .tree()
            .syntax()
            .clone_for_update()
            .descendants_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == WHITESPACE && it.text() == " ")
            .unwrap()
    }

    pub fn whitespace(text: &str) -> SyntaxToken {
        assert!(text.trim().is_empty());
        let sf = SourceFile::parse(text).ok().unwrap();
        sf.syntax().clone_for_update().first_child_or_token().unwrap().into_token().unwrap()
    }

    pub fn doc_comment(text: &str) -> SyntaxToken {
        assert!(!text.trim().is_empty());
        let sf = SourceFile::parse(text).ok().unwrap();
        sf.syntax().first_child_or_token().unwrap().into_token().unwrap()
    }

    pub fn literal(text: &str) -> SyntaxToken {
        assert_eq!(text.trim(), text);
        let lit: ast::Literal = super::ast_from_text(&format!("fn f() {{ let _ = {text}; }}"));
        lit.syntax().first_child_or_token().unwrap().into_token().unwrap()
    }

    pub fn single_newline() -> SyntaxToken {
        let res = SOURCE_FILE
            .tree()
            .syntax()
            .clone_for_update()
            .descendants_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == WHITESPACE && it.text() == "\n")
            .unwrap();
        res.detach();
        res
    }

    pub fn blank_line() -> SyntaxToken {
        SOURCE_FILE
            .tree()
            .syntax()
            .clone_for_update()
            .descendants_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == WHITESPACE && it.text() == "\n\n")
            .unwrap()
    }

    pub struct WsBuilder(SourceFile);

    impl WsBuilder {
        pub fn new(text: &str) -> WsBuilder {
            WsBuilder(SourceFile::parse(text).ok().unwrap())
        }
        pub fn ws(&self) -> SyntaxToken {
            self.0.syntax().first_child_or_token().unwrap().into_token().unwrap()
        }
    }
}
