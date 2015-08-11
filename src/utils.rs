use rustc::lint::{Context, Lint, Level};
use syntax::ast::{DefId, Expr, Name, NodeId, Path};
use syntax::codemap::{ExpnInfo, Span};
use syntax::ptr::P;
use rustc::ast_map::Node::NodeExpr;
use rustc::middle::ty;
use std::borrow::{Cow, IntoCow};
use std::convert::From;

/// returns true if the macro that expanded the crate was outside of
/// the current crate or was a compiler plugin
pub fn in_macro(cx: &Context, opt_info: Option<&ExpnInfo>) -> bool {
    // no ExpnInfo = no macro
    opt_info.map_or(false, |info| {
        // no span for the callee = external macro
        info.callee.span.map_or(true, |span| {
            // no snippet = external macro or compiler-builtin expansion
            cx.sess().codemap().span_to_snippet(span).ok().map_or(true, |code|
                // macro doesn't start with "macro_rules"
                // = compiler plugin
                !code.starts_with("macro_rules")
            )
        })
    })
}

/// invokes in_macro with the expansion info of the given span
pub fn in_external_macro(cx: &Context, span: Span) -> bool {
    cx.sess().codemap().with_expn_info(span.expn_id,
            |info| in_macro(cx, info))
}

/// check if a DefId's path matches the given absolute type path
/// usage e.g. with
/// `match_def_path(cx, id, &["core", "option", "Option"])`
pub fn match_def_path(cx: &Context, def_id: DefId, path: &[&str]) -> bool {
    cx.tcx.with_path(def_id, |iter| iter.map(|elem| elem.name())
        .zip(path.iter()).all(|(nm, p)| &nm.as_str() == p))
}

/// match a Path against a slice of segment string literals, e.g.
/// `match_path(path, &["std", "rt", "begin_unwind"])`
pub fn match_path(path: &Path, segments: &[&str]) -> bool {
    path.segments.iter().rev().zip(segments.iter().rev()).all(
        |(a,b)| &a.identifier.name.as_str() == b)
}

/// convert a span to a code snippet if available, otherwise use default, e.g.
/// `snippet(cx, expr.span, "..")`
pub fn snippet<'a>(cx: &Context, span: Span, default: &'a str) -> Cow<'a, str> {
    cx.sess().codemap().span_to_snippet(span).map(From::from).unwrap_or(Cow::Borrowed(default))
}

/// get a parent expr if any – this is useful to constrain a lint
pub fn get_parent_expr<'c>(cx: &'c Context, e: &Expr) -> Option<&'c Expr> {
    let map = &cx.tcx.map;
    let node_id : NodeId = e.id;
    let parent_id : NodeId = map.get_parent_node(node_id);
    if node_id == parent_id { return None; }
    map.find(parent_id).and_then(|node|
        if let NodeExpr(parent) = node { Some(parent) } else { None } )
}

/// dereference a P<T> and return a ref on the result
pub fn de_p<T>(p: &P<T>) -> &T { &*p }

#[cfg(not(feature="structured_logging"))]
pub fn span_lint(cx: &Context, lint: &'static Lint, sp: Span, msg: &str) {
    cx.span_lint(lint, sp, msg);
}

#[cfg(feature="structured_logging")]
pub fn span_lint(cx: &Context, lint: &'static Lint, sp: Span, msg: &str) {
    // lint.name / lint.desc is can give details of the lint
    // cx.sess().codemap() has all these nice functions for line/column/snippet details
    // http://doc.rust-lang.org/syntax/codemap/struct.CodeMap.html#method.span_to_string
    cx.span_lint(lint, sp, msg);
}

pub fn span_help_and_lint(cx: &Context, lint: &'static Lint, span: Span,
        msg: &str, help: &str) {
    span_lint(cx, lint, span, msg);
    if cx.current_level(lint) != Level::Allow {
        cx.sess().fileline_help(span, help);
    }
}
