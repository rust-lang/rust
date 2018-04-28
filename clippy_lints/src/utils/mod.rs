use reexport::*;
use rustc::hir;
use rustc::hir::*;
use rustc::hir::def_id::{DefId, CRATE_DEF_INDEX};
use rustc::hir::def::Def;
use rustc::hir::intravisit::{NestedVisitorMap, Visitor};
use rustc::hir::map::Node;
use rustc::lint::{LateContext, Level, Lint, LintContext};
use rustc::session::Session;
use rustc::traits;
use rustc::ty::{self, Ty, TyCtxt, layout::{self, IntegerExt}};
use rustc_errors;
use std::borrow::Cow;
use std::env;
use std::mem;
use std::str::FromStr;
use std::rc::Rc;
use syntax::ast::{self, LitKind};
use syntax::attr;
use syntax::codemap::{CompilerDesugaringKind, ExpnFormat, ExpnInfo, Span, DUMMY_SP};
use syntax::errors::DiagnosticBuilder;
use syntax::ptr::P;
use syntax::symbol::keywords;

pub mod comparisons;
pub mod conf;
pub mod constants;
mod hir_utils;
pub mod paths;
pub mod sugg;
pub mod inspector;
pub mod internal_lints;
pub mod author;
pub mod ptr;
pub use self::hir_utils::{SpanlessEq, SpanlessHash};

pub type MethodArgs = HirVec<P<Expr>>;

pub mod higher;

/// Returns true if the two spans come from differing expansions (i.e. one is
/// from a macro and one
/// isn't).
pub fn differing_macro_contexts(lhs: Span, rhs: Span) -> bool {
    rhs.ctxt() != lhs.ctxt()
}

pub fn in_constant(cx: &LateContext, id: NodeId) -> bool {
    let parent_id = cx.tcx.hir.get_parent(id);
    match cx.tcx.hir.body_owner_kind(parent_id) {
        hir::BodyOwnerKind::Fn => false,
        hir::BodyOwnerKind::Const | hir::BodyOwnerKind::Static(..) => true,
    }
}

/// Returns true if this `expn_info` was expanded by any macro.
pub fn in_macro(span: Span) -> bool {
    span.ctxt().outer().expn_info().map_or(false, |info| {
        match info.callee.format {
            // don't treat range expressions desugared to structs as "in_macro"
            ExpnFormat::CompilerDesugaring(kind) => kind != CompilerDesugaringKind::DotFill,
            _ => true,
        }
    })
}

/// Returns true if `expn_info` was expanded by range expressions.
pub fn is_range_expression(span: Span) -> bool {
    span.ctxt().outer().expn_info().map_or(false, |info| {
        match info.callee.format {
            ExpnFormat::CompilerDesugaring(CompilerDesugaringKind::DotFill) => true,
            _ => false,
        }
    })
}

/// Returns true if the macro that expanded the crate was outside of the
/// current crate or was a
/// compiler plugin.
pub fn in_external_macro<'a, T: LintContext<'a>>(cx: &T, span: Span) -> bool {
    /// Invokes `in_macro` with the expansion info of the given span slightly
    /// heavy, try to use
    /// this after other checks have already happened.
    fn in_macro_ext<'a, T: LintContext<'a>>(cx: &T, info: &ExpnInfo) -> bool {
        // no ExpnInfo = no macro
        if let ExpnFormat::MacroAttribute(..) = info.callee.format {
            // these are all plugins
            return true;
        }
        // no span for the callee = external macro
        info.callee.span.map_or(true, |span| {
            // no snippet = external macro or compiler-builtin expansion
            cx.sess()
                .codemap()
                .span_to_snippet(span)
                .ok()
                .map_or(true, |code| !code.starts_with("macro_rules"))
        })
    }

    span.ctxt()
        .outer()
        .expn_info()
        .map_or(false, |info| in_macro_ext(cx, &info))
}

/// Check if a `DefId`'s path matches the given absolute type path usage.
///
/// # Examples
/// ```rust,ignore
/// match_def_path(cx.tcx, id, &["core", "option", "Option"])
/// ```
///
/// See also the `paths` module.
pub fn match_def_path(tcx: TyCtxt, def_id: DefId, path: &[&str]) -> bool {
    use syntax::symbol;

    struct AbsolutePathBuffer {
        names: Vec<symbol::LocalInternedString>,
    }

    impl ty::item_path::ItemPathBuffer for AbsolutePathBuffer {
        fn root_mode(&self) -> &ty::item_path::RootMode {
            const ABSOLUTE: &ty::item_path::RootMode = &ty::item_path::RootMode::Absolute;
            ABSOLUTE
        }

        fn push(&mut self, text: &str) {
            self.names.push(symbol::Symbol::intern(text).as_str());
        }
    }

    let mut apb = AbsolutePathBuffer { names: vec![] };

    tcx.push_item_path(&mut apb, def_id);

    apb.names.len() == path.len()
        && apb.names
            .into_iter()
            .zip(path.iter())
            .all(|(a, &b)| *a == *b)
}

/// Check if type is struct, enum or union type with given def path.
pub fn match_type(cx: &LateContext, ty: Ty, path: &[&str]) -> bool {
    match ty.sty {
        ty::TyAdt(adt, _) => match_def_path(cx.tcx, adt.did, path),
        _ => false,
    }
}

/// Check if the method call given in `expr` belongs to given type.
pub fn match_impl_method(cx: &LateContext, expr: &Expr, path: &[&str]) -> bool {
    let method_call = cx.tables.type_dependent_defs()[expr.hir_id];
    let trt_id = cx.tcx.impl_of_method(method_call.def_id());
    if let Some(trt_id) = trt_id {
        match_def_path(cx.tcx, trt_id, path)
    } else {
        false
    }
}

/// Check if the method call given in `expr` belongs to given trait.
pub fn match_trait_method(cx: &LateContext, expr: &Expr, path: &[&str]) -> bool {
    let method_call = cx.tables.type_dependent_defs()[expr.hir_id];
    let trt_id = cx.tcx.trait_of_item(method_call.def_id());
    if let Some(trt_id) = trt_id {
        match_def_path(cx.tcx, trt_id, path)
    } else {
        false
    }
}

/// Check if an expression references a variable of the given name.
pub fn match_var(expr: &Expr, var: Name) -> bool {
    if let ExprPath(QPath::Resolved(None, ref path)) = expr.node {
        if path.segments.len() == 1 && path.segments[0].name == var {
            return true;
        }
    }
    false
}


pub fn last_path_segment(path: &QPath) -> &PathSegment {
    match *path {
        QPath::Resolved(_, ref path) => path.segments
            .last()
            .expect("A path must have at least one segment"),
        QPath::TypeRelative(_, ref seg) => seg,
    }
}

pub fn single_segment_path(path: &QPath) -> Option<&PathSegment> {
    match *path {
        QPath::Resolved(_, ref path) if path.segments.len() == 1 => Some(&path.segments[0]),
        QPath::Resolved(..) => None,
        QPath::TypeRelative(_, ref seg) => Some(seg),
    }
}

/// Match a `Path` against a slice of segment string literals.
///
/// # Examples
/// ```rust,ignore
/// match_qpath(path, &["std", "rt", "begin_unwind"])
/// ```
pub fn match_qpath(path: &QPath, segments: &[&str]) -> bool {
    match *path {
        QPath::Resolved(_, ref path) => match_path(path, segments),
        QPath::TypeRelative(ref ty, ref segment) => match ty.node {
            TyPath(ref inner_path) => {
                !segments.is_empty() && match_qpath(inner_path, &segments[..(segments.len() - 1)])
                    && segment.name == segments[segments.len() - 1]
            },
            _ => false,
        },
    }
}

pub fn match_path(path: &Path, segments: &[&str]) -> bool {
    path.segments
        .iter()
        .rev()
        .zip(segments.iter().rev())
        .all(|(a, b)| a.name == *b)
}

/// Match a `Path` against a slice of segment string literals, e.g.
///
/// # Examples
/// ```rust,ignore
/// match_qpath(path, &["std", "rt", "begin_unwind"])
/// ```
pub fn match_path_ast(path: &ast::Path, segments: &[&str]) -> bool {
    path.segments
        .iter()
        .rev()
        .zip(segments.iter().rev())
        .all(|(a, b)| a.ident.name == *b)
}

/// Get the definition associated to a path.
pub fn path_to_def(cx: &LateContext, path: &[&str]) -> Option<def::Def> {
    let crates = cx.tcx.crates();
    let krate = crates
        .iter()
        .find(|&&krate| cx.tcx.crate_name(krate) == path[0]);
    if let Some(krate) = krate {
        let krate = DefId {
            krate: *krate,
            index: CRATE_DEF_INDEX,
        };
        let mut items = cx.tcx.item_children(krate);
        let mut path_it = path.iter().skip(1).peekable();

        loop {
            let segment = match path_it.next() {
                Some(segment) => segment,
                None => return None,
            };

            for item in mem::replace(&mut items, Rc::new(vec![])).iter() {
                if item.ident.name == *segment {
                    if path_it.peek().is_none() {
                        return Some(item.def);
                    }

                    items = cx.tcx.item_children(item.def.def_id());
                    break;
                }
            }
        }
    } else {
        None
    }
}

/// Convenience function to get the `DefId` of a trait by path.
pub fn get_trait_def_id(cx: &LateContext, path: &[&str]) -> Option<DefId> {
    let def = match path_to_def(cx, path) {
        Some(def) => def,
        None => return None,
    };

    match def {
        def::Def::Trait(trait_id) => Some(trait_id),
        _ => None,
    }
}

/// Check whether a type implements a trait.
/// See also `get_trait_def_id`.
pub fn implements_trait<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    ty: Ty<'tcx>,
    trait_id: DefId,
    ty_params: &[Ty<'tcx>],
) -> bool {
    let ty = cx.tcx.erase_regions(&ty);
    let obligation =
        cx.tcx
            .predicate_for_trait_def(cx.param_env, traits::ObligationCause::dummy(), trait_id, 0, ty, ty_params);
    cx.tcx.infer_ctxt().enter(|infcx| {
        traits::SelectionContext::new(&infcx).infcx().predicate_must_hold(&obligation)
    })
}

/// Check whether this type implements Drop.
pub fn has_drop(cx: &LateContext, expr: &Expr) -> bool {
    let struct_ty = cx.tables.expr_ty(expr);
    match struct_ty.ty_adt_def() {
        Some(def) => def.has_dtor(cx.tcx),
        _ => false,
    }
}

/// Resolve the definition of a node from its `HirId`.
pub fn resolve_node(cx: &LateContext, qpath: &QPath, id: HirId) -> def::Def {
    cx.tables.qpath_def(qpath, id)
}

/// Match an `Expr` against a chain of methods, and return the matched `Expr`s.
///
/// For example, if `expr` represents the `.baz()` in `foo.bar().baz()`,
/// `matched_method_chain(expr, &["bar", "baz"])` will return a `Vec`
/// containing the `Expr`s for
/// `.bar()` and `.baz()`
pub fn method_chain_args<'a>(expr: &'a Expr, methods: &[&str]) -> Option<Vec<&'a [Expr]>> {
    let mut current = expr;
    let mut matched = Vec::with_capacity(methods.len());
    for method_name in methods.iter().rev() {
        // method chains are stored last -> first
        if let ExprMethodCall(ref path, _, ref args) = current.node {
            if path.name == *method_name {
                if args.iter().any(|e| in_macro(e.span)) {
                    return None;
                }
                matched.push(&**args); // build up `matched` backwards
                current = &args[0] // go to parent expression
            } else {
                return None;
            }
        } else {
            return None;
        }
    }
    matched.reverse(); // reverse `matched`, so that it is in the same order as `methods`
    Some(matched)
}


/// Get the name of the item the expression is in, if available.
pub fn get_item_name(cx: &LateContext, expr: &Expr) -> Option<Name> {
    let parent_id = cx.tcx.hir.get_parent(expr.id);
    match cx.tcx.hir.find(parent_id) {
        Some(Node::NodeItem(&Item { ref name, .. })) |
        Some(Node::NodeTraitItem(&TraitItem { ref name, .. })) |
        Some(Node::NodeImplItem(&ImplItem { ref name, .. })) => Some(*name),
        _ => None,
    }
}

/// Get the name of a `Pat`, if any
pub fn get_pat_name(pat: &Pat) -> Option<Name> {
    match pat.node {
        PatKind::Binding(_, _, ref spname, _) => Some(spname.node),
        PatKind::Path(ref qpath) => single_segment_path(qpath).map(|ps| ps.name),
        PatKind::Box(ref p) | PatKind::Ref(ref p, _) => get_pat_name(&*p),
        _ => None,
    }
}

struct ContainsName {
    name: Name,
    result: bool,
}

impl<'tcx> Visitor<'tcx> for ContainsName {
    fn visit_name(&mut self, _: Span, name: Name) {
        if self.name == name {
            self.result = true;
        }
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

/// check if an `Expr` contains a certain name
pub fn contains_name(name: Name, expr: &Expr) -> bool {
    let mut cn = ContainsName {
        name,
        result: false,
    };
    cn.visit_expr(expr);
    cn.result
}


/// Convert a span to a code snippet if available, otherwise use default.
///
/// # Example
/// ```rust,ignore
/// snippet(cx, expr.span, "..")
/// ```
pub fn snippet<'a, 'b, T: LintContext<'b>>(cx: &T, span: Span, default: &'a str) -> Cow<'a, str> {
    snippet_opt(cx, span).map_or_else(|| Cow::Borrowed(default), From::from)
}

/// Convert a span to a code snippet. Returns `None` if not available.
pub fn snippet_opt<'a, T: LintContext<'a>>(cx: &T, span: Span) -> Option<String> {
    cx.sess().codemap().span_to_snippet(span).ok()
}

/// Convert a span (from a block) to a code snippet if available, otherwise use
/// default.
/// This trims the code of indentation, except for the first line. Use it for
/// blocks or block-like
/// things which need to be printed as such.
///
/// # Example
/// ```rust,ignore
/// snippet(cx, expr.span, "..")
/// ```
pub fn snippet_block<'a, 'b, T: LintContext<'b>>(cx: &T, span: Span, default: &'a str) -> Cow<'a, str> {
    let snip = snippet(cx, span, default);
    trim_multiline(snip, true)
}

/// Returns a new Span that covers the full last line of the given Span
pub fn last_line_of_span<'a, T: LintContext<'a>>(cx: &T, span: Span) -> Span {
    let file_map_and_line = cx.sess().codemap().lookup_line(span.lo()).unwrap();
    let line_no = file_map_and_line.line;
    let line_start = &file_map_and_line.fm.lines.clone().into_inner()[line_no];
    Span::new(*line_start, span.hi(), span.ctxt())
}

/// Like `snippet_block`, but add braces if the expr is not an `ExprBlock`.
/// Also takes an `Option<String>` which can be put inside the braces.
pub fn expr_block<'a, 'b, T: LintContext<'b>>(
    cx: &T,
    expr: &Expr,
    option: Option<String>,
    default: &'a str,
) -> Cow<'a, str> {
    let code = snippet_block(cx, expr.span, default);
    let string = option.unwrap_or_default();
    if let ExprBlock(_) = expr.node {
        Cow::Owned(format!("{}{}", code, string))
    } else if string.is_empty() {
        Cow::Owned(format!("{{ {} }}", code))
    } else {
        Cow::Owned(format!("{{\n{};\n{}\n}}", code, string))
    }
}

/// Trim indentation from a multiline string with possibility of ignoring the
/// first line.
pub fn trim_multiline(s: Cow<str>, ignore_first: bool) -> Cow<str> {
    let s_space = trim_multiline_inner(s, ignore_first, ' ');
    let s_tab = trim_multiline_inner(s_space, ignore_first, '\t');
    trim_multiline_inner(s_tab, ignore_first, ' ')
}

fn trim_multiline_inner(s: Cow<str>, ignore_first: bool, ch: char) -> Cow<str> {
    let x = s.lines()
        .skip(ignore_first as usize)
        .filter_map(|l| {
            if l.is_empty() {
                None
            } else {
                // ignore empty lines
                Some(
                    l.char_indices()
                        .find(|&(_, x)| x != ch)
                        .unwrap_or((l.len(), ch))
                        .0,
                )
            }
        })
        .min()
        .unwrap_or(0);
    if x > 0 {
        Cow::Owned(
            s.lines()
                .enumerate()
                .map(|(i, l)| {
                    if (ignore_first && i == 0) || l.is_empty() {
                        l
                    } else {
                        l.split_at(x).1
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        )
    } else {
        s
    }
}

/// Get a parent expressions if any – this is useful to constrain a lint.
pub fn get_parent_expr<'c>(cx: &'c LateContext, e: &Expr) -> Option<&'c Expr> {
    let map = &cx.tcx.hir;
    let node_id: NodeId = e.id;
    let parent_id: NodeId = map.get_parent_node(node_id);
    if node_id == parent_id {
        return None;
    }
    map.find(parent_id).and_then(|node| {
        if let Node::NodeExpr(parent) = node {
            Some(parent)
        } else {
            None
        }
    })
}

pub fn get_enclosing_block<'a, 'tcx: 'a>(cx: &LateContext<'a, 'tcx>, node: NodeId) -> Option<&'tcx Block> {
    let map = &cx.tcx.hir;
    let enclosing_node = map.get_enclosing_scope(node)
        .and_then(|enclosing_id| map.find(enclosing_id));
    if let Some(node) = enclosing_node {
        match node {
            Node::NodeBlock(block) => Some(block),
            Node::NodeItem(&Item {
                node: ItemFn(_, _, _, _, _, eid),
                ..
            }) | Node::NodeImplItem(&ImplItem {
                node: ImplItemKind::Method(_, eid),
                ..
            }) => match cx.tcx.hir.body(eid).value.node {
                ExprBlock(ref block) => Some(block),
                _ => None,
            },
            _ => None,
        }
    } else {
        None
    }
}

pub struct DiagnosticWrapper<'a>(pub DiagnosticBuilder<'a>);

impl<'a> Drop for DiagnosticWrapper<'a> {
    fn drop(&mut self) {
        self.0.emit();
    }
}

impl<'a> DiagnosticWrapper<'a> {
    fn docs_link(&mut self, lint: &'static Lint) {
        if env::var("CLIPPY_DISABLE_DOCS_LINKS").is_err() {
            self.0.help(&format!(
                "for further information visit https://rust-lang-nursery.github.io/rust-clippy/v{}/index.html#{}",
                env!("CARGO_PKG_VERSION"),
                lint.name_lower()
            ));
        }
    }
}

pub fn span_lint<'a, T: LintContext<'a>>(cx: &T, lint: &'static Lint, sp: Span, msg: &str) {
    DiagnosticWrapper(cx.struct_span_lint(lint, sp, msg)).docs_link(lint);
}

pub fn span_help_and_lint<'a, 'tcx: 'a, T: LintContext<'tcx>>(
    cx: &'a T,
    lint: &'static Lint,
    span: Span,
    msg: &str,
    help: &str,
) {
    let mut db = DiagnosticWrapper(cx.struct_span_lint(lint, span, msg));
    db.0.help(help);
    db.docs_link(lint);
}

pub fn span_note_and_lint<'a, 'tcx: 'a, T: LintContext<'tcx>>(
    cx: &'a T,
    lint: &'static Lint,
    span: Span,
    msg: &str,
    note_span: Span,
    note: &str,
) {
    let mut db = DiagnosticWrapper(cx.struct_span_lint(lint, span, msg));
    if note_span == span {
        db.0.note(note);
    } else {
        db.0.span_note(note_span, note);
    }
    db.docs_link(lint);
}

pub fn span_lint_and_then<'a, 'tcx: 'a, T: LintContext<'tcx>, F>(
    cx: &'a T,
    lint: &'static Lint,
    sp: Span,
    msg: &str,
    f: F,
) where
    F: for<'b> FnOnce(&mut DiagnosticBuilder<'b>),
{
    let mut db = DiagnosticWrapper(cx.struct_span_lint(lint, sp, msg));
    f(&mut db.0);
    db.docs_link(lint);
}

/// Add a span lint with a suggestion on how to fix it.
///
/// These suggestions can be parsed by rustfix to allow it to automatically fix your code.
/// In the example below, `help` is `"try"` and `sugg` is the suggested replacement `".any(|x| x > 2)"`.
///
/// ```
/// error: This `.fold` can be more succinctly expressed as `.any`
/// --> $DIR/methods.rs:390:13
///     |
/// 390 |     let _ = (0..3).fold(false, |acc, x| acc || x > 2);
///     |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `.any(|x| x > 2)`
///     |
///     = note: `-D fold-any` implied by `-D warnings`
/// ```
pub fn span_lint_and_sugg<'a, 'tcx: 'a, T: LintContext<'tcx>>(
    cx: &'a T,
    lint: &'static Lint,
    sp: Span,
    msg: &str,
    help: &str,
    sugg: String,
) {
    span_lint_and_then(cx, lint, sp, msg, |db| {
        db.span_suggestion(sp, help, sugg);
    });
}

/// Create a suggestion made from several `span → replacement`.
///
/// Note: in the JSON format (used by `compiletest_rs`), the help message will
/// appear once per
/// replacement. In human-readable format though, it only appears once before
/// the whole suggestion.
pub fn multispan_sugg<I>(db: &mut DiagnosticBuilder, help_msg: String, sugg: I)
where
    I: IntoIterator<Item = (Span, String)>,
{
    let sugg = rustc_errors::CodeSuggestion {
        substitutions: vec![
            rustc_errors::Substitution {
                parts: sugg.into_iter()
                    .map(|(span, snippet)| {
                        rustc_errors::SubstitutionPart {
                            snippet,
                            span,
                        }
                    })
                    .collect(),
            }
        ],
        msg: help_msg,
        show_code_when_inline: true,
        approximate: false,
    };
    db.suggestions.push(sugg);
}

/// Return the base type for HIR references and pointers.
pub fn walk_ptrs_hir_ty(ty: &hir::Ty) -> &hir::Ty {
    match ty.node {
        TyPtr(ref mut_ty) | TyRptr(_, ref mut_ty) => walk_ptrs_hir_ty(&mut_ty.ty),
        _ => ty,
    }
}

/// Return the base type for references and raw pointers.
pub fn walk_ptrs_ty(ty: Ty) -> Ty {
    match ty.sty {
        ty::TyRef(_, ref tm) => walk_ptrs_ty(tm.ty),
        _ => ty,
    }
}

/// Return the base type for references and raw pointers, and count reference
/// depth.
pub fn walk_ptrs_ty_depth(ty: Ty) -> (Ty, usize) {
    fn inner(ty: Ty, depth: usize) -> (Ty, usize) {
        match ty.sty {
            ty::TyRef(_, ref tm) => inner(tm.ty, depth + 1),
            _ => (ty, depth),
        }
    }
    inner(ty, 0)
}

/// Check whether the given expression is a constant literal of the given value.
pub fn is_integer_literal(expr: &Expr, value: u128) -> bool {
    // FIXME: use constant folding
    if let ExprLit(ref spanned) = expr.node {
        if let LitKind::Int(v, _) = spanned.node {
            return v == value;
        }
    }
    false
}

pub fn is_adjusted(cx: &LateContext, e: &Expr) -> bool {
    cx.tables.adjustments().get(e.hir_id).is_some()
}

pub struct LimitStack {
    stack: Vec<u64>,
}

impl Drop for LimitStack {
    fn drop(&mut self) {
        assert_eq!(self.stack.len(), 1);
    }
}

impl LimitStack {
    pub fn new(limit: u64) -> Self {
        Self { stack: vec![limit] }
    }
    pub fn limit(&self) -> u64 {
        *self.stack
            .last()
            .expect("there should always be a value in the stack")
    }
    pub fn push_attrs(&mut self, sess: &Session, attrs: &[ast::Attribute], name: &'static str) {
        let stack = &mut self.stack;
        parse_attrs(sess, attrs, name, |val| stack.push(val));
    }
    pub fn pop_attrs(&mut self, sess: &Session, attrs: &[ast::Attribute], name: &'static str) {
        let stack = &mut self.stack;
        parse_attrs(sess, attrs, name, |val| assert_eq!(stack.pop(), Some(val)));
    }
}

fn parse_attrs<F: FnMut(u64)>(sess: &Session, attrs: &[ast::Attribute], name: &'static str, mut f: F) {
    for attr in attrs {
        if attr.is_sugared_doc {
            continue;
        }
        if let Some(ref value) = attr.value_str() {
            if attr.name().map_or(false, |n| n == name) {
                if let Ok(value) = FromStr::from_str(&value.as_str()) {
                    attr::mark_used(attr);
                    f(value)
                } else {
                    sess.span_err(attr.span, "not a number");
                }
            }
        }
    }
}

/// Return the pre-expansion span if is this comes from an expansion of the
/// macro `name`.
/// See also `is_direct_expn_of`.
pub fn is_expn_of(mut span: Span, name: &str) -> Option<Span> {
    loop {
        let span_name_span = span.ctxt()
            .outer()
            .expn_info()
            .map(|ei| (ei.callee.name(), ei.call_site));

        match span_name_span {
            Some((mac_name, new_span)) if mac_name == name => return Some(new_span),
            None => return None,
            Some((_, new_span)) => span = new_span,
        }
    }
}

/// Return the pre-expansion span if is this directly comes from an expansion
/// of the macro `name`.
/// The difference with `is_expn_of` is that in
/// ```rust,ignore
/// foo!(bar!(42));
/// ```
/// `42` is considered expanded from `foo!` and `bar!` by `is_expn_of` but only
/// `bar!` by
/// `is_direct_expn_of`.
pub fn is_direct_expn_of(span: Span, name: &str) -> Option<Span> {
    let span_name_span = span.ctxt()
        .outer()
        .expn_info()
        .map(|ei| (ei.callee.name(), ei.call_site));

    match span_name_span {
        Some((mac_name, new_span)) if mac_name == name => Some(new_span),
        _ => None,
    }
}

/// Return the index of the character after the first camel-case component of
/// `s`.
pub fn camel_case_until(s: &str) -> usize {
    let mut iter = s.char_indices();
    if let Some((_, first)) = iter.next() {
        if !first.is_uppercase() {
            return 0;
        }
    } else {
        return 0;
    }
    let mut up = true;
    let mut last_i = 0;
    for (i, c) in iter {
        if up {
            if c.is_lowercase() {
                up = false;
            } else {
                return last_i;
            }
        } else if c.is_uppercase() {
            up = true;
            last_i = i;
        } else if !c.is_lowercase() {
            return i;
        }
    }
    if up {
        last_i
    } else {
        s.len()
    }
}

/// Return index of the last camel-case component of `s`.
pub fn camel_case_from(s: &str) -> usize {
    let mut iter = s.char_indices().rev();
    if let Some((_, first)) = iter.next() {
        if !first.is_lowercase() {
            return s.len();
        }
    } else {
        return s.len();
    }
    let mut down = true;
    let mut last_i = s.len();
    for (i, c) in iter {
        if down {
            if c.is_uppercase() {
                down = false;
                last_i = i;
            } else if !c.is_lowercase() {
                return last_i;
            }
        } else if c.is_lowercase() {
            down = true;
        } else {
            return last_i;
        }
    }
    last_i
}

/// Convenience function to get the return type of a function
pub fn return_ty<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, fn_item: NodeId) -> Ty<'tcx> {
    let fn_def_id = cx.tcx.hir.local_def_id(fn_item);
    let ret_ty = cx.tcx.fn_sig(fn_def_id).output();
    cx.tcx.erase_late_bound_regions(&ret_ty)
}

/// Check if two types are the same.
// FIXME: this works correctly for lifetimes bounds (`for <'a> Foo<'a>` == `for
// <'b> Foo<'b>` but
// not for type parameters.
pub fn same_tys<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
    cx.tcx
        .infer_ctxt()
        .enter(|infcx| infcx.can_eq(cx.param_env, a, b).is_ok())
}

/// Return whether the given type is an `unsafe` function.
pub fn type_is_unsafe_function<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.sty {
        ty::TyFnDef(..) | ty::TyFnPtr(_) => ty.fn_sig(cx.tcx).unsafety() == Unsafety::Unsafe,
        _ => false,
    }
}

pub fn is_copy<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    !ty.moves_by_default(cx.tcx.global_tcx(), cx.param_env, DUMMY_SP)
}

/// Return whether a pattern is refutable.
pub fn is_refutable(cx: &LateContext, pat: &Pat) -> bool {
    fn is_enum_variant(cx: &LateContext, qpath: &QPath, id: HirId) -> bool {
        matches!(
            cx.tables.qpath_def(qpath, id),
            def::Def::Variant(..) | def::Def::VariantCtor(..)
        )
    }

    fn are_refutable<'a, I: Iterator<Item = &'a Pat>>(cx: &LateContext, mut i: I) -> bool {
        i.any(|pat| is_refutable(cx, pat))
    }

    match pat.node {
        PatKind::Binding(..) | PatKind::Wild => false,
        PatKind::Box(ref pat) | PatKind::Ref(ref pat, _) => is_refutable(cx, pat),
        PatKind::Lit(..) | PatKind::Range(..) => true,
        PatKind::Path(ref qpath) => is_enum_variant(cx, qpath, pat.hir_id),
        PatKind::Tuple(ref pats, _) => are_refutable(cx, pats.iter().map(|pat| &**pat)),
        PatKind::Struct(ref qpath, ref fields, _) => if is_enum_variant(cx, qpath, pat.hir_id) {
            true
        } else {
            are_refutable(cx, fields.iter().map(|field| &*field.node.pat))
        },
        PatKind::TupleStruct(ref qpath, ref pats, _) => if is_enum_variant(cx, qpath, pat.hir_id) {
            true
        } else {
            are_refutable(cx, pats.iter().map(|pat| &**pat))
        },
        PatKind::Slice(ref head, ref middle, ref tail) => are_refutable(
            cx,
            head.iter()
                .chain(middle)
                .chain(tail.iter())
                .map(|pat| &**pat),
        ),
    }
}

/// Checks for the `#[automatically_derived]` attribute all `#[derive]`d
/// implementations have.
pub fn is_automatically_derived(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, "automatically_derived")
}

/// Remove blocks around an expression.
///
/// Ie. `x`, `{ x }` and `{{{{ x }}}}` all give `x`. `{ x; y }` and `{}` return
/// themselves.
pub fn remove_blocks(expr: &Expr) -> &Expr {
    if let ExprBlock(ref block) = expr.node {
        if block.stmts.is_empty() {
            if let Some(ref expr) = block.expr {
                remove_blocks(expr)
            } else {
                expr
            }
        } else {
            expr
        }
    } else {
        expr
    }
}

pub fn opt_def_id(def: Def) -> Option<DefId> {
    match def {
        Def::Fn(id) |
        Def::Mod(id) |
        Def::Static(id, _) |
        Def::Variant(id) |
        Def::VariantCtor(id, ..) |
        Def::Enum(id) |
        Def::TyAlias(id) |
        Def::AssociatedTy(id) |
        Def::TyParam(id) |
        Def::TyForeign(id) |
        Def::Struct(id) |
        Def::StructCtor(id, ..) |
        Def::Union(id) |
        Def::Trait(id) |
        Def::TraitAlias(id) |
        Def::Method(id) |
        Def::Const(id) |
        Def::AssociatedConst(id) |
        Def::Macro(id, ..) |
        Def::GlobalAsm(id) => Some(id),

        Def::Upvar(..) | Def::Local(_) | Def::Label(..) | Def::PrimTy(..) | Def::SelfTy(..) | Def::Err => None,
    }
}

pub fn is_self(slf: &Arg) -> bool {
    if let PatKind::Binding(_, _, name, _) = slf.pat.node {
        name.node == keywords::SelfValue.name()
    } else {
        false
    }
}

pub fn is_self_ty(slf: &hir::Ty) -> bool {
    if_chain! {
        if let TyPath(ref qp) = slf.node;
        if let QPath::Resolved(None, ref path) = *qp;
        if let Def::SelfTy(..) = path.def;
        then {
            return true
        }
    }
    false
}

pub fn iter_input_pats<'tcx>(decl: &FnDecl, body: &'tcx Body) -> impl Iterator<Item = &'tcx Arg> {
    (0..decl.inputs.len()).map(move |i| &body.arguments[i])
}

/// Check if a given expression is a match expression
/// expanded from `?` operator or `try` macro.
pub fn is_try(expr: &Expr) -> Option<&Expr> {
    fn is_ok(arm: &Arm) -> bool {
        if_chain! {
            if let PatKind::TupleStruct(ref path, ref pat, None) = arm.pats[0].node;
            if match_qpath(path, &paths::RESULT_OK[1..]);
            if let PatKind::Binding(_, defid, _, None) = pat[0].node;
            if let ExprPath(QPath::Resolved(None, ref path)) = arm.body.node;
            if let Def::Local(lid) = path.def;
            if lid == defid;
            then {
                return true;
            }
        }
        false
    }

    fn is_err(arm: &Arm) -> bool {
        if let PatKind::TupleStruct(ref path, _, _) = arm.pats[0].node {
            match_qpath(path, &paths::RESULT_ERR[1..])
        } else {
            false
        }
    }

    if let ExprMatch(_, ref arms, ref source) = expr.node {
        // desugared from a `?` operator
        if let MatchSource::TryDesugar = *source {
            return Some(expr);
        }

        if_chain! {
            if arms.len() == 2;
            if arms[0].pats.len() == 1 && arms[0].guard.is_none();
            if arms[1].pats.len() == 1 && arms[1].guard.is_none();
            if (is_ok(&arms[0]) && is_err(&arms[1])) ||
                (is_ok(&arms[1]) && is_err(&arms[0]));
            then {
                return Some(expr);
            }
        }
    }

    None
}

/// Returns true if the lint is allowed in the current context
///
/// Useful for skipping long running code when it's unnecessary
pub fn is_allowed(cx: &LateContext, lint: &'static Lint, id: NodeId) -> bool {
    cx.tcx.lint_level_at_node(lint, id).0 == Level::Allow
}

pub fn get_arg_name(pat: &Pat) -> Option<ast::Name> {
    match pat.node {
        PatKind::Binding(_, _, name, None) => Some(name.node),
        PatKind::Ref(ref subpat, _) => get_arg_name(subpat),
        _ => None,
    }
}

pub fn int_bits(tcx: TyCtxt, ity: ast::IntTy) -> u64 {
    layout::Integer::from_attr(tcx, attr::IntType::SignedInt(ity)).size().bits()
}

/// Turn a constant int byte representation into an i128
pub fn sext(tcx: TyCtxt, u: u128, ity: ast::IntTy) -> i128 {
    let amt = 128 - int_bits(tcx, ity);
    ((u as i128) << amt) >> amt
}

/// clip unused bytes
pub fn unsext(tcx: TyCtxt, u: i128, ity: ast::IntTy) -> u128 {
    let amt = 128 - int_bits(tcx, ity);
    ((u as u128) << amt) >> amt
}

/// clip unused bytes
pub fn clip(tcx: TyCtxt, u: u128, ity: ast::UintTy) -> u128 {
    let bits = layout::Integer::from_attr(tcx, attr::IntType::UnsignedInt(ity)).size().bits();
    let amt = 128 - bits;
    (u << amt) >> amt
}

/// Remove block comments from the given Vec of lines
///
/// # Examples
///
/// ```rust,ignore
/// without_block_comments(vec!["/*", "foo", "*/"]);
/// // => vec![]
///
/// without_block_comments(vec!["bar", "/*", "foo", "*/"]);
/// // => vec!["bar"]
/// ```
pub fn without_block_comments(lines: Vec<&str>) -> Vec<&str> {
    let mut without = vec![];

    let mut nest_level = 0;

    for line in lines {
        if line.contains("/*") {
            nest_level += 1;
            continue;
        } else if line.contains("*/") {
            nest_level -= 1;
            continue;
        }

        if nest_level == 0 {
            without.push(line);
        }
    }

    without
}
