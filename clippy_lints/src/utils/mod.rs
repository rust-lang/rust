use reexport::*;
use rustc::hir::*;
use rustc::hir::def_id::DefId;
use rustc::hir::map::Node;
use rustc::lint::{LintContext, LateContext, Level, Lint};
use rustc::middle::cstore;
use rustc::session::Session;
use rustc::traits::ProjectionMode;
use rustc::traits;
use rustc::ty::subst::Subst;
use rustc::ty;
use std::borrow::Cow;
use std::env;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::str::FromStr;
use syntax::ast::{self, LitKind, RangeLimits};
use syntax::codemap::{ExpnInfo, Span, ExpnFormat};
use syntax::errors::DiagnosticBuilder;
use syntax::ptr::P;

pub mod comparisons;
pub mod conf;
mod hir;
pub mod paths;
pub use self::hir::{SpanlessEq, SpanlessHash};
pub mod cargo;

pub type MethodArgs = HirVec<P<Expr>>;

/// Produce a nested chain of if-lets and ifs from the patterns:
///
///     if_let_chain! {[
///         let Some(y) = x,
///         y.len() == 2,
///         let Some(z) = y,
///     ], {
///         block
///     }}
///
/// becomes
///
///     if let Some(y) = x {
///         if y.len() == 2 {
///             if let Some(z) = y {
///                 block
///             }
///         }
///     }
#[macro_export]
macro_rules! if_let_chain {
    ([let $pat:pat = $expr:expr, $($tt:tt)+], $block:block) => {
        if let $pat = $expr {
           if_let_chain!{ [$($tt)+], $block }
        }
    };
    ([let $pat:pat = $expr:expr], $block:block) => {
        if let $pat = $expr {
           $block
        }
    };
    ([let $pat:pat = $expr:expr,], $block:block) => {
        if let $pat = $expr {
           $block
        }
    };
    ([$expr:expr, $($tt:tt)+], $block:block) => {
        if $expr {
           if_let_chain!{ [$($tt)+], $block }
        }
    };
    ([$expr:expr], $block:block) => {
        if $expr {
           $block
        }
    };
    ([$expr:expr,], $block:block) => {
        if $expr {
           $block
        }
    };
}

/// Returns true if the two spans come from differing expansions (i.e. one is from a macro and one
/// isn't).
pub fn differing_macro_contexts(lhs: Span, rhs: Span) -> bool {
    rhs.expn_id != lhs.expn_id
}
/// Returns true if this `expn_info` was expanded by any macro.
pub fn in_macro<T: LintContext>(cx: &T, span: Span) -> bool {
    cx.sess().codemap().with_expn_info(span.expn_id, |info| info.is_some())
}

/// Returns true if the macro that expanded the crate was outside of the current crate or was a
/// compiler plugin.
pub fn in_external_macro<T: LintContext>(cx: &T, span: Span) -> bool {
    /// Invokes `in_macro` with the expansion info of the given span slightly heavy, try to use
    /// this after other checks have already happened.
    fn in_macro_ext<T: LintContext>(cx: &T, opt_info: Option<&ExpnInfo>) -> bool {
        // no ExpnInfo = no macro
        opt_info.map_or(false, |info| {
            if let ExpnFormat::MacroAttribute(..) = info.callee.format {
                // these are all plugins
                return true;
            }
            // no span for the callee = external macro
            info.callee.span.map_or(true, |span| {
                // no snippet = external macro or compiler-builtin expansion
                cx.sess().codemap().span_to_snippet(span).ok().map_or(true, |code| !code.starts_with("macro_rules"))
            })
        })
    }

    cx.sess().codemap().with_expn_info(span.expn_id, |info| in_macro_ext(cx, info))
}

/// Check if a `DefId`'s path matches the given absolute type path usage.
///
/// # Examples
/// ```
/// match_def_path(cx, id, &["core", "option", "Option"])
/// ```
///
/// See also the `paths` module.
pub fn match_def_path(cx: &LateContext, def_id: DefId, path: &[&str]) -> bool {
    use syntax::parse::token;

    struct AbsolutePathBuffer {
        names: Vec<token::InternedString>,
    }

    impl ty::item_path::ItemPathBuffer for AbsolutePathBuffer {
        fn root_mode(&self) -> &ty::item_path::RootMode {
            const ABSOLUTE: &'static ty::item_path::RootMode = &ty::item_path::RootMode::Absolute;
            ABSOLUTE
        }

        fn push(&mut self, text: &str) {
            self.names.push(token::intern(text).as_str());
        }
    }

    let mut apb = AbsolutePathBuffer { names: vec![] };

    cx.tcx.push_item_path(&mut apb, def_id);

    apb.names == path
}

/// Check if type is struct or enum type with given def path.
pub fn match_type(cx: &LateContext, ty: ty::Ty, path: &[&str]) -> bool {
    match ty.sty {
        ty::TyEnum(ref adt, _) |
        ty::TyStruct(ref adt, _) => match_def_path(cx, adt.did, path),
        _ => false,
    }
}

/// Check if the method call given in `expr` belongs to given type.
pub fn match_impl_method(cx: &LateContext, expr: &Expr, path: &[&str]) -> bool {
    let method_call = ty::MethodCall::expr(expr.id);

    let trt_id = cx.tcx
                   .tables
                   .borrow()
                   .method_map
                   .get(&method_call)
                   .and_then(|callee| cx.tcx.impl_of_method(callee.def_id));
    if let Some(trt_id) = trt_id {
        match_def_path(cx, trt_id, path)
    } else {
        false
    }
}

/// Check if the method call given in `expr` belongs to given trait.
pub fn match_trait_method(cx: &LateContext, expr: &Expr, path: &[&str]) -> bool {
    let method_call = ty::MethodCall::expr(expr.id);

    let trt_id = cx.tcx
                   .tables
                   .borrow()
                   .method_map
                   .get(&method_call)
                   .and_then(|callee| cx.tcx.trait_of_item(callee.def_id));
    if let Some(trt_id) = trt_id {
        match_def_path(cx, trt_id, path)
    } else {
        false
    }
}

/// Match a `Path` against a slice of segment string literals.
///
/// # Examples
/// ```
/// match_path(path, &["std", "rt", "begin_unwind"])
/// ```
pub fn match_path(path: &Path, segments: &[&str]) -> bool {
    path.segments.iter().rev().zip(segments.iter().rev()).all(|(a, b)| a.name.as_str() == *b)
}

/// Match a `Path` against a slice of segment string literals, e.g.
///
/// # Examples
/// ```
/// match_path(path, &["std", "rt", "begin_unwind"])
/// ```
pub fn match_path_ast(path: &ast::Path, segments: &[&str]) -> bool {
    path.segments.iter().rev().zip(segments.iter().rev()).all(|(a, b)| a.identifier.name.as_str() == *b)
}

/// Get the definition associated to a path.
/// TODO: investigate if there is something more efficient for that.
pub fn path_to_def(cx: &LateContext, path: &[&str]) -> Option<cstore::DefLike> {
    let cstore = &cx.tcx.sess.cstore;

    let crates = cstore.crates();
    let krate = crates.iter().find(|&&krate| cstore.crate_name(krate) == path[0]);
    if let Some(krate) = krate {
        let mut items = cstore.crate_top_level_items(*krate);
        let mut path_it = path.iter().skip(1).peekable();

        loop {
            let segment = match path_it.next() {
                Some(segment) => segment,
                None => return None,
            };

            for item in &mem::replace(&mut items, vec![]) {
                if item.name.as_str() == *segment {
                    if path_it.peek().is_none() {
                        return Some(item.def);
                    }

                    let def_id = match item.def {
                        cstore::DefLike::DlDef(def) => def.def_id(),
                        cstore::DefLike::DlImpl(def_id) => def_id,
                        _ => panic!("Unexpected {:?}", item.def),
                    };

                    items = cstore.item_children(def_id);
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
        cstore::DlDef(def::Def::Trait(trait_id)) => Some(trait_id),
        _ => None,
    }
}

/// Check whether a type implements a trait.
/// See also `get_trait_def_id`.
pub fn implements_trait<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, ty: ty::Ty<'tcx>, trait_id: DefId,
                                  ty_params: Vec<ty::Ty<'tcx>>)
                                  -> bool {
    cx.tcx.populate_implementations_for_trait_if_necessary(trait_id);

    let ty = cx.tcx.erase_regions(&ty);
    cx.tcx.infer_ctxt(None, None, ProjectionMode::Any).enter(|infcx| {
        let obligation = cx.tcx.predicate_for_trait_def(traits::ObligationCause::dummy(),
                                                        trait_id,
                                                        0,
                                                        ty,
                                                        ty_params);

        traits::SelectionContext::new(&infcx).evaluate_obligation_conservatively(&obligation)
    })
}

/// Match an `Expr` against a chain of methods, and return the matched `Expr`s.
///
/// For example, if `expr` represents the `.baz()` in `foo.bar().baz()`,
/// `matched_method_chain(expr, &["bar", "baz"])` will return a `Vec` containing the `Expr`s for
/// `.bar()` and `.baz()`
pub fn method_chain_args<'a>(expr: &'a Expr, methods: &[&str]) -> Option<Vec<&'a MethodArgs>> {
    let mut current = expr;
    let mut matched = Vec::with_capacity(methods.len());
    for method_name in methods.iter().rev() {
        // method chains are stored last -> first
        if let ExprMethodCall(ref name, _, ref args) = current.node {
            if name.node.as_str() == *method_name {
                matched.push(args); // build up `matched` backwards
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
    let parent_id = cx.tcx.map.get_parent(expr.id);
    match cx.tcx.map.find(parent_id) {
        Some(Node::NodeItem(&Item { ref name, .. })) |
        Some(Node::NodeTraitItem(&TraitItem { ref name, .. })) |
        Some(Node::NodeImplItem(&ImplItem { ref name, .. })) => Some(*name),
        _ => None,
    }
}

/// Checks if a `let` decl is from a `for` loop desugaring.
pub fn is_from_for_desugar(decl: &Decl) -> bool {
    if_let_chain! {[
        let DeclLocal(ref loc) = decl.node,
        let Some(ref expr) = loc.init,
        let ExprMatch(_, _, MatchSource::ForLoopDesugar) = expr.node
    ], {
        return true;
    }}
    false
}


/// Convert a span to a code snippet if available, otherwise use default.
///
/// # Example
/// ```
/// snippet(cx, expr.span, "..")
/// ```
pub fn snippet<'a, T: LintContext>(cx: &T, span: Span, default: &'a str) -> Cow<'a, str> {
    cx.sess().codemap().span_to_snippet(span).map(From::from).unwrap_or_else(|_| Cow::Borrowed(default))
}

/// Convert a span to a code snippet. Returns `None` if not available.
pub fn snippet_opt<T: LintContext>(cx: &T, span: Span) -> Option<String> {
    cx.sess().codemap().span_to_snippet(span).ok()
}

/// Convert a span (from a block) to a code snippet if available, otherwise use default.
/// This trims the code of indentation, except for the first line. Use it for blocks or block-like
/// things which need to be printed as such.
///
/// # Example
/// ```
/// snippet(cx, expr.span, "..")
/// ```
pub fn snippet_block<'a, T: LintContext>(cx: &T, span: Span, default: &'a str) -> Cow<'a, str> {
    let snip = snippet(cx, span, default);
    trim_multiline(snip, true)
}

/// Like `snippet_block`, but add braces if the expr is not an `ExprBlock`.
/// Also takes an `Option<String>` which can be put inside the braces.
pub fn expr_block<'a, T: LintContext>(cx: &T, expr: &Expr, option: Option<String>, default: &'a str) -> Cow<'a, str> {
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

/// Trim indentation from a multiline string with possibility of ignoring the first line.
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
                     Some(l.char_indices()
                           .find(|&(_, x)| x != ch)
                           .unwrap_or((l.len(), ch))
                           .0)
                 }
             })
             .min()
             .unwrap_or(0);
    if x > 0 {
        Cow::Owned(s.lines()
                    .enumerate()
                    .map(|(i, l)| {
                        if (ignore_first && i == 0) || l.is_empty() {
                            l
                        } else {
                            l.split_at(x).1
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n"))
    } else {
        s
    }
}

/// Get a parent expressions if any – this is useful to constrain a lint.
pub fn get_parent_expr<'c>(cx: &'c LateContext, e: &Expr) -> Option<&'c Expr> {
    let map = &cx.tcx.map;
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

pub fn get_enclosing_block<'c>(cx: &'c LateContext, node: NodeId) -> Option<&'c Block> {
    let map = &cx.tcx.map;
    let enclosing_node = map.get_enclosing_scope(node)
                            .and_then(|enclosing_id| map.find(enclosing_id));
    if let Some(node) = enclosing_node {
        match node {
            Node::NodeBlock(ref block) => Some(block),
            Node::NodeItem(&Item { node: ItemFn(_, _, _, _, _, ref block), .. }) => Some(block),
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

impl<'a> DerefMut for DiagnosticWrapper<'a> {
    fn deref_mut(&mut self) -> &mut DiagnosticBuilder<'a> {
        &mut self.0
    }
}

impl<'a> Deref for DiagnosticWrapper<'a> {
    type Target = DiagnosticBuilder<'a>;
    fn deref(&self) -> &DiagnosticBuilder<'a> {
        &self.0
    }
}

impl<'a> DiagnosticWrapper<'a> {
    fn wiki_link(&mut self, lint: &'static Lint) {
        if env::var("CLIPPY_DISABLE_WIKI_LINKS").is_err() {
            self.help(&format!("for further information visit https://github.com/Manishearth/rust-clippy/wiki#{}",
                               lint.name_lower()));
        }
    }
}

pub fn span_lint<'a, T: LintContext>(cx: &'a T, lint: &'static Lint, sp: Span, msg: &str) -> DiagnosticWrapper<'a> {
    let mut db = DiagnosticWrapper(cx.struct_span_lint(lint, sp, msg));
    if cx.current_level(lint) != Level::Allow {
        db.wiki_link(lint);
    }
    db
}

pub fn span_help_and_lint<'a, T: LintContext>(cx: &'a T, lint: &'static Lint, span: Span, msg: &str, help: &str)
                                              -> DiagnosticWrapper<'a> {
    let mut db = DiagnosticWrapper(cx.struct_span_lint(lint, span, msg));
    if cx.current_level(lint) != Level::Allow {
        db.help(help);
        db.wiki_link(lint);
    }
    db
}

pub fn span_note_and_lint<'a, T: LintContext>(cx: &'a T, lint: &'static Lint, span: Span, msg: &str, note_span: Span,
                                              note: &str)
                                              -> DiagnosticWrapper<'a> {
    let mut db = DiagnosticWrapper(cx.struct_span_lint(lint, span, msg));
    if cx.current_level(lint) != Level::Allow {
        if note_span == span {
            db.note(note);
        } else {
            db.span_note(note_span, note);
        }
        db.wiki_link(lint);
    }
    db
}

pub fn span_lint_and_then<'a, T: LintContext, F>(cx: &'a T, lint: &'static Lint, sp: Span, msg: &str, f: F)
                                                 -> DiagnosticWrapper<'a>
    where F: FnOnce(&mut DiagnosticWrapper)
{
    let mut db = DiagnosticWrapper(cx.struct_span_lint(lint, sp, msg));
    if cx.current_level(lint) != Level::Allow {
        f(&mut db);
        db.wiki_link(lint);
    }
    db
}

/// Return the base type for references and raw pointers.
pub fn walk_ptrs_ty(ty: ty::Ty) -> ty::Ty {
    match ty.sty {
        ty::TyRef(_, ref tm) |
        ty::TyRawPtr(ref tm) => walk_ptrs_ty(tm.ty),
        _ => ty,
    }
}

/// Return the base type for references and raw pointers, and count reference depth.
pub fn walk_ptrs_ty_depth(ty: ty::Ty) -> (ty::Ty, usize) {
    fn inner(ty: ty::Ty, depth: usize) -> (ty::Ty, usize) {
        match ty.sty {
            ty::TyRef(_, ref tm) |
            ty::TyRawPtr(ref tm) => inner(tm.ty, depth + 1),
            _ => (ty, depth),
        }
    }
    inner(ty, 0)
}

/// Check whether the given expression is a constant literal of the given value.
pub fn is_integer_literal(expr: &Expr, value: u64) -> bool {
    // FIXME: use constant folding
    if let ExprLit(ref spanned) = expr.node {
        if let LitKind::Int(v, _) = spanned.node {
            return v == value;
        }
    }
    false
}

pub fn is_adjusted(cx: &LateContext, e: &Expr) -> bool {
    cx.tcx.tables.borrow().adjustments.get(&e.id).is_some()
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
    pub fn new(limit: u64) -> LimitStack {
        LimitStack { stack: vec![limit] }
    }
    pub fn limit(&self) -> u64 {
        *self.stack.last().expect("there should always be a value in the stack")
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
        let attr = &attr.node;
        if attr.is_sugared_doc {
            continue;
        }
        if let ast::MetaItemKind::NameValue(ref key, ref value) = attr.value.node {
            if *key == name {
                if let LitKind::Str(ref s, _) = value.node {
                    if let Ok(value) = FromStr::from_str(s) {
                        f(value)
                    } else {
                        sess.span_err(value.span, "not a number");
                    }
                } else {
                    unreachable!()
                }
            }
        }
    }
}

/// Return the pre-expansion span if is this comes from an expansion of the macro `name`.
/// See also `is_direct_expn_of`.
pub fn is_expn_of(cx: &LateContext, mut span: Span, name: &str) -> Option<Span> {
    loop {
        let span_name_span = cx.tcx
                               .sess
                               .codemap()
                               .with_expn_info(span.expn_id, |expn| expn.map(|ei| (ei.callee.name(), ei.call_site)));

        match span_name_span {
            Some((mac_name, new_span)) if mac_name.as_str() == name => return Some(new_span),
            None => return None,
            Some((_, new_span)) => span = new_span,
        }
    }
}

/// Return the pre-expansion span if is this directly comes from an expansion of the macro `name`.
/// The difference with `is_expn_of` is that in
/// ```rust,ignore
/// foo!(bar!(42));
/// ```
/// `42` is considered expanded from `foo!` and `bar!` by `is_expn_of` but only `bar!` by
/// `is_direct_expn_of`.
pub fn is_direct_expn_of(cx: &LateContext, span: Span, name: &str) -> Option<Span> {
    let span_name_span = cx.tcx
                           .sess
                           .codemap()
                           .with_expn_info(span.expn_id, |expn| expn.map(|ei| (ei.callee.name(), ei.call_site)));

    match span_name_span {
        Some((mac_name, new_span)) if mac_name.as_str() == name => Some(new_span),
        _ => None,
    }
}

/// Return the index of the character after the first camel-case component of `s`.
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

/// Represent a range akin to `ast::ExprKind::Range`.
#[derive(Debug, Copy, Clone)]
pub struct UnsugaredRange<'a> {
    pub start: Option<&'a Expr>,
    pub end: Option<&'a Expr>,
    pub limits: RangeLimits,
}

/// Unsugar a `hir` range.
pub fn unsugar_range(expr: &Expr) -> Option<UnsugaredRange> {
    // To be removed when ranges get stable.
    fn unwrap_unstable(expr: &Expr) -> &Expr {
        if let ExprBlock(ref block) = expr.node {
            if block.rules == BlockCheckMode::PushUnstableBlock || block.rules == BlockCheckMode::PopUnstableBlock {
                if let Some(ref expr) = block.expr {
                    return expr;
                }
            }
        }

        expr
    }

    fn get_field<'a>(name: &str, fields: &'a [Field]) -> Option<&'a Expr> {
        let expr = &fields.iter()
                          .find(|field| field.name.node.as_str() == name)
                          .unwrap_or_else(|| panic!("missing {} field for range", name))
                          .expr;

        Some(unwrap_unstable(expr))
    }

    // The range syntax is expanded to literal paths starting with `core` or `std` depending on
    // `#[no_std]`. Testing both instead of resolving the paths.

    match unwrap_unstable(expr).node {
        ExprPath(None, ref path) => {
            if match_path(path, &paths::RANGE_FULL_STD) || match_path(path, &paths::RANGE_FULL) {
                Some(UnsugaredRange {
                    start: None,
                    end: None,
                    limits: RangeLimits::HalfOpen,
                })
            } else {
                None
            }
        }
        ExprStruct(ref path, ref fields, None) => {
            if match_path(path, &paths::RANGE_FROM_STD) || match_path(path, &paths::RANGE_FROM) {
                Some(UnsugaredRange {
                    start: get_field("start", fields),
                    end: None,
                    limits: RangeLimits::HalfOpen,
                })
            } else if match_path(path, &paths::RANGE_INCLUSIVE_NON_EMPTY_STD) ||
               match_path(path, &paths::RANGE_INCLUSIVE_NON_EMPTY) {
                Some(UnsugaredRange {
                    start: get_field("start", fields),
                    end: get_field("end", fields),
                    limits: RangeLimits::Closed,
                })
            } else if match_path(path, &paths::RANGE_STD) || match_path(path, &paths::RANGE) {
                Some(UnsugaredRange {
                    start: get_field("start", fields),
                    end: get_field("end", fields),
                    limits: RangeLimits::HalfOpen,
                })
            } else if match_path(path, &paths::RANGE_TO_INCLUSIVE_STD) || match_path(path, &paths::RANGE_TO_INCLUSIVE) {
                Some(UnsugaredRange {
                    start: None,
                    end: get_field("end", fields),
                    limits: RangeLimits::Closed,
                })
            } else if match_path(path, &paths::RANGE_TO_STD) || match_path(path, &paths::RANGE_TO) {
                Some(UnsugaredRange {
                    start: None,
                    end: get_field("end", fields),
                    limits: RangeLimits::HalfOpen,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Convenience function to get the return type of a function or `None` if the function diverges.
pub fn return_ty<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, fn_item: NodeId) -> Option<ty::Ty<'tcx>> {
    let parameter_env = ty::ParameterEnvironment::for_item(cx.tcx, fn_item);
    let fn_sig = cx.tcx.node_id_to_type(fn_item).fn_sig().subst(cx.tcx, parameter_env.free_substs);
    let fn_sig = cx.tcx.liberate_late_bound_regions(parameter_env.free_id_outlive, &fn_sig);
    if let ty::FnConverging(ret_ty) = fn_sig.output {
        Some(ret_ty)
    } else {
        None
    }
}

/// Check if two types are the same.
// FIXME: this works correctly for lifetimes bounds (`for <'a> Foo<'a>` == `for <'b> Foo<'b>` but
// not for type parameters.
pub fn same_tys<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, a: ty::Ty<'tcx>, b: ty::Ty<'tcx>, parameter_item: NodeId) -> bool {
    let parameter_env = ty::ParameterEnvironment::for_item(cx.tcx, parameter_item);
    cx.tcx.infer_ctxt(None, Some(parameter_env), ProjectionMode::Any).enter(|infcx| {
        let new_a = a.subst(infcx.tcx, infcx.parameter_environment.free_substs);
        let new_b = b.subst(infcx.tcx, infcx.parameter_environment.free_substs);
        infcx.can_equate(&new_a, &new_b).is_ok()
    })
}

/// Recover the essential nodes of a desugared for loop:
/// `for pat in arg { body }` becomes `(pat, arg, body)`.
pub fn recover_for_loop(expr: &Expr) -> Option<(&Pat, &Expr, &Expr)> {
    if_let_chain! {[
        let ExprMatch(ref iterexpr, ref arms, _) = expr.node,
        let ExprCall(_, ref iterargs) = iterexpr.node,
        iterargs.len() == 1 && arms.len() == 1 && arms[0].guard.is_none(),
        let ExprLoop(ref block, _) = arms[0].body.node,
        block.stmts.is_empty(),
        let Some(ref loopexpr) = block.expr,
        let ExprMatch(_, ref innerarms, MatchSource::ForLoopDesugar) = loopexpr.node,
        innerarms.len() == 2 && innerarms[0].pats.len() == 1,
        let PatKind::TupleStruct(_, ref somepats, _) = innerarms[0].pats[0].node,
        somepats.len() == 1
    ], {
        return Some((&somepats[0],
                     &iterargs[0],
                     &innerarms[0].body));
    }}
    None
}
