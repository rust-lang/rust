use rustc::lint::*;
use rustc_front::hir::*;
use reexport::*;
use syntax::codemap::{ExpnInfo, Span, ExpnFormat};
use rustc::front::map::Node::*;
use rustc::middle::def_id::DefId;
use rustc::middle::ty;
use std::borrow::Cow;
use syntax::ast::Lit_::*;
use syntax::ast;
use syntax::errors::DiagnosticBuilder;
use syntax::ptr::P;
use consts::constant;

use rustc::session::Session;
use std::str::FromStr;
use std::ops::{Deref, DerefMut};

pub type MethodArgs = HirVec<P<Expr>>;

// module DefPaths for certain structs/enums we check for
pub const BEGIN_UNWIND: [&'static str; 3] = ["std", "rt", "begin_unwind"];
pub const BTREEMAP_PATH: [&'static str; 4] = ["collections", "btree", "map", "BTreeMap"];
pub const CLONE_PATH: [&'static str; 2] = ["Clone", "clone"];
pub const COW_PATH: [&'static str; 3] = ["collections", "borrow", "Cow"];
pub const HASHMAP_PATH: [&'static str; 5] = ["std", "collections", "hash", "map", "HashMap"];
pub const LL_PATH: [&'static str; 3] = ["collections", "linked_list", "LinkedList"];
pub const MUTEX_PATH: [&'static str; 4] = ["std", "sync", "mutex", "Mutex"];
pub const OPEN_OPTIONS_PATH: [&'static str; 3] = ["std", "fs", "OpenOptions"];
pub const OPTION_PATH: [&'static str; 3] = ["core", "option", "Option"];
pub const RESULT_PATH: [&'static str; 3] = ["core", "result", "Result"];
pub const STRING_PATH: [&'static str; 3] = ["collections", "string", "String"];
pub const VEC_PATH: [&'static str; 3] = ["collections", "vec", "Vec"];

/// Produce a nested chain of if-lets and ifs from the patterns:
///
///     if_let_chain! {
///         [
///             let Some(y) = x,
///             y.len() == 2,
///             let Some(z) = y,
///         ],
///         {
///             block
///         }
///     }
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
}

/// Returns true if the two spans come from differing expansions (i.e. one is from a macro and one
/// isn't).
pub fn differing_macro_contexts(sp1: Span, sp2: Span) -> bool {
    sp1.expn_id != sp2.expn_id
}
/// Returns true if this `expn_info` was expanded by any macro.
pub fn in_macro<T: LintContext>(cx: &T, span: Span) -> bool {
    cx.sess().codemap().with_expn_info(span.expn_id, |info| info.is_some())
}

/// Returns true if the macro that expanded the crate was outside of the current crate or was a
/// compiler plugin.
pub fn in_external_macro<T: LintContext>(cx: &T, span: Span) -> bool {
    /// Invokes in_macro with the expansion info of the given span slightly heavy, try to use this
    /// after other checks have already happened.
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
pub fn match_def_path(cx: &LateContext, def_id: DefId, path: &[&str]) -> bool {
    cx.tcx.with_path(def_id, |iter| {
        iter.zip(path)
            .all(|(nm, p)| nm.name().as_str() == *p)
    })
}

/// Check if type is struct or enum type with given def path.
pub fn match_type(cx: &LateContext, ty: ty::Ty, path: &[&str]) -> bool {
    match ty.sty {
        ty::TyEnum(ref adt, _) | ty::TyStruct(ref adt, _) => match_def_path(cx, adt.did, path),
        _ => false,
    }
}

/// Check if the method call given in `expr` belongs to given trait.
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
    path.segments.iter().rev().zip(segments.iter().rev()).all(|(a, b)| a.identifier.name.as_str() == *b)
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
        Some(NodeItem(&Item{ ref name, .. })) |
        Some(NodeTraitItem(&TraitItem{ ref name, .. })) |
        Some(NodeImplItem(&ImplItem{ ref name, .. })) => Some(*name),
        _ => None,
    }
}

/// Checks if a `let` decl is from a `for` loop desugaring.
pub fn is_from_for_desugar(decl: &Decl) -> bool {
    if_let_chain! {
        [
            let DeclLocal(ref loc) = decl.node,
            let Some(ref expr) = loc.init,
            let ExprMatch(_, _, MatchSource::ForLoopDesugar) = expr.node
        ],
        { return true; }
    };
    false
}


/// Convert a span to a code snippet if available, otherwise use default.
///
/// # Example
/// ```
/// snippet(cx, expr.span, "..")
/// ```
pub fn snippet<'a, T: LintContext>(cx: &T, span: Span, default: &'a str) -> Cow<'a, str> {
    cx.sess().codemap().span_to_snippet(span).map(From::from).unwrap_or(Cow::Borrowed(default))
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
                 if l.len() > 0 {
                     // ignore empty lines
                     Some(l.char_indices()
                           .find(|&(_, x)| x != ch)
                           .unwrap_or((l.len(), ch))
                           .0)
                 } else {
                     None
                 }
             })
             .min()
             .unwrap_or(0);
    if x > 0 {
        Cow::Owned(s.lines()
                    .enumerate()
                    .map(|(i, l)| {
                        if (ignore_first && i == 0) || l.len() == 0 {
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
        if let NodeExpr(parent) = node {
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
            NodeBlock(ref block) => Some(block),
            NodeItem(&Item{ node: ItemFn(_, _, _, _, _, ref block), .. }) => Some(block),
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

pub fn span_lint<'a, T: LintContext>(cx: &'a T, lint: &'static Lint, sp: Span, msg: &str) -> DiagnosticWrapper<'a> {
    let mut db = cx.struct_span_lint(lint, sp, msg);
    if cx.current_level(lint) != Level::Allow {
        db.fileline_help(sp,
                         &format!("for further information visit https://github.com/Manishearth/rust-clippy/wiki#{}",
                                  lint.name_lower()));
    }
    DiagnosticWrapper(db)
}

pub fn span_help_and_lint<'a, T: LintContext>(cx: &'a T, lint: &'static Lint, span: Span, msg: &str, help: &str)
                                              -> DiagnosticWrapper<'a> {
    let mut db = cx.struct_span_lint(lint, span, msg);
    if cx.current_level(lint) != Level::Allow {
        db.fileline_help(span,
                         &format!("{}\nfor further information visit \
                                   https://github.com/Manishearth/rust-clippy/wiki#{}",
                                  help,
                                  lint.name_lower()));
    }
    DiagnosticWrapper(db)
}

pub fn span_note_and_lint<'a, T: LintContext>(cx: &'a T, lint: &'static Lint, span: Span, msg: &str, note_span: Span,
                                              note: &str)
                                              -> DiagnosticWrapper<'a> {
    let mut db = cx.struct_span_lint(lint, span, msg);
    if cx.current_level(lint) != Level::Allow {
        if note_span == span {
            db.fileline_note(note_span, note);
        } else {
            db.span_note(note_span, note);
        }
        db.fileline_help(span,
                         &format!("for further information visit https://github.com/Manishearth/rust-clippy/wiki#{}",
                                  lint.name_lower()));
    }
    DiagnosticWrapper(db)
}

pub fn span_lint_and_then<'a, T: LintContext, F>(cx: &'a T, lint: &'static Lint, sp: Span, msg: &str, f: F)
                                                 -> DiagnosticWrapper<'a>
    where F: Fn(&mut DiagnosticWrapper)
{
    let mut db = DiagnosticWrapper(cx.struct_span_lint(lint, sp, msg));
    if cx.current_level(lint) != Level::Allow {
        f(&mut db);
        db.fileline_help(sp,
                         &format!("for further information visit https://github.com/Manishearth/rust-clippy/wiki#{}",
                                  lint.name_lower()));
    }
    db
}

/// Return the base type for references and raw pointers.
pub fn walk_ptrs_ty(ty: ty::Ty) -> ty::Ty {
    match ty.sty {
        ty::TyRef(_, ref tm) | ty::TyRawPtr(ref tm) => walk_ptrs_ty(tm.ty),
        _ => ty,
    }
}

/// Return the base type for references and raw pointers, and count reference depth.
pub fn walk_ptrs_ty_depth(ty: ty::Ty) -> (ty::Ty, usize) {
    fn inner(ty: ty::Ty, depth: usize) -> (ty::Ty, usize) {
        match ty.sty {
            ty::TyRef(_, ref tm) | ty::TyRawPtr(ref tm) => inner(tm.ty, depth + 1),
            _ => (ty, depth),
        }
    }
    inner(ty, 0)
}

pub fn is_integer_literal(expr: &Expr, value: u64) -> bool {
    // FIXME: use constant folding
    if let ExprLit(ref spanned) = expr.node {
        if let LitInt(v, _) = spanned.node {
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
        if let ast::MetaNameValue(ref key, ref value) = attr.value.node {
            if *key == name {
                if let LitStr(ref s, _) = value.node {
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

pub fn is_exp_equal(cx: &LateContext, left: &Expr, right: &Expr) -> bool {
    if let (Some(l), Some(r)) = (constant(cx, left), constant(cx, right)) {
        if l == r {
            return true;
        }
    }
    match (&left.node, &right.node) {
        (&ExprField(ref lfexp, ref lfident), &ExprField(ref rfexp, ref rfident)) => {
            lfident.node == rfident.node && is_exp_equal(cx, lfexp, rfexp)
        }
        (&ExprLit(ref l), &ExprLit(ref r)) => l.node == r.node,
        (&ExprPath(ref lqself, ref lsubpath), &ExprPath(ref rqself, ref rsubpath)) => {
            both(lqself, rqself, is_qself_equal) && is_path_equal(lsubpath, rsubpath)
        }
        (&ExprTup(ref ltup), &ExprTup(ref rtup)) => is_exps_equal(cx, ltup, rtup),
        (&ExprVec(ref l), &ExprVec(ref r)) => is_exps_equal(cx, l, r),
        (&ExprCast(ref lx, ref lt), &ExprCast(ref rx, ref rt)) => is_exp_equal(cx, lx, rx) && is_cast_ty_equal(lt, rt),
        _ => false,
    }
}

fn is_exps_equal(cx: &LateContext, left: &[P<Expr>], right: &[P<Expr>]) -> bool {
    over(left, right, |l, r| is_exp_equal(cx, l, r))
}

fn is_path_equal(left: &Path, right: &Path) -> bool {
    // The == of idents doesn't work with different contexts,
    // we have to be explicit about hygiene
    left.global == right.global &&
    over(&left.segments,
         &right.segments,
         |l, r| l.identifier.name == r.identifier.name && l.parameters == r.parameters)
}

fn is_qself_equal(left: &QSelf, right: &QSelf) -> bool {
    left.ty.node == right.ty.node && left.position == right.position
}

fn over<X, F>(left: &[X], right: &[X], mut eq_fn: F) -> bool
    where F: FnMut(&X, &X) -> bool
{
    left.len() == right.len() && left.iter().zip(right).all(|(x, y)| eq_fn(x, y))
}

fn both<X, F>(l: &Option<X>, r: &Option<X>, mut eq_fn: F) -> bool
    where F: FnMut(&X, &X) -> bool
{
    l.as_ref().map_or_else(|| r.is_none(), |x| r.as_ref().map_or(false, |y| eq_fn(x, y)))
}

fn is_cast_ty_equal(left: &Ty, right: &Ty) -> bool {
    match (&left.node, &right.node) {
        (&TyVec(ref lvec), &TyVec(ref rvec)) => is_cast_ty_equal(lvec, rvec),
        (&TyPtr(ref lmut), &TyPtr(ref rmut)) => lmut.mutbl == rmut.mutbl && is_cast_ty_equal(&*lmut.ty, &*rmut.ty),
        (&TyRptr(_, ref lrmut), &TyRptr(_, ref rrmut)) => {
            lrmut.mutbl == rrmut.mutbl && is_cast_ty_equal(&*lrmut.ty, &*rrmut.ty)
        }
        (&TyPath(ref lq, ref lpath), &TyPath(ref rq, ref rpath)) => {
            both(lq, rq, is_qself_equal) && is_path_equal(lpath, rpath)
        }
        (&TyInfer, &TyInfer) => true,
        _ => false,
    }
}
