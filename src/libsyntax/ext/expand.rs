// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{Block, Crate, PatKind};
use ast::{Local, Ident, Mac_, Name, SpannedIdent};
use ast::{MacStmtStyle, Mrk, Stmt, StmtKind, ItemKind};
use ast;
use attr::HasAttrs;
use ext::mtwt;
use attr;
use attr::AttrMetaMethods;
use codemap::{Spanned, ExpnInfo, NameAndSpan, MacroBang, MacroAttribute};
use syntax_pos::{self, Span, ExpnId};
use config::StripUnconfigured;
use ext::base::*;
use feature_gate::{self, Features};
use fold;
use fold::*;
use util::move_map::MoveMap;
use parse::token::{fresh_mark, fresh_name, intern, keywords};
use ptr::P;
use tokenstream::TokenTree;
use util::small_vector::SmallVector;
use visit;
use visit::Visitor;
use std_inject;

use std::collections::HashSet;

// A trait for AST nodes and AST node lists into which macro invocations may expand.
trait MacroGenerable: Sized {
    // Expand the given MacResult using its appropriate `make_*` method.
    fn make_with<'a>(result: Box<MacResult + 'a>) -> Option<Self>;

    // Fold this node or list of nodes using the given folder.
    fn fold_with<F: Folder>(self, folder: &mut F) -> Self;
    fn visit_with<V: Visitor>(&self, visitor: &mut V);

    // Return a placeholder expansion to allow compilation to continue after an erroring expansion.
    fn dummy(span: Span) -> Self;

    // The user-friendly name of the node type (e.g. "expression", "item", etc.) for diagnostics.
    fn kind_name() -> &'static str;
}

macro_rules! impl_macro_generable {
    ($($ty:ty: $kind_name:expr, .$make:ident,
               $(.$fold:ident)*  $(lift .$fold_elt:ident)*,
               $(.$visit:ident)* $(lift .$visit_elt:ident)*,
               |$span:ident| $dummy:expr;)*) => { $(
        impl MacroGenerable for $ty {
            fn kind_name() -> &'static str { $kind_name }
            fn make_with<'a>(result: Box<MacResult + 'a>) -> Option<Self> { result.$make() }
            fn fold_with<F: Folder>(self, folder: &mut F) -> Self {
                $( folder.$fold(self) )*
                $( self.into_iter().flat_map(|item| folder. $fold_elt (item)).collect() )*
            }
            fn visit_with<V: Visitor>(&self, visitor: &mut V) {
                $( visitor.$visit(self) )*
                $( for item in self.as_slice() { visitor. $visit_elt (item) } )*
            }
            fn dummy($span: Span) -> Self { $dummy }
        }
    )* }
}

impl_macro_generable! {
    P<ast::Pat>: "pattern", .make_pat, .fold_pat, .visit_pat, |span| P(DummyResult::raw_pat(span));
    P<ast::Ty>:  "type",    .make_ty,  .fold_ty,  .visit_ty,  |span| DummyResult::raw_ty(span);
    P<ast::Expr>:
        "expression", .make_expr, .fold_expr, .visit_expr, |span| DummyResult::raw_expr(span);
    SmallVector<ast::Stmt>:
        "statement",  .make_stmts, lift .fold_stmt, lift .visit_stmt, |_span| SmallVector::zero();
    SmallVector<P<ast::Item>>:
        "item",       .make_items, lift .fold_item, lift .visit_item, |_span| SmallVector::zero();
    SmallVector<ast::TraitItem>:
        "trait item", .make_trait_items, lift .fold_trait_item, lift .visit_trait_item,
        |_span| SmallVector::zero();
    SmallVector<ast::ImplItem>:
        "impl item",  .make_impl_items,  lift .fold_impl_item,  lift .visit_impl_item,
        |_span| SmallVector::zero();
}

impl MacroGenerable for Option<P<ast::Expr>> {
    fn kind_name() -> &'static str { "expression" }
    fn dummy(_span: Span) -> Self { None }
    fn make_with<'a>(result: Box<MacResult + 'a>) -> Option<Self> {
        result.make_expr().map(Some)
    }
    fn fold_with<F: Folder>(self, folder: &mut F) -> Self {
        self.and_then(|expr| folder.fold_opt_expr(expr))
    }
    fn visit_with<V: Visitor>(&self, visitor: &mut V) {
        self.as_ref().map(|expr| visitor.visit_expr(expr));
    }
}

pub fn expand_expr(mut expr: ast::Expr, fld: &mut MacroExpander) -> P<ast::Expr> {
    match expr.node {
        // expr_mac should really be expr_ext or something; it's the
        // entry-point for all syntax extensions.
        ast::ExprKind::Mac(mac) => {
            return expand_mac_invoc(mac, None, expr.attrs.into(), expr.span, fld);
        }

        ast::ExprKind::While(cond, body, opt_ident) => {
            let cond = fld.fold_expr(cond);
            let (body, opt_ident) = expand_loop_block(body, opt_ident, fld);
            expr.node = ast::ExprKind::While(cond, body, opt_ident);
        }

        ast::ExprKind::WhileLet(pat, cond, body, opt_ident) => {
            let pat = fld.fold_pat(pat);
            let cond = fld.fold_expr(cond);

            // Hygienic renaming of the body.
            let ((body, opt_ident), mut rewritten_pats) =
                rename_in_scope(vec![pat],
                                fld,
                                (body, opt_ident),
                                |rename_fld, fld, (body, opt_ident)| {
                expand_loop_block(rename_fld.fold_block(body), opt_ident, fld)
            });
            assert!(rewritten_pats.len() == 1);

            expr.node = ast::ExprKind::WhileLet(rewritten_pats.remove(0), cond, body, opt_ident);
        }

        ast::ExprKind::Loop(loop_block, opt_ident) => {
            let (loop_block, opt_ident) = expand_loop_block(loop_block, opt_ident, fld);
            expr.node = ast::ExprKind::Loop(loop_block, opt_ident);
        }

        ast::ExprKind::ForLoop(pat, head, body, opt_ident) => {
            let pat = fld.fold_pat(pat);

            // Hygienic renaming of the for loop body (for loop binds its pattern).
            let ((body, opt_ident), mut rewritten_pats) =
                rename_in_scope(vec![pat],
                                fld,
                                (body, opt_ident),
                                |rename_fld, fld, (body, opt_ident)| {
                expand_loop_block(rename_fld.fold_block(body), opt_ident, fld)
            });
            assert!(rewritten_pats.len() == 1);

            let head = fld.fold_expr(head);
            expr.node = ast::ExprKind::ForLoop(rewritten_pats.remove(0), head, body, opt_ident);
        }

        ast::ExprKind::IfLet(pat, sub_expr, body, else_opt) => {
            let pat = fld.fold_pat(pat);

            // Hygienic renaming of the body.
            let (body, mut rewritten_pats) =
                rename_in_scope(vec![pat],
                                fld,
                                body,
                                |rename_fld, fld, body| {
                fld.fold_block(rename_fld.fold_block(body))
            });
            assert!(rewritten_pats.len() == 1);

            let else_opt = else_opt.map(|else_opt| fld.fold_expr(else_opt));
            let sub_expr = fld.fold_expr(sub_expr);
            expr.node = ast::ExprKind::IfLet(rewritten_pats.remove(0), sub_expr, body, else_opt);
        }

        ast::ExprKind::Closure(capture_clause, fn_decl, block, fn_decl_span) => {
            let (rewritten_fn_decl, rewritten_block)
                = expand_and_rename_fn_decl_and_block(fn_decl, block, fld);
            expr.node = ast::ExprKind::Closure(capture_clause,
                                               rewritten_fn_decl,
                                               rewritten_block,
                                               fn_decl_span);
        }

        _ => expr = noop_fold_expr(expr, fld),
    };
    P(expr)
}

/// Expand a macro invocation. Returns the result of expansion.
fn expand_mac_invoc<T>(mac: ast::Mac, ident: Option<Ident>, attrs: Vec<ast::Attribute>, span: Span,
                       fld: &mut MacroExpander) -> T
    where T: MacroGenerable,
{
    // It would almost certainly be cleaner to pass the whole macro invocation in,
    // rather than pulling it apart and marking the tts and the ctxt separately.
    let Mac_ { path, tts, .. } = mac.node;
    let mark = fresh_mark();

    fn mac_result<'a>(path: &ast::Path, ident: Option<Ident>, tts: Vec<TokenTree>, mark: Mrk,
                      attrs: Vec<ast::Attribute>, call_site: Span, fld: &'a mut MacroExpander)
                      -> Option<Box<MacResult + 'a>> {
        // Detect use of feature-gated or invalid attributes on macro invoations
        // since they will not be detected after macro expansion.
        for attr in attrs.iter() {
            feature_gate::check_attribute(&attr, &fld.cx.parse_sess.span_diagnostic,
                                          &fld.cx.parse_sess.codemap(),
                                          &fld.cx.ecfg.features.unwrap());
        }

        if path.segments.len() > 1 {
            fld.cx.span_err(path.span, "expected macro name without module separators");
            return None;
        }

        let extname = path.segments[0].identifier.name;
        let extension = if let Some(extension) = fld.cx.syntax_env.find(extname) {
            extension
        } else {
            let mut err = fld.cx.struct_span_err(path.span,
                                                 &format!("macro undefined: '{}!'", &extname));
            fld.cx.suggest_macro_name(&extname.as_str(), &mut err);
            err.emit();
            return None;
        };

        let ident = ident.unwrap_or(keywords::Invalid.ident());
        match *extension {
            NormalTT(ref expandfun, exp_span, allow_internal_unstable) => {
                if ident.name != keywords::Invalid.name() {
                    let msg =
                        format!("macro {}! expects no ident argument, given '{}'", extname, ident);
                    fld.cx.span_err(path.span, &msg);
                    return None;
                }

                fld.cx.bt_push(ExpnInfo {
                    call_site: call_site,
                    callee: NameAndSpan {
                        format: MacroBang(extname),
                        span: exp_span,
                        allow_internal_unstable: allow_internal_unstable,
                    },
                });

                let marked_tts = mark_tts(tts, mark);
                Some(expandfun.expand(fld.cx, call_site, &marked_tts))
            }

            IdentTT(ref expander, tt_span, allow_internal_unstable) => {
                if ident.name == keywords::Invalid.name() {
                    fld.cx.span_err(path.span,
                                    &format!("macro {}! expects an ident argument", extname));
                    return None;
                };

                fld.cx.bt_push(ExpnInfo {
                    call_site: call_site,
                    callee: NameAndSpan {
                        format: MacroBang(extname),
                        span: tt_span,
                        allow_internal_unstable: allow_internal_unstable,
                    }
                });

                let marked_tts = mark_tts(tts, mark);
                Some(expander.expand(fld.cx, call_site, ident, marked_tts))
            }

            MacroRulesTT => {
                if ident.name == keywords::Invalid.name() {
                    fld.cx.span_err(path.span,
                                    &format!("macro {}! expects an ident argument", extname));
                    return None;
                };

                fld.cx.bt_push(ExpnInfo {
                    call_site: call_site,
                    callee: NameAndSpan {
                        format: MacroBang(extname),
                        span: None,
                        // `macro_rules!` doesn't directly allow unstable
                        // (this is orthogonal to whether the macro it creates allows it)
                        allow_internal_unstable: false,
                    }
                });

                // DON'T mark before expansion.
                fld.cx.insert_macro(ast::MacroDef {
                    ident: ident,
                    id: ast::DUMMY_NODE_ID,
                    span: call_site,
                    imported_from: None,
                    use_locally: true,
                    body: tts,
                    export: attr::contains_name(&attrs, "macro_export"),
                    allow_internal_unstable: attr::contains_name(&attrs, "allow_internal_unstable"),
                    attrs: attrs,
                });

                // macro_rules! has a side effect but expands to nothing.
                fld.cx.bt_pop();
                None
            }

            MultiDecorator(..) | MultiModifier(..) => {
                fld.cx.span_err(path.span,
                                &format!("`{}` can only be used in attributes", extname));
                None
            }
        }
    }

    let opt_expanded = T::make_with(match mac_result(&path, ident, tts, mark, attrs, span, fld) {
        Some(result) => result,
        None => return T::dummy(span),
    });

    let expanded = if let Some(expanded) = opt_expanded {
        expanded
    } else {
        let msg = format!("non-{kind} macro in {kind} position: {name}",
                          name = path.segments[0].identifier.name, kind = T::kind_name());
        fld.cx.span_err(path.span, &msg);
        return T::dummy(span);
    };

    let marked = expanded.fold_with(&mut Marker { mark: mark, expn_id: Some(fld.cx.backtrace()) });
    let configured = marked.fold_with(&mut fld.strip_unconfigured());
    fld.load_macros(&configured);
    let fully_expanded = configured.fold_with(fld);
    fld.cx.bt_pop();
    fully_expanded
}

/// Rename loop label and expand its loop body
///
/// The renaming procedure for loop is different in the sense that the loop
/// body is in a block enclosed by loop head so the renaming of loop label
/// must be propagated to the enclosed context.
fn expand_loop_block(loop_block: P<Block>,
                     opt_ident: Option<SpannedIdent>,
                     fld: &mut MacroExpander) -> (P<Block>, Option<SpannedIdent>) {
    match opt_ident {
        Some(label) => {
            let new_label = fresh_name(label.node);
            let rename = (label.node, new_label);

            // The rename *must not* be added to the pending list of current
            // syntax context otherwise an unrelated `break` or `continue` in
            // the same context will pick that up in the deferred renaming pass
            // and be renamed incorrectly.
            let mut rename_list = vec!(rename);
            let mut rename_fld = IdentRenamer{renames: &mut rename_list};
            let renamed_ident = rename_fld.fold_ident(label.node);

            // The rename *must* be added to the enclosed syntax context for
            // `break` or `continue` to pick up because by definition they are
            // in a block enclosed by loop head.
            fld.cx.syntax_env.push_frame();
            fld.cx.syntax_env.info().pending_renames.push(rename);
            let expanded_block = expand_block_elts(loop_block, fld);
            fld.cx.syntax_env.pop_frame();

            (expanded_block, Some(Spanned { node: renamed_ident, span: label.span }))
        }
        None => (fld.fold_block(loop_block), opt_ident)
    }
}

// eval $e with a new exts frame.
// must be a macro so that $e isn't evaluated too early.
macro_rules! with_exts_frame {
    ($extsboxexpr:expr,$macros_escape:expr,$e:expr) =>
    ({$extsboxexpr.push_frame();
      $extsboxexpr.info().macros_escape = $macros_escape;
      let result = $e;
      $extsboxexpr.pop_frame();
      result
     })
}

// When we enter a module, record it, for the sake of `module!`
pub fn expand_item(it: P<ast::Item>, fld: &mut MacroExpander)
                   -> SmallVector<P<ast::Item>> {
    expand_annotatable(Annotatable::Item(it), fld)
        .into_iter().map(|i| i.expect_item()).collect()
}

/// Expand item_kind
fn expand_item_kind(item: ast::ItemKind, fld: &mut MacroExpander) -> ast::ItemKind {
    match item {
        ast::ItemKind::Fn(decl, unsafety, constness, abi, generics, body) => {
            let (rewritten_fn_decl, rewritten_body)
                = expand_and_rename_fn_decl_and_block(decl, body, fld);
            let expanded_generics = fold::noop_fold_generics(generics,fld);
            ast::ItemKind::Fn(rewritten_fn_decl, unsafety, constness, abi,
                        expanded_generics, rewritten_body)
        }
        _ => noop_fold_item_kind(item, fld)
    }
}

// does this attribute list contain "macro_use" ?
fn contains_macro_use(fld: &mut MacroExpander, attrs: &[ast::Attribute]) -> bool {
    for attr in attrs {
        let mut is_use = attr.check_name("macro_use");
        if attr.check_name("macro_escape") {
            let mut err =
                fld.cx.struct_span_warn(attr.span,
                                        "macro_escape is a deprecated synonym for macro_use");
            is_use = true;
            if let ast::AttrStyle::Inner = attr.node.style {
                err.help("consider an outer attribute, \
                          #[macro_use] mod ...").emit();
            } else {
                err.emit();
            }
        };

        if is_use {
            match attr.node.value.node {
                ast::MetaItemKind::Word(..) => (),
                _ => fld.cx.span_err(attr.span, "arguments to macro_use are not allowed here"),
            }
            return true;
        }
    }
    false
}

/// Expand a stmt
fn expand_stmt(stmt: Stmt, fld: &mut MacroExpander) -> SmallVector<Stmt> {
    // perform all pending renames
    let stmt = {
        let pending_renames = &mut fld.cx.syntax_env.info().pending_renames;
        let mut rename_fld = IdentRenamer{renames:pending_renames};
        rename_fld.fold_stmt(stmt).expect_one("rename_fold didn't return one value")
    };

    let (mac, style, attrs) = match stmt.node {
        StmtKind::Mac(mac) => mac.unwrap(),
        _ => return expand_non_macro_stmt(stmt, fld)
    };

    let mut fully_expanded: SmallVector<ast::Stmt> =
        expand_mac_invoc(mac, None, attrs.into(), stmt.span, fld);

    // If this is a macro invocation with a semicolon, then apply that
    // semicolon to the final statement produced by expansion.
    if style == MacStmtStyle::Semicolon {
        if let Some(stmt) = fully_expanded.pop() {
            fully_expanded.push(Stmt {
                id: stmt.id,
                node: match stmt.node {
                    StmtKind::Expr(expr) => StmtKind::Semi(expr),
                    _ => stmt.node /* might already have a semi */
                },
                span: stmt.span,
            });
        }
    }

    fully_expanded
}

// expand a non-macro stmt. this is essentially the fallthrough for
// expand_stmt, above.
fn expand_non_macro_stmt(stmt: Stmt, fld: &mut MacroExpander)
                         -> SmallVector<Stmt> {
    // is it a let?
    match stmt.node {
        StmtKind::Local(local) => {
            // take it apart:
            let rewritten_local = local.map(|Local {id, pat, ty, init, span, attrs}| {
                // expand the ty since TyKind::FixedLengthVec contains an Expr
                // and thus may have a macro use
                let expanded_ty = ty.map(|t| fld.fold_ty(t));
                // expand the pat (it might contain macro uses):
                let expanded_pat = fld.fold_pat(pat);
                // find the PatIdents in the pattern:
                // oh dear heaven... this is going to include the enum
                // names, as well... but that should be okay, as long as
                // the new names are gensyms for the old ones.
                // generate fresh names, push them to a new pending list
                let idents = pattern_bindings(&expanded_pat);
                let mut new_pending_renames =
                    idents.iter().map(|ident| (*ident, fresh_name(*ident))).collect();
                // rewrite the pattern using the new names (the old
                // ones have already been applied):
                let rewritten_pat = {
                    // nested binding to allow borrow to expire:
                    let mut rename_fld = IdentRenamer{renames: &mut new_pending_renames};
                    rename_fld.fold_pat(expanded_pat)
                };
                // add them to the existing pending renames:
                fld.cx.syntax_env.info().pending_renames
                      .extend(new_pending_renames);
                Local {
                    id: id,
                    ty: expanded_ty,
                    pat: rewritten_pat,
                    // also, don't forget to expand the init:
                    init: init.map(|e| fld.fold_expr(e)),
                    span: span,
                    attrs: fold::fold_thin_attrs(attrs, fld),
                }
            });
            SmallVector::one(Stmt {
                id: stmt.id,
                node: StmtKind::Local(rewritten_local),
                span: stmt.span,
            })
        }
        _ => noop_fold_stmt(stmt, fld),
    }
}

// expand the arm of a 'match', renaming for macro hygiene
fn expand_arm(arm: ast::Arm, fld: &mut MacroExpander) -> ast::Arm {
    // expand pats... they might contain macro uses:
    let expanded_pats = arm.pats.move_map(|pat| fld.fold_pat(pat));
    if expanded_pats.is_empty() {
        panic!("encountered match arm with 0 patterns");
    }

    // apply renaming and then expansion to the guard and the body:
    let ((rewritten_guard, rewritten_body), rewritten_pats) =
        rename_in_scope(expanded_pats,
                        fld,
                        (arm.guard, arm.body),
                        |rename_fld, fld, (ag, ab)|{
        let rewritten_guard = ag.map(|g| fld.fold_expr(rename_fld.fold_expr(g)));
        let rewritten_body = fld.fold_expr(rename_fld.fold_expr(ab));
        (rewritten_guard, rewritten_body)
    });

    ast::Arm {
        attrs: fold::fold_attrs(arm.attrs, fld),
        pats: rewritten_pats,
        guard: rewritten_guard,
        body: rewritten_body,
    }
}

fn rename_in_scope<X, F>(pats: Vec<P<ast::Pat>>,
                         fld: &mut MacroExpander,
                         x: X,
                         f: F)
                         -> (X, Vec<P<ast::Pat>>)
    where F: Fn(&mut IdentRenamer, &mut MacroExpander, X) -> X
{
    // all of the pats must have the same set of bindings, so use the
    // first one to extract them and generate new names:
    let idents = pattern_bindings(&pats[0]);
    let new_renames = idents.into_iter().map(|id| (id, fresh_name(id))).collect();
    // apply the renaming, but only to the PatIdents:
    let mut rename_pats_fld = PatIdentRenamer{renames:&new_renames};
    let rewritten_pats = pats.move_map(|pat| rename_pats_fld.fold_pat(pat));

    let mut rename_fld = IdentRenamer{ renames:&new_renames };
    (f(&mut rename_fld, fld, x), rewritten_pats)
}

/// A visitor that extracts the PatKind::Ident (binding) paths
/// from a given thingy and puts them in a mutable
/// array
#[derive(Clone)]
struct PatIdentFinder {
    ident_accumulator: Vec<ast::Ident>
}

impl Visitor for PatIdentFinder {
    fn visit_pat(&mut self, pattern: &ast::Pat) {
        match *pattern {
            ast::Pat { id: _, node: PatKind::Ident(_, ref path1, ref inner), span: _ } => {
                self.ident_accumulator.push(path1.node);
                // visit optional subpattern of PatKind::Ident:
                if let Some(ref subpat) = *inner {
                    self.visit_pat(subpat)
                }
            }
            // use the default traversal for non-PatIdents
            _ => visit::walk_pat(self, pattern)
        }
    }
}

/// find the PatKind::Ident paths in a pattern
fn pattern_bindings(pat: &ast::Pat) -> Vec<ast::Ident> {
    let mut name_finder = PatIdentFinder{ident_accumulator:Vec::new()};
    name_finder.visit_pat(pat);
    name_finder.ident_accumulator
}

/// find the PatKind::Ident paths in a
fn fn_decl_arg_bindings(fn_decl: &ast::FnDecl) -> Vec<ast::Ident> {
    let mut pat_idents = PatIdentFinder{ident_accumulator:Vec::new()};
    for arg in &fn_decl.inputs {
        pat_idents.visit_pat(&arg.pat);
    }
    pat_idents.ident_accumulator
}

// expand a block. pushes a new exts_frame, then calls expand_block_elts
pub fn expand_block(blk: P<Block>, fld: &mut MacroExpander) -> P<Block> {
    // see note below about treatment of exts table
    with_exts_frame!(fld.cx.syntax_env,false,
                     expand_block_elts(blk, fld))
}

// expand the elements of a block.
pub fn expand_block_elts(b: P<Block>, fld: &mut MacroExpander) -> P<Block> {
    b.map(|Block {id, stmts, rules, span}| {
        let new_stmts = stmts.into_iter().flat_map(|x| {
            // perform pending renames and expand macros in the statement
            fld.fold_stmt(x).into_iter()
        }).collect();
        Block {
            id: fld.new_id(id),
            stmts: new_stmts,
            rules: rules,
            span: span
        }
    })
}

fn expand_pat(p: P<ast::Pat>, fld: &mut MacroExpander) -> P<ast::Pat> {
    match p.node {
        PatKind::Mac(_) => {}
        _ => return noop_fold_pat(p, fld)
    }
    p.and_then(|ast::Pat {node, span, ..}| {
        match node {
            PatKind::Mac(mac) => expand_mac_invoc(mac, None, Vec::new(), span, fld),
            _ => unreachable!()
        }
    })
}

/// A tree-folder that applies every rename in its (mutable) list
/// to every identifier, including both bindings and varrefs
/// (and lots of things that will turn out to be neither)
pub struct IdentRenamer<'a> {
    renames: &'a mtwt::RenameList,
}

impl<'a> Folder for IdentRenamer<'a> {
    fn fold_ident(&mut self, id: Ident) -> Ident {
        mtwt::apply_renames(self.renames, id)
    }
    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        fold::noop_fold_mac(mac, self)
    }
}

/// A tree-folder that applies every rename in its list to
/// the idents that are in PatKind::Ident patterns. This is more narrowly
/// focused than IdentRenamer, and is needed for FnDecl,
/// where we want to rename the args but not the fn name or the generics etc.
pub struct PatIdentRenamer<'a> {
    renames: &'a mtwt::RenameList,
}

impl<'a> Folder for PatIdentRenamer<'a> {
    fn fold_pat(&mut self, pat: P<ast::Pat>) -> P<ast::Pat> {
        match pat.node {
            PatKind::Ident(..) => {},
            _ => return noop_fold_pat(pat, self)
        }

        pat.map(|ast::Pat {id, node, span}| match node {
            PatKind::Ident(binding_mode, Spanned{span: sp, node: ident}, sub) => {
                let new_ident = mtwt::apply_renames(self.renames, ident);
                let new_node =
                    PatKind::Ident(binding_mode,
                                  Spanned{span: sp, node: new_ident},
                                  sub.map(|p| self.fold_pat(p)));
                ast::Pat {
                    id: id,
                    node: new_node,
                    span: span,
                }
            },
            _ => unreachable!()
        })
    }
    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        fold::noop_fold_mac(mac, self)
    }
}

fn expand_multi_modified(a: Annotatable, fld: &mut MacroExpander) -> SmallVector<Annotatable> {
    let new_items: SmallVector<Annotatable> = match a {
        Annotatable::Item(it) => match it.node {
            ast::ItemKind::Mac(..) => {
                it.and_then(|it| match it.node {
                    ItemKind::Mac(mac) =>
                        expand_mac_invoc(mac, Some(it.ident), it.attrs, it.span, fld),
                    _ => unreachable!(),
                })
            }
            ast::ItemKind::Mod(_) | ast::ItemKind::ForeignMod(_) => {
                let valid_ident =
                    it.ident.name != keywords::Invalid.name();

                if valid_ident {
                    fld.cx.mod_push(it.ident);
                }
                let macro_use = contains_macro_use(fld, &it.attrs);
                let result = with_exts_frame!(fld.cx.syntax_env,
                                              macro_use,
                                              noop_fold_item(it, fld));
                if valid_ident {
                    fld.cx.mod_pop();
                }
                result
            },
            _ => noop_fold_item(it, fld),
        }.into_iter().map(|i| Annotatable::Item(i)).collect(),

        Annotatable::TraitItem(it) => {
            expand_trait_item(it.unwrap(), fld).into_iter().
                map(|it| Annotatable::TraitItem(P(it))).collect()
        }

        Annotatable::ImplItem(ii) => {
            expand_impl_item(ii.unwrap(), fld).into_iter().
                map(|ii| Annotatable::ImplItem(P(ii))).collect()
        }
    };

    new_items.into_iter().flat_map(|a| decorate(a, fld)).collect()
}

fn decorate(a: Annotatable, fld: &mut MacroExpander) -> SmallVector<Annotatable> {
    let mut decorator_items = SmallVector::zero();
    let mut new_attrs = Vec::new();
    expand_decorators(a.clone(), fld, &mut decorator_items, &mut new_attrs);

    let mut new_items = SmallVector::one(a.fold_attrs(new_attrs));
    new_items.push_all(decorator_items);
    new_items
}

fn expand_decorators(a: Annotatable,
                     fld: &mut MacroExpander,
                     decorator_items: &mut SmallVector<Annotatable>,
                     new_attrs: &mut Vec<ast::Attribute>)
{
    for attr in a.attrs() {
        let mname = intern(&attr.name());
        match fld.cx.syntax_env.find(mname) {
            Some(rc) => match *rc {
                MultiDecorator(ref dec) => {
                    attr::mark_used(&attr);

                    fld.cx.bt_push(ExpnInfo {
                        call_site: attr.span,
                        callee: NameAndSpan {
                            format: MacroAttribute(mname),
                            span: Some(attr.span),
                            // attributes can do whatever they like,
                            // for now.
                            allow_internal_unstable: true,
                        }
                    });

                    let mut items: SmallVector<Annotatable> = SmallVector::zero();
                    dec.expand(fld.cx,
                               attr.span,
                               &attr.node.value,
                               &a,
                               &mut |ann| items.push(ann));

                    for item in items {
                        for configured_item in item.fold_with(&mut fld.strip_unconfigured()) {
                            decorator_items.extend(expand_annotatable(configured_item, fld));
                        }
                    }

                    fld.cx.bt_pop();
                }
                _ => new_attrs.push((*attr).clone()),
            },
            _ => new_attrs.push((*attr).clone()),
        }
    }
}

fn expand_annotatable(mut item: Annotatable, fld: &mut MacroExpander) -> SmallVector<Annotatable> {
    let mut multi_modifier = None;
    item = item.map_attrs(|mut attrs| {
        for i in 0..attrs.len() {
            if let Some(extension) = fld.cx.syntax_env.find(intern(&attrs[i].name())) {
                if let MultiModifier(..) = *extension {
                    multi_modifier = Some((attrs.remove(i), extension));
                    break;
                }
            }
        }
        attrs
    });

    match multi_modifier {
        None => expand_multi_modified(item, fld),
        Some((attr, extension)) => match *extension {
            MultiModifier(ref mac) => {
                attr::mark_used(&attr);
                fld.cx.bt_push(ExpnInfo {
                    call_site: attr.span,
                    callee: NameAndSpan {
                        format: MacroAttribute(intern(&attr.name())),
                        span: Some(attr.span),
                        // attributes can do whatever they like, for now
                        allow_internal_unstable: true,
                    }
                });
                let modified = mac.expand(fld.cx, attr.span, &attr.node.value, item);
                fld.cx.bt_pop();
                modified.into_iter().flat_map(|it| expand_annotatable(it, fld)).collect()
            }
            _ => unreachable!(),
        }
    }
}

fn expand_impl_item(ii: ast::ImplItem, fld: &mut MacroExpander)
                 -> SmallVector<ast::ImplItem> {
    match ii.node {
        ast::ImplItemKind::Method(..) => SmallVector::one(ast::ImplItem {
            id: ii.id,
            ident: ii.ident,
            attrs: ii.attrs,
            vis: ii.vis,
            defaultness: ii.defaultness,
            node: match ii.node {
                ast::ImplItemKind::Method(sig, body) => {
                    let (sig, body) = expand_and_rename_method(sig, body, fld);
                    ast::ImplItemKind::Method(sig, body)
                }
                _ => unreachable!()
            },
            span: ii.span,
        }),
        ast::ImplItemKind::Macro(mac) => {
            expand_mac_invoc(mac, None, ii.attrs, ii.span, fld)
        }
        _ => fold::noop_fold_impl_item(ii, fld)
    }
}

fn expand_trait_item(ti: ast::TraitItem, fld: &mut MacroExpander)
                     -> SmallVector<ast::TraitItem> {
    match ti.node {
        ast::TraitItemKind::Method(_, Some(_)) => {
            SmallVector::one(ast::TraitItem {
                id: ti.id,
                ident: ti.ident,
                attrs: ti.attrs,
                node: match ti.node  {
                    ast::TraitItemKind::Method(sig, Some(body)) => {
                        let (sig, body) = expand_and_rename_method(sig, body, fld);
                        ast::TraitItemKind::Method(sig, Some(body))
                    }
                    _ => unreachable!()
                },
                span: ti.span,
            })
        }
        ast::TraitItemKind::Macro(mac) => {
            expand_mac_invoc(mac, None, ti.attrs, ti.span, fld)
        }
        _ => fold::noop_fold_trait_item(ti, fld)
    }
}

/// Given a fn_decl and a block and a MacroExpander, expand the fn_decl, then use the
/// PatIdents in its arguments to perform renaming in the FnDecl and
/// the block, returning both the new FnDecl and the new Block.
fn expand_and_rename_fn_decl_and_block(fn_decl: P<ast::FnDecl>, block: P<ast::Block>,
                                       fld: &mut MacroExpander)
                                       -> (P<ast::FnDecl>, P<ast::Block>) {
    let expanded_decl = fld.fold_fn_decl(fn_decl);
    let idents = fn_decl_arg_bindings(&expanded_decl);
    let renames =
        idents.iter().map(|id| (*id,fresh_name(*id))).collect();
    // first, a renamer for the PatIdents, for the fn_decl:
    let mut rename_pat_fld = PatIdentRenamer{renames: &renames};
    let rewritten_fn_decl = rename_pat_fld.fold_fn_decl(expanded_decl);
    // now, a renamer for *all* idents, for the body:
    let mut rename_fld = IdentRenamer{renames: &renames};
    let rewritten_body = fld.fold_block(rename_fld.fold_block(block));
    (rewritten_fn_decl,rewritten_body)
}

fn expand_and_rename_method(sig: ast::MethodSig, body: P<ast::Block>,
                            fld: &mut MacroExpander)
                            -> (ast::MethodSig, P<ast::Block>) {
    let (rewritten_fn_decl, rewritten_body)
        = expand_and_rename_fn_decl_and_block(sig.decl, body, fld);
    (ast::MethodSig {
        generics: fld.fold_generics(sig.generics),
        abi: sig.abi,
        unsafety: sig.unsafety,
        constness: sig.constness,
        decl: rewritten_fn_decl
    }, rewritten_body)
}

pub fn expand_type(t: P<ast::Ty>, fld: &mut MacroExpander) -> P<ast::Ty> {
    let t = match t.node.clone() {
        ast::TyKind::Mac(mac) => {
            if fld.cx.ecfg.features.unwrap().type_macros {
                expand_mac_invoc(mac, None, Vec::new(), t.span, fld)
            } else {
                feature_gate::emit_feature_err(
                    &fld.cx.parse_sess.span_diagnostic,
                    "type_macros",
                    t.span,
                    feature_gate::GateIssue::Language,
                    "type macros are experimental");

                DummyResult::raw_ty(t.span)
            }
        }
        _ => t
    };

    fold::noop_fold_ty(t, fld)
}

/// A tree-folder that performs macro expansion
pub struct MacroExpander<'a, 'b:'a> {
    pub cx: &'a mut ExtCtxt<'b>,
}

impl<'a, 'b> MacroExpander<'a, 'b> {
    pub fn new(cx: &'a mut ExtCtxt<'b>) -> MacroExpander<'a, 'b> {
        MacroExpander { cx: cx }
    }

    fn strip_unconfigured(&mut self) -> StripUnconfigured {
        StripUnconfigured {
            config: &self.cx.cfg,
            should_test: self.cx.ecfg.should_test,
            sess: self.cx.parse_sess,
            features: self.cx.ecfg.features,
        }
    }

    fn load_macros<T: MacroGenerable>(&mut self, node: &T) {
        struct MacroLoadingVisitor<'a, 'b: 'a>{
            cx: &'a mut ExtCtxt<'b>,
            at_crate_root: bool,
        }

        impl<'a, 'b> Visitor for MacroLoadingVisitor<'a, 'b> {
            fn visit_mac(&mut self, _: &ast::Mac) {}
            fn visit_item(&mut self, item: &ast::Item) {
                if let ast::ItemKind::ExternCrate(..) = item.node {
                    // We need to error on `#[macro_use] extern crate` when it isn't at the
                    // crate root, because `$crate` won't work properly.
                    for def in self.cx.loader.load_crate(item, self.at_crate_root) {
                        self.cx.insert_macro(def);
                    }
                } else {
                    let at_crate_root = ::std::mem::replace(&mut self.at_crate_root, false);
                    visit::walk_item(self, item);
                    self.at_crate_root = at_crate_root;
                }
            }
            fn visit_block(&mut self, block: &ast::Block) {
                let at_crate_root = ::std::mem::replace(&mut self.at_crate_root, false);
                visit::walk_block(self, block);
                self.at_crate_root = at_crate_root;
            }
        }

        node.visit_with(&mut MacroLoadingVisitor {
            at_crate_root: self.cx.syntax_env.is_crate_root(),
            cx: self.cx,
        });
    }
}

impl<'a, 'b> Folder for MacroExpander<'a, 'b> {
    fn fold_crate(&mut self, c: Crate) -> Crate {
        self.cx.filename = Some(self.cx.parse_sess.codemap().span_to_filename(c.span));
        noop_fold_crate(c, self)
    }

    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        expr.and_then(|expr| expand_expr(expr, self))
    }

    fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        expr.and_then(|expr| match expr.node {
            ast::ExprKind::Mac(mac) =>
                expand_mac_invoc(mac, None, expr.attrs.into(), expr.span, self),
            _ => Some(expand_expr(expr, self)),
        })
    }

    fn fold_pat(&mut self, pat: P<ast::Pat>) -> P<ast::Pat> {
        expand_pat(pat, self)
    }

    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        use std::mem::replace;
        let result;
        if let ast::ItemKind::Mod(ast::Mod { inner, .. }) = item.node {
            if item.span.contains(inner) {
                self.push_mod_path(item.ident, &item.attrs);
                result = expand_item(item, self);
                self.pop_mod_path();
            } else {
                let filename = if inner != syntax_pos::DUMMY_SP {
                    Some(self.cx.parse_sess.codemap().span_to_filename(inner))
                } else { None };
                let orig_filename = replace(&mut self.cx.filename, filename);
                let orig_mod_path_stack = replace(&mut self.cx.mod_path_stack, Vec::new());
                result = expand_item(item, self);
                self.cx.filename = orig_filename;
                self.cx.mod_path_stack = orig_mod_path_stack;
            }
        } else {
            result = expand_item(item, self);
        }
        result
    }

    fn fold_item_kind(&mut self, item: ast::ItemKind) -> ast::ItemKind {
        expand_item_kind(item, self)
    }

    fn fold_stmt(&mut self, stmt: ast::Stmt) -> SmallVector<ast::Stmt> {
        expand_stmt(stmt, self)
    }

    fn fold_block(&mut self, block: P<Block>) -> P<Block> {
        let was_in_block = ::std::mem::replace(&mut self.cx.in_block, true);
        let result = expand_block(block, self);
        self.cx.in_block = was_in_block;
        result
    }

    fn fold_arm(&mut self, arm: ast::Arm) -> ast::Arm {
        expand_arm(arm, self)
    }

    fn fold_trait_item(&mut self, i: ast::TraitItem) -> SmallVector<ast::TraitItem> {
        expand_annotatable(Annotatable::TraitItem(P(i)), self)
            .into_iter().map(|i| i.expect_trait_item()).collect()
    }

    fn fold_impl_item(&mut self, i: ast::ImplItem) -> SmallVector<ast::ImplItem> {
        expand_annotatable(Annotatable::ImplItem(P(i)), self)
            .into_iter().map(|i| i.expect_impl_item()).collect()
    }

    fn fold_ty(&mut self, ty: P<ast::Ty>) -> P<ast::Ty> {
        expand_type(ty, self)
    }
}

impl<'a, 'b> MacroExpander<'a, 'b> {
    fn push_mod_path(&mut self, id: Ident, attrs: &[ast::Attribute]) {
        let default_path = id.name.as_str();
        let file_path = match ::attr::first_attr_value_str_by_name(attrs, "path") {
            Some(d) => d,
            None => default_path,
        };
        self.cx.mod_path_stack.push(file_path)
    }

    fn pop_mod_path(&mut self) {
        self.cx.mod_path_stack.pop().unwrap();
    }
}

pub struct ExpansionConfig<'feat> {
    pub crate_name: String,
    pub features: Option<&'feat Features>,
    pub recursion_limit: usize,
    pub trace_mac: bool,
    pub should_test: bool, // If false, strip `#[test]` nodes
}

macro_rules! feature_tests {
    ($( fn $getter:ident = $field:ident, )*) => {
        $(
            pub fn $getter(&self) -> bool {
                match self.features {
                    Some(&Features { $field: true, .. }) => true,
                    _ => false,
                }
            }
        )*
    }
}

impl<'feat> ExpansionConfig<'feat> {
    pub fn default(crate_name: String) -> ExpansionConfig<'static> {
        ExpansionConfig {
            crate_name: crate_name,
            features: None,
            recursion_limit: 64,
            trace_mac: false,
            should_test: false,
        }
    }

    feature_tests! {
        fn enable_quotes = quote,
        fn enable_asm = asm,
        fn enable_log_syntax = log_syntax,
        fn enable_concat_idents = concat_idents,
        fn enable_trace_macros = trace_macros,
        fn enable_allow_internal_unstable = allow_internal_unstable,
        fn enable_custom_derive = custom_derive,
        fn enable_pushpop_unsafe = pushpop_unsafe,
    }
}

pub fn expand_crate(mut cx: ExtCtxt,
                    user_exts: Vec<NamedSyntaxExtension>,
                    mut c: Crate) -> (Crate, HashSet<Name>) {
    if std_inject::no_core(&c) {
        cx.crate_root = None;
    } else if std_inject::no_std(&c) {
        cx.crate_root = Some("core");
    } else {
        cx.crate_root = Some("std");
    }
    let ret = {
        let mut expander = MacroExpander::new(&mut cx);

        for (name, extension) in user_exts {
            expander.cx.syntax_env.insert(name, extension);
        }

        let items = SmallVector::many(c.module.items);
        expander.load_macros(&items);
        c.module.items = items.into();

        let err_count = cx.parse_sess.span_diagnostic.err_count();
        let mut ret = expander.fold_crate(c);
        ret.exported_macros = expander.cx.exported_macros.clone();

        if cx.parse_sess.span_diagnostic.err_count() > err_count {
            cx.parse_sess.span_diagnostic.abort_if_errors();
        }

        ret
    };
    return (ret, cx.syntax_env.names);
}

// HYGIENIC CONTEXT EXTENSION:
// all of these functions are for walking over
// ASTs and making some change to the context of every
// element that has one. a CtxtFn is a trait-ified
// version of a closure in (SyntaxContext -> SyntaxContext).
// the ones defined here include:
// Marker - add a mark to a context

// A Marker adds the given mark to the syntax context and
// sets spans' `expn_id` to the given expn_id (unless it is `None`).
struct Marker { mark: Mrk, expn_id: Option<ExpnId> }

impl Folder for Marker {
    fn fold_ident(&mut self, id: Ident) -> Ident {
        ast::Ident::new(id.name, mtwt::apply_mark(self.mark, id.ctxt))
    }
    fn fold_mac(&mut self, Spanned {node, span}: ast::Mac) -> ast::Mac {
        Spanned {
            node: Mac_ {
                path: self.fold_path(node.path),
                tts: self.fold_tts(node.tts),
            },
            span: self.new_span(span),
        }
    }

    fn new_span(&mut self, mut span: Span) -> Span {
        if let Some(expn_id) = self.expn_id {
            span.expn_id = expn_id;
        }
        span
    }
}

// apply a given mark to the given token trees. Used prior to expansion of a macro.
fn mark_tts(tts: Vec<TokenTree>, m: Mrk) -> Vec<TokenTree> {
    noop_fold_tts(tts, &mut Marker{mark:m, expn_id: None})
}


#[cfg(test)]
mod tests {
    use super::{pattern_bindings, expand_crate};
    use super::{PatIdentFinder, IdentRenamer, PatIdentRenamer, ExpansionConfig};
    use ast;
    use ast::Name;
    use syntax_pos;
    use ext::base::{ExtCtxt, DummyMacroLoader};
    use ext::mtwt;
    use fold::Folder;
    use parse;
    use parse::token;
    use util::parser_testing::{string_to_parser};
    use util::parser_testing::{string_to_pat, string_to_crate, strs_to_idents};
    use visit;
    use visit::Visitor;

    // a visitor that extracts the paths
    // from a given thingy and puts them in a mutable
    // array (passed in to the traversal)
    #[derive(Clone)]
    struct PathExprFinderContext {
        path_accumulator: Vec<ast::Path> ,
    }

    impl Visitor for PathExprFinderContext {
        fn visit_expr(&mut self, expr: &ast::Expr) {
            if let ast::ExprKind::Path(None, ref p) = expr.node {
                self.path_accumulator.push(p.clone());
            }
            visit::walk_expr(self, expr);
        }
    }

    // find the variable references in a crate
    fn crate_varrefs(the_crate : &ast::Crate) -> Vec<ast::Path> {
        let mut path_finder = PathExprFinderContext{path_accumulator:Vec::new()};
        visit::walk_crate(&mut path_finder, the_crate);
        path_finder.path_accumulator
    }

    /// A Visitor that extracts the identifiers from a thingy.
    // as a side note, I'm starting to want to abstract over these....
    struct IdentFinder {
        ident_accumulator: Vec<ast::Ident>
    }

    impl Visitor for IdentFinder {
        fn visit_ident(&mut self, _: syntax_pos::Span, id: ast::Ident){
            self.ident_accumulator.push(id);
        }
    }

    /// Find the idents in a crate
    fn crate_idents(the_crate: &ast::Crate) -> Vec<ast::Ident> {
        let mut ident_finder = IdentFinder{ident_accumulator: Vec::new()};
        visit::walk_crate(&mut ident_finder, the_crate);
        ident_finder.ident_accumulator
    }

    // these following tests are quite fragile, in that they don't test what
    // *kind* of failure occurs.

    fn test_ecfg() -> ExpansionConfig<'static> {
        ExpansionConfig::default("test".to_string())
    }

    // make sure that macros can't escape fns
    #[should_panic]
    #[test] fn macros_cant_escape_fns_test () {
        let src = "fn bogus() {macro_rules! z (() => (3+4));}\
                   fn inty() -> i32 { z!() }".to_string();
        let sess = parse::ParseSess::new();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess).unwrap();
        // should fail:
        let mut loader = DummyMacroLoader;
        let ecx = ExtCtxt::new(&sess, vec![], test_ecfg(), &mut loader);
        expand_crate(ecx, vec![], crate_ast);
    }

    // make sure that macros can't escape modules
    #[should_panic]
    #[test] fn macros_cant_escape_mods_test () {
        let src = "mod foo {macro_rules! z (() => (3+4));}\
                   fn inty() -> i32 { z!() }".to_string();
        let sess = parse::ParseSess::new();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess).unwrap();
        let mut loader = DummyMacroLoader;
        let ecx = ExtCtxt::new(&sess, vec![], test_ecfg(), &mut loader);
        expand_crate(ecx, vec![], crate_ast);
    }

    // macro_use modules should allow macros to escape
    #[test] fn macros_can_escape_flattened_mods_test () {
        let src = "#[macro_use] mod foo {macro_rules! z (() => (3+4));}\
                   fn inty() -> i32 { z!() }".to_string();
        let sess = parse::ParseSess::new();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess).unwrap();
        let mut loader = DummyMacroLoader;
        let ecx = ExtCtxt::new(&sess, vec![], test_ecfg(), &mut loader);
        expand_crate(ecx, vec![], crate_ast);
    }

    fn expand_crate_str(crate_str: String) -> ast::Crate {
        let ps = parse::ParseSess::new();
        let crate_ast = panictry!(string_to_parser(&ps, crate_str).parse_crate_mod());
        // the cfg argument actually does matter, here...
        let mut loader = DummyMacroLoader;
        let ecx = ExtCtxt::new(&ps, vec![], test_ecfg(), &mut loader);
        expand_crate(ecx, vec![], crate_ast).0
    }

    // find the pat_ident paths in a crate
    fn crate_bindings(the_crate : &ast::Crate) -> Vec<ast::Ident> {
        let mut name_finder = PatIdentFinder{ident_accumulator:Vec::new()};
        visit::walk_crate(&mut name_finder, the_crate);
        name_finder.ident_accumulator
    }

    #[test] fn macro_tokens_should_match(){
        expand_crate_str(
            "macro_rules! m((a)=>(13)) ;fn main(){m!(a);}".to_string());
    }

    // should be able to use a bound identifier as a literal in a macro definition:
    #[test] fn self_macro_parsing(){
        expand_crate_str(
            "macro_rules! foo ((zz) => (287;));
            fn f(zz: i32) {foo!(zz);}".to_string()
            );
    }

    // create a really evil test case where a $x appears inside a binding of $x
    // but *shouldn't* bind because it was inserted by a different macro....
    // can't write this test case until we have macro-generating macros.

    #[test]
    fn fmt_in_macro_used_inside_module_macro() {
        let crate_str = "macro_rules! fmt_wrap(($b:expr)=>($b.to_string()));
macro_rules! foo_module (() => (mod generated { fn a() { let xx = 147; fmt_wrap!(xx);}}));
foo_module!();
".to_string();
        let cr = expand_crate_str(crate_str);
        // find the xx binding
        let bindings = crate_bindings(&cr);
        let cxbinds: Vec<&ast::Ident> =
            bindings.iter().filter(|b| b.name.as_str() == "xx").collect();
        let cxbinds: &[&ast::Ident] = &cxbinds[..];
        let cxbind = match (cxbinds.len(), cxbinds.get(0)) {
            (1, Some(b)) => *b,
            _ => panic!("expected just one binding for ext_cx")
        };
        let resolved_binding = mtwt::resolve(*cxbind);
        let varrefs = crate_varrefs(&cr);

        // the xx binding should bind all of the xx varrefs:
        for (idx,v) in varrefs.iter().filter(|p| {
            p.segments.len() == 1
            && p.segments[0].identifier.name.as_str() == "xx"
        }).enumerate() {
            if mtwt::resolve(v.segments[0].identifier) != resolved_binding {
                println!("uh oh, xx binding didn't match xx varref:");
                println!("this is xx varref \\# {}", idx);
                println!("binding: {}", cxbind);
                println!("resolves to: {}", resolved_binding);
                println!("varref: {}", v.segments[0].identifier);
                println!("resolves to: {}",
                         mtwt::resolve(v.segments[0].identifier));
                mtwt::with_sctable(|x| mtwt::display_sctable(x));
            }
            assert_eq!(mtwt::resolve(v.segments[0].identifier),
                       resolved_binding);
        };
    }

    #[test]
    fn pat_idents(){
        let pat = string_to_pat(
            "(a,Foo{x:c @ (b,9),y:Bar(4,d)})".to_string());
        let idents = pattern_bindings(&pat);
        assert_eq!(idents, strs_to_idents(vec!("a","c","b","d")));
    }

    // test the list of identifier patterns gathered by the visitor. Note that
    // 'None' is listed as an identifier pattern because we don't yet know that
    // it's the name of a 0-ary variant, and that 'i' appears twice in succession.
    #[test]
    fn crate_bindings_test(){
        let the_crate = string_to_crate("fn main (a: i32) -> i32 {|b| {
        match 34 {None => 3, Some(i) | i => j, Foo{k:z,l:y} => \"banana\"}} }".to_string());
        let idents = crate_bindings(&the_crate);
        assert_eq!(idents, strs_to_idents(vec!("a","b","None","i","i","z","y")));
    }

    // test the IdentRenamer directly
    #[test]
    fn ident_renamer_test () {
        let the_crate = string_to_crate("fn f(x: i32){let x = x; x}".to_string());
        let f_ident = token::str_to_ident("f");
        let x_ident = token::str_to_ident("x");
        let int_ident = token::str_to_ident("i32");
        let renames = vec!((x_ident,Name(16)));
        let mut renamer = IdentRenamer{renames: &renames};
        let renamed_crate = renamer.fold_crate(the_crate);
        let idents = crate_idents(&renamed_crate);
        let resolved : Vec<ast::Name> = idents.iter().map(|id| mtwt::resolve(*id)).collect();
        assert_eq!(resolved, [f_ident.name,Name(16),int_ident.name,Name(16),Name(16),Name(16)]);
    }

    // test the PatIdentRenamer; only PatIdents get renamed
    #[test]
    fn pat_ident_renamer_test () {
        let the_crate = string_to_crate("fn f(x: i32){let x = x; x}".to_string());
        let f_ident = token::str_to_ident("f");
        let x_ident = token::str_to_ident("x");
        let int_ident = token::str_to_ident("i32");
        let renames = vec!((x_ident,Name(16)));
        let mut renamer = PatIdentRenamer{renames: &renames};
        let renamed_crate = renamer.fold_crate(the_crate);
        let idents = crate_idents(&renamed_crate);
        let resolved : Vec<ast::Name> = idents.iter().map(|id| mtwt::resolve(*id)).collect();
        let x_name = x_ident.name;
        assert_eq!(resolved, [f_ident.name,Name(16),int_ident.name,Name(16),x_name,x_name]);
    }
}
