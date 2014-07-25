// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{P, Block, Crate, DeclLocal, ExprMac, PatMac};
use ast::{Local, Ident, MacInvocTT};
use ast::{ItemMac, Mrk, Stmt, StmtDecl, StmtMac, StmtExpr, StmtSemi};
use ast::TokenTree;
use ast;
use ext::mtwt;
use ext::build::AstBuilder;
use attr;
use attr::AttrMetaMethods;
use codemap;
use codemap::{Span, Spanned, ExpnInfo, NameAndSpan, MacroBang, MacroAttribute};
use ext::base::*;
use fold;
use fold::*;
use parse;
use parse::token::{fresh_mark, fresh_name, intern};
use parse::token;
use visit;
use visit::Visitor;
use util::small_vector::SmallVector;

use std::gc::{Gc, GC};


fn expand_expr(e: Gc<ast::Expr>, fld: &mut MacroExpander) -> Gc<ast::Expr> {
    match e.node {
        // expr_mac should really be expr_ext or something; it's the
        // entry-point for all syntax extensions.
        ExprMac(ref mac) => {
            let expanded_expr = match expand_mac_invoc(mac,&e.span,
                                                       |r|{r.make_expr()},
                                                       |expr,fm|{mark_expr(expr,fm)},
                                                       fld) {
                Some(expr) => expr,
                None => {
                    return DummyResult::raw_expr(e.span);
                }
            };

            // Keep going, outside-in.
            //
            // FIXME(pcwalton): Is it necessary to clone the
            // node here?
            let fully_expanded =
                fld.fold_expr(expanded_expr).node.clone();
            fld.cx.bt_pop();

            box(GC) ast::Expr {
                id: ast::DUMMY_NODE_ID,
                node: fully_expanded,
                span: e.span,
            }
        }

        ast::ExprLoop(loop_block, opt_ident) => {
            let (loop_block, opt_ident) = expand_loop_block(loop_block, opt_ident, fld);
            fld.cx.expr(e.span, ast::ExprLoop(loop_block, opt_ident))
        }

        ast::ExprForLoop(pat, head, body, opt_ident) => {
            let pat = fld.fold_pat(pat);
            let head = fld.fold_expr(head);
            let (body, opt_ident) = expand_loop_block(body, opt_ident, fld);
            fld.cx.expr(e.span, ast::ExprForLoop(pat, head, body, opt_ident))
        }

        ast::ExprFnBlock(fn_decl, block) => {
            let (rewritten_fn_decl, rewritten_block)
                = expand_and_rename_fn_decl_and_block(&*fn_decl, block, fld);
            let new_node = ast::ExprFnBlock(rewritten_fn_decl, rewritten_block);
            box(GC) ast::Expr{id:e.id, node: new_node, span: fld.new_span(e.span)}
        }

        ast::ExprProc(fn_decl, block) => {
            let (rewritten_fn_decl, rewritten_block)
                = expand_and_rename_fn_decl_and_block(&*fn_decl, block, fld);
            let new_node = ast::ExprProc(rewritten_fn_decl, rewritten_block);
            box(GC) ast::Expr{id:e.id, node: new_node, span: fld.new_span(e.span)}
        }

        _ => noop_fold_expr(e, fld)
    }
}

/// Expand a (not-ident-style) macro invocation. Returns the result
/// of expansion and the mark which must be applied to the result.
/// Our current interface doesn't allow us to apply the mark to the
/// result until after calling make_expr, make_items, etc.
fn expand_mac_invoc<T>(mac: &ast::Mac, span: &codemap::Span,
                       parse_thunk: |Box<MacResult>|->Option<T>,
                       mark_thunk: |T,Mrk|->T,
                       fld: &mut MacroExpander)
    -> Option<T> {
    match (*mac).node {
        // it would almost certainly be cleaner to pass the whole
        // macro invocation in, rather than pulling it apart and
        // marking the tts and the ctxt separately. This also goes
        // for the other three macro invocation chunks of code
        // in this file.
        // Token-tree macros:
        MacInvocTT(ref pth, ref tts, _) => {
            if pth.segments.len() > 1u {
                fld.cx.span_err(pth.span,
                                "expected macro name without module \
                                separators");
                // let compilation continue
                return None;
            }
            let extname = pth.segments.get(0).identifier;
            let extnamestr = token::get_ident(extname);
            match fld.cx.syntax_env.find(&extname.name) {
                None => {
                    fld.cx.span_err(
                        pth.span,
                        format!("macro undefined: '{}!'",
                                extnamestr.get()).as_slice());

                    // let compilation continue
                    None
                }
                Some(rc) => match *rc {
                    NormalTT(ref expandfun, exp_span) => {
                        fld.cx.bt_push(ExpnInfo {
                                call_site: *span,
                                callee: NameAndSpan {
                                    name: extnamestr.get().to_string(),
                                    format: MacroBang,
                                    span: exp_span,
                                },
                            });
                        let fm = fresh_mark();
                        let marked_before = mark_tts(tts.as_slice(), fm);

                        // The span that we pass to the expanders we want to
                        // be the root of the call stack. That's the most
                        // relevant span and it's the actual invocation of
                        // the macro.
                        let mac_span = original_span(fld.cx);

                        let expanded = expandfun.expand(fld.cx,
                                                        mac_span.call_site,
                                                        marked_before.as_slice());
                        let parsed = match parse_thunk(expanded) {
                            Some(e) => e,
                            None => {
                                fld.cx.span_err(
                                    pth.span,
                                    format!("non-expression macro in expression position: {}",
                                            extnamestr.get().as_slice()
                                            ).as_slice());
                                return None;
                            }
                        };
                        Some(mark_thunk(parsed,fm))
                    }
                    _ => {
                        fld.cx.span_err(
                            pth.span,
                            format!("'{}' is not a tt-style macro",
                                    extnamestr.get()).as_slice());
                        None
                    }
                }
            }
        }
    }
}

/// Rename loop label and expand its loop body
///
/// The renaming procedure for loop is different in the sense that the loop
/// body is in a block enclosed by loop head so the renaming of loop label
/// must be propagated to the enclosed context.
fn expand_loop_block(loop_block: P<Block>,
                     opt_ident: Option<Ident>,
                     fld: &mut MacroExpander) -> (P<Block>, Option<Ident>) {
    match opt_ident {
        Some(label) => {
            let new_label = fresh_name(&label);
            let rename = (label, new_label);

            // The rename *must not* be added to the pending list of current
            // syntax context otherwise an unrelated `break` or `continue` in
            // the same context will pick that up in the deferred renaming pass
            // and be renamed incorrectly.
            let mut rename_list = vec!(rename);
            let mut rename_fld = IdentRenamer{renames: &mut rename_list};
            let renamed_ident = rename_fld.fold_ident(label);

            // The rename *must* be added to the enclosed syntax context for
            // `break` or `continue` to pick up because by definition they are
            // in a block enclosed by loop head.
            fld.cx.syntax_env.push_frame();
            fld.cx.syntax_env.info().pending_renames.push(rename);
            let expanded_block = expand_block_elts(&*loop_block, fld);
            fld.cx.syntax_env.pop_frame();

            (expanded_block, Some(renamed_ident))
        }
        None => (fld.fold_block(loop_block), opt_ident)
    }
}

// eval $e with a new exts frame.
// must be a macro so that $e isn't evaluated too early.
macro_rules! with_exts_frame (
    ($extsboxexpr:expr,$macros_escape:expr,$e:expr) =>
    ({$extsboxexpr.push_frame();
      $extsboxexpr.info().macros_escape = $macros_escape;
      let result = $e;
      $extsboxexpr.pop_frame();
      result
     })
)

// When we enter a module, record it, for the sake of `module!`
fn expand_item(it: Gc<ast::Item>, fld: &mut MacroExpander)
                   -> SmallVector<Gc<ast::Item>> {
    let it = expand_item_modifiers(it, fld);

    let mut decorator_items = SmallVector::zero();
    let mut new_attrs = Vec::new();
    for attr in it.attrs.iter() {
        let mname = attr.name();

        match fld.cx.syntax_env.find(&intern(mname.get())) {
            Some(rc) => match *rc {
                ItemDecorator(dec_fn) => {
                    attr::mark_used(attr);

                    fld.cx.bt_push(ExpnInfo {
                        call_site: attr.span,
                        callee: NameAndSpan {
                            name: mname.get().to_string(),
                            format: MacroAttribute,
                            span: None
                        }
                    });

                    // we'd ideally decorator_items.push_all(expand_item(item, fld)),
                    // but that double-mut-borrows fld
                    let mut items: SmallVector<Gc<ast::Item>> = SmallVector::zero();
                    dec_fn(fld.cx, attr.span, attr.node.value, it,
                        |item| items.push(item));
                    decorator_items.extend(items.move_iter()
                        .flat_map(|item| expand_item(item, fld).move_iter()));

                    fld.cx.bt_pop();
                }
                _ => new_attrs.push((*attr).clone()),
            },
            _ => new_attrs.push((*attr).clone()),
        }
    }

    let mut new_items = match it.node {
        ast::ItemMac(..) => expand_item_mac(it, fld),
        ast::ItemMod(_) | ast::ItemForeignMod(_) => {
            fld.cx.mod_push(it.ident);
            let macro_escape = contains_macro_escape(new_attrs.as_slice());
            let result = with_exts_frame!(fld.cx.syntax_env,
                                          macro_escape,
                                          noop_fold_item(&*it, fld));
            fld.cx.mod_pop();
            result
        },
        _ => {
            let it = box(GC) ast::Item {
                attrs: new_attrs,
                ..(*it).clone()

            };
            noop_fold_item(&*it, fld)
        }
    };

    new_items.push_all(decorator_items);
    new_items
}

fn expand_item_modifiers(mut it: Gc<ast::Item>, fld: &mut MacroExpander)
                         -> Gc<ast::Item> {
    // partition the attributes into ItemModifiers and others
    let (modifiers, other_attrs) = it.attrs.partitioned(|attr| {
        match fld.cx.syntax_env.find(&intern(attr.name().get())) {
            Some(rc) => match *rc { ItemModifier(_) => true, _ => false },
            _ => false
        }
    });
    // update the attrs, leave everything else alone. Is this mutation really a good idea?
    it = box(GC) ast::Item {
        attrs: other_attrs,
        ..(*it).clone()
    };

    if modifiers.is_empty() {
        return it;
    }

    for attr in modifiers.iter() {
        let mname = attr.name();

        match fld.cx.syntax_env.find(&intern(mname.get())) {
            Some(rc) => match *rc {
                ItemModifier(dec_fn) => {
                    attr::mark_used(attr);
                    fld.cx.bt_push(ExpnInfo {
                        call_site: attr.span,
                        callee: NameAndSpan {
                            name: mname.get().to_string(),
                            format: MacroAttribute,
                            span: None,
                        }
                    });
                    it = dec_fn(fld.cx, attr.span, attr.node.value, it);
                    fld.cx.bt_pop();
                }
                _ => unreachable!()
            },
            _ => unreachable!()
        }
    }

    // expansion may have added new ItemModifiers
    expand_item_modifiers(it, fld)
}

/// Expand item_underscore
fn expand_item_underscore(item: &ast::Item_, fld: &mut MacroExpander) -> ast::Item_ {
    match *item {
        ast::ItemFn(decl, fn_style, abi, ref generics, body) => {
            let (rewritten_fn_decl, rewritten_body)
                = expand_and_rename_fn_decl_and_block(&*decl, body, fld);
            let expanded_generics = fold::fold_generics(generics,fld);
            ast::ItemFn(rewritten_fn_decl, fn_style, abi, expanded_generics, rewritten_body)
        }
        _ => noop_fold_item_underscore(&*item, fld)
    }
}

// does this attribute list contain "macro_escape" ?
fn contains_macro_escape(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, "macro_escape")
}

// Support for item-position macro invocations, exactly the same
// logic as for expression-position macro invocations.
fn expand_item_mac(it: Gc<ast::Item>, fld: &mut MacroExpander)
                       -> SmallVector<Gc<ast::Item>> {
    let (pth, tts) = match it.node {
        ItemMac(codemap::Spanned {
            node: MacInvocTT(ref pth, ref tts, _),
            ..
        }) => {
            (pth, (*tts).clone())
        }
        _ => fld.cx.span_bug(it.span, "invalid item macro invocation")
    };

    let extname = pth.segments.get(0).identifier;
    let extnamestr = token::get_ident(extname);
    let fm = fresh_mark();
    let expanded = match fld.cx.syntax_env.find(&extname.name) {
        None => {
            fld.cx.span_err(pth.span,
                            format!("macro undefined: '{}!'",
                                    extnamestr).as_slice());
            // let compilation continue
            return SmallVector::zero();
        }

        Some(rc) => match *rc {
            NormalTT(ref expander, span) => {
                if it.ident.name != parse::token::special_idents::invalid.name {
                    fld.cx
                    .span_err(pth.span,
                                format!("macro {}! expects no ident argument, \
                                        given '{}'",
                                        extnamestr,
                                        token::get_ident(it.ident)).as_slice());
                    return SmallVector::zero();
                }
                fld.cx.bt_push(ExpnInfo {
                    call_site: it.span,
                    callee: NameAndSpan {
                        name: extnamestr.get().to_string(),
                        format: MacroBang,
                        span: span
                    }
                });
                // mark before expansion:
                let marked_before = mark_tts(tts.as_slice(), fm);
                expander.expand(fld.cx, it.span, marked_before.as_slice())
            }
            IdentTT(ref expander, span) => {
                if it.ident.name == parse::token::special_idents::invalid.name {
                    fld.cx.span_err(pth.span,
                                    format!("macro {}! expects an ident argument",
                                            extnamestr.get()).as_slice());
                    return SmallVector::zero();
                }
                fld.cx.bt_push(ExpnInfo {
                    call_site: it.span,
                    callee: NameAndSpan {
                        name: extnamestr.get().to_string(),
                        format: MacroBang,
                        span: span
                    }
                });
                // mark before expansion:
                let marked_tts = mark_tts(tts.as_slice(), fm);
                expander.expand(fld.cx, it.span, it.ident, marked_tts)
            }
            LetSyntaxTT(ref expander, span) => {
                if it.ident.name == parse::token::special_idents::invalid.name {
                    fld.cx.span_err(pth.span,
                                    format!("macro {}! expects an ident argument",
                                            extnamestr.get()).as_slice());
                    return SmallVector::zero();
                }
                fld.cx.bt_push(ExpnInfo {
                    call_site: it.span,
                    callee: NameAndSpan {
                        name: extnamestr.get().to_string(),
                        format: MacroBang,
                        span: span
                    }
                });
                // DON'T mark before expansion:
                expander.expand(fld.cx, it.span, it.ident, tts)
            }
            _ => {
                fld.cx.span_err(it.span,
                                format!("{}! is not legal in item position",
                                        extnamestr.get()).as_slice());
                return SmallVector::zero();
            }
        }
    };

    let items = match expanded.make_def() {
        Some(MacroDef { name, ext }) => {
            // hidden invariant: this should only be possible as the
            // result of expanding a LetSyntaxTT, and thus doesn't
            // need to be marked. Not that it could be marked anyway.
            // create issue to recommend refactoring here?
            fld.cx.syntax_env.insert(intern(name.as_slice()), ext);
            if attr::contains_name(it.attrs.as_slice(), "macro_export") {
                fld.cx.exported_macros.push(it);
            }
            SmallVector::zero()
        }
        None => {
            match expanded.make_items() {
                Some(items) => {
                    items.move_iter()
                        .map(|i| mark_item(i, fm))
                        .flat_map(|i| fld.fold_item(i).move_iter())
                        .collect()
                }
                None => {
                    fld.cx.span_err(pth.span,
                                    format!("non-item macro in item position: {}",
                                            extnamestr.get()).as_slice());
                    return SmallVector::zero();
                }
            }
        }
    };
    fld.cx.bt_pop();
    return items;
}

/// Expand a stmt
//
// I don't understand why this returns a vector... it looks like we're
// half done adding machinery to allow macros to expand into multiple statements.
fn expand_stmt(s: &Stmt, fld: &mut MacroExpander) -> SmallVector<Gc<Stmt>> {
    let (mac, semi) = match s.node {
        StmtMac(ref mac, semi) => (mac, semi),
        _ => return expand_non_macro_stmt(s, fld)
    };
    let expanded_stmt = match expand_mac_invoc(mac,&s.span,
                                                |r|{r.make_stmt()},
                                                |sts,mrk| {
                                                    mark_stmt(&*sts,mrk)
                                                },
                                                fld) {
        Some(stmt) => stmt,
        None => {
            return SmallVector::zero();
        }
    };

    // Keep going, outside-in.
    let fully_expanded = fld.fold_stmt(&*expanded_stmt);
    fld.cx.bt_pop();
    let fully_expanded: SmallVector<Gc<Stmt>> = fully_expanded.move_iter()
            .map(|s| box(GC) Spanned { span: s.span, node: s.node.clone() })
            .collect();

    fully_expanded.move_iter().map(|s| {
        match s.node {
            StmtExpr(e, stmt_id) if semi => {
                box(GC) Spanned {
                    span: s.span,
                    node: StmtSemi(e, stmt_id)
                }
            }
            _ => s /* might already have a semi */
        }
    }).collect()
}

// expand a non-macro stmt. this is essentially the fallthrough for
// expand_stmt, above.
fn expand_non_macro_stmt(s: &Stmt, fld: &mut MacroExpander)
                         -> SmallVector<Gc<Stmt>> {
    // is it a let?
    match s.node {
        StmtDecl(decl, node_id) => {
            match *decl {
                Spanned {
                    node: DeclLocal(ref local),
                    span: stmt_span
                } => {
                    // take it apart:
                    let Local {
                        ty: _,
                        pat: pat,
                        init: init,
                        id: id,
                        span: span,
                        source: source,
                    } = **local;
                    // expand the pat (it might contain macro uses):
                    let expanded_pat = fld.fold_pat(pat);
                    // find the PatIdents in the pattern:
                    // oh dear heaven... this is going to include the enum
                    // names, as well... but that should be okay, as long as
                    // the new names are gensyms for the old ones.
                    // generate fresh names, push them to a new pending list
                    let idents = pattern_bindings(&*expanded_pat);
                    let mut new_pending_renames =
                        idents.iter().map(|ident| (*ident, fresh_name(ident))).collect();
                    // rewrite the pattern using the new names (the old
                    // ones have already been applied):
                    let rewritten_pat = {
                        // nested binding to allow borrow to expire:
                        let mut rename_fld = IdentRenamer{renames: &mut new_pending_renames};
                        rename_fld.fold_pat(expanded_pat)
                    };
                    // add them to the existing pending renames:
                    fld.cx.syntax_env.info().pending_renames.push_all_move(new_pending_renames);
                    // also, don't forget to expand the init:
                    let new_init_opt = init.map(|e| fld.fold_expr(e));
                    let rewritten_local =
                        box(GC) Local {
                            ty: local.ty,
                            pat: rewritten_pat,
                            init: new_init_opt,
                            id: id,
                            span: span,
                            source: source
                        };
                    SmallVector::one(box(GC) Spanned {
                        node: StmtDecl(box(GC) Spanned {
                                node: DeclLocal(rewritten_local),
                                span: stmt_span
                            },
                            node_id),
                        span: span
                    })
                }
                _ => noop_fold_stmt(s, fld),
            }
        },
        _ => noop_fold_stmt(s, fld),
    }
}

// expand the arm of a 'match', renaming for macro hygiene
fn expand_arm(arm: &ast::Arm, fld: &mut MacroExpander) -> ast::Arm {
    // expand pats... they might contain macro uses:
    let expanded_pats : Vec<Gc<ast::Pat>> = arm.pats.iter().map(|pat| fld.fold_pat(*pat)).collect();
    if expanded_pats.len() == 0 {
        fail!("encountered match arm with 0 patterns");
    }
    // all of the pats must have the same set of bindings, so use the
    // first one to extract them and generate new names:
    let first_pat = expanded_pats.get(0);
    let idents = pattern_bindings(&**first_pat);
    let new_renames =
        idents.iter().map(|id| (*id,fresh_name(id))).collect();
    // apply the renaming, but only to the PatIdents:
    let mut rename_pats_fld = PatIdentRenamer{renames:&new_renames};
    let rewritten_pats =
        expanded_pats.iter().map(|pat| rename_pats_fld.fold_pat(*pat)).collect();
    // apply renaming and then expansion to the guard and the body:
    let mut rename_fld = IdentRenamer{renames:&new_renames};
    let rewritten_guard =
        arm.guard.map(|g| fld.fold_expr(rename_fld.fold_expr(g)));
    let rewritten_body = fld.fold_expr(rename_fld.fold_expr(arm.body));
    ast::Arm {
        attrs: arm.attrs.iter().map(|x| fld.fold_attribute(*x)).collect(),
        pats: rewritten_pats,
        guard: rewritten_guard,
        body: rewritten_body,
    }
}

/// A visitor that extracts the PatIdent (binding) paths
/// from a given thingy and puts them in a mutable
/// array
#[deriving(Clone)]
struct PatIdentFinder {
    ident_accumulator: Vec<ast::Ident> ,
}

impl Visitor<()> for PatIdentFinder {
    fn visit_pat(&mut self, pattern: &ast::Pat, _: ()) {
        match *pattern {
            ast::Pat { id: _, node: ast::PatIdent(_, ref path1, ref inner), span: _ } => {
                self.ident_accumulator.push(path1.node);
                // visit optional subpattern of PatIdent:
                for subpat in inner.iter() {
                    self.visit_pat(&**subpat, ())
                }
            }
            // use the default traversal for non-PatIdents
            _ => visit::walk_pat(self, pattern, ())
        }
    }
}

/// find the PatIdent paths in a pattern
fn pattern_bindings(pat : &ast::Pat) -> Vec<ast::Ident> {
    let mut name_finder = PatIdentFinder{ident_accumulator:Vec::new()};
    name_finder.visit_pat(pat,());
    name_finder.ident_accumulator
}

/// find the PatIdent paths in a
fn fn_decl_arg_bindings(fn_decl: &ast::FnDecl) -> Vec<ast::Ident> {
    let mut pat_idents = PatIdentFinder{ident_accumulator:Vec::new()};
    for arg in fn_decl.inputs.iter() {
        pat_idents.visit_pat(&*arg.pat, ());
    }
    pat_idents.ident_accumulator
}

// expand a block. pushes a new exts_frame, then calls expand_block_elts
fn expand_block(blk: &Block, fld: &mut MacroExpander) -> P<Block> {
    // see note below about treatment of exts table
    with_exts_frame!(fld.cx.syntax_env,false,
                     expand_block_elts(blk, fld))
}

// expand the elements of a block.
fn expand_block_elts(b: &Block, fld: &mut MacroExpander) -> P<Block> {
    let new_view_items = b.view_items.iter().map(|x| fld.fold_view_item(x)).collect();
    let new_stmts =
        b.stmts.iter().flat_map(|x| {
            // perform all pending renames
            let renamed_stmt = {
                let pending_renames = &mut fld.cx.syntax_env.info().pending_renames;
                let mut rename_fld = IdentRenamer{renames:pending_renames};
                rename_fld.fold_stmt(&**x).expect_one("rename_fold didn't return one value")
            };
            // expand macros in the statement
            fld.fold_stmt(&*renamed_stmt).move_iter()
        }).collect();
    let new_expr = b.expr.map(|x| {
        let expr = {
            let pending_renames = &mut fld.cx.syntax_env.info().pending_renames;
            let mut rename_fld = IdentRenamer{renames:pending_renames};
            rename_fld.fold_expr(x)
        };
        fld.fold_expr(expr)
    });
    P(Block {
        view_items: new_view_items,
        stmts: new_stmts,
        expr: new_expr,
        id: fld.new_id(b.id),
        rules: b.rules,
        span: b.span,
    })
}

fn expand_pat(p: Gc<ast::Pat>, fld: &mut MacroExpander) -> Gc<ast::Pat> {
    let (pth, tts) = match p.node {
        PatMac(ref mac) => {
            match mac.node {
                MacInvocTT(ref pth, ref tts, _) => {
                    (pth, (*tts).clone())
                }
            }
        }
        _ => return noop_fold_pat(p, fld),
    };
    if pth.segments.len() > 1u {
        fld.cx.span_err(pth.span, "expected macro name without module separators");
        return DummyResult::raw_pat(p.span);
    }
    let extname = pth.segments.get(0).identifier;
    let extnamestr = token::get_ident(extname);
    let marked_after = match fld.cx.syntax_env.find(&extname.name) {
        None => {
            fld.cx.span_err(pth.span,
                            format!("macro undefined: '{}!'",
                                    extnamestr).as_slice());
            // let compilation continue
            return DummyResult::raw_pat(p.span);
        }

        Some(rc) => match *rc {
            NormalTT(ref expander, span) => {
                fld.cx.bt_push(ExpnInfo {
                    call_site: p.span,
                    callee: NameAndSpan {
                        name: extnamestr.get().to_string(),
                        format: MacroBang,
                        span: span
                    }
                });

                let fm = fresh_mark();
                let marked_before = mark_tts(tts.as_slice(), fm);
                let mac_span = original_span(fld.cx);
                let expanded = match expander.expand(fld.cx,
                                    mac_span.call_site,
                                    marked_before.as_slice()).make_pat() {
                    Some(e) => e,
                    None => {
                        fld.cx.span_err(
                            pth.span,
                            format!(
                                "non-pattern macro in pattern position: {}",
                                extnamestr.get()
                            ).as_slice()
                        );
                        return DummyResult::raw_pat(p.span);
                    }
                };

                // mark after:
                mark_pat(expanded,fm)
            }
            _ => {
                fld.cx.span_err(p.span,
                                format!("{}! is not legal in pattern position",
                                        extnamestr.get()).as_slice());
                return DummyResult::raw_pat(p.span);
            }
        }
    };

    let fully_expanded =
        fld.fold_pat(marked_after).node.clone();
    fld.cx.bt_pop();

    box(GC) ast::Pat {
        id: ast::DUMMY_NODE_ID,
        node: fully_expanded,
        span: p.span,
    }
}

/// A tree-folder that applies every rename in its (mutable) list
/// to every identifier, including both bindings and varrefs
/// (and lots of things that will turn out to be neither)
pub struct IdentRenamer<'a> {
    renames: &'a mtwt::RenameList,
}

impl<'a> Folder for IdentRenamer<'a> {
    fn fold_ident(&mut self, id: Ident) -> Ident {
        Ident {
            name: id.name,
            ctxt: mtwt::apply_renames(self.renames, id.ctxt),
        }
    }
    fn fold_mac(&mut self, macro: &ast::Mac) -> ast::Mac {
        fold::fold_mac(macro, self)
    }
}

/// A tree-folder that applies every rename in its list to
/// the idents that are in PatIdent patterns. This is more narrowly
/// focused than IdentRenamer, and is needed for FnDecl,
/// where we want to rename the args but not the fn name or the generics etc.
pub struct PatIdentRenamer<'a> {
    renames: &'a mtwt::RenameList,
}

impl<'a> Folder for PatIdentRenamer<'a> {
    fn fold_pat(&mut self, pat: Gc<ast::Pat>) -> Gc<ast::Pat> {
        match pat.node {
            ast::PatIdent(binding_mode, Spanned{span: ref sp, node: id}, ref sub) => {
                let new_ident = Ident{name: id.name,
                                      ctxt: mtwt::apply_renames(self.renames, id.ctxt)};
                let new_node =
                    ast::PatIdent(binding_mode,
                                  Spanned{span: self.new_span(*sp), node: new_ident},
                                  sub.map(|p| self.fold_pat(p)));
                box(GC) ast::Pat {
                    id: pat.id,
                    span: self.new_span(pat.span),
                    node: new_node,
                }
            },
            _ => noop_fold_pat(pat, self)
        }
    }
    fn fold_mac(&mut self, macro: &ast::Mac) -> ast::Mac {
        fold::fold_mac(macro, self)
    }
}

// expand a method
fn expand_method(m: &ast::Method, fld: &mut MacroExpander) -> SmallVector<Gc<ast::Method>> {
    let id = fld.new_id(m.id);
    match m.node {
        ast::MethDecl(ident,
                      ref generics,
                      abi,
                      ref explicit_self,
                      fn_style,
                      decl,
                      body,
                      vis) => {
            let (rewritten_fn_decl, rewritten_body)
                = expand_and_rename_fn_decl_and_block(&*decl,body,fld);
            SmallVector::one(box(GC) ast::Method {
                    attrs: m.attrs.iter().map(|a| fld.fold_attribute(*a)).collect(),
                    id: id,
                    span: fld.new_span(m.span),
                    node: ast::MethDecl(fld.fold_ident(ident),
                                        fold_generics(generics, fld),
                                        abi,
                                        fld.fold_explicit_self(explicit_self),
                                        fn_style,
                                        rewritten_fn_decl,
                                        rewritten_body,
                                        vis)
                })
        },
        ast::MethMac(ref mac) => {
            let maybe_new_methods =
                expand_mac_invoc(mac, &m.span,
                                 |r|{r.make_methods()},
                                 |meths,mark|{
                    meths.move_iter().map(|m|{mark_method(m,mark)})
                        .collect()},
                                 fld);

            let new_methods = match maybe_new_methods {
                Some(methods) => methods,
                None => SmallVector::zero()
            };

            // expand again if necessary
            new_methods.move_iter().flat_map(|m| fld.fold_method(m).move_iter()).collect()
        }
    }
}

/// Given a fn_decl and a block and a MacroExpander, expand the fn_decl, then use the
/// PatIdents in its arguments to perform renaming in the FnDecl and
/// the block, returning both the new FnDecl and the new Block.
fn expand_and_rename_fn_decl_and_block(fn_decl: &ast::FnDecl, block: Gc<ast::Block>,
                                       fld: &mut MacroExpander)
    -> (Gc<ast::FnDecl>, Gc<ast::Block>) {
    let expanded_decl = fld.fold_fn_decl(fn_decl);
    let idents = fn_decl_arg_bindings(&*expanded_decl);
    let renames =
        idents.iter().map(|id : &ast::Ident| (*id,fresh_name(id))).collect();
    // first, a renamer for the PatIdents, for the fn_decl:
    let mut rename_pat_fld = PatIdentRenamer{renames: &renames};
    let rewritten_fn_decl = rename_pat_fld.fold_fn_decl(&*expanded_decl);
    // now, a renamer for *all* idents, for the body:
    let mut rename_fld = IdentRenamer{renames: &renames};
    let rewritten_body = fld.fold_block(rename_fld.fold_block(block));
    (rewritten_fn_decl,rewritten_body)
}

/// A tree-folder that performs macro expansion
pub struct MacroExpander<'a, 'b> {
    pub cx: &'a mut ExtCtxt<'b>,
}

impl<'a, 'b> Folder for MacroExpander<'a, 'b> {
    fn fold_expr(&mut self, expr: Gc<ast::Expr>) -> Gc<ast::Expr> {
        expand_expr(expr, self)
    }

    fn fold_pat(&mut self, pat: Gc<ast::Pat>) -> Gc<ast::Pat> {
        expand_pat(pat, self)
    }

    fn fold_item(&mut self, item: Gc<ast::Item>) -> SmallVector<Gc<ast::Item>> {
        expand_item(item, self)
    }

    fn fold_item_underscore(&mut self, item: &ast::Item_) -> ast::Item_ {
        expand_item_underscore(item, self)
    }

    fn fold_stmt(&mut self, stmt: &ast::Stmt) -> SmallVector<Gc<ast::Stmt>> {
        expand_stmt(stmt, self)
    }

    fn fold_block(&mut self, block: P<Block>) -> P<Block> {
        expand_block(&*block, self)
    }

    fn fold_arm(&mut self, arm: &ast::Arm) -> ast::Arm {
        expand_arm(arm, self)
    }

    fn fold_method(&mut self, method: Gc<ast::Method>) -> SmallVector<Gc<ast::Method>> {
        expand_method(&*method, self)
    }

    fn new_span(&mut self, span: Span) -> Span {
        new_span(self.cx, span)
    }
}

fn new_span(cx: &ExtCtxt, sp: Span) -> Span {
    /* this discards information in the case of macro-defining macros */
    Span {
        lo: sp.lo,
        hi: sp.hi,
        expn_info: cx.backtrace(),
    }
}

pub struct ExpansionConfig {
    pub deriving_hash_type_parameter: bool,
    pub crate_name: String,
}

pub struct ExportedMacros {
    pub crate_name: Ident,
    pub macros: Vec<String>,
}

pub fn expand_crate(parse_sess: &parse::ParseSess,
                    cfg: ExpansionConfig,
                    // these are the macros being imported to this crate:
                    imported_macros: Vec<ExportedMacros>,
                    user_exts: Vec<NamedSyntaxExtension>,
                    c: Crate) -> Crate {
    let mut cx = ExtCtxt::new(parse_sess, c.config.clone(), cfg);
    let mut expander = MacroExpander {
        cx: &mut cx,
    };

    for ExportedMacros { crate_name, macros } in imported_macros.move_iter() {
        let name = format!("<{} macros>", token::get_ident(crate_name))
            .into_string();

        for source in macros.move_iter() {
            let item = parse::parse_item_from_source_str(name.clone(),
                                                         source,
                                                         expander.cx.cfg(),
                                                         expander.cx.parse_sess())
                    .expect("expected a serialized item");
            expand_item_mac(item, &mut expander);
        }
    }

    for (name, extension) in user_exts.move_iter() {
        expander.cx.syntax_env.insert(name, extension);
    }

    let mut ret = expander.fold_crate(c);
    ret.exported_macros = expander.cx.exported_macros.clone();
    parse_sess.span_diagnostic.handler().abort_if_errors();
    return ret;
}

// HYGIENIC CONTEXT EXTENSION:
// all of these functions are for walking over
// ASTs and making some change to the context of every
// element that has one. a CtxtFn is a trait-ified
// version of a closure in (SyntaxContext -> SyntaxContext).
// the ones defined here include:
// Marker - add a mark to a context

// A Marker adds the given mark to the syntax context
struct Marker { mark: Mrk }

impl Folder for Marker {
    fn fold_ident(&mut self, id: Ident) -> Ident {
        ast::Ident {
            name: id.name,
            ctxt: mtwt::apply_mark(self.mark, id.ctxt)
        }
    }
    fn fold_mac(&mut self, m: &ast::Mac) -> ast::Mac {
        let macro = match m.node {
            MacInvocTT(ref path, ref tts, ctxt) => {
                MacInvocTT(self.fold_path(path),
                           fold_tts(tts.as_slice(), self),
                           mtwt::apply_mark(self.mark, ctxt))
            }
        };
        Spanned {
            node: macro,
            span: m.span,
        }
    }
}

// apply a given mark to the given token trees. Used prior to expansion of a macro.
fn mark_tts(tts: &[TokenTree], m: Mrk) -> Vec<TokenTree> {
    fold_tts(tts, &mut Marker{mark:m})
}

// apply a given mark to the given expr. Used following the expansion of a macro.
fn mark_expr(expr: Gc<ast::Expr>, m: Mrk) -> Gc<ast::Expr> {
    Marker{mark:m}.fold_expr(expr)
}

// apply a given mark to the given pattern. Used following the expansion of a macro.
fn mark_pat(pat: Gc<ast::Pat>, m: Mrk) -> Gc<ast::Pat> {
    Marker{mark:m}.fold_pat(pat)
}

// apply a given mark to the given stmt. Used following the expansion of a macro.
fn mark_stmt(expr: &ast::Stmt, m: Mrk) -> Gc<ast::Stmt> {
    Marker{mark:m}.fold_stmt(expr)
        .expect_one("marking a stmt didn't return exactly one stmt")
}

// apply a given mark to the given item. Used following the expansion of a macro.
fn mark_item(expr: Gc<ast::Item>, m: Mrk) -> Gc<ast::Item> {
    Marker{mark:m}.fold_item(expr)
        .expect_one("marking an item didn't return exactly one item")
}

// apply a given mark to the given item. Used following the expansion of a macro.
fn mark_method(expr: Gc<ast::Method>, m: Mrk) -> Gc<ast::Method> {
    Marker{mark:m}.fold_method(expr)
        .expect_one("marking an item didn't return exactly one method")
}

fn original_span(cx: &ExtCtxt) -> Gc<codemap::ExpnInfo> {
    let mut relevant_info = cx.backtrace();
    let mut einfo = relevant_info.unwrap();
    loop {
        match relevant_info {
            None => { break }
            Some(e) => {
                einfo = e;
                relevant_info = einfo.call_site.expn_info;
            }
        }
    }
    return einfo;
}

/// Check that there are no macro invocations left in the AST:
pub fn check_for_macros(sess: &parse::ParseSess, krate: &ast::Crate) {
    visit::walk_crate(&mut MacroExterminator{sess:sess}, krate, ());
}

/// A visitor that ensures that no macro invocations remain in an AST.
struct MacroExterminator<'a>{
    sess: &'a parse::ParseSess
}

impl<'a> visit::Visitor<()> for MacroExterminator<'a> {
    fn visit_mac(&mut self, macro: &ast::Mac, _:()) {
        self.sess.span_diagnostic.span_bug(macro.span,
                                           "macro exterminator: expected AST \
                                           with no macro invocations");
    }
}


#[cfg(test)]
mod test {
    use super::{pattern_bindings, expand_crate, contains_macro_escape};
    use super::{PatIdentFinder, IdentRenamer, PatIdentRenamer};
    use ast;
    use ast::{Attribute_, AttrOuter, MetaWord, Name};
    use attr;
    use codemap;
    use codemap::Spanned;
    use ext::mtwt;
    use fold::Folder;
    use parse;
    use parse::token;
    use util::parser_testing::{string_to_parser};
    use util::parser_testing::{string_to_pat, string_to_crate, strs_to_idents};
    use visit;
    use visit::Visitor;

    use std::gc::GC;

    // a visitor that extracts the paths
    // from a given thingy and puts them in a mutable
    // array (passed in to the traversal)
    #[deriving(Clone)]
    struct PathExprFinderContext {
        path_accumulator: Vec<ast::Path> ,
    }

    impl Visitor<()> for PathExprFinderContext {

        fn visit_expr(&mut self, expr: &ast::Expr, _: ()) {
            match *expr {
                ast::Expr{id:_,span:_,node:ast::ExprPath(ref p)} => {
                    self.path_accumulator.push(p.clone());
                    // not calling visit_path, but it should be fine.
                }
                _ => visit::walk_expr(self,expr,())
            }
        }
    }

    // find the variable references in a crate
    fn crate_varrefs(the_crate : &ast::Crate) -> Vec<ast::Path> {
        let mut path_finder = PathExprFinderContext{path_accumulator:Vec::new()};
        visit::walk_crate(&mut path_finder, the_crate, ());
        path_finder.path_accumulator
    }

    /// A Visitor that extracts the identifiers from a thingy.
    // as a side note, I'm starting to want to abstract over these....
    struct IdentFinder{
        ident_accumulator: Vec<ast::Ident>
    }

    impl Visitor<()> for IdentFinder {
        fn visit_ident(&mut self, _: codemap::Span, id: ast::Ident, _: ()){
            self.ident_accumulator.push(id);
        }
    }

    /// Find the idents in a crate
    fn crate_idents(the_crate: &ast::Crate) -> Vec<ast::Ident> {
        let mut ident_finder = IdentFinder{ident_accumulator: Vec::new()};
        visit::walk_crate(&mut ident_finder, the_crate, ());
        ident_finder.ident_accumulator
    }

    // these following tests are quite fragile, in that they don't test what
    // *kind* of failure occurs.

    // make sure that macros can't escape fns
    #[should_fail]
    #[test] fn macros_cant_escape_fns_test () {
        let src = "fn bogus() {macro_rules! z (() => (3+4))}\
                   fn inty() -> int { z!() }".to_string();
        let sess = parse::new_parse_sess();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess);
        // should fail:
        let cfg = ::syntax::ext::expand::ExpansionConfig {
            deriving_hash_type_parameter: false,
            crate_name: "test".to_string(),
        };
        expand_crate(&sess,cfg,vec!(),vec!(),crate_ast);
    }

    // make sure that macros can't escape modules
    #[should_fail]
    #[test] fn macros_cant_escape_mods_test () {
        let src = "mod foo {macro_rules! z (() => (3+4))}\
                   fn inty() -> int { z!() }".to_string();
        let sess = parse::new_parse_sess();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess);
        let cfg = ::syntax::ext::expand::ExpansionConfig {
            deriving_hash_type_parameter: false,
            crate_name: "test".to_string(),
        };
        expand_crate(&sess,cfg,vec!(),vec!(),crate_ast);
    }

    // macro_escape modules should allow macros to escape
    #[test] fn macros_can_escape_flattened_mods_test () {
        let src = "#[macro_escape] mod foo {macro_rules! z (() => (3+4))}\
                   fn inty() -> int { z!() }".to_string();
        let sess = parse::new_parse_sess();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess);
        let cfg = ::syntax::ext::expand::ExpansionConfig {
            deriving_hash_type_parameter: false,
            crate_name: "test".to_string(),
        };
        expand_crate(&sess, cfg, vec!(), vec!(), crate_ast);
    }

    #[test] fn test_contains_flatten (){
        let attr1 = make_dummy_attr ("foo");
        let attr2 = make_dummy_attr ("bar");
        let escape_attr = make_dummy_attr ("macro_escape");
        let attrs1 = vec!(attr1, escape_attr, attr2);
        assert_eq!(contains_macro_escape(attrs1.as_slice()),true);
        let attrs2 = vec!(attr1,attr2);
        assert_eq!(contains_macro_escape(attrs2.as_slice()),false);
    }

    // make a MetaWord outer attribute with the given name
    fn make_dummy_attr(s: &str) -> ast::Attribute {
        Spanned {
            span:codemap::DUMMY_SP,
            node: Attribute_ {
                id: attr::mk_attr_id(),
                style: AttrOuter,
                value: box(GC) Spanned {
                    node: MetaWord(token::intern_and_get_ident(s)),
                    span: codemap::DUMMY_SP,
                },
                is_sugared_doc: false,
            }
        }
    }

    fn expand_crate_str(crate_str: String) -> ast::Crate {
        let ps = parse::new_parse_sess();
        let crate_ast = string_to_parser(&ps, crate_str).parse_crate_mod();
        // the cfg argument actually does matter, here...
        let cfg = ::syntax::ext::expand::ExpansionConfig {
            deriving_hash_type_parameter: false,
            crate_name: "test".to_string(),
        };
        expand_crate(&ps,cfg,vec!(),vec!(),crate_ast)
    }

    // find the pat_ident paths in a crate
    fn crate_bindings(the_crate : &ast::Crate) -> Vec<ast::Ident> {
        let mut name_finder = PatIdentFinder{ident_accumulator:Vec::new()};
        visit::walk_crate(&mut name_finder, the_crate, ());
        name_finder.ident_accumulator
    }

    #[test] fn macro_tokens_should_match(){
        expand_crate_str(
            "macro_rules! m((a)=>(13)) fn main(){m!(a);}".to_string());
    }

    // should be able to use a bound identifier as a literal in a macro definition:
    #[test] fn self_macro_parsing(){
        expand_crate_str(
            "macro_rules! foo ((zz) => (287u;))
            fn f(zz : int) {foo!(zz);}".to_string()
            );
    }

    // renaming tests expand a crate and then check that the bindings match
    // the right varrefs. The specification of the test case includes the
    // text of the crate, and also an array of arrays.  Each element in the
    // outer array corresponds to a binding in the traversal of the AST
    // induced by visit.  Each of these arrays contains a list of indexes,
    // interpreted as the varrefs in the varref traversal that this binding
    // should match.  So, for instance, in a program with two bindings and
    // three varrefs, the array ~[~[1,2],~[0]] would indicate that the first
    // binding should match the second two varrefs, and the second binding
    // should match the first varref.
    //
    // Put differently; this is a sparse representation of a boolean matrix
    // indicating which bindings capture which identifiers.
    //
    // Note also that this matrix is dependent on the implicit ordering of
    // the bindings and the varrefs discovered by the name-finder and the path-finder.
    //
    // The comparisons are done post-mtwt-resolve, so we're comparing renamed
    // names; differences in marks don't matter any more.
    //
    // oog... I also want tests that check "bound-identifier-=?". That is,
    // not just "do these have the same name", but "do they have the same
    // name *and* the same marks"? Understanding this is really pretty painful.
    // in principle, you might want to control this boolean on a per-varref basis,
    // but that would make things even harder to understand, and might not be
    // necessary for thorough testing.
    type RenamingTest = (&'static str, Vec<Vec<uint>>, bool);

    #[test]
    fn automatic_renaming () {
        let tests: Vec<RenamingTest> =
            vec!(// b & c should get new names throughout, in the expr too:
                ("fn a() -> int { let b = 13; let c = b; b+c }",
                 vec!(vec!(0,1),vec!(2)), false),
                // both x's should be renamed (how is this causing a bug?)
                ("fn main () {let x: int = 13;x;}",
                 vec!(vec!(0)), false),
                // the use of b after the + should be renamed, the other one not:
                ("macro_rules! f (($x:ident) => (b + $x)) fn a() -> int { let b = 13; f!(b)}",
                 vec!(vec!(1)), false),
                // the b before the plus should not be renamed (requires marks)
                ("macro_rules! f (($x:ident) => ({let b=9; ($x + b)})) fn a() -> int { f!(b)}",
                 vec!(vec!(1)), false),
                // the marks going in and out of letty should cancel, allowing that $x to
                // capture the one following the semicolon.
                // this was an awesome test case, and caught a *lot* of bugs.
                ("macro_rules! letty(($x:ident) => (let $x = 15;))
                  macro_rules! user(($x:ident) => ({letty!($x); $x}))
                  fn main() -> int {user!(z)}",
                 vec!(vec!(0)), false)
                );
        for (idx,s) in tests.iter().enumerate() {
            run_renaming_test(s,idx);
        }
    }

    // no longer a fixme #8062: this test exposes a *potential* bug; our system does
    // not behave exactly like MTWT, but a conversation with Matthew Flatt
    // suggests that this can only occur in the presence of local-expand, which
    // we have no plans to support. ... unless it's needed for item hygiene....
    #[ignore]
    #[test] fn issue_8062(){
        run_renaming_test(
            &("fn main() {let hrcoo = 19; macro_rules! getx(()=>(hrcoo)); getx!();}",
              vec!(vec!(0)), true), 0)
    }

    // FIXME #6994:
    // the z flows into and out of two macros (g & f) along one path, and one
    // (just g) along the other, so the result of the whole thing should
    // be "let z_123 = 3; z_123"
    #[ignore]
    #[test] fn issue_6994(){
        run_renaming_test(
            &("macro_rules! g (($x:ident) =>
              ({macro_rules! f(($y:ident)=>({let $y=3;$x}));f!($x)}))
              fn a(){g!(z)}",
              vec!(vec!(0)),false),
            0)
    }

    // match variable hygiene. Should expand into
    // fn z() {match 8 {x_1 => {match 9 {x_2 | x_2 if x_2 == x_1 => x_2 + x_1}}}}
    #[test] fn issue_9384(){
        run_renaming_test(
            &("macro_rules! bad_macro (($ex:expr) => ({match 9 {x | x if x == $ex => x + $ex}}))
              fn z() {match 8 {x => bad_macro!(x)}}",
              // NB: the third "binding" is the repeat of the second one.
              vec!(vec!(1,3),vec!(0,2),vec!(0,2)),
              true),
            0)
    }

    // interpolated nodes weren't getting labeled.
    // should expand into
    // fn main(){let g1_1 = 13; g1_1}}
    #[test] fn pat_expand_issue_15221(){
        run_renaming_test(
            &("macro_rules! inner ( ($e:pat ) => ($e))
              macro_rules! outer ( ($e:pat ) => (inner!($e)))
              fn main() { let outer!(g) = 13; g;}",
              vec!(vec!(0)),
              true),
            0)
    }

    // create a really evil test case where a $x appears inside a binding of $x
    // but *shouldn't* bind because it was inserted by a different macro....
    // can't write this test case until we have macro-generating macros.

    // method arg hygiene
    // method expands to fn get_x(&self_0, x_1:int) {self_0 + self_2 + x_3 + x_1}
    #[test] fn method_arg_hygiene(){
        run_renaming_test(
            &("macro_rules! inject_x (()=>(x))
              macro_rules! inject_self (()=>(self))
              struct A;
              impl A{fn get_x(&self, x: int) {self + inject_self!() + inject_x!() + x;} }",
              vec!(vec!(0),vec!(3)),
              true),
            0)
    }

    // ooh, got another bite?
    // expands to struct A; impl A {fn thingy(&self_1) {self_1;}}
    #[test] fn method_arg_hygiene_2(){
        run_renaming_test(
            &("struct A;
              macro_rules! add_method (($T:ty) =>
              (impl $T {  fn thingy(&self) {self;} }))
              add_method!(A)",
              vec!(vec!(0)),
              true),
            0)
    }

    // item fn hygiene
    // expands to fn q(x_1:int){fn g(x_2:int){x_2 + x_1};}
    #[test] fn issue_9383(){
        run_renaming_test(
            &("macro_rules! bad_macro (($ex:expr) => (fn g(x:int){ x + $ex }))
              fn q(x:int) { bad_macro!(x); }",
              vec!(vec!(1),vec!(0)),true),
            0)
    }

    // closure arg hygiene (ExprFnBlock)
    // expands to fn f(){(|x_1 : int| {(x_2 + x_1)})(3);}
    #[test] fn closure_arg_hygiene(){
        run_renaming_test(
            &("macro_rules! inject_x (()=>(x))
            fn f(){(|x : int| {(inject_x!() + x)})(3);}",
              vec!(vec!(1)),
              true),
            0)
    }

    // closure arg hygiene (ExprProc)
    // expands to fn f(){(proc(x_1 : int) {(x_2 + x_1)})(3);}
    #[test] fn closure_arg_hygiene_2(){
        run_renaming_test(
            &("macro_rules! inject_x (()=>(x))
              fn f(){ (proc(x : int){(inject_x!() + x)})(3); }",
              vec!(vec!(1)),
              true),
            0)
    }

    // macro_rules in method position. Sadly, unimplemented.
    #[test] fn macro_in_method_posn(){
        expand_crate_str(
            "macro_rules! my_method (() => (fn thirteen(&self) -> int {13}))
            struct A;
            impl A{ my_method!()}
            fn f(){A.thirteen;}".to_string());
    }

    // another nested macro
    // expands to impl Entries {fn size_hint(&self_1) {self_1;}
    #[test] fn item_macro_workaround(){
        run_renaming_test(
            &("macro_rules! item { ($i:item) => {$i}}
              struct Entries;
              macro_rules! iterator_impl {
              () => { item!( impl Entries { fn size_hint(&self) { self;}})}}
              iterator_impl! { }",
              vec!(vec!(0)), true),
            0)
    }

    // run one of the renaming tests
    fn run_renaming_test(t: &RenamingTest, test_idx: uint) {
        let invalid_name = token::special_idents::invalid.name;
        let (teststr, bound_connections, bound_ident_check) = match *t {
            (ref str,ref conns, bic) => (str.to_string(), conns.clone(), bic)
        };
        let cr = expand_crate_str(teststr.to_string());
        let bindings = crate_bindings(&cr);
        let varrefs = crate_varrefs(&cr);

        // must be one check clause for each binding:
        assert_eq!(bindings.len(),bound_connections.len());
        for (binding_idx,shouldmatch) in bound_connections.iter().enumerate() {
            let binding_name = mtwt::resolve(*bindings.get(binding_idx));
            let binding_marks = mtwt::marksof(bindings.get(binding_idx).ctxt, invalid_name);
            // shouldmatch can't name varrefs that don't exist:
            assert!((shouldmatch.len() == 0) ||
                    (varrefs.len() > *shouldmatch.iter().max().unwrap()));
            for (idx,varref) in varrefs.iter().enumerate() {
                let print_hygiene_debug_info = || {
                    // good lord, you can't make a path with 0 segments, can you?
                    let final_varref_ident = match varref.segments.last() {
                        Some(pathsegment) => pathsegment.identifier,
                        None => fail!("varref with 0 path segments?")
                    };
                    let varref_name = mtwt::resolve(final_varref_ident);
                    let varref_idents : Vec<ast::Ident>
                        = varref.segments.iter().map(|s| s.identifier)
                        .collect();
                    println!("varref #{}: {}, resolves to {}",idx, varref_idents, varref_name);
                    let string = token::get_ident(final_varref_ident);
                    println!("varref's first segment's string: \"{}\"", string.get());
                    println!("binding #{}: {}, resolves to {}",
                             binding_idx, *bindings.get(binding_idx), binding_name);
                    mtwt::with_sctable(|x| mtwt::display_sctable(x));
                };
                if shouldmatch.contains(&idx) {
                    // it should be a path of length 1, and it should
                    // be free-identifier=? or bound-identifier=? to the given binding
                    assert_eq!(varref.segments.len(),1);
                    let varref_name = mtwt::resolve(varref.segments.get(0).identifier);
                    let varref_marks = mtwt::marksof(varref.segments
                                                           .get(0)
                                                           .identifier
                                                           .ctxt,
                                                     invalid_name);
                    if !(varref_name==binding_name) {
                        println!("uh oh, should match but doesn't:");
                        print_hygiene_debug_info();
                    }
                    assert_eq!(varref_name,binding_name);
                    if bound_ident_check {
                        // we're checking bound-identifier=?, and the marks
                        // should be the same, too:
                        assert_eq!(varref_marks,binding_marks.clone());
                    }
                } else {
                    let varref_name = mtwt::resolve(varref.segments.get(0).identifier);
                    let fail = (varref.segments.len() == 1)
                        && (varref_name == binding_name);
                    // temp debugging:
                    if fail {
                        println!("failure on test {}",test_idx);
                        println!("text of test case: \"{}\"", teststr);
                        println!("");
                        println!("uh oh, matches but shouldn't:");
                        print_hygiene_debug_info();
                    }
                    assert!(!fail);
                }
            }
        }
    }

    #[test] fn fmt_in_macro_used_inside_module_macro() {
        let crate_str = "macro_rules! fmt_wrap(($b:expr)=>($b.to_string()))
macro_rules! foo_module (() => (mod generated { fn a() { let xx = 147; fmt_wrap!(xx);}}))
foo_module!()
".to_string();
        let cr = expand_crate_str(crate_str);
        // find the xx binding
        let bindings = crate_bindings(&cr);
        let cxbinds: Vec<&ast::Ident> =
            bindings.iter().filter(|b| {
                let ident = token::get_ident(**b);
                let string = ident.get();
                "xx" == string
            }).collect();
        let cxbinds: &[&ast::Ident] = cxbinds.as_slice();
        let cxbind = match cxbinds {
            [b] => b,
            _ => fail!("expected just one binding for ext_cx")
        };
        let resolved_binding = mtwt::resolve(*cxbind);
        let varrefs = crate_varrefs(&cr);

        // the xx binding should bind all of the xx varrefs:
        for (idx,v) in varrefs.iter().filter(|p| {
            p.segments.len() == 1
            && "xx" == token::get_ident(p.segments.get(0).identifier).get()
        }).enumerate() {
            if mtwt::resolve(v.segments.get(0).identifier) != resolved_binding {
                println!("uh oh, xx binding didn't match xx varref:");
                println!("this is xx varref \\# {:?}",idx);
                println!("binding: {:?}",cxbind);
                println!("resolves to: {:?}",resolved_binding);
                println!("varref: {:?}",v.segments.get(0).identifier);
                println!("resolves to: {:?}",
                         mtwt::resolve(v.segments.get(0).identifier));
                mtwt::with_sctable(|x| mtwt::display_sctable(x));
            }
            assert_eq!(mtwt::resolve(v.segments.get(0).identifier),
                       resolved_binding);
        };
    }

    #[test]
    fn pat_idents(){
        let pat = string_to_pat(
            "(a,Foo{x:c @ (b,9),y:Bar(4,d)})".to_string());
        let idents = pattern_bindings(&*pat);
        assert_eq!(idents, strs_to_idents(vec!("a","c","b","d")));
    }

    // test the list of identifier patterns gathered by the visitor. Note that
    // 'None' is listed as an identifier pattern because we don't yet know that
    // it's the name of a 0-ary variant, and that 'i' appears twice in succession.
    #[test]
    fn crate_bindings_test(){
        let the_crate = string_to_crate("fn main (a : int) -> int {|b| {
        match 34 {None => 3, Some(i) | i => j, Foo{k:z,l:y} => \"banana\"}} }".to_string());
        let idents = crate_bindings(&the_crate);
        assert_eq!(idents, strs_to_idents(vec!("a","b","None","i","i","z","y")));
    }

    // test the IdentRenamer directly
    #[test]
    fn ident_renamer_test () {
        let the_crate = string_to_crate("fn f(x : int){let x = x; x}".to_string());
        let f_ident = token::str_to_ident("f");
        let x_ident = token::str_to_ident("x");
        let int_ident = token::str_to_ident("int");
        let renames = vec!((x_ident,Name(16)));
        let mut renamer = IdentRenamer{renames: &renames};
        let renamed_crate = renamer.fold_crate(the_crate);
        let idents = crate_idents(&renamed_crate);
        let resolved : Vec<ast::Name> = idents.iter().map(|id| mtwt::resolve(*id)).collect();
        assert_eq!(resolved,vec!(f_ident.name,Name(16),int_ident.name,Name(16),Name(16),Name(16)));
    }

    // test the PatIdentRenamer; only PatIdents get renamed
    #[test]
    fn pat_ident_renamer_test () {
        let the_crate = string_to_crate("fn f(x : int){let x = x; x}".to_string());
        let f_ident = token::str_to_ident("f");
        let x_ident = token::str_to_ident("x");
        let int_ident = token::str_to_ident("int");
        let renames = vec!((x_ident,Name(16)));
        let mut renamer = PatIdentRenamer{renames: &renames};
        let renamed_crate = renamer.fold_crate(the_crate);
        let idents = crate_idents(&renamed_crate);
        let resolved : Vec<ast::Name> = idents.iter().map(|id| mtwt::resolve(*id)).collect();
        let x_name = x_ident.name;
        assert_eq!(resolved,vec!(f_ident.name,Name(16),int_ident.name,Name(16),x_name,x_name));
    }


}
