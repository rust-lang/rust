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
use crateid::CrateId;
use ext::base::*;
use fold::*;
use parse;
use parse::token::{fresh_mark, fresh_name, intern};
use parse::token;
use visit;
use visit::Visitor;
use util::small_vector::SmallVector;

use std::gc::{Gc, GC};


pub fn expand_expr(e: Gc<ast::Expr>, fld: &mut MacroExpander) -> Gc<ast::Expr> {
    match e.node {
        // expr_mac should really be expr_ext or something; it's the
        // entry-point for all syntax extensions.
        ExprMac(ref mac) => {
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
                        return DummyResult::raw_expr(e.span);
                    }
                    let extname = pth.segments.get(0).identifier;
                    let extnamestr = token::get_ident(extname);
                    let marked_after = match fld.extsbox.find(&extname.name) {
                        None => {
                            fld.cx.span_err(
                                pth.span,
                                format!("macro undefined: '{}'",
                                        extnamestr.get()).as_slice());

                            // let compilation continue
                            return DummyResult::raw_expr(e.span);
                        }
                        Some(&NormalTT(ref expandfun, exp_span)) => {
                            fld.cx.bt_push(ExpnInfo {
                                call_site: e.span,
                                callee: NameAndSpan {
                                    name: extnamestr.get().to_string(),
                                    format: MacroBang,
                                    span: exp_span,
                                },
                            });
                            let fm = fresh_mark();
                            // mark before:
                            let marked_before = mark_tts(tts.as_slice(), fm);

                            // The span that we pass to the expanders we want to
                            // be the root of the call stack. That's the most
                            // relevant span and it's the actual invocation of
                            // the macro.
                            let mac_span = original_span(fld.cx);

                            let expanded = match expandfun.expand(fld.cx,
                                                   mac_span.call_site,
                                                   marked_before.as_slice()).make_expr() {
                                Some(e) => e,
                                None => {
                                    fld.cx.span_err(
                                        pth.span,
                                        format!("non-expression macro in expression position: {}",
                                                extnamestr.get().as_slice()
                                        ).as_slice());
                                    return DummyResult::raw_expr(e.span);
                                }
                            };

                            // mark after:
                            mark_expr(expanded,fm)
                        }
                        _ => {
                            fld.cx.span_err(
                                pth.span,
                                format!("'{}' is not a tt-style macro",
                                        extnamestr.get()).as_slice());
                            return DummyResult::raw_expr(e.span);
                        }
                    };

                    // Keep going, outside-in.
                    //
                    // FIXME(pcwalton): Is it necessary to clone the
                    // node here?
                    let fully_expanded =
                        fld.fold_expr(marked_after).node.clone();
                    fld.cx.bt_pop();

                    box(GC) ast::Expr {
                        id: ast::DUMMY_NODE_ID,
                        node: fully_expanded,
                        span: e.span,
                    }
                }
            }
        }

        // Desugar expr_for_loop
        // From: `['<ident>:] for <src_pat> in <src_expr> <src_loop_block>`
        // FIXME #6993: change type of opt_ident to Option<Name>
        ast::ExprForLoop(src_pat, src_expr, src_loop_block, opt_ident) => {

            let span = e.span;

            // to:
            //
            //   match &mut <src_expr> {
            //     i => {
            //       ['<ident>:] loop {
            //         match i.next() {
            //           None => break ['<ident>],
            //           Some(mut value) => {
            //             let <src_pat> = value;
            //             <src_loop_block>
            //           }
            //         }
            //       }
            //     }
            //   }
            //
            // (The use of the `let` is to give better error messages
            // when the pattern is refutable.)

            let local_ident = token::gensym_ident("i");
            let next_ident = fld.cx.ident_of("next");
            let none_ident = fld.cx.ident_of("None");

            let local_path = fld.cx.path_ident(span, local_ident);
            let some_path = fld.cx.path_ident(span, fld.cx.ident_of("Some"));

            // `None => break ['<ident>],`
            let none_arm = {
                let break_expr = fld.cx.expr(span, ast::ExprBreak(opt_ident));
                let none_pat = fld.cx.pat_ident(span, none_ident);
                fld.cx.arm(span, vec!(none_pat), break_expr)
            };

            // let <src_pat> = value;
            // use underscore to suppress lint error:
            let value_ident = token::gensym_ident("_value");
            // this is careful to use src_pat.span so that error
            // messages point exact at that.
            let local = box(GC) ast::Local {
                ty: fld.cx.ty_infer(src_pat.span),
                pat: src_pat,
                init: Some(fld.cx.expr_ident(src_pat.span, value_ident)),
                id: ast::DUMMY_NODE_ID,
                span: src_pat.span,
                source: ast::LocalFor
            };
            let local = codemap::respan(src_pat.span, ast::DeclLocal(local));
            let local = box(GC) codemap::respan(span, ast::StmtDecl(box(GC) local,
                                                            ast::DUMMY_NODE_ID));

            // { let ...; <src_loop_block> }
            let block = fld.cx.block(span, vec![local],
                                     Some(fld.cx.expr_block(src_loop_block)));

            // `Some(mut value) => { ... }`
            // Note the _'s in the name will stop any unused mutability warnings.
            let value_pat = fld.cx.pat_ident_binding_mode(span, value_ident,
                                                          ast::BindByValue(ast::MutMutable));
            let some_arm =
                fld.cx.arm(span,
                           vec!(fld.cx.pat_enum(span, some_path, vec!(value_pat))),
                           fld.cx.expr_block(block));

            // `match i.next() { ... }`
            let match_expr = {
                let next_call_expr =
                    fld.cx.expr_method_call(span,
                                            fld.cx.expr_path(local_path),
                                            next_ident,
                                            Vec::new());

                fld.cx.expr_match(span, next_call_expr, vec!(none_arm, some_arm))
            };

            // ['ident:] loop { ... }
            let loop_expr = fld.cx.expr(span,
                                        ast::ExprLoop(fld.cx.block_expr(match_expr),
                                                      opt_ident));

            // `i => loop { ... }`

            // `match &mut <src_expr> { i => loop { ... } }`
            let discrim = fld.cx.expr_mut_addr_of(span, src_expr);
            let i_pattern = fld.cx.pat_ident(span, local_ident);
            let arm = fld.cx.arm(span, vec!(i_pattern), loop_expr);
            // why these clone()'s everywhere? I guess I'll follow the pattern....
            let match_expr = fld.cx.expr_match(span, discrim, vec!(arm));
            fld.fold_expr(match_expr).clone()
        }

        ast::ExprLoop(loop_block, opt_ident) => {
            let (loop_block, opt_ident) = expand_loop_block(loop_block, opt_ident, fld);
            fld.cx.expr(e.span, ast::ExprLoop(loop_block, opt_ident))
        }

        _ => noop_fold_expr(e, fld)
    }
}

// Rename loop label and expand its loop body
//
// The renaming procedure for loop is different in the sense that the loop
// body is in a block enclosed by loop head so the renaming of loop label
// must be propagated to the enclosed context.
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
            fld.extsbox.push_frame();
            fld.extsbox.info().pending_renames.push(rename);
            let expanded_block = expand_block_elts(&*loop_block, fld);
            fld.extsbox.pop_frame();

            (expanded_block, Some(renamed_ident))
        }
        None => (fld.fold_block(loop_block), opt_ident)
    }
}

// eval $e with a new exts frame:
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

        match fld.extsbox.find(&intern(mname.get())) {
            Some(&ItemDecorator(dec_fn)) => {
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
        }
    }

    let mut new_items = match it.node {
        ast::ItemMac(..) => expand_item_mac(it, fld),
        ast::ItemMod(_) | ast::ItemForeignMod(_) => {
            fld.cx.mod_push(it.ident);
            let macro_escape = contains_macro_escape(new_attrs.as_slice());
            let result = with_exts_frame!(fld.extsbox,
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
    let (modifiers, attrs) = it.attrs.partitioned(|attr| {
        match fld.extsbox.find(&intern(attr.name().get())) {
            Some(&ItemModifier(_)) => true,
            _ => false
        }
    });

    it = box(GC) ast::Item {
        attrs: attrs,
        ..(*it).clone()
    };

    if modifiers.is_empty() {
        return it;
    }

    for attr in modifiers.iter() {
        let mname = attr.name();

        match fld.extsbox.find(&intern(mname.get())) {
            Some(&ItemModifier(dec_fn)) => {
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
        }
    }

    // expansion may have added new ItemModifiers
    expand_item_modifiers(it, fld)
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
    let expanded = match fld.extsbox.find(&extname.name) {
        None => {
            fld.cx.span_err(pth.span,
                            format!("macro undefined: '{}!'",
                                    extnamestr).as_slice());
            // let compilation continue
            return SmallVector::zero();
        }

        Some(&NormalTT(ref expander, span)) => {
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
        Some(&IdentTT(ref expander, span)) => {
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
        _ => {
            fld.cx.span_err(it.span,
                            format!("{}! is not legal in item position",
                                    extnamestr.get()).as_slice());
            return SmallVector::zero();
        }
    };

    let items = match expanded.make_def() {
        Some(MacroDef { name, ext }) => {
            // yikes... no idea how to apply the mark to this. I'm afraid
            // we're going to have to wait-and-see on this one.
            fld.extsbox.insert(intern(name.as_slice()), ext);
            if attr::contains_name(it.attrs.as_slice(), "macro_export") {
                SmallVector::one(it)
            } else {
                SmallVector::zero()
            }
        }
        None => {
            match expanded.make_items() {
                Some(items) => {
                    items.move_iter()
                        .flat_map(|i| mark_item(i, fm).move_iter())
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

// expand a stmt
fn expand_stmt(s: &Stmt, fld: &mut MacroExpander) -> SmallVector<Gc<Stmt>> {
    // why the copying here and not in expand_expr?
    // looks like classic changed-in-only-one-place
    let (pth, tts, semi) = match s.node {
        StmtMac(ref mac, semi) => {
            match mac.node {
                MacInvocTT(ref pth, ref tts, _) => {
                    (pth, (*tts).clone(), semi)
                }
            }
        }
        _ => return expand_non_macro_stmt(s, fld)
    };
    if pth.segments.len() > 1u {
        fld.cx.span_err(pth.span, "expected macro name without module separators");
        return SmallVector::zero();
    }
    let extname = pth.segments.get(0).identifier;
    let extnamestr = token::get_ident(extname);
    let marked_after = match fld.extsbox.find(&extname.name) {
        None => {
            fld.cx.span_err(pth.span,
                            format!("macro undefined: '{}'",
                                    extnamestr).as_slice());
            return SmallVector::zero();
        }

        Some(&NormalTT(ref expandfun, exp_span)) => {
            fld.cx.bt_push(ExpnInfo {
                call_site: s.span,
                callee: NameAndSpan {
                    name: extnamestr.get().to_string(),
                    format: MacroBang,
                    span: exp_span,
                }
            });
            let fm = fresh_mark();
            // mark before expansion:
            let marked_tts = mark_tts(tts.as_slice(), fm);

            // See the comment in expand_expr for why we want the original span,
            // not the current mac.span.
            let mac_span = original_span(fld.cx);

            let expanded = match expandfun.expand(fld.cx,
                                                  mac_span.call_site,
                                                  marked_tts.as_slice()).make_stmt() {
                Some(stmt) => stmt,
                None => {
                    fld.cx.span_err(pth.span,
                                    format!("non-statement macro in statement position: {}",
                                            extnamestr).as_slice());
                    return SmallVector::zero();
                }
            };

            mark_stmt(&*expanded,fm)
        }

        _ => {
            fld.cx.span_err(pth.span, format!("'{}' is not a tt-style macro",
                                              extnamestr).as_slice());
            return SmallVector::zero();
        }
    };

    // Keep going, outside-in.
    let fully_expanded = fld.fold_stmt(&*marked_after);
    if fully_expanded.is_empty() {
        fld.cx.span_err(pth.span, "macro didn't expand to a statement");
        return SmallVector::zero();
    }
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
                    // find the pat_idents in the pattern:
                    // oh dear heaven... this is going to include the enum
                    // names, as well... but that should be okay, as long as
                    // the new names are gensyms for the old ones.
                    // generate fresh names, push them to a new pending list
                    let idents = pattern_bindings(expanded_pat);
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
                    fld.extsbox.info().pending_renames.push_all_move(new_pending_renames);
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

fn expand_arm(arm: &ast::Arm, fld: &mut MacroExpander) -> ast::Arm {
    // expand pats... they might contain macro uses:
    let expanded_pats : Vec<Gc<ast::Pat>> = arm.pats.iter().map(|pat| fld.fold_pat(*pat)).collect();
    if expanded_pats.len() == 0 {
        fail!("encountered match arm with 0 patterns");
    }
    // all of the pats must have the same set of bindings, so use the
    // first one to extract them and generate new names:
    let first_pat = expanded_pats.get(0);
    // code duplicated from 'let', above. Perhaps this can be lifted
    // into a separate function:
    let idents = pattern_bindings(*first_pat);
    let mut new_pending_renames =
        idents.iter().map(|id| (*id,fresh_name(id))).collect();
    // rewrite all of the patterns using the new names (the old
    // ones have already been applied). Note that we depend here
    // on the guarantee that after expansion, there can't be any
    // Path expressions (a.k.a. varrefs) left in the pattern. If
    // this were false, we'd need to apply this renaming only to
    // the bindings, and not to the varrefs, using a more targeted
    // fold-er.
    let mut rename_fld = IdentRenamer{renames:&mut new_pending_renames};
    let rewritten_pats =
        expanded_pats.iter().map(|pat| rename_fld.fold_pat(*pat)).collect();
    // apply renaming and then expansion to the guard and the body:
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



// a visitor that extracts the pat_ident (binding) paths
// from a given thingy and puts them in a mutable
// array
#[deriving(Clone)]
struct NameFinderContext {
    ident_accumulator: Vec<ast::Ident> ,
}

impl Visitor<()> for NameFinderContext {
    fn visit_pat(&mut self, pattern: &ast::Pat, _: ()) {
        match *pattern {
            // we found a pat_ident!
            ast::Pat {
                id: _,
                node: ast::PatIdent(_, ref path, ref inner),
                span: _
            } => {
                match path {
                    // a path of length one:
                    &ast::Path {
                        global: false,
                        span: _,
                        segments: ref segments
                    } if segments.len() == 1 => {
                        self.ident_accumulator.push(segments.get(0)
                                                            .identifier)
                    }
                    // I believe these must be enums...
                    _ => ()
                }
                // visit optional subpattern of pat_ident:
                for subpat in inner.iter() {
                    self.visit_pat(&**subpat, ())
                }
            }
            // use the default traversal for non-pat_idents
            _ => visit::walk_pat(self, pattern, ())
        }
    }

}

// find the pat_ident paths in a pattern
fn pattern_bindings(pat : &ast::Pat) -> Vec<ast::Ident> {
    let mut name_finder = NameFinderContext{ident_accumulator:Vec::new()};
    name_finder.visit_pat(pat,());
    name_finder.ident_accumulator
}

// expand a block. pushes a new exts_frame, then calls expand_block_elts
fn expand_block(blk: &Block, fld: &mut MacroExpander) -> P<Block> {
    // see note below about treatment of exts table
    with_exts_frame!(fld.extsbox,false,
                     expand_block_elts(blk, fld))
}

// expand the elements of a block.
fn expand_block_elts(b: &Block, fld: &mut MacroExpander) -> P<Block> {
    let new_view_items = b.view_items.iter().map(|x| fld.fold_view_item(x)).collect();
    let new_stmts =
        b.stmts.iter().flat_map(|x| {
            // perform all pending renames
            let renamed_stmt = {
                let pending_renames = &mut fld.extsbox.info().pending_renames;
                let mut rename_fld = IdentRenamer{renames:pending_renames};
                rename_fld.fold_stmt(&**x).expect_one("rename_fold didn't return one value")
            };
            // expand macros in the statement
            fld.fold_stmt(&*renamed_stmt).move_iter()
        }).collect();
    let new_expr = b.expr.map(|x| {
        let expr = {
            let pending_renames = &mut fld.extsbox.info().pending_renames;
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
    let marked_after = match fld.extsbox.find(&extname.name) {
        None => {
            fld.cx.span_err(pth.span,
                            format!("macro undefined: '{}!'",
                                    extnamestr).as_slice());
            // let compilation continue
            return DummyResult::raw_pat(p.span);
        }

        Some(&NormalTT(ref expander, span)) => {
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

// a tree-folder that applies every rename in its (mutable) list
// to every identifier, including both bindings and varrefs
// (and lots of things that will turn out to be neither)
pub struct IdentRenamer<'a> {
    renames: &'a mut RenameList,
}

impl<'a> Folder for IdentRenamer<'a> {
    fn fold_ident(&mut self, id: Ident) -> Ident {
        let new_ctxt = self.renames.iter().fold(id.ctxt, |ctxt, &(from, to)| {
            mtwt::new_rename(from, to, ctxt)
        });
        Ident {
            name: id.name,
            ctxt: new_ctxt,
        }
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

pub struct MacroExpander<'a, 'b> {
    pub extsbox: SyntaxEnv,
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

    fn fold_stmt(&mut self, stmt: &ast::Stmt) -> SmallVector<Gc<ast::Stmt>> {
        expand_stmt(stmt, self)
    }

    fn fold_block(&mut self, block: P<Block>) -> P<Block> {
        expand_block(&*block, self)
    }

    fn fold_arm(&mut self, arm: &ast::Arm) -> ast::Arm {
        expand_arm(arm, self)
    }

    fn new_span(&mut self, span: Span) -> Span {
        new_span(self.cx, span)
    }
}

pub struct ExpansionConfig {
    pub deriving_hash_type_parameter: bool,
    pub crate_id: CrateId,
}

pub struct ExportedMacros {
    pub crate_name: Ident,
    pub macros: Vec<String>,
}

pub fn expand_crate(parse_sess: &parse::ParseSess,
                    cfg: ExpansionConfig,
                    macros: Vec<ExportedMacros>,
                    user_exts: Vec<NamedSyntaxExtension>,
                    c: Crate) -> Crate {
    let mut cx = ExtCtxt::new(parse_sess, c.config.clone(), cfg);
    let mut expander = MacroExpander {
        extsbox: syntax_expander_table(),
        cx: &mut cx,
    };

    for ExportedMacros { crate_name, macros } in macros.move_iter() {
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
        expander.extsbox.insert(name, extension);
    }

    let ret = expander.fold_crate(c);
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
            ctxt: mtwt::new_mark(self.mark, id.ctxt)
        }
    }
    fn fold_mac(&mut self, m: &ast::Mac) -> ast::Mac {
        let macro = match m.node {
            MacInvocTT(ref path, ref tts, ctxt) => {
                MacInvocTT(self.fold_path(path),
                           fold_tts(tts.as_slice(), self),
                           mtwt::new_mark(self.mark, ctxt))
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
            .expect_one("marking a stmt didn't return a stmt")
}

// apply a given mark to the given item. Used following the expansion of a macro.
fn mark_item(expr: Gc<ast::Item>, m: Mrk) -> SmallVector<Gc<ast::Item>> {
    Marker{mark:m}.fold_item(expr)
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

#[cfg(test)]
mod test {
    use super::{pattern_bindings, expand_crate, contains_macro_escape};
    use super::{NameFinderContext};
    use ast;
    use ast::{Attribute_, AttrOuter, MetaWord};
    use attr;
    use codemap;
    use codemap::Spanned;
    use ext::mtwt;
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
            crate_id: from_str("test").unwrap(),
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
            crate_id: from_str("test").unwrap(),
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
            crate_id: from_str("test").unwrap(),
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
            crate_id: from_str("test").unwrap(),
        };
        expand_crate(&ps,cfg,vec!(),vec!(),crate_ast)
    }

    // find the pat_ident paths in a crate
    fn crate_bindings(the_crate : &ast::Crate) -> Vec<ast::Ident> {
        let mut name_finder = NameFinderContext{ident_accumulator:Vec::new()};
        visit::walk_crate(&mut name_finder, the_crate, ());
        name_finder.ident_accumulator
    }


    //fn expand_and_resolve(crate_str: @str) -> ast::crate {
        //let expanded_ast = expand_crate_str(crate_str);
        // println!("expanded: {:?}\n",expanded_ast);
        //mtwt_resolve_crate(expanded_ast)
    //}
    //fn expand_and_resolve_and_pretty_print (crate_str: @str) -> String {
        //let resolved_ast = expand_and_resolve(crate_str);
        //pprust::to_str(&resolved_ast,fake_print_crate,get_ident_interner())
    //}

    #[test] fn macro_tokens_should_match(){
        expand_crate_str(
            "macro_rules! m((a)=>(13)) fn main(){m!(a);}".to_string());
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
    // but *shouldnt* bind because it was inserted by a different macro....
    // can't write this test case until we have macro-generating macros.

    // FIXME #9383 : lambda var hygiene
    // interesting... can't even write this test, yet, because the name-finder
    // only finds pattern vars. Time to upgrade test framework.
    /*#[test]
    fn issue_9383(){
        run_renaming_test(
            &("macro_rules! bad_macro (($ex:expr) => ({(|_x| { $ex }) (9) }))
              fn takes_x(_x : int) { assert_eq!(bad_macro!(_x),8); }
              fn main() { takes_x(8); }",
              vec!(vec!()),false),
            0)
    }*/

    // run one of the renaming tests
    fn run_renaming_test(t: &RenamingTest, test_idx: uint) {
        let invalid_name = token::special_idents::invalid.name;
        let (teststr, bound_connections, bound_ident_check) = match *t {
            (ref str,ref conns, bic) => (str.to_owned(), conns.clone(), bic)
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
                if shouldmatch.contains(&idx) {
                    // it should be a path of length 1, and it should
                    // be free-identifier=? or bound-identifier=? to the given binding
                    assert_eq!(varref.segments.len(),1);
                    let varref_name = mtwt::resolve(varref.segments
                                                          .get(0)
                                                          .identifier);
                    let varref_marks = mtwt::marksof(varref.segments
                                                           .get(0)
                                                           .identifier
                                                           .ctxt,
                                                     invalid_name);
                    if !(varref_name==binding_name) {
                        let varref_idents : Vec<ast::Ident>
                            = varref.segments.iter().map(|s|
                                                         s.identifier)
                            .collect();
                        println!("uh oh, should match but doesn't:");
                        println!("varref #{}: {}",idx, varref_idents);
                        println!("binding #{}: {}", binding_idx, *bindings.get(binding_idx));
                        mtwt::with_sctable(|x| mtwt::display_sctable(x));
                    }
                    assert_eq!(varref_name,binding_name);
                    if bound_ident_check {
                        // we're checking bound-identifier=?, and the marks
                        // should be the same, too:
                        assert_eq!(varref_marks,binding_marks.clone());
                    }
                } else {
                    let fail = (varref.segments.len() == 1)
                        && (mtwt::resolve(varref.segments.get(0).identifier)
                            == binding_name);
                    // temp debugging:
                    if fail {
                        let varref_idents : Vec<ast::Ident>
                            = varref.segments.iter().map(|s|
                                                         s.identifier)
                            .collect();
                        println!("failure on test {}",test_idx);
                        println!("text of test case: \"{}\"", teststr);
                        println!("");
                        println!("uh oh, matches but shouldn't:");
                        println!("varref: {}",varref_idents);
                        // good lord, you can't make a path with 0 segments, can you?
                        let string = token::get_ident(varref.segments
                                                            .get(0)
                                                            .identifier);
                        println!("varref's first segment's uint: {}, and string: \"{}\"",
                                 varref.segments.get(0).identifier.name,
                                 string.get());
                        println!("binding: {}", *bindings.get(binding_idx));
                        mtwt::with_sctable(|x| mtwt::display_sctable(x));
                    }
                    assert!(!fail);
                }
            }
        }
    }

    #[test] fn fmt_in_macro_used_inside_module_macro() {
        let crate_str = "macro_rules! fmt_wrap(($b:expr)=>($b.to_str()))
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
        let idents = pattern_bindings(pat);
        assert_eq!(idents, strs_to_idents(vec!("a","c","b","d")));
    }

    // test the list of identifier patterns gathered by the visitor. Note that
    // 'None' is listed as an identifier pattern because we don't yet know that
    // it's the name of a 0-ary variant, and that 'i' appears twice in succession.
    #[test]
    fn crate_idents(){
        let the_crate = string_to_crate("fn main (a : int) -> int {|b| {
        match 34 {None => 3, Some(i) | i => j, Foo{k:z,l:y} => \"banana\"}} }".to_string());
        let idents = crate_bindings(&the_crate);
        assert_eq!(idents, strs_to_idents(vec!("a","b","None","i","i","z","y")));
    }

    //

}
