// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{P, Block, Crate, DeclLocal, ExprMac};
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

use std::mem;
use std::os;
use std::unstable::dynamic_lib::DynamicLibrary;

pub fn expand_expr(e: @ast::Expr, fld: &mut MacroExpander) -> @ast::Expr {
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
                    // leaving explicit deref here to highlight unbox op:
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
                                    name: extnamestr.get().to_strbuf(),
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
                                        format!("non-expr macro in expr pos: \
                                                 {}",
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

                    @ast::Expr {
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
            // Expand any interior macros etc.
            // NB: we don't fold pats yet. Curious.
            let src_expr = fld.fold_expr(src_expr).clone();
            let (src_loop_block, opt_ident) = expand_loop_block(src_loop_block, opt_ident, fld);

            let span = e.span;

            // to:
            //
            //   match &mut <src_expr> {
            //     i => {
            //       ['<ident>:] loop {
            //         match i.next() {
            //           None => break,
            //           Some(<src_pat>) => <src_loop_block>
            //         }
            //       }
            //     }
            //   }

            let local_ident = token::gensym_ident("__i"); // FIXME #13573
            let next_ident = fld.cx.ident_of("next");
            let none_ident = fld.cx.ident_of("None");

            let local_path = fld.cx.path_ident(span, local_ident);
            let some_path = fld.cx.path_ident(span, fld.cx.ident_of("Some"));

            // `None => break ['<ident>];`
            let none_arm = {
                let break_expr = fld.cx.expr(span, ast::ExprBreak(opt_ident));
                let none_pat = fld.cx.pat_ident(span, none_ident);
                fld.cx.arm(span, vec!(none_pat), break_expr)
            };

            // `Some(<src_pat>) => <src_loop_block>`
            let some_arm =
                fld.cx.arm(span,
                           vec!(fld.cx.pat_enum(span, some_path, vec!(src_pat))),
                           fld.cx.expr_block(src_loop_block));

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
            fld.cx.expr_match(span, discrim, vec!(arm))
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
            let mut rename_fld = renames_to_fold(&mut rename_list);
            let renamed_ident = rename_fld.fold_ident(label);

            // The rename *must* be added to the enclosed syntax context for
            // `break` or `continue` to pick up because by definition they are
            // in a block enclosed by loop head.
            fld.extsbox.push_frame();
            fld.extsbox.info().pending_renames.push(rename);
            let expanded_block = expand_block_elts(loop_block, fld);
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
pub fn expand_item(it: @ast::Item, fld: &mut MacroExpander)
                   -> SmallVector<@ast::Item> {
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
                        name: mname.get().to_strbuf(),
                        format: MacroAttribute,
                        span: None
                    }
                });

                // we'd ideally decorator_items.push_all(expand_item(item, fld)),
                // but that double-mut-borrows fld
                let mut items: SmallVector<@ast::Item> = SmallVector::zero();
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
                                          noop_fold_item(it, fld));
            fld.cx.mod_pop();
            result
        },
        _ => {
            let it = @ast::Item {
                attrs: new_attrs,
                ..(*it).clone()

            };
            noop_fold_item(it, fld)
        }
    };

    new_items.push_all(decorator_items);
    new_items
}

fn expand_item_modifiers(mut it: @ast::Item, fld: &mut MacroExpander)
                         -> @ast::Item {
    let (modifiers, attrs) = it.attrs.partitioned(|attr| {
        match fld.extsbox.find(&intern(attr.name().get())) {
            Some(&ItemModifier(_)) => true,
            _ => false
        }
    });

    it = @ast::Item {
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
                        name: mname.get().to_strbuf(),
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
pub fn contains_macro_escape(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, "macro_escape")
}

// Support for item-position macro invocations, exactly the same
// logic as for expression-position macro invocations.
pub fn expand_item_mac(it: @ast::Item, fld: &mut MacroExpander)
                       -> SmallVector<@ast::Item> {
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
                    name: extnamestr.get().to_strbuf(),
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
                    name: extnamestr.get().to_strbuf(),
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
                                    format!("expr macro in item position: {}",
                                            extnamestr.get()).as_slice());
                    return SmallVector::zero();
                }
            }
        }
    };
    fld.cx.bt_pop();
    return items;
}

// load macros from syntax-phase crates
pub fn expand_view_item(vi: &ast::ViewItem,
                        fld: &mut MacroExpander)
                        -> ast::ViewItem {
    match vi.node {
        ast::ViewItemExternCrate(..) => {
            let should_load = vi.attrs.iter().any(|attr| {
                attr.check_name("phase") &&
                    attr.meta_item_list().map_or(false, |phases| {
                        attr::contains_name(phases, "syntax")
                    })
            });

            if should_load {
                load_extern_macros(vi, fld);
            }
        }
        ast::ViewItemUse(_) => {}
    }

    noop_fold_view_item(vi, fld)
}

fn load_extern_macros(krate: &ast::ViewItem, fld: &mut MacroExpander) {
    let MacroCrate { lib, macros, registrar_symbol } =
        fld.cx.ecfg.loader.load_crate(krate);

    let crate_name = match krate.node {
        ast::ViewItemExternCrate(name, _, _) => name,
        _ => unreachable!()
    };
    let name = format!("<{} macros>", token::get_ident(crate_name));
    let name = name.to_strbuf();

    for source in macros.iter() {
        let item = parse::parse_item_from_source_str(name.clone(),
                                                     (*source).clone(),
                                                     fld.cx.cfg(),
                                                     fld.cx.parse_sess())
                .expect("expected a serialized item");
        expand_item_mac(item, fld);
    }

    let path = match lib {
        Some(path) => path,
        None => return
    };
    // Make sure the path contains a / or the linker will search for it.
    let path = os::make_absolute(&path);

    let registrar = match registrar_symbol {
        Some(registrar) => registrar,
        None => return
    };

    debug!("load_extern_macros: mapped crate {} to path {} and registrar {:s}",
           crate_name, path.display(), registrar);

    let lib = match DynamicLibrary::open(Some(&path)) {
        Ok(lib) => lib,
        // this is fatal: there are almost certainly macros we need
        // inside this crate, so continue would spew "macro undefined"
        // errors
        Err(err) => fld.cx.span_fatal(krate.span, err.as_slice())
    };

    unsafe {
        let registrar: MacroCrateRegistrationFun =
            match lib.symbol(registrar.as_slice()) {
                Ok(registrar) => registrar,
                // again fatal if we can't register macros
                Err(err) => fld.cx.span_fatal(krate.span, err.as_slice())
            };
        registrar(|name, extension| {
            let extension = match extension {
                NormalTT(ext, _) => NormalTT(ext, Some(krate.span)),
                IdentTT(ext, _) => IdentTT(ext, Some(krate.span)),
                ItemDecorator(ext) => ItemDecorator(ext),
                ItemModifier(ext) => ItemModifier(ext),
            };
            fld.extsbox.insert(name, extension);
        });

        // Intentionally leak the dynamic library. We can't ever unload it
        // since the library can do things that will outlive the expansion
        // phase (e.g. make an @-box cycle or launch a task).
        mem::forget(lib);
    }
}

// expand a stmt
pub fn expand_stmt(s: &Stmt, fld: &mut MacroExpander) -> SmallVector<@Stmt> {
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
                    name: extnamestr.get().to_strbuf(),
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
                                    format!("non-stmt macro in stmt pos: {}",
                                            extnamestr).as_slice());
                    return SmallVector::zero();
                }
            };

            mark_stmt(expanded,fm)
        }

        _ => {
            fld.cx.span_err(pth.span, format!("'{}' is not a tt-style macro",
                                              extnamestr).as_slice());
            return SmallVector::zero();
        }
    };

    // Keep going, outside-in.
    let fully_expanded = fld.fold_stmt(marked_after);
    if fully_expanded.is_empty() {
        fld.cx.span_err(pth.span, "macro didn't expand to a statement");
        return SmallVector::zero();
    }
    fld.cx.bt_pop();
    let fully_expanded: SmallVector<@Stmt> = fully_expanded.move_iter()
            .map(|s| @Spanned { span: s.span, node: s.node.clone() })
            .collect();

    fully_expanded.move_iter().map(|s| {
        match s.node {
            StmtExpr(e, stmt_id) if semi => {
                @Spanned {
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
                         -> SmallVector<@Stmt> {
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
                        span: span
                    } = **local;
                    // expand the pat (it might contain exprs... #:(o)>
                    let expanded_pat = fld.fold_pat(pat);
                    // find the pat_idents in the pattern:
                    // oh dear heaven... this is going to include the enum
                    // names, as well... but that should be okay, as long as
                    // the new names are gensyms for the old ones.
                    let mut name_finder = new_name_finder(Vec::new());
                    name_finder.visit_pat(expanded_pat,());
                    // generate fresh names, push them to a new pending list
                    let mut new_pending_renames = Vec::new();
                    for ident in name_finder.ident_accumulator.iter() {
                        let new_name = fresh_name(ident);
                        new_pending_renames.push((*ident,new_name));
                    }
                    let rewritten_pat = {
                        let mut rename_fld =
                            renames_to_fold(&mut new_pending_renames);
                        // rewrite the pattern using the new names (the old
                        // ones have already been applied):
                        rename_fld.fold_pat(expanded_pat)
                    };
                    // add them to the existing pending renames:
                    fld.extsbox.info().pending_renames.push_all_move(new_pending_renames);
                    // also, don't forget to expand the init:
                    let new_init_opt = init.map(|e| fld.fold_expr(e));
                    let rewritten_local =
                        @Local {
                            ty: local.ty,
                            pat: rewritten_pat,
                            init: new_init_opt,
                            id: id,
                            span: span,
                        };
                    SmallVector::one(@Spanned {
                        node: StmtDecl(@Spanned {
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

// a visitor that extracts the pat_ident paths
// from a given thingy and puts them in a mutable
// array (passed in to the traversal)
#[deriving(Clone)]
pub struct NewNameFinderContext {
    ident_accumulator: Vec<ast::Ident> ,
}

impl Visitor<()> for NewNameFinderContext {
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
                    self.visit_pat(*subpat, ())
                }
            }
            // use the default traversal for non-pat_idents
            _ => visit::walk_pat(self, pattern, ())
        }
    }

    fn visit_ty(&mut self, typ: &ast::Ty, _: ()) {
        visit::walk_ty(self, typ, ())
    }

}

// return a visitor that extracts the pat_ident paths
// from a given thingy and puts them in a mutable
// array (passed in to the traversal)
pub fn new_name_finder(idents: Vec<ast::Ident> ) -> NewNameFinderContext {
    NewNameFinderContext {
        ident_accumulator: idents,
    }
}

// expand a block. pushes a new exts_frame, then calls expand_block_elts
pub fn expand_block(blk: &Block, fld: &mut MacroExpander) -> P<Block> {
    // see note below about treatment of exts table
    with_exts_frame!(fld.extsbox,false,
                     expand_block_elts(blk, fld))
}

// expand the elements of a block.
pub fn expand_block_elts(b: &Block, fld: &mut MacroExpander) -> P<Block> {
    let new_view_items = b.view_items.iter().map(|x| fld.fold_view_item(x)).collect();
    let new_stmts =
        b.stmts.iter().flat_map(|x| {
            let renamed_stmt = {
                let pending_renames = &mut fld.extsbox.info().pending_renames;
                let mut rename_fld = renames_to_fold(pending_renames);
                rename_fld.fold_stmt(*x).expect_one("rename_fold didn't return one value")
            };
            fld.fold_stmt(renamed_stmt).move_iter()
        }).collect();
    let new_expr = b.expr.map(|x| {
        let expr = {
            let pending_renames = &mut fld.extsbox.info().pending_renames;
            let mut rename_fld = renames_to_fold(pending_renames);
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

// given a mutable list of renames, return a tree-folder that applies those
// renames.
pub fn renames_to_fold<'a>(renames: &'a mut RenameList) -> IdentRenamer<'a> {
    IdentRenamer {
        renames: renames,
    }
}

pub fn new_span(cx: &ExtCtxt, sp: Span) -> Span {
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
    fn fold_expr(&mut self, expr: @ast::Expr) -> @ast::Expr {
        expand_expr(expr, self)
    }

    fn fold_item(&mut self, item: @ast::Item) -> SmallVector<@ast::Item> {
        expand_item(item, self)
    }

    fn fold_view_item(&mut self, vi: &ast::ViewItem) -> ast::ViewItem {
        expand_view_item(vi, self)
    }

    fn fold_stmt(&mut self, stmt: &ast::Stmt) -> SmallVector<@ast::Stmt> {
        expand_stmt(stmt, self)
    }

    fn fold_block(&mut self, block: P<Block>) -> P<Block> {
        expand_block(block, self)
    }

    fn new_span(&mut self, span: Span) -> Span {
        new_span(self.cx, span)
    }
}

pub struct ExpansionConfig<'a> {
    pub loader: &'a mut CrateLoader,
    pub deriving_hash_type_parameter: bool,
    pub crate_id: CrateId,
}

pub fn expand_crate(parse_sess: &parse::ParseSess,
                    cfg: ExpansionConfig,
                    c: Crate) -> Crate {
    let mut cx = ExtCtxt::new(parse_sess, c.config.clone(), cfg);
    let mut expander = MacroExpander {
        extsbox: syntax_expander_table(),
        cx: &mut cx,
    };

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

// just a convenience:
fn new_mark_folder(m: Mrk) -> Marker {
    Marker {mark: m}
}

// apply a given mark to the given token trees. Used prior to expansion of a macro.
fn mark_tts(tts: &[TokenTree], m: Mrk) -> Vec<TokenTree> {
    fold_tts(tts, &mut new_mark_folder(m))
}

// apply a given mark to the given expr. Used following the expansion of a macro.
fn mark_expr(expr: @ast::Expr, m: Mrk) -> @ast::Expr {
    new_mark_folder(m).fold_expr(expr)
}

// apply a given mark to the given stmt. Used following the expansion of a macro.
fn mark_stmt(expr: &ast::Stmt, m: Mrk) -> @ast::Stmt {
    new_mark_folder(m).fold_stmt(expr)
            .expect_one("marking a stmt didn't return a stmt")
}

// apply a given mark to the given item. Used following the expansion of a macro.
fn mark_item(expr: @ast::Item, m: Mrk) -> SmallVector<@ast::Item> {
    new_mark_folder(m).fold_item(expr)
}

fn original_span(cx: &ExtCtxt) -> @codemap::ExpnInfo {
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
    use super::*;
    use ast;
    use ast::{Attribute_, AttrOuter, MetaWord};
    use attr;
    use codemap;
    use codemap::Spanned;
    use ext::base::{CrateLoader, MacroCrate};
    use ext::mtwt;
    use parse;
    use parse::token;
    use util::parser_testing::{string_to_parser};
    use util::parser_testing::{string_to_pat, strs_to_idents};
    use visit;
    use visit::Visitor;

    // a visitor that extracts the paths
    // from a given thingy and puts them in a mutable
    // array (passed in to the traversal)
    #[deriving(Clone)]
    struct NewPathExprFinderContext {
        path_accumulator: Vec<ast::Path> ,
    }

    impl Visitor<()> for NewPathExprFinderContext {

        fn visit_expr(&mut self, expr: &ast::Expr, _: ()) {
            match *expr {
                ast::Expr{id:_,span:_,node:ast::ExprPath(ref p)} => {
                    self.path_accumulator.push(p.clone());
                    // not calling visit_path, should be fine.
                }
                _ => visit::walk_expr(self,expr,())
            }
        }

        fn visit_ty(&mut self, typ: &ast::Ty, _: ()) {
            visit::walk_ty(self, typ, ())
        }

    }

    // return a visitor that extracts the paths
    // from a given pattern and puts them in a mutable
    // array (passed in to the traversal)
    pub fn new_path_finder(paths: Vec<ast::Path> ) -> NewPathExprFinderContext {
        NewPathExprFinderContext {
            path_accumulator: paths
        }
    }

    struct ErrLoader;

    impl CrateLoader for ErrLoader {
        fn load_crate(&mut self, _: &ast::ViewItem) -> MacroCrate {
            fail!("lolwut")
        }
    }

    // these following tests are quite fragile, in that they don't test what
    // *kind* of failure occurs.

    // make sure that macros can leave scope
    #[should_fail]
    #[test] fn macros_cant_escape_fns_test () {
        let src = "fn bogus() {macro_rules! z (() => (3+4))}\
                   fn inty() -> int { z!() }".to_strbuf();
        let sess = parse::new_parse_sess();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_strbuf(),
            src,
            Vec::new(), &sess);
        // should fail:
        let mut loader = ErrLoader;
        let cfg = ::syntax::ext::expand::ExpansionConfig {
            loader: &mut loader,
            deriving_hash_type_parameter: false,
            crate_id: from_str("test").unwrap(),
        };
        expand_crate(&sess,cfg,crate_ast);
    }

    // make sure that macros can leave scope for modules
    #[should_fail]
    #[test] fn macros_cant_escape_mods_test () {
        let src = "mod foo {macro_rules! z (() => (3+4))}\
                   fn inty() -> int { z!() }".to_strbuf();
        let sess = parse::new_parse_sess();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_strbuf(),
            src,
            Vec::new(), &sess);
        // should fail:
        let mut loader = ErrLoader;
        let cfg = ::syntax::ext::expand::ExpansionConfig {
            loader: &mut loader,
            deriving_hash_type_parameter: false,
            crate_id: from_str("test").unwrap(),
        };
        expand_crate(&sess,cfg,crate_ast);
    }

    // macro_escape modules shouldn't cause macros to leave scope
    #[test] fn macros_can_escape_flattened_mods_test () {
        let src = "#[macro_escape] mod foo {macro_rules! z (() => (3+4))}\
                   fn inty() -> int { z!() }".to_strbuf();
        let sess = parse::new_parse_sess();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_strbuf(),
            src,
            Vec::new(), &sess);
        // should fail:
        let mut loader = ErrLoader;
        let cfg = ::syntax::ext::expand::ExpansionConfig {
            loader: &mut loader,
            deriving_hash_type_parameter: false,
            crate_id: from_str("test").unwrap(),
        };
        expand_crate(&sess, cfg, crate_ast);
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
                value: @Spanned {
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
        let mut loader = ErrLoader;
        let cfg = ::syntax::ext::expand::ExpansionConfig {
            loader: &mut loader,
            deriving_hash_type_parameter: false,
            crate_id: from_str("test").unwrap(),
        };
        expand_crate(&ps,cfg,crate_ast)
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
            "macro_rules! m((a)=>(13)) fn main(){m!(a);}".to_strbuf());
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
    // The comparisons are done post-mtwt-resolve, so we're comparing renamed
    // names; differences in marks don't matter any more.
    //
    // oog... I also want tests that check "binding-identifier-=?". That is,
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
                 vec!(vec!(0)), false));
        for (idx,s) in tests.iter().enumerate() {
            run_renaming_test(s,idx);
        }
    }

    // run one of the renaming tests
    fn run_renaming_test(t: &RenamingTest, test_idx: uint) {
        let invalid_name = token::special_idents::invalid.name;
        let (teststr, bound_connections, bound_ident_check) = match *t {
            (ref str,ref conns, bic) => (str.to_owned(), conns.clone(), bic)
        };
        let cr = expand_crate_str(teststr.to_strbuf());
        // find the bindings:
        let mut name_finder = new_name_finder(Vec::new());
        visit::walk_crate(&mut name_finder,&cr,());
        let bindings = name_finder.ident_accumulator;

        // find the varrefs:
        let mut path_finder = new_path_finder(Vec::new());
        visit::walk_crate(&mut path_finder,&cr,());
        let varrefs = path_finder.path_accumulator;

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
                        println!("uh oh, should match but doesn't:");
                        println!("varref: {:?}",varref);
                        println!("binding: {:?}", *bindings.get(binding_idx));
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
                        println!("failure on test {}",test_idx);
                        println!("text of test case: \"{}\"", teststr);
                        println!("");
                        println!("uh oh, matches but shouldn't:");
                        println!("varref: {:?}",varref);
                        // good lord, you can't make a path with 0 segments, can you?
                        let string = token::get_ident(varref.segments
                                                            .get(0)
                                                            .identifier);
                        println!("varref's first segment's uint: {}, and string: \"{}\"",
                                 varref.segments.get(0).identifier.name,
                                 string.get());
                        println!("binding: {:?}", *bindings.get(binding_idx));
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
".to_strbuf();
        let cr = expand_crate_str(crate_str);
        // find the xx binding
        let mut name_finder = new_name_finder(Vec::new());
        visit::walk_crate(&mut name_finder, &cr, ());
        let bindings = name_finder.ident_accumulator;

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
        // find all the xx varrefs:
        let mut path_finder = new_path_finder(Vec::new());
        visit::walk_crate(&mut path_finder, &cr, ());
        let varrefs = path_finder.path_accumulator;

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
            "(a,Foo{x:c @ (b,9),y:Bar(4,d)})".to_strbuf());
        let mut pat_idents = new_name_finder(Vec::new());
        pat_idents.visit_pat(pat, ());
        assert_eq!(pat_idents.ident_accumulator,
                   strs_to_idents(vec!("a","c","b","d")));
    }

}
