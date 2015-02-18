// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{Block, Crate, DeclLocal, ExprMac, PatMac};
use ast::{Local, Ident, MacInvocTT};
use ast::{ItemMac, MacStmtWithSemicolon, Mrk, Stmt, StmtDecl, StmtMac};
use ast::{StmtExpr, StmtSemi};
use ast::TokenTree;
use ast;
use ast_util::path_to_ident;
use ext::mtwt;
use ext::build::AstBuilder;
use attr;
use attr::AttrMetaMethods;
use codemap;
use codemap::{Span, Spanned, ExpnInfo, NameAndSpan, MacroBang, MacroAttribute};
use ext::base::*;
use feature_gate::{Features};
use fold;
use fold::*;
use parse;
use parse::token::{fresh_mark, fresh_name, intern};
use parse::token;
use ptr::P;
use util::small_vector::SmallVector;
use visit;
use visit::Visitor;
use std_inject;

pub fn expand_type(t: P<ast::Ty>,
                   fld: &mut MacroExpander,
                   impl_ty: Option<P<ast::Ty>>)
                   -> P<ast::Ty> {
    debug!("expanding type {:?} with impl_ty {:?}", t, impl_ty);
    let t = match (t.node.clone(), impl_ty) {
        // Expand uses of `Self` in impls to the concrete type.
        (ast::Ty_::TyPath(ref path, _), Some(ref impl_ty)) => {
            let path_as_ident = path_to_ident(path);
            // Note unhygenic comparison here. I think this is correct, since
            // even though `Self` is almost just a type parameter, the treatment
            // for this expansion is as if it were a keyword.
            if path_as_ident.is_some() &&
               path_as_ident.unwrap().name == token::special_idents::type_self.name {
                impl_ty.clone()
            } else {
                t
            }
        }
        _ => t
    };
    fold::noop_fold_ty(t, fld)
}

pub fn expand_expr(e: P<ast::Expr>, fld: &mut MacroExpander) -> P<ast::Expr> {
    e.and_then(|ast::Expr {id, node, span}| match node {
        // expr_mac should really be expr_ext or something; it's the
        // entry-point for all syntax extensions.
        ast::ExprMac(mac) => {
            let expanded_expr = match expand_mac_invoc(mac, span,
                                                       |r| r.make_expr(),
                                                       mark_expr, fld) {
                Some(expr) => expr,
                None => {
                    return DummyResult::raw_expr(span);
                }
            };

            // Keep going, outside-in.
            //
            let fully_expanded = fld.fold_expr(expanded_expr);
            fld.cx.bt_pop();

            fully_expanded.map(|e| ast::Expr {
                id: ast::DUMMY_NODE_ID,
                node: e.node,
                span: span,
            })
        }

        ast::ExprWhile(cond, body, opt_ident) => {
            let cond = fld.fold_expr(cond);
            let (body, opt_ident) = expand_loop_block(body, opt_ident, fld);
            fld.cx.expr(span, ast::ExprWhile(cond, body, opt_ident))
        }

        // Desugar ExprWhileLet
        // From: `[opt_ident]: while let <pat> = <expr> <body>`
        ast::ExprWhileLet(pat, expr, body, opt_ident) => {
            // to:
            //
            //   [opt_ident]: loop {
            //     match <expr> {
            //       <pat> => <body>,
            //       _ => break
            //     }
            //   }

            // `<pat> => <body>`
            let pat_arm = {
                let body_expr = fld.cx.expr_block(body);
                fld.cx.arm(pat.span, vec![pat], body_expr)
            };

            // `_ => break`
            let break_arm = {
                let pat_under = fld.cx.pat_wild(span);
                let break_expr = fld.cx.expr_break(span);
                fld.cx.arm(span, vec![pat_under], break_expr)
            };

            // `match <expr> { ... }`
            let arms = vec![pat_arm, break_arm];
            let match_expr = fld.cx.expr(span,
                                    ast::ExprMatch(expr, arms, ast::MatchSource::WhileLetDesugar));

            // `[opt_ident]: loop { ... }`
            let loop_block = fld.cx.block_expr(match_expr);
            let (loop_block, opt_ident) = expand_loop_block(loop_block, opt_ident, fld);
            fld.cx.expr(span, ast::ExprLoop(loop_block, opt_ident))
        }

        // Desugar ExprIfLet
        // From: `if let <pat> = <expr> <body> [<elseopt>]`
        ast::ExprIfLet(pat, expr, body, mut elseopt) => {
            // to:
            //
            //   match <expr> {
            //     <pat> => <body>,
            //     [_ if <elseopt_if_cond> => <elseopt_if_body>,]
            //     _ => [<elseopt> | ()]
            //   }

            // `<pat> => <body>`
            let pat_arm = {
                let body_expr = fld.cx.expr_block(body);
                fld.cx.arm(pat.span, vec![pat], body_expr)
            };

            // `[_ if <elseopt_if_cond> => <elseopt_if_body>,]`
            let else_if_arms = {
                let mut arms = vec![];
                loop {
                    let elseopt_continue = elseopt
                        .and_then(|els| els.and_then(|els| match els.node {
                        // else if
                        ast::ExprIf(cond, then, elseopt) => {
                            let pat_under = fld.cx.pat_wild(span);
                            arms.push(ast::Arm {
                                attrs: vec![],
                                pats: vec![pat_under],
                                guard: Some(cond),
                                body: fld.cx.expr_block(then)
                            });
                            elseopt.map(|elseopt| (elseopt, true))
                        }
                        _ => Some((P(els), false))
                    }));
                    match elseopt_continue {
                        Some((e, true)) => {
                            elseopt = Some(e);
                        }
                        Some((e, false)) => {
                            elseopt = Some(e);
                            break;
                        }
                        None => {
                            elseopt = None;
                            break;
                        }
                    }
                }
                arms
            };

            let contains_else_clause = elseopt.is_some();

            // `_ => [<elseopt> | ()]`
            let else_arm = {
                let pat_under = fld.cx.pat_wild(span);
                let else_expr = elseopt.unwrap_or_else(|| fld.cx.expr_tuple(span, vec![]));
                fld.cx.arm(span, vec![pat_under], else_expr)
            };

            let mut arms = Vec::with_capacity(else_if_arms.len() + 2);
            arms.push(pat_arm);
            arms.extend(else_if_arms.into_iter());
            arms.push(else_arm);

            let match_expr = fld.cx.expr(span,
                                         ast::ExprMatch(expr, arms,
                                                ast::MatchSource::IfLetDesugar {
                                                    contains_else_clause: contains_else_clause,
                                                }));
            fld.fold_expr(match_expr)
        }

        // Desugar support for ExprIfLet in the ExprIf else position
        ast::ExprIf(cond, blk, elseopt) => {
            let elseopt = elseopt.map(|els| els.and_then(|els| match els.node {
                ast::ExprIfLet(..) => {
                    // wrap the if-let expr in a block
                    let span = els.span;
                    let blk = P(ast::Block {
                        stmts: vec![],
                        expr: Some(P(els)),
                        id: ast::DUMMY_NODE_ID,
                        rules: ast::DefaultBlock,
                        span: span
                    });
                    fld.cx.expr_block(blk)
                }
                _ => P(els)
            }));
            let if_expr = fld.cx.expr(span, ast::ExprIf(cond, blk, elseopt));
            if_expr.map(|e| noop_fold_expr(e, fld))
        }

        ast::ExprLoop(loop_block, opt_ident) => {
            let (loop_block, opt_ident) = expand_loop_block(loop_block, opt_ident, fld);
            fld.cx.expr(span, ast::ExprLoop(loop_block, opt_ident))
        }

        // Desugar ExprForLoop
        // From: `[opt_ident]: for <pat> in <head> <body>`
        ast::ExprForLoop(pat, head, body, opt_ident) => {
            // to:
            //
            //   {
            //     let result = match ::std::iter::IntoIterator::into_iter(<head>) {
            //       mut iter => {
            //         [opt_ident]: loop {
            //           match ::std::iter::Iterator::next(&mut iter) {
            //             ::std::option::Option::Some(<pat>) => <body>,
            //             ::std::option::Option::None => break
            //           }
            //         }
            //       }
            //     };
            //     result
            //   }

            // expand <head>
            let head = fld.fold_expr(head);

            // create an hygienic ident
            let iter = {
                let ident = fld.cx.ident_of("iter");
                let new_ident = fresh_name(&ident);
                let rename = (ident, new_ident);
                let mut rename_list = vec![rename];
                let mut rename_fld = IdentRenamer{ renames: &mut rename_list };

                rename_fld.fold_ident(ident)
            };

            let pat_span = pat.span;
            // `:;std::option::Option::Some(<pat>) => <body>`
            let pat_arm = {
                let body_expr = fld.cx.expr_block(body);
                let some_pat = fld.cx.pat_some(pat_span, pat);

                fld.cx.arm(pat_span, vec![some_pat], body_expr)
            };

            // `::std::option::Option::None => break`
            let break_arm = {
                let break_expr = fld.cx.expr_break(span);

                fld.cx.arm(span, vec![fld.cx.pat_none(span)], break_expr)
            };

            // `match ::std::iter::Iterator::next(&mut iter) { ... }`
            let match_expr = {
                let next_path = {
                    let strs = vec![
                        fld.cx.ident_of_std("core"),
                        fld.cx.ident_of("iter"),
                        fld.cx.ident_of("Iterator"),
                        fld.cx.ident_of("next"),
                    ];

                    fld.cx.path_global(span, strs)
                };
                let ref_mut_iter = fld.cx.expr_mut_addr_of(span, fld.cx.expr_ident(span, iter));
                let next_expr =
                    fld.cx.expr_call(span, fld.cx.expr_path(next_path), vec![ref_mut_iter]);
                let arms = vec![pat_arm, break_arm];

                fld.cx.expr(pat_span,
                            ast::ExprMatch(next_expr, arms, ast::MatchSource::ForLoopDesugar))
            };

            // `[opt_ident]: loop { ... }`
            let loop_block = fld.cx.block_expr(match_expr);
            let (loop_block, opt_ident) = expand_loop_block(loop_block, opt_ident, fld);
            let loop_expr = fld.cx.expr(span, ast::ExprLoop(loop_block, opt_ident));

            // `mut iter => { ... }`
            let iter_arm = {
                let iter_pat =
                    fld.cx.pat_ident_binding_mode(span, iter, ast::BindByValue(ast::MutMutable));
                fld.cx.arm(span, vec![iter_pat], loop_expr)
            };

            // `match ::std::iter::IntoIterator::into_iter(<head>) { ... }`
            let into_iter_expr = {
                let into_iter_path = {
                    let strs = vec![
                        fld.cx.ident_of_std("core"),
                        fld.cx.ident_of("iter"),
                        fld.cx.ident_of("IntoIterator"),
                        fld.cx.ident_of("into_iter"),
                    ];

                    fld.cx.path_global(span, strs)
                };

                fld.cx.expr_call(span, fld.cx.expr_path(into_iter_path), vec![head])
            };

            let match_expr = fld.cx.expr_match(span, into_iter_expr, vec![iter_arm]);

            // `{ let result = ...; result }`
            let result_ident = token::gensym_ident("result");
            fld.cx.expr_block(
                fld.cx.block_all(
                    span,
                    vec![fld.cx.stmt_let(span, false, result_ident, match_expr)],
                    Some(fld.cx.expr_ident(span, result_ident))))
        }

        ast::ExprClosure(capture_clause, fn_decl, block) => {
            let (rewritten_fn_decl, rewritten_block)
                = expand_and_rename_fn_decl_and_block(fn_decl, block, fld);
            let new_node = ast::ExprClosure(capture_clause,
                                            rewritten_fn_decl,
                                            rewritten_block);
            P(ast::Expr{id:id, node: new_node, span: fld.new_span(span)})
        }

        _ => {
            P(noop_fold_expr(ast::Expr {
                id: id,
                node: node,
                span: span
            }, fld))
        }
    })
}

/// Expand a (not-ident-style) macro invocation. Returns the result
/// of expansion and the mark which must be applied to the result.
/// Our current interface doesn't allow us to apply the mark to the
/// result until after calling make_expr, make_items, etc.
fn expand_mac_invoc<T, F, G>(mac: ast::Mac, span: codemap::Span,
                             parse_thunk: F,
                             mark_thunk: G,
                             fld: &mut MacroExpander)
                             -> Option<T> where
    F: for<'a> FnOnce(Box<MacResult+'a>) -> Option<T>,
    G: FnOnce(T, Mrk) -> T,
{
    match mac.node {
        // it would almost certainly be cleaner to pass the whole
        // macro invocation in, rather than pulling it apart and
        // marking the tts and the ctxt separately. This also goes
        // for the other three macro invocation chunks of code
        // in this file.
        // Token-tree macros:
        MacInvocTT(pth, tts, _) => {
            if pth.segments.len() > 1 {
                fld.cx.span_err(pth.span,
                                "expected macro name without module \
                                separators");
                // let compilation continue
                return None;
            }
            let extname = pth.segments[0].identifier;
            let extnamestr = token::get_ident(extname);
            match fld.cx.syntax_env.find(&extname.name) {
                None => {
                    fld.cx.span_err(
                        pth.span,
                        &format!("macro undefined: '{}!'",
                                &extnamestr)[]);

                    // let compilation continue
                    None
                }
                Some(rc) => match *rc {
                    NormalTT(ref expandfun, exp_span) => {
                        fld.cx.bt_push(ExpnInfo {
                                call_site: span,
                                callee: NameAndSpan {
                                    name: extnamestr.to_string(),
                                    format: MacroBang,
                                    span: exp_span,
                                },
                            });
                        let fm = fresh_mark();
                        let marked_before = mark_tts(&tts[..], fm);

                        // The span that we pass to the expanders we want to
                        // be the root of the call stack. That's the most
                        // relevant span and it's the actual invocation of
                        // the macro.
                        let mac_span = fld.cx.original_span();

                        let opt_parsed = {
                            let expanded = expandfun.expand(fld.cx,
                                                            mac_span,
                                                            &marked_before[..]);
                            parse_thunk(expanded)
                        };
                        let parsed = match opt_parsed {
                            Some(e) => e,
                            None => {
                                fld.cx.span_err(
                                    pth.span,
                                    &format!("non-expression macro in expression position: {}",
                                            &extnamestr[..]
                                            )[]);
                                return None;
                            }
                        };
                        Some(mark_thunk(parsed,fm))
                    }
                    _ => {
                        fld.cx.span_err(
                            pth.span,
                            &format!("'{}' is not a tt-style macro",
                                    &extnamestr)[]);
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
            let expanded_block = expand_block_elts(loop_block, fld);
            fld.cx.syntax_env.pop_frame();

            (expanded_block, Some(renamed_ident))
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
    let it = expand_item_modifiers(it, fld);

    expand_annotatable(Annotatable::Item(it), fld)
        .into_iter().map(|i| i.expect_item()).collect()
}

fn expand_item_modifiers(mut it: P<ast::Item>, fld: &mut MacroExpander)
                         -> P<ast::Item> {
    // partition the attributes into ItemModifiers and others
    let (modifiers, other_attrs) = modifiers(&it.attrs, fld);

    // update the attrs, leave everything else alone. Is this mutation really a good idea?
    it = P(ast::Item {
        attrs: other_attrs,
        ..(*it).clone()
    });

    if modifiers.is_empty() {
        let it = expand_item_multi_modifier(Annotatable::Item(it), fld);
        return it.expect_item();
    }

    for attr in &modifiers {
        let mname = attr.name();

        match fld.cx.syntax_env.find(&intern(&mname)) {
            Some(rc) => match *rc {
                Modifier(ref mac) => {
                    attr::mark_used(attr);
                    fld.cx.bt_push(ExpnInfo {
                        call_site: attr.span,
                        callee: NameAndSpan {
                            name: mname.to_string(),
                            format: MacroAttribute,
                            span: None,
                        }
                    });
                    it = mac.expand(fld.cx, attr.span, &*attr.node.value, it);
                    fld.cx.bt_pop();
                }
                _ => unreachable!()
            },
            _ => unreachable!()
        }
    }

    // Expansion may have added new ItemModifiers.
    // It is possible, that an item modifier could expand to a multi-modifier or
    // vice versa. In this case we will expand all modifiers before multi-modifiers,
    // which might give an odd ordering. However, I think it is unlikely that the
    // two kinds will be mixed, and I old-style multi-modifiers should be deprecated
    // anyway.
    expand_item_modifiers(it, fld)
}

/// Expand item_underscore
fn expand_item_underscore(item: ast::Item_, fld: &mut MacroExpander) -> ast::Item_ {
    match item {
        ast::ItemFn(decl, fn_style, abi, generics, body) => {
            let (rewritten_fn_decl, rewritten_body)
                = expand_and_rename_fn_decl_and_block(decl, body, fld);
            let expanded_generics = fold::noop_fold_generics(generics,fld);
            ast::ItemFn(rewritten_fn_decl, fn_style, abi, expanded_generics, rewritten_body)
        }
        _ => noop_fold_item_underscore(item, fld)
    }
}

// does this attribute list contain "macro_use" ?
fn contains_macro_use(fld: &mut MacroExpander, attrs: &[ast::Attribute]) -> bool {
    for attr in attrs {
        let mut is_use = attr.check_name("macro_use");
        if attr.check_name("macro_escape") {
            fld.cx.span_warn(attr.span, "macro_escape is a deprecated synonym for macro_use");
            is_use = true;
            if let ast::AttrInner = attr.node.style {
                fld.cx.span_help(attr.span, "consider an outer attribute, \
                                             #[macro_use] mod ...");
            }
        };

        if is_use {
            match attr.node.value.node {
                ast::MetaWord(..) => (),
                _ => fld.cx.span_err(attr.span, "arguments to macro_use are not allowed here"),
            }
            return true;
        }
    }
    false
}

// Support for item-position macro invocations, exactly the same
// logic as for expression-position macro invocations.
pub fn expand_item_mac(it: P<ast::Item>,
                       fld: &mut MacroExpander) -> SmallVector<P<ast::Item>> {
    let (extname, path_span, tts) = match it.node {
        ItemMac(codemap::Spanned {
            node: MacInvocTT(ref pth, ref tts, _),
            ..
        }) => {
            (pth.segments[0].identifier, pth.span, (*tts).clone())
        }
        _ => fld.cx.span_bug(it.span, "invalid item macro invocation")
    };

    let extnamestr = token::get_ident(extname);
    let fm = fresh_mark();
    let items = {
        let expanded = match fld.cx.syntax_env.find(&extname.name) {
            None => {
                fld.cx.span_err(path_span,
                                &format!("macro undefined: '{}!'",
                                        extnamestr)[]);
                // let compilation continue
                return SmallVector::zero();
            }

            Some(rc) => match *rc {
                NormalTT(ref expander, span) => {
                    if it.ident.name != parse::token::special_idents::invalid.name {
                        fld.cx
                            .span_err(path_span,
                                      &format!("macro {}! expects no ident argument, \
                                        given '{}'",
                                      extnamestr,
                                      token::get_ident(it.ident))[]);
                        return SmallVector::zero();
                    }
                    fld.cx.bt_push(ExpnInfo {
                        call_site: it.span,
                        callee: NameAndSpan {
                            name: extnamestr.to_string(),
                            format: MacroBang,
                            span: span
                        }
                    });
                    // mark before expansion:
                    let marked_before = mark_tts(&tts[..], fm);
                    expander.expand(fld.cx, it.span, &marked_before[..])
                }
                IdentTT(ref expander, span) => {
                    if it.ident.name == parse::token::special_idents::invalid.name {
                        fld.cx.span_err(path_span,
                                        &format!("macro {}! expects an ident argument",
                                                &extnamestr)[]);
                        return SmallVector::zero();
                    }
                    fld.cx.bt_push(ExpnInfo {
                        call_site: it.span,
                        callee: NameAndSpan {
                            name: extnamestr.to_string(),
                            format: MacroBang,
                            span: span
                        }
                    });
                    // mark before expansion:
                    let marked_tts = mark_tts(&tts[..], fm);
                    expander.expand(fld.cx, it.span, it.ident, marked_tts)
                }
                MacroRulesTT => {
                    if it.ident.name == parse::token::special_idents::invalid.name {
                        fld.cx.span_err(path_span,
                                        &format!("macro_rules! expects an ident argument")
                                        []);
                        return SmallVector::zero();
                    }
                    fld.cx.bt_push(ExpnInfo {
                        call_site: it.span,
                        callee: NameAndSpan {
                            name: extnamestr.to_string(),
                            format: MacroBang,
                            span: None,
                        }
                    });
                    // DON'T mark before expansion.

                    let def = ast::MacroDef {
                        ident: it.ident,
                        attrs: it.attrs.clone(),
                        id: ast::DUMMY_NODE_ID,
                        span: it.span,
                        imported_from: None,
                        export: attr::contains_name(&it.attrs, "macro_export"),
                        use_locally: true,
                        body: tts,
                    };
                    fld.cx.insert_macro(def);

                    // macro_rules! has a side effect but expands to nothing.
                    fld.cx.bt_pop();
                    return SmallVector::zero();
                }
                _ => {
                    fld.cx.span_err(it.span,
                                    &format!("{}! is not legal in item position",
                                            &extnamestr)[]);
                    return SmallVector::zero();
                }
            }
        };

        expanded.make_items()
    };

    let items = match items {
        Some(items) => {
            items.into_iter()
                .map(|i| mark_item(i, fm))
                .flat_map(|i| fld.fold_item(i).into_iter())
                .collect()
        }
        None => {
            fld.cx.span_err(path_span,
                            &format!("non-item macro in item position: {}",
                                    &extnamestr)[]);
            return SmallVector::zero();
        }
    };

    fld.cx.bt_pop();
    items
}

/// Expand a stmt
fn expand_stmt(s: Stmt, fld: &mut MacroExpander) -> SmallVector<P<Stmt>> {
    let (mac, style) = match s.node {
        StmtMac(mac, style) => (mac, style),
        _ => return expand_non_macro_stmt(s, fld)
    };
    let expanded_stmt = match expand_mac_invoc(mac.and_then(|m| m), s.span,
                                                |r| r.make_stmt(),
                                                mark_stmt, fld) {
        Some(stmt) => stmt,
        None => {
            return SmallVector::zero();
        }
    };

    // Keep going, outside-in.
    let fully_expanded = fld.fold_stmt(expanded_stmt);
    fld.cx.bt_pop();

    if style == MacStmtWithSemicolon {
        fully_expanded.into_iter().map(|s| s.map(|Spanned {node, span}| {
            Spanned {
                node: match node {
                    StmtExpr(e, stmt_id) => StmtSemi(e, stmt_id),
                    _ => node /* might already have a semi */
                },
                span: span
            }
        })).collect()
    } else {
        fully_expanded
    }
}

// expand a non-macro stmt. this is essentially the fallthrough for
// expand_stmt, above.
fn expand_non_macro_stmt(Spanned {node, span: stmt_span}: Stmt, fld: &mut MacroExpander)
                         -> SmallVector<P<Stmt>> {
    // is it a let?
    match node {
        StmtDecl(decl, node_id) => decl.and_then(|Spanned {node: decl, span}| match decl {
            DeclLocal(local) => {
                // take it apart:
                let rewritten_local = local.map(|Local {id, pat, ty, init, source, span}| {
                    // expand the ty since TyFixedLengthVec contains an Expr
                    // and thus may have a macro use
                    let expanded_ty = ty.map(|t| fld.fold_ty(t));
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
                    fld.cx.syntax_env.info().pending_renames
                          .extend(new_pending_renames.into_iter());
                    Local {
                        id: id,
                        ty: expanded_ty,
                        pat: rewritten_pat,
                        // also, don't forget to expand the init:
                        init: init.map(|e| fld.fold_expr(e)),
                        source: source,
                        span: span
                    }
                });
                SmallVector::one(P(Spanned {
                    node: StmtDecl(P(Spanned {
                            node: DeclLocal(rewritten_local),
                            span: span
                        }),
                        node_id),
                    span: stmt_span
                }))
            }
            _ => {
                noop_fold_stmt(Spanned {
                    node: StmtDecl(P(Spanned {
                            node: decl,
                            span: span
                        }),
                        node_id),
                    span: stmt_span
                }, fld)
            }
        }),
        _ => {
            noop_fold_stmt(Spanned {
                node: node,
                span: stmt_span
            }, fld)
        }
    }
}

// expand the arm of a 'match', renaming for macro hygiene
fn expand_arm(arm: ast::Arm, fld: &mut MacroExpander) -> ast::Arm {
    // expand pats... they might contain macro uses:
    let expanded_pats = arm.pats.move_map(|pat| fld.fold_pat(pat));
    if expanded_pats.len() == 0 {
        panic!("encountered match arm with 0 patterns");
    }
    // all of the pats must have the same set of bindings, so use the
    // first one to extract them and generate new names:
    let idents = pattern_bindings(&*expanded_pats[0]);
    let new_renames = idents.into_iter().map(|id| (id, fresh_name(&id))).collect();
    // apply the renaming, but only to the PatIdents:
    let mut rename_pats_fld = PatIdentRenamer{renames:&new_renames};
    let rewritten_pats = expanded_pats.move_map(|pat| rename_pats_fld.fold_pat(pat));
    // apply renaming and then expansion to the guard and the body:
    let mut rename_fld = IdentRenamer{renames:&new_renames};
    let rewritten_guard =
        arm.guard.map(|g| fld.fold_expr(rename_fld.fold_expr(g)));
    let rewritten_body = fld.fold_expr(rename_fld.fold_expr(arm.body));
    ast::Arm {
        attrs: fold::fold_attrs(arm.attrs, fld),
        pats: rewritten_pats,
        guard: rewritten_guard,
        body: rewritten_body,
    }
}

/// A visitor that extracts the PatIdent (binding) paths
/// from a given thingy and puts them in a mutable
/// array
#[derive(Clone)]
struct PatIdentFinder {
    ident_accumulator: Vec<ast::Ident>
}

impl<'v> Visitor<'v> for PatIdentFinder {
    fn visit_pat(&mut self, pattern: &ast::Pat) {
        match *pattern {
            ast::Pat { id: _, node: ast::PatIdent(_, ref path1, ref inner), span: _ } => {
                self.ident_accumulator.push(path1.node);
                // visit optional subpattern of PatIdent:
                if let Some(ref subpat) = *inner {
                    self.visit_pat(&**subpat)
                }
            }
            // use the default traversal for non-PatIdents
            _ => visit::walk_pat(self, pattern)
        }
    }
}

/// find the PatIdent paths in a pattern
fn pattern_bindings(pat: &ast::Pat) -> Vec<ast::Ident> {
    let mut name_finder = PatIdentFinder{ident_accumulator:Vec::new()};
    name_finder.visit_pat(pat);
    name_finder.ident_accumulator
}

/// find the PatIdent paths in a
fn fn_decl_arg_bindings(fn_decl: &ast::FnDecl) -> Vec<ast::Ident> {
    let mut pat_idents = PatIdentFinder{ident_accumulator:Vec::new()};
    for arg in &fn_decl.inputs {
        pat_idents.visit_pat(&*arg.pat);
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
    b.map(|Block {id, stmts, expr, rules, span}| {
        let new_stmts = stmts.into_iter().flat_map(|x| {
            // perform all pending renames
            let renamed_stmt = {
                let pending_renames = &mut fld.cx.syntax_env.info().pending_renames;
                let mut rename_fld = IdentRenamer{renames:pending_renames};
                rename_fld.fold_stmt(x).expect_one("rename_fold didn't return one value")
            };
            // expand macros in the statement
            fld.fold_stmt(renamed_stmt).into_iter()
        }).collect();
        let new_expr = expr.map(|x| {
            let expr = {
                let pending_renames = &mut fld.cx.syntax_env.info().pending_renames;
                let mut rename_fld = IdentRenamer{renames:pending_renames};
                rename_fld.fold_expr(x)
            };
            fld.fold_expr(expr)
        });
        Block {
            id: fld.new_id(id),
            stmts: new_stmts,
            expr: new_expr,
            rules: rules,
            span: span
        }
    })
}

fn expand_pat(p: P<ast::Pat>, fld: &mut MacroExpander) -> P<ast::Pat> {
    match p.node {
        PatMac(_) => {}
        _ => return noop_fold_pat(p, fld)
    }
    p.map(|ast::Pat {node, span, ..}| {
        let (pth, tts) = match node {
            PatMac(mac) => match mac.node {
                MacInvocTT(pth, tts, _) => {
                    (pth, tts)
                }
            },
            _ => unreachable!()
        };
        if pth.segments.len() > 1 {
            fld.cx.span_err(pth.span, "expected macro name without module separators");
            return DummyResult::raw_pat(span);
        }
        let extname = pth.segments[0].identifier;
        let extnamestr = token::get_ident(extname);
        let marked_after = match fld.cx.syntax_env.find(&extname.name) {
            None => {
                fld.cx.span_err(pth.span,
                                &format!("macro undefined: '{}!'",
                                        extnamestr)[]);
                // let compilation continue
                return DummyResult::raw_pat(span);
            }

            Some(rc) => match *rc {
                NormalTT(ref expander, tt_span) => {
                    fld.cx.bt_push(ExpnInfo {
                        call_site: span,
                        callee: NameAndSpan {
                            name: extnamestr.to_string(),
                            format: MacroBang,
                            span: tt_span
                        }
                    });

                    let fm = fresh_mark();
                    let marked_before = mark_tts(&tts[..], fm);
                    let mac_span = fld.cx.original_span();
                    let expanded = match expander.expand(fld.cx,
                                        mac_span,
                                        &marked_before[..]).make_pat() {
                        Some(e) => e,
                        None => {
                            fld.cx.span_err(
                                pth.span,
                                &format!(
                                    "non-pattern macro in pattern position: {}",
                                    &extnamestr
                                    )[]
                            );
                            return DummyResult::raw_pat(span);
                        }
                    };

                    // mark after:
                    mark_pat(expanded,fm)
                }
                _ => {
                    fld.cx.span_err(span,
                                    &format!("{}! is not legal in pattern position",
                                            &extnamestr)[]);
                    return DummyResult::raw_pat(span);
                }
            }
        };

        let fully_expanded =
            fld.fold_pat(marked_after).node.clone();
        fld.cx.bt_pop();

        ast::Pat {
            id: ast::DUMMY_NODE_ID,
            node: fully_expanded,
            span: span
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
        Ident {
            name: id.name,
            ctxt: mtwt::apply_renames(self.renames, id.ctxt),
        }
    }
    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        fold::noop_fold_mac(mac, self)
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
    fn fold_pat(&mut self, pat: P<ast::Pat>) -> P<ast::Pat> {
        match pat.node {
            ast::PatIdent(..) => {},
            _ => return noop_fold_pat(pat, self)
        }

        pat.map(|ast::Pat {id, node, span}| match node {
            ast::PatIdent(binding_mode, Spanned{span: sp, node: ident}, sub) => {
                let new_ident = Ident{name: ident.name,
                                      ctxt: mtwt::apply_renames(self.renames, ident.ctxt)};
                let new_node =
                    ast::PatIdent(binding_mode,
                                  Spanned{span: self.new_span(sp), node: new_ident},
                                  sub.map(|p| self.fold_pat(p)));
                ast::Pat {
                    id: id,
                    node: new_node,
                    span: self.new_span(span)
                }
            },
            _ => unreachable!()
        })
    }
    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        fold::noop_fold_mac(mac, self)
    }
}

fn expand_annotatable(a: Annotatable,
                      fld: &mut MacroExpander)
                      -> SmallVector<Annotatable> {
    let a = expand_item_multi_modifier(a, fld);

    let mut decorator_items = SmallVector::zero();
    let mut new_attrs = Vec::new();
    for attr in a.attrs() {
        let mname = attr.name();

        match fld.cx.syntax_env.find(&intern(&mname)) {
            Some(rc) => match *rc {
                Decorator(ref dec) => {
                    let it = match a {
                        Annotatable::Item(ref it) => it,
                        // ItemDecorators are only implemented for Items.
                        _ => break,
                    };

                    attr::mark_used(attr);

                    fld.cx.bt_push(ExpnInfo {
                        call_site: attr.span,
                        callee: NameAndSpan {
                            name: mname.to_string(),
                            format: MacroAttribute,
                            span: None
                        }
                    });

                    // we'd ideally decorator_items.push_all(expand_item(item, fld)),
                    // but that double-mut-borrows fld
                    let mut items: SmallVector<P<ast::Item>> = SmallVector::zero();
                    dec.expand(fld.cx, attr.span, &*attr.node.value, &**it,
                               &mut |item| items.push(item));
                    decorator_items.extend(
                        items.into_iter()
                             .flat_map(|item| expand_item(item, fld).into_iter()));

                    fld.cx.bt_pop();
                }
                _ => new_attrs.push((*attr).clone()),
            },
            _ => new_attrs.push((*attr).clone()),
        }
    }

    let mut new_items: SmallVector<Annotatable> = match a {
        Annotatable::Item(it) => match it.node {
            ast::ItemMac(..) => {
                expand_item_mac(it, fld).into_iter().map(|i| Annotatable::Item(i)).collect()
            }
            ast::ItemMod(_) | ast::ItemForeignMod(_) => {
                let valid_ident =
                    it.ident.name != parse::token::special_idents::invalid.name;

                if valid_ident {
                    fld.cx.mod_push(it.ident);
                }
                let macro_use = contains_macro_use(fld, &new_attrs[..]);
                let result = with_exts_frame!(fld.cx.syntax_env,
                                              macro_use,
                                              noop_fold_item(it, fld));
                if valid_ident {
                    fld.cx.mod_pop();
                }
                result.into_iter().map(|i| Annotatable::Item(i)).collect()
            },
            _ => {
                let it = P(ast::Item {
                    attrs: new_attrs,
                    ..(*it).clone()
                });
                noop_fold_item(it, fld).into_iter().map(|i| Annotatable::Item(i)).collect()
            }
        },
        Annotatable::TraitItem(it) => match it {
            ast::TraitItem::ProvidedMethod(m) => {
                expand_method(m, fld).into_iter().map(|m|
                    Annotatable::TraitItem(ast::TraitItem::ProvidedMethod(m))).collect()
            }
            ast::TraitItem::RequiredMethod(m) => {
                SmallVector::one(Annotatable::TraitItem(
                    ast::TraitItem::RequiredMethod(fld.fold_type_method(m))))
            }
            ast::TraitItem::TypeTraitItem(t) => {
                SmallVector::one(Annotatable::TraitItem(
                    ast::TraitItem::TypeTraitItem(P(fld.fold_associated_type((*t).clone())))))
            }
        },
        Annotatable::ImplItem(it) => match it {
            ast::ImplItem::MethodImplItem(m) => {
                expand_method(m, fld).into_iter().map(|m|
                    Annotatable::ImplItem(ast::ImplItem::MethodImplItem(m))).collect()
            }
            ast::ImplItem::TypeImplItem(t) => {
                SmallVector::one(Annotatable::ImplItem(
                    ast::ImplItem::TypeImplItem(P(fld.fold_typedef((*t).clone())))))
            }
        }
    };

    new_items.push_all(decorator_items.into_iter().map(|i| Annotatable::Item(i)).collect());
    new_items
}

fn expand_trait_item(i: ast::TraitItem,
                     fld: &mut MacroExpander)
                     -> SmallVector<ast::TraitItem> {
    expand_annotatable(Annotatable::TraitItem(i), fld)
        .into_iter().map(|i| i.expect_trait_item()).collect()

}

fn expand_impl_item(i: ast::ImplItem,
                    fld: &mut MacroExpander)
                    -> SmallVector<ast::ImplItem> {
    expand_annotatable(Annotatable::ImplItem(i), fld)
        .into_iter().map(|i| i.expect_impl_item()).collect()
}

// partition the attributes into ItemModifiers and others
fn modifiers(attrs: &Vec<ast::Attribute>,
             fld: &MacroExpander)
             -> (Vec<ast::Attribute>, Vec<ast::Attribute>) {
    attrs.iter().cloned().partition(|attr| {
        match fld.cx.syntax_env.find(&intern(&attr.name())) {
            Some(rc) => match *rc {
                Modifier(_) => true,
                _ => false
            },
            _ => false
        }
    })
}

// partition the attributes into MultiModifiers and others
fn multi_modifiers(attrs: &[ast::Attribute],
                   fld: &MacroExpander)
                   -> (Vec<ast::Attribute>, Vec<ast::Attribute>) {
    attrs.iter().cloned().partition(|attr| {
        match fld.cx.syntax_env.find(&intern(&attr.name())) {
            Some(rc) => match *rc {
                MultiModifier(_) => true,
                _ => false
            },
            _ => false
        }
    })
}

fn expand_item_multi_modifier(mut it: Annotatable,
                              fld: &mut MacroExpander)
                              -> Annotatable {
    let (modifiers, other_attrs) = multi_modifiers(it.attrs(), fld);

    // Update the attrs, leave everything else alone. Is this mutation really a good idea?
    it = it.fold_attrs(other_attrs);

    if modifiers.is_empty() {
        return it
    }

    for attr in &modifiers {
        let mname = attr.name();

        match fld.cx.syntax_env.find(&intern(&mname)) {
            Some(rc) => match *rc {
                MultiModifier(ref mac) => {
                    attr::mark_used(attr);
                    fld.cx.bt_push(ExpnInfo {
                        call_site: attr.span,
                        callee: NameAndSpan {
                            name: mname.to_string(),
                            format: MacroAttribute,
                            span: None,
                        }
                    });
                    it = mac.expand(fld.cx, attr.span, &*attr.node.value, it);
                    fld.cx.bt_pop();
                }
                _ => unreachable!()
            },
            _ => unreachable!()
        }
    }

    // Expansion may have added new ItemModifiers.
    expand_item_multi_modifier(it, fld)
}

// expand a method
fn expand_method(m: P<ast::Method>, fld: &mut MacroExpander) -> SmallVector<P<ast::Method>> {
    m.and_then(|m| match m.node {
        ast::MethDecl(ident,
                      generics,
                      abi,
                      explicit_self,
                      fn_style,
                      decl,
                      body,
                      vis) => {
            let id = fld.new_id(m.id);
            let (rewritten_fn_decl, rewritten_body)
                = expand_and_rename_fn_decl_and_block(decl, body, fld);
            SmallVector::one(P(ast::Method {
                    attrs: fold::fold_attrs(m.attrs, fld),
                    id: id,
                    span: fld.new_span(m.span),
                    node: ast::MethDecl(fld.fold_ident(ident),
                                        noop_fold_generics(generics, fld),
                                        abi,
                                        fld.fold_explicit_self(explicit_self),
                                        fn_style,
                                        rewritten_fn_decl,
                                        rewritten_body,
                                        vis)
                }))
        },
        ast::MethMac(mac) => {
            let maybe_new_methods =
                expand_mac_invoc(mac, m.span,
                                 |r| r.make_methods(),
                                 |meths, mark| meths.move_map(|m| mark_method(m, mark)),
                                 fld);

            match maybe_new_methods {
                Some(methods) => {
                    // expand again if necessary
                    let new_methods = methods.into_iter()
                                             .flat_map(|m| fld.fold_method(m).into_iter())
                                             .collect();
                    fld.cx.bt_pop();
                    new_methods
                }
                None => SmallVector::zero()
            }
        }
    })
}

/// Given a fn_decl and a block and a MacroExpander, expand the fn_decl, then use the
/// PatIdents in its arguments to perform renaming in the FnDecl and
/// the block, returning both the new FnDecl and the new Block.
fn expand_and_rename_fn_decl_and_block(fn_decl: P<ast::FnDecl>, block: P<ast::Block>,
                                       fld: &mut MacroExpander)
    -> (P<ast::FnDecl>, P<ast::Block>) {
    let expanded_decl = fld.fold_fn_decl(fn_decl);
    let idents = fn_decl_arg_bindings(&*expanded_decl);
    let renames =
        idents.iter().map(|id : &ast::Ident| (*id,fresh_name(id))).collect();
    // first, a renamer for the PatIdents, for the fn_decl:
    let mut rename_pat_fld = PatIdentRenamer{renames: &renames};
    let rewritten_fn_decl = rename_pat_fld.fold_fn_decl(expanded_decl);
    // now, a renamer for *all* idents, for the body:
    let mut rename_fld = IdentRenamer{renames: &renames};
    let rewritten_body = fld.fold_block(rename_fld.fold_block(block));
    (rewritten_fn_decl,rewritten_body)
}

/// A tree-folder that performs macro expansion
pub struct MacroExpander<'a, 'b:'a> {
    pub cx: &'a mut ExtCtxt<'b>,
    // The type of the impl currently being expanded.
    current_impl_type: Option<P<ast::Ty>>,
}

impl<'a, 'b> MacroExpander<'a, 'b> {
    pub fn new(cx: &'a mut ExtCtxt<'b>) -> MacroExpander<'a, 'b> {
        MacroExpander { cx: cx, current_impl_type: None }
    }
}

impl<'a, 'b> Folder for MacroExpander<'a, 'b> {
    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        expand_expr(expr, self)
    }

    fn fold_pat(&mut self, pat: P<ast::Pat>) -> P<ast::Pat> {
        expand_pat(pat, self)
    }

    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        let prev_type = self.current_impl_type.clone();
        if let ast::Item_::ItemImpl(_, _, _, _, ref ty, _) = item.node {
            self.current_impl_type = Some(ty.clone());
        }

        let result = expand_item(item, self);
        self.current_impl_type = prev_type;
        result
    }

    fn fold_item_underscore(&mut self, item: ast::Item_) -> ast::Item_ {
        expand_item_underscore(item, self)
    }

    fn fold_stmt(&mut self, stmt: P<ast::Stmt>) -> SmallVector<P<ast::Stmt>> {
        stmt.and_then(|stmt| expand_stmt(stmt, self))
    }

    fn fold_block(&mut self, block: P<Block>) -> P<Block> {
        expand_block(block, self)
    }

    fn fold_arm(&mut self, arm: ast::Arm) -> ast::Arm {
        expand_arm(arm, self)
    }

    fn fold_trait_item(&mut self, i: ast::TraitItem) -> SmallVector<ast::TraitItem> {
        expand_trait_item(i, self)
    }

    fn fold_impl_item(&mut self, i: ast::ImplItem) -> SmallVector<ast::ImplItem> {
        expand_impl_item(i, self)
    }

    fn fold_method(&mut self, method: P<ast::Method>) -> SmallVector<P<ast::Method>> {
        expand_method(method, self)
    }

    fn fold_ty(&mut self, t: P<ast::Ty>) -> P<ast::Ty> {
        let impl_type = self.current_impl_type.clone();
        expand_type(t, self, impl_type)
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
        expn_id: cx.backtrace(),
    }
}

pub struct ExpansionConfig<'feat> {
    pub crate_name: String,
    pub features: Option<&'feat Features>,
    pub recursion_limit: usize,
}

impl<'feat> ExpansionConfig<'feat> {
    pub fn default(crate_name: String) -> ExpansionConfig<'static> {
        ExpansionConfig {
            crate_name: crate_name,
            features: None,
            recursion_limit: 64,
        }
    }

    pub fn enable_quotes(&self) -> bool {
        match self.features {
            Some(&Features { allow_quote: true, .. }) => true,
            _ => false,
        }
    }

    pub fn enable_asm(&self) -> bool {
        match self.features {
            Some(&Features { allow_asm: true, .. }) => true,
            _ => false,
        }
    }

    pub fn enable_log_syntax(&self) -> bool {
        match self.features {
            Some(&Features { allow_log_syntax: true, .. }) => true,
            _ => false,
        }
    }

    pub fn enable_concat_idents(&self) -> bool {
        match self.features {
            Some(&Features { allow_concat_idents: true, .. }) => true,
            _ => false,
        }
    }

    pub fn enable_trace_macros(&self) -> bool {
        match self.features {
            Some(&Features { allow_trace_macros: true, .. }) => true,
            _ => false,
        }
    }
}

pub fn expand_crate<'feat>(parse_sess: &parse::ParseSess,
                           cfg: ExpansionConfig<'feat>,
                           // these are the macros being imported to this crate:
                           imported_macros: Vec<ast::MacroDef>,
                           user_exts: Vec<NamedSyntaxExtension>,
                           c: Crate) -> Crate {
    let mut cx = ExtCtxt::new(parse_sess, c.config.clone(), cfg);
    cx.use_std = std_inject::use_std(&c);

    let mut expander = MacroExpander::new(&mut cx);

    for def in imported_macros {
        expander.cx.insert_macro(def);
    }

    for (name, extension) in user_exts {
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
    fn fold_mac(&mut self, Spanned {node, span}: ast::Mac) -> ast::Mac {
        Spanned {
            node: match node {
                MacInvocTT(path, tts, ctxt) => {
                    MacInvocTT(self.fold_path(path),
                               self.fold_tts(&tts[..]),
                               mtwt::apply_mark(self.mark, ctxt))
                }
            },
            span: span,
        }
    }
}

// apply a given mark to the given token trees. Used prior to expansion of a macro.
fn mark_tts(tts: &[TokenTree], m: Mrk) -> Vec<TokenTree> {
    noop_fold_tts(tts, &mut Marker{mark:m})
}

// apply a given mark to the given expr. Used following the expansion of a macro.
fn mark_expr(expr: P<ast::Expr>, m: Mrk) -> P<ast::Expr> {
    Marker{mark:m}.fold_expr(expr)
}

// apply a given mark to the given pattern. Used following the expansion of a macro.
fn mark_pat(pat: P<ast::Pat>, m: Mrk) -> P<ast::Pat> {
    Marker{mark:m}.fold_pat(pat)
}

// apply a given mark to the given stmt. Used following the expansion of a macro.
fn mark_stmt(expr: P<ast::Stmt>, m: Mrk) -> P<ast::Stmt> {
    Marker{mark:m}.fold_stmt(expr)
        .expect_one("marking a stmt didn't return exactly one stmt")
}

// apply a given mark to the given item. Used following the expansion of a macro.
fn mark_item(expr: P<ast::Item>, m: Mrk) -> P<ast::Item> {
    Marker{mark:m}.fold_item(expr)
        .expect_one("marking an item didn't return exactly one item")
}

// apply a given mark to the given item. Used following the expansion of a macro.
fn mark_method(expr: P<ast::Method>, m: Mrk) -> P<ast::Method> {
    Marker{mark:m}.fold_method(expr)
        .expect_one("marking an item didn't return exactly one method")
}

/// Check that there are no macro invocations left in the AST:
pub fn check_for_macros(sess: &parse::ParseSess, krate: &ast::Crate) {
    visit::walk_crate(&mut MacroExterminator{sess:sess}, krate);
}

/// A visitor that ensures that no macro invocations remain in an AST.
struct MacroExterminator<'a>{
    sess: &'a parse::ParseSess
}

impl<'a, 'v> Visitor<'v> for MacroExterminator<'a> {
    fn visit_mac(&mut self, mac: &ast::Mac) {
        self.sess.span_diagnostic.span_bug(mac.span,
                                           "macro exterminator: expected AST \
                                           with no macro invocations");
    }
}


#[cfg(test)]
mod test {
    use super::{pattern_bindings, expand_crate};
    use super::{PatIdentFinder, IdentRenamer, PatIdentRenamer, ExpansionConfig};
    use ast;
    use ast::Name;
    use codemap;
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

    impl<'v> Visitor<'v> for PathExprFinderContext {
        fn visit_expr(&mut self, expr: &ast::Expr) {
            match expr.node {
                ast::ExprPath(ref p) => {
                    self.path_accumulator.push(p.clone());
                    // not calling visit_path, but it should be fine.
                }
                _ => visit::walk_expr(self, expr)
            }
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

    impl<'v> Visitor<'v> for IdentFinder {
        fn visit_ident(&mut self, _: codemap::Span, id: ast::Ident){
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
    #[should_fail]
    #[test] fn macros_cant_escape_fns_test () {
        let src = "fn bogus() {macro_rules! z (() => (3+4));}\
                   fn inty() -> i32 { z!() }".to_string();
        let sess = parse::new_parse_sess();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess);
        // should fail:
        expand_crate(&sess,test_ecfg(),vec!(),vec!(),crate_ast);
    }

    // make sure that macros can't escape modules
    #[should_fail]
    #[test] fn macros_cant_escape_mods_test () {
        let src = "mod foo {macro_rules! z (() => (3+4));}\
                   fn inty() -> i32 { z!() }".to_string();
        let sess = parse::new_parse_sess();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess);
        expand_crate(&sess,test_ecfg(),vec!(),vec!(),crate_ast);
    }

    // macro_use modules should allow macros to escape
    #[test] fn macros_can_escape_flattened_mods_test () {
        let src = "#[macro_use] mod foo {macro_rules! z (() => (3+4));}\
                   fn inty() -> i32 { z!() }".to_string();
        let sess = parse::new_parse_sess();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess);
        expand_crate(&sess, test_ecfg(), vec!(), vec!(), crate_ast);
    }

    fn expand_crate_str(crate_str: String) -> ast::Crate {
        let ps = parse::new_parse_sess();
        let crate_ast = string_to_parser(&ps, crate_str).parse_crate_mod();
        // the cfg argument actually does matter, here...
        expand_crate(&ps,test_ecfg(),vec!(),vec!(),crate_ast)
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
    type RenamingTest = (&'static str, Vec<Vec<usize>>, bool);

    #[test]
    fn automatic_renaming () {
        let tests: Vec<RenamingTest> =
            vec!(// b & c should get new names throughout, in the expr too:
                ("fn a() -> i32 { let b = 13; let c = b; b+c }",
                 vec!(vec!(0,1),vec!(2)), false),
                // both x's should be renamed (how is this causing a bug?)
                ("fn main () {let x: i32 = 13;x;}",
                 vec!(vec!(0)), false),
                // the use of b after the + should be renamed, the other one not:
                ("macro_rules! f (($x:ident) => (b + $x)); fn a() -> i32 { let b = 13; f!(b)}",
                 vec!(vec!(1)), false),
                // the b before the plus should not be renamed (requires marks)
                ("macro_rules! f (($x:ident) => ({let b=9; ($x + b)})); fn a() -> i32 { f!(b)}",
                 vec!(vec!(1)), false),
                // the marks going in and out of letty should cancel, allowing that $x to
                // capture the one following the semicolon.
                // this was an awesome test case, and caught a *lot* of bugs.
                ("macro_rules! letty(($x:ident) => (let $x = 15;));
                  macro_rules! user(($x:ident) => ({letty!($x); $x}));
                  fn main() -> i32 {user!(z)}",
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
              ({macro_rules! f(($y:ident)=>({let $y=3;$x}));f!($x)}));
              fn a(){g!(z)}",
              vec!(vec!(0)),false),
            0)
    }

    // match variable hygiene. Should expand into
    // fn z() {match 8 {x_1 => {match 9 {x_2 | x_2 if x_2 == x_1 => x_2 + x_1}}}}
    #[test] fn issue_9384(){
        run_renaming_test(
            &("macro_rules! bad_macro (($ex:expr) => ({match 9 {x | x if x == $ex => x + $ex}}));
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
            &("macro_rules! inner ( ($e:pat ) => ($e));
              macro_rules! outer ( ($e:pat ) => (inner!($e)));
              fn main() { let outer!(g) = 13; g;}",
              vec!(vec!(0)),
              true),
            0)
    }

    // create a really evil test case where a $x appears inside a binding of $x
    // but *shouldn't* bind because it was inserted by a different macro....
    // can't write this test case until we have macro-generating macros.

    // method arg hygiene
    // method expands to fn get_x(&self_0, x_1: i32) {self_0 + self_2 + x_3 + x_1}
    #[test] fn method_arg_hygiene(){
        run_renaming_test(
            &("macro_rules! inject_x (()=>(x));
              macro_rules! inject_self (()=>(self));
              struct A;
              impl A{fn get_x(&self, x: i32) {self + inject_self!() + inject_x!() + x;} }",
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
              (impl $T {  fn thingy(&self) {self;} }));
              add_method!(A);",
              vec!(vec!(0)),
              true),
            0)
    }

    // item fn hygiene
    // expands to fn q(x_1: i32){fn g(x_2: i32){x_2 + x_1};}
    #[test] fn issue_9383(){
        run_renaming_test(
            &("macro_rules! bad_macro (($ex:expr) => (fn g(x: i32){ x + $ex }));
              fn q(x: i32) { bad_macro!(x); }",
              vec!(vec!(1),vec!(0)),true),
            0)
    }

    // closure arg hygiene (ExprClosure)
    // expands to fn f(){(|x_1 : i32| {(x_2 + x_1)})(3);}
    #[test] fn closure_arg_hygiene(){
        run_renaming_test(
            &("macro_rules! inject_x (()=>(x));
            fn f(){(|x : i32| {(inject_x!() + x)})(3);}",
              vec!(vec!(1)),
              true),
            0)
    }

    // macro_rules in method position. Sadly, unimplemented.
    #[test] fn macro_in_method_posn(){
        expand_crate_str(
            "macro_rules! my_method (() => (fn thirteen(&self) -> i32 {13}));
            struct A;
            impl A{ my_method!(); }
            fn f(){A.thirteen;}".to_string());
    }

    // another nested macro
    // expands to impl Entries {fn size_hint(&self_1) {self_1;}
    #[test] fn item_macro_workaround(){
        run_renaming_test(
            &("macro_rules! item { ($i:item) => {$i}}
              struct Entries;
              macro_rules! iterator_impl {
              () => { item!( impl Entries { fn size_hint(&self) { self;}});}}
              iterator_impl! { }",
              vec!(vec!(0)), true),
            0)
    }

    // run one of the renaming tests
    fn run_renaming_test(t: &RenamingTest, test_idx: usize) {
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
            let binding_name = mtwt::resolve(bindings[binding_idx]);
            let binding_marks = mtwt::marksof(bindings[binding_idx].ctxt, invalid_name);
            // shouldmatch can't name varrefs that don't exist:
            assert!((shouldmatch.len() == 0) ||
                    (varrefs.len() > *shouldmatch.iter().max().unwrap()));
            for (idx,varref) in varrefs.iter().enumerate() {
                let print_hygiene_debug_info = || {
                    // good lord, you can't make a path with 0 segments, can you?
                    let final_varref_ident = match varref.segments.last() {
                        Some(pathsegment) => pathsegment.identifier,
                        None => panic!("varref with 0 path segments?")
                    };
                    let varref_name = mtwt::resolve(final_varref_ident);
                    let varref_idents : Vec<ast::Ident>
                        = varref.segments.iter().map(|s| s.identifier)
                        .collect();
                    println!("varref #{}: {:?}, resolves to {}",idx, varref_idents, varref_name);
                    let string = token::get_ident(final_varref_ident);
                    println!("varref's first segment's string: \"{}\"", &string[..]);
                    println!("binding #{}: {}, resolves to {}",
                             binding_idx, bindings[binding_idx], binding_name);
                    mtwt::with_sctable(|x| mtwt::display_sctable(x));
                };
                if shouldmatch.contains(&idx) {
                    // it should be a path of length 1, and it should
                    // be free-identifier=? or bound-identifier=? to the given binding
                    assert_eq!(varref.segments.len(),1);
                    let varref_name = mtwt::resolve(varref.segments[0].identifier);
                    let varref_marks = mtwt::marksof(varref.segments[0]
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
                    let varref_name = mtwt::resolve(varref.segments[0].identifier);
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
        let crate_str = "macro_rules! fmt_wrap(($b:expr)=>($b.to_string()));
macro_rules! foo_module (() => (mod generated { fn a() { let xx = 147; fmt_wrap!(xx);}}));
foo_module!();
".to_string();
        let cr = expand_crate_str(crate_str);
        // find the xx binding
        let bindings = crate_bindings(&cr);
        let cxbinds: Vec<&ast::Ident> =
            bindings.iter().filter(|b| {
                let ident = token::get_ident(**b);
                let string = &ident[..];
                "xx" == string
            }).collect();
        let cxbinds: &[&ast::Ident] = &cxbinds[..];
        let cxbind = match cxbinds {
            [b] => b,
            _ => panic!("expected just one binding for ext_cx")
        };
        let resolved_binding = mtwt::resolve(*cxbind);
        let varrefs = crate_varrefs(&cr);

        // the xx binding should bind all of the xx varrefs:
        for (idx,v) in varrefs.iter().filter(|p| {
            p.segments.len() == 1
            && "xx" == &token::get_ident(p.segments[0].identifier)[]
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
        let idents = pattern_bindings(&*pat);
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
        assert_eq!(resolved,vec!(f_ident.name,Name(16),int_ident.name,Name(16),Name(16),Name(16)));
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
        assert_eq!(resolved,vec!(f_ident.name,Name(16),int_ident.name,Name(16),x_name,x_name));
    }


}
