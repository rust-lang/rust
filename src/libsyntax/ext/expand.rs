// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{Block, Crate, DeclLocal, Expr_, ExprMac, SyntaxContext};
use ast::{Local, Ident, mac_invoc_tt};
use ast::{item_mac, Mrk, Stmt, StmtDecl, StmtMac, StmtExpr, StmtSemi};
use ast::{token_tree};
use ast;
use ast_util::{mtwt_outer_mark, new_rename, new_mark};
use ext::build::AstBuilder;
use attr;
use attr::AttrMetaMethods;
use codemap;
use codemap::{Span, Spanned, ExpnInfo, NameAndSpan};
use ext::base::*;
use fold::*;
use opt_vec;
use parse;
use parse::{parse_item_from_source_str};
use parse::token;
use parse::token::{fresh_mark, fresh_name, ident_to_str, intern};
use visit;
use visit::Visitor;

use std::vec;

pub fn expand_expr(extsbox: @mut SyntaxEnv,
                   cx: @ExtCtxt,
                   e: @ast::Expr,
                   fld: &MacroExpander)
                   -> @ast::Expr {
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
                mac_invoc_tt(ref pth, ref tts, ctxt) => {
                    if (pth.segments.len() > 1u) {
                        cx.span_fatal(
                            pth.span,
                            fmt!("expected macro name without module \
                                  separators"));
                    }
                    let extname = &pth.segments[0].identifier;
                    let extnamestr = ident_to_str(extname);
                    // leaving explicit deref here to highlight unbox op:
                    match (*extsbox).find(&extname.name) {
                        None => {
                            cx.span_fatal(
                                pth.span,
                                fmt!("macro undefined: '%s'", extnamestr))
                        }
                        Some(@SE(NormalTT(expandfun, exp_span))) => {
                            cx.bt_push(ExpnInfo {
                                call_site: e.span,
                                callee: NameAndSpan {
                                    name: extnamestr,
                                    span: exp_span,
                                },
                            });
                            let fm = fresh_mark();
                            // mark before:
                            let marked_before = mark_tts(*tts,fm);
                            let marked_ctxt = new_mark(fm, ctxt);

                            // The span that we pass to the expanders we want to
                            // be the root of the call stack. That's the most
                            // relevant span and it's the actual invocation of
                            // the macro.
                            let mac_span = original_span(cx);

                            let expanded =
                                match expandfun.expand(cx,
                                                       mac_span.call_site,
                                                       marked_before,
                                                       marked_ctxt) {
                                    MRExpr(e) => e,
                                    MRAny(any_macro) => any_macro.make_expr(),
                                    _ => {
                                        cx.span_fatal(
                                            pth.span,
                                            fmt!(
                                                "non-expr macro in expr pos: %s",
                                                extnamestr
                                            )
                                        )
                                    }
                                };
                            // mark after:
                            let marked_after = mark_expr(expanded,fm);

                            // Keep going, outside-in.
                            //
                            // XXX(pcwalton): Is it necessary to clone the
                            // node here?
                            let fully_expanded =
                                fld.fold_expr(marked_after).node.clone();
                            cx.bt_pop();

                            @ast::Expr {
                                id: ast::DUMMY_NODE_ID,
                                node: fully_expanded,
                                span: e.span,
                            }
                        }
                        _ => {
                            cx.span_fatal(
                                pth.span,
                                fmt!("'%s' is not a tt-style macro", extnamestr)
                            )
                        }
                    }
                }
            }
        }

        // Desugar expr_for_loop
        // From: `['<ident>:] for <src_pat> in <src_expr> <src_loop_block>`
        // FIXME #6993 : change type of opt_ident to Option<Name>
        ast::ExprForLoop(src_pat, src_expr, ref src_loop_block, opt_ident) => {
            // Expand any interior macros etc.
            // NB: we don't fold pats yet. Curious.
            let src_expr = fld.fold_expr(src_expr).clone();
            let src_loop_block = fld.fold_block(src_loop_block).clone();

            let span = e.span;

            pub fn mk_expr(_: @ExtCtxt, span: Span, node: Expr_)
                           -> @ast::Expr {
                @ast::Expr {
                    id: ast::DUMMY_NODE_ID,
                    node: node,
                    span: span,
                }
            }

            fn mk_block(_: @ExtCtxt,
                        stmts: &[@ast::Stmt],
                        expr: Option<@ast::Expr>,
                        span: Span)
                        -> ast::Block {
                ast::Block {
                    view_items: ~[],
                    stmts: stmts.to_owned(),
                    expr: expr,
                    id: ast::DUMMY_NODE_ID,
                    rules: ast::DefaultBlock,
                    span: span,
                }
            }

            fn mk_simple_path(ident: ast::Ident, span: Span) -> ast::Path {
                ast::Path {
                    span: span,
                    global: false,
                    segments: ~[
                        ast::PathSegment {
                            identifier: ident,
                            lifetime: None,
                            types: opt_vec::Empty,
                        }
                    ],
                }
            }

            // to:
            //
            // {
            //   let _i = &mut <src_expr>;
            //   ['<ident>:] loop {
            //       match i.next() {
            //           None => break,
            //           Some(<src_pat>) => <src_loop_block>
            //       }
            //   }
            // }

            let local_ident = token::gensym_ident("i");
            let next_ident = cx.ident_of("next");
            let none_ident = cx.ident_of("None");

            let local_path = cx.path_ident(span, local_ident);
            let some_path = cx.path_ident(span, cx.ident_of("Some"));

            // `let i = &mut <src_expr>`
            let iter_decl_stmt = cx.stmt_let(span, false, local_ident,
                                             cx.expr_mut_addr_of(span, src_expr));

            // `None => break ['<ident>];`
            let none_arm = {
                // FIXME #6993: this map goes away:
                let break_expr = cx.expr(span, ast::ExprBreak(opt_ident.map(|x| x.name)));
                let none_pat = cx.pat_ident(span, none_ident);
                cx.arm(span, ~[none_pat], break_expr)
            };

            // `Some(<src_pat>) => <src_loop_block>`
            let some_arm =
                cx.arm(span,
                       ~[cx.pat_enum(span, some_path, ~[src_pat])],
                       cx.expr_block(src_loop_block));

            // `match i.next() { ... }`
            let match_expr = {
                let next_call_expr =
                    cx.expr_method_call(span, cx.expr_path(local_path), next_ident, ~[]);

                cx.expr_match(span, next_call_expr, ~[none_arm, some_arm])
            };

            // ['ident:] loop { ... }
            let loop_expr = cx.expr(span,
                                    ast::ExprLoop(cx.block_expr(match_expr), opt_ident));

            // `{ let ... ;  loop { ... } }`
            let block = cx.block(span,
                                 ~[iter_decl_stmt],
                                 Some(loop_expr));

            @ast::Expr {
                id: ast::DUMMY_NODE_ID,
                node: ast::ExprBlock(block),
                span: span,
            }
        }

        _ => noop_fold_expr(e, fld)
    }
}

// This is a secondary mechanism for invoking syntax extensions on items:
// "decorator" attributes, such as #[auto_encode]. These are invoked by an
// attribute prefixing an item, and are interpreted by feeding the item
// through the named attribute _as a syntax extension_ and splicing in the
// resulting item vec into place in favour of the decorator. Note that
// these do _not_ work for macro extensions, just ItemDecorator ones.
//
// NB: there is some redundancy between this and expand_item, below, and
// they might benefit from some amount of semantic and language-UI merger.
pub fn expand_mod_items(extsbox: @mut SyntaxEnv,
                        cx: @ExtCtxt,
                        module_: &ast::_mod,
                        fld: &MacroExpander)
                        -> ast::_mod {
    // Fold the contents first:
    let module_ = noop_fold_mod(module_, fld);

    // For each item, look through the attributes.  If any of them are
    // decorated with "item decorators", then use that function to transform
    // the item into a new set of items.
    let new_items = do vec::flat_map(module_.items) |item| {
        do item.attrs.rev_iter().fold(~[*item]) |items, attr| {
            let mname = attr.name();

            match (*extsbox).find(&intern(mname)) {
              Some(@SE(ItemDecorator(dec_fn))) => {
                  cx.bt_push(ExpnInfo {
                      call_site: attr.span,
                      callee: NameAndSpan {
                          name: mname,
                          span: None
                      }
                  });
                  let r = dec_fn(cx, attr.span, attr.node.value, items);
                  cx.bt_pop();
                  r
              },
              _ => items,
            }
        }
    };

    ast::_mod {
        items: new_items,
        ..module_
    }
}

// eval $e with a new exts frame:
macro_rules! with_exts_frame (
    ($extsboxexpr:expr,$macros_escape:expr,$e:expr) =>
    ({let extsbox = $extsboxexpr;
      let oldexts = *extsbox;
      *extsbox = oldexts.push_frame();
      extsbox.insert(intern(special_block_name),
                     @BlockInfo(BlockInfo{macros_escape:$macros_escape,pending_renames:@mut ~[]}));
      let result = $e;
      *extsbox = oldexts;
      result
     })
)

static special_block_name : &'static str = " block";

// When we enter a module, record it, for the sake of `module!`
pub fn expand_item(extsbox: @mut SyntaxEnv,
                   cx: @ExtCtxt,
                   it: @ast::item,
                   fld: &MacroExpander)
                   -> Option<@ast::item> {
    match it.node {
        ast::item_mac(*) => expand_item_mac(extsbox, cx, it, fld),
        ast::item_mod(_) | ast::item_foreign_mod(_) => {
            cx.mod_push(it.ident);
            let macro_escape = contains_macro_escape(it.attrs);
            let result = with_exts_frame!(extsbox,
                                          macro_escape,
                                          noop_fold_item(it, fld));
            cx.mod_pop();
            result
        },
        _ => noop_fold_item(it, fld)
    }
}

// does this attribute list contain "macro_escape" ?
pub fn contains_macro_escape(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, "macro_escape")
}

// Support for item-position macro invocations, exactly the same
// logic as for expression-position macro invocations.
pub fn expand_item_mac(extsbox: @mut SyntaxEnv,
                       cx: @ExtCtxt,
                       it: @ast::item,
                       fld: &MacroExpander)
                       -> Option<@ast::item> {
    let (pth, tts, ctxt) = match it.node {
        item_mac(codemap::Spanned {
            node: mac_invoc_tt(ref pth, ref tts, ctxt),
            _
        }) => {
            (pth, (*tts).clone(), ctxt)
        }
        _ => cx.span_bug(it.span, "invalid item macro invocation")
    };

    let extname = &pth.segments[0].identifier;
    let extnamestr = ident_to_str(extname);
    let fm = fresh_mark();
    let expanded = match (*extsbox).find(&extname.name) {
        None => cx.span_fatal(pth.span,
                              fmt!("macro undefined: '%s!'", extnamestr)),

        Some(@SE(NormalTT(expander, span))) => {
            if it.ident.name != parse::token::special_idents::invalid.name {
                cx.span_fatal(pth.span,
                              fmt!("macro %s! expects no ident argument, \
                                    given '%s'", extnamestr,
                                   ident_to_str(&it.ident)));
            }
            cx.bt_push(ExpnInfo {
                call_site: it.span,
                callee: NameAndSpan {
                    name: extnamestr,
                    span: span
                }
            });
            // mark before expansion:
            let marked_before = mark_tts(tts,fm);
            let marked_ctxt = new_mark(fm,ctxt);
            expander.expand(cx, it.span, marked_before, marked_ctxt)
        }
        Some(@SE(IdentTT(expander, span))) => {
            if it.ident.name == parse::token::special_idents::invalid.name {
                cx.span_fatal(pth.span,
                              fmt!("macro %s! expects an ident argument",
                                   extnamestr));
            }
            cx.bt_push(ExpnInfo {
                call_site: it.span,
                callee: NameAndSpan {
                    name: extnamestr,
                    span: span
                }
            });
            // mark before expansion:
            let marked_tts = mark_tts(tts,fm);
            let marked_ctxt = new_mark(fm,ctxt);
            expander.expand(cx, it.span, it.ident, marked_tts, marked_ctxt)
        }
        _ => cx.span_fatal(
            it.span, fmt!("%s! is not legal in item position", extnamestr))
    };

    let maybe_it = match expanded {
        MRItem(it) => {
            mark_item(it,fm)
                .and_then(|i| fld.fold_item(i))
        }
        MRExpr(_) => {
            cx.span_fatal(pth.span, fmt!("expr macro in item position: %s", extnamestr))
        }
        MRAny(any_macro) => {
            any_macro.make_item()
                     .and_then(|i| mark_item(i,fm))
                     .and_then(|i| fld.fold_item(i))
        }
        MRDef(ref mdef) => {
            // yikes... no idea how to apply the mark to this. I'm afraid
            // we're going to have to wait-and-see on this one.
            insert_macro(*extsbox,intern(mdef.name), @SE((*mdef).ext));
            None
        }
    };
    cx.bt_pop();
    return maybe_it;
}


// insert a macro into the innermost frame that doesn't have the
// macro_escape tag.
fn insert_macro(exts: SyntaxEnv, name: ast::Name, transformer: @Transformer) {
    let is_non_escaping_block =
        |t : &@Transformer| -> bool{
        match t {
            &@BlockInfo(BlockInfo {macros_escape:false,_}) => true,
            &@BlockInfo(BlockInfo {_}) => false,
            _ => fail!(fmt!("special identifier %? was bound to a non-BlockInfo",
                            special_block_name))
        }
    };
    exts.insert_into_frame(name,transformer,intern(special_block_name),
                           is_non_escaping_block)
}

// expand a stmt
pub fn expand_stmt(extsbox: @mut SyntaxEnv,
                   cx: @ExtCtxt,
                   s: &Stmt,
                   fld: &MacroExpander)
                   -> Option<@Stmt> {
    // why the copying here and not in expand_expr?
    // looks like classic changed-in-only-one-place
    let (pth, tts, semi, ctxt) = match s.node {
        StmtMac(ref mac, semi) => {
            match mac.node {
                mac_invoc_tt(ref pth, ref tts, ctxt) => {
                    (pth, (*tts).clone(), semi, ctxt)
                }
            }
        }
        _ => return expand_non_macro_stmt(*extsbox, s, fld)
    };
    if (pth.segments.len() > 1u) {
        cx.span_fatal(pth.span,
                      "expected macro name without module separators");
    }
    let extname = &pth.segments[0].identifier;
    let extnamestr = ident_to_str(extname);
    let fully_expanded: @ast::Stmt = match (*extsbox).find(&extname.name) {
        None => {
            cx.span_fatal(pth.span, fmt!("macro undefined: '%s'", extnamestr))
        }

        Some(@SE(NormalTT(expandfun, exp_span))) => {
            cx.bt_push(ExpnInfo {
                call_site: s.span,
                callee: NameAndSpan {
                    name: extnamestr,
                    span: exp_span,
                }
            });
            let fm = fresh_mark();
            // mark before expansion:
            let marked_tts = mark_tts(tts,fm);
            let marked_ctxt = new_mark(fm,ctxt);

            // See the comment in expand_expr for why we want the original span,
            // not the current mac.span.
            let mac_span = original_span(cx);

            let expanded = match expandfun.expand(cx,
                                                  mac_span.call_site,
                                                  marked_tts,
                                                  marked_ctxt) {
                MRExpr(e) => {
                    @codemap::Spanned {
                        node: StmtExpr(e, ast::DUMMY_NODE_ID),
                        span: e.span,
                    }
                }
                MRAny(any_macro) => any_macro.make_stmt(),
                _ => cx.span_fatal(
                    pth.span,
                    fmt!("non-stmt macro in stmt pos: %s", extnamestr))
            };
            let marked_after = mark_stmt(expanded,fm);

            // Keep going, outside-in.
            let fully_expanded = match fld.fold_stmt(marked_after) {
                Some(stmt) => {
                    let fully_expanded = &stmt.node;
                    cx.bt_pop();
                    @Spanned {
                        span: stmt.span,
                        node: (*fully_expanded).clone(),
                    }
                }
                None => {
                    cx.span_fatal(pth.span,
                                  "macro didn't expand to a statement")
                }
            };

            fully_expanded
        }

        _ => {
            cx.span_fatal(pth.span,
                          fmt!("'%s' is not a tt-style macro", extnamestr))
        }
    };

    match fully_expanded.node {
        StmtExpr(e, stmt_id) if semi => {
            Some(@Spanned {
                span: fully_expanded.span,
                node: StmtSemi(e, stmt_id),
            })
        }
        _ => Some(fully_expanded), /* might already have a semi */
    }
}

// expand a non-macro stmt. this is essentially the fallthrough for
// expand_stmt, above.
fn expand_non_macro_stmt(exts: SyntaxEnv, s: &Stmt, fld: &MacroExpander)
                         -> Option<@Stmt> {
    // is it a let?
    match s.node {
        StmtDecl(@Spanned {
            node: DeclLocal(ref local),
            span: stmt_span
        },
        node_id) => {
            let block_info = get_block_info(exts);
            let pending_renames = block_info.pending_renames;

            // take it apart:
            let @Local{is_mutbl:is_mutbl,
                       ty:_,
                       pat:pat,
                       init:init,
                       id:id,
                       span:span
                      } = *local;
            // types can't be copied automatically because of the owned ptr in ty_tup...
            let ty = local.ty.clone();
            // expand the pat (it might contain exprs... #:(o)>
            let expanded_pat = fld.fold_pat(pat);
            // find the pat_idents in the pattern:
            // oh dear heaven... this is going to include the enum names, as well....
            // ... but that should be okay, as long as the new names are gensyms
            // for the old ones.
            let idents = @mut ~[];
            let name_finder = new_name_finder(idents);
            name_finder.visit_pat(expanded_pat,());
            // generate fresh names, push them to a new pending list
            let new_pending_renames = @mut ~[];
            for ident in idents.iter() {
                let new_name = fresh_name(ident);
                new_pending_renames.push((*ident,new_name));
            }
            let rename_fld = renames_to_fold(new_pending_renames);
            // rewrite the pattern using the new names (the old ones
            // have already been applied):
            let rewritten_pat = rename_fld.fold_pat(expanded_pat);
            // add them to the existing pending renames:
            for pr in new_pending_renames.iter() {pending_renames.push(*pr)}
            // also, don't forget to expand the init:
            let new_init_opt = init.map(|e| fld.fold_expr(*e));
            let rewritten_local =
                @Local {
                    is_mutbl: is_mutbl,
                    ty: ty,
                    pat: rewritten_pat,
                    init: new_init_opt,
                    id: id,
                    span: span,
                };
            Some(@Spanned {
                node: StmtDecl(@Spanned {
                        node: DeclLocal(rewritten_local),
                        span: stmt_span
                    },
                    node_id),
                span: span
            })
        },
        _ => noop_fold_stmt(s, fld),
    }
}

// a visitor that extracts the pat_ident paths
// from a given thingy and puts them in a mutable
// array (passed in to the traversal)
#[deriving(Clone)]
struct NewNameFinderContext {
    ident_accumulator: @mut ~[ast::Ident],
}

impl Visitor<()> for NewNameFinderContext {
    fn visit_pat(&mut self, pattern: @ast::Pat, _: ()) {
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
                        segments: [
                            ast::PathSegment {
                                identifier: id,
                                lifetime: _,
                                types: _
                            }
                        ]
                    } => self.ident_accumulator.push(id),
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

// a visitor that extracts the paths
// from a given thingy and puts them in a mutable
// array (passed in to the traversal)
#[deriving(Clone)]
struct NewPathExprFinderContext {
    path_accumulator: @mut ~[ast::Path],
}

impl Visitor<()> for NewPathExprFinderContext {

    fn visit_expr(&mut self, expr: @ast::Expr, _: ()) {
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

// return a visitor that extracts the pat_ident paths
// from a given thingy and puts them in a mutable
// array (passed in to the traversal)
pub fn new_name_finder(idents: @mut ~[ast::Ident]) -> @mut Visitor<()> {
    let context = @mut NewNameFinderContext {
        ident_accumulator: idents,
    };
    context as @mut Visitor<()>
}

// return a visitor that extracts the paths
// from a given pattern and puts them in a mutable
// array (passed in to the traversal)
pub fn new_path_finder(paths: @mut ~[ast::Path]) -> @mut Visitor<()> {
    let context = @mut NewPathExprFinderContext {
        path_accumulator: paths,
    };
    context as @mut Visitor<()>
}

// expand a block. pushes a new exts_frame, then calls expand_block_elts
pub fn expand_block(extsbox: @mut SyntaxEnv,
                    _: @ExtCtxt,
                    blk: &Block,
                    fld: &MacroExpander)
                    -> Block {
    // see note below about treatment of exts table
    with_exts_frame!(extsbox,false,
                     expand_block_elts(*extsbox, blk, fld))
}

// expand the elements of a block.
pub fn expand_block_elts(exts: SyntaxEnv, b: &Block, fld: &MacroExpander)
                         -> Block {
    let block_info = get_block_info(exts);
    let pending_renames = block_info.pending_renames;
    let rename_fld = renames_to_fold(pending_renames);
    let new_view_items = b.view_items.map(|x| fld.fold_view_item(x));
    let mut new_stmts = ~[];
    for x in b.stmts.iter() {
        match fld.fold_stmt(mustbesome(rename_fld.fold_stmt(*x))) {
            Some(s) => new_stmts.push(s),
            None => ()
        }
    }
    let new_expr = b.expr.map(|x| fld.fold_expr(rename_fld.fold_expr(*x)));
    Block{
        view_items: new_view_items,
        stmts: new_stmts,
        expr: new_expr,
        id: fld.new_id(b.id),
        rules: b.rules,
        span: b.span,
    }
}

// rename_fold should never return "None".
// (basically, just .get() with a better message...)
fn mustbesome<T>(val : Option<T>) -> T {
    match val {
        Some(v) => v,
        None => fail!("rename_fold returned None")
    }
}

// get the (innermost) BlockInfo from an exts stack
fn get_block_info(exts : SyntaxEnv) -> BlockInfo {
    match exts.find_in_topmost_frame(&intern(special_block_name)) {
        Some(@BlockInfo(bi)) => bi,
        _ => fail!(fmt!("special identifier %? was bound to a non-BlockInfo",
                       @" block"))
    }
}

struct IdentRenamer {
    renames: @mut ~[(ast::Ident,ast::Name)],
}

impl ast_fold for IdentRenamer {
    fn fold_ident(&self, id: ast::Ident) -> ast::Ident {
        let new_ctxt = self.renames.iter().fold(id.ctxt, |ctxt, &(from, to)| {
            new_rename(from, to, ctxt)
        });
        ast::Ident {
            name: id.name,
            ctxt: new_ctxt,
        }
    }
}

// given a mutable list of renames, return a tree-folder that applies those
// renames.
pub fn renames_to_fold(renames: @mut ~[(ast::Ident,ast::Name)]) -> @ast_fold {
    @IdentRenamer {
        renames: renames,
    } as @ast_fold
}

// perform a bunch of renames
fn apply_pending_renames(folder : @ast_fold, stmt : ast::Stmt) -> @ast::Stmt {
    match folder.fold_stmt(&stmt) {
        Some(s) => s,
        None => fail!(fmt!("renaming of stmt produced None"))
    }
}



pub fn new_span(cx: @ExtCtxt, sp: Span) -> Span {
    /* this discards information in the case of macro-defining macros */
    Span {
        lo: sp.lo,
        hi: sp.hi,
        expn_info: cx.backtrace(),
    }
}

// FIXME (#2247): this is a moderately bad kludge to inject some macros into
// the default compilation environment in that it injects strings, rather than
// syntax elements.

pub fn std_macros() -> @str {
    return
@"mod __std_macros {
    #[macro_escape];
    #[doc(hidden)];

    macro_rules! ignore (($($x:tt)*) => (()))

    macro_rules! log(
        ($lvl:expr, $arg:expr) => ({
            let lvl = $lvl;
            if lvl <= __log_level() {
                format_args!(|args| {
                    ::std::logging::log(lvl, args)
                }, \"{}\", fmt!(\"%?\", $arg))
            }
        });
        ($lvl:expr, $($arg:expr),+) => ({
            let lvl = $lvl;
            if lvl <= __log_level() {
                format_args!(|args| {
                    ::std::logging::log(lvl, args)
                }, \"{}\", fmt!($($arg),+))
            }
        })
    )
    macro_rules! error( ($($arg:tt)*) => (log!(1u32, $($arg)*)) )
    macro_rules! warn ( ($($arg:tt)*) => (log!(2u32, $($arg)*)) )
    macro_rules! info ( ($($arg:tt)*) => (log!(3u32, $($arg)*)) )
    macro_rules! debug( ($($arg:tt)*) => (
        if cfg!(not(ndebug)) { log!(4u32, $($arg)*) }
    ))

    macro_rules! log2(
        ($lvl:expr, $($arg:tt)+) => ({
            let lvl = $lvl;
            if lvl <= __log_level() {
                format_args!(|args| {
                    ::std::logging::log(lvl, args)
                }, $($arg)+)
            }
        })
    )
    macro_rules! error2( ($($arg:tt)*) => (log2!(1u32, $($arg)*)) )
    macro_rules! warn2 ( ($($arg:tt)*) => (log2!(2u32, $($arg)*)) )
    macro_rules! info2 ( ($($arg:tt)*) => (log2!(3u32, $($arg)*)) )
    macro_rules! debug2( ($($arg:tt)*) => (
        if cfg!(not(ndebug)) { log2!(4u32, $($arg)*) }
    ))

    macro_rules! fail(
        () => (
            fail!(\"explicit failure\")
        );
        ($msg:expr) => (
            ::std::sys::FailWithCause::fail_with($msg, file!(), line!())
        );
        ($( $arg:expr ),+) => (
            ::std::sys::FailWithCause::fail_with(fmt!( $($arg),+ ), file!(), line!())
        )
    )

    macro_rules! fail2(
        () => (
            fail!(\"explicit failure\")
        );
        ($($arg:tt)*) => (
            ::std::sys::FailWithCause::fail_with(format!($($arg)*), file!(), line!())
        )
    )

    macro_rules! assert(
        ($cond:expr) => {
            if !$cond {
                ::std::sys::FailWithCause::fail_with(
                    \"assertion failed: \" + stringify!($cond), file!(), line!())
            }
        };
        ($cond:expr, $msg:expr) => {
            if !$cond {
                ::std::sys::FailWithCause::fail_with($msg, file!(), line!())
            }
        };
        ($cond:expr, $( $arg:expr ),+) => {
            if !$cond {
                ::std::sys::FailWithCause::fail_with(fmt!( $($arg),+ ), file!(), line!())
            }
        }
    )

    macro_rules! assert_eq (
        ($given:expr , $expected:expr) => (
            {
                let given_val = $given;
                let expected_val = $expected;
                // check both directions of equality....
                if !((given_val == expected_val) && (expected_val == given_val)) {
                    fail!(\"assertion failed: `(left == right) && (right == \
                    left)` (left: `%?`, right: `%?`)\", given_val, expected_val);
                }
            }
        )
    )

    macro_rules! assert_approx_eq (
        ($given:expr , $expected:expr) => (
            {
                use std::cmp::ApproxEq;

                let given_val = $given;
                let expected_val = $expected;
                // check both directions of equality....
                if !(
                    given_val.approx_eq(&expected_val) &&
                    expected_val.approx_eq(&given_val)
                ) {
                    fail!(\"left: %? does not approximately equal right: %?\",
                          given_val, expected_val);
                }
            }
        );
        ($given:expr , $expected:expr , $epsilon:expr) => (
            {
                use std::cmp::ApproxEq;

                let given_val = $given;
                let expected_val = $expected;
                let epsilon_val = $epsilon;
                // check both directions of equality....
                if !(
                    given_val.approx_eq_eps(&expected_val, &epsilon_val) &&
                    expected_val.approx_eq_eps(&given_val, &epsilon_val)
                ) {
                    fail!(\"left: %? does not approximately equal right: %? with epsilon: %?\",
                          given_val, expected_val, epsilon_val);
                }
            }
        )
    )

    // FIXME(#6266): change the /* to /** when attributes are supported on macros
    // (Though even then—is it going to work according to the clear intent here?)
    /*
    A utility macro for indicating unreachable code. It will fail if
    executed. This is occasionally useful to put after loops that never
    terminate normally, but instead directly return from a function.

    # Example

    ```rust
    fn choose_weighted_item(v: &[Item]) -> Item {
        assert!(!v.is_empty());
        let mut so_far = 0u;
        for v.each |item| {
            so_far += item.weight;
            if so_far > 100 {
                return item;
            }
        }
        // The above loop always returns, so we must hint to the
        // type checker that it isn't possible to get down here
        unreachable!();
    }
    ```

    */
    macro_rules! unreachable (() => (
        fail!(\"internal error: entered unreachable code\");
    ))

    macro_rules! condition (

        { pub $c:ident: $input:ty -> $out:ty; } => {

            pub mod $c {
                #[allow(unused_imports)];
                #[allow(non_uppercase_statics)];

                use super::*;

                local_data_key!(key: @::std::condition::Handler<$input, $out>)

                pub static cond :
                    ::std::condition::Condition<$input,$out> =
                    ::std::condition::Condition {
                        name: stringify!($c),
                        key: key
                    };
            }
        };

        { $c:ident: $input:ty -> $out:ty; } => {

            mod $c {
                #[allow(unused_imports)];
                #[allow(non_uppercase_statics)];

                use super::*;

                local_data_key!(key: @::std::condition::Handler<$input, $out>)

                pub static cond :
                    ::std::condition::Condition<$input,$out> =
                    ::std::condition::Condition {
                        name: stringify!($c),
                        key: key
                    };
            }
        }
    )

    macro_rules! format(($($arg:tt)*) => (
        format_args!(::std::fmt::format, $($arg)*)
    ))
    macro_rules! write(($dst:expr, $($arg:tt)*) => (
        format_args!(|args| { ::std::fmt::write($dst, args) }, $($arg)*)
    ))
    macro_rules! writeln(($dst:expr, $($arg:tt)*) => (
        format_args!(|args| { ::std::fmt::writeln($dst, args) }, $($arg)*)
    ))
    // FIXME(#6846) once stdio is redesigned, this shouldn't perform an
    //              allocation but should rather delegate to an invocation of
    //              write! instead of format!
    macro_rules! print (
        ($($arg:tt)*) => (::std::io::print(format!($($arg)*)))
    )
    // FIXME(#6846) once stdio is redesigned, this shouldn't perform an
    //              allocation but should rather delegate to an io::Writer
    macro_rules! println (
        ($($arg:tt)*) => (::std::io::println(format!($($arg)*)))
    )

    macro_rules! local_data_key (
        ($name:ident: $ty:ty) => (
            static $name: ::std::local_data::Key<$ty> = &::std::local_data::Key;
        );
        (pub $name:ident: $ty:ty) => (
            pub static $name: ::std::local_data::Key<$ty> = &::std::local_data::Key;
        )
    )

    // externfn! declares a wrapper for an external function.
    // It is intended to be used like:
    //
    // externfn!(#[nolink]
    //           #[abi = \"cdecl\"]
    //           fn memcmp(cx: *u8, ct: *u8, n: u32) -> u32)
    //
    // Due to limitations in the macro parser, this pattern must be
    // implemented with 4 distinct patterns (with attrs / without
    // attrs CROSS with args / without ARGS).
    //
    // Also, this macro grammar allows for any number of return types
    // because I couldn't figure out the syntax to specify at most one.
    macro_rules! externfn(
        (fn $name:ident () $(-> $ret_ty:ty),*) => (
            pub unsafe fn $name() $(-> $ret_ty),* {
                // Note: to avoid obscure bug in macros, keep these
                // attributes *internal* to the fn
                #[fixed_stack_segment];
                #[inline(never)];
                #[allow(missing_doc)];

                return $name();

                extern {
                    fn $name() $(-> $ret_ty),*;
                }
            }
        );
        (fn $name:ident ($($arg_name:ident : $arg_ty:ty),*) $(-> $ret_ty:ty),*) => (
            pub unsafe fn $name($($arg_name : $arg_ty),*) $(-> $ret_ty),* {
                // Note: to avoid obscure bug in macros, keep these
                // attributes *internal* to the fn
                #[fixed_stack_segment];
                #[inline(never)];
                #[allow(missing_doc)];

                return $name($($arg_name),*);

                extern {
                    fn $name($($arg_name : $arg_ty),*) $(-> $ret_ty),*;
                }
            }
        );
        ($($attrs:attr)* fn $name:ident () $(-> $ret_ty:ty),*) => (
            pub unsafe fn $name() $(-> $ret_ty),* {
                // Note: to avoid obscure bug in macros, keep these
                // attributes *internal* to the fn
                #[fixed_stack_segment];
                #[inline(never)];
                #[allow(missing_doc)];

                return $name();

                $($attrs)*
                extern {
                    fn $name() $(-> $ret_ty),*;
                }
            }
        );
        ($($attrs:attr)* fn $name:ident ($($arg_name:ident : $arg_ty:ty),*) $(-> $ret_ty:ty),*) => (
            pub unsafe fn $name($($arg_name : $arg_ty),*) $(-> $ret_ty),* {
                // Note: to avoid obscure bug in macros, keep these
                // attributes *internal* to the fn
                #[fixed_stack_segment];
                #[inline(never)];
                #[allow(missing_doc)];

                return $name($($arg_name),*);

                $($attrs)*
                extern {
                    fn $name($($arg_name : $arg_ty),*) $(-> $ret_ty),*;
                }
            }
        )
    )

}";
}

struct Injector {
    sm: @ast::item,
}

impl ast_fold for Injector {
    fn fold_mod(&self, module: &ast::_mod) -> ast::_mod {
        // Just inject the standard macros at the start of the first module
        // in the crate: that is, at the start of the crate file itself.
        let items = vec::append(~[ self.sm ], module.items);
        ast::_mod {
            items: items,
            ..(*module).clone() // FIXME #2543: Bad copy.
        }
    }
}

// add a bunch of macros as though they were placed at the head of the
// program (ick). This should run before cfg stripping.
pub fn inject_std_macros(parse_sess: @mut parse::ParseSess,
                         cfg: ast::CrateConfig,
                         c: @Crate)
                         -> @Crate {
    let sm = match parse_item_from_source_str(@"<std-macros>",
                                              std_macros(),
                                              cfg.clone(),
                                              ~[],
                                              parse_sess) {
        Some(item) => item,
        None => fail!("expected core macros to parse correctly")
    };

    let injector = @Injector {
        sm: sm,
    } as @ast_fold;
    @injector.fold_crate(c)
}

struct NoOpFolder {
    contents: (),
}

impl ast_fold for NoOpFolder {}

struct MacroExpander {
    extsbox: @mut SyntaxEnv,
    cx: @ExtCtxt,
}

impl ast_fold for MacroExpander {
    fn fold_expr(&self, expr: @ast::Expr) -> @ast::Expr {
        expand_expr(self.extsbox,
                    self.cx,
                    expr,
                    self)
    }

    fn fold_mod(&self, module: &ast::_mod) -> ast::_mod {
        expand_mod_items(self.extsbox,
                         self.cx,
                         module,
                         self)
    }

    fn fold_item(&self, item: @ast::item) -> Option<@ast::item> {
        expand_item(self.extsbox,
                    self.cx,
                    item,
                    self)
    }

    fn fold_stmt(&self, stmt: &ast::Stmt) -> Option<@ast::Stmt> {
        expand_stmt(self.extsbox,
                    self.cx,
                    stmt,
                    self)
    }

    fn fold_block(&self, block: &ast::Block) -> ast::Block {
        expand_block(self.extsbox,
                     self.cx,
                     block,
                     self)
    }

    fn new_span(&self, span: Span) -> Span {
        new_span(self.cx, span)
    }
}

pub fn expand_crate(parse_sess: @mut parse::ParseSess,
                    cfg: ast::CrateConfig,
                    c: &Crate) -> @Crate {
    // adding *another* layer of indirection here so that the block
    // visitor can swap out one exts table for another for the duration
    // of the block.  The cleaner alternative would be to thread the
    // exts table through the fold, but that would require updating
    // every method/element of AstFoldFns in fold.rs.
    let extsbox = syntax_expander_table();
    let cx = ExtCtxt::new(parse_sess, cfg.clone());
    let expander = @MacroExpander {
        extsbox: @mut extsbox,
        cx: cx,
    } as @ast_fold;

    let ret = @expander.fold_crate(c);
    parse_sess.span_diagnostic.handler().abort_if_errors();
    return ret;
}

// HYGIENIC CONTEXT EXTENSION:
// all of these functions are for walking over
// ASTs and making some change to the context of every
// element that has one. a CtxtFn is a trait-ified
// version of a closure in (SyntaxContext -> SyntaxContext).
// the ones defined here include:
// Renamer - add a rename to a context
// MultiRenamer - add a set of renames to a context
// Marker - add a mark to a context
// Repainter - replace a context (maybe Replacer would be a better name?)

// a function in SyntaxContext -> SyntaxContext
pub trait CtxtFn{
    fn f(&self, ast::SyntaxContext) -> ast::SyntaxContext;
}

// a renamer adds a rename to the syntax context
pub struct Renamer {
    from : ast::Ident,
    to : ast::Name
}

impl CtxtFn for Renamer {
    fn f(&self, ctxt : ast::SyntaxContext) -> ast::SyntaxContext {
        new_rename(self.from,self.to,ctxt)
    }
}

// a renamer that performs a whole bunch of renames
pub struct MultiRenamer {
    renames : @mut ~[(ast::Ident,ast::Name)]
}

impl CtxtFn for MultiRenamer {
    fn f(&self, starting_ctxt : ast::SyntaxContext) -> ast::SyntaxContext {
        // the individual elements are memoized... it would
        // also be possible to memoize on the whole list at once.
        self.renames.iter().fold(starting_ctxt,|ctxt,&(from,to)| {
            new_rename(from,to,ctxt)
        })
    }
}

// a marker adds the given mark to the syntax context
pub struct Marker { mark : Mrk }

impl CtxtFn for Marker {
    fn f(&self, ctxt : ast::SyntaxContext) -> ast::SyntaxContext {
        new_mark(self.mark,ctxt)
    }
}

// a repainter just replaces the given context with the one it's closed over
pub struct Repainter { ctxt : SyntaxContext }

impl CtxtFn for Repainter {
    fn f(&self, _ctxt : ast::SyntaxContext) -> ast::SyntaxContext {
        self.ctxt
    }
}

pub struct ContextWrapper {
    context_function: @CtxtFn,
}

impl ast_fold for ContextWrapper {
    fn fold_ident(&self, id: ast::Ident) -> ast::Ident {
        let ast::Ident {
            name,
            ctxt
        } = id;
        ast::Ident {
            name: name,
            ctxt: self.context_function.f(ctxt),
        }
    }
    fn fold_mac(&self, m: &ast::mac) -> ast::mac {
        let macro = match m.node {
            mac_invoc_tt(ref path, ref tts, ctxt) => {
                mac_invoc_tt(self.fold_path(path),
                             fold_tts(*tts, self),
                             self.context_function.f(ctxt))
            }
        };
        Spanned {
            node: macro,
            span: m.span,
        }
    }
}

// given a function from ctxts to ctxts, produce
// an ast_fold that applies that function to all ctxts:
pub fn fun_to_ctxt_folder<T : 'static + CtxtFn>(cf: @T) -> @ContextWrapper {
    @ContextWrapper {
        context_function: cf as @CtxtFn,
    }
}

// just a convenience:
pub fn new_mark_folder(m: Mrk) -> @ContextWrapper {
    fun_to_ctxt_folder(@Marker{mark:m})
}

pub fn new_rename_folder(from: ast::Ident, to: ast::Name) -> @ContextWrapper {
    fun_to_ctxt_folder(@Renamer{from:from,to:to})
}

// apply a given mark to the given token trees. Used prior to expansion of a macro.
fn mark_tts(tts : &[token_tree], m : Mrk) -> ~[token_tree] {
    fold_tts(tts,new_mark_folder(m))
}

// apply a given mark to the given expr. Used following the expansion of a macro.
fn mark_expr(expr : @ast::Expr, m : Mrk) -> @ast::Expr {
    new_mark_folder(m).fold_expr(expr)
}

// apply a given mark to the given stmt. Used following the expansion of a macro.
fn mark_stmt(expr : &ast::Stmt, m : Mrk) -> @ast::Stmt {
    new_mark_folder(m).fold_stmt(expr).unwrap()
}

// apply a given mark to the given item. Used following the expansion of a macro.
fn mark_item(expr : @ast::item, m : Mrk) -> Option<@ast::item> {
    new_mark_folder(m).fold_item(expr)
}

// replace all contexts in a given expr with the given mark. Used
// for capturing macros
pub fn replace_ctxts(expr : @ast::Expr, ctxt : SyntaxContext) -> @ast::Expr {
    fun_to_ctxt_folder(@Repainter{ctxt:ctxt}).fold_expr(expr)
}

// take the mark from the given ctxt (that has a mark at the outside),
// and apply it to everything in the token trees, thereby cancelling
// that mark.
pub fn mtwt_cancel_outer_mark(tts: &[ast::token_tree], ctxt: ast::SyntaxContext)
    -> ~[ast::token_tree] {
    let outer_mark = mtwt_outer_mark(ctxt);
    mark_tts(tts,outer_mark)
}

fn original_span(cx: @ExtCtxt) -> @codemap::ExpnInfo {
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
    use ast::{Attribute_, AttrOuter, MetaWord, EMPTY_CTXT};
    use ast_util::{get_sctable, mtwt_marksof, mtwt_resolve, new_rename};
    use ast_util;
    use codemap;
    use codemap::Spanned;
    use fold;
    use parse;
    use parse::token::{fresh_mark, gensym, intern, get_ident_interner, ident_to_str};
    use parse::token;
    use print::pprust;
    use std;
    use util::parser_testing::{string_to_crate, string_to_crate_and_sess};
    use util::parser_testing::{string_to_pat, string_to_tts, strs_to_idents};
    use visit;

    // make sure that fail! is present
    #[test] fn fail_exists_test () {
        let src = @"fn main() { fail!(\"something appropriately gloomy\");}";
        let sess = parse::new_parse_sess(None);
        let crate_ast = parse::parse_crate_from_source_str(
            @"<test>",
            src,
            ~[],sess);
        let crate_ast = inject_std_macros(sess, ~[], crate_ast);
        // don't bother with striping, doesn't affect fail!.
        expand_crate(sess,~[],crate_ast);
    }

    // these following tests are quite fragile, in that they don't test what
    // *kind* of failure occurs.

    // make sure that macros can leave scope
    #[should_fail]
    #[test] fn macros_cant_escape_fns_test () {
        let src = @"fn bogus() {macro_rules! z (() => (3+4))}\
                    fn inty() -> int { z!() }";
        let sess = parse::new_parse_sess(None);
        let crate_ast = parse::parse_crate_from_source_str(
            @"<test>",
            src,
            ~[],sess);
        // should fail:
        expand_crate(sess,~[],crate_ast);
    }

    // make sure that macros can leave scope for modules
    #[should_fail]
    #[test] fn macros_cant_escape_mods_test () {
        let src = @"mod foo {macro_rules! z (() => (3+4))}\
                    fn inty() -> int { z!() }";
        let sess = parse::new_parse_sess(None);
        let crate_ast = parse::parse_crate_from_source_str(
            @"<test>",
            src,
            ~[],sess);
        // should fail:
        expand_crate(sess,~[],crate_ast);
    }

    // macro_escape modules shouldn't cause macros to leave scope
    #[test] fn macros_can_escape_flattened_mods_test () {
        let src = @"#[macro_escape] mod foo {macro_rules! z (() => (3+4))}\
                    fn inty() -> int { z!() }";
        let sess = parse::new_parse_sess(None);
        let crate_ast = parse::parse_crate_from_source_str(
            @"<test>",
            src,
            ~[], sess);
        // should fail:
        expand_crate(sess,~[],crate_ast);
    }

    #[test] fn std_macros_must_parse () {
        let src = super::std_macros();
        let sess = parse::new_parse_sess(None);
        let cfg = ~[];
        let item_ast = parse::parse_item_from_source_str(
            @"<test>",
            src,
            cfg,~[],sess);
        match item_ast {
            Some(_) => (), // success
            None => fail!("expected this to parse")
        }
    }

    #[test] fn test_contains_flatten (){
        let attr1 = make_dummy_attr (@"foo");
        let attr2 = make_dummy_attr (@"bar");
        let escape_attr = make_dummy_attr (@"macro_escape");
        let attrs1 = ~[attr1, escape_attr, attr2];
        assert_eq!(contains_macro_escape (attrs1),true);
        let attrs2 = ~[attr1,attr2];
        assert_eq!(contains_macro_escape (attrs2),false);
    }

    // make a MetaWord outer attribute with the given name
    fn make_dummy_attr(s: @str) -> ast::Attribute {
        Spanned {
            span:codemap::dummy_sp(),
            node: Attribute_ {
                style: AttrOuter,
                value: @Spanned {
                    node: MetaWord(s),
                    span: codemap::dummy_sp(),
                },
                is_sugared_doc: false,
            }
        }
    }

    #[test] fn cancel_outer_mark_test(){
        let invalid_name = token::special_idents::invalid.name;
        let ident_str = @"x";
        let tts = string_to_tts(ident_str);
        let fm = fresh_mark();
        let marked_once = fold::fold_tts(tts,new_mark_folder(fm));
        assert_eq!(marked_once.len(),1);
        let marked_once_ctxt =
            match marked_once[0] {
                ast::tt_tok(_,token::IDENT(id,_)) => id.ctxt,
                _ => fail!(fmt!("unexpected shape for marked tts: %?",marked_once[0]))
            };
        assert_eq!(mtwt_marksof(marked_once_ctxt,invalid_name),~[fm]);
        let remarked = mtwt_cancel_outer_mark(marked_once,marked_once_ctxt);
        assert_eq!(remarked.len(),1);
        match remarked[0] {
            ast::tt_tok(_,token::IDENT(id,_)) =>
            assert_eq!(mtwt_marksof(id.ctxt,invalid_name),~[]),
            _ => fail!(fmt!("unexpected shape for marked tts: %?",remarked[0]))
        }
    }

    #[test]
    fn renaming () {
        let item_ast = string_to_crate(@"fn f() -> int { a }");
        let a_name = intern("a");
        let a2_name = gensym("a2");
        let renamer = new_rename_folder(ast::Ident{name:a_name,ctxt:EMPTY_CTXT},
                                        a2_name);
        let renamed_ast = renamer.fold_crate(item_ast);
        let varrefs = @mut ~[];
        visit::walk_crate(&mut new_path_finder(varrefs), &renamed_ast, ());
        match varrefs {
            @[Path{segments:[ref seg],_}] => assert_eq!(mtwt_resolve(seg.identifier),a2_name),
            _ => assert_eq!(0,1)
        }

        // try a double-rename, with pending_renames.
        let a3_name = gensym("a3");
        // a context that renames from ("a",empty) to "a2" :
        let ctxt2 = new_rename(ast::Ident::new(a_name),a2_name,EMPTY_CTXT);
        let pending_renames = @mut ~[(ast::Ident::new(a_name),a2_name),
                                     (ast::Ident{name:a_name,ctxt:ctxt2},a3_name)];
        let double_renamed = renames_to_fold(pending_renames).fold_crate(item_ast);
        let varrefs = @mut ~[];
        visit::walk_crate(&mut new_path_finder(varrefs), &double_renamed, ());
        match varrefs {
            @[Path{segments:[ref seg],_}] => assert_eq!(mtwt_resolve(seg.identifier),a3_name),
            _ => assert_eq!(0,1)
        }
    }

    fn fake_print_crate(crate: @ast::Crate) {
        let s = pprust::rust_printer(std::io::stderr(),get_ident_interner());
        pprust::print_crate_(s, crate);
    }

    fn expand_crate_str(crate_str: @str) -> @ast::Crate {
        let (crate_ast,ps) = string_to_crate_and_sess(crate_str);
        // the cfg argument actually does matter, here...
        expand_crate(ps,~[],crate_ast)
    }

    //fn expand_and_resolve(crate_str: @str) -> ast::crate {
        //let expanded_ast = expand_crate_str(crate_str);
        // std::io::println(fmt!("expanded: %?\n",expanded_ast));
        //mtwt_resolve_crate(expanded_ast)
    //}
    //fn expand_and_resolve_and_pretty_print (crate_str : @str) -> ~str {
        //let resolved_ast = expand_and_resolve(crate_str);
        //pprust::to_str(&resolved_ast,fake_print_crate,get_ident_interner())
    //}

    #[test] fn macro_tokens_should_match(){
        expand_crate_str(@"macro_rules! m((a)=>(13)) fn main(){m!(a);}");
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
    type renaming_test = (&'static str, ~[~[uint]], bool);

    #[test]
    fn automatic_renaming () {
        let tests : ~[renaming_test] =
            ~[// b & c should get new names throughout, in the expr too:
                ("fn a() -> int { let b = 13; let c = b; b+c }",
                 ~[~[0,1],~[2]], false),
                // both x's should be renamed (how is this causing a bug?)
                ("fn main () {let x : int = 13;x;}",
                 ~[~[0]], false),
                // the use of b after the + should be renamed, the other one not:
                ("macro_rules! f (($x:ident) => (b + $x)) fn a() -> int { let b = 13; f!(b)}",
                 ~[~[1]], false),
                // the b before the plus should not be renamed (requires marks)
                ("macro_rules! f (($x:ident) => ({let b=9; ($x + b)})) fn a() -> int { f!(b)}",
                 ~[~[1]], false),
                // the marks going in and out of letty should cancel, allowing that $x to
                // capture the one following the semicolon.
                // this was an awesome test case, and caught a *lot* of bugs.
                ("macro_rules! letty(($x:ident) => (let $x = 15;))
                  macro_rules! user(($x:ident) => ({letty!($x); $x}))
                  fn main() -> int {user!(z)}",
                 ~[~[0]], false),
                // no longer a fixme #8062: this test exposes a *potential* bug; our system does
                // not behave exactly like MTWT, but a conversation with Matthew Flatt
                // suggests that this can only occur in the presence of local-expand, which
                // we have no plans to support.
                // ("fn main() {let hrcoo = 19; macro_rules! getx(()=>(hrcoo)); getx!();}",
                // ~[~[0]], true)
                // FIXME #6994: the next string exposes the bug referred to in issue 6994, so I'm
                // commenting it out.
                // the z flows into and out of two macros (g & f) along one path, and one
                // (just g) along the other, so the result of the whole thing should
                // be "let z_123 = 3; z_123"
                //"macro_rules! g (($x:ident) =>
                //   ({macro_rules! f(($y:ident)=>({let $y=3;$x}));f!($x)}))
                //   fn a(){g!(z)}"
                // create a really evil test case where a $x appears inside a binding of $x
                // but *shouldnt* bind because it was inserted by a different macro....
                // can't write this test case until we have macro-generating macros.
            ];
        for (idx,s) in tests.iter().enumerate() {
            run_renaming_test(s,idx);
        }
    }

    // run one of the renaming tests
    fn run_renaming_test(t : &renaming_test, test_idx: uint) {
        let invalid_name = token::special_idents::invalid.name;
        let (teststr, bound_connections, bound_ident_check) = match *t {
            (ref str,ref conns, bic) => (str.to_managed(), conns.clone(), bic)
        };
        let cr = expand_crate_str(teststr.to_managed());
        // find the bindings:
        let bindings = @mut ~[];
        visit::walk_crate(&mut new_name_finder(bindings),cr,());
        // find the varrefs:
        let varrefs = @mut ~[];
        visit::walk_crate(&mut new_path_finder(varrefs),cr,());
        // must be one check clause for each binding:
        assert_eq!(bindings.len(),bound_connections.len());
        for (binding_idx,shouldmatch) in bound_connections.iter().enumerate() {
            let binding_name = mtwt_resolve(bindings[binding_idx]);
            let binding_marks = mtwt_marksof(bindings[binding_idx].ctxt,invalid_name);
            // shouldmatch can't name varrefs that don't exist:
            assert!((shouldmatch.len() == 0) ||
                    (varrefs.len() > *shouldmatch.iter().max().unwrap()));
            for (idx,varref) in varrefs.iter().enumerate() {
                if shouldmatch.contains(&idx) {
                    // it should be a path of length 1, and it should
                    // be free-identifier=? or bound-identifier=? to the given binding
                    assert_eq!(varref.segments.len(),1);
                    let varref_name = mtwt_resolve(varref.segments[0].identifier);
                    let varref_marks = mtwt_marksof(varref.segments[0].identifier.ctxt,
                                                    invalid_name);
                    if (!(varref_name==binding_name)){
                        std::io::println("uh oh, should match but doesn't:");
                        std::io::println(fmt!("varref: %?",varref));
                        std::io::println(fmt!("binding: %?", bindings[binding_idx]));
                        ast_util::display_sctable(get_sctable());
                    }
                    assert_eq!(varref_name,binding_name);
                    if (bound_ident_check) {
                        // we're checking bound-identifier=?, and the marks
                        // should be the same, too:
                        assert_eq!(varref_marks,binding_marks.clone());
                    }
                } else {
                    let fail = (varref.segments.len() == 1)
                        && (mtwt_resolve(varref.segments[0].identifier) == binding_name);
                    // temp debugging:
                    if (fail) {
                        println!("failure on test {}",test_idx);
                        println!("text of test case: \"{}\"", teststr);
                        println!("");
                        println!("uh oh, matches but shouldn't:");
                        std::io::println(fmt!("varref: %?",varref));
                        // good lord, you can't make a path with 0 segments, can you?
                        println!("varref's first segment's uint: {}, and string: \"{}\"",
                                 varref.segments[0].identifier.name,
                                 ident_to_str(&varref.segments[0].identifier));
                        std::io::println(fmt!("binding: %?", bindings[binding_idx]));
                        ast_util::display_sctable(get_sctable());
                    }
                    assert!(!fail);
                }
            }
        }
    }

    #[test] fn fmt_in_macro_used_inside_module_macro() {
        let crate_str = @"macro_rules! fmt_wrap(($b:expr)=>(fmt!(\"left: %?\", $b)))
macro_rules! foo_module (() => (mod generated { fn a() { let xx = 147; fmt_wrap!(xx);}}))
foo_module!()
";
        let cr = expand_crate_str(crate_str);
        // find the xx binding
        let bindings = @mut ~[];
        visit::walk_crate(&mut new_name_finder(bindings), cr, ());
        let cxbinds : ~[&ast::Ident] =
            bindings.iter().filter(|b|{@"xx" == (ident_to_str(*b))}).collect();
        let cxbind = match cxbinds {
            [b] => b,
            _ => fail!("expected just one binding for ext_cx")
        };
        let resolved_binding = mtwt_resolve(*cxbind);
        // find all the xx varrefs:
        let varrefs = @mut ~[];
        visit::walk_crate(&mut new_path_finder(varrefs), cr, ());
        // the xx binding should bind all of the xx varrefs:
        for (idx,v) in varrefs.iter().filter(|p|{ p.segments.len() == 1
                                          && (@"xx" == (ident_to_str(&p.segments[0].identifier)))
                                     }).enumerate() {
            if (mtwt_resolve(v.segments[0].identifier) != resolved_binding) {
                std::io::println("uh oh, xx binding didn't match xx varref:");
                std::io::println(fmt!("this is xx varref # %?",idx));
                std::io::println(fmt!("binding: %?",cxbind));
                std::io::println(fmt!("resolves to: %?",resolved_binding));
                std::io::println(fmt!("varref: %?",v.segments[0].identifier));
                std::io::println(fmt!("resolves to: %?",mtwt_resolve(v.segments[0].identifier)));
                let table = get_sctable();
                std::io::println("SC table:");
                for (idx,val) in table.table.iter().enumerate() {
                    std::io::println(fmt!("%4u : %?",idx,val));
                }
            }
            assert_eq!(mtwt_resolve(v.segments[0].identifier),resolved_binding);
        };
    }

    #[test]
    fn pat_idents(){
        let pat = string_to_pat(@"(a,Foo{x:c @ (b,9),y:Bar(4,d)})");
        let idents = @mut ~[];
        let pat_idents = new_name_finder(idents);
        pat_idents.visit_pat(pat, ());
        assert_eq!(idents, @mut strs_to_idents(~["a","c","b","d"]));
    }

}
