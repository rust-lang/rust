// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{Block, Crate, NodeId, expr_, expr_mac, ident, mac_invoc_tt};
use ast::{item_mac, stmt_, stmt_mac, stmt_expr, stmt_semi};
use ast::{illegal_ctxt};
use ast;
use ast_util::{new_rename, new_mark, resolve};
use attr;
use attr::AttrMetaMethods;
use codemap;
use codemap::{span, spanned, ExpnInfo, NameAndSpan};
use ext::base::*;
use fold::*;
use parse;
use parse::{parse_item_from_source_str};
use parse::token;
use parse::token::{ident_to_str, intern};
use visit;
use visit::Visitor;

use std::vec;

pub fn expand_expr(extsbox: @mut SyntaxEnv,
                   cx: @ExtCtxt,
                   e: &expr_,
                   s: span,
                   fld: @ast_fold,
                   orig: @fn(&expr_, span, @ast_fold) -> (expr_, span))
                -> (expr_, span) {
    match *e {
        // expr_mac should really be expr_ext or something; it's the
        // entry-point for all syntax extensions.
        expr_mac(ref mac) => {
            match (*mac).node {
                // Token-tree macros:
                mac_invoc_tt(ref pth, ref tts) => {
                    if (pth.idents.len() > 1u) {
                        cx.span_fatal(
                            pth.span,
                            fmt!("expected macro name without module \
                                  separators"));
                    }
                    let extname = &pth.idents[0];
                    let extnamestr = ident_to_str(extname);
                    // leaving explicit deref here to highlight unbox op:
                    match (*extsbox).find(&extname.name) {
                        None => {
                            cx.span_fatal(
                                pth.span,
                                fmt!("macro undefined: '%s'", extnamestr))
                        }
                        Some(@SE(NormalTT(SyntaxExpanderTT{
                            expander: exp,
                            span: exp_sp
                        }))) => {
                            cx.bt_push(ExpnInfo {
                                call_site: s,
                                callee: NameAndSpan {
                                    name: extnamestr,
                                    span: exp_sp,
                                },
                            });

                            let expanded = match exp(cx, mac.span, *tts) {
                                MRExpr(e) => e,
                                MRAny(expr_maker,_,_) => expr_maker(),
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

                            //keep going, outside-in
                            let fully_expanded =
                                fld.fold_expr(expanded).node.clone();
                            cx.bt_pop();

                            (fully_expanded, s)
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
        // From: `for <src_pat> in <src_expr> <src_loop_block>`
        ast::expr_for_loop(src_pat, src_expr, ref src_loop_block) => {
            let src_pat = src_pat.clone();
            let src_expr = src_expr.clone();

            // Expand any interior macros etc.
            // NB: we don't fold pats yet. Curious.
            let src_expr = fld.fold_expr(src_expr).clone();
            let src_loop_block = fld.fold_block(src_loop_block).clone();

            let span = s;
            let lo = s.lo;
            let hi = s.hi;

            pub fn mk_expr(cx: @ExtCtxt, span: span,
                           node: expr_) -> @ast::expr {
                @ast::expr {
                    id: cx.next_id(),
                    node: node,
                    span: span,
                }
            }

            fn mk_block(cx: @ExtCtxt,
                        stmts: &[@ast::stmt],
                        expr: Option<@ast::expr>,
                        span: span) -> ast::Block {
                ast::Block {
                    view_items: ~[],
                    stmts: stmts.to_owned(),
                    expr: expr,
                    id: cx.next_id(),
                    rules: ast::DefaultBlock,
                    span: span,
                }
            }

            fn mk_simple_path(ident: ast::ident, span: span) -> ast::Path {
                ast::Path {
                    span: span,
                    global: false,
                    idents: ~[ident],
                    rp: None,
                    types: ~[]
                }
            }

            // to:
            //
            // {
            //   let _i = &mut <src_expr>;
            //   loop {
            //       match i.next() {
            //           None => break,
            //           Some(<src_pat>) => <src_loop_block>
            //       }
            //   }
            // }

            let local_ident = token::gensym_ident("i");
            let some_ident = token::str_to_ident("Some");
            let none_ident = token::str_to_ident("None");
            let next_ident = token::str_to_ident("next");

            let local_path_1 = mk_simple_path(local_ident, span);
            let local_path_2 = mk_simple_path(local_ident, span);
            let some_path = mk_simple_path(some_ident, span);
            let none_path = mk_simple_path(none_ident, span);

            // `let i = &mut <src_expr>`
            let iter_decl_stmt = {
                let ty = ast::Ty {
                    id: cx.next_id(),
                    node: ast::ty_infer,
                    span: span
                };
                let local = @ast::Local {
                    is_mutbl: false,
                    ty: ty,
                    pat: @ast::pat {
                        id: cx.next_id(),
                        node: ast::pat_ident(ast::bind_infer, local_path_1, None),
                        span: src_expr.span
                    },
                    init: Some(mk_expr(cx, src_expr.span,
                                       ast::expr_addr_of(ast::m_mutbl, src_expr))),
                    id: cx.next_id(),
                    span: src_expr.span,
                };
                let e = @spanned(src_expr.span.lo,
                                 src_expr.span.hi,
                                 ast::decl_local(local));
                @spanned(lo, hi, ast::stmt_decl(e, cx.next_id()))
            };

            // `None => break;`
            let none_arm = {
                let break_expr = mk_expr(cx, span, ast::expr_break(None));
                let break_stmt = @spanned(lo, hi, ast::stmt_expr(break_expr, cx.next_id()));
                let none_block = mk_block(cx, [break_stmt], None, span);
                let none_pat = @ast::pat {
                    id: cx.next_id(),
                    node: ast::pat_ident(ast::bind_infer, none_path, None),
                    span: span
                };
                ast::arm {
                    pats: ~[none_pat],
                    guard: None,
                    body: none_block
                }
            };

            // `Some(<src_pat>) => <src_loop_block>`
            let some_arm = {
                let pat = @ast::pat {
                    id: cx.next_id(),
                    node: ast::pat_enum(some_path, Some(~[src_pat])),
                    span: src_pat.span
                };
                ast::arm {
                    pats: ~[pat],
                    guard: None,
                    body: src_loop_block
                }
            };

            // `match i.next() { ... }`
            let match_stmt = {
                let local_expr = mk_expr(cx, span, ast::expr_path(local_path_2));
                let next_call_expr = mk_expr(cx, span,
                                             ast::expr_method_call(cx.next_id(),
                                                                   local_expr, next_ident,
                                                                   ~[], ~[], ast::NoSugar));
                let match_expr = mk_expr(cx, span, ast::expr_match(next_call_expr,
                                                                   ~[none_arm, some_arm]));
                @spanned(lo, hi, ast::stmt_expr(match_expr, cx.next_id()))
            };

            // `loop { ... }`
            let loop_block = {
                let loop_body_block = mk_block(cx, [match_stmt], None, span);
                let loop_body_expr = mk_expr(cx, span, ast::expr_loop(loop_body_block, None));
                let loop_body_stmt = @spanned(lo, hi, ast::stmt_expr(loop_body_expr, cx.next_id()));
                mk_block(cx, [iter_decl_stmt,
                              loop_body_stmt],
                         None, span)
            };

            (ast::expr_block(loop_block), span)
        }

        _ => orig(e, s, fld)
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
                        fld: @ast_fold,
                        orig: @fn(&ast::_mod, @ast_fold) -> ast::_mod)
                     -> ast::_mod {

    // Fold the contents first:
    let module_ = orig(module_, fld);

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

    ast::_mod { items: new_items, ..module_ }
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
                   fld: @ast_fold,
                   orig: @fn(@ast::item, @ast_fold) -> Option<@ast::item>)
                -> Option<@ast::item> {
    // need to do expansion first... it might turn out to be a module.
    let maybe_it = match it.node {
      ast::item_mac(*) => expand_item_mac(extsbox, cx, it, fld),
      _ => Some(it)
    };
    match maybe_it {
      Some(it) => {
          match it.node {
              ast::item_mod(_) | ast::item_foreign_mod(_) => {
                  cx.mod_push(it.ident);
                  let macro_escape = contains_macro_escape(it.attrs);
                  let result = with_exts_frame!(extsbox,macro_escape,orig(it,fld));
                  cx.mod_pop();
                  result
              }
              _ => orig(it,fld)
          }
      }
      None => None
    }
}

// does this attribute list contain "macro_escape" ?
pub fn contains_macro_escape(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, "macro_escape")
}

// Support for item-position macro invocations, exactly the same
// logic as for expression-position macro invocations.
pub fn expand_item_mac(extsbox: @mut SyntaxEnv,
                       cx: @ExtCtxt, it: @ast::item,
                       fld: @ast_fold)
                    -> Option<@ast::item> {
    let (pth, tts) = match it.node {
        item_mac(codemap::spanned { node: mac_invoc_tt(ref pth, ref tts), _}) => {
            (pth, (*tts).clone())
        }
        _ => cx.span_bug(it.span, "invalid item macro invocation")
    };

    let extname = &pth.idents[0];
    let extnamestr = ident_to_str(extname);
    let expanded = match (*extsbox).find(&extname.name) {
        None => cx.span_fatal(pth.span,
                              fmt!("macro undefined: '%s!'", extnamestr)),

        Some(@SE(NormalTT(ref expand))) => {
            if it.ident != parse::token::special_idents::invalid {
                cx.span_fatal(pth.span,
                              fmt!("macro %s! expects no ident argument, \
                                    given '%s'", extnamestr,
                                   ident_to_str(&it.ident)));
            }
            cx.bt_push(ExpnInfo {
                call_site: it.span,
                callee: NameAndSpan {
                    name: extnamestr,
                    span: expand.span
                }
            });
            ((*expand).expander)(cx, it.span, tts)
        }
        Some(@SE(IdentTT(ref expand))) => {
            if it.ident == parse::token::special_idents::invalid {
                cx.span_fatal(pth.span,
                              fmt!("macro %s! expects an ident argument",
                                   extnamestr));
            }
            cx.bt_push(ExpnInfo {
                call_site: it.span,
                callee: NameAndSpan {
                    name: extnamestr,
                    span: expand.span
                }
            });
            ((*expand).expander)(cx, it.span, it.ident, tts)
        }
        _ => cx.span_fatal(
            it.span, fmt!("%s! is not legal in item position", extnamestr))
    };

    let maybe_it = match expanded {
        MRItem(it) => fld.fold_item(it),
        MRExpr(_) => cx.span_fatal(pth.span,
                                   fmt!("expr macro in item position: %s", extnamestr)),
        MRAny(_, item_maker, _) => item_maker().chain(|i| {fld.fold_item(i)}),
        MRDef(ref mdef) => {
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
                   s: &stmt_,
                   sp: span,
                   fld: @ast_fold,
                   orig: @fn(&stmt_, span, @ast_fold)
                             -> (Option<stmt_>, span))
                -> (Option<stmt_>, span) {
    let (mac, pth, tts, semi) = match *s {
        stmt_mac(ref mac, semi) => {
            match mac.node {
                mac_invoc_tt(ref pth, ref tts) => {
                    ((*mac).clone(), pth, (*tts).clone(), semi)
                }
            }
        }
        _ => return orig(s, sp, fld)
    };
    if (pth.idents.len() > 1u) {
        cx.span_fatal(
            pth.span,
            fmt!("expected macro name without module \
                  separators"));
    }
    let extname = &pth.idents[0];
    let extnamestr = ident_to_str(extname);
    let (fully_expanded, sp) = match (*extsbox).find(&extname.name) {
        None =>
            cx.span_fatal(pth.span, fmt!("macro undefined: '%s'", extnamestr)),

        Some(@SE(NormalTT(
            SyntaxExpanderTT{expander: exp, span: exp_sp}))) => {
            cx.bt_push(ExpnInfo {
                call_site: sp,
                callee: NameAndSpan { name: extnamestr, span: exp_sp }
            });
            let expanded = match exp(cx, mac.span, tts) {
                MRExpr(e) =>
                    @codemap::spanned { node: stmt_expr(e, cx.next_id()),
                                    span: e.span},
                MRAny(_,_,stmt_mkr) => stmt_mkr(),
                _ => cx.span_fatal(
                    pth.span,
                    fmt!("non-stmt macro in stmt pos: %s", extnamestr))
            };

            //keep going, outside-in
            let fully_expanded = match fld.fold_stmt(expanded) {
                Some(stmt) => {
                    let fully_expanded = &stmt.node;
                    cx.bt_pop();
                    (*fully_expanded).clone()
                }
                None => {
                    cx.span_fatal(pth.span,
                                  "macro didn't expand to a statement")
                }
            };

            (fully_expanded, sp)
        }

        _ => {
            cx.span_fatal(pth.span,
                          fmt!("'%s' is not a tt-style macro", extnamestr))
        }
    };

    (match fully_expanded {
        stmt_expr(e, stmt_id) if semi => Some(stmt_semi(e, stmt_id)),
        _ => { Some(fully_expanded) } /* might already have a semi */
    }, sp)

}

#[deriving(Clone)]
struct NewNameFinderContext {
    ident_accumulator: @mut ~[ast::ident],
}

impl Visitor<()> for NewNameFinderContext {
    fn visit_pat(&mut self, pattern: @ast::pat, _: ()) {
        match *pattern {
            // we found a pat_ident!
            ast::pat {
                id: _,
                node: ast::pat_ident(_, ref path, ref inner),
                span: _
            } => {
                match path {
                    // a path of length one:
                    &ast::Path {
                        global: false,
                        idents: [id],
                        span: _,
                        rp: _,
                        types: _
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

    // XXX: Methods below can become default methods.

    fn visit_mod(&mut self, module: &ast::_mod, _: span, _: NodeId, _: ()) {
        visit::walk_mod(self, module, ())
    }

    fn visit_view_item(&mut self, view_item: &ast::view_item, _: ()) {
        visit::walk_view_item(self, view_item, ())
    }

    fn visit_item(&mut self, item: @ast::item, _: ()) {
        visit::walk_item(self, item, ())
    }

    fn visit_foreign_item(&mut self,
                          foreign_item: @ast::foreign_item,
                          _: ()) {
        visit::walk_foreign_item(self, foreign_item, ())
    }

    fn visit_local(&mut self, local: @ast::Local, _: ()) {
        visit::walk_local(self, local, ())
    }

    fn visit_block(&mut self, block: &ast::Block, _: ()) {
        visit::walk_block(self, block, ())
    }

    fn visit_stmt(&mut self, stmt: @ast::stmt, _: ()) {
        visit::walk_stmt(self, stmt, ())
    }

    fn visit_arm(&mut self, arm: &ast::arm, _: ()) {
        visit::walk_arm(self, arm, ())
    }

    fn visit_decl(&mut self, decl: @ast::decl, _: ()) {
        visit::walk_decl(self, decl, ())
    }

    fn visit_expr(&mut self, expr: @ast::expr, _: ()) {
        visit::walk_expr(self, expr, ())
    }

    fn visit_expr_post(&mut self, _: @ast::expr, _: ()) {
        // Empty!
    }

    fn visit_ty(&mut self, typ: &ast::Ty, _: ()) {
        visit::walk_ty(self, typ, ())
    }

    fn visit_generics(&mut self, generics: &ast::Generics, _: ()) {
        visit::walk_generics(self, generics, ())
    }

    fn visit_fn(&mut self,
                function_kind: &visit::fn_kind,
                function_declaration: &ast::fn_decl,
                block: &ast::Block,
                span: span,
                node_id: NodeId,
                _: ()) {
        visit::walk_fn(self,
                        function_kind,
                        function_declaration,
                        block,
                        span,
                        node_id,
                        ())
    }

    fn visit_ty_method(&mut self, ty_method: &ast::TypeMethod, _: ()) {
        visit::walk_ty_method(self, ty_method, ())
    }

    fn visit_trait_method(&mut self,
                          trait_method: &ast::trait_method,
                          _: ()) {
        visit::walk_trait_method(self, trait_method, ())
    }

    fn visit_struct_def(&mut self,
                        struct_def: @ast::struct_def,
                        ident: ident,
                        generics: &ast::Generics,
                        node_id: NodeId,
                        _: ()) {
        visit::walk_struct_def(self,
                                struct_def,
                                ident,
                                generics,
                                node_id,
                                ())
    }

    fn visit_struct_field(&mut self,
                          struct_field: @ast::struct_field,
                          _: ()) {
        visit::walk_struct_field(self, struct_field, ())
    }
}

// return a visitor that extracts the pat_ident paths
// from a given pattern and puts them in a mutable
// array (passed in to the traversal)
pub fn new_name_finder(idents: @mut ~[ast::ident]) -> @mut Visitor<()> {
    let context = @mut NewNameFinderContext {
        ident_accumulator: idents,
    };
    context as @mut Visitor<()>
}

pub fn expand_block(extsbox: @mut SyntaxEnv,
                    _cx: @ExtCtxt,
                    blk: &Block,
                    fld: @ast_fold,
                    orig: @fn(&Block, @ast_fold) -> Block)
                 -> Block {
    // see note below about treatment of exts table
    with_exts_frame!(extsbox,false,orig(blk,fld))
}


// get the (innermost) BlockInfo from an exts stack
fn get_block_info(exts : SyntaxEnv) -> BlockInfo {
    match exts.find_in_topmost_frame(&intern(special_block_name)) {
        Some(@BlockInfo(bi)) => bi,
        _ => fail!(fmt!("special identifier %? was bound to a non-BlockInfo",
                       @" block"))
    }
}


// given a mutable list of renames, return a tree-folder that applies those
// renames.
fn renames_to_fold(renames : @mut ~[(ast::ident,ast::Name)]) -> @ast_fold {
    let afp = default_ast_fold();
    let f_pre = @AstFoldFns {
        fold_ident: |id,_| {
            // the individual elements are memoized... it would
            // also be possible to memoize on the whole list at once.
            let new_ctxt = renames.iter().fold(id.ctxt,|ctxt,&(from,to)| {
                new_rename(from,to,ctxt)
            });
            ast::ident{name:id.name,ctxt:new_ctxt}
        },
        .. *afp
    };
    make_fold(f_pre)
}

// perform a bunch of renames
fn apply_pending_renames(folder : @ast_fold, stmt : ast::stmt) -> @ast::stmt {
    match folder.fold_stmt(&stmt) {
        Some(s) => s,
        None => fail!(fmt!("renaming of stmt produced None"))
    }
}



pub fn new_span(cx: @ExtCtxt, sp: span) -> span {
    /* this discards information in the case of macro-defining macros */
    return span {lo: sp.lo, hi: sp.hi, expn_info: cx.backtrace()};
}

// FIXME (#2247): this is a moderately bad kludge to inject some macros into
// the default compilation environment. It would be much nicer to use
// a mechanism like syntax_quote to ensure hygiene.

pub fn std_macros() -> @str {
    return
@"mod __std_macros {
    #[macro_escape];
    #[doc(hidden)];

    macro_rules! ignore (($($x:tt)*) => (()))

    macro_rules! error (
        ($arg:expr) => (
            __log(1u32, fmt!( \"%?\", $arg ))
        );
        ($( $arg:expr ),+) => (
            __log(1u32, fmt!( $($arg),+ ))
        )
    )

    macro_rules! warn (
        ($arg:expr) => (
            __log(2u32, fmt!( \"%?\", $arg ))
        );
        ($( $arg:expr ),+) => (
            __log(2u32, fmt!( $($arg),+ ))
        )
    )

    macro_rules! info (
        ($arg:expr) => (
            __log(3u32, fmt!( \"%?\", $arg ))
        );
        ($( $arg:expr ),+) => (
            __log(3u32, fmt!( $($arg),+ ))
        )
    )

    macro_rules! debug (
        ($arg:expr) => (
            if cfg!(debug) { __log(4u32, fmt!( \"%?\", $arg )) }
        );
        ($( $arg:expr ),+) => (
            if cfg!(debug) { __log(4u32, fmt!( $($arg),+ )) }
        )
    )

    macro_rules! error2 (
        ($($arg:tt)*) => ( __log(1u32, format!($($arg)*)))
    )

    macro_rules! warn2 (
        ($($arg:tt)*) => ( __log(2u32, format!($($arg)*)))
    )

    macro_rules! info2 (
        ($($arg:tt)*) => ( __log(3u32, format!($($arg)*)))
    )

    macro_rules! debug2 (
        ($($arg:tt)*) => (
            if cfg!(debug) { __log(4u32, format!($($arg)*)) }
        )
    )

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
        ($($arg:tt)+) => (
            ::std::sys::FailWithCause::fail_with(format!($($arg)+), file!(), line!())
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

    macro_rules! condition (

        { pub $c:ident: $input:ty -> $out:ty; } => {

            pub mod $c {
                #[allow(non_uppercase_statics)];
                static key: ::std::local_data::Key<
                    @::std::condition::Handler<$input, $out>> =
                    &::std::local_data::Key;

                pub static cond :
                    ::std::condition::Condition<$input,$out> =
                    ::std::condition::Condition {
                        name: stringify!($c),
                        key: key
                    };
            }
        };

        { $c:ident: $input:ty -> $out:ty; } => {

            // FIXME (#6009): remove mod's `pub` below once variant above lands.
            pub mod $c {
                #[allow(non_uppercase_statics)];
                static key: ::std::local_data::Key<
                    @::std::condition::Handler<$input, $out>> =
                    &::std::local_data::Key;

                pub static cond :
                    ::std::condition::Condition<$input,$out> =
                    ::std::condition::Condition {
                        name: stringify!($c),
                        key: key
                    };
            }
        }
    )

    //
    // A scheme-style conditional that helps to improve code clarity in some instances when
    // the `if`, `else if`, and `else` keywords obscure predicates undesirably.
    //
    // # Example
    //
    // ~~~
    // let clamped =
    //     if x > mx { mx }
    //     else if x < mn { mn }
    //     else { x };
    // ~~~
    //
    // Using `cond!`, the above could be written as:
    //
    // ~~~
    // let clamped = cond!(
    //     (x > mx) { mx }
    //     (x < mn) { mn }
    //     _        { x  }
    // );
    // ~~~
    //
    // The optional default case is denoted by `_`.
    //
    macro_rules! cond (
        ( $(($pred:expr) $body:block)+ _ $default:block ) => (
            $(if $pred $body else)+
            $default
        );
        // for if the default case was ommitted
        ( $(($pred:expr) $body:block)+ ) => (
            $(if $pred $body)else+
        );
    )

    // NOTE(acrichto): start removing this after the next snapshot
    macro_rules! printf (
        ($arg:expr) => (
            print(fmt!(\"%?\", $arg))
        );
        ($( $arg:expr ),+) => (
            print(fmt!($($arg),+))
        )
    )

    // NOTE(acrichto): start removing this after the next snapshot
    macro_rules! printfln (
        ($arg:expr) => (
            println(fmt!(\"%?\", $arg))
        );
        ($( $arg:expr ),+) => (
            println(fmt!($($arg),+))
        )
    )

    // FIXME(#6846) once stdio is redesigned, this shouldn't perform an
    //              allocation but should rather delegate to an invocation of
    //              write! instead of format!
    macro_rules! print (
        ($($arg:tt)+) => ( ::std::io::print(format!($($arg)+)))
    )

    // FIXME(#6846) once stdio is redesigned, this shouldn't perform an
    //              allocation but should rather delegate to an io::Writer
    macro_rules! println (
        ($($arg:tt)+) => ({ print!($($arg)+); ::std::io::println(\"\"); })
    )

    // NOTE: use this after a snapshot lands to abstract the details
    // of the TLS interface.
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

// add a bunch of macros as though they were placed at the head of the
// program (ick). This should run before cfg stripping.
pub fn inject_std_macros(parse_sess: @mut parse::ParseSess,
                         cfg: ast::CrateConfig, c: &Crate) -> @Crate {
    let sm = match parse_item_from_source_str(@"<std-macros>",
                                              std_macros(),
                                              cfg.clone(),
                                              ~[],
                                              parse_sess) {
        Some(item) => item,
        None => fail!("expected core macros to parse correctly")
    };

    let injecter = @AstFoldFns {
        fold_mod: |modd, _| {
            // just inject the std macros at the start of the first
            // module in the crate (i.e the crate file itself.)
            let items = vec::append(~[sm], modd.items);
            ast::_mod {
                items: items,
                // FIXME #2543: Bad copy.
                .. (*modd).clone()
            }
        },
        .. *default_ast_fold()
    };
    @make_fold(injecter).fold_crate(c)
}

pub fn expand_crate(parse_sess: @mut parse::ParseSess,
                    cfg: ast::CrateConfig, c: &Crate) -> @Crate {
    // adding *another* layer of indirection here so that the block
    // visitor can swap out one exts table for another for the duration
    // of the block.  The cleaner alternative would be to thread the
    // exts table through the fold, but that would require updating
    // every method/element of AstFoldFns in fold.rs.
    let extsbox = @mut syntax_expander_table();
    let afp = default_ast_fold();
    let cx = ExtCtxt::new(parse_sess, cfg.clone());
    let f_pre = @AstFoldFns {
        fold_expr: |expr,span,recur|
            expand_expr(extsbox, cx, expr, span, recur, afp.fold_expr),
        fold_mod: |modd,recur|
            expand_mod_items(extsbox, cx, modd, recur, afp.fold_mod),
        fold_item: |item,recur|
            expand_item(extsbox, cx, item, recur, afp.fold_item),
        fold_stmt: |stmt,span,recur|
            expand_stmt(extsbox, cx, stmt, span, recur, afp.fold_stmt),
        fold_block: |blk,recur|
            expand_block(extsbox, cx, blk, recur, afp.fold_block),
        new_span: |a| new_span(cx, a),
        .. *afp};
    let f = make_fold(f_pre);

    let ret = @f.fold_crate(c);
    parse_sess.span_diagnostic.handler().abort_if_errors();
    return ret;
}

// given a function from idents to idents, produce
// an ast_fold that applies that function:
pub fn fun_to_ident_folder(f: @fn(ast::ident)->ast::ident) -> @ast_fold{
    let afp = default_ast_fold();
    let f_pre = @AstFoldFns{
        fold_ident : |id, _| f(id),
        .. *afp
    };
    make_fold(f_pre)
}

// update the ctxts in a path to get a rename node
pub fn new_ident_renamer(from: ast::ident,
                      to: ast::Name) ->
    @fn(ast::ident)->ast::ident {
    |id : ast::ident|
    ast::ident{
        name: id.name,
        ctxt: new_rename(from,to,id.ctxt)
    }
}


// update the ctxts in a path to get a mark node
pub fn new_ident_marker(mark: uint) ->
    @fn(ast::ident)->ast::ident {
    |id : ast::ident|
    ast::ident{
        name: id.name,
        ctxt: new_mark(mark,id.ctxt)
    }
}

// perform resolution (in the MTWT sense) on all of the
// idents in the tree. This is the final step in expansion.
pub fn new_ident_resolver() ->
    @fn(ast::ident)->ast::ident {
    |id : ast::ident|
    ast::ident {
        name : resolve(id),
        ctxt : illegal_ctxt
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use ast;
    use ast::{Attribute_, AttrOuter, MetaWord, empty_ctxt};
    use codemap;
    use codemap::spanned;
    use parse;
    use parse::token::{intern, get_ident_interner};
    use print::pprust;
    use util::parser_testing::{string_to_item, string_to_pat, strs_to_idents};

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
        spanned {
            span:codemap::dummy_sp(),
            node: Attribute_ {
                style: AttrOuter,
                value: @spanned {
                    node: MetaWord(s),
                    span: codemap::dummy_sp(),
                },
                is_sugared_doc: false,
            }
        }
    }

    #[test]
    fn renaming () {
        let maybe_item_ast = string_to_item(@"fn a() -> int { let b = 13; b }");
        let item_ast = match maybe_item_ast {
            Some(x) => x,
            None => fail!("test case fail")
        };
        let a_name = intern("a");
        let a2_name = intern("a2");
        let renamer = new_ident_renamer(ast::ident{name:a_name,ctxt:empty_ctxt},
                                        a2_name);
        let renamed_ast = fun_to_ident_folder(renamer).fold_item(item_ast).unwrap();
        let resolver = new_ident_resolver();
        let resolved_ast = fun_to_ident_folder(resolver).fold_item(renamed_ast).unwrap();
        let resolved_as_str = pprust::item_to_str(resolved_ast,
                                                  get_ident_interner());
        assert_eq!(resolved_as_str,~"fn a2() -> int { let b = 13; b }");


    }

    // sigh... it looks like I have two different renaming mechanisms, now...

    #[test]
    fn pat_idents(){
        let pat = string_to_pat(@"(a,Foo{x:c @ (b,9),y:Bar(4,d)})");
        let idents = @mut ~[];
        let pat_idents = new_name_finder(idents);
        pat_idents.visit_pat(pat, ());
        assert_eq!(idents, @mut strs_to_idents(~["a","c","b","d"]));
    }
}
