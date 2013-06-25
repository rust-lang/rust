// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{blk_, attribute_, attr_outer, meta_word};
use ast::{crate, expr_, expr_mac, mac_invoc_tt};
use ast::{item_mac, stmt_, stmt_mac, stmt_expr, stmt_semi};
use ast::{illegal_ctxt};
use ast;
use ast_util::{new_rename, new_mark, resolve};
use attr;
use codemap;
use codemap::{span, CallInfo, ExpandedFrom, NameAndSpan, spanned};
use ext::base::*;
use fold::*;
use parse;
use parse::{parse_item_from_source_str};
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
                mac_invoc_tt(pth, ref tts) => {
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
                            cx.bt_push(ExpandedFrom(CallInfo {
                                call_site: s,
                                callee: NameAndSpan {
                                    name: extnamestr,
                                    span: exp_sp,
                                },
                            }));

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
                                copy fld.fold_expr(expanded).node;
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
            let mname = attr::get_attr_name(attr);

            match (*extsbox).find(&intern(mname)) {
              Some(@SE(ItemDecorator(dec_fn))) => {
                  cx.bt_push(ExpandedFrom(CallInfo {
                      call_site: attr.span,
                      callee: NameAndSpan {
                          name: mname,
                          span: None
                      }
                  }));
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
pub fn contains_macro_escape (attrs: &[ast::attribute]) -> bool {
    attrs.iter().any_(|attr| "macro_escape" == attr::get_attr_name(attr))
}

// Support for item-position macro invocations, exactly the same
// logic as for expression-position macro invocations.
pub fn expand_item_mac(extsbox: @mut SyntaxEnv,
                       cx: @ExtCtxt, it: @ast::item,
                       fld: @ast_fold)
                    -> Option<@ast::item> {
    let (pth, tts) = match it.node {
        item_mac(codemap::spanned { node: mac_invoc_tt(pth, ref tts), _}) => {
            (pth, copy *tts)
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
            cx.bt_push(ExpandedFrom(CallInfo {
                call_site: it.span,
                callee: NameAndSpan {
                    name: extnamestr,
                    span: expand.span
                }
            }));
            ((*expand).expander)(cx, it.span, tts)
        }
        Some(@SE(IdentTT(ref expand))) => {
            if it.ident == parse::token::special_idents::invalid {
                cx.span_fatal(pth.span,
                              fmt!("macro %s! expects an ident argument",
                                   extnamestr));
            }
            cx.bt_push(ExpandedFrom(CallInfo {
                call_site: it.span,
                callee: NameAndSpan {
                    name: extnamestr,
                    span: expand.span
                }
            }));
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
                mac_invoc_tt(pth, ref tts) => {
                    (copy *mac, pth, copy *tts, semi)
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
            cx.bt_push(ExpandedFrom(CallInfo {
                call_site: sp,
                callee: NameAndSpan { name: extnamestr, span: exp_sp }
            }));
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
                    copy *fully_expanded
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

// return a visitor that extracts the pat_ident paths
// from a given pattern and puts them in a mutable
// array (passed in to the traversal)
pub fn new_name_finder() -> @Visitor<@mut ~[ast::ident]> {
    let default_visitor = visit::default_visitor();
    @Visitor{
        visit_pat : |p:@ast::pat,
                     (ident_accum, v): (@mut ~[ast::ident], visit::vt<@mut ~[ast::ident]>)| {
            match *p {
                // we found a pat_ident!
                ast::pat{id:_, node: ast::pat_ident(_,path,ref inner), span:_} => {
                    match path {
                        // a path of length one:
                        @ast::Path{global: false,idents: [id], span:_,rp:_,types:_} =>
                        ident_accum.push(id),
                        // I believe these must be enums...
                        _ => ()
                    }
                    // visit optional subpattern of pat_ident:
                    for inner.iter().advance |subpat: &@ast::pat| {
                        (v.visit_pat)(*subpat, (ident_accum, v))
                    }
                }
                // use the default traversal for non-pat_idents
                _ => visit::visit_pat(p,(ident_accum,v))
            }
        },
        .. *default_visitor
    }
}

pub fn expand_block(extsbox: @mut SyntaxEnv,
                    _cx: @ExtCtxt,
                    blk: &blk_,
                    sp: span,
                    fld: @ast_fold,
                    orig: @fn(&blk_, span, @ast_fold) -> (blk_, span))
                 -> (blk_, span) {
    // see note below about treatment of exts table
    with_exts_frame!(extsbox,false,orig(blk,sp,fld))
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

pub fn core_macros() -> @str {
    return
@"pub mod macros {
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
            __log(4u32, fmt!( \"%?\", $arg ))
        );
        ($( $arg:expr ),+) => (
            __log(4u32, fmt!( $($arg),+ ))
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

    macro_rules! assert(
        ($cond:expr) => {
            if !$cond {
                ::std::sys::FailWithCause::fail_with(
                    ~\"assertion failed: \" + stringify!($cond), file!(), line!())
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
                    fail!(\"left: %? does not equal right: %?\", given_val, expected_val);
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

        { pub $c:ident: $in:ty -> $out:ty; } => {

            pub mod $c {
                fn key(_x: @::std::condition::Handler<$in,$out>) { }

                pub static cond :
                    ::std::condition::Condition<'static,$in,$out> =
                    ::std::condition::Condition {
                        name: stringify!($c),
                        key: key
                    };
            }
        };

        { $c:ident: $in:ty -> $out:ty; } => {

            // FIXME (#6009): remove mod's `pub` below once variant above lands.
            pub mod $c {
                fn key(_x: @::std::condition::Handler<$in,$out>) { }

                pub static cond :
                    ::std::condition::Condition<'static,$in,$out> =
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
}";
}

pub fn expand_crate(parse_sess: @mut parse::ParseSess,
                    cfg: ast::crate_cfg, c: @crate) -> @crate {
    // adding *another* layer of indirection here so that the block
    // visitor can swap out one exts table for another for the duration
    // of the block.  The cleaner alternative would be to thread the
    // exts table through the fold, but that would require updating
    // every method/element of AstFoldFns in fold.rs.
    let extsbox = @mut syntax_expander_table();
    let afp = default_ast_fold();
    let cx = ExtCtxt::new(parse_sess, copy cfg);
    let f_pre = @AstFoldFns {
        fold_expr: |expr,span,recur|
            expand_expr(extsbox, cx, expr, span, recur, afp.fold_expr),
        fold_mod: |modd,recur|
            expand_mod_items(extsbox, cx, modd, recur, afp.fold_mod),
        fold_item: |item,recur|
            expand_item(extsbox, cx, item, recur, afp.fold_item),
        fold_stmt: |stmt,span,recur|
            expand_stmt(extsbox, cx, stmt, span, recur, afp.fold_stmt),
        fold_block: |blk,span,recur|
            expand_block(extsbox, cx, blk, span, recur, afp.fold_block),
        new_span: |a| new_span(cx, a),
        .. *afp};
    let f = make_fold(f_pre);
    // add a bunch of macros as though they were placed at the
    // head of the program (ick).
    let attrs = ~[
        spanned {
            span: codemap::dummy_sp(),
            node: attribute_ {
                style: attr_outer,
                value: @spanned {
                    node: meta_word(@"macro_escape"),
                    span: codemap::dummy_sp(),
                },
                is_sugared_doc: false,
            }
        }
    ];

    let cm = match parse_item_from_source_str(@"<core-macros>",
                                              core_macros(),
                                              copy cfg,
                                              attrs,
                                              parse_sess) {
        Some(item) => item,
        None => cx.bug("expected core macros to parse correctly")
    };
    // This is run for its side-effects on the expander env,
    // as it registers all the core macros as expanders.
    f.fold_item(cm);

    @f.fold_crate(&*c)
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
    use ast::{attribute_, attr_outer, meta_word, empty_ctxt};
    use codemap;
    use codemap::spanned;
    use parse;
    use parse::token::{intern, get_ident_interner};
    use print::pprust;
    use util::parser_testing::{string_to_item, string_to_pat, strs_to_idents};
    use visit::{mk_vt};

    // make sure that fail! is present
    #[test] fn fail_exists_test () {
        let src = @"fn main() { fail!(\"something appropriately gloomy\");}";
        let sess = parse::new_parse_sess(None);
        let crate_ast = parse::parse_crate_from_source_str(
            @"<test>",
            src,
            ~[],sess);
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

    #[test] fn core_macros_must_parse () {
        let src = @"
  pub mod macros {
    macro_rules! ignore (($($x:tt)*) => (()))

    macro_rules! error ( ($( $arg:expr ),+) => (
        log(::core::error, fmt!( $($arg),+ )) ))
}";
        let sess = parse::new_parse_sess(None);
        let cfg = ~[];
        let item_ast = parse::parse_item_from_source_str(
            @"<test>",
            src,
            cfg,~[make_dummy_attr (@"macro_escape")],sess);
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

    // make a "meta_word" outer attribute with the given name
    fn make_dummy_attr(s: @str) -> ast::attribute {
        spanned {
            span:codemap::dummy_sp(),
            node: attribute_ {
                style: attr_outer,
                value: @spanned {
                    node: meta_word(s),
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
        let renamed_ast = fun_to_ident_folder(renamer).fold_item(item_ast).get();
        let resolver = new_ident_resolver();
        let resolved_ast = fun_to_ident_folder(resolver).fold_item(renamed_ast).get();
        let resolved_as_str = pprust::item_to_str(resolved_ast,
                                                  get_ident_interner());
        assert_eq!(resolved_as_str,~"fn a2() -> int { let b = 13; b }");


    }

    // sigh... it looks like I have two different renaming mechanisms, now...

    #[test]
    fn pat_idents(){
        let pat = string_to_pat(@"(a,Foo{x:c @ (b,9),y:Bar(4,d)})");
        let pat_idents = new_name_finder();
        let idents = @mut ~[];
        ((*pat_idents).visit_pat)(pat, (idents, mk_vt(pat_idents)));
        assert_eq!(idents,@mut strs_to_idents(~["a","c","b","d"]));
    }
}
