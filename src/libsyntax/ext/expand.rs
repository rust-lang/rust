use std::map::HashMap;

use ast::{crate, expr_, expr_mac, mac_invoc, mac_invoc_tt,
          tt_delim, tt_tok, item_mac, stmt_, stmt_mac, stmt_expr, stmt_semi};
use fold::*;
use ext::base::*;
use ext::qquote::{qq_helper};
use parse::{parser, parse_expr_from_source_str, new_parser_from_tts};


use codemap::{span, ExpandedFrom};


fn expand_expr(exts: HashMap<~str, syntax_extension>, cx: ext_ctxt,
               e: expr_, s: span, fld: ast_fold,
               orig: fn@(expr_, span, ast_fold) -> (expr_, span))
    -> (expr_, span)
{
    return match e {
      // expr_mac should really be expr_ext or something; it's the
      // entry-point for all syntax extensions.
          expr_mac(mac) => {

            match mac.node {
              // Old-style macros. For compatibility, will erase this whole
              // block once we've transitioned.
              mac_invoc(pth, args, body) => {
                assert (vec::len(pth.idents) > 0u);
                /* using idents and token::special_idents would make the
                the macro names be hygienic */
                let extname = cx.parse_sess().interner.get(pth.idents[0]);
                match exts.find(*extname) {
                  None => {
                    cx.span_fatal(pth.span,
                                  fmt!("macro undefined: '%s'", *extname))
                  }
                  Some(item_decorator(_)) => {
                    cx.span_fatal(
                        pth.span,
                        fmt!("%s can only be used as a decorator", *extname));
                  }
                  Some(normal({expander: exp, span: exp_sp})) => {
                    let expanded = exp(cx, mac.span, args, body);

                    cx.bt_push(ExpandedFrom({call_site: s,
                                callie: {name: *extname, span: exp_sp}}));
                    //keep going, outside-in
                    let fully_expanded = fld.fold_expr(expanded).node;
                    cx.bt_pop();

                    (fully_expanded, s)
                  }
                  Some(macro_defining(ext)) => {
                    let named_extension = ext(cx, mac.span, args, body);
                    exts.insert(named_extension.name, named_extension.ext);
                    (ast::expr_rec(~[], None), s)
                  }
                  Some(normal_tt(_)) => {
                    cx.span_fatal(pth.span,
                                  fmt!("this tt-style macro should be \
                                        invoked '%s!(...)'", *extname))
                  }
                  Some(item_tt(*)) => {
                    cx.span_fatal(pth.span,
                                  ~"cannot use item macros in this context");
                  }
                }
              }

              // Token-tree macros, these will be the only case when we're
              // finished transitioning.
              mac_invoc_tt(pth, tts) => {
                assert (vec::len(pth.idents) == 1u);
                /* using idents and token::special_idents would make the
                the macro names be hygienic */
                let extname = cx.parse_sess().interner.get(pth.idents[0]);
                match exts.find(*extname) {
                  None => {
                    cx.span_fatal(pth.span,
                                  fmt!("macro undefined: '%s'", *extname))
                  }
                  Some(normal_tt({expander: exp, span: exp_sp})) => {
                    let expanded = match exp(cx, mac.span, tts) {
                      mr_expr(e) => e,
                      mr_any(expr_maker,_,_) => expr_maker(),
                      _ => cx.span_fatal(
                          pth.span, fmt!("non-expr macro in expr pos: %s",
                                         *extname))
                    };

                    cx.bt_push(ExpandedFrom({call_site: s,
                                callie: {name: *extname, span: exp_sp}}));
                    //keep going, outside-in
                    let fully_expanded = fld.fold_expr(expanded).node;
                    cx.bt_pop();

                    (fully_expanded, s)
                  }
                  Some(normal({expander: exp, span: exp_sp})) => {
                    //convert the new-style invoc for the old-style macro
                    let arg = base::tt_args_to_original_flavor(cx, pth.span,
                                                               tts);
                    let expanded = exp(cx, mac.span, arg, None);

                    cx.bt_push(ExpandedFrom({call_site: s,
                                callie: {name: *extname, span: exp_sp}}));
                    //keep going, outside-in
                    let fully_expanded = fld.fold_expr(expanded).node;
                    cx.bt_pop();

                    (fully_expanded, s)
                  }
                  _ => {
                    cx.span_fatal(pth.span,
                                  fmt!("'%s' is not a tt-style macro",
                                       *extname))
                  }

                }
              }
              _ => cx.span_bug(mac.span, ~"naked syntactic bit")
            }
          }
          _ => orig(e, s, fld)
        };
}

// This is a secondary mechanism for invoking syntax extensions on items:
// "decorator" attributes, such as #[auto_serialize]. These are invoked by an
// attribute prefixing an item, and are interpreted by feeding the item
// through the named attribute _as a syntax extension_ and splicing in the
// resulting item vec into place in favour of the decorator. Note that
// these do _not_ work for macro extensions, just item_decorator ones.
//
// NB: there is some redundancy between this and expand_item, below, and
// they might benefit from some amount of semantic and language-UI merger.
fn expand_mod_items(exts: HashMap<~str, syntax_extension>, cx: ext_ctxt,
                    module_: ast::_mod, fld: ast_fold,
                    orig: fn@(ast::_mod, ast_fold) -> ast::_mod)
    -> ast::_mod
{
    // Fold the contents first:
    let module_ = orig(module_, fld);

    // For each item, look through the attributes.  If any of them are
    // decorated with "item decorators", then use that function to transform
    // the item into a new set of items.
    let new_items = do vec::flat_map(module_.items) |item| {
        do vec::foldr(item.attrs, ~[*item]) |attr, items| {
            let mname = match attr.node.value.node {
              ast::meta_word(n) => n,
              ast::meta_name_value(n, _) => n,
              ast::meta_list(n, _) => n
            };
            match exts.find(mname) {
              None | Some(normal(_)) | Some(macro_defining(_))
              | Some(normal_tt(_)) | Some(item_tt(*)) => items,
              Some(item_decorator(dec_fn)) => {
                dec_fn(cx, attr.span, attr.node.value, items)
              }
            }
        }
    };

    return {items: new_items, ..module_};
}


// When we enter a module, record it, for the sake of `module!`
fn expand_item(exts: HashMap<~str, syntax_extension>,
               cx: ext_ctxt, &&it: @ast::item, fld: ast_fold,
               orig: fn@(&&v: @ast::item, ast_fold) -> Option<@ast::item>)
    -> Option<@ast::item>
{
    let is_mod = match it.node {
      ast::item_mod(_) | ast::item_foreign_mod(_) => true,
      _ => false
    };
    let maybe_it = match it.node {
      ast::item_mac(*) => expand_item_mac(exts, cx, it, fld),
      _ => Some(it)
    };

    match maybe_it {
      Some(it) => {
        if is_mod { cx.mod_push(it.ident); }
        let ret_val = orig(it, fld);
        if is_mod { cx.mod_pop(); }
        return ret_val;
      }
      None => return None
    }
}

// avoid excess indentation when a series of nested `match`es
// has only one "good" outcome
macro_rules! biased_match (
    (   ($e    :expr) ~ ($p    :pat) else $err    :stmt ;
     $( ($e_cdr:expr) ~ ($p_cdr:pat) else $err_cdr:stmt ; )*
     => $body:expr
    ) => (
        match $e {
            $p => {
                biased_match!($( ($e_cdr) ~ ($p_cdr) else $err_cdr ; )*
                              => $body)
            }
            _ => { $err }
        }
    );
    ( => $body:expr ) => ( $body )
)


// Support for item-position macro invocations, exactly the same
// logic as for expression-position macro invocations.
fn expand_item_mac(exts: HashMap<~str, syntax_extension>,
                   cx: ext_ctxt, &&it: @ast::item,
                   fld: ast_fold) -> Option<@ast::item> {
    let (pth, tts) = biased_match!(
        (it.node) ~ (item_mac({node: mac_invoc_tt(pth, tts), _})) else {
            cx.span_bug(it.span, ~"invalid item macro invocation")
        };
        => (pth, tts)
    );

    let extname = cx.parse_sess().interner.get(pth.idents[0]);
    let (expanded, ex_span) = match exts.find(*extname) {
        None => cx.span_fatal(pth.span,
                              fmt!("macro undefined: '%s!'", *extname)),

        Some(normal_tt(expand)) => {
            if it.ident != parse::token::special_idents::invalid {
                cx.span_fatal(pth.span,
                              fmt!("macro %s! expects no ident argument, \
                                    given '%s'", *extname,
                                   *cx.parse_sess().interner.get(it.ident)));
            }
            (expand.expander(cx, it.span, tts), expand.span)
        }
        Some(item_tt(expand)) => {
            if it.ident == parse::token::special_idents::invalid {
                cx.span_fatal(pth.span,
                              fmt!("macro %s! expects an ident argument",
                                   *extname));
            }
            (expand.expander(cx, it.span, it.ident, tts), expand.span)
        }
        _ => cx.span_fatal(
            it.span, fmt!("%s! is not legal in item position", *extname))
    };

    cx.bt_push(ExpandedFrom({call_site: it.span,
                              callie: {name: *extname, span: ex_span}}));
    let maybe_it = match expanded {
        mr_item(it) => fld.fold_item(it),
        mr_expr(_) => cx.span_fatal(pth.span,
                                    ~"expr macro in item position: "
                                    + *extname),
        mr_any(_, item_maker, _) =>
            option::chain(item_maker(), |i| {fld.fold_item(i)}),
        mr_def(mdef) => {
            exts.insert(mdef.name, mdef.ext);
            None
        }
    };
    cx.bt_pop();
    return maybe_it;
}

fn expand_stmt(exts: HashMap<~str, syntax_extension>, cx: ext_ctxt,
               && s: stmt_, sp: span, fld: ast_fold,
               orig: fn@(&&s: stmt_, span, ast_fold) -> (stmt_, span))
    -> (stmt_, span)
{
    let (mac, pth, tts, semi) = biased_match! (
        (s)        ~ (stmt_mac(mac, semi))    else return orig(s, sp, fld);
        (mac.node) ~ (mac_invoc_tt(pth, tts)) else {
            cx.span_bug(mac.span, ~"naked syntactic bit")
        };
        => (mac, pth, tts, semi));

    assert(vec::len(pth.idents) == 1u);
    let extname = cx.parse_sess().interner.get(pth.idents[0]);
    let (fully_expanded, sp) = match exts.find(*extname) {
        None =>
            cx.span_fatal(pth.span, fmt!("macro undefined: '%s'", *extname)),

        Some(normal_tt({expander: exp, span: exp_sp})) => {
            let expanded = match exp(cx, mac.span, tts) {
                mr_expr(e) =>
                    @{node: stmt_expr(e, cx.next_id()), span: e.span},
                mr_any(_,_,stmt_mkr) => stmt_mkr(),
                _ => cx.span_fatal(
                    pth.span,
                    fmt!("non-stmt macro in stmt pos: %s", *extname))
            };

            cx.bt_push(ExpandedFrom(
                {call_site: sp, callie: {name: *extname, span: exp_sp}}));
            //keep going, outside-in
            let fully_expanded = fld.fold_stmt(expanded).node;
            cx.bt_pop();

            (fully_expanded, sp)
        }

        Some(normal({expander: exp, span: exp_sp})) => {
            //convert the new-style invoc for the old-style macro
            let arg = base::tt_args_to_original_flavor(cx, pth.span, tts);
            let exp_expr = exp(cx, mac.span, arg, None);
            let expanded = @{node: stmt_expr(exp_expr, cx.next_id()),
                             span: exp_expr.span};

            cx.bt_push(ExpandedFrom({call_site: sp,
                                      callie: {name: *extname,
                                               span: exp_sp}}));
            //keep going, outside-in
            let fully_expanded = fld.fold_stmt(expanded).node;
            cx.bt_pop();

            (fully_expanded, sp)
        }

        _ => {
            cx.span_fatal(pth.span,
                          fmt!("'%s' is not a tt-style macro", *extname))
        }
    };

    return (match fully_expanded {
        stmt_expr(e, stmt_id) if semi => stmt_semi(e, stmt_id),
        _ => { fully_expanded } /* might already have a semi */
    }, sp)

}


fn new_span(cx: ext_ctxt, sp: span) -> span {
    /* this discards information in the case of macro-defining macros */
    return span {lo: sp.lo, hi: sp.hi, expn_info: cx.backtrace()};
}

// FIXME (#2247): this is a terrible kludge to inject some macros into
// the default compilation environment. When the macro-definition system
// is substantially more mature, these should move from here, into a
// compiled part of libcore at very least.

fn core_macros() -> ~str {
    return
~"{
    macro_rules! ignore (($($x:tt)*) => (()))

    macro_rules! error ( ($( $arg:expr ),+) => (
        log(core::error, fmt!( $($arg),+ )) ))
    macro_rules! warn ( ($( $arg:expr ),+) => (
        log(core::warn, fmt!( $($arg),+ )) ))
    macro_rules! info ( ($( $arg:expr ),+) => (
        log(core::info, fmt!( $($arg),+ )) ))
    macro_rules! debug ( ($( $arg:expr ),+) => (
        log(core::debug, fmt!( $($arg),+ )) ))

    macro_rules! die(
        ($msg: expr) => (
            {
                do core::str::as_buf($msg) |msg_buf, _msg_len| {
                    do core::str::as_buf(file!()) |file_buf, _file_len| {
                        unsafe {
                            let msg_buf = core::cast::transmute(msg_buf);
                            let file_buf = core::cast::transmute(file_buf);
                            let line = line!() as core::libc::size_t;
                            core::rt::rt_fail_(msg_buf, file_buf, line)
                        }
                    }
                }
            }
        );
        () => (
            die!(\"explicit failure\")
        )
    )
}";
}

fn expand_crate(parse_sess: parse::parse_sess,
                cfg: ast::crate_cfg, c: @crate) -> @crate {
    let exts = syntax_expander_table();
    let afp = default_ast_fold();
    let cx: ext_ctxt = mk_ctxt(parse_sess, cfg);
    let f_pre =
        @{fold_expr: |a,b,c| expand_expr(exts, cx, a, b, c, afp.fold_expr),
          fold_mod: |a,b| expand_mod_items(exts, cx, a, b, afp.fold_mod),
          fold_item: |a,b| expand_item(exts, cx, a, b, afp.fold_item),
          fold_stmt: |a,b,c| expand_stmt(exts, cx, a, b, c, afp.fold_stmt),
          new_span: |a| new_span(cx, a),
          .. *afp};
    let f = make_fold(f_pre);
    let cm = parse_expr_from_source_str(~"<core-macros>",
                                        @core_macros(),
                                        cfg,
                                        parse_sess);

    // This is run for its side-effects on the expander env,
    // as it registers all the core macros as expanders.
    f.fold_expr(cm);

    let res = @f.fold_crate(*c);
    return res;
}
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
