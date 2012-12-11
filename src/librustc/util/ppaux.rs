// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::map::HashMap;
use middle::ty;
use middle::ty::{arg, canon_mode};
use middle::ty::{bound_copy, bound_const, bound_durable, bound_owned,
        bound_trait};
use middle::ty::{bound_region, br_anon, br_named, br_self, br_cap_avoid};
use middle::ty::{ctxt, field, method};
use middle::ty::{mt, t, param_bound};
use middle::ty::{re_bound, re_free, re_scope, re_infer, re_static, Region};
use middle::ty::{ReSkolemized, ReVar};
use middle::ty::{ty_bool, ty_bot, ty_box, ty_struct, ty_enum};
use middle::ty::{ty_err, ty_estr, ty_evec, ty_float, ty_fn, ty_trait, ty_int};
use middle::ty::{ty_nil, ty_opaque_box, ty_opaque_closure_ptr, ty_param};
use middle::ty::{ty_ptr, ty_rec, ty_rptr, ty_self, ty_tup};
use middle::ty::{ty_type, ty_uniq, ty_uint, ty_infer};
use middle::ty::{ty_unboxed_vec, vid};
use metadata::encoder;
use syntax::codemap;
use syntax::codemap::span;
use syntax::print::pprust;
use syntax::print::pprust::{path_to_str, proto_to_str,
                            mode_to_str, purity_to_str,
                            onceness_to_str};
use syntax::{ast, ast_util};
use syntax::ast_map;

fn note_and_explain_region(cx: ctxt,
                           prefix: ~str,
                           region: ty::Region,
                           suffix: ~str) {
    match explain_region_and_span(cx, region) {
      (ref str, Some(span)) => {
        cx.sess.span_note(
            span,
            fmt!("%s%s%s", prefix, (*str), suffix));
      }
      (ref str, None) => {
        cx.sess.note(
            fmt!("%s%s%s", prefix, (*str), suffix));
      }
    }
}

/// Returns a string like "the block at 27:31" that attempts to explain a
/// lifetime in a way it might plausibly be understood.
fn explain_region(cx: ctxt, region: ty::Region) -> ~str {
  let (res, _) = explain_region_and_span(cx, region);
  return res;
}


fn explain_region_and_span(cx: ctxt, region: ty::Region)
    -> (~str, Option<span>)
{
    return match region {
      re_scope(node_id) => {
        match cx.items.find(node_id) {
          Some(ast_map::node_block(ref blk)) => {
            explain_span(cx, ~"block", (*blk).span)
          }
          Some(ast_map::node_expr(expr)) => {
            match expr.node {
              ast::expr_call(*) => explain_span(cx, ~"call", expr.span),
              ast::expr_method_call(*) => {
                explain_span(cx, ~"method call", expr.span)
              },
              ast::expr_match(*) => explain_span(cx, ~"match", expr.span),
              _ => explain_span(cx, ~"expression", expr.span)
            }
          }
          Some(_) | None => {
            // this really should not happen
            (fmt!("unknown scope: %d.  Please report a bug.", node_id),
             None)
          }
        }
      }

      re_free(id, br) => {
        let prefix = match br {
          br_anon(idx) => fmt!("the anonymous lifetime #%u defined on",
                               idx + 1),
          _ => fmt!("the lifetime %s as defined on",
                    bound_region_to_str(cx, br))
        };

        match cx.items.find(id) {
          Some(ast_map::node_block(ref blk)) => {
            let (msg, opt_span) = explain_span(cx, ~"block", (*blk).span);
            (fmt!("%s %s", prefix, msg), opt_span)
          }
          Some(_) | None => {
            // this really should not happen
            (fmt!("%s node %d", prefix, id), None)
          }
        }
      }

      re_static => { (~"the static lifetime", None) }

      // I believe these cases should not occur (except when debugging,
      // perhaps)
      re_infer(_) | re_bound(_) => {
        (fmt!("lifetime %?", region), None)
      }
    };

    fn explain_span(cx: ctxt, heading: ~str, span: span)
        -> (~str, Option<span>)
    {
        let lo = cx.sess.codemap.lookup_char_pos_adj(span.lo);
        (fmt!("the %s at %u:%u", heading,
              lo.line, lo.col.to_uint()), Some(span))
    }
}

fn bound_region_to_str(cx: ctxt, br: bound_region) -> ~str {
    bound_region_to_str_adorned(cx, ~"&", br, ~"")
}

fn bound_region_to_str_adorned(cx: ctxt, prefix: ~str,
                               br: bound_region, sep: ~str) -> ~str {
    if cx.sess.verbose() { return fmt!("%s%?%s", prefix, br, sep); }

    match br {
      br_named(id)         => fmt!("%s%s%s", prefix, cx.sess.str_of(id), sep),
      br_self              => fmt!("%sself%s", prefix, sep),
      br_anon(_)           => prefix,
      br_cap_avoid(_, br)  => bound_region_to_str_adorned(cx, prefix,
                                                          *br, sep)
    }
}

fn re_scope_id_to_str(cx: ctxt, node_id: ast::node_id) -> ~str {
    match cx.items.find(node_id) {
      Some(ast_map::node_block(ref blk)) => {
        fmt!("<block at %s>",
             cx.sess.codemap.span_to_str((*blk).span))
      }
      Some(ast_map::node_expr(expr)) => {
        match expr.node {
          ast::expr_call(*) => {
            fmt!("<call at %s>",
                 cx.sess.codemap.span_to_str(expr.span))
          }
          ast::expr_match(*) => {
            fmt!("<alt at %s>",
                 cx.sess.codemap.span_to_str(expr.span))
          }
          ast::expr_assign_op(*) |
          ast::expr_field(*) |
          ast::expr_unary(*) |
          ast::expr_binary(*) |
          ast::expr_index(*) => {
            fmt!("<method at %s>",
                 cx.sess.codemap.span_to_str(expr.span))
          }
          _ => {
            fmt!("<expression at %s>",
                 cx.sess.codemap.span_to_str(expr.span))
          }
        }
      }
      None => {
        fmt!("<unknown-%d>", node_id)
      }
      _ => { cx.sess.bug(
          fmt!("re_scope refers to %s",
               ast_map::node_id_to_str(cx.items, node_id,
                                       cx.sess.parse_sess.interner))) }
    }
}

// In general, if you are giving a region error message,
// you should use `explain_region()` or, better yet,
// `note_and_explain_region()`
fn region_to_str(cx: ctxt, region: Region) -> ~str {
    region_to_str_adorned(cx, ~"&", region, ~"")
}

fn region_to_str_adorned(cx: ctxt, prefix: ~str,
                         region: Region, sep: ~str) -> ~str {
    if cx.sess.verbose() {
        return fmt!("%s%?%s", prefix, region, sep);
    }

    // These printouts are concise.  They do not contain all the information
    // the user might want to diagnose an error, but there is basically no way
    // to fit that into a short string.  Hence the recommendation to use
    // `explain_region()` or `note_and_explain_region()`.
    match region {
        re_scope(_) => prefix,
        re_bound(br) => bound_region_to_str_adorned(cx, prefix, br, sep),
        re_free(_, br) => bound_region_to_str_adorned(cx, prefix, br, sep),
        re_infer(ReSkolemized(_, br)) => {
            bound_region_to_str_adorned(cx, prefix, br, sep)
        }
        re_infer(ReVar(_)) => prefix,
        re_static => fmt!("%sstatic%s", prefix, sep)
    }
}

fn mt_to_str(cx: ctxt, m: mt) -> ~str {
    let mstr = match m.mutbl {
      ast::m_mutbl => ~"mut ",
      ast::m_imm => ~"",
      ast::m_const => ~"const "
    };
    return mstr + ty_to_str(cx, m.ty);
}

fn vstore_to_str(cx: ctxt, vs: ty::vstore) -> ~str {
    match vs {
      ty::vstore_fixed(n) => fmt!("%u", n),
      ty::vstore_uniq => ~"~",
      ty::vstore_box => ~"@",
      ty::vstore_slice(r) => region_to_str(cx, r)
    }
}

fn vstore_ty_to_str(cx: ctxt, ty: ~str, vs: ty::vstore) -> ~str {
    match vs {
      ty::vstore_fixed(_) => {
        fmt!("%s/%s", ty, vstore_to_str(cx, vs))
      }
      ty::vstore_slice(_) => {
        fmt!("%s/%s", vstore_to_str(cx, vs), ty)
      }
      _ => fmt!("%s%s", vstore_to_str(cx, vs), ty)
    }
}

fn proto_ty_to_str(_cx: ctxt, proto: ast::Proto) -> ~str {
    match proto {
        ast::ProtoBare => ~"",
        ast::ProtoBox => ~"@",
        ast::ProtoBorrowed => ~"&",
        ast::ProtoUniq => ~"~",
    }
}

fn expr_repr(cx: ctxt, expr: @ast::expr) -> ~str {
    fmt!("expr(%d: %s)",
         expr.id,
         pprust::expr_to_str(expr, cx.sess.intr()))
}

fn tys_to_str(cx: ctxt, ts: ~[t]) -> ~str {
    let tstrs = ts.map(|t| ty_to_str(cx, *t));
    fmt!("[%s]", str::connect(tstrs, ", "))
}

fn bound_to_str(cx: ctxt, b: param_bound) -> ~str {
    ty::param_bound_to_str(cx, &b)
}

fn ty_to_str(cx: ctxt, typ: t) -> ~str {
    fn fn_input_to_str(cx: ctxt, input: {mode: ast::mode, ty: t}) ->
       ~str {
        let {mode, ty} = input;
        let modestr = match canon_mode(cx, mode) {
          ast::infer(_) => ~"",
          ast::expl(m) => {
            if !ty::type_needs_infer(ty) &&
                m == ty::default_arg_mode_for_ty(cx, ty) {
                ~""
            } else {
                mode_to_str(ast::expl(m)) + ":"
            }
          }
        };
        modestr + ty_to_str(cx, ty)
    }
    fn fn_to_str(cx: ctxt,
                 proto: ast::Proto,
                 region: ty::Region,
                 purity: ast::purity,
                 onceness: ast::Onceness,
                 ident: Option<ast::ident>,
                 inputs: ~[arg],
                 output: t,
                 cf: ast::ret_style) -> ~str {
        let mut s;

        s = match purity {
            ast::impure_fn => ~"",
            _ => purity_to_str(purity) + ~" "
        };

        s += match onceness {
            ast::Many => ~"",
            ast::Once => onceness_to_str(onceness) + ~" "
        };

        s += proto_ty_to_str(cx, proto);

        match (proto, region) {
            (ast::ProtoBox, ty::re_static) |
            (ast::ProtoUniq, ty::re_static) |
            (ast::ProtoBare, ty::re_static) => {
            }

            (_, region) => {
                s += region_to_str_adorned(cx, ~"", region, ~"/");
            }
        }

        s += ~"fn";

        match ident {
          Some(i) => { s += ~" "; s += cx.sess.str_of(i); }
          _ => { }
        }
        s += ~"(";
        let strs = inputs.map(|a| fn_input_to_str(cx, *a));
        s += str::connect(strs, ~", ");
        s += ~")";
        if ty::get(output).sty != ty_nil {
            s += ~" -> ";
            match cf {
              ast::noreturn => { s += ~"!"; }
              ast::return_val => { s += ty_to_str(cx, output); }
            }
        }
        return s;
    }
    fn method_to_str(cx: ctxt, m: method) -> ~str {
        return fn_to_str(
            cx,
            m.fty.meta.proto,
            m.fty.meta.region,
            m.fty.meta.purity,
            m.fty.meta.onceness,
            Some(m.ident),
            m.fty.sig.inputs,
            m.fty.sig.output,
            m.fty.meta.ret_style) + ~";";
    }
    fn field_to_str(cx: ctxt, f: field) -> ~str {
        return cx.sess.str_of(f.ident) + ~": " + mt_to_str(cx, f.mt);
    }

    // if there is an id, print that instead of the structural type:
    for ty::type_def_id(typ).each |def_id| {
        // note that this typedef cannot have type parameters
        return ast_map::path_to_str(ty::item_path(cx, *def_id),
                                    cx.sess.intr());
    }

    // pretty print the structural type representation:
    return match ty::get(typ).sty {
      ty_nil => ~"()",
      ty_bot => ~"_|_",
      ty_bool => ~"bool",
      ty_int(ast::ty_i) => ~"int",
      ty_int(ast::ty_char) => ~"char",
      ty_int(t) => ast_util::int_ty_to_str(t),
      ty_uint(ast::ty_u) => ~"uint",
      ty_uint(t) => ast_util::uint_ty_to_str(t),
      ty_float(ast::ty_f) => ~"float",
      ty_float(t) => ast_util::float_ty_to_str(t),
      ty_box(tm) => ~"@" + mt_to_str(cx, tm),
      ty_uniq(tm) => ~"~" + mt_to_str(cx, tm),
      ty_ptr(tm) => ~"*" + mt_to_str(cx, tm),
      ty_rptr(r, tm) => {
        region_to_str_adorned(cx, ~"&", r, ~"/") + mt_to_str(cx, tm)
      }
      ty_unboxed_vec(tm) => { ~"unboxed_vec<" + mt_to_str(cx, tm) + ~">" }
      ty_type => ~"type",
      ty_rec(elems) => {
        let strs = elems.map(|fld| field_to_str(cx, *fld));
        ~"{" + str::connect(strs, ~",") + ~"}"
      }
      ty_tup(elems) => {
        let strs = elems.map(|elem| ty_to_str(cx, *elem));
        ~"(" + str::connect(strs, ~",") + ~")"
      }
      ty_fn(ref f) => {
        fn_to_str(cx,
                  f.meta.proto,
                  f.meta.region,
                  f.meta.purity,
                  f.meta.onceness,
                  None,
                  f.sig.inputs,
                  f.sig.output,
                  f.meta.ret_style)
      }
      ty_infer(infer_ty) => infer_ty.to_str(),
      ty_err => ~"[type error]",
      ty_param({idx: id, _}) => {
        ~"'" + str::from_bytes(~[('a' as u8) + (id as u8)])
      }
      ty_self => ~"self",
      ty_enum(did, ref substs) | ty_struct(did, ref substs) => {
        let path = ty::item_path(cx, did);
        let base = ast_map::path_to_str(path, cx.sess.intr());
        parameterized(cx, base, (*substs).self_r, (*substs).tps)
      }
      ty_trait(did, ref substs, vs) => {
        let path = ty::item_path(cx, did);
        let base = ast_map::path_to_str(path, cx.sess.intr());
        let result = parameterized(cx, base, (*substs).self_r, (*substs).tps);
        vstore_ty_to_str(cx, result, vs)
      }
      ty_evec(mt, vs) => {
        vstore_ty_to_str(cx, fmt!("[%s]", mt_to_str(cx, mt)), vs)
      }
      ty_estr(vs) => vstore_ty_to_str(cx, ~"str", vs),
      ty_opaque_box => ~"@?",
      ty_opaque_closure_ptr(ast::ProtoBorrowed) => ~"closure&",
      ty_opaque_closure_ptr(ast::ProtoBox) => ~"closure@",
      ty_opaque_closure_ptr(ast::ProtoUniq) => ~"closure~",
      ty_opaque_closure_ptr(ast::ProtoBare) => ~"closure"
    }
}

fn parameterized(cx: ctxt,
                 base: ~str,
                 self_r: Option<ty::Region>,
                 tps: ~[ty::t]) -> ~str {

    let r_str = match self_r {
      None => ~"",
      Some(r) => {
        fmt!("/%s", region_to_str(cx, r))
      }
    };

    if vec::len(tps) > 0u {
        let strs = vec::map(tps, |t| ty_to_str(cx, *t));
        fmt!("%s%s<%s>", base, r_str, str::connect(strs, ~","))
    } else {
        fmt!("%s%s", base, r_str)
    }
}

fn ty_to_short_str(cx: ctxt, typ: t) -> ~str {
    let mut s = encoder::encoded_ty(cx, typ);
    if str::len(s) >= 32u { s = str::slice(s, 0u, 32u); }
    return s;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
