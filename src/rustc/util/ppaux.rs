import std::map::hashmap;
import middle::ty;
import middle::ty::{arg, canon_mode};
import middle::ty::{bound_region, br_anon, br_named, br_self, br_cap_avoid};
import middle::ty::{ck_block, ck_box, ck_uniq, ctxt, field, method};
import middle::ty::{mt, t};
import middle::ty::{re_bound, re_free, re_scope, re_var, re_static, region};
import middle::ty::{ty_bool, ty_bot, ty_box, ty_class, ty_enum};
import middle::ty::{ty_estr, ty_evec, ty_float, ty_fn, ty_trait, ty_int};
import middle::ty::{ty_nil, ty_opaque_box, ty_opaque_closure_ptr, ty_param};
import middle::ty::{ty_ptr, ty_rec, ty_rptr, ty_self, ty_tup};
import middle::ty::{ty_type, ty_uniq, ty_uint, ty_var, ty_var_integral};
import middle::ty::{ty_unboxed_vec, vid};
import metadata::encoder;
import syntax::codemap;
import syntax::codemap::span;
import syntax::print::pprust;
import syntax::print::pprust::{path_to_str, proto_to_str,
                               mode_to_str, purity_to_str};
import syntax::{ast, ast_util};
import syntax::ast_map;
import driver::session::session;

/// Returns a string like "reference valid for the block at 27:31 in foo.rs"
/// that attempts to explain a lifetime in a way it might plausibly be
/// understood.
fn explain_region(cx: ctxt, region: ty::region) -> ~str {
    return alt region {
      re_scope(node_id) => {
        let scope_str = alt cx.items.find(node_id) {
          some(ast_map::node_block(blk)) => {
            explain_span(cx, ~"block", blk.span)
          }
          some(ast_map::node_expr(expr)) => {
            alt expr.node {
              ast::expr_call(*) => { explain_span(cx, ~"call", expr.span) }
              ast::expr_alt(*) => { explain_span(cx, ~"alt", expr.span) }
              _ => { explain_span(cx, ~"expression", expr.span) }
            }
          }
          some(_) | none => {
            // this really should not happen
            fmt!{"unknown scope: %d.  Please report a bug.", node_id}
          }
        };
        fmt!{"reference valid for the %s", scope_str}
      }

      re_free(id, br) => {
        alt cx.items.find(id) {
          some(ast_map::node_block(blk)) => {
            fmt!{"reference with lifetime %s as defined on %s",
                 bound_region_to_str(cx, br),
                 explain_span(cx, ~"the block", blk.span)}
          }
          some(_) | none => {
            // this really should not happen
            fmt!{"reference with lifetime %s as defined on node %d",
                 bound_region_to_str(cx, br), id}
          }
        }
      }

      re_static => { ~"reference to static data" }

      // I believe these cases should not occur.
      re_var(_) | re_bound(_) => {
        fmt!{"reference with lifetime %?", region}
      }
    };

    fn explain_span(cx: ctxt, heading: ~str, span: span) -> ~str {
        let lo = codemap::lookup_char_pos_adj(cx.sess.codemap, span.lo);
        fmt!{"%s at %u:%u", heading, lo.line, lo.col}
    }
}

fn bound_region_to_str(cx: ctxt, br: bound_region) -> ~str {
    alt br {
      br_anon                        => { ~"&" }
      br_named(str)                  => { fmt!{"&%s", *str} }
      br_self if cx.sess.ppregions() => { ~"&<self>" }
      br_self                        => { ~"&self" }

      // FIXME(#3011) -- even if this arm is removed, exhaustiveness checking
      // does not fail
      br_cap_avoid(id, br) => {
        if cx.sess.ppregions() {
            fmt!{"br_cap_avoid(%?, %s)", id, bound_region_to_str(cx, *br)}
        } else {
            bound_region_to_str(cx, *br)
        }
      }
    }
}

fn re_scope_id_to_str(cx: ctxt, node_id: ast::node_id) -> ~str {
    alt cx.items.find(node_id) {
      some(ast_map::node_block(blk)) => {
        fmt!{"<block at %s>",
             codemap::span_to_str(blk.span, cx.sess.codemap)}
      }
      some(ast_map::node_expr(expr)) => {
        alt expr.node {
          ast::expr_call(*) => {
            fmt!{"<call at %s>",
                 codemap::span_to_str(expr.span, cx.sess.codemap)}
          }
          ast::expr_alt(*) => {
            fmt!{"<alt at %s>",
                 codemap::span_to_str(expr.span, cx.sess.codemap)}
          }
          ast::expr_assign_op(*) |
          ast::expr_field(*) |
          ast::expr_unary(*) |
          ast::expr_binary(*) |
          ast::expr_index(*) => {
            fmt!{"<method at %s>",
                 codemap::span_to_str(expr.span, cx.sess.codemap)}
          }
          _ => {
            fmt!{"<expression at %s>",
                 codemap::span_to_str(expr.span, cx.sess.codemap)}
          }
        }
      }
      none => {
        fmt!{"<unknown-%d>", node_id}
      }
      _ => { cx.sess.bug(
          fmt!{"re_scope refers to %s",
               ast_map::node_id_to_str(cx.items, node_id)}) }
    }
}

fn region_to_str(cx: ctxt, region: region) -> ~str {
    alt region {
      re_scope(node_id) => {
        if cx.sess.ppregions() {
            fmt!{"&%s", re_scope_id_to_str(cx, node_id)}
        } else {
            ~"&"
        }
      }
      re_bound(br) => {
          bound_region_to_str(cx, br)
      }
      re_free(id, br) => {
        if cx.sess.ppregions() {
            // For debugging, this version is sometimes helpful:
            fmt!{"{%d} %s", id, bound_region_to_str(cx, br)}
        } else {
            // But this version is what the user expects to see:
            bound_region_to_str(cx, br)
        }
      }

      // These two should not be seen by end-users (very often, anyhow):
      re_var(id)    => fmt!{"&%s", id.to_str()},
      re_static     => ~"&static"
    }
}

fn mt_to_str(cx: ctxt, m: mt) -> ~str {
    let mstr = alt m.mutbl {
      ast::m_mutbl => ~"mut ",
      ast::m_imm => ~"",
      ast::m_const => ~"const "
    };
    return mstr + ty_to_str(cx, m.ty);
}

fn vstore_to_str(cx: ctxt, vs: ty::vstore) -> ~str {
    alt vs {
      ty::vstore_fixed(n) => fmt!{"%u", n},
      ty::vstore_uniq => ~"~",
      ty::vstore_box => ~"@",
      ty::vstore_slice(r) => region_to_str(cx, r)
    }
}

fn vstore_ty_to_str(cx: ctxt, ty: ~str, vs: ty::vstore) -> ~str {
    alt vs {
      ty::vstore_fixed(_) => {
        fmt!{"%s/%s", ty, vstore_to_str(cx, vs)}
      }
      _ => fmt!{"%s%s", vstore_to_str(cx, vs), ty}
    }
}

fn tys_to_str(cx: ctxt, ts: ~[t]) -> ~str {
    let mut rs = ~"";
    for ts.each |t| { rs += ty_to_str(cx, t); }
    rs
}

fn ty_to_str(cx: ctxt, typ: t) -> ~str {
    fn fn_input_to_str(cx: ctxt, input: {mode: ast::mode, ty: t}) ->
       ~str {
        let {mode, ty} = input;
        let modestr = alt canon_mode(cx, mode) {
          ast::infer(_) => ~"",
          ast::expl(m) => {
            if !ty::type_needs_infer(ty) &&
                m == ty::default_arg_mode_for_ty(ty) {
                ~""
            } else {
                mode_to_str(ast::expl(m))
            }
          }
        };
        modestr + ty_to_str(cx, ty)
    }
    fn fn_to_str(cx: ctxt, purity: ast::purity, proto: ast::proto,
                 ident: option<ast::ident>,
                 inputs: ~[arg], output: t, cf: ast::ret_style) -> ~str {
        let mut s;

        s = alt purity {
          ast::impure_fn => ~"",
          _ => purity_to_str(purity) + ~" "
        };
        s += proto_to_str(proto);
        alt ident {
          some(i) => { s += ~" "; s += *i; }
          _ => { }
        }
        s += ~"(";
        let mut strs = ~[];
        for inputs.each |a| { vec::push(strs, fn_input_to_str(cx, a)); }
        s += str::connect(strs, ~", ");
        s += ~")";
        if ty::get(output).struct != ty_nil {
            s += ~" -> ";
            alt cf {
              ast::noreturn => { s += ~"!"; }
              ast::return_val => { s += ty_to_str(cx, output); }
            }
        }
        return s;
    }
    fn method_to_str(cx: ctxt, m: method) -> ~str {
        return fn_to_str(
            cx, m.fty.purity, m.fty.proto, some(m.ident), m.fty.inputs,
            m.fty.output, m.fty.ret_style) + ~";";
    }
    fn field_to_str(cx: ctxt, f: field) -> ~str {
        return *f.ident + ~": " + mt_to_str(cx, f.mt);
    }

    // if there is an id, print that instead of the structural type:
    for ty::type_def_id(typ).each |def_id| {
        // note that this typedef cannot have type parameters
        return ast_map::path_to_str(ty::item_path(cx, def_id));
    }

    // pretty print the structural type representation:
    return alt ty::get(typ).struct {
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
        let rs = region_to_str(cx, r);
        if rs == ~"&" {
            rs + mt_to_str(cx, tm)
        } else {
            rs + ~"/" + mt_to_str(cx, tm)
        }
      }
      ty_unboxed_vec(tm) => { ~"unboxed_vec<" + mt_to_str(cx, tm) + ~">" }
      ty_type => ~"type",
      ty_rec(elems) => {
        let mut strs: ~[~str] = ~[];
        for elems.each |fld| { vec::push(strs, field_to_str(cx, fld)); }
        ~"{" + str::connect(strs, ~",") + ~"}"
      }
      ty_tup(elems) => {
        let mut strs = ~[];
        for elems.each |elem| { vec::push(strs, ty_to_str(cx, elem)); }
        ~"(" + str::connect(strs, ~",") + ~")"
      }
      ty_fn(f) => {
        fn_to_str(cx, f.purity, f.proto, none, f.inputs,
                  f.output, f.ret_style)
      }
      ty_var(v) => v.to_str(),
      ty_var_integral(v) => v.to_str(),
      ty_param({idx: id, _}) => {
        ~"'" + str::from_bytes(~[('a' as u8) + (id as u8)])
      }
      ty_self => ~"self",
      ty_enum(did, substs) | ty_class(did, substs) => {
        let path = ty::item_path(cx, did);
        let base = ast_map::path_to_str(path);
        parameterized(cx, base, substs.self_r, substs.tps)
      }
      ty_trait(did, substs) => {
        let path = ty::item_path(cx, did);
        let base = ast_map::path_to_str(path);
        parameterized(cx, base, substs.self_r, substs.tps)
      }
      ty_evec(mt, vs) => {
        vstore_ty_to_str(cx, fmt!{"[%s]", mt_to_str(cx, mt)}, vs)
      }
      ty_estr(vs) => vstore_ty_to_str(cx, ~"str", vs),
      ty_opaque_box => ~"@?",
      ty_opaque_closure_ptr(ck_block) => ~"closure&",
      ty_opaque_closure_ptr(ck_box) => ~"closure@",
      ty_opaque_closure_ptr(ck_uniq) => ~"closure~"
    }
}

fn parameterized(cx: ctxt,
                 base: ~str,
                 self_r: option<ty::region>,
                 tps: ~[ty::t]) -> ~str {

    let r_str = alt self_r {
      none => ~"",
      some(r) => {
        fmt!{"/%s", region_to_str(cx, r)}
      }
    };

    if vec::len(tps) > 0u {
        let strs = vec::map(tps, |t| ty_to_str(cx, t) );
        fmt!{"%s%s<%s>", base, r_str, str::connect(strs, ~",")}
    } else {
        fmt!{"%s%s", base, r_str}
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
