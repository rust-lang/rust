import std::map::hashmap;
import middle::ty;
import middle::ty::*;
import metadata::encoder;
import syntax::codemap;
import syntax::print::pprust;
import syntax::print::pprust::{path_to_str, constr_args_to_str, proto_to_str,
                               mode_to_str};
import syntax::{ast, ast_util};
import middle::ast_map;
import driver::session::session;

fn region_to_str(cx: ctxt, region: region) -> str {
    alt region {
      re_named(_)   { "<name>" }    // TODO: include name
      re_caller(def_id) {
        if def_id.crate == ast::local_crate {
            alt cx.items.get(def_id.node) {
              ast_map::node_item(item, path) {
                #fmt("<caller of %s::%s>", ast_map::path_to_str(*path),
                     item.ident)
              }
              _ { "<caller>" }
            }
        } else {
            "<caller>"
        }
      }
      re_block(node_id) {
        alt cx.items.get(node_id) {
            ast_map::node_block(blk) {
                #fmt("<block at %s>", codemap::span_to_str(blk.span,
                                                           cx.sess.codemap))
            }
            _ { cx.sess.bug("re_block refers to non-block") }
        }
      }
      re_self(_)    { "self" }
      re_inferred   { "" }
    }
}

iface pp_ctxt<T> {
    fn tcx() -> ty::ctxt;
    fn is_nil(T) -> bool;
    fn t_to_str(t: T) -> str;
}

fn sty_base_to_str<T,C:pp_ctxt<T>>(
    cx: C,
    typ: sty_base<T>) -> str {

    fn fn_input_to_str<T,C:pp_ctxt<T>>(
        cx: C,
        input: {mode: ast::mode, ty: T}) -> str {

        let modestr = alt canon_mode(cx.tcx(), input.mode) {
          ast::infer(_) { "" }
          ast::expl(m) { mode_to_str(ast::expl(m)) }
        };
        modestr + cx.t_to_str(input.ty)
    };

    fn fn_to_str<T,C:pp_ctxt<T>>(
        cx: C,
        proto: ast::proto,
        ident: option<ast::ident>,
        inputs: [arg_base<T>],
        output: T,
        cf: ast::ret_style,
        constrs: [@constr]) -> str {

        let s = proto_to_str(proto);
        alt ident { some(i) { s += " "; s += i; } _ { } }
        s += "(";
        let strs = [];
        for a in inputs { strs += [fn_input_to_str(cx, a)]; }
        s += str::connect(strs, ", ");
        s += ")";
        if !cx.is_nil(output) {
            s += " -> ";
            alt cf {
              ast::noreturn { s += "!"; }
              ast::return_val { s += cx.t_to_str(output); }
            }
        }
        s += constrs_str(constrs);
        ret s;
    };

    fn mt_to_str<T,C:pp_ctxt<T>>(
        cx: C,
        m: mt_base<T>) -> str {

        let mstr = alt m.mutbl {
          ast::m_mutbl { "mut " }
          ast::m_imm { "" }
          ast::m_const { "const " }
        };
        ret mstr + cx.t_to_str(m.ty);
    };

    fn field_to_str<T,C:pp_ctxt<T>>(
        cx: C,
        f: field_base<T>) -> str {

        ret f.ident + ": " + mt_to_str(cx, f.mt);
    };

    fn parameterized<T,C:pp_ctxt<T>>(
        cx: C,
        base: str,
        tps: [T]) -> str {

        if vec::len(tps) > 0u {
            let strs = vec::map(tps, cx.t_to_str);
            #fmt["%s<%s>", base, str::connect(strs, ",")]
        } else {
            base
        }
    };

    ret alt typ {
      ty_nil { "()" }
      ty_bot { "_|_" }
      ty_bool { "bool" }
      ty_int(ast::ty_i) { "int" }
      ty_int(ast::ty_char) { "char" }
      ty_int(t) { ast_util::int_ty_to_str(t) }
      ty_uint(ast::ty_u) { "uint" }
      ty_uint(t) { ast_util::uint_ty_to_str(t) }
      ty_float(ast::ty_f) { "float" }
      ty_float(t) { ast_util::float_ty_to_str(t) }
      ty_str { "str" }
      ty_self(ts) { parameterized(cx, "self", ts) }
      ty_box(tm) { "@" + mt_to_str(cx, tm) }
      ty_uniq(tm) { "~" + mt_to_str(cx, tm) }
      ty_ptr(tm) { "*" + mt_to_str(cx, tm) }
      ty_rptr(r, tm) { "&" + region_to_str(cx.tcx(), r) + "." + mt_to_str(cx, tm) }
      ty_vec(tm) { "[" + mt_to_str(cx, tm) + "]" }
      ty_type { "type" }
      ty_rec(flds) {
        let strs = vec::map(flds) {|f| field_to_str(cx, f) };
        "{" + str::connect(strs, ",") + "}"
      }
      ty_tup(elems) {
        let strs = vec::map(elems, cx.t_to_str);
        "(" + str::connect(strs, ",") + ")"
      }
      ty_fn(f) {
        fn_to_str(cx, f.proto, none, f.inputs, f.output, f.ret_style, f.constraints)
      }
      ty_param(id, _) {
        "'" + str::from_bytes([('a' as u8) + (id as u8)])
      }
      ty_enum(did, tps) | ty_res(did, _, tps) | ty_iface(did, tps) |
      ty_class(did, tps) {
        let path = ty::item_path(cx.tcx(), did);
        let base = ast_map::path_to_str(path);
        parameterized(cx, base, tps)
      }
      ty_constr(t, ts) { #fmt["%s:...", cx.t_to_str(t)] }
      ty_opaque_box { "@?" }
      ty_opaque_closure_ptr(ty::ck_block) { "fn&(...)" }
      ty_opaque_closure_ptr(ty::ck_box) { "fn@(...)" }
      ty_opaque_closure_ptr(ty::ck_uniq) { "fn~(...)" }
    }
}

type ti_pp_ctxt = {
    vb: @var_bindings,
    mut visited: [int]
};

impl of pp_ctxt<t_i> for ti_pp_ctxt {
    fn tcx() -> ctxt {
        self.vb.tcx
    }

    fn is_nil(&&t: ty::t_i) -> bool {
        alt t {
          @ty::sty_i(ty_nil) { true }
          _ { false }
        }
    }

    fn t_to_str(&&t: ty::t_i) -> str {
        alt t {
          @ty::ty_var_i(vid) if vec::contains(self.visited, vid) {
            #fmt["<T%d>", vid]
          }
          @ty::ty_var_i(vid) {
            unify::get_var_binding(
                self.vb, vid,
                {|vid| #fmt["<T%d>", vid] }, //...if unbound
                {|sty| // ...if bound
                    self.visited += [vid];
                    sty_base_to_str(self, sty)
                })
          }
          @ty::sty_i(sty) {
            sty_base_to_str(self, sty)
          }
        }
    }
}

fn ty_i_to_str(vb: @var_bindings,
               ti: t_i) -> str {
    let cx: ti_pp_ctxt = {vb: vb, mut visited: []};
    cx.t_to_str(ti)
}

impl of pp_ctxt<t> for ctxt {
    fn tcx() -> ctxt { self }

    fn is_nil(&&t: t) -> bool {
        type_is_nil(t)
    }

    fn t_to_str(&&t: t) -> str {
        sty_base_to_str(self, ty::get(t).struct)
    }
}

fn ty_to_str(cx: ctxt, typ: t) -> str {
    ret cx.t_to_str(typ);
}

fn ty_to_short_str(cx: ctxt, typ: t) -> str {
    let s = encoder::encoded_ty(cx, typ);
    if str::len(s) >= 32u { s = str::slice(s, 0u, 32u); }
    ret s;
}

fn constr_to_str(c: @constr) -> str {
    ret path_to_str(c.node.path) +
            pprust::constr_args_to_str(pprust::uint_to_str, c.node.args);
}

fn constrs_str(constrs: [@constr]) -> str {
    let s = "";
    let colon = true;
    for c: @constr in constrs {
        if colon { s += " : "; colon = false; } else { s += ", "; }
        s += constr_to_str(c);
    }
    ret s;
}

fn ty_constr_to_str<Q>(c: @ast::spanned<ast::constr_general_<@ast::path, Q>>)
   -> str {
    ret path_to_str(c.node.path) +
            constr_args_to_str::<@ast::path>(path_to_str, c.node.args);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
