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
      re_block(node_id) {
        alt cx.items.get(node_id) {
            ast_map::node_block(blk) {
                #fmt("<block at %s>", codemap::span_to_str(blk.span,
                                                           cx.sess.codemap))
            }
            _ { cx.sess.bug("re_block refers to non-block") }
        }
      }
      re_self       { "self" }
      re_inferred   { "" }
      re_param(id)  { #fmt("<P%u>", id) }    // TODO: do better than this
      re_var(id)    { #fmt("<R%u>", id) }    // TODO: do better than this
    }
}

fn mt_to_str(cx: ctxt, m: mt) -> str {
    let mstr = alt m.mutbl {
      ast::m_mutbl { "mut " }
      ast::m_imm { "" }
      ast::m_const { "const " }
    };
    ret mstr + ty_to_str(cx, m.ty);
}

fn ty_to_str(cx: ctxt, typ: t) -> str {
    fn fn_input_to_str(cx: ctxt, input: {mode: ast::mode, ty: t}) ->
       str {
        let {mode, ty} = input;
        let modestr = alt canon_mode(cx, mode) {
          ast::infer(_) { "" }
          ast::expl(m) {
            if !ty::type_has_vars(ty) &&
                m == ty::default_arg_mode_for_ty(ty) {
                ""
            } else {
                mode_to_str(ast::expl(m))
            }
          }
        };
        modestr + ty_to_str(cx, ty)
    }
    fn fn_to_str(cx: ctxt, proto: ast::proto, ident: option<ast::ident>,
                 inputs: [arg], output: t, cf: ast::ret_style,
                 constrs: [@constr]) -> str {
        let mut s = proto_to_str(proto);
        alt ident { some(i) { s += " "; s += i; } _ { } }
        s += "(";
        let mut strs = [];
        for a: arg in inputs { strs += [fn_input_to_str(cx, a)]; }
        s += str::connect(strs, ", ");
        s += ")";
        if ty::get(output).struct != ty_nil {
            s += " -> ";
            alt cf {
              ast::noreturn { s += "!"; }
              ast::return_val { s += ty_to_str(cx, output); }
            }
        }
        s += constrs_str(constrs);
        ret s;
    }
    fn method_to_str(cx: ctxt, m: method) -> str {
        ret fn_to_str(cx, m.fty.proto, some(m.ident), m.fty.inputs,
                      m.fty.output, m.fty.ret_style, m.fty.constraints) + ";";
    }
    fn field_to_str(cx: ctxt, f: field) -> str {
        ret f.ident + ": " + mt_to_str(cx, f.mt);
    }
    fn parameterized(cx: ctxt, base: str, tps: [ty::t]) -> str {
        if vec::len(tps) > 0u {
            let strs = vec::map(tps, {|t| ty_to_str(cx, t)});
            #fmt["%s<%s>", base, str::connect(strs, ",")]
        } else {
            base
        }
    }

    // if there is an id, print that instead of the structural type:
    alt ty::type_def_id(typ) {
      some(def_id) {
        let cs = ast_map::path_to_str(ty::item_path(cx, def_id));
        ret alt ty::get(typ).struct {
          ty_enum(_, tps) | ty_res(_, _, tps) | ty_iface(_, tps) |
          ty_class(_, tps) {
            parameterized(cx, cs, tps)
          }
          _ { cs }
        };
      }
      none { /* fallthrough */}
    }

    // pretty print the structural type representation:
    ret alt ty::get(typ).struct {
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
      ty_rptr(r, tm) { "&" + region_to_str(cx, r) + "." + mt_to_str(cx, tm) }
      ty_vec(tm) { "[" + mt_to_str(cx, tm) + "]" }
      ty_type { "type" }
      ty_rec(elems) {
        let mut strs: [str] = [];
        for fld: field in elems { strs += [field_to_str(cx, fld)]; }
        "{" + str::connect(strs, ",") + "}"
      }
      ty_tup(elems) {
        let mut strs = [];
        for elem in elems { strs += [ty_to_str(cx, elem)]; }
        "(" + str::connect(strs, ",") + ")"
      }
      ty_fn(f) {
        fn_to_str(cx, f.proto, none, f.inputs, f.output, f.ret_style,
                  f.constraints)
      }
      ty_var(v) { "<T" + int::str(v) + ">" }
      ty_param(id, _) {
        "'" + str::from_bytes([('a' as u8) + (id as u8)])
      }
      ty_enum(did, tps) | ty_res(did, _, tps) | ty_iface(did, tps) |
      ty_class(did, tps) {
        let path = ty::item_path(cx, did);
        let base = ast_map::path_to_str(path);
        parameterized(cx, base, tps)
      }
      _ { ty_to_short_str(cx, typ) }
    }
}

fn ty_to_short_str(cx: ctxt, typ: t) -> str {
    let mut s = encoder::encoded_ty(cx, typ);
    if str::len(s) >= 32u { s = str::slice(s, 0u, 32u); }
    ret s;
}

fn constr_to_str(c: @constr) -> str {
    ret path_to_str(c.node.path) +
            pprust::constr_args_to_str(pprust::uint_to_str, c.node.args);
}

fn constrs_str(constrs: [@constr]) -> str {
    let mut s = "";
    let mut colon = true;
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
