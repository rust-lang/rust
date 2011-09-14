import std::{vec, str, int, option};
import std::option::{none, some};
import middle::ty;
import middle::ty::*;
import metadata::encoder;
import syntax::print::{pp, pprust};
import syntax::print::pprust::{path_to_str, constr_args_to_str, proto_to_str};
import pp::{word, eof, zerobreak, hardbreak};
import syntax::ast_util::ty_mach_to_str;
import syntax::ast;
import middle::ast_map;
import metadata::csearch;

fn mode_str(m: ty::mode) -> str {
    alt m {
      ast::by_ref. { "" }
      ast::by_mut_ref. { "&" }
      ast::by_move. { "-" }
    }
}

fn mode_str_1(m: ty::mode) -> str {
    alt m { ast::by_ref. { "ref" } _ { mode_str(m) } }
}

fn fn_ident_to_string(id: ast::node_id, i: ast::fn_ident) -> str {
    ret alt i { none. { "anon" + int::str(id) } some(s) { s } };
}

fn get_id_ident(cx: ctxt, id: ast::def_id) -> str {
    if id.crate != ast::local_crate {
        alt cx.ext_map.find(id) {
          some(j) { str::connect(j, "::") }
          none. { "<#" + int::str(id.crate) + ":" + int::str(id.node) + ">" }
        }
    } else {
        alt cx.items.find(id.node) {
          some(ast_map::node_item(it)) { it.ident }
          _ { fail "get_id_ident: can't find item in ast_map" }
        }
    }
}

fn ty_to_str(cx: ctxt, typ: t) -> str {
    fn fn_input_to_str(cx: ctxt, input: {mode: middle::ty::mode, ty: t}) ->
       str {
        let s = mode_str(input.mode);
        ret s + ty_to_str(cx, input.ty);
    }
    fn fn_to_str(cx: ctxt, proto: ast::proto, ident: option::t<ast::ident>,
                 inputs: [arg], output: t, cf: ast::ret_style,
                 constrs: [@constr]) -> str {
        let s = proto_to_str(proto);
        alt ident { some(i) { s += " "; s += i; } _ { } }
        s += "(";
        let strs = [];
        for a: arg in inputs { strs += [fn_input_to_str(cx, a)]; }
        s += str::connect(strs, ", ");
        s += ")";
        if struct(cx, output) != ty_nil {
            s += " -> ";
            alt cf {
              ast::noreturn. { s += "!"; }
              ast::return_ref(mut) {
                s += mut ? "&!" : "&";
                s += ty_to_str(cx, output);
              }
              ast::return_val. { s += ty_to_str(cx, output); }
            }
        }
        s += constrs_str(constrs);
        ret s;
    }
    fn method_to_str(cx: ctxt, m: method) -> str {
        ret fn_to_str(cx, m.proto, some::<ast::ident>(m.ident), m.inputs,
                      m.output, m.cf, m.constrs) + ";";
    }
    fn field_to_str(cx: ctxt, f: field) -> str {
        ret f.ident + ": " + mt_to_str(cx, f.mt);
    }
    fn mt_to_str(cx: ctxt, m: mt) -> str {
        let mstr;
        alt m.mut {
          ast::mut. { mstr = "mutable "; }
          ast::imm. { mstr = ""; }
          ast::maybe_mut. { mstr = "mutable? "; }
        }
        ret mstr + ty_to_str(cx, m.ty);
    }
    alt cname(cx, typ) { some(cs) { ret cs; } _ { } }
    ret alt struct(cx, typ) {
          ty_native(_) { "native" }
          ty_nil. { "()" }
          ty_bot. { "_|_" }
          ty_bool. { "bool" }
          ty_int. { "int" }
          ty_float. { "float" }
          ty_uint. { "uint" }
          ty_machine(tm) { ty_mach_to_str(tm) }
          ty_char. { "char" }
          ty_str. { "str" }
          ty_box(tm) { "@" + mt_to_str(cx, tm) }
          ty_uniq(t) { "~" + ty_to_str(cx, t) }
          ty_vec(tm) { "[" + mt_to_str(cx, tm) + "]" }
          ty_type. { "type" }
          ty_rec(elems) {
            let strs: [str] = [];
            for fld: field in elems { strs += [field_to_str(cx, fld)]; }
            "{" + str::connect(strs, ",") + "}"
          }
          ty_tup(elems) {
            let strs = [];
            for elem in elems { strs += [ty_to_str(cx, elem)]; }
            "(" + str::connect(strs, ",") + ")"
          }
          ty_tag(id, tps) {
            let s = get_id_ident(cx, id);
            if vec::len::<t>(tps) > 0u {
                let strs: [str] = [];
                for typ: t in tps { strs += [ty_to_str(cx, typ)]; }
                s += "[" + str::connect(strs, ",") + "]";
            }
            s
          }
          ty_fn(proto, inputs, output, cf, constrs) {
            fn_to_str(cx, proto, none, inputs, output, cf, constrs)
          }
          ty_native_fn(_, inputs, output) {
            fn_to_str(cx, ast::proto_fn, none, inputs, output,
                      ast::return_val, [])
          }
          ty_obj(meths) {
            let strs = [];
            for m: method in meths { strs += [method_to_str(cx, m)]; }
            "obj {\n\t" + str::connect(strs, "\n\t") + "\n}"
          }
          ty_res(id, _, _) { get_id_ident(cx, id) }
          ty_var(v) { "<T" + int::str(v) + ">" }
          ty_param(id, _) {
            "'" + str::unsafe_from_bytes([('a' as u8) + (id as u8)])
          }
          _ { ty_to_short_str(cx, typ) }
        }
}

fn ty_to_short_str(cx: ctxt, typ: t) -> str {
    let s = encoder::encoded_ty(cx, typ);
    if str::byte_len(s) >= 32u { s = str::substr(s, 0u, 32u); }
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

fn ty_constr_to_str<Q>(c: @ast::spanned<ast::constr_general_<ast::path, Q>>)
   -> str {
    ret path_to_str(c.node.path) +
            constr_args_to_str::<ast::path>(path_to_str, c.node.args);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
