import std::vec;
import std::str;
import std::int;
import std::option;
import std::option::none;
import std::option::some;
import middle::ty;
import middle::ty::*;
import metadata::encoder;
import syntax::print::pp;
import syntax::print::pprust;
import syntax::print::pprust::path_to_str;
import syntax::print::pprust::constr_args_to_str;
import syntax::print::pprust::proto_to_str;
import pp::word;
import pp::eof;
import pp::zerobreak;
import pp::hardbreak;
import ast::ty_mach_to_str;
import syntax::ast;

fn mode_str(m: &ty::mode) -> str {
    alt m {
      mo_val. { "" }
      mo_alias(false) { "&" }
      mo_alias(true) { "&mutable " }
    }
}

fn mode_str_1(m: &ty::mode) -> str {
    alt m { mo_val. { "val" } _ { mode_str(m) } }
}

fn fn_ident_to_string(id: ast::node_id, i: &ast::fn_ident) -> str {
    ret alt i { none. { "anon" + int::str(id) } some(s) { s } };
}

fn ty_to_str(cx: &ctxt, typ: &t) -> str {
    fn fn_input_to_str(cx: &ctxt, input: &{mode: middle::ty::mode, ty: t}) ->
       str {
        let s = mode_str(input.mode);
        ret s + ty_to_str(cx, input.ty);
    }
    fn fn_to_str(cx: &ctxt, proto: ast::proto, ident: option::t[ast::ident],
                 inputs: &[arg], output: t, cf: ast::controlflow,
                 constrs: &[@constr]) -> str {
        let s = proto_to_str(proto);
        alt ident { some(i) { s += " "; s += i; } _ { } }
        s += "(";
        let strs = ~[];
        for a: arg in inputs { strs += ~[fn_input_to_str(cx, a)]; }
        s += str::connect(strs, ", ");
        s += ")";
        if struct(cx, output) != ty_nil {
            alt cf {
              ast::noreturn. { s += " -> !"; }
              ast::return. { s += " -> " + ty_to_str(cx, output); }
            }
        }
        s += constrs_str(constrs);
        ret s;
    }
    fn method_to_str(cx: &ctxt, m: &method) -> str {
        ret fn_to_str(cx, m.proto, some[ast::ident](m.ident), m.inputs,
                      m.output, m.cf, m.constrs) + ";";
    }
    fn field_to_str(cx: &ctxt, f: &field) -> str {
        ret f.ident + ": " + mt_to_str(cx, f.mt);
    }
    fn mt_to_str(cx: &ctxt, m: &mt) -> str {
        let mstr;
        alt m.mut {
          ast::mut. { mstr = "mutable "; }
          ast::imm. { mstr = ""; }
          ast::maybe_mut. { mstr = "mutable? "; }
        }
        ret mstr + ty_to_str(cx, m.ty);
    }
    alt cname(cx, typ) { some(cs) { ret cs; } _ { } }
    let s = "";
    alt struct(cx, typ) {
      ty_native(_) { s += "native"; }
      ty_nil. { s += "()"; }
      ty_bot. { s += "_|_"; }
      ty_bool. { s += "bool"; }
      ty_int. { s += "int"; }
      ty_float. { s += "float"; }
      ty_uint. { s += "uint"; }
      ty_machine(tm) { s += ty_mach_to_str(tm); }
      ty_char. { s += "char"; }
      ty_str. { s += "str"; }
      ty_istr. { s += "istr"; }
      ty_box(tm) { s += "@" + mt_to_str(cx, tm); }
      ty_uniq(t) { s += "~" + ty_to_str(cx, t); }
      ty_vec(tm) { s += "vec[" + mt_to_str(cx, tm) + "]"; }
      ty_ivec(tm) { s += "[" + mt_to_str(cx, tm) + "]"; }
      ty_port(t) { s += "port[" + ty_to_str(cx, t) + "]"; }
      ty_chan(t) { s += "chan[" + ty_to_str(cx, t) + "]"; }
      ty_type. { s += "type"; }
      ty_task. { s += "task"; }
      ty_rec(elems) {
        let strs: [str] = ~[];
        for fld: field in elems { strs += ~[field_to_str(cx, fld)]; }
        s += "{" + str::connect(strs, ",") + "}";
      }
      ty_tup(elems) {
        let strs = ~[];
        for elem in elems { strs += ~[ty_to_str(cx, elem)]; }
        s += "(" + str::connect(strs, ",") + ")";
      }
      ty_tag(id, tps) {
        // The user should never see this if the cname is set properly!

        s += "<tag#" + int::str(id.crate) + ":" + int::str(id.node) + ">";
        if vec::len[t](tps) > 0u {
            let strs: [str] = ~[];
            for typ: t in tps { strs += ~[ty_to_str(cx, typ)]; }
            s += "[" + str::connect(strs, ",") + "]";
        }
      }
      ty_fn(proto, inputs, output, cf, constrs) {
        s += fn_to_str(cx, proto, none, inputs, output, cf, constrs);
      }
      ty_native_fn(_, inputs, output) {
        s +=
            fn_to_str(cx, ast::proto_fn, none, inputs, output, ast::return,
                      ~[]);
      }
      ty_obj(meths) {
        let strs = ~[];
        for m: method in meths { strs += ~[method_to_str(cx, m)]; }
        s += "obj {\n\t" + str::connect(strs, "\n\t") + "\n}";
      }
      ty_res(id, _, _) {
        s +=
            "<resource#" + int::str(id.node) + ":" + int::str(id.crate) + ">";
      }
      ty_var(v) { s += "<T" + int::str(v) + ">"; }
      ty_param(id,_) {
        s += "'" + str::unsafe_from_bytes(~[('a' as u8) + (id as u8)]);
      }
      _ { s += ty_to_short_str(cx, typ); }
    }
    ret s;
}

fn ty_to_short_str(cx: &ctxt, typ: t) -> str {
    let s = encoder::encoded_ty(cx, typ);
    if str::byte_len(s) >= 32u { s = str::substr(s, 0u, 32u); }
    ret s;
}

fn constr_to_str(c: &@constr) -> str {
    ret path_to_str(c.node.path) +
            pprust::constr_args_to_str(pprust::uint_to_str, c.node.args);
}

fn constrs_str(constrs: &[@constr]) -> str {
    let s = "";
    let colon = true;
    for c: @constr in constrs {
        if colon { s += " : "; colon = false; } else { s += ", "; }
        s += constr_to_str(c);
    }
    ret s;
}

fn ty_constr_to_str[Q](c: &@ast::spanned[ast::constr_general_[ast::path, Q]])
   -> str {
    ret path_to_str(c.node.path) +
            constr_args_to_str[ast::path](path_to_str, c.node.args);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
