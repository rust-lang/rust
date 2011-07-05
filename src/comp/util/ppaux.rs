import std::io;
import std::vec;
import std::str;
import std::option;
import std::option::none;
import std::option::some;
import middle::ty;
import middle::ty::*;
import syntax::print::pp;
import syntax::print::pprust;
import pp::word;
import pp::eof;
import pp::zerobreak;
import pp::hardbreak;
import syntax::_std::istr;
import syntax::_std::uistr;
import ast::ty_mach_to_str;
import syntax::ast;

fn mode_str(&ty::mode m) -> str {
    alt (m) {
        case (mo_val) { "" }
        case (mo_alias(false)) { "&" }
        case (mo_alias(true)) { "&mutable " }
    }
}

fn mode_str_1(&ty::mode m) -> str {
    alt (m) {
        case (mo_val) { "val" }
        case (_)      { mode_str(m) }
    }
}

fn fn_ident_to_string(ast::node_id id, &ast::fn_ident i) -> str {
    ret alt (i) {
        case (none) { "anon" + istr(id) }
        case (some(?s)) { s }
    };
}

fn ty_to_str(&ctxt cx, &t typ) -> str {
    fn fn_input_to_str(&ctxt cx, &rec(middle::ty::mode mode, t ty) input) ->
       str {
        auto s = mode_str(input.mode);
        ret s + ty_to_str(cx, input.ty);
    }
    fn fn_to_str(&ctxt cx, ast::proto proto, option::t[ast::ident] ident,
                 &arg[] inputs, t output, ast::controlflow cf,
                 &vec[@constr_def] constrs) -> str {
        auto s;
        alt (proto) {
            case (ast::proto_iter) { s = "iter"; }
            case (ast::proto_fn) { s = "fn"; }
        }
        alt (ident) { case (some(?i)) { s += " "; s += i; } case (_) { } }
        s += "(";
        auto strs = [];
        for (arg a in inputs) { strs += [fn_input_to_str(cx, a)]; }
        s += str::connect(strs, ", ");
        s += ")";
        if (struct(cx, output) != ty_nil) {
            alt (cf) {
                case (ast::noreturn) { s += " -> !"; }
                case (ast::return) { s += " -> " + ty_to_str(cx, output); }
            }
        }
        s += constrs_str(constrs);
        ret s;
    }
    fn method_to_str(&ctxt cx, &method m) -> str {
        ret fn_to_str(cx, m.proto, some[ast::ident](m.ident), m.inputs,
                      m.output, m.cf, m.constrs) + ";";
    }
    fn field_to_str(&ctxt cx, &field f) -> str {
        ret mt_to_str(cx, f.mt) + " " + f.ident;
    }
    fn mt_to_str(&ctxt cx, &mt m) -> str {
        auto mstr;
        alt (m.mut) {
            case (ast::mut) { mstr = "mutable "; }
            case (ast::imm) { mstr = ""; }
            case (ast::maybe_mut) { mstr = "mutable? "; }
        }
        ret mstr + ty_to_str(cx, m.ty);
    }
    alt (cname(cx, typ)) { case (some(?cs)) { ret cs; } case (_) { } }
    auto s = "";
    alt (struct(cx, typ)) {
        case (ty_native(_)) { s += "native"; }
        case (ty_nil) { s += "()"; }
        case (ty_bot) { s += "_|_"; }
        case (ty_bool) { s += "bool"; }
        case (ty_int) { s += "int"; }
        case (ty_float) { s += "float"; }
        case (ty_uint) { s += "uint"; }
        case (ty_machine(?tm)) { s += ty_mach_to_str(tm); }
        case (ty_char) { s += "char"; }
        case (ty_str) { s += "str"; }
        case (ty_istr) { s += "istr"; }
        case (ty_box(?tm)) { s += "@" + mt_to_str(cx, tm); }
        case (ty_vec(?tm)) { s += "vec[" + mt_to_str(cx, tm) + "]"; }
        case (ty_ivec(?tm)) { s += "ivec[" + mt_to_str(cx, tm) + "]"; }
        case (ty_port(?t)) { s += "port[" + ty_to_str(cx, t) + "]"; }
        case (ty_chan(?t)) { s += "chan[" + ty_to_str(cx, t) + "]"; }
        case (ty_type) { s += "type"; }
        case (ty_task) { s += "task"; }
        case (ty_tup(?elems)) {
            let vec[str] strs = [];
            for (mt tm in elems) { strs += [mt_to_str(cx, tm)]; }
            s += "tup(" + str::connect(strs, ",") + ")";
        }
        case (ty_rec(?elems)) {
            let vec[str] strs = [];
            for (field fld in elems) { strs += [field_to_str(cx, fld)]; }
            s += "rec(" + str::connect(strs, ",") + ")";
        }
        case (ty_tag(?id, ?tps)) {
            // The user should never see this if the cname is set properly!

            s += "<tag#" + istr(id._0) + ":" + istr(id._1) + ">";
            if (vec::len[t](tps) > 0u) {
                auto f = bind ty_to_str(cx, _);
                auto strs = vec::map[t, str](f, tps);
                s += "[" + str::connect(strs, ",") + "]";
            }
        }
        case (ty_fn(?proto, ?inputs, ?output, ?cf, ?constrs)) {
            s += fn_to_str(cx, proto, none, inputs, output, cf, constrs);
        }
        case (ty_native_fn(_, ?inputs, ?output)) {
            s += fn_to_str(cx, ast::proto_fn, none, inputs, output,
                           ast::return, []);
        }
        case (ty_obj(?meths)) {
            auto f = bind method_to_str(cx, _);
            auto m = vec::map[method, str](f, meths);
            s += "obj {\n\t" + str::connect(m, "\n\t") + "\n}";
        }
        case (ty_res(?id, _, _)) {
            s += "<resource#" + istr(id._0) + ":" + istr(id._1) + ">";
        }
        case (ty_var(?v)) { s += "<T" + istr(v) + ">"; }
        case (ty_param(?id)) {
            s += "'" + str::unsafe_from_bytes([('a' as u8) + (id as u8)]);
        }
        case (_) { s += ty_to_short_str(cx, typ); }
    }
    ret s;
}

fn ty_to_short_str(&ctxt cx, t typ) -> str {
    auto f = def_to_str;
    auto ecx = @rec(ds=f, tcx=cx, abbrevs=metadata::tyencode::ac_no_abbrevs);
    auto s = metadata::tyencode::ty_str(ecx, typ);
    if (str::byte_len(s) >= 32u) { s = str::substr(s, 0u, 32u); }
    ret s;
}

fn constr_to_str(&@constr_def c) -> str {
    ret ast::path_to_str(c.node.path) +
        pprust::constr_args_to_str(pprust::uint_to_str, c.node.args);
}

fn constrs_str(&vec[@constr_def] constrs) -> str {
    auto s = "";
    auto colon = true;
    for (@constr_def c in constrs) {
        if (colon) { s += " : "; colon = false; } else { s += ", "; }
        s += constr_to_str(c);
    }
    ret s;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
