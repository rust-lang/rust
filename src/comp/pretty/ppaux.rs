
import std::io;
import std::vec;
import std::str;
import std::option;
import std::option::none;
import std::option::some;
import middle::ty;
import middle::ty::*;
import front::lexer;
import front::ast;
import pp::word;
import pp::eof;
import pp::zerobreak;
import pp::hardbreak;
import front::codemap;
import front::codemap::codemap;
import util::common::istr;
import util::common::uistr;
import util::common::ty_mach_to_str;

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
        case (ty_native) { s += "native"; }
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

fn constr_arg_to_str[T](fn(&T) -> str  f, &ast::constr_arg_general_[T] c) ->
   str {
    alt (c) {
        case (ast::carg_base) { ret "*"; }
        case (ast::carg_ident(?i)) { ret f(i); }
        case (ast::carg_lit(?l)) { ret lit_to_str(l); }
    }
}

fn constr_arg_to_str_1(&front::ast::constr_arg_general_[str] c) -> str {
    alt (c) {
        case (ast::carg_base) { ret "*"; }
        case (ast::carg_ident(?i)) { ret i; }
        case (ast::carg_lit(?l)) { ret lit_to_str(l); }
    }
}

fn constr_args_to_str[T](fn(&T) -> str  f,
                         &vec[@ast::constr_arg_general[T]] args) -> str {
    auto comma = false;
    auto s = "(";
    for (@ast::constr_arg_general[T] a in args) {
        if (comma) { s += ", "; } else { comma = true; }
        s += constr_arg_to_str[T](f, a.node);
    }
    s += ")";
    ret s;
}

fn print_literal(&ps s, &@front::ast::lit lit) {
    maybe_print_comment(s, lit.span.lo);
    alt (next_lit(s)) {
        case (some(?lt)) {
            if (lt.pos == lit.span.lo) {
                word(s.s, lt.lit);
                s.cur_lit += 1u;
                ret;
            }
        }
        case (_) { }
    }
    alt (lit.node) {
        case (ast::lit_str(?st, ?kind)) {
            if (kind == ast::sk_unique) { word(s.s, "~"); }
            print_string(s, st);
        }
        case (ast::lit_char(?ch)) {
            word(s.s,
                 "'" + escape_str(str::from_bytes([ch as u8]), '\'') + "'");
        }
        case (ast::lit_int(?val)) { word(s.s, istr(val)); }
        case (ast::lit_uint(?val)) { word(s.s, uistr(val) + "u"); }
        case (ast::lit_float(?fstr)) { word(s.s, fstr); }
        case (ast::lit_mach_int(?mach, ?val)) {
            word(s.s, istr(val as int));
            word(s.s, ty_mach_to_str(mach));
        }
        case (ast::lit_mach_float(?mach, ?val)) {
            // val is already a str

            word(s.s, val);
            word(s.s, ty_mach_to_str(mach));
        }
        case (ast::lit_nil) { word(s.s, "()"); }
        case (ast::lit_bool(?val)) {
            if (val) { word(s.s, "true"); } else { word(s.s, "false"); }
        }
    }
}

fn lit_to_str(&@front::ast::lit l) -> str { be to_str(l, print_literal); }

fn next_lit(&ps s) -> option::t[lexer::lit] {
    alt (s.literals) {
        case (some(?lits)) {
            if (s.cur_lit < vec::len(lits)) {
                ret some(lits.(s.cur_lit));
            } else { ret none[lexer::lit]; }
        }
        case (_) { ret none[lexer::lit]; }
    }
}

fn maybe_print_comment(&ps s, uint pos) {
    while (true) {
        alt (next_comment(s)) {
            case (some(?cmnt)) {
                if (cmnt.pos < pos) {
                    print_comment(s, cmnt);
                    s.cur_cmnt += 1u;
                } else { break; }
            }
            case (_) { break; }
        }
    }
}

fn print_comment(&ps s, lexer::cmnt cmnt) {
    alt (cmnt.style) {
        case (lexer::mixed) {
            assert (vec::len(cmnt.lines) == 1u);
            zerobreak(s.s);
            word(s.s, cmnt.lines.(0));
            zerobreak(s.s);
        }
        case (lexer::isolated) {
            pprust::hardbreak_if_not_bol(s);
            for (str line in cmnt.lines) { word(s.s, line); hardbreak(s.s); }
        }
        case (lexer::trailing) {
            word(s.s, " ");
            if (vec::len(cmnt.lines) == 1u) {
                word(s.s, cmnt.lines.(0));
                hardbreak(s.s);
            } else {
                ibox(s, 0u);
                for (str line in cmnt.lines) {
                    word(s.s, line);
                    hardbreak(s.s);
                }
                end(s);
            }
        }
        case (lexer::blank_line) {
            // We need to do at least one, possibly two hardbreaks.
            pprust::hardbreak_if_not_bol(s);
            hardbreak(s.s);
        }
    }
}

fn print_string(&ps s, &str st) {
    word(s.s, "\"");
    word(s.s, escape_str(st, '"'));
    word(s.s, "\"");
}

fn escape_str(str st, char to_escape) -> str {
    let str out = "";
    auto len = str::byte_len(st);
    auto i = 0u;
    while (i < len) {
        alt (st.(i) as char) {
            case ('\n') { out += "\\n"; }
            case ('\t') { out += "\\t"; }
            case ('\r') { out += "\\r"; }
            case ('\\') { out += "\\\\"; }
            case (?cur) {
                if (cur == to_escape) { out += "\\"; }
                // FIXME some (or all?) non-ascii things should be escaped

                str::push_char(out, cur);
            }
        }
        i += 1u;
    }
    ret out;
}

fn to_str[T](&T t, fn(&ps, &T)  f) -> str {
    auto writer = io::string_writer();
    auto s = rust_printer(writer.get_writer());
    f(s, t);
    eof(s.s);
    ret writer.get_str();
}

fn next_comment(&ps s) -> option::t[lexer::cmnt] {
    alt (s.comments) {
        case (some(?cmnts)) {
            if (s.cur_cmnt < vec::len(cmnts)) {
                ret some(cmnts.(s.cur_cmnt));
            } else { ret none[lexer::cmnt]; }
        }
        case (_) { ret none[lexer::cmnt]; }
    }
}

type ps =
    @rec(pp::printer s,
         option::t[codemap] cm,
         option::t[vec[lexer::cmnt]] comments,
         option::t[vec[lexer::lit]] literals,
         mutable uint cur_cmnt,
         mutable uint cur_lit,
         mutable vec[pp::breaks] boxes,
         mode mode);

fn ibox(&ps s, uint u) {
    vec::push(s.boxes, pp::inconsistent);
    pp::ibox(s.s, u);
}

fn end(&ps s) { vec::pop(s.boxes); pp::end(s.s); }

tag mode { mo_untyped; mo_typed(ctxt); mo_identified; }

fn rust_printer(io::writer writer) -> ps {
    let vec[pp::breaks] boxes = [];
    ret @rec(s=pp::mk_printer(writer, default_columns),
             cm=none[codemap],
             comments=none[vec[lexer::cmnt]],
             literals=none[vec[lexer::lit]],
             mutable cur_cmnt=0u,
             mutable cur_lit=0u,
             mutable boxes=boxes,
             mode=mo_untyped);
}

const uint indent_unit = 4u;

const uint default_columns = 78u;


// needed b/c constr_args_to_str needs
// something that takes an alias
// (argh)
fn uint_to_str(&uint i) -> str { ret uistr(i); }

fn constr_to_str(&@constr_def c) -> str {
    ret path_to_str(c.node.path) +
            constr_args_to_str(uint_to_str, c.node.args);
}

fn ast_constr_to_str(&@front::ast::constr c) -> str {
    ret path_to_str(c.node.path) +
            constr_args_to_str(uint_to_str, c.node.args);
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

fn ast_constrs_str(&vec[@ast::constr] constrs) -> str {
    auto s = "";
    auto colon = true;
    for (@ast::constr c in constrs) {
        if (colon) { s += " : "; colon = false; } else { s += ", "; }
        s += ast_constr_to_str(c);
    }
    ret s;
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
