// Type encoding

import std::io;
import std::map::hashmap;
import std::option::some;
import std::option::none;
import std::uint;
import front::ast::*;
import middle::ty;
import pretty::ppaux::lit_to_str;
import util::common;

export ctxt;
export ty_abbrev;
export ac_no_abbrevs;
export ac_use_abbrevs;
export enc_ty;
export ty_str;

type ctxt =
    rec(fn(&def_id) -> str  ds, // Def -> str Callback:
        ty::ctxt tcx, // The type context.
        abbrev_ctxt abbrevs);

// Compact string representation for ty.t values. API ty_str & parse_from_str.
// Extra parameters are for converting to/from def_ids in the string rep.
// Whatever format you choose should not contain pipe characters.
type ty_abbrev = rec(uint pos, uint len, str s);

tag abbrev_ctxt { ac_no_abbrevs; ac_use_abbrevs(hashmap[ty::t, ty_abbrev]); }

fn cx_uses_abbrevs(&@ctxt cx) -> bool {
    alt (cx.abbrevs) {
        case (ac_no_abbrevs) { ret false; }
        case (ac_use_abbrevs(_)) { ret true; }
    }
}
fn ty_str(&@ctxt cx, &ty::t t) -> str {
    assert (!cx_uses_abbrevs(cx));
    auto sw = io::string_writer();
    enc_ty(sw.get_writer(), cx, t);
    ret sw.get_str();
}
fn enc_ty(&io::writer w, &@ctxt cx, &ty::t t) {
    alt (cx.abbrevs) {
        case (ac_no_abbrevs) {
            auto result_str;
            alt (cx.tcx.short_names_cache.find(t)) {
                case (some(?s)) { result_str = s; }
                case (none) {
                    auto sw = io::string_writer();
                    enc_sty(sw.get_writer(), cx, ty::struct(cx.tcx, t));
                    result_str = sw.get_str();
                    cx.tcx.short_names_cache.insert(t, result_str);
                }
            }
            w.write_str(result_str);
        }
        case (ac_use_abbrevs(?abbrevs)) {
            alt (abbrevs.find(t)) {
                case (some(?a)) { w.write_str(a.s); ret; }
                case (none) {
                    auto pos = w.get_buf_writer().tell();
                    enc_sty(w, cx, ty::struct(cx.tcx, t));
                    auto end = w.get_buf_writer().tell();
                    auto len = end - pos;
                    fn estimate_sz(uint u) -> uint {
                        auto n = u;
                        auto len = 0u;
                        while (n != 0u) { len += 1u; n = n >> 4u; }
                        ret len;
                    }
                    auto abbrev_len =
                        3u + estimate_sz(pos) + estimate_sz(len);
                    if (abbrev_len < len) {
                        // I.e. it's actually an abbreviation.

                        auto s =
                            "#" + uint::to_str(pos, 16u) + ":" +
                            uint::to_str(len, 16u) + "#";
                        auto a = rec(pos=pos, len=len, s=s);
                        abbrevs.insert(t, a);
                    }
                    ret;
                }
            }
        }
    }
}
fn enc_mt(&io::writer w, &@ctxt cx, &ty::mt mt) {
    alt (mt.mut) {
        case (imm) { }
        case (mut) { w.write_char('m'); }
        case (maybe_mut) { w.write_char('?'); }
    }
    enc_ty(w, cx, mt.ty);
}
fn enc_sty(&io::writer w, &@ctxt cx, &ty::sty st) {
    alt (st) {
        case (ty::ty_nil) { w.write_char('n'); }
        case (ty::ty_bot) { w.write_char('z'); }
        case (ty::ty_bool) { w.write_char('b'); }
        case (ty::ty_int) { w.write_char('i'); }
        case (ty::ty_uint) { w.write_char('u'); }
        case (ty::ty_float) { w.write_char('l'); }
        case (ty::ty_machine(?mach)) {
            alt (mach) {
                case (common::ty_u8) { w.write_str("Mb"); }
                case (common::ty_u16) { w.write_str("Mw"); }
                case (common::ty_u32) { w.write_str("Ml"); }
                case (common::ty_u64) { w.write_str("Md"); }
                case (common::ty_i8) { w.write_str("MB"); }
                case (common::ty_i16) { w.write_str("MW"); }
                case (common::ty_i32) { w.write_str("ML"); }
                case (common::ty_i64) { w.write_str("MD"); }
                case (common::ty_f32) { w.write_str("Mf"); }
                case (common::ty_f64) { w.write_str("MF"); }
            }
        }
        case (ty::ty_char) { w.write_char('c'); }
        case (ty::ty_str) { w.write_char('s'); }
        case (ty::ty_istr) { w.write_char('S'); }
        case (ty::ty_tag(?def, ?tys)) {
            w.write_str("t[");
            w.write_str(cx.ds(def));
            w.write_char('|');
            for (ty::t t in tys) { enc_ty(w, cx, t); }
            w.write_char(']');
        }
        case (ty::ty_box(?mt)) { w.write_char('@'); enc_mt(w, cx, mt); }
        case (ty::ty_ptr(?mt)) { w.write_char('*'); enc_mt(w, cx, mt); }
        case (ty::ty_vec(?mt)) { w.write_char('V'); enc_mt(w, cx, mt); }
        case (ty::ty_ivec(?mt)) { w.write_char('I'); enc_mt(w, cx, mt); }
        case (ty::ty_port(?t)) { w.write_char('P'); enc_ty(w, cx, t); }
        case (ty::ty_chan(?t)) { w.write_char('C'); enc_ty(w, cx, t); }
        case (ty::ty_tup(?mts)) {
            w.write_str("T[");
            for (ty::mt mt in mts) { enc_mt(w, cx, mt); }
            w.write_char(']');
        }
        case (ty::ty_rec(?fields)) {
            w.write_str("R[");
            for (ty::field field in fields) {
                w.write_str(field.ident);
                w.write_char('=');
                enc_mt(w, cx, field.mt);
            }
            w.write_char(']');
        }
        case (ty::ty_fn(?proto, ?args, ?out, ?cf, ?constrs)) {
            enc_proto(w, proto);
            enc_ty_fn(w, cx, args, out, cf, constrs);
        }
        case (ty::ty_native_fn(?abi, ?args, ?out)) {
            w.write_char('N');
            alt (abi) {
                case (native_abi_rust) { w.write_char('r'); }
                case (native_abi_rust_intrinsic) {
                    w.write_char('i');
                }
                case (native_abi_cdecl) { w.write_char('c'); }
                case (native_abi_llvm) { w.write_char('l'); }
            }
            enc_ty_fn(w, cx, args, out, return, []);
        }
        case (ty::ty_obj(?methods)) {
            w.write_str("O[");
            for (ty::method m in methods) {
                enc_proto(w, m.proto);
                w.write_str(m.ident);
                enc_ty_fn(w, cx, m.inputs, m.output, m.cf, m.constrs);
            }
            w.write_char(']');
        }
        case (ty::ty_res(?def, ?ty, ?tps)) {
            w.write_str("r[");
            w.write_str(cx.ds(def));
            w.write_char('|');
            enc_ty(w, cx, ty);
            for (ty::t t in tps) { enc_ty(w, cx, t); }
            w.write_char(']');
        }
        case (ty::ty_var(?id)) {
            w.write_char('X');
            w.write_str(common::istr(id));
        }
        case (ty::ty_native(?def)) {
            w.write_char('E');
            w.write_str(cx.ds(def));
            w.write_char('|');
        }
        case (ty::ty_param(?id)) {
            w.write_char('p');
            w.write_str(common::uistr(id));
        }
        case (ty::ty_type) { w.write_char('Y'); }
        case (ty::ty_task) { w.write_char('a'); }
    }
}
fn enc_proto(&io::writer w, proto proto) {
    alt (proto) {
        case (proto_iter) { w.write_char('W'); }
        case (proto_fn) { w.write_char('F'); }
    }
}
fn enc_ty_fn(&io::writer w, &@ctxt cx, &ty::arg[] args, &ty::t out,
             &controlflow cf, &vec[@ty::constr_def] constrs) {
    w.write_char('[');
    for (ty::arg arg in args) {
        alt (arg.mode) {
            case (ty::mo_alias(?mut)) {
                w.write_char('&');
                if (mut) { w.write_char('m'); }
            }
            case (ty::mo_val) { }
        }
        enc_ty(w, cx, arg.ty);
    }
    w.write_char(']');
    auto colon = true;
    for (@ty::constr_def c in constrs) {
        if (colon) {
            w.write_char(':');
            colon = false;
        } else { w.write_char(';'); }
        enc_constr(w, cx, c);
    }
    alt (cf) {
        case (noreturn) { w.write_char('!'); }
        case (_) { enc_ty(w, cx, out); }
    }

}
fn enc_constr(&io::writer w, &@ctxt cx, &@ty::constr_def c) {
    w.write_str(path_to_str(c.node.path));
    w.write_char('(');
    w.write_str(cx.ds(c.node.id));
    w.write_char('|');
    auto semi = false;
    for (@constr_arg a in c.node.args) {
        if (semi) { w.write_char(';'); } else { semi = true; }
        alt (a.node) {
            case (carg_base) { w.write_char('*'); }
            case (carg_ident(?i)) { 
                w.write_uint(i);
            }
            case (carg_lit(?l)) { w.write_str(lit_to_str(l)); }
        }
    }
    w.write_char(')');
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
