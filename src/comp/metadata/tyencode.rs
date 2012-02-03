// Type encoding

import std::io;
import io::writer_util;
import std::map::hashmap;
import option::{some, none};
import syntax::ast::*;
import driver::session::session;
import middle::ty;
import syntax::print::pprust::*;

export ctxt;
export ty_abbrev;
export ac_no_abbrevs;
export ac_use_abbrevs;
export enc_ty;
export enc_bounds;

type ctxt =
    // Def -> str Callback:
    // The type context.
    {ds: fn@(def_id) -> str, tcx: ty::ctxt, abbrevs: abbrev_ctxt};

// Compact string representation for ty.t values. API ty_str & parse_from_str.
// Extra parameters are for converting to/from def_ids in the string rep.
// Whatever format you choose should not contain pipe characters.
type ty_abbrev = {pos: uint, len: uint, s: @str};

enum abbrev_ctxt { ac_no_abbrevs, ac_use_abbrevs(hashmap<ty::t, ty_abbrev>), }

fn cx_uses_abbrevs(cx: @ctxt) -> bool {
    alt cx.abbrevs {
      ac_no_abbrevs { ret false; }
      ac_use_abbrevs(_) { ret true; }
    }
}

fn enc_ty(w: io::writer, cx: @ctxt, t: ty::t) {
    alt cx.abbrevs {
      ac_no_abbrevs {
        let result_str = alt cx.tcx.short_names_cache.find(t) {
          some(s) { *s }
          none {
            let buf = io::mk_mem_buffer();
            enc_sty(io::mem_buffer_writer(buf), cx,
                    ty::struct_raw(cx.tcx, t));
            cx.tcx.short_names_cache.insert(t, @io::mem_buffer_str(buf));
            io::mem_buffer_str(buf)
          }
        };
        w.write_str(result_str);
      }
      ac_use_abbrevs(abbrevs) {
        alt abbrevs.find(t) {
          some(a) { w.write_str(*a.s); ret; }
          none {
            let pos = w.tell();
            enc_sty(w, cx, ty::struct_raw(cx.tcx, t));
            let end = w.tell();
            let len = end - pos;
            fn estimate_sz(u: uint) -> uint {
                let n = u;
                let len = 0u;
                while n != 0u { len += 1u; n = n >> 4u; }
                ret len;
            }
            let abbrev_len = 3u + estimate_sz(pos) + estimate_sz(len);
            if abbrev_len < len {
                // I.e. it's actually an abbreviation.
                let s = "#" + uint::to_str(pos, 16u) + ":" +
                    uint::to_str(len, 16u) + "#";
                let a = {pos: pos, len: len, s: @s};
                abbrevs.insert(t, a);
            }
            ret;
          }
        }
      }
    }
}
fn enc_mt(w: io::writer, cx: @ctxt, mt: ty::mt) {
    alt mt.mut {
      imm { }
      mut { w.write_char('m'); }
      maybe_mut { w.write_char('?'); }
    }
    enc_ty(w, cx, mt.ty);
}
fn enc_sty(w: io::writer, cx: @ctxt, st: ty::sty) {
    alt st {
      ty::ty_nil { w.write_char('n'); }
      ty::ty_bot { w.write_char('z'); }
      ty::ty_bool { w.write_char('b'); }
      ty::ty_int(t) {
        alt t {
          ty_i { w.write_char('i'); }
          ty_char { w.write_char('c'); }
          ty_i8 { w.write_str("MB"); }
          ty_i16 { w.write_str("MW"); }
          ty_i32 { w.write_str("ML"); }
          ty_i64 { w.write_str("MD"); }
        }
      }
      ty::ty_uint(t) {
        alt t {
          ty_u { w.write_char('u'); }
          ty_u8 { w.write_str("Mb"); }
          ty_u16 { w.write_str("Mw"); }
          ty_u32 { w.write_str("Ml"); }
          ty_u64 { w.write_str("Md"); }
        }
      }
      ty::ty_float(t) {
        alt t {
          ty_f { w.write_char('l'); }
          ty_f32 { w.write_str("Mf"); }
          ty_f64 { w.write_str("MF"); }
        }
      }
      ty::ty_str { w.write_char('S'); }
      ty::ty_enum(def, tys) {
        w.write_str("t[");
        w.write_str(cx.ds(def));
        w.write_char('|');
        for t: ty::t in tys { enc_ty(w, cx, t); }
        w.write_char(']');
      }
      ty::ty_iface(def, tys) {
        w.write_str("x[");
        w.write_str(cx.ds(def));
        w.write_char('|');
        for t: ty::t in tys { enc_ty(w, cx, t); }
        w.write_char(']');
      }
      ty::ty_tup(ts) {
        w.write_str("T[");
        for t in ts { enc_ty(w, cx, t); }
        w.write_char(']');
      }
      ty::ty_box(mt) { w.write_char('@'); enc_mt(w, cx, mt); }
      ty::ty_uniq(mt) { w.write_char('~'); enc_mt(w, cx, mt); }
      ty::ty_ptr(mt) { w.write_char('*'); enc_mt(w, cx, mt); }
      ty::ty_vec(mt) { w.write_char('I'); enc_mt(w, cx, mt); }
      ty::ty_rec(fields) {
        w.write_str("R[");
        for field: ty::field in fields {
            w.write_str(field.ident);
            w.write_char('=');
            enc_mt(w, cx, field.mt);
        }
        w.write_char(']');
      }
      ty::ty_fn(f) {
        enc_proto(w, f.proto);
        enc_ty_fn(w, cx, f);
      }
      ty::ty_res(def, ty, tps) {
        w.write_str("r[");
        w.write_str(cx.ds(def));
        w.write_char('|');
        enc_ty(w, cx, ty);
        for t: ty::t in tps { enc_ty(w, cx, t); }
        w.write_char(']');
      }
      ty::ty_var(id) { w.write_char('X'); w.write_str(int::str(id)); }
      ty::ty_param(id, did) {
        w.write_char('p');
        w.write_str(cx.ds(did));
        w.write_char('|');
        w.write_str(uint::str(id));
      }
      ty::ty_type { w.write_char('Y'); }
      ty::ty_send_type { w.write_char('y'); }
      ty::ty_opaque_closure_ptr(ty::ck_block) { w.write_str("C&"); }
      ty::ty_opaque_closure_ptr(ty::ck_box) { w.write_str("C@"); }
      ty::ty_opaque_closure_ptr(ty::ck_uniq) { w.write_str("C~"); }
      ty::ty_constr(ty, cs) {
        w.write_str("A[");
        enc_ty(w, cx, ty);
        for tc: @ty::type_constr in cs { enc_ty_constr(w, cx, tc); }
        w.write_char(']');
      }
      ty::ty_named(t, name) {
        if cx.abbrevs != ac_no_abbrevs {
            w.write_char('"');
            w.write_str(*name);
            w.write_char('"');
        }
        enc_ty(w, cx, t);
      }
    }
}
fn enc_proto(w: io::writer, proto: proto) {
    alt proto {
      proto_uniq { w.write_str("f~"); }
      proto_box { w.write_str("f@"); }
      proto_block { w.write_str("f&"); }
      proto_any { w.write_str("f*"); }
      proto_bare { w.write_str("fn"); }
    }
}

fn enc_ty_fn(w: io::writer, cx: @ctxt, ft: ty::fn_ty) {
    w.write_char('[');
    for arg: ty::arg in ft.inputs {
        alt ty::resolved_mode(cx.tcx, arg.mode) {
          by_mut_ref { w.write_char('&'); }
          by_move { w.write_char('-'); }
          by_copy { w.write_char('+'); }
          by_ref { w.write_char('='); }
          by_val { w.write_char('#'); }
        }
        enc_ty(w, cx, arg.ty);
    }
    w.write_char(']');
    let colon = true;
    for c: @ty::constr in ft.constraints {
        if colon {
            w.write_char(':');
            colon = false;
        } else { w.write_char(';'); }
        enc_constr(w, cx, c);
    }
    alt ft.ret_style {
      noreturn { w.write_char('!'); }
      _ { enc_ty(w, cx, ft.output); }
    }
}

// FIXME less copy-and-paste
fn enc_constr(w: io::writer, cx: @ctxt, c: @ty::constr) {
    w.write_str(path_to_str(c.node.path));
    w.write_char('(');
    w.write_str(cx.ds(c.node.id));
    w.write_char('|');
    let semi = false;
    for a: @constr_arg in c.node.args {
        if semi { w.write_char(';'); } else { semi = true; }
        alt a.node {
          carg_base { w.write_char('*'); }
          carg_ident(i) { w.write_uint(i); }
          carg_lit(l) { w.write_str(lit_to_str(l)); }
        }
    }
    w.write_char(')');
}

fn enc_ty_constr(w: io::writer, cx: @ctxt, c: @ty::type_constr) {
    w.write_str(path_to_str(c.node.path));
    w.write_char('(');
    w.write_str(cx.ds(c.node.id));
    w.write_char('|');
    let semi = false;
    for a: @ty::ty_constr_arg in c.node.args {
        if semi { w.write_char(';'); } else { semi = true; }
        alt a.node {
          carg_base { w.write_char('*'); }
          carg_ident(p) { w.write_str(path_to_str(p)); }
          carg_lit(l) { w.write_str(lit_to_str(l)); }
        }
    }
    w.write_char(')');
}

fn enc_bounds(w: io::writer, cx: @ctxt, bs: @[ty::param_bound]) {
    for bound in *bs {
        alt bound {
          ty::bound_send { w.write_char('S'); }
          ty::bound_copy { w.write_char('C'); }
          ty::bound_iface(tp) {
            w.write_char('I');
            enc_ty(w, cx, tp);
          }
        }
    }
    w.write_char('.');
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
