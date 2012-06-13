// Type encoding

import io::writer_util;
import std::map::hashmap;
import syntax::ast::*;
import syntax::diagnostic::span_handler;
import middle::ty;
import middle::ty::vid;
import syntax::print::pprust::*;

export ctxt;
export ty_abbrev;
export ac_no_abbrevs;
export ac_use_abbrevs;
export enc_ty;
export enc_bounds;
export enc_mode;

type ctxt = {
    diag: span_handler,
    // Def -> str Callback:
    ds: fn@(def_id) -> str,
    // The type context.
    tcx: ty::ctxt,
    reachable: fn@(node_id) -> bool,
    abbrevs: abbrev_ctxt
};

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
            let buf = io::mem_buffer();
            enc_sty(io::mem_buffer_writer(buf), cx, ty::get(t).struct);
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
            alt ty::type_def_id(t) {
              some(def_id) {
                // Do not emit node ids that map to unexported names.  Those
                // are not helpful.
                if def_id.crate != local_crate ||
                    cx.reachable(def_id.node) {
                    w.write_char('"');
                    w.write_str(cx.ds(def_id));
                    w.write_char('|');
                }
              }
              _ {}
            }
            enc_sty(w, cx, ty::get(t).struct);
            let end = w.tell();
            let len = end - pos;
            fn estimate_sz(u: uint) -> uint {
                let mut n = u;
                let mut len = 0u;
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
    alt mt.mutbl {
      m_imm { }
      m_mutbl { w.write_char('m'); }
      m_const { w.write_char('?'); }
    }
    enc_ty(w, cx, mt.ty);
}

fn enc_opt<T>(w: io::writer, t: option<T>, enc_f: fn(T)) {
    alt t {
      none { w.write_char('n') }
      some(v) {
        w.write_char('s');
        enc_f(v);
      }
    }
}

fn enc_substs(w: io::writer, cx: @ctxt, substs: ty::substs) {
    enc_opt(w, substs.self_r) { |r| enc_region(w, cx, r) }
    enc_opt(w, substs.self_ty) { |t| enc_ty(w, cx, t) }
    w.write_char('[');
    for substs.tps.each { |t| enc_ty(w, cx, t); }
    w.write_char(']');
}

fn enc_region(w: io::writer, cx: @ctxt, r: ty::region) {
    alt r {
      ty::re_bound(br) {
        w.write_char('b');
        enc_bound_region(w, br);
      }
      ty::re_free(id, br) {
        w.write_char('f');
        w.write_char('[');
        w.write_int(id);
        w.write_char('|');
        enc_bound_region(w, br);
        w.write_char(']');
      }
      ty::re_scope(nid) {
        w.write_char('s');
        w.write_int(nid);
        w.write_char('|');
      }
      ty::re_static {
        w.write_char('t');
      }
      ty::re_var(_) {
        // these should not crop up after typeck
        cx.diag.handler().bug("Cannot encode region variables");
      }
    }
}

fn enc_bound_region(w: io::writer, br: ty::bound_region) {
    alt br {
      ty::br_self { w.write_char('s') }
      ty::br_anon { w.write_char('a') }
      ty::br_named(s) {
        w.write_char('[');
        w.write_str(*s);
        w.write_char(']')
      }
    }
}

fn enc_vstore(w: io::writer, cx: @ctxt, v: ty::vstore) {
    w.write_char('/');
    alt v {
      ty::vstore_fixed(u)  {
        w.write_uint(u);
        w.write_char('|');
      }
      ty::vstore_uniq {
        w.write_char('~');
      }
      ty::vstore_box {
        w.write_char('@');
      }
      ty::vstore_slice(r) {
        w.write_char('&');
        enc_region(w, cx, r);
      }
    }
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
          ty_i8 { w.write_str("MB"/&); }
          ty_i16 { w.write_str("MW"/&); }
          ty_i32 { w.write_str("ML"/&); }
          ty_i64 { w.write_str("MD"/&); }
        }
      }
      ty::ty_uint(t) {
        alt t {
          ty_u { w.write_char('u'); }
          ty_u8 { w.write_str("Mb"/&); }
          ty_u16 { w.write_str("Mw"/&); }
          ty_u32 { w.write_str("Ml"/&); }
          ty_u64 { w.write_str("Md"/&); }
        }
      }
      ty::ty_float(t) {
        alt t {
          ty_f { w.write_char('l'); }
          ty_f32 { w.write_str("Mf"/&); }
          ty_f64 { w.write_str("MF"/&); }
        }
      }
      ty::ty_str { w.write_char('S'); }
      ty::ty_enum(def, substs) {
        w.write_str("t["/&);
        w.write_str(cx.ds(def));
        w.write_char('|');
        enc_substs(w, cx, substs);
        w.write_char(']');
      }
      ty::ty_iface(def, substs) {
        w.write_str("x["/&);
        w.write_str(cx.ds(def));
        w.write_char('|');
        enc_substs(w, cx, substs);
        w.write_char(']');
      }
      ty::ty_tup(ts) {
        w.write_str("T["/&);
        for ts.each {|t| enc_ty(w, cx, t); }
        w.write_char(']');
      }
      ty::ty_box(mt) { w.write_char('@'); enc_mt(w, cx, mt); }
      ty::ty_uniq(mt) { w.write_char('~'); enc_mt(w, cx, mt); }
      ty::ty_ptr(mt) { w.write_char('*'); enc_mt(w, cx, mt); }
      ty::ty_rptr(r, mt) {
        w.write_char('&');
        enc_region(w, cx, r);
        enc_mt(w, cx, mt);
      }
      ty::ty_evec(mt, v) {
        w.write_char('V');
        enc_mt(w, cx, mt);
        enc_vstore(w, cx, v);
      }
      ty::ty_estr(v) {
        w.write_char('v');
        enc_vstore(w, cx, v);
      }
      ty::ty_vec(mt) { w.write_char('I'); enc_mt(w, cx, mt); }
      ty::ty_unboxed_vec(mt) { w.write_char('U'); enc_mt(w, cx, mt); }
      ty::ty_rec(fields) {
        w.write_str("R["/&);
        for fields.each {|field|
            w.write_str(*field.ident);
            w.write_char('=');
            enc_mt(w, cx, field.mt);
        }
        w.write_char(']');
      }
      ty::ty_fn(f) {
        enc_ty_fn(w, cx, f);
      }
      ty::ty_res(def, ty, substs) {
        w.write_str("r["/&);
        w.write_str(cx.ds(def));
        w.write_char('|');
        enc_ty(w, cx, ty);
        enc_substs(w, cx, substs);
        w.write_char(']');
      }
      ty::ty_var(id) {
        w.write_char('X');
        w.write_uint(id.to_uint());
      }
      ty::ty_var_integral(id) {
        // TODO: should we have a different character for these? (Issue #1425)
        w.write_char('X');
        w.write_uint(id.to_uint());
        w.write_str("(integral)");
      }
      ty::ty_param(id, did) {
        w.write_char('p');
        w.write_str(cx.ds(did));
        w.write_char('|');
        w.write_str(uint::str(id));
      }
      ty::ty_self {
        w.write_char('s');
      }
      ty::ty_type { w.write_char('Y'); }
      ty::ty_opaque_closure_ptr(ty::ck_block) { w.write_str("C&"/&); }
      ty::ty_opaque_closure_ptr(ty::ck_box) { w.write_str("C@"/&); }
      ty::ty_opaque_closure_ptr(ty::ck_uniq) { w.write_str("C~"/&); }
      ty::ty_constr(ty, cs) {
        w.write_str("A["/&);
        enc_ty(w, cx, ty);
        for cs.each {|tc| enc_ty_constr(w, cx, tc); }
        w.write_char(']');
      }
      ty::ty_opaque_box { w.write_char('B'); }
      ty::ty_class(def, substs) {
          #debug("~~~~ %s", "a[");
          w.write_str("a["/&);
          let s = cx.ds(def);
          #debug("~~~~ %s", s);
          w.write_str(s);
          #debug("~~~~ %s", "|");
          w.write_char('|');
          enc_substs(w, cx, substs);
          #debug("~~~~ %s", "]");
          w.write_char(']');
      }
    }
}
fn enc_proto(w: io::writer, proto: proto) {
    alt proto {
      proto_uniq { w.write_str("f~"/&); }
      proto_box { w.write_str("f@"/&); }
      proto_block { w.write_str("f&"); }
      proto_any { w.write_str("f*"/&); }
      proto_bare { w.write_str("fn"/&); }
    }
}

fn enc_mode(w: io::writer, cx: @ctxt, m: mode) {
    alt ty::resolved_mode(cx.tcx, m) {
      by_mutbl_ref { w.write_char('&'); }
      by_move { w.write_char('-'); }
      by_copy { w.write_char('+'); }
      by_ref { w.write_char('='); }
      by_val { w.write_char('#'); }
    }
}

fn enc_purity(w: io::writer, p: purity) {
    alt p {
      pure_fn { w.write_char('p'); }
      impure_fn { w.write_char('i'); }
      unsafe_fn { w.write_char('u'); }
      crust_fn { w.write_char('c'); }
    }
}

fn enc_ty_fn(w: io::writer, cx: @ctxt, ft: ty::fn_ty) {
    enc_proto(w, ft.proto);
    enc_purity(w, ft.purity);
    w.write_char('[');
    for ft.inputs.each {|arg|
        enc_mode(w, cx, arg.mode);
        enc_ty(w, cx, arg.ty);
    }
    w.write_char(']');
    let mut colon = true;
    for ft.constraints.each {|c|
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

fn enc_constr_gen<T>(w: io::writer, cx: @ctxt,
                  c: @ty::constr_general<T>,
                  write_arg: fn(@sp_constr_arg<T>)) {
    w.write_str(path_to_str(c.node.path));
    w.write_char('(');
    w.write_str(cx.ds(c.node.id));
    w.write_char('|');
    let mut semi = false;
    for c.node.args.each {|a|
        if semi { w.write_char(';'); } else { semi = true; }
        write_arg(a);
    }
    w.write_char(')');
}

fn enc_constr(w: io::writer, cx: @ctxt, c: @ty::constr) {
    enc_constr_gen(w, cx, c, {|a|
      alt a.node {
        carg_base     { w.write_char('*'); }
        carg_ident(i) { w.write_uint(i); }
        carg_lit(l)   { w.write_str(lit_to_str(l)); }
      }
    });
}

fn enc_ty_constr(w: io::writer, cx: @ctxt, c: @ty::type_constr) {
    enc_constr_gen(w, cx, c, {|a|
      alt a.node {
        carg_base     { w.write_char('*'); }
        carg_ident(p) { w.write_str(path_to_str(p)); }
        carg_lit(l)  { w.write_str(lit_to_str(l)); }
      }
    });
}

fn enc_bounds(w: io::writer, cx: @ctxt, bs: @[ty::param_bound]) {
    for vec::each(*bs) {|bound|
        alt bound {
          ty::bound_send { w.write_char('S'); }
          ty::bound_copy { w.write_char('C'); }
          ty::bound_const { w.write_char('K'); }
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
