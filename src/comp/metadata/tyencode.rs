// Type encoding

import std::istr;
import std::io;
import std::map::hashmap;
import std::option::some;
import std::option::none;
import std::int;
import std::uint;
import syntax::ast::*;
import middle::ty;
import syntax::print::pprust::*;
import util::common;

export ctxt;
export ty_abbrev;
export ac_no_abbrevs;
export ac_use_abbrevs;
export enc_ty;

type ctxt =
     // Def -> str Callback:
     // The type context.
     {ds: fn(&def_id) -> istr, tcx: ty::ctxt, abbrevs: abbrev_ctxt};

// Compact string representation for ty.t values. API ty_str & parse_from_str.
// Extra parameters are for converting to/from def_ids in the string rep.
// Whatever format you choose should not contain pipe characters.
type ty_abbrev = {pos: uint, len: uint, s: @istr};

tag abbrev_ctxt { ac_no_abbrevs; ac_use_abbrevs(hashmap<ty::t, ty_abbrev>); }

fn cx_uses_abbrevs(cx: &@ctxt) -> bool {
    alt cx.abbrevs {
      ac_no_abbrevs. { ret false; }
      ac_use_abbrevs(_) { ret true; }
    }
}

fn enc_ty(w: &io::writer, cx: &@ctxt, t: ty::t) {
    alt cx.abbrevs {
      ac_no_abbrevs. {
        let result_str: @istr;
        alt cx.tcx.short_names_cache.find(t) {
          some(s) { result_str = s; }
          none. {
            let sw = io::string_writer();
            enc_sty(sw.get_writer(), cx, ty::struct(cx.tcx, t));
            result_str = @sw.get_str();
            cx.tcx.short_names_cache.insert(t, result_str);
          }
        }
        w.write_str(*result_str);
      }
      ac_use_abbrevs(abbrevs) {
        alt abbrevs.find(t) {
          some(a) { w.write_str(*a.s); ret; }
          none. {
            let pos = w.get_buf_writer().tell();
            enc_sty(w, cx, ty::struct(cx.tcx, t));
            let end = w.get_buf_writer().tell();
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

                let s =
                    ~"#" + uint::to_str(pos, 16u) + ~":" +
                    uint::to_str(len, 16u) + ~"#";
                let a = {pos: pos, len: len, s: @s};
                abbrevs.insert(t, a);
            }
            ret;
          }
        }
      }
    }
}
fn enc_mt(w: &io::writer, cx: &@ctxt, mt: &ty::mt) {
    alt mt.mut {
      imm. { }
      mut. { w.write_char('m'); }
      maybe_mut. { w.write_char('?'); }
    }
    enc_ty(w, cx, mt.ty);
}
fn enc_sty(w: &io::writer, cx: &@ctxt, st: &ty::sty) {
    alt st {
      ty::ty_nil. { w.write_char('n'); }
      ty::ty_bot. { w.write_char('z'); }
      ty::ty_bool. { w.write_char('b'); }
      ty::ty_int. { w.write_char('i'); }
      ty::ty_uint. { w.write_char('u'); }
      ty::ty_float. { w.write_char('l'); }
      ty::ty_machine(mach) {
        alt mach {
          ty_u8. { w.write_str(~"Mb"); }
          ty_u16. { w.write_str(~"Mw"); }
          ty_u32. { w.write_str(~"Ml"); }
          ty_u64. { w.write_str(~"Md"); }
          ty_i8. { w.write_str(~"MB"); }
          ty_i16. { w.write_str(~"MW"); }
          ty_i32. { w.write_str(~"ML"); }
          ty_i64. { w.write_str(~"MD"); }
          ty_f32. { w.write_str(~"Mf"); }
          ty_f64. { w.write_str(~"MF"); }
        }
      }
      ty::ty_char. { w.write_char('c'); }
      ty::ty_str. { w.write_char('s'); }
      ty::ty_istr. { w.write_char('S'); }
      ty::ty_tag(def, tys) {
        w.write_str(~"t[");
        w.write_str(cx.ds(def));
        w.write_char('|');
        for t: ty::t in tys { enc_ty(w, cx, t); }
        w.write_char(']');
      }
      ty::ty_tup(ts) {
        w.write_str(~"T[");
        for t in ts { enc_ty(w, cx, t); }
        w.write_char(']');
      }
      ty::ty_box(mt) { w.write_char('@'); enc_mt(w, cx, mt); }
      ty::ty_uniq(t) { w.write_char('~'); enc_ty(w, cx, t); }
      ty::ty_ptr(mt) { w.write_char('*'); enc_mt(w, cx, mt); }
      ty::ty_vec(mt) { w.write_char('I'); enc_mt(w, cx, mt); }
      ty::ty_rec(fields) {
        w.write_str(~"R[");
        for field: ty::field in fields {
            w.write_str(field.ident);
            w.write_char('=');
            enc_mt(w, cx, field.mt);
        }
        w.write_char(']');
      }
      ty::ty_fn(proto, args, out, cf, constrs) {
        enc_proto(w, proto);
        enc_ty_fn(w, cx, args, out, cf, constrs);
      }
      ty::ty_native_fn(abi, args, out) {
        w.write_char('N');
        alt abi {
          native_abi_rust. { w.write_char('r'); }
          native_abi_rust_intrinsic. { w.write_char('i'); }
          native_abi_cdecl. { w.write_char('c'); }
          native_abi_llvm. { w.write_char('l'); }
          native_abi_x86stdcall. { w.write_char('s'); }
        }
        enc_ty_fn(w, cx, args, out, return, []);
      }
      ty::ty_obj(methods) {
        w.write_str(~"O[");
        for m: ty::method in methods {
            enc_proto(w, m.proto);
            w.write_str(m.ident);
            enc_ty_fn(w, cx, m.inputs, m.output, m.cf, m.constrs);
        }
        w.write_char(']');
      }
      ty::ty_res(def, ty, tps) {
        w.write_str(~"r[");
        w.write_str(cx.ds(def));
        w.write_char('|');
        enc_ty(w, cx, ty);
        for t: ty::t in tps { enc_ty(w, cx, t); }
        w.write_char(']');
      }
      ty::ty_var(id) {
        w.write_char('X');
        w.write_str(int::str(id));
      }
      ty::ty_native(def) {
        w.write_char('E');
        w.write_str(cx.ds(def));
        w.write_char('|');
      }
      ty::ty_param(id, k) {
        alt k {
          kind_unique. { w.write_str(~"pu"); }
          kind_shared. { w.write_str(~"ps"); }
          kind_pinned. { w.write_str(~"pp"); }
        }
        w.write_str(uint::str(id));
      }
      ty::ty_type. { w.write_char('Y'); }
      ty::ty_constr(ty, cs) {
        w.write_str(~"A[");
        enc_ty(w, cx, ty);
        for tc: @ty::type_constr in cs { enc_ty_constr(w, cx, tc); }
        w.write_char(']');
      }
    }
}
fn enc_proto(w: &io::writer, proto: proto) {
    alt proto {
      proto_iter. { w.write_char('W'); }
      proto_fn. { w.write_char('F'); }
      proto_block. { w.write_char('B'); }
    }
}

fn enc_ty_fn(w: &io::writer, cx: &@ctxt, args: &[ty::arg], out: ty::t,
             cf: &controlflow, constrs: &[@ty::constr]) {
    w.write_char('[');
    for arg: ty::arg in args {
        alt arg.mode {
          ty::mo_alias(mut) {
            w.write_char('&');
            if mut { w.write_char('m'); }
          }
          ty::mo_move. { w.write_char('-'); }
          ty::mo_val. { }
        }
        enc_ty(w, cx, arg.ty);
    }
    w.write_char(']');
    let colon = true;
    for c: @ty::constr in constrs {
        if colon {
            w.write_char(':');
            colon = false;
        } else { w.write_char(';'); }
        enc_constr(w, cx, c);
    }
    alt cf { noreturn. { w.write_char('!'); } _ { enc_ty(w, cx, out); } }

}

// FIXME less copy-and-paste
fn enc_constr(w: &io::writer, cx: &@ctxt, c: &@ty::constr) {
    w.write_str(path_to_str(c.node.path));
    w.write_char('(');
    w.write_str(cx.ds(c.node.id));
    w.write_char('|');
    let semi = false;
    for a: @constr_arg in c.node.args {
        if semi { w.write_char(';'); } else { semi = true; }
        alt a.node {
          carg_base. { w.write_char('*'); }
          carg_ident(i) { w.write_uint(i); }
          carg_lit(l) {
            w.write_str(lit_to_str(l));
          }
        }
    }
    w.write_char(')');
}

fn enc_ty_constr(w: &io::writer, cx: &@ctxt, c: &@ty::type_constr) {
    w.write_str(path_to_str(c.node.path));
    w.write_char('(');
    w.write_str(cx.ds(c.node.id));
    w.write_char('|');
    let semi = false;
    for a: @ty::ty_constr_arg in c.node.args {
        if semi { w.write_char(';'); } else { semi = true; }
        alt a.node {
          carg_base. { w.write_char('*'); }
          carg_ident(p) {
            w.write_str(path_to_str(p)); }
          carg_lit(l) {
            w.write_str(lit_to_str(l)); }
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
