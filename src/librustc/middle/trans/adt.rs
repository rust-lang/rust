// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::libc::c_ulonglong;
use core::option::{Option, Some, None};
use core::vec;
use lib::llvm::{ValueRef, TypeRef, True, False};
use middle::trans::_match;
use middle::trans::build::*;
use middle::trans::common::*;
use middle::trans::machine;
use middle::trans::type_of;
use middle::ty;
use syntax::ast;
use util::ppaux::ty_to_str;


// XXX: should this be done with boxed traits instead of ML-style?
pub enum Repr {
    Unit(int),
    CEnum(int, int), /* discriminant range */
    Univariant(Struct, Destructor),
    General(~[Struct])
}

enum Destructor {
    DtorPresent,
    DtorAbsent,
    NoDtor
}

struct Struct {
    size: u64,
    align: u64,
    fields: ~[ty::t]
}


pub fn represent_node(bcx: block, node: ast::node_id)
    -> Repr {
    represent_type(bcx.ccx(), node_id_type(bcx, node))
}

pub fn represent_type(cx: @CrateContext, t: ty::t) -> Repr {
    debug!("Representing: %s", ty_to_str(cx.tcx, t));
    // XXX: cache this
    match ty::get(t).sty {
        ty::ty_tup(ref elems) => {
            Univariant(mk_struct(cx, *elems), NoDtor)
        }
        ty::ty_rec(ref fields) => {
            // XXX: Are these in the right order?
            Univariant(mk_struct(cx, fields.map(|f| f.mt.ty)), DtorAbsent)
        }
        ty::ty_struct(def_id, ref substs) => {
            let fields = ty::lookup_struct_fields(cx.tcx, def_id);
            let dt = ty::ty_dtor(cx.tcx, def_id).is_present();
            Univariant(mk_struct(cx, fields.map(|field| {
                ty::lookup_field_type(cx.tcx, def_id, field.id, substs)
            })), if dt { DtorPresent } else { DtorAbsent })
        }
        ty::ty_enum(def_id, ref substs) => {
            struct Case { discr: int, tys: ~[ty::t] };

            let cases = do ty::enum_variants(cx.tcx, def_id).map |vi| {
                let arg_tys = do vi.args.map |&raw_ty| {
                    ty::subst(cx.tcx, substs, raw_ty)
                };
                Case { discr: vi.disr_val, tys: arg_tys }
            };
            if cases.len() == 0 {
                // Uninhabitable; represent as unit
                Univariant(mk_struct(cx, ~[]), NoDtor)
            } else if cases.len() == 1 && cases[0].tys.len() == 0 {
                Unit(cases[0].discr)
            } else if cases.len() == 1 {
                // struct, tuple, newtype, etc.
                assert cases[0].discr == 0;
                Univariant(mk_struct(cx, cases[0].tys), NoDtor)
            } else if cases.all(|c| c.tys.len() == 0) {
                let discrs = cases.map(|c| c.discr);
                CEnum(discrs.min(), discrs.max())
            } else {
                if !cases.alli(|i,c| c.discr == (i as int)) {
                    cx.sess.bug(fmt!("non-C-like enum %s with specified \
                                      discriminants",
                                     ty::item_path_str(cx.tcx, def_id)))
                }
                General(cases.map(|c| mk_struct(cx, c.tys)))
            }
        }
        _ => cx.sess.bug(~"adt::represent_type called on non-ADT type")
    }
}

fn mk_struct(cx: @CrateContext, tys: &[ty::t]) -> Struct {
    let lltys = tys.map(|&ty| type_of::sizing_type_of(cx, ty));
    let llty_rec = T_struct(lltys);
    Struct {
        size: machine::llsize_of_alloc(cx, llty_rec) /*bad*/as u64,
        align: machine::llalign_of_min(cx, llty_rec) /*bad*/as u64,
        fields: vec::from_slice(tys)
    }
}


pub fn sizing_fields_of(cx: @CrateContext, r: &Repr) -> ~[TypeRef] {
    generic_fields_of(cx, r, true)
}
pub fn fields_of(cx: @CrateContext, r: &Repr) -> ~[TypeRef] {
    generic_fields_of(cx, r, false)
}
fn generic_fields_of(cx: @CrateContext, r: &Repr, sizing: bool)
    -> ~[TypeRef] {
    match *r {
        Unit(*) => ~[],
        CEnum(*) => ~[T_enum_discrim(cx)],
        Univariant(ref st, dt) => {
            let f = if sizing {
                st.fields.map(|&ty| type_of::sizing_type_of(cx, ty))
            } else {
                st.fields.map(|&ty| type_of::type_of(cx, ty))
            };
            match dt {
                NoDtor => f,
                DtorAbsent => ~[T_struct(f)],
                DtorPresent => ~[T_struct(f), T_i8()]
            }
        }
        General(ref sts) => {
            ~[T_enum_discrim(cx),
              T_array(T_i8(), sts.map(|st| st.size).max() /*bad*/as uint)]
        }
    }
}

fn load_discr(bcx: block, scrutinee: ValueRef, min: int, max: int)
    -> ValueRef {
    let ptr = GEPi(bcx, scrutinee, [0, 0]);
    if max + 1 == min {
        // i.e., if the range is everything.  The lo==hi case would be
        // rejected by the LLVM verifier (it would mean either an
        // empty set, which is impossible, or the entire range of the
        // type, which is pointless).
        Load(bcx, ptr)
    } else {
        // llvm::ConstantRange can deal with ranges that wrap around,
        // so an overflow on (max + 1) is fine.
        LoadRangeAssert(bcx, ptr, min as c_ulonglong,
                        (max + 1) as c_ulonglong,
                        /* signed: */ True)
    }
}

pub fn trans_switch(bcx: block, r: &Repr, scrutinee: ValueRef)
    -> (_match::branch_kind, Option<ValueRef>) {
    match *r {
        CEnum(*) | General(*) => {
            (_match::switch, Some(trans_cast_to_int(bcx, r, scrutinee)))
        }
        Unit(*) | Univariant(*) => {
            (_match::single, None)
        }
    }
}

pub fn trans_cast_to_int(bcx: block, r: &Repr, scrutinee: ValueRef)
    -> ValueRef {
    match *r {
        Unit(the_disc) => C_int(bcx.ccx(), the_disc),
        CEnum(min, max) => load_discr(bcx, scrutinee, min, max),
        Univariant(*) => bcx.ccx().sess.bug(~"type has no explicit \
                                              discriminant"),
        General(ref cases) => load_discr(bcx, scrutinee, 0,
                                         (cases.len() - 1) as int)
    }
}

pub fn trans_case(bcx: block, r: &Repr, discr: int) -> _match::opt_result {
    match *r {
        CEnum(*) => {
            _match::single_result(rslt(bcx, C_int(bcx.ccx(), discr)))
        }
        Unit(*) | Univariant(*)=> {
            bcx.ccx().sess.bug(~"no cases for univariants or structs")
        }
        General(*) => {
            _match::single_result(rslt(bcx, C_int(bcx.ccx(), discr)))
        }
    }
}

pub fn trans_set_discr(bcx: block, r: &Repr, val: ValueRef, discr: int) {
    match *r {
        Unit(the_discr) => {
            assert discr == the_discr;
        }
        CEnum(min, max) => {
            assert min <= discr && discr <= max;
            Store(bcx, C_int(bcx.ccx(), discr), GEPi(bcx, val, [0, 0]))
        }
        Univariant(_, DtorPresent) => {
            assert discr == 0;
            Store(bcx, C_u8(1), GEPi(bcx, val, [0, 1]))
        }
        Univariant(*) => {
            assert discr == 0;
        }
        General(*) => {
            Store(bcx, C_int(bcx.ccx(), discr), GEPi(bcx, val, [0, 0]))
        }
    }
}

pub fn num_args(r: &Repr, discr: int) -> uint {
    match *r {
        Unit(*) | CEnum(*) => 0,
        Univariant(ref st, _dt) => { assert discr == 0; st.fields.len() }
        General(ref cases) => cases[discr as uint].fields.len()
    }
}

pub fn trans_GEP(bcx: block, r: &Repr, val: ValueRef, discr: int, ix: uint)
    -> ValueRef {
    // Note: if this ever needs to generate conditionals (e.g., if we
    // decide to do some kind of cdr-coding-like non-unique repr
    // someday), it'll need to return a possibly-new bcx as well.
    match *r {
        Unit(*) | CEnum(*) => {
            bcx.ccx().sess.bug(~"element access in C-like enum")
        }
        Univariant(ref st, dt) => {
            assert discr == 0;
            let val = match dt {
                NoDtor => val,
                DtorPresent | DtorAbsent => GEPi(bcx, val, [0, 0])
            };
            struct_GEP(bcx, st, val, ix, false)
        }
        General(ref cases) => {
            struct_GEP(bcx, &cases[discr as uint],
                       GEPi(bcx, val, [0, 1]), ix, true)
        }
    }
}

fn struct_GEP(bcx: block, st: &Struct, val: ValueRef, ix: uint,
              needs_cast: bool) -> ValueRef {
    let ccx = bcx.ccx();

    let val = if needs_cast {
        let real_llty = T_struct(st.fields.map(
            |&ty| type_of::type_of(ccx, ty)));
        PointerCast(bcx, val, T_ptr(real_llty))
    } else {
        val
    };

    GEPi(bcx, val, [0, ix])
}

pub fn trans_const(ccx: @CrateContext, r: &Repr, discr: int,
                   vals: &[ValueRef]) -> ValueRef {
    match *r {
        Unit(*) => {
            C_struct(~[])
        }
        CEnum(min, max) => {
            assert vals.len() == 0;
            assert min <= discr && discr <= max;
            C_int(ccx, discr)
        }
        Univariant(ref st, dt) => {
            assert discr == 0;
            let s = C_struct(build_const_struct(ccx, st, vals));
            match dt {
                NoDtor => s,
                // The actual destructor flag doesn't need to be present.
                // But add an extra struct layer for compatibility.
                DtorPresent | DtorAbsent => C_struct(~[s])
            }
        }
        General(ref cases) => {
            let case = &cases[discr as uint];
            let max_sz = cases.map(|s| s.size).max();
            let body = build_const_struct(ccx, case, vals);

            C_struct([C_int(ccx, discr),
                      C_packed_struct([C_struct(body)]),
                      padding(max_sz - case.size)])
        }
    }
}

fn padding(size: u64) -> ValueRef {
    C_undef(T_array(T_i8(), size /*bad*/as uint))
}

fn build_const_struct(ccx: @CrateContext, st: &Struct, vals: &[ValueRef])
    -> ~[ValueRef] {
    assert vals.len() == st.fields.len();

    let mut offset = 0;
    let mut cfields = ~[];
    for st.fields.eachi |i, &ty| {
        let llty = type_of::sizing_type_of(ccx, ty);
        let type_align = machine::llalign_of_min(ccx, llty)
            /*bad*/as u64;
        let val_align = machine::llalign_of_min(ccx, val_ty(vals[i]))
            /*bad*/as u64;
        let target_offset = roundup(offset, type_align);
        offset = roundup(offset, val_align);
        if (offset != target_offset) {
            cfields.push(padding(target_offset - offset));
            offset = target_offset;
        }
        assert !is_undef(vals[i]);
        // If that assert fails, could change it to wrap in a struct?
        cfields.push(vals[i]);
    }

    return cfields;
}

#[always_inline]
fn roundup(x: u64, a: u64) -> u64 { ((x + (a - 1)) / a) * a }


pub fn const_get_discrim(ccx: @CrateContext, r: &Repr, val: ValueRef)
    -> int {
    match *r {
        Unit(discr) => discr,
        CEnum(*) => const_to_int(val) as int,
        Univariant(*) => 0,
        General(*) => const_to_int(const_get_elt(ccx, val, [0])) as int,
    }
}

pub fn const_get_element(ccx: @CrateContext, r: &Repr, val: ValueRef,
                         _discr: int, ix: uint) -> ValueRef {
    // Not to be confused with common::const_get_elt.
    match *r {
        Unit(*) | CEnum(*) => ccx.sess.bug(~"element access in C-like enum \
                                             const"),
        Univariant(_, NoDtor) => const_struct_field(ccx, val, ix),
        Univariant(*) => const_struct_field(ccx, const_get_elt(ccx, val,
                                                               [0]), ix),
        General(*) => const_struct_field(ccx, const_get_elt(ccx, val,
                                                            [1, 0]), ix)
    }
}

fn const_struct_field(ccx: @CrateContext, val: ValueRef, ix: uint)
    -> ValueRef {
    // Get the ix-th non-undef element of the struct.
    let mut real_ix = 0; // actual position in the struct
    let mut ix = ix; // logical index relative to real_ix
    let mut field;
    loop {
        loop {
            field = const_get_elt(ccx, val, [real_ix]);
            if !is_undef(field) {
                break;
            }
            real_ix = real_ix + 1;
        }
        if ix == 0 {
            return field;
        }
        ix = ix - 1;
        real_ix = real_ix + 1;
    }
}

/// Is it safe to bitcast a value to the one field of its one variant?
pub fn is_newtypeish(r: &Repr) -> bool {
    match *r {
        Univariant(ref st, DtorAbsent)
        | Univariant(ref st, NoDtor) => st.fields.len() == 1,
        _ => false
    }
}
