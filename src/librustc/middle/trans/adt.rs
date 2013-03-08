// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * # Representation of Algebraic Data Types
 *
 * This module determines how to represent enums, structs, and tuples
 * based on their monomorphized types; it is responsible both for
 * choosing a representation and translating basic operations on
 * values of those types.
 *
 * Note that the interface treats everything as a general case of an
 * enum, so structs/tuples/etc. have one pseudo-variant with
 * discriminant 0; i.e., as if they were a univariant enum.
 *
 * Having everything in one place will enable improvements to data
 * structure representation; possibilities include:
 *
 * - Aligning enum bodies correctly, which in turn makes possible SIMD
 *   vector types (which are strict-alignment even on x86) and ports
 *   to strict-alignment architectures (PowerPC, SPARC, etc.).
 *
 * - User-specified alignment (e.g., cacheline-aligning parts of
 *   concurrently accessed data structures); LLVM can't represent this
 *   directly, so we'd have to insert padding fields in any structure
 *   that might contain one and adjust GEP indices accordingly.  See
 *   issue #4578.
 *
 * - Rendering `Option<&T>` as a possibly-null `*T` instead of using
 *   an extra word (and likewise for `@T` and `~T`).  Can and probably
 *   should also apply to any enum with one empty case and one case
 *   starting with a non-null pointer (e.g., `Result<(), ~str>`).
 *
 * - Using smaller integer types for discriminants.
 *
 * - Store nested enums' discriminants in the same word.  Rather, if
 *   some variants start with enums, and those enums representations
 *   have unused alignment padding between discriminant and body, the
 *   outer enum's discriminant can be stored there and those variants
 *   can start at offset 0.  Kind of fancy, and might need work to
 *   make copies of the inner enum type cooperate, but it could help
 *   with `Option` or `Result` wrapped around another enum.
 *
 * - Tagged pointers would be neat, but given that any type can be
 *   used unboxed and any field can have pointers (including mutable)
 *   taken to it, implementing them for Rust seems difficult.
 */

use core::container::Map;
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


/// Representations.
pub enum Repr {
    /**
     * `Unit` exists only so that an enum with a single C-like variant
     * can occupy no space, for ABI compatibility with rustc from
     * before (and during) the creation of this module.  It may not be
     * worth keeping around; `CEnum` and `Univariant` cover it
     * overwise.
     */
    Unit(int),
    /// C-like enums; basically an int.
    CEnum(int, int), // discriminant range
    /// Single-case variants, and structs/tuples/records.
    Univariant(Struct, Destructor),
    /**
     * General-case enums: discriminant as int, followed by fields.
     * The fields start immediately after the discriminant, meaning
     * that they may not be correctly aligned for the platform's ABI;
     * see above.
     */
    General(~[Struct])
}

/**
 * Structs without destructors have historically had an extra layer of
 * LLVM-struct to make accessing them work the same as structs with
 * destructors.  This could probably be flattened to a boolean now
 * that this module exists.
 */
enum Destructor {
    StructWithDtor,
    StructWithoutDtor,
    NonStruct
}

/// For structs, and struct-like parts of anything fancier.
struct Struct {
    size: u64,
    align: u64,
    fields: ~[ty::t]
}

/**
 * Convenience for `represent_type`.  There should probably be more or
 * these, for places in trans where the `ty::t` isn't directly
 * available.
 */
pub fn represent_node(bcx: block, node: ast::node_id) -> @Repr {
    represent_type(bcx.ccx(), node_id_type(bcx, node))
}

/// Decides how to represent a given type.
pub fn represent_type(cx: @CrateContext, t: ty::t) -> @Repr {
    debug!("Representing: %s", ty_to_str(cx.tcx, t));
    match cx.adt_reprs.find(&t) {
        Some(repr) => return *repr,
        None => { }
    }
    let repr = @match ty::get(t).sty {
        ty::ty_tup(ref elems) => {
            Univariant(mk_struct(cx, *elems), NonStruct)
        }
        ty::ty_struct(def_id, ref substs) => {
            let fields = ty::lookup_struct_fields(cx.tcx, def_id);
            let dt = ty::ty_dtor(cx.tcx, def_id).is_present();
            Univariant(mk_struct(cx, fields.map(|field| {
                ty::lookup_field_type(cx.tcx, def_id, field.id, substs)
            })), if dt { StructWithDtor } else { StructWithoutDtor })
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
                Unit(0)
            } else if cases.len() == 1 && cases[0].tys.len() == 0 {
                // `()`-like; see comment on definition of `Unit`.
                Unit(cases[0].discr)
            } else if cases.len() == 1 {
                // Equivalent to a struct/tuple/newtype.
                fail_unless!(cases[0].discr == 0);
                Univariant(mk_struct(cx, cases[0].tys), NonStruct)
            } else if cases.all(|c| c.tys.len() == 0) {
                // All bodies empty -> intlike
                let discrs = cases.map(|c| c.discr);
                CEnum(discrs.min(), discrs.max())
            } else {
                // The general case.  Since there's at least one
                // non-empty body, explicit discriminants should have
                // been rejected by a checker before this point.
                if !cases.alli(|i,c| c.discr == (i as int)) {
                    cx.sess.bug(fmt!("non-C-like enum %s with specified \
                                      discriminants",
                                     ty::item_path_str(cx.tcx, def_id)))
                }
                General(cases.map(|c| mk_struct(cx, c.tys)))
            }
        }
        _ => cx.sess.bug(~"adt::represent_type called on non-ADT type")
    };
    cx.adt_reprs.insert(t, repr);
    return repr;
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

/**
 * Returns the fields of a struct for the given representation.
 * All nominal types are LLVM structs, in order to be able to use
 * forward-declared opaque types to prevent circularity in `type_of`.
 */
pub fn fields_of(cx: @CrateContext, r: &Repr) -> ~[TypeRef] {
    generic_fields_of(cx, r, false)
}
/// Like `fields_of`, but for `type_of::sizing_type_of` (q.v.).
pub fn sizing_fields_of(cx: @CrateContext, r: &Repr) -> ~[TypeRef] {
    generic_fields_of(cx, r, true)
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
                NonStruct => f,
                StructWithoutDtor => ~[T_struct(f)],
                StructWithDtor => ~[T_struct(f), T_i8()]
            }
        }
        General(ref sts) => {
            ~[T_enum_discrim(cx),
              T_array(T_i8(), sts.map(|st| st.size).max() /*bad*/as uint)]
        }
    }
}

/**
 * Obtain a representation of the discriminant sufficient to translate
 * destructuring; this may or may not involve the actual discriminant.
 *
 * This should ideally be less tightly tied to `_match`.
 */
pub fn trans_switch(bcx: block, r: &Repr, scrutinee: ValueRef)
    -> (_match::branch_kind, Option<ValueRef>) {
    match *r {
        CEnum(*) | General(*) => {
            (_match::switch, Some(trans_get_discr(bcx, r, scrutinee)))
        }
        Unit(*) | Univariant(*) => {
            (_match::single, None)
        }
    }
}

/// Obtain the actual discriminant of a value.
pub fn trans_get_discr(bcx: block, r: &Repr, scrutinee: ValueRef)
    -> ValueRef {
    match *r {
        Unit(the_disc) => C_int(bcx.ccx(), the_disc),
        CEnum(min, max) => load_discr(bcx, scrutinee, min, max),
        Univariant(*) => C_int(bcx.ccx(), 0),
        General(ref cases) => load_discr(bcx, scrutinee, 0,
                                         (cases.len() - 1) as int)
    }
}

/// Helper for cases where the discriminant is simply loaded.
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

/**
 * Yield information about how to dispatch a case of the
 * discriminant-like value returned by `trans_switch`.
 *
 * This should ideally be less tightly tied to `_match`.
 */
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

/**
 * Begin initializing a new value of the given case of the given
 * representation.  The fields, if any, should then be initialized via
 * `trans_field_ptr`.
 */
pub fn trans_start_init(bcx: block, r: &Repr, val: ValueRef, discr: int) {
    match *r {
        Unit(the_discr) => {
            fail_unless!(discr == the_discr);
        }
        CEnum(min, max) => {
            fail_unless!(min <= discr && discr <= max);
            Store(bcx, C_int(bcx.ccx(), discr), GEPi(bcx, val, [0, 0]))
        }
        Univariant(_, StructWithDtor) => {
            fail_unless!(discr == 0);
            Store(bcx, C_u8(1), GEPi(bcx, val, [0, 1]))
        }
        Univariant(*) => {
            fail_unless!(discr == 0);
        }
        General(*) => {
            Store(bcx, C_int(bcx.ccx(), discr), GEPi(bcx, val, [0, 0]))
        }
    }
}

/**
 * The number of fields in a given case; for use when obtaining this
 * information from the type or definition is less convenient.
 */
pub fn num_args(r: &Repr, discr: int) -> uint {
    match *r {
        Unit(*) | CEnum(*) => 0,
        Univariant(ref st, _) => { fail_unless!(discr == 0); st.fields.len() }
        General(ref cases) => cases[discr as uint].fields.len()
    }
}

/// Access a field, at a point when the value's case is known.
pub fn trans_field_ptr(bcx: block, r: &Repr, val: ValueRef, discr: int,
                       ix: uint) -> ValueRef {
    // Note: if this ever needs to generate conditionals (e.g., if we
    // decide to do some kind of cdr-coding-like non-unique repr
    // someday), it will need to return a possibly-new bcx as well.
    match *r {
        Unit(*) | CEnum(*) => {
            bcx.ccx().sess.bug(~"element access in C-like enum")
        }
        Univariant(ref st, dt) => {
            fail_unless!(discr == 0);
            let val = match dt {
                NonStruct => val,
                StructWithDtor | StructWithoutDtor => GEPi(bcx, val, [0, 0])
            };
            struct_field_ptr(bcx, st, val, ix, false)
        }
        General(ref cases) => {
            struct_field_ptr(bcx, &cases[discr as uint],
                                 GEPi(bcx, val, [0, 1]), ix, true)
        }
    }
}

fn struct_field_ptr(bcx: block, st: &Struct, val: ValueRef, ix: uint,
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

/// Access the struct drop flag, if present.
pub fn trans_drop_flag_ptr(bcx: block, r: &Repr, val: ValueRef) -> ValueRef {
    match *r {
        Univariant(_, StructWithDtor) => GEPi(bcx, val, [0, 1]),
        _ => bcx.ccx().sess.bug(~"tried to get drop flag of non-droppable \
                                  type")
    }
}

/**
 * Construct a constant value, suitable for initializing a
 * GlobalVariable, given a case and constant values for its fields.
 * Note that this may have a different LLVM type (and different
 * alignment!) from the representation's `type_of`, so it needs a
 * pointer cast before use.
 *
 * The LLVM type system does not directly support unions, and only
 * pointers can be bitcast, so a constant (and, by extension, the
 * GlobalVariable initialized by it) will have a type that can vary
 * depending on which case of an enum it is.
 *
 * To understand the alignment situation, consider `enum E { V64(u64),
 * V32(u32, u32) }` on win32.  The type should have 8-byte alignment
 * to accommodate the u64 (currently it doesn't; this is a known bug),
 * but `V32(x, y)` would have LLVM type `{i32, i32, i32}`, which is
 * 4-byte aligned.
 *
 * Currently the returned value has the same size as the type, but
 * this may be changed in the future to avoid allocating unnecessary
 * space after values of shorter-than-maximum cases.
 */
pub fn trans_const(ccx: @CrateContext, r: &Repr, discr: int,
                   vals: &[ValueRef]) -> ValueRef {
    match *r {
        Unit(*) => {
            C_struct(~[])
        }
        CEnum(min, max) => {
            fail_unless!(vals.len() == 0);
            fail_unless!(min <= discr && discr <= max);
            C_int(ccx, discr)
        }
        Univariant(ref st, dt) => {
            fail_unless!(discr == 0);
            let s = C_struct(build_const_struct(ccx, st, vals));
            match dt {
                NonStruct => s,
                // The actual destructor flag doesn't need to be present.
                // But add an extra struct layer for compatibility.
                StructWithDtor | StructWithoutDtor => C_struct(~[s])
            }
        }
        General(ref cases) => {
            let case = &cases[discr as uint];
            let max_sz = cases.map(|s| s.size).max();
            let body = build_const_struct(ccx, case, vals);

            // The unary packed struct has alignment 1 regardless of
            // its contents, so it will always be located at the
            // expected offset at runtime.
            C_struct([C_int(ccx, discr),
                      C_packed_struct([C_struct(body)]),
                      padding(max_sz - case.size)])
        }
    }
}

/**
 * Building structs is a little complicated, because we might need to
 * insert padding if a field's value is less aligned than its type.
 *
 * Continuing the example from `trans_const`, a value of type `(u32,
 * E)` should have the `E` at offset 8, but if that field's
 * initializer is 4-byte aligned then simply translating the tuple as
 * a two-element struct will locate it at offset 4, and accesses to it
 * will read the wrong memory.
 */
fn build_const_struct(ccx: @CrateContext, st: &Struct, vals: &[ValueRef])
    -> ~[ValueRef] {
    fail_unless!(vals.len() == st.fields.len());

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
        fail_unless!(!is_undef(vals[i]));
        // If that assert fails, could change it to wrap in a struct?
        // (See `const_struct_field` for why real fields must not be undef.)
        cfields.push(vals[i]);
    }

    return cfields;
}

fn padding(size: u64) -> ValueRef {
    C_undef(T_array(T_i8(), size /*bad*/as uint))
}

// XXX this utility routine should be somewhere more general
#[always_inline]
fn roundup(x: u64, a: u64) -> u64 { ((x + (a - 1)) / a) * a }

/// Get the discriminant of a constant value.  (Not currently used.)
pub fn const_get_discrim(ccx: @CrateContext, r: &Repr, val: ValueRef)
    -> int {
    match *r {
        Unit(discr) => discr,
        CEnum(*) => const_to_int(val) as int,
        Univariant(*) => 0,
        General(*) => const_to_int(const_get_elt(ccx, val, [0])) as int,
    }
}

/**
 * Extract a field of a constant value, as appropriate for its
 * representation.
 *
 * (Not to be confused with `common::const_get_elt`, which operates on
 * raw LLVM-level structs and arrays.)
 */
pub fn const_get_field(ccx: @CrateContext, r: &Repr, val: ValueRef,
                       _discr: int, ix: uint) -> ValueRef {
    match *r {
        Unit(*) | CEnum(*) => ccx.sess.bug(~"element access in C-like enum \
                                             const"),
        Univariant(_, NonStruct) => const_struct_field(ccx, val, ix),
        Univariant(*) => const_struct_field(ccx, const_get_elt(ccx, val,
                                                               [0]), ix),
        General(*) => const_struct_field(ccx, const_get_elt(ccx, val,
                                                            [1, 0]), ix)
    }
}

/// Extract field of struct-like const, skipping our alignment padding.
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
        Univariant(ref st, StructWithoutDtor)
        | Univariant(ref st, NonStruct) => st.fields.len() == 1,
        _ => false
    }
}
