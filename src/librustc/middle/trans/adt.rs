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
 * - User-specified alignment (e.g., cacheline-aligning parts of
 *   concurrently accessed data structures); LLVM can't represent this
 *   directly, so we'd have to insert padding fields in any structure
 *   that might contain one and adjust GEP indices accordingly.  See
 *   issue #4578.
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

use std::container::Map;
use std::libc::c_ulonglong;
use std::option::{Option, Some, None};
use std::vec;

use lib::llvm::{ValueRef, True, IntEQ, IntNE};
use middle::trans::_match;
use middle::trans::build::*;
use middle::trans::common::*;
use middle::trans::machine;
use middle::trans::type_of;
use middle::ty;
use syntax::ast;
use util::ppaux::ty_to_str;

use middle::trans::type_::Type;


/// Representations.
pub enum Repr {
    /// C-like enums; basically an int.
    CEnum(int, int), // discriminant range
    /**
     * Single-case variants, and structs/tuples/records.
     *
     * Structs with destructors need a dynamic destroyedness flag to
     * avoid running the destructor too many times; this is included
     * in the `Struct` if present.
     */
    Univariant(Struct, bool),
    /**
     * General-case enums: for each case there is a struct, and they
     * all start with a field for the discriminant.
     */
    General(~[Struct]),
    /**
     * Two cases distinguished by a nullable pointer: the case with discriminant
     * `nndiscr` is represented by the struct `nonnull`, where the `ptrfield`th
     * field is known to be nonnull due to its type; if that field is null, then
     * it represents the other case, which is inhabited by at most one value
     * (and all other fields are undefined/unused).
     *
     * For example, `std::option::Option` instantiated at a safe pointer type
     * is represented such that `None` is a null pointer and `Some` is the
     * identity function.
     */
    NullablePointer{ nonnull: Struct, nndiscr: int, ptrfield: uint,
                     nullfields: ~[ty::t] }
}

/// For structs, and struct-like parts of anything fancier.
struct Struct {
    size: u64,
    align: u64,
    packed: bool,
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
pub fn represent_type(cx: &mut CrateContext, t: ty::t) -> @Repr {
    debug!("Representing: %s", ty_to_str(cx.tcx, t));
    match cx.adt_reprs.find(&t) {
        Some(repr) => return *repr,
        None => { }
    }
    let repr = @represent_type_uncached(cx, t);
    debug!("Represented as: %?", repr)
    cx.adt_reprs.insert(t, repr);
    return repr;
}

fn represent_type_uncached(cx: &mut CrateContext, t: ty::t) -> Repr {
    match ty::get(t).sty {
        ty::ty_tup(ref elems) => {
            return Univariant(mk_struct(cx, *elems, false), false)
        }
        ty::ty_struct(def_id, ref substs) => {
            let fields = ty::lookup_struct_fields(cx.tcx, def_id);
            let mut ftys = do fields.map |field| {
                ty::lookup_field_type(cx.tcx, def_id, field.id, substs)
            };
            let packed = ty::lookup_packed(cx.tcx, def_id);
            let dtor = ty::ty_dtor(cx.tcx, def_id).has_drop_flag();
            if dtor { ftys.push(ty::mk_bool()); }

            return Univariant(mk_struct(cx, ftys, packed), dtor)
        }
        ty::ty_enum(def_id, ref substs) => {
            struct Case { discr: int, tys: ~[ty::t] };
            impl Case {
                fn is_zerolen(&self, cx: &mut CrateContext) -> bool {
                    mk_struct(cx, self.tys, false).size == 0
                }
                fn find_ptr(&self) -> Option<uint> {
                    self.tys.iter().position_(|&ty| mono_data_classify(ty) == MonoNonNull)
                }
            }

            let cases = do ty::enum_variants(cx.tcx, def_id).map |vi| {
                let arg_tys = do vi.args.map |&raw_ty| {
                    ty::subst(cx.tcx, substs, raw_ty)
                };
                Case { discr: vi.disr_val, tys: arg_tys }
            };

            if cases.len() == 0 {
                // Uninhabitable; represent as unit
                return Univariant(mk_struct(cx, [], false), false);
            }

            if cases.iter().all(|c| c.tys.len() == 0) {
                // All bodies empty -> intlike
                let discrs = cases.map(|c| c.discr);
                return CEnum(*discrs.iter().min().unwrap(), *discrs.iter().max().unwrap());
            }

            if cases.len() == 1 {
                // Equivalent to a struct/tuple/newtype.
                assert_eq!(cases[0].discr, 0);
                return Univariant(mk_struct(cx, cases[0].tys, false), false)
            }

            // Since there's at least one
            // non-empty body, explicit discriminants should have
            // been rejected by a checker before this point.
            if !cases.iter().enumerate().all(|(i,c)| c.discr == (i as int)) {
                cx.sess.bug(fmt!("non-C-like enum %s with specified \
                                  discriminants",
                                 ty::item_path_str(cx.tcx, def_id)))
            }

            if cases.len() == 2 {
                let mut discr = 0;
                while discr < 2 {
                    if cases[1 - discr].is_zerolen(cx) {
                        match cases[discr].find_ptr() {
                            Some(ptrfield) => {
                                return NullablePointer {
                                    nndiscr: discr,
                                    nonnull: mk_struct(cx, cases[discr].tys, false),
                                    ptrfield: ptrfield,
                                    nullfields: copy cases[1 - discr].tys
                                }
                            }
                            None => { }
                        }
                    }
                    discr += 1;
                }
            }

            // The general case.
            let discr = ~[ty::mk_int()];
            return General(cases.map(|c| mk_struct(cx, discr + c.tys, false)))
        }
        _ => cx.sess.bug("adt::represent_type called on non-ADT type")
    }
}

fn mk_struct(cx: &mut CrateContext, tys: &[ty::t], packed: bool) -> Struct {
    let lltys = tys.map(|&ty| type_of::sizing_type_of(cx, ty));
    let llty_rec = Type::struct_(lltys, packed);
    Struct {
        size: machine::llsize_of_alloc(cx, llty_rec) /*bad*/as u64,
        align: machine::llalign_of_min(cx, llty_rec) /*bad*/as u64,
        packed: packed,
        fields: vec::to_owned(tys)
    }
}

/**
 * Returns the fields of a struct for the given representation.
 * All nominal types are LLVM structs, in order to be able to use
 * forward-declared opaque types to prevent circularity in `type_of`.
 */
pub fn fields_of(cx: &mut CrateContext, r: &Repr) -> ~[Type] {
    generic_fields_of(cx, r, false)
}
/// Like `fields_of`, but for `type_of::sizing_type_of` (q.v.).
pub fn sizing_fields_of(cx: &mut CrateContext, r: &Repr) -> ~[Type] {
    generic_fields_of(cx, r, true)
}
fn generic_fields_of(cx: &mut CrateContext, r: &Repr, sizing: bool) -> ~[Type] {
    match *r {
        CEnum(*) => ~[Type::enum_discrim(cx)],
        Univariant(ref st, _dtor) => struct_llfields(cx, st, sizing),
        NullablePointer{ nonnull: ref st, _ } => struct_llfields(cx, st, sizing),
        General(ref sts) => {
            // To get "the" type of a general enum, we pick the case
            // with the largest alignment (so it will always align
            // correctly in containing structures) and pad it out.
            assert!(sts.len() >= 1);
            let mut most_aligned = None;
            let mut largest_align = 0;
            let mut largest_size = 0;
            for sts.iter().advance |st| {
                if largest_size < st.size {
                    largest_size = st.size;
                }
                if largest_align < st.align {
                    // Clang breaks ties by size; it is unclear if
                    // that accomplishes anything important.
                    largest_align = st.align;
                    most_aligned = Some(st);
                }
            }
            let most_aligned = most_aligned.get();
            let padding = largest_size - most_aligned.size;

            struct_llfields(cx, most_aligned, sizing)
                + &[Type::array(&Type::i8(), padding)]
        }
    }
}

fn struct_llfields(cx: &mut CrateContext, st: &Struct, sizing: bool) -> ~[Type] {
    if sizing {
        st.fields.map(|&ty| type_of::sizing_type_of(cx, ty))
    } else {
        st.fields.map(|&ty| type_of::type_of(cx, ty))
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
        NullablePointer{ nonnull: ref nonnull, nndiscr, ptrfield, _ } => {
            (_match::switch, Some(nullable_bitdiscr(bcx, nonnull, nndiscr, ptrfield, scrutinee)))
        }
        Univariant(*) => {
            (_match::single, None)
        }
    }
}



/// Obtain the actual discriminant of a value.
pub fn trans_get_discr(bcx: block, r: &Repr, scrutinee: ValueRef)
    -> ValueRef {
    match *r {
        CEnum(min, max) => load_discr(bcx, scrutinee, min, max),
        Univariant(*) => C_int(bcx.ccx(), 0),
        General(ref cases) => load_discr(bcx, scrutinee, 0,
                                         (cases.len() - 1) as int),
        NullablePointer{ nonnull: ref nonnull, nndiscr, ptrfield, _ } => {
            ZExt(bcx, nullable_bitdiscr(bcx, nonnull, nndiscr, ptrfield, scrutinee),
                 Type::enum_discrim(bcx.ccx()))
        }
    }
}

fn nullable_bitdiscr(bcx: block, nonnull: &Struct, nndiscr: int, ptrfield: uint,
                     scrutinee: ValueRef) -> ValueRef {
    let cmp = if nndiscr == 0 { IntEQ } else { IntNE };
    let llptr = Load(bcx, GEPi(bcx, scrutinee, [0, ptrfield]));
    let llptrty = type_of::type_of(bcx.ccx(), nonnull.fields[ptrfield]);
    ICmp(bcx, cmp, llptr, C_null(llptrty))
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
        Univariant(*) => {
            bcx.ccx().sess.bug("no cases for univariants or structs")
        }
        General(*) => {
            _match::single_result(rslt(bcx, C_int(bcx.ccx(), discr)))
        }
        NullablePointer{ _ } => {
            assert!(discr == 0 || discr == 1);
            _match::single_result(rslt(bcx, C_i1(discr != 0)))
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
        CEnum(min, max) => {
            assert!(min <= discr && discr <= max);
            Store(bcx, C_int(bcx.ccx(), discr), GEPi(bcx, val, [0, 0]))
        }
        Univariant(ref st, true) => {
            assert_eq!(discr, 0);
            Store(bcx, C_bool(true),
                  GEPi(bcx, val, [0, st.fields.len() - 1]))
        }
        Univariant(*) => {
            assert_eq!(discr, 0);
        }
        General(*) => {
            Store(bcx, C_int(bcx.ccx(), discr), GEPi(bcx, val, [0, 0]))
        }
        NullablePointer{ nonnull: ref nonnull, nndiscr, ptrfield, _ } => {
            if discr != nndiscr {
                let llptrptr = GEPi(bcx, val, [0, ptrfield]);
                let llptrty = type_of::type_of(bcx.ccx(), nonnull.fields[ptrfield]);
                Store(bcx, C_null(llptrty), llptrptr)
            }
        }
    }
}

/**
 * The number of fields in a given case; for use when obtaining this
 * information from the type or definition is less convenient.
 */
pub fn num_args(r: &Repr, discr: int) -> uint {
    match *r {
        CEnum(*) => 0,
        Univariant(ref st, dtor) => {
            assert_eq!(discr, 0);
            st.fields.len() - (if dtor { 1 } else { 0 })
        }
        General(ref cases) => cases[discr as uint].fields.len() - 1,
        NullablePointer{ nonnull: ref nonnull, nndiscr, nullfields: ref nullfields, _ } => {
            if discr == nndiscr { nonnull.fields.len() } else { nullfields.len() }
        }
    }
}

/// Access a field, at a point when the value's case is known.
pub fn trans_field_ptr(bcx: block, r: &Repr, val: ValueRef, discr: int,
                       ix: uint) -> ValueRef {
    // Note: if this ever needs to generate conditionals (e.g., if we
    // decide to do some kind of cdr-coding-like non-unique repr
    // someday), it will need to return a possibly-new bcx as well.
    match *r {
        CEnum(*) => {
            bcx.ccx().sess.bug("element access in C-like enum")
        }
        Univariant(ref st, _dtor) => {
            assert_eq!(discr, 0);
            struct_field_ptr(bcx, st, val, ix, false)
        }
        General(ref cases) => {
            struct_field_ptr(bcx, &cases[discr as uint], val, ix + 1, true)
        }
        NullablePointer{ nonnull: ref nonnull, nullfields: ref nullfields, nndiscr, _ } => {
            if (discr == nndiscr) {
                struct_field_ptr(bcx, nonnull, val, ix, false)
            } else {
                // The unit-like case might have a nonzero number of unit-like fields.
                // (e.g., Result or Either with () as one side.)
                let ty = type_of::type_of(bcx.ccx(), nullfields[ix]);
                assert_eq!(machine::llsize_of_alloc(bcx.ccx(), ty), 0);
                // The contents of memory at this pointer can't matter, but use
                // the value that's "reasonable" in case of pointer comparison.
                PointerCast(bcx, val, ty.ptr_to())
            }
        }
    }
}

fn struct_field_ptr(bcx: block, st: &Struct, val: ValueRef, ix: uint,
              needs_cast: bool) -> ValueRef {
    let ccx = bcx.ccx();

    let val = if needs_cast {
        let fields = do st.fields.map |&ty| {
            type_of::type_of(ccx, ty)
        };
        let real_ty = Type::struct_(fields, st.packed);
        PointerCast(bcx, val, real_ty.ptr_to())
    } else {
        val
    };

    GEPi(bcx, val, [0, ix])
}

/// Access the struct drop flag, if present.
pub fn trans_drop_flag_ptr(bcx: block, r: &Repr, val: ValueRef) -> ValueRef {
    match *r {
        Univariant(ref st, true) => GEPi(bcx, val, [0, st.fields.len() - 1]),
        _ => bcx.ccx().sess.bug("tried to get drop flag of non-droppable type")
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
 * V32(u32, u32) }` on win32.  The type has 8-byte alignment to
 * accommodate the u64, but `V32(x, y)` would have LLVM type `{i32,
 * i32, i32}`, which is 4-byte aligned.
 *
 * Currently the returned value has the same size as the type, but
 * this could be changed in the future to avoid allocating unnecessary
 * space after values of shorter-than-maximum cases.
 */
pub fn trans_const(ccx: &mut CrateContext, r: &Repr, discr: int,
                   vals: &[ValueRef]) -> ValueRef {
    match *r {
        CEnum(min, max) => {
            assert_eq!(vals.len(), 0);
            assert!(min <= discr && discr <= max);
            C_int(ccx, discr)
        }
        Univariant(ref st, _dro) => {
            assert_eq!(discr, 0);
            C_struct(build_const_struct(ccx, st, vals))
        }
        General(ref cases) => {
            let case = &cases[discr as uint];
            let max_sz = cases.iter().transform(|x| x.size).max().unwrap();
            let discr_ty = C_int(ccx, discr);
            let contents = build_const_struct(ccx, case,
                                              ~[discr_ty] + vals);
            C_struct(contents + &[padding(max_sz - case.size)])
        }
        NullablePointer{ nonnull: ref nonnull, nndiscr, ptrfield, _ } => {
            if discr == nndiscr {
                C_struct(build_const_struct(ccx, nonnull, vals))
            } else {
                assert_eq!(vals.len(), 0);
                let vals = do nonnull.fields.iter().enumerate().transform |(i, &ty)| {
                    let llty = type_of::sizing_type_of(ccx, ty);
                    if i == ptrfield { C_null(llty) } else { C_undef(llty) }
                }.collect::<~[ValueRef]>();
                C_struct(build_const_struct(ccx, nonnull, vals))
            }
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
fn build_const_struct(ccx: &mut CrateContext, st: &Struct, vals: &[ValueRef])
    -> ~[ValueRef] {
    assert_eq!(vals.len(), st.fields.len());

    let mut offset = 0;
    let mut cfields = ~[];
    for st.fields.iter().enumerate().advance |(i, &ty)| {
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
        let val = if is_undef(vals[i]) {
            let wrapped = C_struct([vals[i]]);
            assert!(!is_undef(wrapped));
            wrapped
        } else {
            vals[i]
        };
        cfields.push(val);
        offset += machine::llsize_of_alloc(ccx, llty) as u64
    }

    return cfields;
}

fn padding(size: u64) -> ValueRef {
    C_undef(Type::array(&Type::i8(), size))
}

// XXX this utility routine should be somewhere more general
#[inline]
fn roundup(x: u64, a: u64) -> u64 { ((x + (a - 1)) / a) * a }

/// Get the discriminant of a constant value.  (Not currently used.)
pub fn const_get_discrim(ccx: &mut CrateContext, r: &Repr, val: ValueRef)
    -> int {
    match *r {
        CEnum(*) => const_to_int(val) as int,
        Univariant(*) => 0,
        General(*) => const_to_int(const_get_elt(ccx, val, [0])) as int,
        NullablePointer{ nndiscr, ptrfield, _ } => {
            if is_null(const_struct_field(ccx, val, ptrfield)) { 1 - nndiscr } else { nndiscr }
        }
    }
}

/**
 * Extract a field of a constant value, as appropriate for its
 * representation.
 *
 * (Not to be confused with `common::const_get_elt`, which operates on
 * raw LLVM-level structs and arrays.)
 */
pub fn const_get_field(ccx: &mut CrateContext, r: &Repr, val: ValueRef,
                       _discr: int, ix: uint) -> ValueRef {
    match *r {
        CEnum(*) => ccx.sess.bug("element access in C-like enum const"),
        Univariant(*) => const_struct_field(ccx, val, ix),
        General(*) => const_struct_field(ccx, val, ix + 1),
        NullablePointer{ _ } => const_struct_field(ccx, val, ix)
    }
}

/// Extract field of struct-like const, skipping our alignment padding.
fn const_struct_field(ccx: &mut CrateContext, val: ValueRef, ix: uint)
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
        Univariant(ref st, false) => st.fields.len() == 1,
        _ => false
    }
}
