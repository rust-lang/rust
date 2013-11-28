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
 * values of those types.  (Note: exporting the representations for
 * debuggers is handled in debuginfo.rs, not here.)
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

use lib::llvm::{ValueRef, True, IntEQ, IntNE};
use middle::trans::_match;
use middle::trans::build::*;
use middle::trans::common::*;
use middle::trans::machine;
use middle::trans::type_of;
use middle::ty;
use middle::ty::Disr;
use syntax::abi::{X86, X86_64, Arm, Mips};
use syntax::ast;
use syntax::attr;
use syntax::attr::IntType;
use util::ppaux::ty_to_str;

use middle::trans::type_::Type;

type Hint = attr::ReprAttr;


/// Representations.
pub enum Repr {
    /// C-like enums; basically an int.
    CEnum(IntType, Disr, Disr), // discriminant range (signedness based on the IntType)
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
    General(IntType, ~[Struct]),
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
    NullablePointer{ nonnull: Struct, nndiscr: Disr, ptrfield: uint,
                     nullfields: ~[ty::t] }
}

/// For structs, and struct-like parts of anything fancier.
pub struct Struct {
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
pub fn represent_node(bcx: @mut Block, node: ast::NodeId) -> @Repr {
    represent_type(bcx.ccx(), node_id_type(bcx, node))
}

/// Decides how to represent a given type.
pub fn represent_type(cx: &mut CrateContext, t: ty::t) -> @Repr {
    debug!("Representing: {}", ty_to_str(cx.tcx, t));
    match cx.adt_reprs.find(&t) {
        Some(repr) => return *repr,
        None => { }
    }
    let repr = @represent_type_uncached(cx, t);
    debug!("Represented as: {:?}", repr)
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
            let mut ftys = fields.map(|field| {
                ty::lookup_field_type(cx.tcx, def_id, field.id, substs)
            });
            let packed = ty::lookup_packed(cx.tcx, def_id);
            let dtor = ty::ty_dtor(cx.tcx, def_id).has_drop_flag();
            if dtor { ftys.push(ty::mk_bool()); }

            return Univariant(mk_struct(cx, ftys, packed), dtor)
        }
        ty::ty_enum(def_id, ref substs) => {
            let cases = get_cases(cx.tcx, def_id, substs);
            let hint = ty::lookup_repr_hint(cx.tcx, def_id);

            if cases.len() == 0 {
                // Uninhabitable; represent as unit
                // (Typechecking will reject discriminant-sizing attrs.)
                assert_eq!(hint, attr::ReprAny);
                return Univariant(mk_struct(cx, [], false), false);
            }

            if cases.iter().all(|c| c.tys.len() == 0) {
                // All bodies empty -> intlike
                let discrs = cases.map(|c| c.discr);
                let bounds = IntBounds {
                    ulo: *discrs.iter().min().unwrap(),
                    uhi: *discrs.iter().max().unwrap(),
                    slo: discrs.iter().map(|n| *n as i64).min().unwrap(),
                    shi: discrs.iter().map(|n| *n as i64).max().unwrap()
                };
                return mk_cenum(cx, hint, &bounds);
            }

            // Since there's at least one
            // non-empty body, explicit discriminants should have
            // been rejected by a checker before this point.
            if !cases.iter().enumerate().all(|(i,c)| c.discr == (i as Disr)) {
                cx.sess.bug(format!("non-C-like enum {} with specified \
                                  discriminants",
                                 ty::item_path_str(cx.tcx, def_id)))
            }

            if cases.len() == 1 {
                // Equivalent to a struct/tuple/newtype.
                // (Typechecking will reject discriminant-sizing attrs.)
                assert_eq!(hint, attr::ReprAny);
                return Univariant(mk_struct(cx, cases[0].tys, false), false)
            }

            if cases.len() == 2 && hint == attr::ReprAny {
                // Nullable pointer optimization
                let mut discr = 0;
                while discr < 2 {
                    if cases[1 - discr].is_zerolen(cx) {
                        match cases[discr].find_ptr() {
                            Some(ptrfield) => {
                                return NullablePointer {
                                    nndiscr: discr,
                                    nonnull: mk_struct(cx,
                                                       cases[discr].tys,
                                                       false),
                                    ptrfield: ptrfield,
                                    nullfields: cases[1 - discr].tys.clone()
                                }
                            }
                            None => { }
                        }
                    }
                    discr += 1;
                }
            }

            // The general case.
            assert!((cases.len() - 1) as i64 >= 0);
            let bounds = IntBounds { ulo: 0, uhi: (cases.len() - 1) as u64,
                                     slo: 0, shi: (cases.len() - 1) as i64 };
            let ity = range_to_inttype(cx, hint, &bounds);
            let discr = ~[ty_of_inttype(ity)];
            return General(ity, cases.map(|c| mk_struct(cx, discr + c.tys, false)))
        }
        _ => cx.sess.bug("adt::represent_type called on non-ADT type")
    }
}

/// Determine, without doing translation, whether an ADT must be FFI-safe.
/// For use in lint or similar, where being sound but slightly incomplete is acceptable.
pub fn is_ffi_safe(tcx: ty::ctxt, def_id: ast::DefId) -> bool {
    match ty::get(ty::lookup_item_type(tcx, def_id).ty).sty {
        ty::ty_enum(def_id, _) => {
            let variants = ty::enum_variants(tcx, def_id);
            // Univariant => like struct/tuple.
            if variants.len() <= 1 {
                return true;
            }
            let hint = ty::lookup_repr_hint(tcx, def_id);
            // Appropriate representation explicitly selected?
            if hint.is_ffi_safe() {
                return true;
            }
            // Option<~T> and similar are used in FFI.  Rather than try to resolve type parameters
            // and recognize this case exactly, this overapproximates -- assuming that if a
            // non-C-like enum is being used in FFI then the user knows what they're doing.
            if variants.iter().any(|vi| !vi.args.is_empty()) {
                return true;
            }
            false
        }
        // struct, tuple, etc.
        // (is this right in the present of typedefs?)
        _ => true
    }
}

// NOTE this should probably all be in ty
struct Case { discr: Disr, tys: ~[ty::t] }
impl Case {
    fn is_zerolen(&self, cx: &mut CrateContext) -> bool {
        mk_struct(cx, self.tys, false).size == 0
    }
    fn find_ptr(&self) -> Option<uint> {
        self.tys.iter().position(|&ty| mono_data_classify(ty) == MonoNonNull)
    }
}

fn get_cases(tcx: ty::ctxt, def_id: ast::DefId, substs: &ty::substs) -> ~[Case] {
    ty::enum_variants(tcx, def_id).map(|vi| {
        let arg_tys = vi.args.map(|&raw_ty| {
            ty::subst(tcx, substs, raw_ty)
        });
        Case { discr: vi.disr_val, tys: arg_tys }
    })
}


fn mk_struct(cx: &mut CrateContext, tys: &[ty::t], packed: bool) -> Struct {
    let lltys = tys.map(|&ty| type_of::sizing_type_of(cx, ty));
    let llty_rec = Type::struct_(lltys, packed);
    Struct {
        size: machine::llsize_of_alloc(cx, llty_rec) /*bad*/as u64,
        align: machine::llalign_of_min(cx, llty_rec) /*bad*/as u64,
        packed: packed,
        fields: tys.to_owned(),
    }
}

struct IntBounds {
    slo: i64,
    shi: i64,
    ulo: u64,
    uhi: u64
}

fn mk_cenum(cx: &mut CrateContext, hint: Hint, bounds: &IntBounds) -> Repr {
    let it = range_to_inttype(cx, hint, bounds);
    match it {
        attr::SignedInt(_) => CEnum(it, bounds.slo as Disr, bounds.shi as Disr),
        attr::UnsignedInt(_) => CEnum(it, bounds.ulo, bounds.uhi)
    }
}

fn range_to_inttype(cx: &mut CrateContext, hint: Hint, bounds: &IntBounds) -> IntType {
    debug!("range_to_inttype: {:?} {:?}", hint, bounds);
    // Lists of sizes to try.  u64 is always allowed as a fallback.
    static choose_shortest: &'static[IntType] = &[
        attr::UnsignedInt(ast::ty_u8), attr::SignedInt(ast::ty_i8),
        attr::UnsignedInt(ast::ty_u16), attr::SignedInt(ast::ty_i16),
        attr::UnsignedInt(ast::ty_u32), attr::SignedInt(ast::ty_i32)];
    static at_least_32: &'static[IntType] = &[
        attr::UnsignedInt(ast::ty_u32), attr::SignedInt(ast::ty_i32)];

    let attempts;
    match hint {
        attr::ReprInt(span, ity) => {
            if !bounds_usable(cx, ity, bounds) {
                cx.sess.span_bug(span, "representation hint insufficient for discriminant range")
            }
            return ity;
        }
        attr::ReprExtern => {
            attempts = match cx.sess.targ_cfg.arch {
                X86 | X86_64 => at_least_32,
                // WARNING: the ARM EABI has two variants; the one corresponding to `at_least_32`
                // appears to be used on Linux and NetBSD, but some systems may use the variant
                // corresponding to `choose_shortest`.  However, we don't run on those yet...?
                Arm => at_least_32,
                Mips => at_least_32,
            }
        }
        attr::ReprAny => {
            attempts = choose_shortest;
        }
    }
    for &ity in attempts.iter() {
        if bounds_usable(cx, ity, bounds) {
            return ity;
        }
    }
    return attr::UnsignedInt(ast::ty_u64);
}

pub fn ll_inttype(cx: &mut CrateContext, ity: IntType) -> Type {
    match ity {
        attr::SignedInt(t) => Type::int_from_ty(cx, t),
        attr::UnsignedInt(t) => Type::uint_from_ty(cx, t)
    }
}

fn bounds_usable(cx: &mut CrateContext, ity: IntType, bounds: &IntBounds) -> bool {
    debug!("bounds_usable: {:?} {:?}", ity, bounds);
    match ity {
        attr::SignedInt(_) => {
            let lllo = C_integral(ll_inttype(cx, ity), bounds.slo as u64, true);
            let llhi = C_integral(ll_inttype(cx, ity), bounds.shi as u64, true);
            bounds.slo == const_to_int(lllo) as i64 && bounds.shi == const_to_int(llhi) as i64
        }
        attr::UnsignedInt(_) => {
            let lllo = C_integral(ll_inttype(cx, ity), bounds.ulo, false);
            let llhi = C_integral(ll_inttype(cx, ity), bounds.uhi, false);
            bounds.ulo == const_to_uint(lllo) as u64 && bounds.uhi == const_to_uint(llhi) as u64
        }
    }
}

pub fn ty_of_inttype(ity: IntType) -> ty::t {
    match ity {
        attr::SignedInt(t) => ty::mk_mach_int(t),
        attr::UnsignedInt(t) => ty::mk_mach_uint(t)
    }
}


/**
 * LLVM-level types are a little complicated.
 *
 * C-like enums need to be actual ints, not wrapped in a struct,
 * because that changes the ABI on some platforms (see issue #10308).
 *
 * For nominal types, in some cases, we need to use LLVM named structs
 * and fill in the actual contents in a second pass to prevent
 * unbounded recursion; see also the comments in `trans::type_of`.
 */
pub fn type_of(cx: &mut CrateContext, r: &Repr) -> Type {
    generic_type_of(cx, r, None, false)
}
pub fn sizing_type_of(cx: &mut CrateContext, r: &Repr) -> Type {
    generic_type_of(cx, r, None, true)
}
pub fn incomplete_type_of(cx: &mut CrateContext, r: &Repr, name: &str) -> Type {
    generic_type_of(cx, r, Some(name), false)
}
pub fn finish_type_of(cx: &mut CrateContext, r: &Repr, llty: &mut Type) {
    match *r {
        CEnum(*) | General(*) => { }
        Univariant(ref st, _) | NullablePointer{ nonnull: ref st, _ } =>
            llty.set_struct_body(struct_llfields(cx, st, false), st.packed)
    }
}

fn generic_type_of(cx: &mut CrateContext, r: &Repr, name: Option<&str>, sizing: bool) -> Type {
    match *r {
        CEnum(ity, _, _) => ll_inttype(cx, ity),
        Univariant(ref st, _) | NullablePointer{ nonnull: ref st, _ } => {
            match name {
                None => Type::struct_(struct_llfields(cx, st, sizing), st.packed),
                Some(name) => { assert_eq!(sizing, false); Type::named_struct(name) }
            }
        }
        General(ity, ref sts) => {
            // We need a representation that has:
            // * The alignment of the most-aligned field
            // * The size of the largest variant (rounded up to that alignment)
            // * No alignment padding anywhere any variant has actual data
            //   (currently matters only for enums small enough to be immediate)
            // * The discriminant in an obvious place.
            //
            // So we start with the discriminant, pad it up to the alignment with
            // more of its own type, then use alignment-sized ints to get the rest
            // of the size.
            //
            // FIXME #10604: this breaks when vector types are present.
            let size = sts.iter().map(|st| st.size).max().unwrap();
            let most_aligned = sts.iter().max_by(|st| st.align).unwrap();
            let align = most_aligned.align;
            let discr_ty = ll_inttype(cx, ity);
            let discr_size = machine::llsize_of_alloc(cx, discr_ty) as u64;
            let pad_ty = match align {
                1 => Type::i8(),
                2 => Type::i16(),
                4 => Type::i32(),
                8 if machine::llalign_of_min(cx, Type::i64()) == 8 => Type::i64(),
                _ => fail!("Unsupported enum alignment: {:?}", align)
            };
            assert_eq!(machine::llalign_of_min(cx, pad_ty) as u64, align);
            let align_units = (size + align - 1) / align;
            assert_eq!(align % discr_size, 0);
            let fields = ~[discr_ty,
              Type::array(&discr_ty, align / discr_size - 1),
              Type::array(&pad_ty, align_units - 1)];
            match name {
                None => Type::struct_(fields, false),
                Some(name) => {
                    let mut llty = Type::named_struct(name);
                    llty.set_struct_body(fields, false);
                    llty
                }
            }
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
pub fn trans_switch(bcx: @mut Block, r: &Repr, scrutinee: ValueRef)
    -> (_match::branch_kind, Option<ValueRef>) {
    match *r {
        CEnum(*) | General(*) => {
            (_match::switch, Some(trans_get_discr(bcx, r, scrutinee, None)))
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
pub fn trans_get_discr(bcx: @mut Block, r: &Repr, scrutinee: ValueRef, cast_to: Option<Type>)
    -> ValueRef {
    let signed;
    let val;
    match *r {
        CEnum(ity, min, max) => {
            val = load_discr(bcx, ity, scrutinee, min, max);
            signed = ity.is_signed();
        }
        General(ity, ref cases) => {
            let ptr = GEPi(bcx, scrutinee, [0, 0]);
            val = load_discr(bcx, ity, ptr, 0, (cases.len() - 1) as Disr);
            signed = ity.is_signed();
        }
        Univariant(*) => {
            val = C_u8(0);
            signed = false;
        }
        NullablePointer{ nonnull: ref nonnull, nndiscr, ptrfield, _ } => {
            val = nullable_bitdiscr(bcx, nonnull, nndiscr, ptrfield, scrutinee);
            signed = false;
        }
    }
    match cast_to {
        None => val,
        Some(llty) => if signed { SExt(bcx, val, llty) } else { ZExt(bcx, val, llty) }
    }
}

fn nullable_bitdiscr(bcx: @mut Block, nonnull: &Struct, nndiscr: Disr, ptrfield: uint,
                     scrutinee: ValueRef) -> ValueRef {
    let cmp = if nndiscr == 0 { IntEQ } else { IntNE };
    let llptr = Load(bcx, GEPi(bcx, scrutinee, [0, ptrfield]));
    let llptrty = type_of::type_of(bcx.ccx(), nonnull.fields[ptrfield]);
    ICmp(bcx, cmp, llptr, C_null(llptrty))
}

/// Helper for cases where the discriminant is simply loaded.
fn load_discr(bcx: @mut Block, ity: IntType, ptr: ValueRef, min: Disr, max: Disr)
    -> ValueRef {
    let llty = ll_inttype(bcx.ccx(), ity);
    assert_eq!(val_ty(ptr), llty.ptr_to());
    let bits = machine::llbitsize_of_real(bcx.ccx(), llty);
    assert!(bits <= 64);
    let mask = (-1u64 >> (64 - bits)) as Disr;
    if (max + 1) & mask == min & mask {
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
pub fn trans_case(bcx: @mut Block, r: &Repr, discr: Disr) -> _match::opt_result {
    match *r {
        CEnum(ity, _, _) => {
            _match::single_result(rslt(bcx, C_integral(ll_inttype(bcx.ccx(), ity),
                                                       discr as u64, true)))
        }
        General(ity, _) => {
            _match::single_result(rslt(bcx, C_integral(ll_inttype(bcx.ccx(), ity),
                                                       discr as u64, true)))
        }
        Univariant(*) => {
            bcx.ccx().sess.bug("no cases for univariants or structs")
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
pub fn trans_start_init(bcx: @mut Block, r: &Repr, val: ValueRef, discr: Disr) {
    match *r {
        CEnum(ity, min, max) => {
            assert_discr_in_range(ity, min, max, discr);
            Store(bcx, C_integral(ll_inttype(bcx.ccx(), ity), discr as u64, true),
                  val)
        }
        General(ity, _) => {
            Store(bcx, C_integral(ll_inttype(bcx.ccx(), ity), discr as u64, true),
                  GEPi(bcx, val, [0, 0]))
        }
        Univariant(ref st, true) => {
            assert_eq!(discr, 0);
            Store(bcx, C_bool(true),
                  GEPi(bcx, val, [0, st.fields.len() - 1]))
        }
        Univariant(*) => {
            assert_eq!(discr, 0);
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

fn assert_discr_in_range(ity: IntType, min: Disr, max: Disr, discr: Disr) {
    match ity {
        attr::UnsignedInt(_) => assert!(min <= discr && discr <= max),
        attr::SignedInt(_) => assert!(min as i64 <= discr as i64 && discr as i64 <= max as i64)
    }
}

/**
 * The number of fields in a given case; for use when obtaining this
 * information from the type or definition is less convenient.
 */
pub fn num_args(r: &Repr, discr: Disr) -> uint {
    match *r {
        CEnum(*) => 0,
        Univariant(ref st, dtor) => {
            assert_eq!(discr, 0);
            st.fields.len() - (if dtor { 1 } else { 0 })
        }
        General(_, ref cases) => cases[discr].fields.len() - 1,
        NullablePointer{ nonnull: ref nonnull, nndiscr, nullfields: ref nullfields, _ } => {
            if discr == nndiscr { nonnull.fields.len() } else { nullfields.len() }
        }
    }
}

/// Access a field, at a point when the value's case is known.
pub fn trans_field_ptr(bcx: @mut Block, r: &Repr, val: ValueRef, discr: Disr,
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
        General(_, ref cases) => {
            struct_field_ptr(bcx, &cases[discr], val, ix + 1, true)
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

fn struct_field_ptr(bcx: @mut Block, st: &Struct, val: ValueRef, ix: uint,
              needs_cast: bool) -> ValueRef {
    let ccx = bcx.ccx();

    let val = if needs_cast {
        let fields = st.fields.map(|&ty| type_of::type_of(ccx, ty));
        let real_ty = Type::struct_(fields, st.packed);
        PointerCast(bcx, val, real_ty.ptr_to())
    } else {
        val
    };

    GEPi(bcx, val, [0, ix])
}

/// Access the struct drop flag, if present.
pub fn trans_drop_flag_ptr(bcx: @mut Block, r: &Repr, val: ValueRef) -> ValueRef {
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
pub fn trans_const(ccx: &mut CrateContext, r: &Repr, discr: Disr,
                   vals: &[ValueRef]) -> ValueRef {
    match *r {
        CEnum(ity, min, max) => {
            assert_eq!(vals.len(), 0);
            assert_discr_in_range(ity, min, max, discr);
            C_integral(ll_inttype(ccx, ity), discr as u64, true)
        }
        General(ity, ref cases) => {
            let case = &cases[discr];
            let max_sz = cases.iter().map(|x| x.size).max().unwrap();
            let lldiscr = C_integral(ll_inttype(ccx, ity), discr as u64, true);
            let contents = build_const_struct(ccx, case, ~[lldiscr] + vals);
            C_struct(contents + &[padding(max_sz - case.size)], false)
        }
        Univariant(ref st, _dro) => {
            assert!(discr == 0);
            let contents = build_const_struct(ccx, st, vals);
            C_struct(contents, st.packed)
        }
        NullablePointer{ nonnull: ref nonnull, nndiscr, ptrfield, _ } => {
            if discr == nndiscr {
                C_struct(build_const_struct(ccx, nonnull, vals), false)
            } else {
                assert_eq!(vals.len(), 0);
                let vals = nonnull.fields.iter().enumerate().map(|(i, &ty)| {
                    let llty = type_of::sizing_type_of(ccx, ty);
                    if i == ptrfield { C_null(llty) } else { C_undef(llty) }
                }).collect::<~[ValueRef]>();
                C_struct(build_const_struct(ccx, nonnull, vals), false)
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
    for (i, &ty) in st.fields.iter().enumerate() {
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
            let wrapped = C_struct([vals[i]], false);
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
    -> Disr {
    match *r {
        CEnum(ity, _, _) => {
            match ity {
                attr::SignedInt(*) => const_to_int(val) as Disr,
                attr::UnsignedInt(*) => const_to_uint(val) as Disr
            }
        }
        General(ity, _) => {
            match ity {
                attr::SignedInt(*) => const_to_int(const_get_elt(ccx, val, [0])) as Disr,
                attr::UnsignedInt(*) => const_to_uint(const_get_elt(ccx, val, [0])) as Disr
            }
        }
        Univariant(*) => 0,
        NullablePointer{ nndiscr, ptrfield, _ } => {
            if is_null(const_struct_field(ccx, val, ptrfield)) {
                /* subtraction as uint is ok because nndiscr is either 0 or 1 */
                (1 - nndiscr) as Disr
            } else {
                nndiscr
            }
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
                       _discr: Disr, ix: uint) -> ValueRef {
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
