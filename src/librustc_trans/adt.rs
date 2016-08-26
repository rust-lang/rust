// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Representation of Algebraic Data Types
//!
//! This module determines how to represent enums, structs, and tuples
//! based on their monomorphized types; it is responsible both for
//! choosing a representation and translating basic operations on
//! values of those types.  (Note: exporting the representations for
//! debuggers is handled in debuginfo.rs, not here.)
//!
//! Note that the interface treats everything as a general case of an
//! enum, so structs/tuples/etc. have one pseudo-variant with
//! discriminant 0; i.e., as if they were a univariant enum.
//!
//! Having everything in one place will enable improvements to data
//! structure representation; possibilities include:
//!
//! - User-specified alignment (e.g., cacheline-aligning parts of
//!   concurrently accessed data structures); LLVM can't represent this
//!   directly, so we'd have to insert padding fields in any structure
//!   that might contain one and adjust GEP indices accordingly.  See
//!   issue #4578.
//!
//! - Store nested enums' discriminants in the same word.  Rather, if
//!   some variants start with enums, and those enums representations
//!   have unused alignment padding between discriminant and body, the
//!   outer enum's discriminant can be stored there and those variants
//!   can start at offset 0.  Kind of fancy, and might need work to
//!   make copies of the inner enum type cooperate, but it could help
//!   with `Option` or `Result` wrapped around another enum.
//!
//! - Tagged pointers would be neat, but given that any type can be
//!   used unboxed and any field can have pointers (including mutable)
//!   taken to it, implementing them for Rust seems difficult.

pub use self::Repr::*;
use super::Disr;

use std;
use std::rc::Rc;

use llvm::{ValueRef, True, IntEQ, IntNE};
use rustc::ty::subst::Substs;
use rustc::ty::{self, Ty, TyCtxt};
use syntax::ast;
use syntax::attr;
use syntax::attr::IntType;
use abi::FAT_PTR_ADDR;
use build::*;
use common::*;
use debuginfo::DebugLoc;
use glue;
use machine;
use monomorphize;
use type_::Type;
use type_of;
use value::Value;

#[derive(Copy, Clone, PartialEq)]
pub enum BranchKind {
    Switch,
    Single
}

type Hint = attr::ReprAttr;

/// Representations.
#[derive(Eq, PartialEq, Debug)]
pub enum Repr<'tcx> {
    /// C-like enums; basically an int.
    CEnum(IntType, Disr, Disr), // discriminant range (signedness based on the IntType)
    /// Single-case variants, and structs/tuples/records.
    Univariant(Struct<'tcx>),
    /// Untagged unions.
    UntaggedUnion(Union<'tcx>),
    /// General-case enums: for each case there is a struct, and they
    /// all start with a field for the discriminant.
    General(IntType, Vec<Struct<'tcx>>),
    /// Two cases distinguished by a nullable pointer: the case with discriminant
    /// `nndiscr` must have single field which is known to be nonnull due to its type.
    /// The other case is known to be zero sized. Hence we represent the enum
    /// as simply a nullable pointer: if not null it indicates the `nndiscr` variant,
    /// otherwise it indicates the other case.
    RawNullablePointer {
        nndiscr: Disr,
        nnty: Ty<'tcx>,
        nullfields: Vec<Ty<'tcx>>
    },
    /// Two cases distinguished by a nullable pointer: the case with discriminant
    /// `nndiscr` is represented by the struct `nonnull`, where the `discrfield`th
    /// field is known to be nonnull due to its type; if that field is null, then
    /// it represents the other case, which is inhabited by at most one value
    /// (and all other fields are undefined/unused).
    ///
    /// For example, `std::option::Option` instantiated at a safe pointer type
    /// is represented such that `None` is a null pointer and `Some` is the
    /// identity function.
    StructWrappedNullablePointer {
        nonnull: Struct<'tcx>,
        nndiscr: Disr,
        discrfield: DiscrField,
        nullfields: Vec<Ty<'tcx>>,
    }
}

/// For structs, and struct-like parts of anything fancier.
#[derive(Eq, PartialEq, Debug)]
pub struct Struct<'tcx> {
    // If the struct is DST, then the size and alignment do not take into
    // account the unsized fields of the struct.
    pub size: u64,
    pub align: u32,
    pub sized: bool,
    pub packed: bool,
    pub fields: Vec<Ty<'tcx>>,
}

/// For untagged unions.
#[derive(Eq, PartialEq, Debug)]
pub struct Union<'tcx> {
    pub min_size: u64,
    pub align: u32,
    pub packed: bool,
    pub fields: Vec<Ty<'tcx>>,
}

#[derive(Copy, Clone)]
pub struct MaybeSizedValue {
    pub value: ValueRef,
    pub meta: ValueRef,
}

impl MaybeSizedValue {
    pub fn sized(value: ValueRef) -> MaybeSizedValue {
        MaybeSizedValue {
            value: value,
            meta: std::ptr::null_mut()
        }
    }

    pub fn unsized_(value: ValueRef, meta: ValueRef) -> MaybeSizedValue {
        MaybeSizedValue {
            value: value,
            meta: meta
        }
    }

    pub fn has_meta(&self) -> bool {
        !self.meta.is_null()
    }
}

/// Decides how to represent a given type.
pub fn represent_type<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                t: Ty<'tcx>)
                                -> Rc<Repr<'tcx>> {
    debug!("Representing: {}", t);
    if let Some(repr) = cx.adt_reprs().borrow().get(&t) {
        return repr.clone();
    }

    let repr = Rc::new(represent_type_uncached(cx, t));
    debug!("Represented as: {:?}", repr);
    cx.adt_reprs().borrow_mut().insert(t, repr.clone());
    repr
}

fn represent_type_uncached<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                     t: Ty<'tcx>) -> Repr<'tcx> {
    match t.sty {
        ty::TyTuple(ref elems) => {
            Univariant(mk_struct(cx, &elems[..], false, t))
        }
        ty::TyStruct(def, substs) => {
            let ftys = def.struct_variant().fields.iter().map(|field| {
                monomorphize::field_ty(cx.tcx(), substs, field)
            }).collect::<Vec<_>>();
            let packed = cx.tcx().lookup_packed(def.did);

            Univariant(mk_struct(cx, &ftys[..], packed, t))
        }
        ty::TyUnion(def, substs) => {
            let ftys = def.struct_variant().fields.iter().map(|field| {
                monomorphize::field_ty(cx.tcx(), substs, field)
            }).collect::<Vec<_>>();
            let packed = cx.tcx().lookup_packed(def.did);
            UntaggedUnion(mk_union(cx, &ftys[..], packed, t))
        }
        ty::TyClosure(_, ref substs) => {
            Univariant(mk_struct(cx, &substs.upvar_tys, false, t))
        }
        ty::TyEnum(def, substs) => {
            let cases = get_cases(cx.tcx(), def, substs);
            let hint = *cx.tcx().lookup_repr_hints(def.did).get(0)
                .unwrap_or(&attr::ReprAny);

            if cases.is_empty() {
                // Uninhabitable; represent as unit
                // (Typechecking will reject discriminant-sizing attrs.)
                assert_eq!(hint, attr::ReprAny);
                return Univariant(mk_struct(cx, &[], false, t));
            }

            if cases.iter().all(|c| c.tys.is_empty()) {
                // All bodies empty -> intlike
                let discrs: Vec<_> = cases.iter().map(|c| Disr::from(c.discr)).collect();
                let bounds = IntBounds {
                    ulo: discrs.iter().min().unwrap().0,
                    uhi: discrs.iter().max().unwrap().0,
                    slo: discrs.iter().map(|n| n.0 as i64).min().unwrap(),
                    shi: discrs.iter().map(|n| n.0 as i64).max().unwrap()
                };
                return mk_cenum(cx, hint, &bounds);
            }

            // Since there's at least one
            // non-empty body, explicit discriminants should have
            // been rejected by a checker before this point.
            if !cases.iter().enumerate().all(|(i,c)| c.discr == Disr::from(i)) {
                bug!("non-C-like enum {} with specified discriminants",
                     cx.tcx().item_path_str(def.did));
            }

            if cases.len() == 1 && hint == attr::ReprAny {
                // Equivalent to a struct/tuple/newtype.
                return Univariant(mk_struct(cx, &cases[0].tys, false, t));
            }

            if cases.len() == 2 && hint == attr::ReprAny {
                // Nullable pointer optimization
                let mut discr = 0;
                while discr < 2 {
                    if cases[1 - discr].is_zerolen(cx, t) {
                        let st = mk_struct(cx, &cases[discr].tys,
                                           false, t);
                        match cases[discr].find_ptr(cx) {
                            Some(ref df) if df.len() == 1 && st.fields.len() == 1 => {
                                return RawNullablePointer {
                                    nndiscr: Disr::from(discr),
                                    nnty: st.fields[0],
                                    nullfields: cases[1 - discr].tys.clone()
                                };
                            }
                            Some(mut discrfield) => {
                                discrfield.push(0);
                                discrfield.reverse();
                                return StructWrappedNullablePointer {
                                    nndiscr: Disr::from(discr),
                                    nonnull: st,
                                    discrfield: discrfield,
                                    nullfields: cases[1 - discr].tys.clone()
                                };
                            }
                            None => {}
                        }
                    }
                    discr += 1;
                }
            }

            // The general case.
            assert!((cases.len() - 1) as i64 >= 0);
            let bounds = IntBounds { ulo: 0, uhi: (cases.len() - 1) as u64,
                                     slo: 0, shi: (cases.len() - 1) as i64 };
            let min_ity = range_to_inttype(cx, hint, &bounds);

            // Create the set of structs that represent each variant
            // Use the minimum integer type we figured out above
            let fields : Vec<_> = cases.iter().map(|c| {
                let mut ftys = vec!(ty_of_inttype(cx.tcx(), min_ity));
                ftys.extend_from_slice(&c.tys);
                mk_struct(cx, &ftys, false, t)
            }).collect();


            // Check to see if we should use a different type for the
            // discriminant. If the overall alignment of the type is
            // the same as the first field in each variant, we can safely use
            // an alignment-sized type.
            // We increase the size of the discriminant to avoid LLVM copying
            // padding when it doesn't need to. This normally causes unaligned
            // load/stores and excessive memcpy/memset operations. By using a
            // bigger integer size, LLVM can be sure about it's contents and
            // won't be so conservative.
            // This check is needed to avoid increasing the size of types when
            // the alignment of the first field is smaller than the overall
            // alignment of the type.
            let (_, align) = union_size_and_align(&fields);
            let mut use_align = true;
            for st in &fields {
                // Get the first non-zero-sized field
                let field = st.fields.iter().skip(1).filter(|ty| {
                    let t = type_of::sizing_type_of(cx, **ty);
                    machine::llsize_of_real(cx, t) != 0 ||
                    // This case is only relevant for zero-sized types with large alignment
                    machine::llalign_of_min(cx, t) != 1
                }).next();

                if let Some(field) = field {
                    let field_align = type_of::align_of(cx, *field);
                    if field_align != align {
                        use_align = false;
                        break;
                    }
                }
            }

            // If the alignment is smaller than the chosen discriminant size, don't use the
            // alignment as the final size.
            let min_ty = ll_inttype(&cx, min_ity);
            let min_size = machine::llsize_of_real(cx, min_ty);
            if (align as u64) < min_size {
                use_align = false;
            }

            let ity = if use_align {
                // Use the overall alignment
                match align {
                    1 => attr::UnsignedInt(ast::UintTy::U8),
                    2 => attr::UnsignedInt(ast::UintTy::U16),
                    4 => attr::UnsignedInt(ast::UintTy::U32),
                    8 if machine::llalign_of_min(cx, Type::i64(cx)) == 8 =>
                        attr::UnsignedInt(ast::UintTy::U64),
                    _ => min_ity // use min_ity as a fallback
                }
            } else {
                min_ity
            };

            let fields : Vec<_> = cases.iter().map(|c| {
                let mut ftys = vec!(ty_of_inttype(cx.tcx(), ity));
                ftys.extend_from_slice(&c.tys);
                mk_struct(cx, &ftys[..], false, t)
            }).collect();

            ensure_enum_fits_in_address_space(cx, &fields[..], t);

            General(ity, fields)
        }
        _ => bug!("adt::represent_type called on non-ADT type: {}", t)
    }
}

// this should probably all be in ty
struct Case<'tcx> {
    discr: Disr,
    tys: Vec<Ty<'tcx>>
}

/// This represents the (GEP) indices to follow to get to the discriminant field
pub type DiscrField = Vec<usize>;

fn find_discr_field_candidate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        ty: Ty<'tcx>,
                                        mut path: DiscrField)
                                        -> Option<DiscrField> {
    match ty.sty {
        // Fat &T/&mut T/Box<T> i.e. T is [T], str, or Trait
        ty::TyRef(_, ty::TypeAndMut { ty, .. }) | ty::TyBox(ty) if !type_is_sized(tcx, ty) => {
            path.push(FAT_PTR_ADDR);
            Some(path)
        },

        // Regular thin pointer: &T/&mut T/Box<T>
        ty::TyRef(..) | ty::TyBox(..) => Some(path),

        // Function pointer: `fn() -> i32`
        ty::TyFnPtr(_) => Some(path),

        // Is this the NonZero lang item wrapping a pointer or integer type?
        ty::TyStruct(def, substs) if Some(def.did) == tcx.lang_items.non_zero() => {
            let nonzero_fields = &def.struct_variant().fields;
            assert_eq!(nonzero_fields.len(), 1);
            let field_ty = monomorphize::field_ty(tcx, substs, &nonzero_fields[0]);
            match field_ty.sty {
                ty::TyRawPtr(ty::TypeAndMut { ty, .. }) if !type_is_sized(tcx, ty) => {
                    path.extend_from_slice(&[0, FAT_PTR_ADDR]);
                    Some(path)
                },
                ty::TyRawPtr(..) | ty::TyInt(..) | ty::TyUint(..) => {
                    path.push(0);
                    Some(path)
                },
                _ => None
            }
        },

        // Perhaps one of the fields of this struct is non-zero
        // let's recurse and find out
        ty::TyStruct(def, substs) => {
            for (j, field) in def.struct_variant().fields.iter().enumerate() {
                let field_ty = monomorphize::field_ty(tcx, substs, field);
                if let Some(mut fpath) = find_discr_field_candidate(tcx, field_ty, path.clone()) {
                    fpath.push(j);
                    return Some(fpath);
                }
            }
            None
        },

        // Perhaps one of the upvars of this struct is non-zero
        // Let's recurse and find out!
        ty::TyClosure(_, ref substs) => {
            for (j, &ty) in substs.upvar_tys.iter().enumerate() {
                if let Some(mut fpath) = find_discr_field_candidate(tcx, ty, path.clone()) {
                    fpath.push(j);
                    return Some(fpath);
                }
            }
            None
        },

        // Can we use one of the fields in this tuple?
        ty::TyTuple(ref tys) => {
            for (j, &ty) in tys.iter().enumerate() {
                if let Some(mut fpath) = find_discr_field_candidate(tcx, ty, path.clone()) {
                    fpath.push(j);
                    return Some(fpath);
                }
            }
            None
        },

        // Is this a fixed-size array of something non-zero
        // with at least one element?
        ty::TyArray(ety, d) if d > 0 => {
            if let Some(mut vpath) = find_discr_field_candidate(tcx, ety, path) {
                vpath.push(0);
                Some(vpath)
            } else {
                None
            }
        },

        // Anything else is not a pointer
        _ => None
    }
}

impl<'tcx> Case<'tcx> {
    fn is_zerolen<'a>(&self, cx: &CrateContext<'a, 'tcx>, scapegoat: Ty<'tcx>) -> bool {
        mk_struct(cx, &self.tys, false, scapegoat).size == 0
    }

    fn find_ptr<'a>(&self, cx: &CrateContext<'a, 'tcx>) -> Option<DiscrField> {
        for (i, &ty) in self.tys.iter().enumerate() {
            if let Some(mut path) = find_discr_field_candidate(cx.tcx(), ty, vec![]) {
                path.push(i);
                return Some(path);
            }
        }
        None
    }
}

fn get_cases<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                       adt: ty::AdtDef<'tcx>,
                       substs: &Substs<'tcx>)
                       -> Vec<Case<'tcx>> {
    adt.variants.iter().map(|vi| {
        let field_tys = vi.fields.iter().map(|field| {
            monomorphize::field_ty(tcx, substs, field)
        }).collect();
        Case { discr: Disr::from(vi.disr_val), tys: field_tys }
    }).collect()
}

fn mk_struct<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                       tys: &[Ty<'tcx>], packed: bool,
                       scapegoat: Ty<'tcx>)
                       -> Struct<'tcx> {
    let sized = tys.iter().all(|&ty| type_is_sized(cx.tcx(), ty));
    let lltys : Vec<Type> = if sized {
        tys.iter().map(|&ty| type_of::sizing_type_of(cx, ty)).collect()
    } else {
        tys.iter().filter(|&ty| type_is_sized(cx.tcx(), *ty))
           .map(|&ty| type_of::sizing_type_of(cx, ty)).collect()
    };

    ensure_struct_fits_in_address_space(cx, &lltys[..], packed, scapegoat);

    let llty_rec = Type::struct_(cx, &lltys[..], packed);
    Struct {
        size: machine::llsize_of_alloc(cx, llty_rec),
        align: machine::llalign_of_min(cx, llty_rec),
        sized: sized,
        packed: packed,
        fields: tys.to_vec(),
    }
}

fn mk_union<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                      tys: &[Ty<'tcx>], packed: bool,
                      _scapegoat: Ty<'tcx>)
                      -> Union<'tcx> {
    let mut min_size = 0;
    let mut align = 0;
    for llty in tys.iter().map(|&ty| type_of::sizing_type_of(cx, ty)) {
        let field_size = machine::llsize_of_alloc(cx, llty);
        if min_size < field_size {
            min_size = field_size;
        }
        let field_align = machine::llalign_of_min(cx, llty);
        if align < field_align {
            align = field_align;
        }
    }

    Union {
        min_size: min_size,
        align: if packed { 1 } else { align },
        packed: packed,
        fields: tys.to_vec(),
    }
}

#[derive(Debug)]
struct IntBounds {
    slo: i64,
    shi: i64,
    ulo: u64,
    uhi: u64
}

fn mk_cenum<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                      hint: Hint, bounds: &IntBounds)
                      -> Repr<'tcx> {
    let it = range_to_inttype(cx, hint, bounds);
    match it {
        attr::SignedInt(_) => CEnum(it, Disr(bounds.slo as u64), Disr(bounds.shi as u64)),
        attr::UnsignedInt(_) => CEnum(it, Disr(bounds.ulo), Disr(bounds.uhi))
    }
}

fn range_to_inttype(cx: &CrateContext, hint: Hint, bounds: &IntBounds) -> IntType {
    debug!("range_to_inttype: {:?} {:?}", hint, bounds);
    // Lists of sizes to try.  u64 is always allowed as a fallback.
    #[allow(non_upper_case_globals)]
    const choose_shortest: &'static [IntType] = &[
        attr::UnsignedInt(ast::UintTy::U8), attr::SignedInt(ast::IntTy::I8),
        attr::UnsignedInt(ast::UintTy::U16), attr::SignedInt(ast::IntTy::I16),
        attr::UnsignedInt(ast::UintTy::U32), attr::SignedInt(ast::IntTy::I32)];
    #[allow(non_upper_case_globals)]
    const at_least_32: &'static [IntType] = &[
        attr::UnsignedInt(ast::UintTy::U32), attr::SignedInt(ast::IntTy::I32)];

    let attempts;
    match hint {
        attr::ReprInt(span, ity) => {
            if !bounds_usable(cx, ity, bounds) {
                span_bug!(span, "representation hint insufficient for discriminant range")
            }
            return ity;
        }
        attr::ReprExtern => {
            attempts = match &cx.sess().target.target.arch[..] {
                // WARNING: the ARM EABI has two variants; the one corresponding to `at_least_32`
                // appears to be used on Linux and NetBSD, but some systems may use the variant
                // corresponding to `choose_shortest`.  However, we don't run on those yet...?
                "arm" => at_least_32,
                _ => at_least_32,
            }
        }
        attr::ReprAny => {
            attempts = choose_shortest;
        },
        attr::ReprPacked => {
            bug!("range_to_inttype: found ReprPacked on an enum");
        }
        attr::ReprSimd => {
            bug!("range_to_inttype: found ReprSimd on an enum");
        }
    }
    for &ity in attempts {
        if bounds_usable(cx, ity, bounds) {
            return ity;
        }
    }
    return attr::UnsignedInt(ast::UintTy::U64);
}

pub fn ll_inttype(cx: &CrateContext, ity: IntType) -> Type {
    match ity {
        attr::SignedInt(t) => Type::int_from_ty(cx, t),
        attr::UnsignedInt(t) => Type::uint_from_ty(cx, t)
    }
}

fn bounds_usable(cx: &CrateContext, ity: IntType, bounds: &IntBounds) -> bool {
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

pub fn ty_of_inttype<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ity: IntType) -> Ty<'tcx> {
    match ity {
        attr::SignedInt(t) => tcx.mk_mach_int(t),
        attr::UnsignedInt(t) => tcx.mk_mach_uint(t)
    }
}

// LLVM doesn't like types that don't fit in the address space
fn ensure_struct_fits_in_address_space<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                 fields: &[Type],
                                                 packed: bool,
                                                 scapegoat: Ty<'tcx>) {
    let mut offset = 0;
    for &llty in fields {
        // Invariant: offset < ccx.obj_size_bound() <= 1<<61
        if !packed {
            let type_align = machine::llalign_of_min(ccx, llty);
            offset = roundup(offset, type_align);
        }
        // type_align is a power-of-2, so still offset < ccx.obj_size_bound()
        // llsize_of_alloc(ccx, llty) is also less than ccx.obj_size_bound()
        // so the sum is less than 1<<62 (and therefore can't overflow).
        offset += machine::llsize_of_alloc(ccx, llty);

        if offset >= ccx.obj_size_bound() {
            ccx.report_overbig_object(scapegoat);
        }
    }
}

fn union_size_and_align(sts: &[Struct]) -> (machine::llsize, machine::llalign) {
    let size = sts.iter().map(|st| st.size).max().unwrap();
    let align = sts.iter().map(|st| st.align).max().unwrap();
    (roundup(size, align), align)
}

fn ensure_enum_fits_in_address_space<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                               fields: &[Struct],
                                               scapegoat: Ty<'tcx>) {
    let (total_size, _) = union_size_and_align(fields);

    if total_size >= ccx.obj_size_bound() {
        ccx.report_overbig_object(scapegoat);
    }
}


/// LLVM-level types are a little complicated.
///
/// C-like enums need to be actual ints, not wrapped in a struct,
/// because that changes the ABI on some platforms (see issue #10308).
///
/// For nominal types, in some cases, we need to use LLVM named structs
/// and fill in the actual contents in a second pass to prevent
/// unbounded recursion; see also the comments in `trans::type_of`.
pub fn type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, r: &Repr<'tcx>) -> Type {
    generic_type_of(cx, r, None, false, false)
}


// Pass dst=true if the type you are passing is a DST. Yes, we could figure
// this out, but if you call this on an unsized type without realising it, you
// are going to get the wrong type (it will not include the unsized parts of it).
pub fn sizing_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                r: &Repr<'tcx>, dst: bool) -> Type {
    generic_type_of(cx, r, None, true, dst)
}

pub fn incomplete_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                    r: &Repr<'tcx>, name: &str) -> Type {
    generic_type_of(cx, r, Some(name), false, false)
}

pub fn finish_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                r: &Repr<'tcx>, llty: &mut Type) {
    match *r {
        CEnum(..) | General(..) | UntaggedUnion(..) | RawNullablePointer { .. } => { }
        Univariant(ref st) | StructWrappedNullablePointer { nonnull: ref st, .. } =>
            llty.set_struct_body(&struct_llfields(cx, st, false, false),
                                 st.packed)
    }
}

fn generic_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                             r: &Repr<'tcx>,
                             name: Option<&str>,
                             sizing: bool,
                             dst: bool) -> Type {
    debug!("adt::generic_type_of r: {:?} name: {:?} sizing: {} dst: {}",
           r, name, sizing, dst);
    match *r {
        CEnum(ity, _, _) => ll_inttype(cx, ity),
        RawNullablePointer { nnty, .. } =>
            type_of::sizing_type_of(cx, nnty),
        StructWrappedNullablePointer { nonnull: ref st, .. } => {
            match name {
                None => {
                    Type::struct_(cx, &struct_llfields(cx, st, sizing, dst),
                                  st.packed)
                }
                Some(name) => {
                    assert_eq!(sizing, false);
                    Type::named_struct(cx, name)
                }
            }
        }
        Univariant(ref st) => {
            match name {
                None => {
                    let fields = struct_llfields(cx, st, sizing, dst);
                    Type::struct_(cx, &fields, st.packed)
                }
                Some(name) => {
                    // Hypothesis: named_struct's can never need a
                    // drop flag. (... needs validation.)
                    assert_eq!(sizing, false);
                    Type::named_struct(cx, name)
                }
            }
        }
        UntaggedUnion(ref un) => {
            // Use alignment-sized ints to fill all the union storage.
            let (size, align) = (roundup(un.min_size, un.align), un.align);

            let align_s = align as u64;
            assert_eq!(size % align_s, 0); // Ensure division in align_units comes out evenly
            let align_units = size / align_s;
            let fill_ty = match align_s {
                1 => Type::array(&Type::i8(cx), align_units),
                2 => Type::array(&Type::i16(cx), align_units),
                4 => Type::array(&Type::i32(cx), align_units),
                8 if machine::llalign_of_min(cx, Type::i64(cx)) == 8 =>
                                 Type::array(&Type::i64(cx), align_units),
                a if a.count_ones() == 1 => Type::array(&Type::vector(&Type::i32(cx), a / 4),
                                                              align_units),
                _ => bug!("unsupported union alignment: {}", align)
            };
            match name {
                None => {
                    Type::struct_(cx, &[fill_ty], un.packed)
                }
                Some(name) => {
                    let mut llty = Type::named_struct(cx, name);
                    llty.set_struct_body(&[fill_ty], un.packed);
                    llty
                }
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
            let (size, align) = union_size_and_align(&sts[..]);
            let align_s = align as u64;
            let discr_ty = ll_inttype(cx, ity);
            let discr_size = machine::llsize_of_alloc(cx, discr_ty);
            let padded_discr_size = roundup(discr_size, align);
            assert_eq!(size % align_s, 0); // Ensure division in align_units comes out evenly
            let align_units = (size - padded_discr_size) / align_s;
            let fill_ty = match align_s {
                1 => Type::array(&Type::i8(cx), align_units),
                2 => Type::array(&Type::i16(cx), align_units),
                4 => Type::array(&Type::i32(cx), align_units),
                8 if machine::llalign_of_min(cx, Type::i64(cx)) == 8 =>
                                 Type::array(&Type::i64(cx), align_units),
                a if a.count_ones() == 1 => Type::array(&Type::vector(&Type::i32(cx), a / 4),
                                                              align_units),
                _ => bug!("unsupported enum alignment: {}", align)
            };
            assert_eq!(machine::llalign_of_min(cx, fill_ty), align);
            assert_eq!(padded_discr_size % discr_size, 0); // Ensure discr_ty can fill pad evenly
            let fields: Vec<Type> =
                [discr_ty,
                 Type::array(&discr_ty, (padded_discr_size - discr_size)/discr_size),
                 fill_ty].iter().cloned().collect();
            match name {
                None => {
                    Type::struct_(cx, &fields[..], false)
                }
                Some(name) => {
                    let mut llty = Type::named_struct(cx, name);
                    llty.set_struct_body(&fields[..], false);
                    llty
                }
            }
        }
    }
}

fn struct_llfields<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, st: &Struct<'tcx>,
                             sizing: bool, dst: bool) -> Vec<Type> {
    if sizing {
        st.fields.iter().filter(|&ty| !dst || type_is_sized(cx.tcx(), *ty))
            .map(|&ty| type_of::sizing_type_of(cx, ty)).collect()
    } else {
        st.fields.iter().map(|&ty| type_of::in_memory_type_of(cx, ty)).collect()
    }
}

/// Obtain a representation of the discriminant sufficient to translate
/// destructuring; this may or may not involve the actual discriminant.
pub fn trans_switch<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                r: &Repr<'tcx>,
                                scrutinee: ValueRef,
                                range_assert: bool)
                                -> (BranchKind, Option<ValueRef>) {
    match *r {
        CEnum(..) | General(..) |
        RawNullablePointer { .. } | StructWrappedNullablePointer { .. } => {
            (BranchKind::Switch, Some(trans_get_discr(bcx, r, scrutinee, None, range_assert)))
        }
        Univariant(..) | UntaggedUnion(..) => {
            // N.B.: Univariant means <= 1 enum variants (*not* == 1 variants).
            (BranchKind::Single, None)
        }
    }
}

pub fn is_discr_signed<'tcx>(r: &Repr<'tcx>) -> bool {
    match *r {
        CEnum(ity, _, _) => ity.is_signed(),
        General(ity, _) => ity.is_signed(),
        Univariant(..) | UntaggedUnion(..) => false,
        RawNullablePointer { .. } => false,
        StructWrappedNullablePointer { .. } => false,
    }
}

/// Obtain the actual discriminant of a value.
pub fn trans_get_discr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, r: &Repr<'tcx>,
                                   scrutinee: ValueRef, cast_to: Option<Type>,
                                   range_assert: bool)
    -> ValueRef {
    debug!("trans_get_discr r: {:?}", r);
    let val = match *r {
        CEnum(ity, min, max) => {
            load_discr(bcx, ity, scrutinee, min, max, range_assert)
        }
        General(ity, ref cases) => {
            let ptr = StructGEP(bcx, scrutinee, 0);
            load_discr(bcx, ity, ptr, Disr(0), Disr(cases.len() as u64 - 1),
                       range_assert)
        }
        Univariant(..) | UntaggedUnion(..) => C_u8(bcx.ccx(), 0),
        RawNullablePointer { nndiscr, nnty, .. } =>  {
            let cmp = if nndiscr == Disr(0) { IntEQ } else { IntNE };
            let llptrty = type_of::sizing_type_of(bcx.ccx(), nnty);
            ICmp(bcx, cmp, Load(bcx, scrutinee), C_null(llptrty), DebugLoc::None)
        }
        StructWrappedNullablePointer { nndiscr, ref discrfield, .. } => {
            struct_wrapped_nullable_bitdiscr(bcx, nndiscr, discrfield, scrutinee)
        }
    };
    match cast_to {
        None => val,
        Some(llty) => if is_discr_signed(r) { SExt(bcx, val, llty) } else { ZExt(bcx, val, llty) }
    }
}

fn struct_wrapped_nullable_bitdiscr(bcx: Block, nndiscr: Disr, discrfield: &DiscrField,
                                    scrutinee: ValueRef) -> ValueRef {
    let llptrptr = GEPi(bcx, scrutinee, &discrfield[..]);
    let llptr = Load(bcx, llptrptr);
    let cmp = if nndiscr == Disr(0) { IntEQ } else { IntNE };
    ICmp(bcx, cmp, llptr, C_null(val_ty(llptr)), DebugLoc::None)
}

/// Helper for cases where the discriminant is simply loaded.
fn load_discr(bcx: Block, ity: IntType, ptr: ValueRef, min: Disr, max: Disr,
              range_assert: bool)
    -> ValueRef {
    let llty = ll_inttype(bcx.ccx(), ity);
    assert_eq!(val_ty(ptr), llty.ptr_to());
    let bits = machine::llbitsize_of_real(bcx.ccx(), llty);
    assert!(bits <= 64);
    let bits = bits as usize;
    let mask = Disr(!0u64 >> (64 - bits));
    // For a (max) discr of -1, max will be `-1 as usize`, which overflows.
    // However, that is fine here (it would still represent the full range),
    if max.wrapping_add(Disr(1)) & mask == min & mask || !range_assert {
        // i.e., if the range is everything.  The lo==hi case would be
        // rejected by the LLVM verifier (it would mean either an
        // empty set, which is impossible, or the entire range of the
        // type, which is pointless).
        Load(bcx, ptr)
    } else {
        // llvm::ConstantRange can deal with ranges that wrap around,
        // so an overflow on (max + 1) is fine.
        LoadRangeAssert(bcx, ptr, min.0, max.0.wrapping_add(1), /* signed: */ True)
    }
}

/// Yield information about how to dispatch a case of the
/// discriminant-like value returned by `trans_switch`.
///
/// This should ideally be less tightly tied to `_match`.
pub fn trans_case<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, r: &Repr, discr: Disr)
                              -> ValueRef {
    match *r {
        CEnum(ity, _, _) => {
            C_integral(ll_inttype(bcx.ccx(), ity), discr.0, true)
        }
        General(ity, _) => {
            C_integral(ll_inttype(bcx.ccx(), ity), discr.0, true)
        }
        Univariant(..) | UntaggedUnion(..) => {
            bug!("no cases for univariants, structs or unions")
        }
        RawNullablePointer { .. } |
        StructWrappedNullablePointer { .. } => {
            assert!(discr == Disr(0) || discr == Disr(1));
            C_bool(bcx.ccx(), discr != Disr(0))
        }
    }
}

/// Set the discriminant for a new value of the given case of the given
/// representation.
pub fn trans_set_discr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, r: &Repr<'tcx>,
                                   val: ValueRef, discr: Disr) {
    match *r {
        CEnum(ity, min, max) => {
            assert_discr_in_range(ity, min, max, discr);
            Store(bcx, C_integral(ll_inttype(bcx.ccx(), ity), discr.0, true),
                  val);
        }
        General(ity, _) => {
            Store(bcx, C_integral(ll_inttype(bcx.ccx(), ity), discr.0, true),
                  StructGEP(bcx, val, 0));
        }
        Univariant(_) => {
            assert_eq!(discr, Disr(0));
        }
        UntaggedUnion(..) => {
            assert_eq!(discr, Disr(0));
        }
        RawNullablePointer { nndiscr, nnty, ..} => {
            if discr != nndiscr {
                let llptrty = type_of::sizing_type_of(bcx.ccx(), nnty);
                Store(bcx, C_null(llptrty), val);
            }
        }
        StructWrappedNullablePointer { nndiscr, ref discrfield, .. } => {
            if discr != nndiscr {
                let llptrptr = GEPi(bcx, val, &discrfield[..]);
                let llptrty = val_ty(llptrptr).element_type();
                Store(bcx, C_null(llptrty), llptrptr);
            }
        }
    }
}

fn assert_discr_in_range(ity: IntType, min: Disr, max: Disr, discr: Disr) {
    match ity {
        attr::UnsignedInt(_) => {
            assert!(min <= discr);
            assert!(discr <= max);
        },
        attr::SignedInt(_) => {
            assert!(min.0 as i64 <= discr.0 as i64);
            assert!(discr.0 as i64 <= max.0 as i64);
        },
    }
}

/// Access a field, at a point when the value's case is known.
pub fn trans_field_ptr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, r: &Repr<'tcx>,
                                   val: MaybeSizedValue, discr: Disr, ix: usize) -> ValueRef {
    trans_field_ptr_builder(&bcx.build(), r, val, discr, ix)
}

/// Access a field, at a point when the value's case is known.
pub fn trans_field_ptr_builder<'blk, 'tcx>(bcx: &BlockAndBuilder<'blk, 'tcx>,
                                           r: &Repr<'tcx>,
                                           val: MaybeSizedValue,
                                           discr: Disr, ix: usize)
                                           -> ValueRef {
    // Note: if this ever needs to generate conditionals (e.g., if we
    // decide to do some kind of cdr-coding-like non-unique repr
    // someday), it will need to return a possibly-new bcx as well.
    match *r {
        CEnum(..) => {
            bug!("element access in C-like enum")
        }
        Univariant(ref st) => {
            assert_eq!(discr, Disr(0));
            struct_field_ptr(bcx, st, val, ix, false)
        }
        General(_, ref cases) => {
            struct_field_ptr(bcx, &cases[discr.0 as usize], val, ix + 1, true)
        }
        UntaggedUnion(ref un) => {
            let ty = type_of::in_memory_type_of(bcx.ccx(), un.fields[ix]);
            if bcx.is_unreachable() { return C_undef(ty.ptr_to()); }
            bcx.pointercast(val.value, ty.ptr_to())
        }
        RawNullablePointer { nndiscr, ref nullfields, .. } |
        StructWrappedNullablePointer { nndiscr, ref nullfields, .. } if discr != nndiscr => {
            // The unit-like case might have a nonzero number of unit-like fields.
            // (e.d., Result of Either with (), as one side.)
            let ty = type_of::type_of(bcx.ccx(), nullfields[ix]);
            assert_eq!(machine::llsize_of_alloc(bcx.ccx(), ty), 0);
            // The contents of memory at this pointer can't matter, but use
            // the value that's "reasonable" in case of pointer comparison.
            if bcx.is_unreachable() { return C_undef(ty.ptr_to()); }
            bcx.pointercast(val.value, ty.ptr_to())
        }
        RawNullablePointer { nndiscr, nnty, .. } => {
            assert_eq!(ix, 0);
            assert_eq!(discr, nndiscr);
            let ty = type_of::type_of(bcx.ccx(), nnty);
            if bcx.is_unreachable() { return C_undef(ty.ptr_to()); }
            bcx.pointercast(val.value, ty.ptr_to())
        }
        StructWrappedNullablePointer { ref nonnull, nndiscr, .. } => {
            assert_eq!(discr, nndiscr);
            struct_field_ptr(bcx, nonnull, val, ix, false)
        }
    }
}

fn struct_field_ptr<'blk, 'tcx>(bcx: &BlockAndBuilder<'blk, 'tcx>,
                                st: &Struct<'tcx>, val: MaybeSizedValue,
                                ix: usize, needs_cast: bool) -> ValueRef {
    let ccx = bcx.ccx();
    let fty = st.fields[ix];
    let ll_fty = type_of::in_memory_type_of(bcx.ccx(), fty);
    if bcx.is_unreachable() {
        return C_undef(ll_fty.ptr_to());
    }

    let ptr_val = if needs_cast {
        let fields = st.fields.iter().map(|&ty| {
            type_of::in_memory_type_of(ccx, ty)
        }).collect::<Vec<_>>();
        let real_ty = Type::struct_(ccx, &fields[..], st.packed);
        bcx.pointercast(val.value, real_ty.ptr_to())
    } else {
        val.value
    };

    // Simple case - we can just GEP the field
    //   * First field - Always aligned properly
    //   * Packed struct - There is no alignment padding
    //   * Field is sized - pointer is properly aligned already
    if ix == 0 || st.packed || type_is_sized(bcx.tcx(), fty) {
        return bcx.struct_gep(ptr_val, ix);
    }

    // If the type of the last field is [T] or str, then we don't need to do
    // any adjusments
    match fty.sty {
        ty::TySlice(..) | ty::TyStr => {
            return bcx.struct_gep(ptr_val, ix);
        }
        _ => ()
    }

    // There's no metadata available, log the case and just do the GEP.
    if !val.has_meta() {
        debug!("Unsized field `{}`, of `{:?}` has no metadata for adjustment",
               ix, Value(ptr_val));
        return bcx.struct_gep(ptr_val, ix);
    }

    let dbloc = DebugLoc::None;

    // We need to get the pointer manually now.
    // We do this by casting to a *i8, then offsetting it by the appropriate amount.
    // We do this instead of, say, simply adjusting the pointer from the result of a GEP
    // because the field may have an arbitrary alignment in the LLVM representation
    // anyway.
    //
    // To demonstrate:
    //   struct Foo<T: ?Sized> {
    //      x: u16,
    //      y: T
    //   }
    //
    // The type Foo<Foo<Trait>> is represented in LLVM as { u16, { u16, u8 }}, meaning that
    // the `y` field has 16-bit alignment.

    let meta = val.meta;

    // Calculate the unaligned offset of the unsized field.
    let mut offset = 0;
    for &ty in &st.fields[0..ix] {
        let llty = type_of::sizing_type_of(ccx, ty);
        let type_align = type_of::align_of(ccx, ty);
        offset = roundup(offset, type_align);
        offset += machine::llsize_of_alloc(ccx, llty);
    }
    let unaligned_offset = C_uint(bcx.ccx(), offset);

    // Get the alignment of the field
    let (_, align) = glue::size_and_align_of_dst(bcx, fty, meta);

    // Bump the unaligned offset up to the appropriate alignment using the
    // following expression:
    //
    //   (unaligned offset + (align - 1)) & -align

    // Calculate offset
    dbloc.apply(bcx.fcx());
    let align_sub_1 = bcx.sub(align, C_uint(bcx.ccx(), 1u64));
    let offset = bcx.and(bcx.add(unaligned_offset, align_sub_1),
                         bcx.neg(align));

    debug!("struct_field_ptr: DST field offset: {:?}", Value(offset));

    // Cast and adjust pointer
    let byte_ptr = bcx.pointercast(ptr_val, Type::i8p(bcx.ccx()));
    let byte_ptr = bcx.gep(byte_ptr, &[offset]);

    // Finally, cast back to the type expected
    let ll_fty = type_of::in_memory_type_of(bcx.ccx(), fty);
    debug!("struct_field_ptr: Field type is {:?}", ll_fty);
    bcx.pointercast(byte_ptr, ll_fty.ptr_to())
}

/// Construct a constant value, suitable for initializing a
/// GlobalVariable, given a case and constant values for its fields.
/// Note that this may have a different LLVM type (and different
/// alignment!) from the representation's `type_of`, so it needs a
/// pointer cast before use.
///
/// The LLVM type system does not directly support unions, and only
/// pointers can be bitcast, so a constant (and, by extension, the
/// GlobalVariable initialized by it) will have a type that can vary
/// depending on which case of an enum it is.
///
/// To understand the alignment situation, consider `enum E { V64(u64),
/// V32(u32, u32) }` on Windows.  The type has 8-byte alignment to
/// accommodate the u64, but `V32(x, y)` would have LLVM type `{i32,
/// i32, i32}`, which is 4-byte aligned.
///
/// Currently the returned value has the same size as the type, but
/// this could be changed in the future to avoid allocating unnecessary
/// space after values of shorter-than-maximum cases.
pub fn trans_const<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, r: &Repr<'tcx>, discr: Disr,
                             vals: &[ValueRef]) -> ValueRef {
    match *r {
        CEnum(ity, min, max) => {
            assert_eq!(vals.len(), 0);
            assert_discr_in_range(ity, min, max, discr);
            C_integral(ll_inttype(ccx, ity), discr.0, true)
        }
        General(ity, ref cases) => {
            let case = &cases[discr.0 as usize];
            let (max_sz, _) = union_size_and_align(&cases[..]);
            let lldiscr = C_integral(ll_inttype(ccx, ity), discr.0 as u64, true);
            let mut f = vec![lldiscr];
            f.extend_from_slice(vals);
            let mut contents = build_const_struct(ccx, case, &f[..]);
            contents.extend_from_slice(&[padding(ccx, max_sz - case.size)]);
            C_struct(ccx, &contents[..], false)
        }
        UntaggedUnion(ref un) => {
            assert_eq!(discr, Disr(0));
            let contents = build_const_union(ccx, un, vals[0]);
            C_struct(ccx, &contents, un.packed)
        }
        Univariant(ref st) => {
            assert_eq!(discr, Disr(0));
            let contents = build_const_struct(ccx, st, vals);
            C_struct(ccx, &contents[..], st.packed)
        }
        RawNullablePointer { nndiscr, nnty, .. } => {
            if discr == nndiscr {
                assert_eq!(vals.len(), 1);
                vals[0]
            } else {
                C_null(type_of::sizing_type_of(ccx, nnty))
            }
        }
        StructWrappedNullablePointer { ref nonnull, nndiscr, .. } => {
            if discr == nndiscr {
                C_struct(ccx, &build_const_struct(ccx,
                                                 nonnull,
                                                 vals),
                         false)
            } else {
                let vals = nonnull.fields.iter().map(|&ty| {
                    // Always use null even if it's not the `discrfield`th
                    // field; see #8506.
                    C_null(type_of::sizing_type_of(ccx, ty))
                }).collect::<Vec<ValueRef>>();
                C_struct(ccx, &build_const_struct(ccx,
                                                 nonnull,
                                                 &vals[..]),
                         false)
            }
        }
    }
}

/// Compute struct field offsets relative to struct begin.
fn compute_struct_field_offsets<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                          st: &Struct<'tcx>) -> Vec<u64> {
    let mut offsets = vec!();

    let mut offset = 0;
    for &ty in &st.fields {
        let llty = type_of::sizing_type_of(ccx, ty);
        if !st.packed {
            let type_align = type_of::align_of(ccx, ty);
            offset = roundup(offset, type_align);
        }
        offsets.push(offset);
        offset += machine::llsize_of_alloc(ccx, llty);
    }
    assert_eq!(st.fields.len(), offsets.len());
    offsets
}

/// Building structs is a little complicated, because we might need to
/// insert padding if a field's value is less aligned than its type.
///
/// Continuing the example from `trans_const`, a value of type `(u32,
/// E)` should have the `E` at offset 8, but if that field's
/// initializer is 4-byte aligned then simply translating the tuple as
/// a two-element struct will locate it at offset 4, and accesses to it
/// will read the wrong memory.
fn build_const_struct<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                st: &Struct<'tcx>, vals: &[ValueRef])
                                -> Vec<ValueRef> {
    assert_eq!(vals.len(), st.fields.len());

    let target_offsets = compute_struct_field_offsets(ccx, st);

    // offset of current value
    let mut offset = 0;
    let mut cfields = Vec::new();
    for (&val, target_offset) in vals.iter().zip(target_offsets) {
        if !st.packed {
            let val_align = machine::llalign_of_min(ccx, val_ty(val));
            offset = roundup(offset, val_align);
        }
        if offset != target_offset {
            cfields.push(padding(ccx, target_offset - offset));
            offset = target_offset;
        }
        assert!(!is_undef(val));
        cfields.push(val);
        offset += machine::llsize_of_alloc(ccx, val_ty(val));
    }

    assert!(st.sized && offset <= st.size);
    if offset != st.size {
        cfields.push(padding(ccx, st.size - offset));
    }

    cfields
}

fn build_const_union<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                               un: &Union<'tcx>,
                               field_val: ValueRef)
                               -> Vec<ValueRef> {
    let mut cfields = vec![field_val];

    let offset = machine::llsize_of_alloc(ccx, val_ty(field_val));
    let size = roundup(un.min_size, un.align);
    if offset != size {
        cfields.push(padding(ccx, size - offset));
    }

    cfields
}

fn padding(ccx: &CrateContext, size: u64) -> ValueRef {
    C_undef(Type::array(&Type::i8(ccx), size))
}

// FIXME this utility routine should be somewhere more general
#[inline]
fn roundup(x: u64, a: u32) -> u64 { let a = a as u64; ((x + (a - 1)) / a) * a }

/// Extract a field of a constant value, as appropriate for its
/// representation.
///
/// (Not to be confused with `common::const_get_elt`, which operates on
/// raw LLVM-level structs and arrays.)
pub fn const_get_field(r: &Repr, val: ValueRef, _discr: Disr,
                       ix: usize) -> ValueRef {
    match *r {
        CEnum(..) => bug!("element access in C-like enum const"),
        Univariant(..) => const_struct_field(val, ix),
        UntaggedUnion(..) => const_struct_field(val, 0),
        General(..) => const_struct_field(val, ix + 1),
        RawNullablePointer { .. } => {
            assert_eq!(ix, 0);
            val
        },
        StructWrappedNullablePointer{ .. } => const_struct_field(val, ix)
    }
}

/// Extract field of struct-like const, skipping our alignment padding.
fn const_struct_field(val: ValueRef, ix: usize) -> ValueRef {
    // Get the ix-th non-undef element of the struct.
    let mut real_ix = 0; // actual position in the struct
    let mut ix = ix; // logical index relative to real_ix
    let mut field;
    loop {
        loop {
            field = const_get_elt(val, &[real_ix]);
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
