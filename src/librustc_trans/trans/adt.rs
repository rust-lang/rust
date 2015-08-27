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

use std::rc::Rc;

use llvm::{ValueRef, True, IntEQ, IntNE};
use back::abi::FAT_PTR_ADDR;
use middle::subst;
use middle::ty::{self, Ty};
use middle::ty::Disr;
use syntax::ast;
use syntax::attr;
use syntax::attr::IntType;
use trans::_match;
use trans::build::*;
use trans::cleanup;
use trans::cleanup::CleanupMethods;
use trans::common::*;
use trans::datum;
use trans::debuginfo::DebugLoc;
use trans::machine;
use trans::monomorphize;
use trans::type_::Type;
use trans::type_of;

type Hint = attr::ReprAttr;

// Representation of the context surrounding an unsized type. I want
// to be able to track the drop flags that are injected by trans.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TypeContext {
    prefix: Type,
    needs_drop_flag: bool,
}

impl TypeContext {
    pub fn prefix(&self) -> Type { self.prefix }
    pub fn needs_drop_flag(&self) -> bool { self.needs_drop_flag }

    fn direct(t: Type) -> TypeContext {
        TypeContext { prefix: t, needs_drop_flag: false }
    }
    fn may_need_drop_flag(t: Type, needs_drop_flag: bool) -> TypeContext {
        TypeContext { prefix: t, needs_drop_flag: needs_drop_flag }
    }
    pub fn to_string(self) -> String {
        let TypeContext { prefix, needs_drop_flag } = self;
        format!("TypeContext {{ prefix: {}, needs_drop_flag: {} }}",
                prefix.to_string(), needs_drop_flag)
    }
}

/// Representations.
#[derive(Eq, PartialEq, Debug)]
pub enum Repr<'tcx> {
    /// C-like enums; basically an int.
    CEnum(IntType, Disr, Disr), // discriminant range (signedness based on the IntType)
    /// Single-case variants, and structs/tuples/records.
    ///
    /// Structs with destructors need a dynamic destroyedness flag to
    /// avoid running the destructor too many times; this is included
    /// in the `Struct` if present.
    /// (The flag if nonzero, represents the initialization value to use;
    ///  if zero, then use no flag at all.)
    Univariant(Struct<'tcx>, u8),
    /// General-case enums: for each case there is a struct, and they
    /// all start with a field for the discriminant.
    ///
    /// Types with destructors need a dynamic destroyedness flag to
    /// avoid running the destructor too many times; the last argument
    /// indicates whether such a flag is present.
    /// (The flag, if nonzero, represents the initialization value to use;
    ///  if zero, then use no flag at all.)
    General(IntType, Vec<Struct<'tcx>>, u8),
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

/// Convenience for `represent_type`.  There should probably be more or
/// these, for places in trans where the `Ty` isn't directly
/// available.
pub fn represent_node<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                  node: ast::NodeId) -> Rc<Repr<'tcx>> {
    represent_type(bcx.ccx(), node_id_type(bcx, node))
}

/// Decides how to represent a given type.
pub fn represent_type<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                t: Ty<'tcx>)
                                -> Rc<Repr<'tcx>> {
    debug!("Representing: {}", t);
    match cx.adt_reprs().borrow().get(&t) {
        Some(repr) => return repr.clone(),
        None => {}
    }

    let repr = Rc::new(represent_type_uncached(cx, t));
    debug!("Represented as: {:?}", repr);
    cx.adt_reprs().borrow_mut().insert(t, repr.clone());
    repr
}

macro_rules! repeat_u8_as_u32 {
    ($name:expr) => { (($name as u32) << 24 |
                       ($name as u32) << 16 |
                       ($name as u32) <<  8 |
                       ($name as u32)) }
}
macro_rules! repeat_u8_as_u64 {
    ($name:expr) => { ((repeat_u8_as_u32!($name) as u64) << 32 |
                       (repeat_u8_as_u32!($name) as u64)) }
}

/// `DTOR_NEEDED_HINT` is a stack-local hint that just means
/// "we do not know whether the destructor has run or not; check the
/// drop-flag embedded in the value itself."
pub const DTOR_NEEDED_HINT: u8 = 0x3d;

/// `DTOR_MOVED_HINT` is a stack-local hint that means "this value has
/// definitely been moved; you do not need to run its destructor."
///
/// (However, for now, such values may still end up being explicitly
/// zeroed by the generated code; this is the distinction between
/// `datum::DropFlagInfo::ZeroAndMaintain` versus
/// `datum::DropFlagInfo::DontZeroJustUse`.)
pub const DTOR_MOVED_HINT: u8 = 0x2d;

pub const DTOR_NEEDED: u8 = 0xd4;
pub const DTOR_NEEDED_U32: u32 = repeat_u8_as_u32!(DTOR_NEEDED);
pub const DTOR_NEEDED_U64: u64 = repeat_u8_as_u64!(DTOR_NEEDED);
#[allow(dead_code)]
pub fn dtor_needed_usize(ccx: &CrateContext) -> usize {
    match &ccx.tcx().sess.target.target.target_pointer_width[..] {
        "32" => DTOR_NEEDED_U32 as usize,
        "64" => DTOR_NEEDED_U64 as usize,
        tws => panic!("Unsupported target word size for int: {}", tws),
    }
}

pub const DTOR_DONE: u8 = 0x1d;
pub const DTOR_DONE_U32: u32 = repeat_u8_as_u32!(DTOR_DONE);
pub const DTOR_DONE_U64: u64 = repeat_u8_as_u64!(DTOR_DONE);
#[allow(dead_code)]
pub fn dtor_done_usize(ccx: &CrateContext) -> usize {
    match &ccx.tcx().sess.target.target.target_pointer_width[..] {
        "32" => DTOR_DONE_U32 as usize,
        "64" => DTOR_DONE_U64 as usize,
        tws => panic!("Unsupported target word size for int: {}", tws),
    }
}

fn dtor_to_init_u8(dtor: bool) -> u8 {
    if dtor { DTOR_NEEDED } else { 0 }
}

pub trait GetDtorType<'tcx> { fn dtor_type(&self) -> Ty<'tcx>; }
impl<'tcx> GetDtorType<'tcx> for ty::ctxt<'tcx> {
    fn dtor_type(&self) -> Ty<'tcx> { self.types.u8 }
}

fn dtor_active(flag: u8) -> bool {
    flag != 0
}

fn represent_type_uncached<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                     t: Ty<'tcx>) -> Repr<'tcx> {
    match t.sty {
        ty::TyTuple(ref elems) => {
            Univariant(mk_struct(cx, &elems[..], false, t), 0)
        }
        ty::TyStruct(def, substs) => {
            let mut ftys = def.struct_variant().fields.iter().map(|field| {
                monomorphize::field_ty(cx.tcx(), substs, field)
            }).collect::<Vec<_>>();
            let packed = cx.tcx().lookup_packed(def.did);
            let dtor = def.dtor_kind().has_drop_flag();
            if dtor {
                ftys.push(cx.tcx().dtor_type());
            }

            Univariant(mk_struct(cx, &ftys[..], packed, t), dtor_to_init_u8(dtor))
        }
        ty::TyClosure(_, ref substs) => {
            Univariant(mk_struct(cx, &substs.upvar_tys, false, t), 0)
        }
        ty::TyEnum(def, substs) => {
            let cases = get_cases(cx.tcx(), def, substs);
            let hint = *cx.tcx().lookup_repr_hints(def.did).get(0)
                .unwrap_or(&attr::ReprAny);

            let dtor = def.dtor_kind().has_drop_flag();

            if cases.is_empty() {
                // Uninhabitable; represent as unit
                // (Typechecking will reject discriminant-sizing attrs.)
                assert_eq!(hint, attr::ReprAny);
                let ftys = if dtor { vec!(cx.tcx().dtor_type()) } else { vec!() };
                return Univariant(mk_struct(cx, &ftys[..], false, t),
                                  dtor_to_init_u8(dtor));
            }

            if !dtor && cases.iter().all(|c| c.tys.is_empty()) {
                // All bodies empty -> intlike
                let discrs: Vec<u64> = cases.iter().map(|c| c.discr).collect();
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
                cx.sess().bug(&format!("non-C-like enum {} with specified \
                                        discriminants",
                                       cx.tcx().item_path_str(def.did)));
            }

            if cases.len() == 1 {
                // Equivalent to a struct/tuple/newtype.
                // (Typechecking will reject discriminant-sizing attrs.)
                assert_eq!(hint, attr::ReprAny);
                let mut ftys = cases[0].tys.clone();
                if dtor { ftys.push(cx.tcx().dtor_type()); }
                return Univariant(mk_struct(cx, &ftys[..], false, t),
                                  dtor_to_init_u8(dtor));
            }

            if !dtor && cases.len() == 2 && hint == attr::ReprAny {
                // Nullable pointer optimization
                let mut discr = 0;
                while discr < 2 {
                    if cases[1 - discr].is_zerolen(cx, t) {
                        let st = mk_struct(cx, &cases[discr].tys,
                                           false, t);
                        match cases[discr].find_ptr(cx) {
                            Some(ref df) if df.len() == 1 && st.fields.len() == 1 => {
                                return RawNullablePointer {
                                    nndiscr: discr as Disr,
                                    nnty: st.fields[0],
                                    nullfields: cases[1 - discr].tys.clone()
                                };
                            }
                            Some(mut discrfield) => {
                                discrfield.push(0);
                                discrfield.reverse();
                                return StructWrappedNullablePointer {
                                    nndiscr: discr as Disr,
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
                ftys.push_all(&c.tys);
                if dtor { ftys.push(cx.tcx().dtor_type()); }
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
            let ity = if use_align {
                // Use the overall alignment
                match align {
                    1 => attr::UnsignedInt(ast::TyU8),
                    2 => attr::UnsignedInt(ast::TyU16),
                    4 => attr::UnsignedInt(ast::TyU32),
                    8 if machine::llalign_of_min(cx, Type::i64(cx)) == 8 =>
                        attr::UnsignedInt(ast::TyU64),
                    _ => min_ity // use min_ity as a fallback
                }
            } else {
                min_ity
            };

            let fields : Vec<_> = cases.iter().map(|c| {
                let mut ftys = vec!(ty_of_inttype(cx.tcx(), ity));
                ftys.push_all(&c.tys);
                if dtor { ftys.push(cx.tcx().dtor_type()); }
                mk_struct(cx, &ftys[..], false, t)
            }).collect();

            ensure_enum_fits_in_address_space(cx, &fields[..], t);

            General(ity, fields, dtor_to_init_u8(dtor))
        }
        _ => cx.sess().bug(&format!("adt::represent_type called on non-ADT type: {}", t))
    }
}

// this should probably all be in ty
struct Case<'tcx> {
    discr: Disr,
    tys: Vec<Ty<'tcx>>
}

/// This represents the (GEP) indices to follow to get to the discriminant field
pub type DiscrField = Vec<usize>;

fn find_discr_field_candidate<'tcx>(tcx: &ty::ctxt<'tcx>,
                                    ty: Ty<'tcx>,
                                    mut path: DiscrField) -> Option<DiscrField> {
    match ty.sty {
        // Fat &T/&mut T/Box<T> i.e. T is [T], str, or Trait
        ty::TyRef(_, ty::TypeAndMut { ty, .. }) | ty::TyBox(ty) if !type_is_sized(tcx, ty) => {
            path.push(FAT_PTR_ADDR);
            Some(path)
        },

        // Regular thin pointer: &T/&mut T/Box<T>
        ty::TyRef(..) | ty::TyBox(..) => Some(path),

        // Functions are just pointers
        ty::TyBareFn(..) => Some(path),

        // Is this the NonZero lang item wrapping a pointer or integer type?
        ty::TyStruct(def, substs) if Some(def.did) == tcx.lang_items.non_zero() => {
            let nonzero_fields = &def.struct_variant().fields;
            assert_eq!(nonzero_fields.len(), 1);
            let field_ty = monomorphize::field_ty(tcx, substs, &nonzero_fields[0]);
            match field_ty.sty {
                ty::TyRawPtr(ty::TypeAndMut { ty, .. }) if !type_is_sized(tcx, ty) => {
                    path.push_all(&[0, FAT_PTR_ADDR]);
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

fn get_cases<'tcx>(tcx: &ty::ctxt<'tcx>,
                   adt: ty::AdtDef<'tcx>,
                   substs: &subst::Substs<'tcx>)
                   -> Vec<Case<'tcx>> {
    adt.variants.iter().map(|vi| {
        let field_tys = vi.fields.iter().map(|field| {
            monomorphize::field_ty(tcx, substs, field)
        }).collect();
        Case { discr: vi.disr_val, tys: field_tys }
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
        attr::SignedInt(_) => CEnum(it, bounds.slo as Disr, bounds.shi as Disr),
        attr::UnsignedInt(_) => CEnum(it, bounds.ulo, bounds.uhi)
    }
}

fn range_to_inttype(cx: &CrateContext, hint: Hint, bounds: &IntBounds) -> IntType {
    debug!("range_to_inttype: {:?} {:?}", hint, bounds);
    // Lists of sizes to try.  u64 is always allowed as a fallback.
    #[allow(non_upper_case_globals)]
    const choose_shortest: &'static [IntType] = &[
        attr::UnsignedInt(ast::TyU8), attr::SignedInt(ast::TyI8),
        attr::UnsignedInt(ast::TyU16), attr::SignedInt(ast::TyI16),
        attr::UnsignedInt(ast::TyU32), attr::SignedInt(ast::TyI32)];
    #[allow(non_upper_case_globals)]
    const at_least_32: &'static [IntType] = &[
        attr::UnsignedInt(ast::TyU32), attr::SignedInt(ast::TyI32)];

    let attempts;
    match hint {
        attr::ReprInt(span, ity) => {
            if !bounds_usable(cx, ity, bounds) {
                cx.sess().span_bug(span, "representation hint insufficient for discriminant range")
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
            cx.tcx().sess.bug("range_to_inttype: found ReprPacked on an enum");
        }
        attr::ReprSimd => {
            cx.tcx().sess.bug("range_to_inttype: found ReprSimd on an enum");
        }
    }
    for &ity in attempts {
        if bounds_usable(cx, ity, bounds) {
            return ity;
        }
    }
    return attr::UnsignedInt(ast::TyU64);
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

pub fn ty_of_inttype<'tcx>(tcx: &ty::ctxt<'tcx>, ity: IntType) -> Ty<'tcx> {
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
    let c = generic_type_of(cx, r, None, false, false, false);
    assert!(!c.needs_drop_flag);
    c.prefix
}


// Pass dst=true if the type you are passing is a DST. Yes, we could figure
// this out, but if you call this on an unsized type without realising it, you
// are going to get the wrong type (it will not include the unsized parts of it).
pub fn sizing_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                r: &Repr<'tcx>, dst: bool) -> Type {
    let c = generic_type_of(cx, r, None, true, dst, false);
    assert!(!c.needs_drop_flag);
    c.prefix
}
pub fn sizing_type_context_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                        r: &Repr<'tcx>, dst: bool) -> TypeContext {
    generic_type_of(cx, r, None, true, dst, true)
}
pub fn incomplete_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                    r: &Repr<'tcx>, name: &str) -> Type {
    let c = generic_type_of(cx, r, Some(name), false, false, false);
    assert!(!c.needs_drop_flag);
    c.prefix
}
pub fn finish_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                r: &Repr<'tcx>, llty: &mut Type) {
    match *r {
        CEnum(..) | General(..) | RawNullablePointer { .. } => { }
        Univariant(ref st, _) | StructWrappedNullablePointer { nonnull: ref st, .. } =>
            llty.set_struct_body(&struct_llfields(cx, st, false, false),
                                 st.packed)
    }
}

fn generic_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                             r: &Repr<'tcx>,
                             name: Option<&str>,
                             sizing: bool,
                             dst: bool,
                             delay_drop_flag: bool) -> TypeContext {
    debug!("adt::generic_type_of r: {:?} name: {:?} sizing: {} dst: {} delay_drop_flag: {}",
           r, name, sizing, dst, delay_drop_flag);
    match *r {
        CEnum(ity, _, _) => TypeContext::direct(ll_inttype(cx, ity)),
        RawNullablePointer { nnty, .. } =>
            TypeContext::direct(type_of::sizing_type_of(cx, nnty)),
        StructWrappedNullablePointer { nonnull: ref st, .. } => {
            match name {
                None => {
                    TypeContext::direct(
                        Type::struct_(cx, &struct_llfields(cx, st, sizing, dst),
                                      st.packed))
                }
                Some(name) => {
                    assert_eq!(sizing, false);
                    TypeContext::direct(Type::named_struct(cx, name))
                }
            }
        }
        Univariant(ref st, dtor_needed) => {
            let dtor_needed = dtor_needed != 0;
            match name {
                None => {
                    let mut fields = struct_llfields(cx, st, sizing, dst);
                    if delay_drop_flag && dtor_needed {
                        fields.pop();
                    }
                    TypeContext::may_need_drop_flag(
                        Type::struct_(cx, &fields,
                                      st.packed),
                        delay_drop_flag && dtor_needed)
                }
                Some(name) => {
                    // Hypothesis: named_struct's can never need a
                    // drop flag. (... needs validation.)
                    assert_eq!(sizing, false);
                    TypeContext::direct(Type::named_struct(cx, name))
                }
            }
        }
        General(ity, ref sts, dtor_needed) => {
            let dtor_needed = dtor_needed != 0;
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
            assert_eq!(size % align_s, 0);
            let align_units = size / align_s - 1;

            let discr_ty = ll_inttype(cx, ity);
            let discr_size = machine::llsize_of_alloc(cx, discr_ty);
            let fill_ty = match align_s {
                1 => Type::array(&Type::i8(cx), align_units),
                2 => Type::array(&Type::i16(cx), align_units),
                4 => Type::array(&Type::i32(cx), align_units),
                8 if machine::llalign_of_min(cx, Type::i64(cx)) == 8 =>
                                 Type::array(&Type::i64(cx), align_units),
                a if a.count_ones() == 1 => Type::array(&Type::vector(&Type::i32(cx), a / 4),
                                                              align_units),
                _ => panic!("unsupported enum alignment: {}", align)
            };
            assert_eq!(machine::llalign_of_min(cx, fill_ty), align);
            assert_eq!(align_s % discr_size, 0);
            let mut fields: Vec<Type> =
                [discr_ty,
                 Type::array(&discr_ty, align_s / discr_size - 1),
                 fill_ty].iter().cloned().collect();
            if delay_drop_flag && dtor_needed {
                fields.pop();
            }
            match name {
                None => {
                    TypeContext::may_need_drop_flag(
                        Type::struct_(cx, &fields[..], false),
                        delay_drop_flag && dtor_needed)
                }
                Some(name) => {
                    let mut llty = Type::named_struct(cx, name);
                    llty.set_struct_body(&fields[..], false);
                    TypeContext::may_need_drop_flag(
                        llty,
                        delay_drop_flag && dtor_needed)
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
///
/// This should ideally be less tightly tied to `_match`.
pub fn trans_switch<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                r: &Repr<'tcx>, scrutinee: ValueRef)
                                -> (_match::BranchKind, Option<ValueRef>) {
    match *r {
        CEnum(..) | General(..) |
        RawNullablePointer { .. } | StructWrappedNullablePointer { .. } => {
            (_match::Switch, Some(trans_get_discr(bcx, r, scrutinee, None)))
        }
        Univariant(..) => {
            // N.B.: Univariant means <= 1 enum variants (*not* == 1 variants).
            (_match::Single, None)
        }
    }
}

pub fn is_discr_signed<'tcx>(r: &Repr<'tcx>) -> bool {
    match *r {
        CEnum(ity, _, _) => ity.is_signed(),
        General(ity, _, _) => ity.is_signed(),
        Univariant(..) => false,
        RawNullablePointer { .. } => false,
        StructWrappedNullablePointer { .. } => false,
    }
}

/// Obtain the actual discriminant of a value.
pub fn trans_get_discr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, r: &Repr<'tcx>,
                                   scrutinee: ValueRef, cast_to: Option<Type>)
    -> ValueRef {
    debug!("trans_get_discr r: {:?}", r);
    let val = match *r {
        CEnum(ity, min, max) => load_discr(bcx, ity, scrutinee, min, max),
        General(ity, ref cases, _) => {
            let ptr = StructGEP(bcx, scrutinee, 0);
            load_discr(bcx, ity, ptr, 0, (cases.len() - 1) as Disr)
        }
        Univariant(..) => C_u8(bcx.ccx(), 0),
        RawNullablePointer { nndiscr, nnty, .. } =>  {
            let cmp = if nndiscr == 0 { IntEQ } else { IntNE };
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
    let cmp = if nndiscr == 0 { IntEQ } else { IntNE };
    ICmp(bcx, cmp, llptr, C_null(val_ty(llptr)), DebugLoc::None)
}

/// Helper for cases where the discriminant is simply loaded.
fn load_discr(bcx: Block, ity: IntType, ptr: ValueRef, min: Disr, max: Disr)
    -> ValueRef {
    let llty = ll_inttype(bcx.ccx(), ity);
    assert_eq!(val_ty(ptr), llty.ptr_to());
    let bits = machine::llbitsize_of_real(bcx.ccx(), llty);
    assert!(bits <= 64);
    let  bits = bits as usize;
    let mask = (!0u64 >> (64 - bits)) as Disr;
    // For a (max) discr of -1, max will be `-1 as usize`, which overflows.
    // However, that is fine here (it would still represent the full range),
    if (max.wrapping_add(1)) & mask == min & mask {
        // i.e., if the range is everything.  The lo==hi case would be
        // rejected by the LLVM verifier (it would mean either an
        // empty set, which is impossible, or the entire range of the
        // type, which is pointless).
        Load(bcx, ptr)
    } else {
        // llvm::ConstantRange can deal with ranges that wrap around,
        // so an overflow on (max + 1) is fine.
        LoadRangeAssert(bcx, ptr, min, (max.wrapping_add(1)), /* signed: */ True)
    }
}

/// Yield information about how to dispatch a case of the
/// discriminant-like value returned by `trans_switch`.
///
/// This should ideally be less tightly tied to `_match`.
pub fn trans_case<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, r: &Repr, discr: Disr)
                              -> _match::OptResult<'blk, 'tcx> {
    match *r {
        CEnum(ity, _, _) => {
            _match::SingleResult(Result::new(bcx, C_integral(ll_inttype(bcx.ccx(), ity),
                                                              discr as u64, true)))
        }
        General(ity, _, _) => {
            _match::SingleResult(Result::new(bcx, C_integral(ll_inttype(bcx.ccx(), ity),
                                                              discr as u64, true)))
        }
        Univariant(..) => {
            bcx.ccx().sess().bug("no cases for univariants or structs")
        }
        RawNullablePointer { .. } |
        StructWrappedNullablePointer { .. } => {
            assert!(discr == 0 || discr == 1);
            _match::SingleResult(Result::new(bcx, C_bool(bcx.ccx(), discr != 0)))
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
            Store(bcx, C_integral(ll_inttype(bcx.ccx(), ity), discr as u64, true),
                  val);
        }
        General(ity, ref cases, dtor) => {
            if dtor_active(dtor) {
                let ptr = trans_field_ptr(bcx, r, val, discr,
                                          cases[discr as usize].fields.len() - 2);
                Store(bcx, C_u8(bcx.ccx(), DTOR_NEEDED), ptr);
            }
            Store(bcx, C_integral(ll_inttype(bcx.ccx(), ity), discr as u64, true),
                  StructGEP(bcx, val, 0));
        }
        Univariant(ref st, dtor) => {
            assert_eq!(discr, 0);
            if dtor_active(dtor) {
                Store(bcx, C_u8(bcx.ccx(), DTOR_NEEDED),
                      StructGEP(bcx, val, st.fields.len() - 1));
            }
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
        attr::UnsignedInt(_) => assert!(min <= discr && discr <= max),
        attr::SignedInt(_) => assert!(min as i64 <= discr as i64 && discr as i64 <= max as i64)
    }
}

/// The number of fields in a given case; for use when obtaining this
/// information from the type or definition is less convenient.
pub fn num_args(r: &Repr, discr: Disr) -> usize {
    match *r {
        CEnum(..) => 0,
        Univariant(ref st, dtor) => {
            assert_eq!(discr, 0);
            st.fields.len() - (if dtor_active(dtor) { 1 } else { 0 })
        }
        General(_, ref cases, dtor) => {
            cases[discr as usize].fields.len() - 1 - (if dtor_active(dtor) { 1 } else { 0 })
        }
        RawNullablePointer { nndiscr, ref nullfields, .. } => {
            if discr == nndiscr { 1 } else { nullfields.len() }
        }
        StructWrappedNullablePointer { ref nonnull, nndiscr,
                                       ref nullfields, .. } => {
            if discr == nndiscr { nonnull.fields.len() } else { nullfields.len() }
        }
    }
}

/// Access a field, at a point when the value's case is known.
pub fn trans_field_ptr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, r: &Repr<'tcx>,
                                   val: ValueRef, discr: Disr, ix: usize) -> ValueRef {
    // Note: if this ever needs to generate conditionals (e.g., if we
    // decide to do some kind of cdr-coding-like non-unique repr
    // someday), it will need to return a possibly-new bcx as well.
    match *r {
        CEnum(..) => {
            bcx.ccx().sess().bug("element access in C-like enum")
        }
        Univariant(ref st, _dtor) => {
            assert_eq!(discr, 0);
            struct_field_ptr(bcx, st, val, ix, false)
        }
        General(_, ref cases, _) => {
            struct_field_ptr(bcx, &cases[discr as usize], val, ix + 1, true)
        }
        RawNullablePointer { nndiscr, ref nullfields, .. } |
        StructWrappedNullablePointer { nndiscr, ref nullfields, .. } if discr != nndiscr => {
            // The unit-like case might have a nonzero number of unit-like fields.
            // (e.d., Result of Either with (), as one side.)
            let ty = type_of::type_of(bcx.ccx(), nullfields[ix]);
            assert_eq!(machine::llsize_of_alloc(bcx.ccx(), ty), 0);
            // The contents of memory at this pointer can't matter, but use
            // the value that's "reasonable" in case of pointer comparison.
            PointerCast(bcx, val, ty.ptr_to())
        }
        RawNullablePointer { nndiscr, nnty, .. } => {
            assert_eq!(ix, 0);
            assert_eq!(discr, nndiscr);
            let ty = type_of::type_of(bcx.ccx(), nnty);
            PointerCast(bcx, val, ty.ptr_to())
        }
        StructWrappedNullablePointer { ref nonnull, nndiscr, .. } => {
            assert_eq!(discr, nndiscr);
            struct_field_ptr(bcx, nonnull, val, ix, false)
        }
    }
}

pub fn struct_field_ptr<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, st: &Struct<'tcx>, val: ValueRef,
                                    ix: usize, needs_cast: bool) -> ValueRef {
    let val = if needs_cast {
        let ccx = bcx.ccx();
        let fields = st.fields.iter().map(|&ty| type_of::type_of(ccx, ty)).collect::<Vec<_>>();
        let real_ty = Type::struct_(ccx, &fields[..], st.packed);
        PointerCast(bcx, val, real_ty.ptr_to())
    } else {
        val
    };

    StructGEP(bcx, val, ix)
}

pub fn fold_variants<'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                                    r: &Repr<'tcx>,
                                    value: ValueRef,
                                    mut f: F)
                                    -> Block<'blk, 'tcx> where
    F: FnMut(Block<'blk, 'tcx>, &Struct<'tcx>, ValueRef) -> Block<'blk, 'tcx>,
{
    let fcx = bcx.fcx;
    match *r {
        Univariant(ref st, _) => {
            f(bcx, st, value)
        }
        General(ity, ref cases, _) => {
            let ccx = bcx.ccx();

            // See the comments in trans/base.rs for more information (inside
            // iter_structural_ty), but the gist here is that if the enum's
            // discriminant is *not* in the range that we're expecting (in which
            // case we'll take the fall-through branch on the switch
            // instruction) then we can't just optimize this to an Unreachable
            // block.
            //
            // Currently we still have filling drop, so this means that the drop
            // glue for enums may be called when the enum has been paved over
            // with the "I've been dropped" value. In this case the default
            // branch of the switch instruction will actually be taken at
            // runtime, so the basic block isn't actually unreachable, so we
            // need to make it do something with defined behavior. In this case
            // we just return early from the function.
            let ret_void_cx = fcx.new_temp_block("enum-variant-iter-ret-void");
            RetVoid(ret_void_cx, DebugLoc::None);

            let discr_val = trans_get_discr(bcx, r, value, None);
            let llswitch = Switch(bcx, discr_val, ret_void_cx.llbb, cases.len());
            let bcx_next = fcx.new_temp_block("enum-variant-iter-next");

            for (discr, case) in cases.iter().enumerate() {
                let mut variant_cx = fcx.new_temp_block(
                    &format!("enum-variant-iter-{}", &discr.to_string())
                );
                let rhs_val = C_integral(ll_inttype(ccx, ity), discr as u64, true);
                AddCase(llswitch, rhs_val, variant_cx.llbb);

                let fields = case.fields.iter().map(|&ty|
                    type_of::type_of(bcx.ccx(), ty)).collect::<Vec<_>>();
                let real_ty = Type::struct_(ccx, &fields[..], case.packed);
                let variant_value = PointerCast(variant_cx, value, real_ty.ptr_to());

                variant_cx = f(variant_cx, case, variant_value);
                Br(variant_cx, bcx_next.llbb, DebugLoc::None);
            }

            bcx_next
        }
        _ => unreachable!()
    }
}

/// Access the struct drop flag, if present.
pub fn trans_drop_flag_ptr<'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                       r: &Repr<'tcx>,
                                       val: ValueRef)
                                       -> datum::DatumBlock<'blk, 'tcx, datum::Expr>
{
    let tcx = bcx.tcx();
    let ptr_ty = bcx.tcx().mk_imm_ptr(tcx.dtor_type());
    match *r {
        Univariant(ref st, dtor) if dtor_active(dtor) => {
            let flag_ptr = StructGEP(bcx, val, st.fields.len() - 1);
            datum::immediate_rvalue_bcx(bcx, flag_ptr, ptr_ty).to_expr_datumblock()
        }
        General(_, _, dtor) if dtor_active(dtor) => {
            let fcx = bcx.fcx;
            let custom_cleanup_scope = fcx.push_custom_cleanup_scope();
            let scratch = unpack_datum!(bcx, datum::lvalue_scratch_datum(
                bcx, tcx.dtor_type(), "drop_flag",
                cleanup::CustomScope(custom_cleanup_scope), (), |_, bcx, _| bcx
            ));
            bcx = fold_variants(bcx, r, val, |variant_cx, st, value| {
                let ptr = struct_field_ptr(variant_cx, st, value, (st.fields.len() - 1), false);
                datum::Datum::new(ptr, ptr_ty, datum::Lvalue::new("adt::trans_drop_flag_ptr"))
                    .store_to(variant_cx, scratch.val)
            });
            let expr_datum = scratch.to_expr_datum();
            fcx.pop_custom_cleanup_scope(custom_cleanup_scope);
            datum::DatumBlock::new(bcx, expr_datum)
        }
        _ => bcx.ccx().sess().bug("tried to get drop flag of non-droppable type")
    }
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
            C_integral(ll_inttype(ccx, ity), discr as u64, true)
        }
        General(ity, ref cases, _) => {
            let case = &cases[discr as usize];
            let (max_sz, _) = union_size_and_align(&cases[..]);
            let lldiscr = C_integral(ll_inttype(ccx, ity), discr as u64, true);
            let mut f = vec![lldiscr];
            f.push_all(vals);
            let mut contents = build_const_struct(ccx, case, &f[..]);
            contents.push_all(&[padding(ccx, max_sz - case.size)]);
            C_struct(ccx, &contents[..], false)
        }
        Univariant(ref st, _dro) => {
            assert!(discr == 0);
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

fn padding(ccx: &CrateContext, size: u64) -> ValueRef {
    C_undef(Type::array(&Type::i8(ccx), size))
}

// FIXME this utility routine should be somewhere more general
#[inline]
fn roundup(x: u64, a: u32) -> u64 { let a = a as u64; ((x + (a - 1)) / a) * a }

/// Get the discriminant of a constant value.
pub fn const_get_discrim(ccx: &CrateContext, r: &Repr, val: ValueRef) -> Disr {
    match *r {
        CEnum(ity, _, _) => {
            match ity {
                attr::SignedInt(..) => const_to_int(val) as Disr,
                attr::UnsignedInt(..) => const_to_uint(val) as Disr
            }
        }
        General(ity, _, _) => {
            match ity {
                attr::SignedInt(..) => const_to_int(const_get_elt(ccx, val, &[0])) as Disr,
                attr::UnsignedInt(..) => const_to_uint(const_get_elt(ccx, val, &[0])) as Disr
            }
        }
        Univariant(..) => 0,
        RawNullablePointer { .. } | StructWrappedNullablePointer { .. } => {
            ccx.sess().bug("const discrim access of non c-like enum")
        }
    }
}

/// Extract a field of a constant value, as appropriate for its
/// representation.
///
/// (Not to be confused with `common::const_get_elt`, which operates on
/// raw LLVM-level structs and arrays.)
pub fn const_get_field(ccx: &CrateContext, r: &Repr, val: ValueRef,
                       _discr: Disr, ix: usize) -> ValueRef {
    match *r {
        CEnum(..) => ccx.sess().bug("element access in C-like enum const"),
        Univariant(..) => const_struct_field(ccx, val, ix),
        General(..) => const_struct_field(ccx, val, ix + 1),
        RawNullablePointer { .. } => {
            assert_eq!(ix, 0);
            val
        },
        StructWrappedNullablePointer{ .. } => const_struct_field(ccx, val, ix)
    }
}

/// Extract field of struct-like const, skipping our alignment padding.
fn const_struct_field(ccx: &CrateContext, val: ValueRef, ix: usize) -> ValueRef {
    // Get the ix-th non-undef element of the struct.
    let mut real_ix = 0; // actual position in the struct
    let mut ix = ix; // logical index relative to real_ix
    let mut field;
    loop {
        loop {
            field = const_get_elt(ccx, val, &[real_ix]);
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
