// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! misc. type-system utilities too small to deserve their own file

use hir::svh::Svh;
use hir::def_id::DefId;
use ty::subst;
use infer::InferCtxt;
use hir::pat_util;
use traits::{self, ProjectionMode};
use ty::{self, Ty, TyCtxt, TypeAndMut, TypeFlags, TypeFoldable};
use ty::{Disr, ParameterEnvironment};
use ty::layout::{Layout, LayoutError};
use ty::TypeVariants::*;

use rustc_const_math::{ConstInt, ConstIsize, ConstUsize};

use std::cmp;
use std::hash::{Hash, SipHasher, Hasher};
use syntax::ast::{self, Name};
use syntax::attr::{self, SignedInt, UnsignedInt};
use syntax_pos::Span;

use hir;

pub trait IntTypeExt {
    fn to_ty<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx>;
    fn disr_incr<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, val: Option<Disr>)
                           -> Option<Disr>;
    fn assert_ty_matches(&self, val: Disr);
    fn initial_discriminant<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Disr;
}

impl IntTypeExt for attr::IntType {
    fn to_ty<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx> {
        match *self {
            SignedInt(ast::IntTy::I8)      => tcx.types.i8,
            SignedInt(ast::IntTy::I16)     => tcx.types.i16,
            SignedInt(ast::IntTy::I32)     => tcx.types.i32,
            SignedInt(ast::IntTy::I64)     => tcx.types.i64,
            SignedInt(ast::IntTy::Is)   => tcx.types.isize,
            UnsignedInt(ast::UintTy::U8)    => tcx.types.u8,
            UnsignedInt(ast::UintTy::U16)   => tcx.types.u16,
            UnsignedInt(ast::UintTy::U32)   => tcx.types.u32,
            UnsignedInt(ast::UintTy::U64)   => tcx.types.u64,
            UnsignedInt(ast::UintTy::Us) => tcx.types.usize,
        }
    }

    fn initial_discriminant<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Disr {
        match *self {
            SignedInt(ast::IntTy::I8)    => ConstInt::I8(0),
            SignedInt(ast::IntTy::I16)   => ConstInt::I16(0),
            SignedInt(ast::IntTy::I32)   => ConstInt::I32(0),
            SignedInt(ast::IntTy::I64)   => ConstInt::I64(0),
            SignedInt(ast::IntTy::Is) => match tcx.sess.target.int_type {
                ast::IntTy::I16 => ConstInt::Isize(ConstIsize::Is16(0)),
                ast::IntTy::I32 => ConstInt::Isize(ConstIsize::Is32(0)),
                ast::IntTy::I64 => ConstInt::Isize(ConstIsize::Is64(0)),
                _ => bug!(),
            },
            UnsignedInt(ast::UintTy::U8)  => ConstInt::U8(0),
            UnsignedInt(ast::UintTy::U16) => ConstInt::U16(0),
            UnsignedInt(ast::UintTy::U32) => ConstInt::U32(0),
            UnsignedInt(ast::UintTy::U64) => ConstInt::U64(0),
            UnsignedInt(ast::UintTy::Us) => match tcx.sess.target.uint_type {
                ast::UintTy::U16 => ConstInt::Usize(ConstUsize::Us16(0)),
                ast::UintTy::U32 => ConstInt::Usize(ConstUsize::Us32(0)),
                ast::UintTy::U64 => ConstInt::Usize(ConstUsize::Us64(0)),
                _ => bug!(),
            },
        }
    }

    fn assert_ty_matches(&self, val: Disr) {
        match (*self, val) {
            (SignedInt(ast::IntTy::I8), ConstInt::I8(_)) => {},
            (SignedInt(ast::IntTy::I16), ConstInt::I16(_)) => {},
            (SignedInt(ast::IntTy::I32), ConstInt::I32(_)) => {},
            (SignedInt(ast::IntTy::I64), ConstInt::I64(_)) => {},
            (SignedInt(ast::IntTy::Is), ConstInt::Isize(_)) => {},
            (UnsignedInt(ast::UintTy::U8), ConstInt::U8(_)) => {},
            (UnsignedInt(ast::UintTy::U16), ConstInt::U16(_)) => {},
            (UnsignedInt(ast::UintTy::U32), ConstInt::U32(_)) => {},
            (UnsignedInt(ast::UintTy::U64), ConstInt::U64(_)) => {},
            (UnsignedInt(ast::UintTy::Us), ConstInt::Usize(_)) => {},
            _ => bug!("disr type mismatch: {:?} vs {:?}", self, val),
        }
    }

    fn disr_incr<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, val: Option<Disr>)
                           -> Option<Disr> {
        if let Some(val) = val {
            self.assert_ty_matches(val);
            (val + ConstInt::Infer(1)).ok()
        } else {
            Some(self.initial_discriminant(tcx))
        }
    }
}


#[derive(Copy, Clone)]
pub enum CopyImplementationError {
    InfrigingField(Name),
    InfrigingVariant(Name),
    NotAnAdt,
    HasDestructor
}

/// Describes whether a type is representable. For types that are not
/// representable, 'SelfRecursive' and 'ContainsRecursive' are used to
/// distinguish between types that are recursive with themselves and types that
/// contain a different recursive type. These cases can therefore be treated
/// differently when reporting errors.
///
/// The ordering of the cases is significant. They are sorted so that cmp::max
/// will keep the "more erroneous" of two values.
#[derive(Copy, Clone, PartialOrd, Ord, Eq, PartialEq, Debug)]
pub enum Representability {
    Representable,
    ContainsRecursive,
    SelfRecursive,
}

impl<'tcx> ParameterEnvironment<'tcx> {
    pub fn can_type_implement_copy<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       self_type: Ty<'tcx>, span: Span)
                                       -> Result<(),CopyImplementationError> {
        // FIXME: (@jroesch) float this code up
        tcx.infer_ctxt(None, Some(self.clone()),
                       ProjectionMode::Topmost).enter(|infcx| {
            let adt = match self_type.sty {
                ty::TyStruct(struct_def, substs) => {
                    for field in struct_def.all_fields() {
                        let field_ty = field.ty(tcx, substs);
                        if infcx.type_moves_by_default(field_ty, span) {
                            return Err(CopyImplementationError::InfrigingField(
                                field.name))
                        }
                    }
                    struct_def
                }
                ty::TyEnum(enum_def, substs) => {
                    for variant in &enum_def.variants {
                        for field in &variant.fields {
                            let field_ty = field.ty(tcx, substs);
                            if infcx.type_moves_by_default(field_ty, span) {
                                return Err(CopyImplementationError::InfrigingVariant(
                                    variant.name))
                            }
                        }
                    }
                    enum_def
                }
                _ => return Err(CopyImplementationError::NotAnAdt)
            };

            if adt.has_dtor() {
                return Err(CopyImplementationError::HasDestructor);
            }

            Ok(())
        })
    }
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn pat_contains_ref_binding(self, pat: &hir::Pat) -> Option<hir::Mutability> {
        pat_util::pat_contains_ref_binding(pat)
    }

    pub fn arm_contains_ref_binding(self, arm: &hir::Arm) -> Option<hir::Mutability> {
        pat_util::arm_contains_ref_binding(arm)
    }

    /// Returns the type of element at index `i` in tuple or tuple-like type `t`.
    /// For an enum `t`, `variant` is None only if `t` is a univariant enum.
    pub fn positional_element_ty(self,
                                 ty: Ty<'tcx>,
                                 i: usize,
                                 variant: Option<DefId>) -> Option<Ty<'tcx>> {
        match (&ty.sty, variant) {
            (&TyStruct(def, substs), None) => {
                def.struct_variant().fields.get(i).map(|f| f.ty(self, substs))
            }
            (&TyEnum(def, substs), Some(vid)) => {
                def.variant_with_id(vid).fields.get(i).map(|f| f.ty(self, substs))
            }
            (&TyEnum(def, substs), None) => {
                assert!(def.is_univariant());
                def.variants[0].fields.get(i).map(|f| f.ty(self, substs))
            }
            (&TyTuple(ref v), None) => v.get(i).cloned(),
            _ => None
        }
    }

    /// Returns the type of element at field `n` in struct or struct-like type `t`.
    /// For an enum `t`, `variant` must be some def id.
    pub fn named_element_ty(self,
                            ty: Ty<'tcx>,
                            n: Name,
                            variant: Option<DefId>) -> Option<Ty<'tcx>> {
        match (&ty.sty, variant) {
            (&TyStruct(def, substs), None) => {
                def.struct_variant().find_field_named(n).map(|f| f.ty(self, substs))
            }
            (&TyEnum(def, substs), Some(vid)) => {
                def.variant_with_id(vid).find_field_named(n).map(|f| f.ty(self, substs))
            }
            _ => return None
        }
    }

    /// Returns the IntType representation.
    /// This used to ensure `int_ty` doesn't contain `usize` and `isize`
    /// by converting them to their actual types. That doesn't happen anymore.
    pub fn enum_repr_type(self, opt_hint: Option<&attr::ReprAttr>) -> attr::IntType {
        match opt_hint {
            // Feed in the given type
            Some(&attr::ReprInt(_, int_t)) => int_t,
            // ... but provide sensible default if none provided
            //
            // NB. Historically `fn enum_variants` generate i64 here, while
            // rustc_typeck::check would generate isize.
            _ => SignedInt(ast::IntTy::Is),
        }
    }

    /// Returns the deeply last field of nested structures, or the same type,
    /// if not a structure at all. Corresponds to the only possible unsized
    /// field, and its type can be used to determine unsizing strategy.
    pub fn struct_tail(self, mut ty: Ty<'tcx>) -> Ty<'tcx> {
        while let TyStruct(def, substs) = ty.sty {
            match def.struct_variant().fields.last() {
                Some(f) => ty = f.ty(self, substs),
                None => break
            }
        }
        ty
    }

    /// Same as applying struct_tail on `source` and `target`, but only
    /// keeps going as long as the two types are instances of the same
    /// structure definitions.
    /// For `(Foo<Foo<T>>, Foo<Trait>)`, the result will be `(Foo<T>, Trait)`,
    /// whereas struct_tail produces `T`, and `Trait`, respectively.
    pub fn struct_lockstep_tails(self,
                                 source: Ty<'tcx>,
                                 target: Ty<'tcx>)
                                 -> (Ty<'tcx>, Ty<'tcx>) {
        let (mut a, mut b) = (source, target);
        while let (&TyStruct(a_def, a_substs), &TyStruct(b_def, b_substs)) = (&a.sty, &b.sty) {
            if a_def != b_def {
                break;
            }
            if let Some(f) = a_def.struct_variant().fields.last() {
                a = f.ty(self, a_substs);
                b = f.ty(self, b_substs);
            } else {
                break;
            }
        }
        (a, b)
    }

    /// Given a set of predicates that apply to an object type, returns
    /// the region bounds that the (erased) `Self` type must
    /// outlive. Precisely *because* the `Self` type is erased, the
    /// parameter `erased_self_ty` must be supplied to indicate what type
    /// has been used to represent `Self` in the predicates
    /// themselves. This should really be a unique type; `FreshTy(0)` is a
    /// popular choice.
    ///
    /// NB: in some cases, particularly around higher-ranked bounds,
    /// this function returns a kind of conservative approximation.
    /// That is, all regions returned by this function are definitely
    /// required, but there may be other region bounds that are not
    /// returned, as well as requirements like `for<'a> T: 'a`.
    ///
    /// Requires that trait definitions have been processed so that we can
    /// elaborate predicates and walk supertraits.
    pub fn required_region_bounds(self,
                                  erased_self_ty: Ty<'tcx>,
                                  predicates: Vec<ty::Predicate<'tcx>>)
                                  -> Vec<ty::Region>    {
        debug!("required_region_bounds(erased_self_ty={:?}, predicates={:?})",
               erased_self_ty,
               predicates);

        assert!(!erased_self_ty.has_escaping_regions());

        traits::elaborate_predicates(self, predicates)
            .filter_map(|predicate| {
                match predicate {
                    ty::Predicate::Projection(..) |
                    ty::Predicate::Trait(..) |
                    ty::Predicate::Rfc1592(..) |
                    ty::Predicate::Equate(..) |
                    ty::Predicate::WellFormed(..) |
                    ty::Predicate::ObjectSafe(..) |
                    ty::Predicate::ClosureKind(..) |
                    ty::Predicate::RegionOutlives(..) => {
                        None
                    }
                    ty::Predicate::TypeOutlives(ty::Binder(ty::OutlivesPredicate(t, r))) => {
                        // Search for a bound of the form `erased_self_ty
                        // : 'a`, but be wary of something like `for<'a>
                        // erased_self_ty : 'a` (we interpret a
                        // higher-ranked bound like that as 'static,
                        // though at present the code in `fulfill.rs`
                        // considers such bounds to be unsatisfiable, so
                        // it's kind of a moot point since you could never
                        // construct such an object, but this seems
                        // correct even if that code changes).
                        if t == erased_self_ty && !r.has_escaping_regions() {
                            Some(r)
                        } else {
                            None
                        }
                    }
                }
            })
            .collect()
    }

    /// Creates a hash of the type `Ty` which will be the same no matter what crate
    /// context it's calculated within. This is used by the `type_id` intrinsic.
    pub fn hash_crate_independent(self, ty: Ty<'tcx>, svh: &Svh) -> u64 {
        let mut state = SipHasher::new();
        helper(self, ty, svh, &mut state);
        return state.finish();

        fn helper<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                  ty: Ty<'tcx>, svh: &Svh,
                                  state: &mut SipHasher) {
            macro_rules! byte { ($b:expr) => { ($b as u8).hash(state) } }
            macro_rules! hash { ($e:expr) => { $e.hash(state) }  }

            let region = |state: &mut SipHasher, r: ty::Region| {
                match r {
                    ty::ReStatic | ty::ReErased => {}
                    ty::ReLateBound(db, ty::BrAnon(i)) => {
                        db.hash(state);
                        i.hash(state);
                    }
                    ty::ReEmpty |
                    ty::ReEarlyBound(..) |
                    ty::ReLateBound(..) |
                    ty::ReFree(..) |
                    ty::ReScope(..) |
                    ty::ReVar(..) |
                    ty::ReSkolemized(..) => {
                        bug!("unexpected region found when hashing a type")
                    }
                }
            };
            let did = |state: &mut SipHasher, did: DefId| {
                let h = if did.is_local() {
                    svh.clone()
                } else {
                    tcx.sess.cstore.crate_hash(did.krate)
                };
                h.hash(state);
                did.index.hash(state);
            };
            let mt = |state: &mut SipHasher, mt: TypeAndMut| {
                mt.mutbl.hash(state);
            };
            let fn_sig = |state: &mut SipHasher, sig: &ty::Binder<ty::FnSig<'tcx>>| {
                let sig = tcx.anonymize_late_bound_regions(sig).0;
                for a in &sig.inputs { helper(tcx, *a, svh, state); }
                if let ty::FnConverging(output) = sig.output {
                    helper(tcx, output, svh, state);
                }
            };
            ty.maybe_walk(|ty| {
                match ty.sty {
                    TyBool => byte!(2),
                    TyChar => byte!(3),
                    TyInt(i) => {
                        byte!(4);
                        hash!(i);
                    }
                    TyUint(u) => {
                        byte!(5);
                        hash!(u);
                    }
                    TyFloat(f) => {
                        byte!(6);
                        hash!(f);
                    }
                    TyStr => {
                        byte!(7);
                    }
                    TyEnum(d, _) => {
                        byte!(8);
                        did(state, d.did);
                    }
                    TyBox(_) => {
                        byte!(9);
                    }
                    TyArray(_, n) => {
                        byte!(10);
                        n.hash(state);
                    }
                    TySlice(_) => {
                        byte!(11);
                    }
                    TyRawPtr(m) => {
                        byte!(12);
                        mt(state, m);
                    }
                    TyRef(r, m) => {
                        byte!(13);
                        region(state, *r);
                        mt(state, m);
                    }
                    TyFnDef(def_id, _, _) => {
                        byte!(14);
                        hash!(def_id);
                    }
                    TyFnPtr(ref b) => {
                        byte!(15);
                        hash!(b.unsafety);
                        hash!(b.abi);
                        fn_sig(state, &b.sig);
                        return false;
                    }
                    TyTrait(ref data) => {
                        byte!(17);
                        did(state, data.principal_def_id());
                        hash!(data.bounds);

                        let principal = tcx.anonymize_late_bound_regions(&data.principal).0;
                        for subty in &principal.substs.types {
                            helper(tcx, subty, svh, state);
                        }

                        return false;
                    }
                    TyStruct(d, _) => {
                        byte!(18);
                        did(state, d.did);
                    }
                    TyTuple(ref inner) => {
                        byte!(19);
                        hash!(inner.len());
                    }
                    TyParam(p) => {
                        byte!(20);
                        hash!(p.space);
                        hash!(p.idx);
                        hash!(p.name.as_str());
                    }
                    TyInfer(_) => bug!(),
                    TyError => byte!(21),
                    TyClosure(d, _) => {
                        byte!(22);
                        did(state, d);
                    }
                    TyProjection(ref data) => {
                        byte!(23);
                        did(state, data.trait_ref.def_id);
                        hash!(data.item_name.as_str());
                    }
                }
                true
            });
        }
    }

    /// Returns true if this ADT is a dtorck type.
    ///
    /// Invoking the destructor of a dtorck type during usual cleanup
    /// (e.g. the glue emitted for stack unwinding) requires all
    /// lifetimes in the type-structure of `adt` to strictly outlive
    /// the adt value itself.
    ///
    /// If `adt` is not dtorck, then the adt's destructor can be
    /// invoked even when there are lifetimes in the type-structure of
    /// `adt` that do not strictly outlive the adt value itself.
    /// (This allows programs to make cyclic structures without
    /// resorting to unasfe means; see RFCs 769 and 1238).
    pub fn is_adt_dtorck(self, adt: ty::AdtDef) -> bool {
        let dtor_method = match adt.destructor() {
            Some(dtor) => dtor,
            None => return false
        };

        // RFC 1238: if the destructor method is tagged with the
        // attribute `unsafe_destructor_blind_to_params`, then the
        // compiler is being instructed to *assume* that the
        // destructor will not access borrowed data,
        // even if such data is otherwise reachable.
        //
        // Such access can be in plain sight (e.g. dereferencing
        // `*foo.0` of `Foo<'a>(&'a u32)`) or indirectly hidden
        // (e.g. calling `foo.0.clone()` of `Foo<T:Clone>`).
        return !self.has_attr(dtor_method, "unsafe_destructor_blind_to_params");
    }
}

impl<'a, 'tcx> ty::TyS<'tcx> {
    fn impls_bound(&'tcx self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                   param_env: &ParameterEnvironment<'tcx>,
                   bound: ty::BuiltinBound, span: Span) -> bool
    {
        tcx.infer_ctxt(None, Some(param_env.clone()), ProjectionMode::Topmost).enter(|infcx| {
            traits::type_known_to_meet_builtin_bound(&infcx, self, bound, span)
        })
    }

    // FIXME (@jroesch): I made this public to use it, not sure if should be private
    pub fn moves_by_default(&'tcx self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            param_env: &ParameterEnvironment<'tcx>,
                            span: Span) -> bool {
        if self.flags.get().intersects(TypeFlags::MOVENESS_CACHED) {
            return self.flags.get().intersects(TypeFlags::MOVES_BY_DEFAULT);
        }

        assert!(!self.needs_infer());

        // Fast-path for primitive types
        let result = match self.sty {
            TyBool | TyChar | TyInt(..) | TyUint(..) | TyFloat(..) |
            TyRawPtr(..) | TyFnDef(..) | TyFnPtr(_) | TyRef(_, TypeAndMut {
                mutbl: hir::MutImmutable, ..
            }) => Some(false),

            TyStr | TyBox(..) | TyRef(_, TypeAndMut {
                mutbl: hir::MutMutable, ..
            }) => Some(true),

            TyArray(..) | TySlice(_) | TyTrait(..) | TyTuple(..) |
            TyClosure(..) | TyEnum(..) | TyStruct(..) |
            TyProjection(..) | TyParam(..) | TyInfer(..) | TyError => None
        }.unwrap_or_else(|| !self.impls_bound(tcx, param_env, ty::BoundCopy, span));

        if !self.has_param_types() && !self.has_self_ty() {
            self.flags.set(self.flags.get() | if result {
                TypeFlags::MOVENESS_CACHED | TypeFlags::MOVES_BY_DEFAULT
            } else {
                TypeFlags::MOVENESS_CACHED
            });
        }

        result
    }

    #[inline]
    pub fn is_sized(&'tcx self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    param_env: &ParameterEnvironment<'tcx>,
                    span: Span) -> bool
    {
        if self.flags.get().intersects(TypeFlags::SIZEDNESS_CACHED) {
            return self.flags.get().intersects(TypeFlags::IS_SIZED);
        }

        self.is_sized_uncached(tcx, param_env, span)
    }

    fn is_sized_uncached(&'tcx self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                         param_env: &ParameterEnvironment<'tcx>,
                         span: Span) -> bool {
        assert!(!self.needs_infer());

        // Fast-path for primitive types
        let result = match self.sty {
            TyBool | TyChar | TyInt(..) | TyUint(..) | TyFloat(..) |
            TyBox(..) | TyRawPtr(..) | TyRef(..) | TyFnDef(..) | TyFnPtr(_) |
            TyArray(..) | TyTuple(..) | TyClosure(..) => Some(true),

            TyStr | TyTrait(..) | TySlice(_) => Some(false),

            TyEnum(..) | TyStruct(..) | TyProjection(..) | TyParam(..) |
            TyInfer(..) | TyError => None
        }.unwrap_or_else(|| self.impls_bound(tcx, param_env, ty::BoundSized, span));

        if !self.has_param_types() && !self.has_self_ty() {
            self.flags.set(self.flags.get() | if result {
                TypeFlags::SIZEDNESS_CACHED | TypeFlags::IS_SIZED
            } else {
                TypeFlags::SIZEDNESS_CACHED
            });
        }

        result
    }

    #[inline]
    pub fn layout<'lcx>(&'tcx self, infcx: &InferCtxt<'a, 'tcx, 'lcx>)
                        -> Result<&'tcx Layout, LayoutError<'tcx>> {
        let tcx = infcx.tcx.global_tcx();
        let can_cache = !self.has_param_types() && !self.has_self_ty();
        if can_cache {
            if let Some(&cached) = tcx.layout_cache.borrow().get(&self) {
                return Ok(cached);
            }
        }

        let layout = Layout::compute_uncached(self, infcx)?;
        let layout = tcx.intern_layout(layout);
        if can_cache {
            tcx.layout_cache.borrow_mut().insert(self, layout);
        }
        Ok(layout)
    }


    /// Check whether a type is representable. This means it cannot contain unboxed
    /// structural recursion. This check is needed for structs and enums.
    pub fn is_representable(&'tcx self, tcx: TyCtxt<'a, 'tcx, 'tcx>, sp: Span)
                            -> Representability {

        // Iterate until something non-representable is found
        fn find_nonrepresentable<'a, 'tcx, It>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                               sp: Span,
                                               seen: &mut Vec<Ty<'tcx>>,
                                               iter: It)
                                               -> Representability
        where It: Iterator<Item=Ty<'tcx>> {
            iter.fold(Representability::Representable,
                      |r, ty| cmp::max(r, is_type_structurally_recursive(tcx, sp, seen, ty)))
        }

        fn are_inner_types_recursive<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, sp: Span,
                                               seen: &mut Vec<Ty<'tcx>>, ty: Ty<'tcx>)
                                               -> Representability {
            match ty.sty {
                TyTuple(ref ts) => {
                    find_nonrepresentable(tcx, sp, seen, ts.iter().cloned())
                }
                // Fixed-length vectors.
                // FIXME(#11924) Behavior undecided for zero-length vectors.
                TyArray(ty, _) => {
                    is_type_structurally_recursive(tcx, sp, seen, ty)
                }
                TyStruct(def, substs) | TyEnum(def, substs) => {
                    find_nonrepresentable(tcx,
                                          sp,
                                          seen,
                                          def.all_fields().map(|f| f.ty(tcx, substs)))
                }
                TyClosure(..) => {
                    // this check is run on type definitions, so we don't expect
                    // to see closure types
                    bug!("requires check invoked on inapplicable type: {:?}", ty)
                }
                _ => Representability::Representable,
            }
        }

        fn same_struct_or_enum<'tcx>(ty: Ty<'tcx>, def: ty::AdtDef<'tcx>) -> bool {
            match ty.sty {
                TyStruct(ty_def, _) | TyEnum(ty_def, _) => {
                     ty_def == def
                }
                _ => false
            }
        }

        fn same_type<'tcx>(a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
            match (&a.sty, &b.sty) {
                (&TyStruct(did_a, ref substs_a), &TyStruct(did_b, ref substs_b)) |
                (&TyEnum(did_a, ref substs_a), &TyEnum(did_b, ref substs_b)) => {
                    if did_a != did_b {
                        return false;
                    }

                    let types_a = substs_a.types.get_slice(subst::TypeSpace);
                    let types_b = substs_b.types.get_slice(subst::TypeSpace);

                    let mut pairs = types_a.iter().zip(types_b);

                    pairs.all(|(&a, &b)| same_type(a, b))
                }
                _ => {
                    a == b
                }
            }
        }

        // Does the type `ty` directly (without indirection through a pointer)
        // contain any types on stack `seen`?
        fn is_type_structurally_recursive<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                    sp: Span,
                                                    seen: &mut Vec<Ty<'tcx>>,
                                                    ty: Ty<'tcx>) -> Representability {
            debug!("is_type_structurally_recursive: {:?}", ty);

            match ty.sty {
                TyStruct(def, _) | TyEnum(def, _) => {
                    {
                        // Iterate through stack of previously seen types.
                        let mut iter = seen.iter();

                        // The first item in `seen` is the type we are actually curious about.
                        // We want to return SelfRecursive if this type contains itself.
                        // It is important that we DON'T take generic parameters into account
                        // for this check, so that Bar<T> in this example counts as SelfRecursive:
                        //
                        // struct Foo;
                        // struct Bar<T> { x: Bar<Foo> }

                        match iter.next() {
                            Some(&seen_type) => {
                                if same_struct_or_enum(seen_type, def) {
                                    debug!("SelfRecursive: {:?} contains {:?}",
                                           seen_type,
                                           ty);
                                    return Representability::SelfRecursive;
                                }
                            }
                            None => {}
                        }

                        // We also need to know whether the first item contains other types
                        // that are structurally recursive. If we don't catch this case, we
                        // will recurse infinitely for some inputs.
                        //
                        // It is important that we DO take generic parameters into account
                        // here, so that code like this is considered SelfRecursive, not
                        // ContainsRecursive:
                        //
                        // struct Foo { Option<Option<Foo>> }

                        for &seen_type in iter {
                            if same_type(ty, seen_type) {
                                debug!("ContainsRecursive: {:?} contains {:?}",
                                       seen_type,
                                       ty);
                                return Representability::ContainsRecursive;
                            }
                        }
                    }

                    // For structs and enums, track all previously seen types by pushing them
                    // onto the 'seen' stack.
                    seen.push(ty);
                    let out = are_inner_types_recursive(tcx, sp, seen, ty);
                    seen.pop();
                    out
                }
                _ => {
                    // No need to push in other cases.
                    are_inner_types_recursive(tcx, sp, seen, ty)
                }
            }
        }

        debug!("is_type_representable: {:?}", self);

        // To avoid a stack overflow when checking an enum variant or struct that
        // contains a different, structurally recursive type, maintain a stack
        // of seen types and check recursion for each of them (issues #3008, #3779).
        let mut seen: Vec<Ty> = Vec::new();
        let r = is_type_structurally_recursive(tcx, sp, &mut seen, self);
        debug!("is_type_representable: {:?} is {:?}", self, r);
        r
    }
}
