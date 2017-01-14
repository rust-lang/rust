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

use hir::def_id::DefId;
use hir::map::DefPathData;
use infer::InferCtxt;
use hir::map as ast_map;
use traits::{self, Reveal};
use ty::{self, Ty, TyCtxt, TypeAndMut, TypeFlags, TypeFoldable};
use ty::{Disr, ParameterEnvironment};
use ty::fold::TypeVisitor;
use ty::layout::{Layout, LayoutError};
use ty::TypeVariants::*;
use util::nodemap::FxHashMap;
use middle::lang_items;

use rustc_const_math::{ConstInt, ConstIsize, ConstUsize};
use rustc_data_structures::stable_hasher::{StableHasher, StableHasherResult};

use std::cell::RefCell;
use std::cmp;
use std::hash::Hash;
use std::intrinsics;
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
            SignedInt(ast::IntTy::I128)     => tcx.types.i128,
            SignedInt(ast::IntTy::Is)   => tcx.types.isize,
            UnsignedInt(ast::UintTy::U8)    => tcx.types.u8,
            UnsignedInt(ast::UintTy::U16)   => tcx.types.u16,
            UnsignedInt(ast::UintTy::U32)   => tcx.types.u32,
            UnsignedInt(ast::UintTy::U64)   => tcx.types.u64,
            UnsignedInt(ast::UintTy::U128)   => tcx.types.u128,
            UnsignedInt(ast::UintTy::Us) => tcx.types.usize,
        }
    }

    fn initial_discriminant<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Disr {
        match *self {
            SignedInt(ast::IntTy::I8)    => ConstInt::I8(0),
            SignedInt(ast::IntTy::I16)   => ConstInt::I16(0),
            SignedInt(ast::IntTy::I32)   => ConstInt::I32(0),
            SignedInt(ast::IntTy::I64)   => ConstInt::I64(0),
            SignedInt(ast::IntTy::I128)   => ConstInt::I128(0),
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
            UnsignedInt(ast::UintTy::U128) => ConstInt::U128(0),
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
            (SignedInt(ast::IntTy::I128), ConstInt::I128(_)) => {},
            (SignedInt(ast::IntTy::Is), ConstInt::Isize(_)) => {},
            (UnsignedInt(ast::UintTy::U8), ConstInt::U8(_)) => {},
            (UnsignedInt(ast::UintTy::U16), ConstInt::U16(_)) => {},
            (UnsignedInt(ast::UintTy::U32), ConstInt::U32(_)) => {},
            (UnsignedInt(ast::UintTy::U64), ConstInt::U64(_)) => {},
            (UnsignedInt(ast::UintTy::U128), ConstInt::U128(_)) => {},
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
pub enum CopyImplementationError<'tcx> {
    InfrigingField(&'tcx ty::FieldDef),
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
                                       -> Result<(), CopyImplementationError> {
        // FIXME: (@jroesch) float this code up
        tcx.infer_ctxt(self.clone(), Reveal::NotSpecializable).enter(|infcx| {
            let (adt, substs) = match self_type.sty {
                ty::TyAdt(adt, substs) => (adt, substs),
                _ => return Err(CopyImplementationError::NotAnAdt)
            };

            let field_implements_copy = |field: &ty::FieldDef| {
                let cause = traits::ObligationCause::dummy();
                match traits::fully_normalize(&infcx, cause, &field.ty(tcx, substs)) {
                    Ok(ty) => !infcx.type_moves_by_default(ty, span),
                    Err(..) => false
                }
            };

            for variant in &adt.variants {
                for field in &variant.fields {
                    if !field_implements_copy(field) {
                        return Err(CopyImplementationError::InfrigingField(field));
                    }
                }
            }

            if adt.has_dtor() {
                return Err(CopyImplementationError::HasDestructor);
            }

            Ok(())
        })
    }
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn has_error_field(self, ty: Ty<'tcx>) -> bool {
        match ty.sty {
            ty::TyAdt(def, substs) => {
                for field in def.all_fields() {
                    let field_ty = field.ty(self, substs);
                    if let TyError = field_ty.sty {
                        return true;
                    }
                }
            }
            _ => ()
        }
        false
    }

    /// Returns the type of element at index `i` in tuple or tuple-like type `t`.
    /// For an enum `t`, `variant` is None only if `t` is a univariant enum.
    pub fn positional_element_ty(self,
                                 ty: Ty<'tcx>,
                                 i: usize,
                                 variant: Option<DefId>) -> Option<Ty<'tcx>> {
        match (&ty.sty, variant) {
            (&TyAdt(adt, substs), Some(vid)) => {
                adt.variant_with_id(vid).fields.get(i).map(|f| f.ty(self, substs))
            }
            (&TyAdt(adt, substs), None) => {
                // Don't use `struct_variant`, this may be a univariant enum.
                adt.variants[0].fields.get(i).map(|f| f.ty(self, substs))
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
            (&TyAdt(adt, substs), Some(vid)) => {
                adt.variant_with_id(vid).find_field_named(n).map(|f| f.ty(self, substs))
            }
            (&TyAdt(adt, substs), None) => {
                adt.struct_variant().find_field_named(n).map(|f| f.ty(self, substs))
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
            Some(&attr::ReprInt(int_t)) => int_t,
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
        while let TyAdt(def, substs) = ty.sty {
            if !def.is_struct() {
                break
            }
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
        while let (&TyAdt(a_def, a_substs), &TyAdt(b_def, b_substs)) = (&a.sty, &b.sty) {
            if a_def != b_def || !a_def.is_struct() {
                break
            }
            match a_def.struct_variant().fields.last() {
                Some(f) => {
                    a = f.ty(self, a_substs);
                    b = f.ty(self, b_substs);
                }
                _ => break
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
                                  -> Vec<&'tcx ty::Region>    {
        debug!("required_region_bounds(erased_self_ty={:?}, predicates={:?})",
               erased_self_ty,
               predicates);

        assert!(!erased_self_ty.has_escaping_regions());

        traits::elaborate_predicates(self, predicates)
            .filter_map(|predicate| {
                match predicate {
                    ty::Predicate::Projection(..) |
                    ty::Predicate::Trait(..) |
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
    pub fn type_id_hash(self, ty: Ty<'tcx>) -> u64 {
        let mut hasher = TypeIdHasher::new(self);
        hasher.visit_ty(ty);
        hasher.finish()
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
    pub fn is_adt_dtorck(self, adt: &ty::AdtDef) -> bool {
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

    pub fn closure_base_def_id(&self, def_id: DefId) -> DefId {
        let mut def_id = def_id;
        while self.def_key(def_id).disambiguated_data.data == DefPathData::ClosureExpr {
            def_id = self.parent_def_id(def_id).unwrap_or_else(|| {
                bug!("closure {:?} has no parent", def_id);
            });
        }
        def_id
    }
}

pub struct TypeIdHasher<'a, 'gcx: 'a+'tcx, 'tcx: 'a, W> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    state: StableHasher<W>,
}

impl<'a, 'gcx, 'tcx, W> TypeIdHasher<'a, 'gcx, 'tcx, W>
    where W: StableHasherResult
{
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Self {
        TypeIdHasher { tcx: tcx, state: StableHasher::new() }
    }

    pub fn finish(self) -> W {
        self.state.finish()
    }

    pub fn hash<T: Hash>(&mut self, x: T) {
        x.hash(&mut self.state);
    }

    fn hash_discriminant_u8<T>(&mut self, x: &T) {
        let v = unsafe {
            intrinsics::discriminant_value(x)
        };
        let b = v as u8;
        assert_eq!(v, b as u64);
        self.hash(b)
    }

    fn def_id(&mut self, did: DefId) {
        // Hash the DefPath corresponding to the DefId, which is independent
        // of compiler internal state.
        let path = self.tcx.def_path(did);
        self.def_path(&path)
    }

    pub fn def_path(&mut self, def_path: &ast_map::DefPath) {
        def_path.deterministic_hash_to(self.tcx, &mut self.state);
    }
}

impl<'a, 'gcx, 'tcx, W> TypeVisitor<'tcx> for TypeIdHasher<'a, 'gcx, 'tcx, W>
    where W: StableHasherResult
{
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> bool {
        // Distinguish between the Ty variants uniformly.
        self.hash_discriminant_u8(&ty.sty);

        match ty.sty {
            TyInt(i) => self.hash(i),
            TyUint(u) => self.hash(u),
            TyFloat(f) => self.hash(f),
            TyArray(_, n) => self.hash(n),
            TyRawPtr(m) |
            TyRef(_, m) => self.hash(m.mutbl),
            TyClosure(def_id, _) |
            TyAnon(def_id, _) |
            TyFnDef(def_id, ..) => self.def_id(def_id),
            TyAdt(d, _) => self.def_id(d.did),
            TyFnPtr(f) => {
                self.hash(f.unsafety);
                self.hash(f.abi);
                self.hash(f.sig.variadic());
                self.hash(f.sig.skip_binder().inputs().len());
            }
            TyDynamic(ref data, ..) => {
                if let Some(p) = data.principal() {
                    self.def_id(p.def_id());
                }
                for d in data.auto_traits() {
                    self.def_id(d);
                }
            }
            TyTuple(tys) => {
                self.hash(tys.len());
            }
            TyParam(p) => {
                self.hash(p.idx);
                self.hash(p.name.as_str());
            }
            TyProjection(ref data) => {
                self.def_id(data.trait_ref.def_id);
                self.hash(data.item_name.as_str());
            }
            TyNever |
            TyBool |
            TyChar |
            TyStr |
            TyBox(_) |
            TySlice(_) => {}

            TyError |
            TyInfer(_) => bug!("TypeIdHasher: unexpected type {}", ty)
        }

        ty.super_visit_with(self)
    }

    fn visit_region(&mut self, r: &'tcx ty::Region) -> bool {
        match *r {
            ty::ReErased => {
                self.hash::<u32>(0);
            }
            ty::ReLateBound(db, ty::BrAnon(i)) => {
                assert!(db.depth > 0);
                self.hash::<u32>(db.depth);
                self.hash(i);
            }
            ty::ReStatic |
            ty::ReEmpty |
            ty::ReEarlyBound(..) |
            ty::ReLateBound(..) |
            ty::ReFree(..) |
            ty::ReScope(..) |
            ty::ReVar(..) |
            ty::ReSkolemized(..) => {
                bug!("TypeIdHasher: unexpected region {:?}", r)
            }
        }
        false
    }

    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, x: &ty::Binder<T>) -> bool {
        // Anonymize late-bound regions so that, for example:
        // `for<'a, b> fn(&'a &'b T)` and `for<'a, b> fn(&'b &'a T)`
        // result in the same TypeId (the two types are equivalent).
        self.tcx.anonymize_late_bound_regions(x).super_visit_with(self)
    }
}

impl<'a, 'tcx> ty::TyS<'tcx> {
    fn impls_bound(&'tcx self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                   param_env: &ParameterEnvironment<'tcx>,
                   def_id: DefId,
                   cache: &RefCell<FxHashMap<Ty<'tcx>, bool>>,
                   span: Span) -> bool
    {
        if self.has_param_types() || self.has_self_ty() {
            if let Some(result) = cache.borrow().get(self) {
                return *result;
            }
        }
        let result =
            tcx.infer_ctxt(param_env.clone(), Reveal::ExactMatch)
            .enter(|infcx| {
                traits::type_known_to_meet_bound(&infcx, self, def_id, span)
            });
        if self.has_param_types() || self.has_self_ty() {
            cache.borrow_mut().insert(self, result);
        }
        return result;
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
            TyBool | TyChar | TyInt(..) | TyUint(..) | TyFloat(..) | TyNever |
            TyRawPtr(..) | TyFnDef(..) | TyFnPtr(_) | TyRef(_, TypeAndMut {
                mutbl: hir::MutImmutable, ..
            }) => Some(false),

            TyStr | TyBox(..) | TyRef(_, TypeAndMut {
                mutbl: hir::MutMutable, ..
            }) => Some(true),

            TyArray(..) | TySlice(..) | TyDynamic(..) | TyTuple(..) |
            TyClosure(..) | TyAdt(..) | TyAnon(..) |
            TyProjection(..) | TyParam(..) | TyInfer(..) | TyError => None
        }.unwrap_or_else(|| {
            !self.impls_bound(tcx, param_env,
                              tcx.require_lang_item(lang_items::CopyTraitLangItem),
                              &param_env.is_copy_cache, span) });

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
            TyArray(..) | TyTuple(..) | TyClosure(..) | TyNever => Some(true),

            TyStr | TyDynamic(..) | TySlice(_) => Some(false),

            TyAdt(..) | TyProjection(..) | TyParam(..) |
            TyInfer(..) | TyAnon(..) | TyError => None
        }.unwrap_or_else(|| {
            self.impls_bound(tcx, param_env, tcx.require_lang_item(lang_items::SizedTraitLangItem),
                              &param_env.is_sized_cache, span) });

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

        let rec_limit = tcx.sess.recursion_limit.get();
        let depth = tcx.layout_depth.get();
        if depth > rec_limit {
            tcx.sess.fatal(
                &format!("overflow representing the type `{}`", self));
        }

        tcx.layout_depth.set(depth+1);
        let layout = Layout::compute_uncached(self, infcx)?;
        if can_cache {
            tcx.layout_cache.borrow_mut().insert(self, layout);
        }
        tcx.layout_depth.set(depth);
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
                TyAdt(def, substs) => {
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

        fn same_struct_or_enum<'tcx>(ty: Ty<'tcx>, def: &'tcx ty::AdtDef) -> bool {
            match ty.sty {
                TyAdt(ty_def, _) => {
                     ty_def == def
                }
                _ => false
            }
        }

        fn same_type<'tcx>(a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
            match (&a.sty, &b.sty) {
                (&TyAdt(did_a, substs_a), &TyAdt(did_b, substs_b)) => {
                    if did_a != did_b {
                        return false;
                    }

                    substs_a.types().zip(substs_b.types()).all(|(a, b)| same_type(a, b))
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
                TyAdt(def, _) => {
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

                        if let Some(&seen_type) = iter.next() {
                            if same_struct_or_enum(seen_type, def) {
                                debug!("SelfRecursive: {:?} contains {:?}",
                                       seen_type,
                                       ty);
                                return Representability::SelfRecursive;
                            }
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
