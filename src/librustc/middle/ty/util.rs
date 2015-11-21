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

use back::svh::Svh;
use middle::const_eval::{self, ConstVal, ErrKind};
use middle::const_eval::EvalHint::UncheckedExprHint;
use middle::def_id::DefId;
use middle::subst::{self, Subst, Substs};
use middle::infer;
use middle::pat_util;
use middle::traits;
use middle::ty::{self, Ty, TypeAndMut, TypeFlags};
use middle::ty::{Disr, ParameterEnvironment};
use middle::ty::{HasTypeFlags, RegionEscape};
use middle::ty::TypeVariants::*;
use util::num::ToPrimitive;

use std::cmp;
use std::hash::{Hash, SipHasher, Hasher};
use std::rc::Rc;
use syntax::ast::{self, Name};
use syntax::attr::{self, AttrMetaMethods, SignedInt, UnsignedInt};
use syntax::codemap::Span;

use rustc_front::hir;

pub trait IntTypeExt {
    fn to_ty<'tcx>(&self, cx: &ty::ctxt<'tcx>) -> Ty<'tcx>;
    fn i64_to_disr(&self, val: i64) -> Option<Disr>;
    fn u64_to_disr(&self, val: u64) -> Option<Disr>;
    fn disr_incr(&self, val: Disr) -> Option<Disr>;
    fn disr_string(&self, val: Disr) -> String;
    fn disr_wrap_incr(&self, val: Option<Disr>) -> Disr;
}

impl IntTypeExt for attr::IntType {
    fn to_ty<'tcx>(&self, cx: &ty::ctxt<'tcx>) -> Ty<'tcx> {
        match *self {
            SignedInt(ast::TyI8)      => cx.types.i8,
            SignedInt(ast::TyI16)     => cx.types.i16,
            SignedInt(ast::TyI32)     => cx.types.i32,
            SignedInt(ast::TyI64)     => cx.types.i64,
            SignedInt(ast::TyIs)   => cx.types.isize,
            UnsignedInt(ast::TyU8)    => cx.types.u8,
            UnsignedInt(ast::TyU16)   => cx.types.u16,
            UnsignedInt(ast::TyU32)   => cx.types.u32,
            UnsignedInt(ast::TyU64)   => cx.types.u64,
            UnsignedInt(ast::TyUs) => cx.types.usize,
        }
    }

    fn i64_to_disr(&self, val: i64) -> Option<Disr> {
        match *self {
            SignedInt(ast::TyI8)    => val.to_i8()  .map(|v| v as Disr),
            SignedInt(ast::TyI16)   => val.to_i16() .map(|v| v as Disr),
            SignedInt(ast::TyI32)   => val.to_i32() .map(|v| v as Disr),
            SignedInt(ast::TyI64)   => val.to_i64() .map(|v| v as Disr),
            UnsignedInt(ast::TyU8)  => val.to_u8()  .map(|v| v as Disr),
            UnsignedInt(ast::TyU16) => val.to_u16() .map(|v| v as Disr),
            UnsignedInt(ast::TyU32) => val.to_u32() .map(|v| v as Disr),
            UnsignedInt(ast::TyU64) => val.to_u64() .map(|v| v as Disr),

            UnsignedInt(ast::TyUs) |
            SignedInt(ast::TyIs) => unreachable!(),
        }
    }

    fn u64_to_disr(&self, val: u64) -> Option<Disr> {
        match *self {
            SignedInt(ast::TyI8)    => val.to_i8()  .map(|v| v as Disr),
            SignedInt(ast::TyI16)   => val.to_i16() .map(|v| v as Disr),
            SignedInt(ast::TyI32)   => val.to_i32() .map(|v| v as Disr),
            SignedInt(ast::TyI64)   => val.to_i64() .map(|v| v as Disr),
            UnsignedInt(ast::TyU8)  => val.to_u8()  .map(|v| v as Disr),
            UnsignedInt(ast::TyU16) => val.to_u16() .map(|v| v as Disr),
            UnsignedInt(ast::TyU32) => val.to_u32() .map(|v| v as Disr),
            UnsignedInt(ast::TyU64) => val.to_u64() .map(|v| v as Disr),

            UnsignedInt(ast::TyUs) |
            SignedInt(ast::TyIs) => unreachable!(),
        }
    }

    fn disr_incr(&self, val: Disr) -> Option<Disr> {
        macro_rules! add1 {
            ($e:expr) => { $e.and_then(|v|v.checked_add(1)).map(|v| v as Disr) }
        }
        match *self {
            // SignedInt repr means we *want* to reinterpret the bits
            // treating the highest bit of Disr as a sign-bit, so
            // cast to i64 before range-checking.
            SignedInt(ast::TyI8)    => add1!((val as i64).to_i8()),
            SignedInt(ast::TyI16)   => add1!((val as i64).to_i16()),
            SignedInt(ast::TyI32)   => add1!((val as i64).to_i32()),
            SignedInt(ast::TyI64)   => add1!(Some(val as i64)),

            UnsignedInt(ast::TyU8)  => add1!(val.to_u8()),
            UnsignedInt(ast::TyU16) => add1!(val.to_u16()),
            UnsignedInt(ast::TyU32) => add1!(val.to_u32()),
            UnsignedInt(ast::TyU64) => add1!(Some(val)),

            UnsignedInt(ast::TyUs) |
            SignedInt(ast::TyIs) => unreachable!(),
        }
    }

    // This returns a String because (1.) it is only used for
    // rendering an error message and (2.) a string can represent the
    // full range from `i64::MIN` through `u64::MAX`.
    fn disr_string(&self, val: Disr) -> String {
        match *self {
            SignedInt(ast::TyI8)    => format!("{}", val as i8 ),
            SignedInt(ast::TyI16)   => format!("{}", val as i16),
            SignedInt(ast::TyI32)   => format!("{}", val as i32),
            SignedInt(ast::TyI64)   => format!("{}", val as i64),
            UnsignedInt(ast::TyU8)  => format!("{}", val as u8 ),
            UnsignedInt(ast::TyU16) => format!("{}", val as u16),
            UnsignedInt(ast::TyU32) => format!("{}", val as u32),
            UnsignedInt(ast::TyU64) => format!("{}", val as u64),

            UnsignedInt(ast::TyUs) |
            SignedInt(ast::TyIs) => unreachable!(),
        }
    }

    fn disr_wrap_incr(&self, val: Option<Disr>) -> Disr {
        macro_rules! add1 {
            ($e:expr) => { ($e).wrapping_add(1) as Disr }
        }
        let val = val.unwrap_or(ty::INITIAL_DISCRIMINANT_VALUE);
        match *self {
            SignedInt(ast::TyI8)    => add1!(val as i8 ),
            SignedInt(ast::TyI16)   => add1!(val as i16),
            SignedInt(ast::TyI32)   => add1!(val as i32),
            SignedInt(ast::TyI64)   => add1!(val as i64),
            UnsignedInt(ast::TyU8)  => add1!(val as u8 ),
            UnsignedInt(ast::TyU16) => add1!(val as u16),
            UnsignedInt(ast::TyU32) => add1!(val as u32),
            UnsignedInt(ast::TyU64) => add1!(val as u64),

            UnsignedInt(ast::TyUs) |
            SignedInt(ast::TyIs) => unreachable!(),
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

impl<'a, 'tcx> ParameterEnvironment<'a, 'tcx> {
    pub fn can_type_implement_copy(&self, self_type: Ty<'tcx>, span: Span)
                                   -> Result<(),CopyImplementationError> {
        let tcx = self.tcx;

        // FIXME: (@jroesch) float this code up
        let infcx = infer::new_infer_ctxt(tcx, &tcx.tables, Some(self.clone()), false);

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
            _ => return Err(CopyImplementationError::NotAnAdt),
        };

        if adt.has_dtor() {
            return Err(CopyImplementationError::HasDestructor)
        }

        Ok(())
    }
}

impl<'tcx> ty::ctxt<'tcx> {
    pub fn pat_contains_ref_binding(&self, pat: &hir::Pat) -> Option<hir::Mutability> {
        pat_util::pat_contains_ref_binding(&self.def_map, pat)
    }

    pub fn arm_contains_ref_binding(&self, arm: &hir::Arm) -> Option<hir::Mutability> {
        pat_util::arm_contains_ref_binding(&self.def_map, arm)
    }

    /// Returns the type of element at index `i` in tuple or tuple-like type `t`.
    /// For an enum `t`, `variant` is None only if `t` is a univariant enum.
    pub fn positional_element_ty(&self,
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
    pub fn named_element_ty(&self,
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

    /// Returns `(normalized_type, ty)`, where `normalized_type` is the
    /// IntType representation of one of {i64,i32,i16,i8,u64,u32,u16,u8},
    /// and `ty` is the original type (i.e. may include `isize` or
    /// `usize`).
    pub fn enum_repr_type(&self, opt_hint: Option<&attr::ReprAttr>)
                          -> (attr::IntType, Ty<'tcx>) {
        let repr_type = match opt_hint {
            // Feed in the given type
            Some(&attr::ReprInt(_, int_t)) => int_t,
            // ... but provide sensible default if none provided
            //
            // NB. Historically `fn enum_variants` generate i64 here, while
            // rustc_typeck::check would generate isize.
            _ => SignedInt(ast::TyIs),
        };

        let repr_type_ty = repr_type.to_ty(self);
        let repr_type = match repr_type {
            SignedInt(ast::TyIs) =>
                SignedInt(self.sess.target.int_type),
            UnsignedInt(ast::TyUs) =>
                UnsignedInt(self.sess.target.uint_type),
            other => other
        };

        (repr_type, repr_type_ty)
    }

    /// Returns the deeply last field of nested structures, or the same type,
    /// if not a structure at all. Corresponds to the only possible unsized
    /// field, and its type can be used to determine unsizing strategy.
    pub fn struct_tail(&self, mut ty: Ty<'tcx>) -> Ty<'tcx> {
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
    pub fn struct_lockstep_tails(&self,
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

    /// Returns the repeat count for a repeating vector expression.
    pub fn eval_repeat_count(&self, count_expr: &hir::Expr) -> usize {
        let hint = UncheckedExprHint(self.types.usize);
        match const_eval::eval_const_expr_partial(self, count_expr, hint, None) {
            Ok(val) => {
                let found = match val {
                    ConstVal::Uint(count) => return count as usize,
                    ConstVal::Int(count) if count >= 0 => return count as usize,
                    const_val => const_val.description(),
                };
                span_err!(self.sess, count_expr.span, E0306,
                    "expected positive integer for repeat count, found {}",
                    found);
            }
            Err(err) => {
                let err_msg = match count_expr.node {
                    hir::ExprPath(None, hir::Path {
                        global: false,
                        ref segments,
                        ..
                    }) if segments.len() == 1 =>
                        format!("found variable"),
                    _ => match err.kind {
                        ErrKind::MiscCatchAll => format!("but found {}", err.description()),
                        _ => format!("but {}", err.description())
                    }
                };
                span_err!(self.sess, count_expr.span, E0307,
                    "expected constant integer for repeat count, {}", err_msg);
            }
        }
        0
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
    pub fn required_region_bounds(&self,
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
                    ty::Predicate::Equate(..) |
                    ty::Predicate::WellFormed(..) |
                    ty::Predicate::ObjectSafe(..) |
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
    pub fn hash_crate_independent(&self, ty: Ty<'tcx>, svh: &Svh) -> u64 {
        let mut state = SipHasher::new();
        helper(self, ty, svh, &mut state);
        return state.finish();

        fn helper<'tcx>(tcx: &ty::ctxt<'tcx>, ty: Ty<'tcx>, svh: &Svh,
                        state: &mut SipHasher) {
            macro_rules! byte { ($b:expr) => { ($b as u8).hash(state) } }
            macro_rules! hash { ($e:expr) => { $e.hash(state) }  }

            let region = |state: &mut SipHasher, r: ty::Region| {
                match r {
                    ty::ReStatic => {}
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
                        tcx.sess.bug("unexpected region found when hashing a type")
                    }
                }
            };
            let did = |state: &mut SipHasher, did: DefId| {
                let h = if did.is_local() {
                    svh.clone()
                } else {
                    tcx.sess.cstore.crate_hash(did.krate)
                };
                h.as_str().hash(state);
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
                    TyBareFn(opt_def_id, ref b) => {
                        byte!(14);
                        hash!(opt_def_id);
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
                    TyInfer(_) => unreachable!(),
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
    pub fn is_adt_dtorck(&self, adt: ty::AdtDef<'tcx>) -> bool {
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

#[derive(Debug)]
pub struct ImplMethod<'tcx> {
    pub method: Rc<ty::Method<'tcx>>,
    pub substs: Substs<'tcx>,
    pub is_provided: bool
}

impl<'tcx> ty::ctxt<'tcx> {
    #[inline(never)] // is this perfy enough?
    pub fn get_impl_method(&self,
                           impl_def_id: DefId,
                           substs: Substs<'tcx>,
                           name: Name)
                           -> ImplMethod<'tcx>
    {
        // there don't seem to be nicer accessors to these:
        let impl_or_trait_items_map = self.impl_or_trait_items.borrow();

        for impl_item in &self.impl_items.borrow()[&impl_def_id] {
            if let ty::MethodTraitItem(ref meth) =
                impl_or_trait_items_map[&impl_item.def_id()] {
                if meth.name == name {
                    return ImplMethod {
                        method: meth.clone(),
                        substs: substs,
                        is_provided: false
                    }
                }
            }
        }

        // It is not in the impl - get the default from the trait.
        let trait_ref = self.impl_trait_ref(impl_def_id).unwrap();
        for trait_item in self.trait_items(trait_ref.def_id).iter() {
            if let &ty::MethodTraitItem(ref meth) = trait_item {
                if meth.name == name {
                    let impl_to_trait_substs = self
                        .make_substs_for_receiver_types(&trait_ref, meth);
                    return ImplMethod {
                        method: meth.clone(),
                        substs: impl_to_trait_substs.subst(self, &substs),
                        is_provided: true
                    }
                }
            }
        }

        self.sess.bug(&format!("method {:?} not found in {:?}",
                               name, impl_def_id))
    }
}

impl<'tcx> ty::TyS<'tcx> {
    fn impls_bound<'a>(&'tcx self, param_env: &ParameterEnvironment<'a,'tcx>,
                       bound: ty::BuiltinBound,
                       span: Span)
                       -> bool
    {
        let tcx = param_env.tcx;
        let infcx = infer::new_infer_ctxt(tcx, &tcx.tables, Some(param_env.clone()), false);

        let is_impld = traits::type_known_to_meet_builtin_bound(&infcx,
                                                                self, bound, span);

        debug!("Ty::impls_bound({:?}, {:?}) = {:?}",
               self, bound, is_impld);

        is_impld
    }

    // FIXME (@jroesch): I made this public to use it, not sure if should be private
    pub fn moves_by_default<'a>(&'tcx self, param_env: &ParameterEnvironment<'a,'tcx>,
                           span: Span) -> bool {
        if self.flags.get().intersects(TypeFlags::MOVENESS_CACHED) {
            return self.flags.get().intersects(TypeFlags::MOVES_BY_DEFAULT);
        }

        assert!(!self.needs_infer());

        // Fast-path for primitive types
        let result = match self.sty {
            TyBool | TyChar | TyInt(..) | TyUint(..) | TyFloat(..) |
            TyRawPtr(..) | TyBareFn(..) | TyRef(_, TypeAndMut {
                mutbl: hir::MutImmutable, ..
            }) => Some(false),

            TyStr | TyBox(..) | TyRef(_, TypeAndMut {
                mutbl: hir::MutMutable, ..
            }) => Some(true),

            TyArray(..) | TySlice(_) | TyTrait(..) | TyTuple(..) |
            TyClosure(..) | TyEnum(..) | TyStruct(..) |
            TyProjection(..) | TyParam(..) | TyInfer(..) | TyError => None
        }.unwrap_or_else(|| !self.impls_bound(param_env, ty::BoundCopy, span));

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
    pub fn is_sized<'a>(&'tcx self, param_env: &ParameterEnvironment<'a,'tcx>,
                        span: Span) -> bool
    {
        if self.flags.get().intersects(TypeFlags::SIZEDNESS_CACHED) {
            return self.flags.get().intersects(TypeFlags::IS_SIZED);
        }

        self.is_sized_uncached(param_env, span)
    }

    fn is_sized_uncached<'a>(&'tcx self, param_env: &ParameterEnvironment<'a,'tcx>,
                             span: Span) -> bool {
        assert!(!self.needs_infer());

        // Fast-path for primitive types
        let result = match self.sty {
            TyBool | TyChar | TyInt(..) | TyUint(..) | TyFloat(..) |
            TyBox(..) | TyRawPtr(..) | TyRef(..) | TyBareFn(..) |
            TyArray(..) | TyTuple(..) | TyClosure(..) => Some(true),

            TyStr | TyTrait(..) | TySlice(_) => Some(false),

            TyEnum(..) | TyStruct(..) | TyProjection(..) | TyParam(..) |
            TyInfer(..) | TyError => None
        }.unwrap_or_else(|| self.impls_bound(param_env, ty::BoundSized, span));

        if !self.has_param_types() && !self.has_self_ty() {
            self.flags.set(self.flags.get() | if result {
                TypeFlags::SIZEDNESS_CACHED | TypeFlags::IS_SIZED
            } else {
                TypeFlags::SIZEDNESS_CACHED
            });
        }

        result
    }


    /// Check whether a type is representable. This means it cannot contain unboxed
    /// structural recursion. This check is needed for structs and enums.
    pub fn is_representable(&'tcx self, cx: &ty::ctxt<'tcx>, sp: Span) -> Representability {

        // Iterate until something non-representable is found
        fn find_nonrepresentable<'tcx, It: Iterator<Item=Ty<'tcx>>>(cx: &ty::ctxt<'tcx>,
                                                                    sp: Span,
                                                                    seen: &mut Vec<Ty<'tcx>>,
                                                                    iter: It)
                                                                    -> Representability {
            iter.fold(Representability::Representable,
                      |r, ty| cmp::max(r, is_type_structurally_recursive(cx, sp, seen, ty)))
        }

        fn are_inner_types_recursive<'tcx>(cx: &ty::ctxt<'tcx>, sp: Span,
                                           seen: &mut Vec<Ty<'tcx>>, ty: Ty<'tcx>)
                                           -> Representability {
            match ty.sty {
                TyTuple(ref ts) => {
                    find_nonrepresentable(cx, sp, seen, ts.iter().cloned())
                }
                // Fixed-length vectors.
                // FIXME(#11924) Behavior undecided for zero-length vectors.
                TyArray(ty, _) => {
                    is_type_structurally_recursive(cx, sp, seen, ty)
                }
                TyStruct(def, substs) | TyEnum(def, substs) => {
                    find_nonrepresentable(cx,
                                          sp,
                                          seen,
                                          def.all_fields().map(|f| f.ty(cx, substs)))
                }
                TyClosure(..) => {
                    // this check is run on type definitions, so we don't expect
                    // to see closure types
                    cx.sess.bug(&format!("requires check invoked on inapplicable type: {:?}", ty))
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
        fn is_type_structurally_recursive<'tcx>(cx: &ty::ctxt<'tcx>,
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
                    let out = are_inner_types_recursive(cx, sp, seen, ty);
                    seen.pop();
                    out
                }
                _ => {
                    // No need to push in other cases.
                    are_inner_types_recursive(cx, sp, seen, ty)
                }
            }
        }

        debug!("is_type_representable: {:?}", self);

        // To avoid a stack overflow when checking an enum variant or struct that
        // contains a different, structurally recursive type, maintain a stack
        // of seen types and check recursion for each of them (issues #3008, #3779).
        let mut seen: Vec<Ty> = Vec::new();
        let r = is_type_structurally_recursive(cx, sp, &mut seen, self);
        debug!("is_type_representable: {:?} is {:?}", self, r);
        r
    }
}
