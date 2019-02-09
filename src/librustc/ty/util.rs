//! Miscellaneous type-system utilities that are too small to deserve their own modules.

use crate::hir::def::Def;
use crate::hir::def_id::DefId;
use crate::hir::map::DefPathData;
use crate::hir::{self, Node};
use crate::mir::interpret::{sign_extend, truncate};
use crate::ich::NodeIdHashingMode;
use crate::traits::{self, ObligationCause};
use crate::ty::{self, Ty, TyCtxt, GenericParamDefKind, TypeFoldable};
use crate::ty::subst::{Subst, InternalSubsts, SubstsRef, UnpackedKind};
use crate::ty::query::TyCtxtAt;
use crate::ty::TyKind::*;
use crate::ty::layout::{Integer, IntegerExt};
use crate::util::common::ErrorReported;
use crate::middle::lang_items;
use crate::mir::interpret::GlobalId;

use rustc_data_structures::stable_hasher::{StableHasher, HashStable};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use std::{cmp, fmt};
use syntax::ast;
use syntax::attr::{self, SignedInt, UnsignedInt};
use syntax_pos::{Span, DUMMY_SP};

#[derive(Copy, Clone, Debug)]
pub struct Discr<'tcx> {
    /// Bit representation of the discriminant (e.g., `-128i8` is `0xFF_u128`).
    pub val: u128,
    pub ty: Ty<'tcx>
}

impl<'tcx> fmt::Display for Discr<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ty.sty {
            ty::Int(ity) => {
                let size = ty::tls::with(|tcx| {
                    Integer::from_attr(&tcx, SignedInt(ity)).size()
                });
                let x = self.val;
                // sign extend the raw representation to be an i128
                let x = sign_extend(x, size) as i128;
                write!(fmt, "{}", x)
            },
            _ => write!(fmt, "{}", self.val),
        }
    }
}

impl<'tcx> Discr<'tcx> {
    /// Adds `1` to the value and wraps around if the maximum for the type is reached.
    pub fn wrap_incr<'a, 'gcx>(self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Self {
        self.checked_add(tcx, 1).0
    }
    pub fn checked_add<'a, 'gcx>(self, tcx: TyCtxt<'a, 'gcx, 'tcx>, n: u128) -> (Self, bool) {
        let (int, signed) = match self.ty.sty {
            Int(ity) => (Integer::from_attr(&tcx, SignedInt(ity)), true),
            Uint(uty) => (Integer::from_attr(&tcx, UnsignedInt(uty)), false),
            _ => bug!("non integer discriminant"),
        };

        let size = int.size();
        let bit_size = int.size().bits();
        let shift = 128 - bit_size;
        if signed {
            let sext = |u| {
                sign_extend(u, size) as i128
            };
            let min = sext(1_u128 << (bit_size - 1));
            let max = i128::max_value() >> shift;
            let val = sext(self.val);
            assert!(n < (i128::max_value() as u128));
            let n = n as i128;
            let oflo = val > max - n;
            let val = if oflo {
                min + (n - (max - val) - 1)
            } else {
                val + n
            };
            // zero the upper bits
            let val = val as u128;
            let val = truncate(val, size);
            (Self {
                val: val as u128,
                ty: self.ty,
            }, oflo)
        } else {
            let max = u128::max_value() >> shift;
            let val = self.val;
            let oflo = val > max - n;
            let val = if oflo {
                n - (max - val) - 1
            } else {
                val + n
            };
            (Self {
                val: val,
                ty: self.ty,
            }, oflo)
        }
    }
}

pub trait IntTypeExt {
    fn to_ty<'a, 'gcx, 'tcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx>;
    fn disr_incr<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, val: Option<Discr<'tcx>>)
                           -> Option<Discr<'tcx>>;
    fn initial_discriminant<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Discr<'tcx>;
}

impl IntTypeExt for attr::IntType {
    fn to_ty<'a, 'gcx, 'tcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        match *self {
            SignedInt(ast::IntTy::I8)       => tcx.types.i8,
            SignedInt(ast::IntTy::I16)      => tcx.types.i16,
            SignedInt(ast::IntTy::I32)      => tcx.types.i32,
            SignedInt(ast::IntTy::I64)      => tcx.types.i64,
            SignedInt(ast::IntTy::I128)     => tcx.types.i128,
            SignedInt(ast::IntTy::Isize)    => tcx.types.isize,
            UnsignedInt(ast::UintTy::U8)    => tcx.types.u8,
            UnsignedInt(ast::UintTy::U16)   => tcx.types.u16,
            UnsignedInt(ast::UintTy::U32)   => tcx.types.u32,
            UnsignedInt(ast::UintTy::U64)   => tcx.types.u64,
            UnsignedInt(ast::UintTy::U128)  => tcx.types.u128,
            UnsignedInt(ast::UintTy::Usize) => tcx.types.usize,
        }
    }

    fn initial_discriminant<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Discr<'tcx> {
        Discr {
            val: 0,
            ty: self.to_ty(tcx)
        }
    }

    fn disr_incr<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        val: Option<Discr<'tcx>>,
    ) -> Option<Discr<'tcx>> {
        if let Some(val) = val {
            assert_eq!(self.to_ty(tcx), val.ty);
            let (new, oflo) = val.checked_add(tcx, 1);
            if oflo {
                None
            } else {
                Some(new)
            }
        } else {
            Some(self.initial_discriminant(tcx))
        }
    }
}


#[derive(Clone)]
pub enum CopyImplementationError<'tcx> {
    InfrigingFields(Vec<&'tcx ty::FieldDef>),
    NotAnAdt,
    HasDestructor,
}

/// Describes whether a type is representable. For types that are not
/// representable, 'SelfRecursive' and 'ContainsRecursive' are used to
/// distinguish between types that are recursive with themselves and types that
/// contain a different recursive type. These cases can therefore be treated
/// differently when reporting errors.
///
/// The ordering of the cases is significant. They are sorted so that cmp::max
/// will keep the "more erroneous" of two values.
#[derive(Clone, PartialOrd, Ord, Eq, PartialEq, Debug)]
pub enum Representability {
    Representable,
    ContainsRecursive,
    SelfRecursive(Vec<Span>),
}

impl<'tcx> ty::ParamEnv<'tcx> {
    pub fn can_type_implement_copy<'a>(self,
                                       tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       self_type: Ty<'tcx>)
                                       -> Result<(), CopyImplementationError<'tcx>> {
        // FIXME: (@jroesch) float this code up
        tcx.infer_ctxt().enter(|infcx| {
            let (adt, substs) = match self_type.sty {
                // These types used to have a builtin impl.
                // Now libcore provides that impl.
                ty::Uint(_) | ty::Int(_) | ty::Bool | ty::Float(_) |
                ty::Char | ty::RawPtr(..) | ty::Never |
                ty::Ref(_, _, hir::MutImmutable) => return Ok(()),

                ty::Adt(adt, substs) => (adt, substs),

                _ => return Err(CopyImplementationError::NotAnAdt),
            };

            let mut infringing = Vec::new();
            for variant in &adt.variants {
                for field in &variant.fields {
                    let ty = field.ty(tcx, substs);
                    if ty.references_error() {
                        continue;
                    }
                    let span = tcx.def_span(field.did);
                    let cause = ObligationCause { span, ..ObligationCause::dummy() };
                    let ctx = traits::FulfillmentContext::new();
                    match traits::fully_normalize(&infcx, ctx, cause, self, &ty) {
                        Ok(ty) => if !infcx.type_is_copy_modulo_regions(self, ty, span) {
                            infringing.push(field);
                        }
                        Err(errors) => {
                            infcx.report_fulfillment_errors(&errors, None, false);
                        }
                    };
                }
            }
            if !infringing.is_empty() {
                return Err(CopyImplementationError::InfrigingFields(infringing));
            }
            if adt.has_dtor(tcx) {
                return Err(CopyImplementationError::HasDestructor);
            }

            Ok(())
        })
    }
}

impl<'a, 'tcx> TyCtxt<'a, 'tcx, 'tcx> {
    /// Creates a hash of the type `Ty` which will be the same no matter what crate
    /// context it's calculated within. This is used by the `type_id` intrinsic.
    pub fn type_id_hash(self, ty: Ty<'tcx>) -> u64 {
        let mut hasher = StableHasher::new();
        let mut hcx = self.create_stable_hashing_context();

        // We want the type_id be independent of the types free regions, so we
        // erase them. The erase_regions() call will also anonymize bound
        // regions, which is desirable too.
        let ty = self.erase_regions(&ty);

        hcx.while_hashing_spans(false, |hcx| {
            hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
                ty.hash_stable(hcx, &mut hasher);
            });
        });
        hasher.finish()
    }
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn has_error_field(self, ty: Ty<'tcx>) -> bool {
        if let ty::Adt(def, substs) = ty.sty {
            for field in def.all_fields() {
                let field_ty = field.ty(self, substs);
                if let Error = field_ty.sty {
                    return true;
                }
            }
        }
        false
    }

    /// Returns the deeply last field of nested structures, or the same type,
    /// if not a structure at all. Corresponds to the only possible unsized
    /// field, and its type can be used to determine unsizing strategy.
    pub fn struct_tail(self, mut ty: Ty<'tcx>) -> Ty<'tcx> {
        loop {
            match ty.sty {
                ty::Adt(def, substs) => {
                    if !def.is_struct() {
                        break;
                    }
                    match def.non_enum_variant().fields.last() {
                        Some(f) => ty = f.ty(self, substs),
                        None => break,
                    }
                }

                ty::Tuple(tys) => {
                    if let Some((&last_ty, _)) = tys.split_last() {
                        ty = last_ty;
                    } else {
                        break;
                    }
                }

                _ => {
                    break;
                }
            }
        }
        ty
    }

    /// Same as applying struct_tail on `source` and `target`, but only
    /// keeps going as long as the two types are instances of the same
    /// structure definitions.
    /// For `(Foo<Foo<T>>, Foo<dyn Trait>)`, the result will be `(Foo<T>, Trait)`,
    /// whereas struct_tail produces `T`, and `Trait`, respectively.
    pub fn struct_lockstep_tails(self,
                                 source: Ty<'tcx>,
                                 target: Ty<'tcx>)
                                 -> (Ty<'tcx>, Ty<'tcx>) {
        let (mut a, mut b) = (source, target);
        loop {
            match (&a.sty, &b.sty) {
                (&Adt(a_def, a_substs), &Adt(b_def, b_substs))
                        if a_def == b_def && a_def.is_struct() => {
                    if let Some(f) = a_def.non_enum_variant().fields.last() {
                        a = f.ty(self, a_substs);
                        b = f.ty(self, b_substs);
                    } else {
                        break;
                    }
                },
                (&Tuple(a_tys), &Tuple(b_tys))
                        if a_tys.len() == b_tys.len() => {
                    if let Some(a_last) = a_tys.last() {
                        a = a_last;
                        b = b_tys.last().unwrap();
                    } else {
                        break;
                    }
                },
                _ => break,
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
    /// N.B., in some cases, particularly around higher-ranked bounds,
    /// this function returns a kind of conservative approximation.
    /// That is, all regions returned by this function are definitely
    /// required, but there may be other region bounds that are not
    /// returned, as well as requirements like `for<'a> T: 'a`.
    ///
    /// Requires that trait definitions have been processed so that we can
    /// elaborate predicates and walk supertraits.
    //
    // FIXME: callers may only have a `&[Predicate]`, not a `Vec`, so that's
    // what this code should accept.
    pub fn required_region_bounds(self,
                                  erased_self_ty: Ty<'tcx>,
                                  predicates: Vec<ty::Predicate<'tcx>>)
                                  -> Vec<ty::Region<'tcx>>    {
        debug!("required_region_bounds(erased_self_ty={:?}, predicates={:?})",
               erased_self_ty,
               predicates);

        assert!(!erased_self_ty.has_escaping_bound_vars());

        traits::elaborate_predicates(self, predicates)
            .filter_map(|predicate| {
                match predicate {
                    ty::Predicate::Projection(..) |
                    ty::Predicate::Trait(..) |
                    ty::Predicate::Subtype(..) |
                    ty::Predicate::WellFormed(..) |
                    ty::Predicate::ObjectSafe(..) |
                    ty::Predicate::ClosureKind(..) |
                    ty::Predicate::RegionOutlives(..) |
                    ty::Predicate::ConstEvaluatable(..) => {
                        None
                    }
                    ty::Predicate::TypeOutlives(predicate) => {
                        // Search for a bound of the form `erased_self_ty
                        // : 'a`, but be wary of something like `for<'a>
                        // erased_self_ty : 'a` (we interpret a
                        // higher-ranked bound like that as 'static,
                        // though at present the code in `fulfill.rs`
                        // considers such bounds to be unsatisfiable, so
                        // it's kind of a moot point since you could never
                        // construct such an object, but this seems
                        // correct even if that code changes).
                        let ty::OutlivesPredicate(ref t, ref r) = predicate.skip_binder();
                        if t == &erased_self_ty && !r.has_escaping_bound_vars() {
                            Some(*r)
                        } else {
                            None
                        }
                    }
                }
            })
            .collect()
    }

    /// Calculate the destructor of a given type.
    pub fn calculate_dtor(
        self,
        adt_did: DefId,
        validate: &mut dyn FnMut(Self, DefId) -> Result<(), ErrorReported>
    ) -> Option<ty::Destructor> {
        let drop_trait = if let Some(def_id) = self.lang_items().drop_trait() {
            def_id
        } else {
            return None;
        };

        self.ensure().coherent_trait(drop_trait);

        let mut dtor_did = None;
        let ty = self.type_of(adt_did);
        self.for_each_relevant_impl(drop_trait, ty, |impl_did| {
            if let Some(item) = self.associated_items(impl_did).next() {
                if validate(self, impl_did).is_ok() {
                    dtor_did = Some(item.def_id);
                }
            }
        });

        Some(ty::Destructor { did: dtor_did? })
    }

    /// Returns the set of types that are required to be alive in
    /// order to run the destructor of `def` (see RFCs 769 and
    /// 1238).
    ///
    /// Note that this returns only the constraints for the
    /// destructor of `def` itself. For the destructors of the
    /// contents, you need `adt_dtorck_constraint`.
    pub fn destructor_constraints(self, def: &'tcx ty::AdtDef)
                                  -> Vec<ty::subst::Kind<'tcx>>
    {
        let dtor = match def.destructor(self) {
            None => {
                debug!("destructor_constraints({:?}) - no dtor", def.did);
                return vec![]
            }
            Some(dtor) => dtor.did
        };

        // RFC 1238: if the destructor method is tagged with the
        // attribute `unsafe_destructor_blind_to_params`, then the
        // compiler is being instructed to *assume* that the
        // destructor will not access borrowed data,
        // even if such data is otherwise reachable.
        //
        // Such access can be in plain sight (e.g., dereferencing
        // `*foo.0` of `Foo<'a>(&'a u32)`) or indirectly hidden
        // (e.g., calling `foo.0.clone()` of `Foo<T:Clone>`).
        if self.has_attr(dtor, "unsafe_destructor_blind_to_params") {
            debug!("destructor_constraint({:?}) - blind", def.did);
            return vec![];
        }

        let impl_def_id = self.associated_item(dtor).container.id();
        let impl_generics = self.generics_of(impl_def_id);

        // We have a destructor - all the parameters that are not
        // pure_wrt_drop (i.e, don't have a #[may_dangle] attribute)
        // must be live.

        // We need to return the list of parameters from the ADTs
        // generics/substs that correspond to impure parameters on the
        // impl's generics. This is a bit ugly, but conceptually simple:
        //
        // Suppose our ADT looks like the following
        //
        //     struct S<X, Y, Z>(X, Y, Z);
        //
        // and the impl is
        //
        //     impl<#[may_dangle] P0, P1, P2> Drop for S<P1, P2, P0>
        //
        // We want to return the parameters (X, Y). For that, we match
        // up the item-substs <X, Y, Z> with the substs on the impl ADT,
        // <P1, P2, P0>, and then look up which of the impl substs refer to
        // parameters marked as pure.

        let impl_substs = match self.type_of(impl_def_id).sty {
            ty::Adt(def_, substs) if def_ == def => substs,
            _ => bug!()
        };

        let item_substs = match self.type_of(def.did).sty {
            ty::Adt(def_, substs) if def_ == def => substs,
            _ => bug!()
        };

        let result = item_substs.iter().zip(impl_substs.iter())
            .filter(|&(_, &k)| {
                match k.unpack() {
                    UnpackedKind::Lifetime(&ty::RegionKind::ReEarlyBound(ref ebr)) => {
                        !impl_generics.region_param(ebr, self).pure_wrt_drop
                    }
                    UnpackedKind::Type(&ty::TyS {
                        sty: ty::Param(ref pt), ..
                    }) => {
                        !impl_generics.type_param(pt, self).pure_wrt_drop
                    }
                    UnpackedKind::Lifetime(_) | UnpackedKind::Type(_) => {
                        // not a type or region param - this should be reported
                        // as an error.
                        false
                    }
                }
            })
            .map(|(&item_param, _)| item_param)
            .collect();
        debug!("destructor_constraint({:?}) = {:?}", def.did, result);
        result
    }

    /// Returns `true` if `def_id` refers to a closure (e.g., `|x| x * 2`). Note
    /// that closures have a `DefId`, but the closure *expression* also
    /// has a `HirId` that is located within the context where the
    /// closure appears (and, sadly, a corresponding `NodeId`, since
    /// those are not yet phased out). The parent of the closure's
    /// `DefId` will also be the context where it appears.
    pub fn is_closure(self, def_id: DefId) -> bool {
        self.def_key(def_id).disambiguated_data.data == DefPathData::ClosureExpr
    }

    /// Returns `true` if `def_id` refers to a trait (i.e., `trait Foo { ... }`).
    pub fn is_trait(self, def_id: DefId) -> bool {
        if let DefPathData::Trait(_) = self.def_key(def_id).disambiguated_data.data {
            true
        } else {
            false
        }
    }

    /// Returns `true` if `def_id` refers to a trait alias (i.e., `trait Foo = ...;`),
    /// and `false` otherwise.
    pub fn is_trait_alias(self, def_id: DefId) -> bool {
        if let DefPathData::TraitAlias(_) = self.def_key(def_id).disambiguated_data.data {
            true
        } else {
            false
        }
    }

    /// Returns `true` if this `DefId` refers to the implicit constructor for
    /// a tuple struct like `struct Foo(u32)`, and `false` otherwise.
    pub fn is_struct_constructor(self, def_id: DefId) -> bool {
        self.def_key(def_id).disambiguated_data.data == DefPathData::StructCtor
    }

    /// Given the `DefId` of a fn or closure, returns the `DefId` of
    /// the innermost fn item that the closure is contained within.
    /// This is a significant `DefId` because, when we do
    /// type-checking, we type-check this fn item and all of its
    /// (transitive) closures together. Therefore, when we fetch the
    /// `typeck_tables_of` the closure, for example, we really wind up
    /// fetching the `typeck_tables_of` the enclosing fn item.
    pub fn closure_base_def_id(self, def_id: DefId) -> DefId {
        let mut def_id = def_id;
        while self.is_closure(def_id) {
            def_id = self.parent_def_id(def_id).unwrap_or_else(|| {
                bug!("closure {:?} has no parent", def_id);
            });
        }
        def_id
    }

    /// Given the `DefId` and substs a closure, creates the type of
    /// `self` argument that the closure expects. For example, for a
    /// `Fn` closure, this would return a reference type `&T` where
    /// `T = closure_ty`.
    ///
    /// Returns `None` if this closure's kind has not yet been inferred.
    /// This should only be possible during type checking.
    ///
    /// Note that the return value is a late-bound region and hence
    /// wrapped in a binder.
    pub fn closure_env_ty(self,
                          closure_def_id: DefId,
                          closure_substs: ty::ClosureSubsts<'tcx>)
                          -> Option<ty::Binder<Ty<'tcx>>>
    {
        let closure_ty = self.mk_closure(closure_def_id, closure_substs);
        let env_region = ty::ReLateBound(ty::INNERMOST, ty::BrEnv);
        let closure_kind_ty = closure_substs.closure_kind_ty(closure_def_id, self);
        let closure_kind = closure_kind_ty.to_opt_closure_kind()?;
        let env_ty = match closure_kind {
            ty::ClosureKind::Fn => self.mk_imm_ref(self.mk_region(env_region), closure_ty),
            ty::ClosureKind::FnMut => self.mk_mut_ref(self.mk_region(env_region), closure_ty),
            ty::ClosureKind::FnOnce => closure_ty,
        };
        Some(ty::Binder::bind(env_ty))
    }

    /// Given the `DefId` of some item that has no type parameters, make
    /// a suitable "empty substs" for it.
    pub fn empty_substs_for_def_id(self, item_def_id: DefId) -> SubstsRef<'tcx> {
        InternalSubsts::for_item(self, item_def_id, |param, _| {
            match param.kind {
                GenericParamDefKind::Lifetime => self.types.re_erased.into(),
                GenericParamDefKind::Type {..} => {
                    bug!("empty_substs_for_def_id: {:?} has type parameters", item_def_id)
                }
            }
        })
    }

    /// Returns `true` if the node pointed to by `def_id` is a static item, and its mutability.
    pub fn is_static(&self, def_id: DefId) -> Option<hir::Mutability> {
        if let Some(node) = self.hir().get_if_local(def_id) {
            match node {
                Node::Item(&hir::Item {
                    node: hir::ItemKind::Static(_, mutbl, _), ..
                }) => Some(mutbl),
                Node::ForeignItem(&hir::ForeignItem {
                    node: hir::ForeignItemKind::Static(_, is_mutbl), ..
                }) =>
                    Some(if is_mutbl {
                        hir::Mutability::MutMutable
                    } else {
                        hir::Mutability::MutImmutable
                    }),
                _ => None
            }
        } else {
            match self.describe_def(def_id) {
                Some(Def::Static(_, is_mutbl)) =>
                    Some(if is_mutbl {
                        hir::Mutability::MutMutable
                    } else {
                        hir::Mutability::MutImmutable
                    }),
                _ => None
            }
        }
    }

    /// Expands the given impl trait type, stopping if the type is recursive.
    pub fn try_expand_impl_trait_type(
        self,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Result<Ty<'tcx>, Ty<'tcx>> {
        use crate::ty::fold::TypeFolder;

        struct OpaqueTypeExpander<'a, 'gcx, 'tcx> {
            // Contains the DefIds of the opaque types that are currently being
            // expanded. When we expand an opaque type we insert the DefId of
            // that type, and when we finish expanding that type we remove the
            // its DefId.
            seen_opaque_tys: FxHashSet<DefId>,
            primary_def_id: DefId,
            found_recursion: bool,
            tcx: TyCtxt<'a, 'gcx, 'tcx>,
        }

        impl<'a, 'gcx, 'tcx> OpaqueTypeExpander<'a, 'gcx, 'tcx> {
            fn expand_opaque_ty(
                &mut self,
                def_id: DefId,
                substs: SubstsRef<'tcx>,
            ) -> Option<Ty<'tcx>> {
                if self.found_recursion {
                    None
                } else if self.seen_opaque_tys.insert(def_id) {
                    let generic_ty = self.tcx.type_of(def_id);
                    let concrete_ty = generic_ty.subst(self.tcx, substs);
                    let Ok(expanded_ty) = self.fold_ty(concrete_ty);
                    self.seen_opaque_tys.remove(&def_id);
                    Some(expanded_ty)
                } else {
                    // If another opaque type that we contain is recursive, then it
                    // will report the error, so we don't have to.
                    self.found_recursion = def_id == self.primary_def_id;
                    None
                }
            }
        }

        impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for OpaqueTypeExpander<'a, 'gcx, 'tcx> {
            type Error = !;

            fn tcx(&self) -> TyCtxt<'_, 'gcx, 'tcx> {
                self.tcx
            }

            fn fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, !> {
                if let ty::Opaque(def_id, substs) = t.sty {
                    Ok(self.expand_opaque_ty(def_id, substs).unwrap_or(t))
                } else {
                    t.super_fold_with(self)
                }
            }
        }

        let mut visitor = OpaqueTypeExpander {
            seen_opaque_tys: FxHashSet::default(),
            primary_def_id: def_id,
            found_recursion: false,
            tcx: self,
        };
        let expanded_type = visitor.expand_opaque_ty(def_id, substs).unwrap();
        if visitor.found_recursion {
            Err(expanded_type)
        } else {
            Ok(expanded_type)
        }
    }

    pub fn force_eval_array_length(self, x: ty::LazyConst<'tcx>) ->  Result<u64, ErrorReported> {
        match x {
            ty::LazyConst::Unevaluated(def_id, substs) => {
                // FIXME(eddyb) get the right param_env.
                let param_env = ty::ParamEnv::empty();
                if let Some(substs) = self.lift_to_global(&substs) {
                    let instance = ty::Instance::resolve(
                        self.global_tcx(),
                        param_env,
                        def_id,
                        substs,
                    );
                    if let Some(instance) = instance {
                        let cid = GlobalId {
                            instance,
                            promoted: None
                        };
                        if let Some(s) = self.const_eval(param_env.and(cid))
                            .ok()
                            .map(|c| c.unwrap_usize(self)) {
                                return Ok(s)
                            }
                    }
                }
                self.sess.delay_span_bug(self.def_span(def_id),
                                         "array length could not be evaluated");
                Err(ErrorReported)
            }
            ty::LazyConst::Evaluated(c) => c.assert_usize(self).ok_or_else(|| {
                self.sess.delay_span_bug(DUMMY_SP,
                                         "array length could not be evaluated");
                ErrorReported
            })
        }
    }
}

impl<'a, 'tcx> ty::TyS<'tcx> {
    /// Checks whether values of this type `T` are *moved* or *copied*
    /// when referenced -- this amounts to a check for whether `T:
    /// Copy`, but note that we **don't** consider lifetimes when
    /// doing this check. This means that we may generate MIR which
    /// does copies even when the type actually doesn't satisfy the
    /// full requirements for the `Copy` trait (cc #29149) -- this
    /// winds up being reported as an error during NLL borrow check.
    pub fn is_copy_modulo_regions(&'tcx self,
                                  tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  param_env: ty::ParamEnv<'tcx>,
                                  span: Span)
                                  -> bool {
        tcx.at(span).is_copy_raw(param_env.and(self))
    }

    /// Checks whether values of this type `T` have a size known at
    /// compile time (i.e., whether `T: Sized`). Lifetimes are ignored
    /// for the purposes of this check, so it can be an
    /// over-approximation in generic contexts, where one can have
    /// strange rules like `<T as Foo<'static>>::Bar: Sized` that
    /// actually carry lifetime requirements.
    pub fn is_sized(&'tcx self,
                    tcx_at: TyCtxtAt<'a, 'tcx, 'tcx>,
                    param_env: ty::ParamEnv<'tcx>)-> bool
    {
        tcx_at.is_sized_raw(param_env.and(self))
    }

    /// Checks whether values of this type `T` implement the `Freeze`
    /// trait -- frozen types are those that do not contain a
    /// `UnsafeCell` anywhere. This is a language concept used to
    /// distinguish "true immutability", which is relevant to
    /// optimization as well as the rules around static values. Note
    /// that the `Freeze` trait is not exposed to end users and is
    /// effectively an implementation detail.
    pub fn is_freeze(&'tcx self,
                     tcx: TyCtxt<'a, 'tcx, 'tcx>,
                     param_env: ty::ParamEnv<'tcx>,
                     span: Span)-> bool
    {
        tcx.at(span).is_freeze_raw(param_env.and(self))
    }

    /// If `ty.needs_drop(...)` returns `true`, then `ty` is definitely
    /// non-copy and *might* have a destructor attached; if it returns
    /// `false`, then `ty` definitely has no destructor (i.e., no drop glue).
    ///
    /// (Note that this implies that if `ty` has a destructor attached,
    /// then `needs_drop` will definitely return `true` for `ty`.)
    #[inline]
    pub fn needs_drop(&'tcx self,
                      tcx: TyCtxt<'a, 'tcx, 'tcx>,
                      param_env: ty::ParamEnv<'tcx>)
                      -> bool {
        tcx.needs_drop_raw(param_env.and(self)).0
    }

    pub fn same_type(a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
        match (&a.sty, &b.sty) {
            (&Adt(did_a, substs_a), &Adt(did_b, substs_b)) => {
                if did_a != did_b {
                    return false;
                }

                substs_a.types().zip(substs_b.types()).all(|(a, b)| Self::same_type(a, b))
            }
            _ => a == b,
        }
    }

    /// Check whether a type is representable. This means it cannot contain unboxed
    /// structural recursion. This check is needed for structs and enums.
    pub fn is_representable(&'tcx self,
                            tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            sp: Span)
                            -> Representability
    {
        // Iterate until something non-representable is found
        fn fold_repr<It: Iterator<Item=Representability>>(iter: It) -> Representability {
            iter.fold(Representability::Representable, |r1, r2| {
                match (r1, r2) {
                    (Representability::SelfRecursive(v1),
                     Representability::SelfRecursive(v2)) => {
                        Representability::SelfRecursive(v1.into_iter().chain(v2).collect())
                    }
                    (r1, r2) => cmp::max(r1, r2)
                }
            })
        }

        fn are_inner_types_recursive<'a, 'tcx>(
            tcx: TyCtxt<'a, 'tcx, 'tcx>, sp: Span,
            seen: &mut Vec<Ty<'tcx>>,
            representable_cache: &mut FxHashMap<Ty<'tcx>, Representability>,
            ty: Ty<'tcx>)
            -> Representability
        {
            match ty.sty {
                Tuple(ref ts) => {
                    // Find non representable
                    fold_repr(ts.iter().map(|ty| {
                        is_type_structurally_recursive(tcx, sp, seen, representable_cache, ty)
                    }))
                }
                // Fixed-length vectors.
                // FIXME(#11924) Behavior undecided for zero-length vectors.
                Array(ty, _) => {
                    is_type_structurally_recursive(tcx, sp, seen, representable_cache, ty)
                }
                Adt(def, substs) => {
                    // Find non representable fields with their spans
                    fold_repr(def.all_fields().map(|field| {
                        let ty = field.ty(tcx, substs);
                        let span = tcx.hir().span_if_local(field.did).unwrap_or(sp);
                        match is_type_structurally_recursive(tcx, span, seen,
                                                             representable_cache, ty)
                        {
                            Representability::SelfRecursive(_) => {
                                Representability::SelfRecursive(vec![span])
                            }
                            x => x,
                        }
                    }))
                }
                Closure(..) => {
                    // this check is run on type definitions, so we don't expect
                    // to see closure types
                    bug!("requires check invoked on inapplicable type: {:?}", ty)
                }
                _ => Representability::Representable,
            }
        }

        fn same_struct_or_enum<'tcx>(ty: Ty<'tcx>, def: &'tcx ty::AdtDef) -> bool {
            match ty.sty {
                Adt(ty_def, _) => {
                     ty_def == def
                }
                _ => false
            }
        }

        // Does the type `ty` directly (without indirection through a pointer)
        // contain any types on stack `seen`?
        fn is_type_structurally_recursive<'a, 'tcx>(
            tcx: TyCtxt<'a, 'tcx, 'tcx>,
            sp: Span,
            seen: &mut Vec<Ty<'tcx>>,
            representable_cache: &mut FxHashMap<Ty<'tcx>, Representability>,
            ty: Ty<'tcx>) -> Representability
        {
            debug!("is_type_structurally_recursive: {:?} {:?}", ty, sp);
            if let Some(representability) = representable_cache.get(ty) {
                debug!("is_type_structurally_recursive: {:?} {:?} - (cached) {:?}",
                       ty, sp, representability);
                return representability.clone();
            }

            let representability = is_type_structurally_recursive_inner(
                tcx, sp, seen, representable_cache, ty);

            representable_cache.insert(ty, representability.clone());
            representability
        }

        fn is_type_structurally_recursive_inner<'a, 'tcx>(
            tcx: TyCtxt<'a, 'tcx, 'tcx>,
            sp: Span,
            seen: &mut Vec<Ty<'tcx>>,
            representable_cache: &mut FxHashMap<Ty<'tcx>, Representability>,
            ty: Ty<'tcx>) -> Representability
        {
            match ty.sty {
                Adt(def, _) => {
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
                                return Representability::SelfRecursive(vec![sp]);
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
                            if ty::TyS::same_type(ty, seen_type) {
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
                    let out = are_inner_types_recursive(tcx, sp, seen, representable_cache, ty);
                    seen.pop();
                    out
                }
                _ => {
                    // No need to push in other cases.
                    are_inner_types_recursive(tcx, sp, seen, representable_cache, ty)
                }
            }
        }

        debug!("is_type_representable: {:?}", self);

        // To avoid a stack overflow when checking an enum variant or struct that
        // contains a different, structurally recursive type, maintain a stack
        // of seen types and check recursion for each of them (issues #3008, #3779).
        let mut seen: Vec<Ty<'_>> = Vec::new();
        let mut representable_cache = FxHashMap::default();
        let r = is_type_structurally_recursive(
            tcx, sp, &mut seen, &mut representable_cache, self);
        debug!("is_type_representable: {:?} is {:?}", self, r);
        r
    }
}

fn is_copy_raw<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                         query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                         -> bool
{
    let (param_env, ty) = query.into_parts();
    let trait_def_id = tcx.require_lang_item(lang_items::CopyTraitLangItem);
    tcx.infer_ctxt()
        .enter(|infcx| traits::type_known_to_meet_bound_modulo_regions(
            &infcx,
            param_env,
            ty,
            trait_def_id,
            DUMMY_SP,
        ))
}

fn is_sized_raw<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                          -> bool
{
    let (param_env, ty) = query.into_parts();
    let trait_def_id = tcx.require_lang_item(lang_items::SizedTraitLangItem);
    tcx.infer_ctxt()
        .enter(|infcx| traits::type_known_to_meet_bound_modulo_regions(
            &infcx,
            param_env,
            ty,
            trait_def_id,
            DUMMY_SP,
        ))
}

fn is_freeze_raw<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                           -> bool
{
    let (param_env, ty) = query.into_parts();
    let trait_def_id = tcx.require_lang_item(lang_items::FreezeTraitLangItem);
    tcx.infer_ctxt()
        .enter(|infcx| traits::type_known_to_meet_bound_modulo_regions(
            &infcx,
            param_env,
            ty,
            trait_def_id,
            DUMMY_SP,
        ))
}

#[derive(Clone)]
pub struct NeedsDrop(pub bool);

fn needs_drop_raw<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                            -> NeedsDrop
{
    let (param_env, ty) = query.into_parts();

    let needs_drop = |ty: Ty<'tcx>| -> bool {
        tcx.needs_drop_raw(param_env.and(ty)).0
    };

    assert!(!ty.needs_infer());

    NeedsDrop(match ty.sty {
        // Fast-path for primitive types
        ty::Infer(ty::FreshIntTy(_)) | ty::Infer(ty::FreshFloatTy(_)) |
        ty::Bool | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Never |
        ty::FnDef(..) | ty::FnPtr(_) | ty::Char | ty::GeneratorWitness(..) |
        ty::RawPtr(_) | ty::Ref(..) | ty::Str => false,

        // Foreign types can never have destructors
        ty::Foreign(..) => false,

        // `ManuallyDrop` doesn't have a destructor regardless of field types.
        ty::Adt(def, _) if Some(def.did) == tcx.lang_items().manually_drop() => false,

        // Issue #22536: We first query `is_copy_modulo_regions`.  It sees a
        // normalized version of the type, and therefore will definitely
        // know whether the type implements Copy (and thus needs no
        // cleanup/drop/zeroing) ...
        _ if ty.is_copy_modulo_regions(tcx, param_env, DUMMY_SP) => false,

        // ... (issue #22536 continued) but as an optimization, still use
        // prior logic of asking for the structural "may drop".

        // FIXME(#22815): Note that this is a conservative heuristic;
        // it may report that the type "may drop" when actual type does
        // not actually have a destructor associated with it. But since
        // the type absolutely did not have the `Copy` bound attached
        // (see above), it is sound to treat it as having a destructor.

        // User destructors are the only way to have concrete drop types.
        ty::Adt(def, _) if def.has_dtor(tcx) => true,

        // Can refer to a type which may drop.
        // FIXME(eddyb) check this against a ParamEnv.
        ty::Dynamic(..) | ty::Projection(..) | ty::Param(_) | ty::Bound(..) |
        ty::Placeholder(..) | ty::Opaque(..) | ty::Infer(_) | ty::Error => true,

        ty::UnnormalizedProjection(..) => bug!("only used with chalk-engine"),

        // Structural recursion.
        ty::Array(ty, _) | ty::Slice(ty) => needs_drop(ty),

        ty::Closure(def_id, ref substs) => substs.upvar_tys(def_id, tcx).any(needs_drop),

        // Pessimistically assume that all generators will require destructors
        // as we don't know if a destructor is a noop or not until after the MIR
        // state transformation pass
        ty::Generator(..) => true,

        ty::Tuple(ref tys) => tys.iter().cloned().any(needs_drop),

        // unions don't have destructors because of the child types,
        // only if they manually implement `Drop` (handled above).
        ty::Adt(def, _) if def.is_union() => false,

        ty::Adt(def, substs) =>
            def.variants.iter().any(
                |variant| variant.fields.iter().any(
                    |field| needs_drop(field.ty(tcx, substs)))),
    })
}

pub enum ExplicitSelf<'tcx> {
    ByValue,
    ByReference(ty::Region<'tcx>, hir::Mutability),
    ByRawPointer(hir::Mutability),
    ByBox,
    Other
}

impl<'tcx> ExplicitSelf<'tcx> {
    /// Categorizes an explicit self declaration like `self: SomeType`
    /// into either `self`, `&self`, `&mut self`, `Box<self>`, or
    /// `Other`.
    /// This is mainly used to require the arbitrary_self_types feature
    /// in the case of `Other`, to improve error messages in the common cases,
    /// and to make `Other` non-object-safe.
    ///
    /// Examples:
    ///
    /// ```
    /// impl<'a> Foo for &'a T {
    ///     // Legal declarations:
    ///     fn method1(self: &&'a T); // ExplicitSelf::ByReference
    ///     fn method2(self: &'a T); // ExplicitSelf::ByValue
    ///     fn method3(self: Box<&'a T>); // ExplicitSelf::ByBox
    ///     fn method4(self: Rc<&'a T>); // ExplicitSelf::Other
    ///
    ///     // Invalid cases will be caught by `check_method_receiver`:
    ///     fn method_err1(self: &'a mut T); // ExplicitSelf::Other
    ///     fn method_err2(self: &'static T) // ExplicitSelf::ByValue
    ///     fn method_err3(self: &&T) // ExplicitSelf::ByReference
    /// }
    /// ```
    ///
    pub fn determine<P>(
        self_arg_ty: Ty<'tcx>,
        is_self_ty: P
    ) -> ExplicitSelf<'tcx>
    where
        P: Fn(Ty<'tcx>) -> bool
    {
        use self::ExplicitSelf::*;

        match self_arg_ty.sty {
            _ if is_self_ty(self_arg_ty) => ByValue,
            ty::Ref(region, ty, mutbl) if is_self_ty(ty) => {
                ByReference(region, mutbl)
            }
            ty::RawPtr(ty::TypeAndMut { ty, mutbl }) if is_self_ty(ty) => {
                ByRawPointer(mutbl)
            }
            ty::Adt(def, _) if def.is_box() && is_self_ty(self_arg_ty.boxed_ty()) => {
                ByBox
            }
            _ => Other
        }
    }
}

pub fn provide(providers: &mut ty::query::Providers<'_>) {
    *providers = ty::query::Providers {
        is_copy_raw,
        is_sized_raw,
        is_freeze_raw,
        needs_drop_raw,
        ..*providers
    };
}
