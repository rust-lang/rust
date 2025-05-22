//! Miscellaneous type-system utilities that are too small to deserve their own modules.

use std::{fmt, iter};

use rustc_abi::{Float, Integer, IntegerType, Size};
use rustc_apfloat::Float as _;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::ErrorGuaranteed;
use rustc_hashes::Hash128;
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId};
use rustc_index::bit_set::GrowableBitSet;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, extension};
use rustc_session::Limit;
use rustc_span::sym;
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

use super::TypingEnv;
use crate::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use crate::mir;
use crate::query::Providers;
use crate::ty::layout::{FloatExt, IntegerExt};
use crate::ty::{
    self, Asyncness, FallibleTypeFolder, GenericArgKind, GenericArgsRef, Ty, TyCtxt, TypeFoldable,
    TypeFolder, TypeSuperFoldable, TypeVisitableExt, Upcast, fold_regions,
};

#[derive(Copy, Clone, Debug)]
pub struct Discr<'tcx> {
    /// Bit representation of the discriminant (e.g., `-128i8` is `0xFF_u128`).
    pub val: u128,
    pub ty: Ty<'tcx>,
}

/// Used as an input to [`TyCtxt::uses_unique_generic_params`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CheckRegions {
    No,
    /// Only permit parameter regions. This should be used
    /// for everything apart from functions, which may use
    /// `ReBound` to represent late-bound regions.
    OnlyParam,
    /// Check region parameters from a function definition.
    /// Allows `ReEarlyParam` and `ReBound` to handle early
    /// and late-bound region parameters.
    FromFunction,
}

#[derive(Copy, Clone, Debug)]
pub enum NotUniqueParam<'tcx> {
    DuplicateParam(ty::GenericArg<'tcx>),
    NotParam(ty::GenericArg<'tcx>),
}

impl<'tcx> fmt::Display for Discr<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self.ty.kind() {
            ty::Int(ity) => {
                let size = ty::tls::with(|tcx| Integer::from_int_ty(&tcx, ity).size());
                let x = self.val;
                // sign extend the raw representation to be an i128
                let x = size.sign_extend(x) as i128;
                write!(fmt, "{x}")
            }
            _ => write!(fmt, "{}", self.val),
        }
    }
}

impl<'tcx> Discr<'tcx> {
    /// Adds `1` to the value and wraps around if the maximum for the type is reached.
    pub fn wrap_incr(self, tcx: TyCtxt<'tcx>) -> Self {
        self.checked_add(tcx, 1).0
    }
    pub fn checked_add(self, tcx: TyCtxt<'tcx>, n: u128) -> (Self, bool) {
        let (size, signed) = self.ty.int_size_and_signed(tcx);
        let (val, oflo) = if signed {
            let min = size.signed_int_min();
            let max = size.signed_int_max();
            let val = size.sign_extend(self.val);
            assert!(n < (i128::MAX as u128));
            let n = n as i128;
            let oflo = val > max - n;
            let val = if oflo { min + (n - (max - val) - 1) } else { val + n };
            // zero the upper bits
            let val = val as u128;
            let val = size.truncate(val);
            (val, oflo)
        } else {
            let max = size.unsigned_int_max();
            let val = self.val;
            let oflo = val > max - n;
            let val = if oflo { n - (max - val) - 1 } else { val + n };
            (val, oflo)
        };
        (Self { val, ty: self.ty }, oflo)
    }
}

#[extension(pub trait IntTypeExt)]
impl IntegerType {
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match self {
            IntegerType::Pointer(true) => tcx.types.isize,
            IntegerType::Pointer(false) => tcx.types.usize,
            IntegerType::Fixed(i, s) => i.to_ty(tcx, *s),
        }
    }

    fn initial_discriminant<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Discr<'tcx> {
        Discr { val: 0, ty: self.to_ty(tcx) }
    }

    fn disr_incr<'tcx>(&self, tcx: TyCtxt<'tcx>, val: Option<Discr<'tcx>>) -> Option<Discr<'tcx>> {
        if let Some(val) = val {
            assert_eq!(self.to_ty(tcx), val.ty);
            let (new, oflo) = val.checked_add(tcx, 1);
            if oflo { None } else { Some(new) }
        } else {
            Some(self.initial_discriminant(tcx))
        }
    }
}

impl<'tcx> TyCtxt<'tcx> {
    /// Creates a hash of the type `Ty` which will be the same no matter what crate
    /// context it's calculated within. This is used by the `type_id` intrinsic.
    pub fn type_id_hash(self, ty: Ty<'tcx>) -> Hash128 {
        // We want the type_id be independent of the types free regions, so we
        // erase them. The erase_regions() call will also anonymize bound
        // regions, which is desirable too.
        let ty = self.erase_regions(ty);

        self.with_stable_hashing_context(|mut hcx| {
            let mut hasher = StableHasher::new();
            hcx.while_hashing_spans(false, |hcx| ty.hash_stable(hcx, &mut hasher));
            hasher.finish()
        })
    }

    pub fn res_generics_def_id(self, res: Res) -> Option<DefId> {
        match res {
            Res::Def(DefKind::Ctor(CtorOf::Variant, _), def_id) => {
                Some(self.parent(self.parent(def_id)))
            }
            Res::Def(DefKind::Variant | DefKind::Ctor(CtorOf::Struct, _), def_id) => {
                Some(self.parent(def_id))
            }
            // Other `DefKind`s don't have generics and would ICE when calling
            // `generics_of`.
            Res::Def(
                DefKind::Struct
                | DefKind::Union
                | DefKind::Enum
                | DefKind::Trait
                | DefKind::OpaqueTy
                | DefKind::TyAlias
                | DefKind::ForeignTy
                | DefKind::TraitAlias
                | DefKind::AssocTy
                | DefKind::Fn
                | DefKind::AssocFn
                | DefKind::AssocConst
                | DefKind::Impl { .. },
                def_id,
            ) => Some(def_id),
            Res::Err => None,
            _ => None,
        }
    }

    /// Checks whether `ty: Copy` holds while ignoring region constraints.
    ///
    /// This impacts whether values of `ty` are *moved* or *copied*
    /// when referenced. This means that we may generate MIR which
    /// does copies even when the type actually doesn't satisfy the
    /// full requirements for the `Copy` trait (cc #29149) -- this
    /// winds up being reported as an error during NLL borrow check.
    ///
    /// This function should not be used if there is an `InferCtxt` available.
    /// Use `InferCtxt::type_is_copy_modulo_regions` instead.
    pub fn type_is_copy_modulo_regions(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> bool {
        ty.is_trivially_pure_clone_copy() || self.is_copy_raw(typing_env.as_query_input(ty))
    }

    /// Checks whether `ty: UseCloned` holds while ignoring region constraints.
    ///
    /// This function should not be used if there is an `InferCtxt` available.
    /// Use `InferCtxt::type_is_copy_modulo_regions` instead.
    pub fn type_is_use_cloned_modulo_regions(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> bool {
        ty.is_trivially_pure_clone_copy() || self.is_use_cloned_raw(typing_env.as_query_input(ty))
    }

    /// Returns the deeply last field of nested structures, or the same type if
    /// not a structure at all. Corresponds to the only possible unsized field,
    /// and its type can be used to determine unsizing strategy.
    ///
    /// Should only be called if `ty` has no inference variables and does not
    /// need its lifetimes preserved (e.g. as part of codegen); otherwise
    /// normalization attempt may cause compiler bugs.
    pub fn struct_tail_for_codegen(
        self,
        ty: Ty<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self;
        tcx.struct_tail_raw(ty, |ty| tcx.normalize_erasing_regions(typing_env, ty), || {})
    }

    /// Returns true if a type has metadata.
    pub fn type_has_metadata(self, ty: Ty<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> bool {
        if ty.is_sized(self, typing_env) {
            return false;
        }

        let tail = self.struct_tail_for_codegen(ty, typing_env);
        match tail.kind() {
            ty::Foreign(..) => false,
            ty::Str | ty::Slice(..) | ty::Dynamic(..) => true,
            _ => bug!("unexpected unsized tail: {:?}", tail),
        }
    }

    /// Returns the deeply last field of nested structures, or the same type if
    /// not a structure at all. Corresponds to the only possible unsized field,
    /// and its type can be used to determine unsizing strategy.
    ///
    /// This is parameterized over the normalization strategy (i.e. how to
    /// handle `<T as Trait>::Assoc` and `impl Trait`). You almost certainly do
    /// **NOT** want to pass the identity function here, unless you know what
    /// you're doing, or you're within normalization code itself and will handle
    /// an unnormalized tail recursively.
    ///
    /// See also `struct_tail_for_codegen`, which is suitable for use
    /// during codegen.
    pub fn struct_tail_raw(
        self,
        mut ty: Ty<'tcx>,
        mut normalize: impl FnMut(Ty<'tcx>) -> Ty<'tcx>,
        // This is currently used to allow us to walk a ValTree
        // in lockstep with the type in order to get the ValTree branch that
        // corresponds to an unsized field.
        mut f: impl FnMut() -> (),
    ) -> Ty<'tcx> {
        let recursion_limit = self.recursion_limit();
        for iteration in 0.. {
            if !recursion_limit.value_within_limit(iteration) {
                let suggested_limit = match recursion_limit {
                    Limit(0) => Limit(2),
                    limit => limit * 2,
                };
                let reported = self
                    .dcx()
                    .emit_err(crate::error::RecursionLimitReached { ty, suggested_limit });
                return Ty::new_error(self, reported);
            }
            match *ty.kind() {
                ty::Adt(def, args) => {
                    if !def.is_struct() {
                        break;
                    }
                    match def.non_enum_variant().tail_opt() {
                        Some(field) => {
                            f();
                            ty = field.ty(self, args);
                        }
                        None => break,
                    }
                }

                ty::Tuple(tys) if let Some((&last_ty, _)) = tys.split_last() => {
                    f();
                    ty = last_ty;
                }

                ty::Tuple(_) => break,

                ty::Pat(inner, _) => {
                    f();
                    ty = inner;
                }

                ty::Alias(..) => {
                    let normalized = normalize(ty);
                    if ty == normalized {
                        return ty;
                    } else {
                        ty = normalized;
                    }
                }

                _ => {
                    break;
                }
            }
        }
        ty
    }

    /// Same as applying `struct_tail` on `source` and `target`, but only
    /// keeps going as long as the two types are instances of the same
    /// structure definitions.
    /// For `(Foo<Foo<T>>, Foo<dyn Trait>)`, the result will be `(Foo<T>, dyn Trait)`,
    /// whereas struct_tail produces `T`, and `Trait`, respectively.
    ///
    /// Should only be called if the types have no inference variables and do
    /// not need their lifetimes preserved (e.g., as part of codegen); otherwise,
    /// normalization attempt may cause compiler bugs.
    pub fn struct_lockstep_tails_for_codegen(
        self,
        source: Ty<'tcx>,
        target: Ty<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> (Ty<'tcx>, Ty<'tcx>) {
        let tcx = self;
        tcx.struct_lockstep_tails_raw(source, target, |ty| {
            tcx.normalize_erasing_regions(typing_env, ty)
        })
    }

    /// Same as applying `struct_tail` on `source` and `target`, but only
    /// keeps going as long as the two types are instances of the same
    /// structure definitions.
    /// For `(Foo<Foo<T>>, Foo<dyn Trait>)`, the result will be `(Foo<T>, Trait)`,
    /// whereas struct_tail produces `T`, and `Trait`, respectively.
    ///
    /// See also `struct_lockstep_tails_for_codegen`, which is suitable for use
    /// during codegen.
    pub fn struct_lockstep_tails_raw(
        self,
        source: Ty<'tcx>,
        target: Ty<'tcx>,
        normalize: impl Fn(Ty<'tcx>) -> Ty<'tcx>,
    ) -> (Ty<'tcx>, Ty<'tcx>) {
        let (mut a, mut b) = (source, target);
        loop {
            match (a.kind(), b.kind()) {
                (&ty::Adt(a_def, a_args), &ty::Adt(b_def, b_args))
                    if a_def == b_def && a_def.is_struct() =>
                {
                    if let Some(f) = a_def.non_enum_variant().tail_opt() {
                        a = f.ty(self, a_args);
                        b = f.ty(self, b_args);
                    } else {
                        break;
                    }
                }
                (&ty::Tuple(a_tys), &ty::Tuple(b_tys)) if a_tys.len() == b_tys.len() => {
                    if let Some(&a_last) = a_tys.last() {
                        a = a_last;
                        b = *b_tys.last().unwrap();
                    } else {
                        break;
                    }
                }
                (ty::Alias(..), _) | (_, ty::Alias(..)) => {
                    // If either side is a projection, attempt to
                    // progress via normalization. (Should be safe to
                    // apply to both sides as normalization is
                    // idempotent.)
                    let a_norm = normalize(a);
                    let b_norm = normalize(b);
                    if a == a_norm && b == b_norm {
                        break;
                    } else {
                        a = a_norm;
                        b = b_norm;
                    }
                }

                _ => break,
            }
        }
        (a, b)
    }

    /// Calculate the destructor of a given type.
    pub fn calculate_dtor(
        self,
        adt_did: LocalDefId,
        validate: impl Fn(Self, LocalDefId) -> Result<(), ErrorGuaranteed>,
    ) -> Option<ty::Destructor> {
        let drop_trait = self.lang_items().drop_trait()?;
        self.ensure_ok().coherent_trait(drop_trait).ok()?;

        let mut dtor_candidate = None;
        // `Drop` impls can only be written in the same crate as the adt, and cannot be blanket impls
        for &impl_did in self.local_trait_impls(drop_trait) {
            let Some(adt_def) = self.type_of(impl_did).skip_binder().ty_adt_def() else { continue };
            if adt_def.did() != adt_did.to_def_id() {
                continue;
            }

            if validate(self, impl_did).is_err() {
                // Already `ErrorGuaranteed`, no need to delay a span bug here.
                continue;
            }

            let Some(item_id) = self.associated_item_def_ids(impl_did).first() else {
                self.dcx()
                    .span_delayed_bug(self.def_span(impl_did), "Drop impl without drop function");
                continue;
            };

            if self.def_kind(item_id) != DefKind::AssocFn {
                self.dcx().span_delayed_bug(self.def_span(item_id), "drop is not a function");
                continue;
            }

            if let Some(old_item_id) = dtor_candidate {
                self.dcx()
                    .struct_span_err(self.def_span(item_id), "multiple drop impls found")
                    .with_span_note(self.def_span(old_item_id), "other impl here")
                    .delay_as_bug();
            }

            dtor_candidate = Some(*item_id);
        }

        let did = dtor_candidate?;
        Some(ty::Destructor { did })
    }

    /// Calculate the async destructor of a given type.
    pub fn calculate_async_dtor(
        self,
        adt_did: LocalDefId,
        validate: impl Fn(Self, LocalDefId) -> Result<(), ErrorGuaranteed>,
    ) -> Option<ty::AsyncDestructor> {
        let async_drop_trait = self.lang_items().async_drop_trait()?;
        self.ensure_ok().coherent_trait(async_drop_trait).ok()?;

        let mut dtor_candidate = None;
        // `AsyncDrop` impls can only be written in the same crate as the adt, and cannot be blanket impls
        for &impl_did in self.local_trait_impls(async_drop_trait) {
            let Some(adt_def) = self.type_of(impl_did).skip_binder().ty_adt_def() else { continue };
            if adt_def.did() != adt_did.to_def_id() {
                continue;
            }

            if validate(self, impl_did).is_err() {
                // Already `ErrorGuaranteed`, no need to delay a span bug here.
                continue;
            }

            if let Some(old_impl_did) = dtor_candidate {
                self.dcx()
                    .struct_span_err(self.def_span(impl_did), "multiple async drop impls found")
                    .with_span_note(self.def_span(old_impl_did), "other impl here")
                    .delay_as_bug();
            }

            dtor_candidate = Some(impl_did);
        }

        Some(ty::AsyncDestructor { impl_did: dtor_candidate?.into() })
    }

    /// Returns the set of types that are required to be alive in
    /// order to run the destructor of `def` (see RFCs 769 and
    /// 1238).
    ///
    /// Note that this returns only the constraints for the
    /// destructor of `def` itself. For the destructors of the
    /// contents, you need `adt_dtorck_constraint`.
    pub fn destructor_constraints(self, def: ty::AdtDef<'tcx>) -> Vec<ty::GenericArg<'tcx>> {
        let dtor = match def.destructor(self) {
            None => {
                debug!("destructor_constraints({:?}) - no dtor", def.did());
                return vec![];
            }
            Some(dtor) => dtor.did,
        };

        let impl_def_id = self.parent(dtor);
        let impl_generics = self.generics_of(impl_def_id);

        // We have a destructor - all the parameters that are not
        // pure_wrt_drop (i.e, don't have a #[may_dangle] attribute)
        // must be live.

        // We need to return the list of parameters from the ADTs
        // generics/args that correspond to impure parameters on the
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
        // up the item-args <X, Y, Z> with the args on the impl ADT,
        // <P1, P2, P0>, and then look up which of the impl args refer to
        // parameters marked as pure.

        let impl_args = match *self.type_of(impl_def_id).instantiate_identity().kind() {
            ty::Adt(def_, args) if def_ == def => args,
            _ => span_bug!(self.def_span(impl_def_id), "expected ADT for self type of `Drop` impl"),
        };

        let item_args = ty::GenericArgs::identity_for_item(self, def.did());

        let result = iter::zip(item_args, impl_args)
            .filter(|&(_, k)| {
                match k.unpack() {
                    GenericArgKind::Lifetime(region) => match region.kind() {
                        ty::ReEarlyParam(ebr) => {
                            !impl_generics.region_param(ebr, self).pure_wrt_drop
                        }
                        // Error: not a region param
                        _ => false,
                    },
                    GenericArgKind::Type(ty) => match *ty.kind() {
                        ty::Param(pt) => !impl_generics.type_param(pt, self).pure_wrt_drop,
                        // Error: not a type param
                        _ => false,
                    },
                    GenericArgKind::Const(ct) => match ct.kind() {
                        ty::ConstKind::Param(pc) => {
                            !impl_generics.const_param(pc, self).pure_wrt_drop
                        }
                        // Error: not a const param
                        _ => false,
                    },
                }
            })
            .map(|(item_param, _)| item_param)
            .collect();
        debug!("destructor_constraint({:?}) = {:?}", def.did(), result);
        result
    }

    /// Checks whether each generic argument is simply a unique generic parameter.
    pub fn uses_unique_generic_params(
        self,
        args: &[ty::GenericArg<'tcx>],
        ignore_regions: CheckRegions,
    ) -> Result<(), NotUniqueParam<'tcx>> {
        let mut seen = GrowableBitSet::default();
        let mut seen_late = FxHashSet::default();
        for arg in args {
            match arg.unpack() {
                GenericArgKind::Lifetime(lt) => match (ignore_regions, lt.kind()) {
                    (CheckRegions::FromFunction, ty::ReBound(di, reg)) => {
                        if !seen_late.insert((di, reg)) {
                            return Err(NotUniqueParam::DuplicateParam(lt.into()));
                        }
                    }
                    (CheckRegions::OnlyParam | CheckRegions::FromFunction, ty::ReEarlyParam(p)) => {
                        if !seen.insert(p.index) {
                            return Err(NotUniqueParam::DuplicateParam(lt.into()));
                        }
                    }
                    (CheckRegions::OnlyParam | CheckRegions::FromFunction, _) => {
                        return Err(NotUniqueParam::NotParam(lt.into()));
                    }
                    (CheckRegions::No, _) => {}
                },
                GenericArgKind::Type(t) => match t.kind() {
                    ty::Param(p) => {
                        if !seen.insert(p.index) {
                            return Err(NotUniqueParam::DuplicateParam(t.into()));
                        }
                    }
                    _ => return Err(NotUniqueParam::NotParam(t.into())),
                },
                GenericArgKind::Const(c) => match c.kind() {
                    ty::ConstKind::Param(p) => {
                        if !seen.insert(p.index) {
                            return Err(NotUniqueParam::DuplicateParam(c.into()));
                        }
                    }
                    _ => return Err(NotUniqueParam::NotParam(c.into())),
                },
            }
        }

        Ok(())
    }

    /// Returns `true` if `def_id` refers to a closure, coroutine, or coroutine-closure
    /// (i.e. an async closure). These are all represented by `hir::Closure`, and all
    /// have the same `DefKind`.
    ///
    /// Note that closures have a `DefId`, but the closure *expression* also has a
    // `HirId` that is located within the context where the closure appears (and, sadly,
    // a corresponding `NodeId`, since those are not yet phased out). The parent of
    // the closure's `DefId` will also be the context where it appears.
    pub fn is_closure_like(self, def_id: DefId) -> bool {
        matches!(self.def_kind(def_id), DefKind::Closure)
    }

    /// Returns `true` if `def_id` refers to a definition that does not have its own
    /// type-checking context, i.e. closure, coroutine or inline const.
    pub fn is_typeck_child(self, def_id: DefId) -> bool {
        matches!(
            self.def_kind(def_id),
            DefKind::Closure | DefKind::InlineConst | DefKind::SyntheticCoroutineBody
        )
    }

    /// Returns `true` if `def_id` refers to a trait (i.e., `trait Foo { ... }`).
    pub fn is_trait(self, def_id: DefId) -> bool {
        self.def_kind(def_id) == DefKind::Trait
    }

    /// Returns `true` if `def_id` refers to a trait alias (i.e., `trait Foo = ...;`),
    /// and `false` otherwise.
    pub fn is_trait_alias(self, def_id: DefId) -> bool {
        self.def_kind(def_id) == DefKind::TraitAlias
    }

    /// Returns `true` if this `DefId` refers to the implicit constructor for
    /// a tuple struct like `struct Foo(u32)`, and `false` otherwise.
    pub fn is_constructor(self, def_id: DefId) -> bool {
        matches!(self.def_kind(def_id), DefKind::Ctor(..))
    }

    /// Given the `DefId`, returns the `DefId` of the innermost item that
    /// has its own type-checking context or "inference environment".
    ///
    /// For example, a closure has its own `DefId`, but it is type-checked
    /// with the containing item. Similarly, an inline const block has its
    /// own `DefId` but it is type-checked together with the containing item.
    ///
    /// Therefore, when we fetch the
    /// `typeck` the closure, for example, we really wind up
    /// fetching the `typeck` the enclosing fn item.
    pub fn typeck_root_def_id(self, def_id: DefId) -> DefId {
        let mut def_id = def_id;
        while self.is_typeck_child(def_id) {
            def_id = self.parent(def_id);
        }
        def_id
    }

    /// Given the `DefId` and args a closure, creates the type of
    /// `self` argument that the closure expects. For example, for a
    /// `Fn` closure, this would return a reference type `&T` where
    /// `T = closure_ty`.
    ///
    /// Returns `None` if this closure's kind has not yet been inferred.
    /// This should only be possible during type checking.
    ///
    /// Note that the return value is a late-bound region and hence
    /// wrapped in a binder.
    pub fn closure_env_ty(
        self,
        closure_ty: Ty<'tcx>,
        closure_kind: ty::ClosureKind,
        env_region: ty::Region<'tcx>,
    ) -> Ty<'tcx> {
        match closure_kind {
            ty::ClosureKind::Fn => Ty::new_imm_ref(self, env_region, closure_ty),
            ty::ClosureKind::FnMut => Ty::new_mut_ref(self, env_region, closure_ty),
            ty::ClosureKind::FnOnce => closure_ty,
        }
    }

    /// Returns `true` if the node pointed to by `def_id` is a `static` item.
    #[inline]
    pub fn is_static(self, def_id: DefId) -> bool {
        matches!(self.def_kind(def_id), DefKind::Static { .. })
    }

    #[inline]
    pub fn static_mutability(self, def_id: DefId) -> Option<hir::Mutability> {
        if let DefKind::Static { mutability, .. } = self.def_kind(def_id) {
            Some(mutability)
        } else {
            None
        }
    }

    /// Returns `true` if this is a `static` item with the `#[thread_local]` attribute.
    pub fn is_thread_local_static(self, def_id: DefId) -> bool {
        self.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::THREAD_LOCAL)
    }

    /// Returns `true` if the node pointed to by `def_id` is a mutable `static` item.
    #[inline]
    pub fn is_mutable_static(self, def_id: DefId) -> bool {
        self.static_mutability(def_id) == Some(hir::Mutability::Mut)
    }

    /// Returns `true` if the item pointed to by `def_id` is a thread local which needs a
    /// thread local shim generated.
    #[inline]
    pub fn needs_thread_local_shim(self, def_id: DefId) -> bool {
        !self.sess.target.dll_tls_export
            && self.is_thread_local_static(def_id)
            && !self.is_foreign_item(def_id)
    }

    /// Returns the type a reference to the thread local takes in MIR.
    pub fn thread_local_ptr_ty(self, def_id: DefId) -> Ty<'tcx> {
        let static_ty = self.type_of(def_id).instantiate_identity();
        if self.is_mutable_static(def_id) {
            Ty::new_mut_ptr(self, static_ty)
        } else if self.is_foreign_item(def_id) {
            Ty::new_imm_ptr(self, static_ty)
        } else {
            // FIXME: These things don't *really* have 'static lifetime.
            Ty::new_imm_ref(self, self.lifetimes.re_static, static_ty)
        }
    }

    /// Get the type of the pointer to the static that we use in MIR.
    pub fn static_ptr_ty(self, def_id: DefId, typing_env: ty::TypingEnv<'tcx>) -> Ty<'tcx> {
        // Make sure that any constants in the static's type are evaluated.
        let static_ty =
            self.normalize_erasing_regions(typing_env, self.type_of(def_id).instantiate_identity());

        // Make sure that accesses to unsafe statics end up using raw pointers.
        // For thread-locals, this needs to be kept in sync with `Rvalue::ty`.
        if self.is_mutable_static(def_id) {
            Ty::new_mut_ptr(self, static_ty)
        } else if self.is_foreign_item(def_id) {
            Ty::new_imm_ptr(self, static_ty)
        } else {
            Ty::new_imm_ref(self, self.lifetimes.re_erased, static_ty)
        }
    }

    /// Return the set of types that should be taken into account when checking
    /// trait bounds on a coroutine's internal state. This properly replaces
    /// `ReErased` with new existential bound lifetimes.
    pub fn coroutine_hidden_types(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<'tcx, ty::Binder<'tcx, &'tcx ty::List<Ty<'tcx>>>> {
        let coroutine_layout = self.mir_coroutine_witnesses(def_id);
        let mut vars = vec![];
        let bound_tys = self.mk_type_list_from_iter(
            coroutine_layout
                .as_ref()
                .map_or_else(|| [].iter(), |l| l.field_tys.iter())
                .filter(|decl| !decl.ignore_for_traits)
                .map(|decl| {
                    let ty = fold_regions(self, decl.ty, |re, debruijn| {
                        assert_eq!(re, self.lifetimes.re_erased);
                        let var = ty::BoundVar::from_usize(vars.len());
                        vars.push(ty::BoundVariableKind::Region(ty::BoundRegionKind::Anon));
                        ty::Region::new_bound(
                            self,
                            debruijn,
                            ty::BoundRegion { var, kind: ty::BoundRegionKind::Anon },
                        )
                    });
                    ty
                }),
        );
        ty::EarlyBinder::bind(ty::Binder::bind_with_vars(
            bound_tys,
            self.mk_bound_variable_kinds(&vars),
        ))
    }

    /// Expands the given impl trait type, stopping if the type is recursive.
    #[instrument(skip(self), level = "debug", ret)]
    pub fn try_expand_impl_trait_type(
        self,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> Result<Ty<'tcx>, Ty<'tcx>> {
        let mut visitor = OpaqueTypeExpander {
            seen_opaque_tys: FxHashSet::default(),
            expanded_cache: FxHashMap::default(),
            primary_def_id: Some(def_id),
            found_recursion: false,
            found_any_recursion: false,
            check_recursion: true,
            tcx: self,
        };

        let expanded_type = visitor.expand_opaque_ty(def_id, args).unwrap();
        if visitor.found_recursion { Err(expanded_type) } else { Ok(expanded_type) }
    }

    /// Query and get an English description for the item's kind.
    pub fn def_descr(self, def_id: DefId) -> &'static str {
        self.def_kind_descr(self.def_kind(def_id), def_id)
    }

    /// Get an English description for the item's kind.
    pub fn def_kind_descr(self, def_kind: DefKind, def_id: DefId) -> &'static str {
        match def_kind {
            DefKind::AssocFn if self.associated_item(def_id).is_method() => "method",
            DefKind::Closure if let Some(coroutine_kind) = self.coroutine_kind(def_id) => {
                match coroutine_kind {
                    hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Async,
                        hir::CoroutineSource::Fn,
                    ) => "async fn",
                    hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Async,
                        hir::CoroutineSource::Block,
                    ) => "async block",
                    hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Async,
                        hir::CoroutineSource::Closure,
                    ) => "async closure",
                    hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::AsyncGen,
                        hir::CoroutineSource::Fn,
                    ) => "async gen fn",
                    hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::AsyncGen,
                        hir::CoroutineSource::Block,
                    ) => "async gen block",
                    hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::AsyncGen,
                        hir::CoroutineSource::Closure,
                    ) => "async gen closure",
                    hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Gen,
                        hir::CoroutineSource::Fn,
                    ) => "gen fn",
                    hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Gen,
                        hir::CoroutineSource::Block,
                    ) => "gen block",
                    hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Gen,
                        hir::CoroutineSource::Closure,
                    ) => "gen closure",
                    hir::CoroutineKind::Coroutine(_) => "coroutine",
                }
            }
            _ => def_kind.descr(def_id),
        }
    }

    /// Gets an English article for the [`TyCtxt::def_descr`].
    pub fn def_descr_article(self, def_id: DefId) -> &'static str {
        self.def_kind_descr_article(self.def_kind(def_id), def_id)
    }

    /// Gets an English article for the [`TyCtxt::def_kind_descr`].
    pub fn def_kind_descr_article(self, def_kind: DefKind, def_id: DefId) -> &'static str {
        match def_kind {
            DefKind::AssocFn if self.associated_item(def_id).is_method() => "a",
            DefKind::Closure if let Some(coroutine_kind) = self.coroutine_kind(def_id) => {
                match coroutine_kind {
                    hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, ..) => "an",
                    hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen, ..) => "an",
                    hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen, ..) => "a",
                    hir::CoroutineKind::Coroutine(_) => "a",
                }
            }
            _ => def_kind.article(),
        }
    }

    /// Return `true` if the supplied `CrateNum` is "user-visible," meaning either a [public]
    /// dependency, or a [direct] private dependency. This is used to decide whether the crate can
    /// be shown in `impl` suggestions.
    ///
    /// [public]: TyCtxt::is_private_dep
    /// [direct]: rustc_session::cstore::ExternCrate::is_direct
    pub fn is_user_visible_dep(self, key: CrateNum) -> bool {
        // `#![rustc_private]` overrides defaults to make private dependencies usable.
        if self.features().enabled(sym::rustc_private) {
            return true;
        }

        // | Private | Direct | Visible |                    |
        // |---------|--------|---------|--------------------|
        // | Yes     | Yes    | Yes     | !true || true   |
        // | No      | Yes    | Yes     | !false || true  |
        // | Yes     | No     | No      | !true || false  |
        // | No      | No     | Yes     | !false || false |
        !self.is_private_dep(key)
            // If `extern_crate` is `None`, then the crate was injected (e.g., by the allocator).
            // Treat that kind of crate as "indirect", since it's an implementation detail of
            // the language.
            || self.extern_crate(key).is_some_and(|e| e.is_direct())
    }

    /// Expand any [free alias types][free] contained within the given `value`.
    ///
    /// This should be used over other normalization routines in situations where
    /// it's important not to normalize other alias types and where the predicates
    /// on the corresponding type alias shouldn't be taken into consideration.
    ///
    /// Whenever possible **prefer not to use this function**! Instead, use standard
    /// normalization routines or if feasible don't normalize at all.
    ///
    /// This function comes in handy if you want to mimic the behavior of eager
    /// type alias expansion in a localized manner.
    ///
    /// <div class="warning">
    /// This delays a bug on overflow! Therefore you need to be certain that the
    /// contained types get fully normalized at a later stage. Note that even on
    /// overflow all well-behaved free alias types get expanded correctly, so the
    /// result is still useful.
    /// </div>
    ///
    /// [free]: ty::Free
    pub fn expand_free_alias_tys<T: TypeFoldable<TyCtxt<'tcx>>>(self, value: T) -> T {
        value.fold_with(&mut FreeAliasTypeExpander { tcx: self, depth: 0 })
    }

    /// Peel off all [free alias types] in this type until there are none left.
    ///
    /// This only expands free alias types in “head” / outermost positions. It can
    /// be used over [expand_free_alias_tys] as an optimization in situations where
    /// one only really cares about the *kind* of the final aliased type but not
    /// the types the other constituent types alias.
    ///
    /// <div class="warning">
    /// This delays a bug on overflow! Therefore you need to be certain that the
    /// type gets fully normalized at a later stage.
    /// </div>
    ///
    /// [free]: ty::Free
    /// [expand_free_alias_tys]: Self::expand_free_alias_tys
    pub fn peel_off_free_alias_tys(self, mut ty: Ty<'tcx>) -> Ty<'tcx> {
        let ty::Alias(ty::Free, _) = ty.kind() else { return ty };

        let limit = self.recursion_limit();
        let mut depth = 0;

        while let ty::Alias(ty::Free, alias) = ty.kind() {
            if !limit.value_within_limit(depth) {
                let guar = self.dcx().delayed_bug("overflow expanding free alias type");
                return Ty::new_error(self, guar);
            }

            ty = self.type_of(alias.def_id).instantiate(self, alias.args);
            depth += 1;
        }

        ty
    }

    // Computes the variances for an alias (opaque or RPITIT) that represent
    // its (un)captured regions.
    pub fn opt_alias_variances(
        self,
        kind: impl Into<ty::AliasTermKind>,
        def_id: DefId,
    ) -> Option<&'tcx [ty::Variance]> {
        match kind.into() {
            ty::AliasTermKind::ProjectionTy => {
                if self.is_impl_trait_in_trait(def_id) {
                    Some(self.variances_of(def_id))
                } else {
                    None
                }
            }
            ty::AliasTermKind::OpaqueTy => Some(self.variances_of(def_id)),
            ty::AliasTermKind::InherentTy
            | ty::AliasTermKind::InherentConst
            | ty::AliasTermKind::FreeTy
            | ty::AliasTermKind::FreeConst
            | ty::AliasTermKind::UnevaluatedConst
            | ty::AliasTermKind::ProjectionConst => None,
        }
    }
}

struct OpaqueTypeExpander<'tcx> {
    // Contains the DefIds of the opaque types that are currently being
    // expanded. When we expand an opaque type we insert the DefId of
    // that type, and when we finish expanding that type we remove the
    // its DefId.
    seen_opaque_tys: FxHashSet<DefId>,
    // Cache of all expansions we've seen so far. This is a critical
    // optimization for some large types produced by async fn trees.
    expanded_cache: FxHashMap<(DefId, GenericArgsRef<'tcx>), Ty<'tcx>>,
    primary_def_id: Option<DefId>,
    found_recursion: bool,
    found_any_recursion: bool,
    /// Whether or not to check for recursive opaque types.
    /// This is `true` when we're explicitly checking for opaque type
    /// recursion, and 'false' otherwise to avoid unnecessary work.
    check_recursion: bool,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> OpaqueTypeExpander<'tcx> {
    fn expand_opaque_ty(&mut self, def_id: DefId, args: GenericArgsRef<'tcx>) -> Option<Ty<'tcx>> {
        if self.found_any_recursion {
            return None;
        }
        let args = args.fold_with(self);
        if !self.check_recursion || self.seen_opaque_tys.insert(def_id) {
            let expanded_ty = match self.expanded_cache.get(&(def_id, args)) {
                Some(expanded_ty) => *expanded_ty,
                None => {
                    let generic_ty = self.tcx.type_of(def_id);
                    let concrete_ty = generic_ty.instantiate(self.tcx, args);
                    let expanded_ty = self.fold_ty(concrete_ty);
                    self.expanded_cache.insert((def_id, args), expanded_ty);
                    expanded_ty
                }
            };
            if self.check_recursion {
                self.seen_opaque_tys.remove(&def_id);
            }
            Some(expanded_ty)
        } else {
            // If another opaque type that we contain is recursive, then it
            // will report the error, so we don't have to.
            self.found_any_recursion = true;
            self.found_recursion = def_id == *self.primary_def_id.as_ref().unwrap();
            None
        }
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for OpaqueTypeExpander<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Alias(ty::Opaque, ty::AliasTy { def_id, args, .. }) = *t.kind() {
            self.expand_opaque_ty(def_id, args).unwrap_or(t)
        } else if t.has_opaque_types() {
            t.super_fold_with(self)
        } else {
            t
        }
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if let ty::PredicateKind::Clause(clause) = p.kind().skip_binder()
            && let ty::ClauseKind::Projection(projection_pred) = clause
        {
            p.kind()
                .rebind(ty::ProjectionPredicate {
                    projection_term: projection_pred.projection_term.fold_with(self),
                    // Don't fold the term on the RHS of the projection predicate.
                    // This is because for default trait methods with RPITITs, we
                    // install a `NormalizesTo(Projection(RPITIT) -> Opaque(RPITIT))`
                    // predicate, which would trivially cause a cycle when we do
                    // anything that requires `TypingEnv::with_post_analysis_normalized`.
                    term: projection_pred.term,
                })
                .upcast(self.tcx)
        } else {
            p.super_fold_with(self)
        }
    }
}

struct FreeAliasTypeExpander<'tcx> {
    tcx: TyCtxt<'tcx>,
    depth: usize,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for FreeAliasTypeExpander<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !ty.has_type_flags(ty::TypeFlags::HAS_TY_FREE_ALIAS) {
            return ty;
        }
        let ty::Alias(ty::Free, alias) = ty.kind() else {
            return ty.super_fold_with(self);
        };
        if !self.tcx.recursion_limit().value_within_limit(self.depth) {
            let guar = self.tcx.dcx().delayed_bug("overflow expanding free alias type");
            return Ty::new_error(self.tcx, guar);
        }

        self.depth += 1;
        ensure_sufficient_stack(|| {
            self.tcx.type_of(alias.def_id).instantiate(self.tcx, alias.args).fold_with(self)
        })
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if !ct.has_type_flags(ty::TypeFlags::HAS_TY_FREE_ALIAS) {
            return ct;
        }
        ct.super_fold_with(self)
    }
}

impl<'tcx> Ty<'tcx> {
    /// Returns the `Size` for primitive types (bool, uint, int, char, float).
    pub fn primitive_size(self, tcx: TyCtxt<'tcx>) -> Size {
        match *self.kind() {
            ty::Bool => Size::from_bytes(1),
            ty::Char => Size::from_bytes(4),
            ty::Int(ity) => Integer::from_int_ty(&tcx, ity).size(),
            ty::Uint(uty) => Integer::from_uint_ty(&tcx, uty).size(),
            ty::Float(fty) => Float::from_float_ty(fty).size(),
            _ => bug!("non primitive type"),
        }
    }

    pub fn int_size_and_signed(self, tcx: TyCtxt<'tcx>) -> (Size, bool) {
        match *self.kind() {
            ty::Int(ity) => (Integer::from_int_ty(&tcx, ity).size(), true),
            ty::Uint(uty) => (Integer::from_uint_ty(&tcx, uty).size(), false),
            _ => bug!("non integer discriminant"),
        }
    }

    /// Returns the minimum and maximum values for the given numeric type (including `char`s) or
    /// returns `None` if the type is not numeric.
    pub fn numeric_min_and_max_as_bits(self, tcx: TyCtxt<'tcx>) -> Option<(u128, u128)> {
        use rustc_apfloat::ieee::{Double, Half, Quad, Single};
        Some(match self.kind() {
            ty::Int(_) | ty::Uint(_) => {
                let (size, signed) = self.int_size_and_signed(tcx);
                let min = if signed { size.truncate(size.signed_int_min() as u128) } else { 0 };
                let max =
                    if signed { size.signed_int_max() as u128 } else { size.unsigned_int_max() };
                (min, max)
            }
            ty::Char => (0, std::char::MAX as u128),
            ty::Float(ty::FloatTy::F16) => ((-Half::INFINITY).to_bits(), Half::INFINITY.to_bits()),
            ty::Float(ty::FloatTy::F32) => {
                ((-Single::INFINITY).to_bits(), Single::INFINITY.to_bits())
            }
            ty::Float(ty::FloatTy::F64) => {
                ((-Double::INFINITY).to_bits(), Double::INFINITY.to_bits())
            }
            ty::Float(ty::FloatTy::F128) => ((-Quad::INFINITY).to_bits(), Quad::INFINITY.to_bits()),
            _ => return None,
        })
    }

    /// Returns the maximum value for the given numeric type (including `char`s)
    /// or returns `None` if the type is not numeric.
    pub fn numeric_max_val(self, tcx: TyCtxt<'tcx>) -> Option<mir::Const<'tcx>> {
        let typing_env = TypingEnv::fully_monomorphized();
        self.numeric_min_and_max_as_bits(tcx)
            .map(|(_, max)| mir::Const::from_bits(tcx, max, typing_env, self))
    }

    /// Returns the minimum value for the given numeric type (including `char`s)
    /// or returns `None` if the type is not numeric.
    pub fn numeric_min_val(self, tcx: TyCtxt<'tcx>) -> Option<mir::Const<'tcx>> {
        let typing_env = TypingEnv::fully_monomorphized();
        self.numeric_min_and_max_as_bits(tcx)
            .map(|(min, _)| mir::Const::from_bits(tcx, min, typing_env, self))
    }

    /// Checks whether values of this type `T` have a size known at
    /// compile time (i.e., whether `T: Sized`). Lifetimes are ignored
    /// for the purposes of this check, so it can be an
    /// over-approximation in generic contexts, where one can have
    /// strange rules like `<T as Foo<'static>>::Bar: Sized` that
    /// actually carry lifetime requirements.
    pub fn is_sized(self, tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> bool {
        self.is_trivially_sized(tcx) || tcx.is_sized_raw(typing_env.as_query_input(self))
    }

    /// Checks whether values of this type `T` implement the `Freeze`
    /// trait -- frozen types are those that do not contain an
    /// `UnsafeCell` anywhere. This is a language concept used to
    /// distinguish "true immutability", which is relevant to
    /// optimization as well as the rules around static values. Note
    /// that the `Freeze` trait is not exposed to end users and is
    /// effectively an implementation detail.
    pub fn is_freeze(self, tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> bool {
        self.is_trivially_freeze() || tcx.is_freeze_raw(typing_env.as_query_input(self))
    }

    /// Fast path helper for testing if a type is `Freeze`.
    ///
    /// Returning true means the type is known to be `Freeze`. Returning
    /// `false` means nothing -- could be `Freeze`, might not be.
    pub fn is_trivially_freeze(self) -> bool {
        match self.kind() {
            ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Bool
            | ty::Char
            | ty::Str
            | ty::Never
            | ty::Ref(..)
            | ty::RawPtr(_, _)
            | ty::FnDef(..)
            | ty::Error(_)
            | ty::FnPtr(..) => true,
            ty::Tuple(fields) => fields.iter().all(Self::is_trivially_freeze),
            ty::Pat(ty, _) | ty::Slice(ty) | ty::Array(ty, _) => ty.is_trivially_freeze(),
            ty::Adt(..)
            | ty::Bound(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Dynamic(..)
            | ty::Foreign(_)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::UnsafeBinder(_)
            | ty::Infer(_)
            | ty::Alias(..)
            | ty::Param(_)
            | ty::Placeholder(_) => false,
        }
    }

    /// Checks whether values of this type `T` implement the `Unpin` trait.
    pub fn is_unpin(self, tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> bool {
        self.is_trivially_unpin() || tcx.is_unpin_raw(typing_env.as_query_input(self))
    }

    /// Fast path helper for testing if a type is `Unpin`.
    ///
    /// Returning true means the type is known to be `Unpin`. Returning
    /// `false` means nothing -- could be `Unpin`, might not be.
    fn is_trivially_unpin(self) -> bool {
        match self.kind() {
            ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Bool
            | ty::Char
            | ty::Str
            | ty::Never
            | ty::Ref(..)
            | ty::RawPtr(_, _)
            | ty::FnDef(..)
            | ty::Error(_)
            | ty::FnPtr(..) => true,
            ty::Tuple(fields) => fields.iter().all(Self::is_trivially_unpin),
            ty::Pat(ty, _) | ty::Slice(ty) | ty::Array(ty, _) => ty.is_trivially_unpin(),
            ty::Adt(..)
            | ty::Bound(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Dynamic(..)
            | ty::Foreign(_)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::UnsafeBinder(_)
            | ty::Infer(_)
            | ty::Alias(..)
            | ty::Param(_)
            | ty::Placeholder(_) => false,
        }
    }

    /// Checks whether this type is an ADT that has unsafe fields.
    pub fn has_unsafe_fields(self) -> bool {
        if let ty::Adt(adt_def, ..) = self.kind() {
            adt_def.all_fields().any(|x| x.safety.is_unsafe())
        } else {
            false
        }
    }

    /// Checks whether values of this type `T` implement the `AsyncDrop` trait.
    pub fn is_async_drop(self, tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> bool {
        !self.is_trivially_not_async_drop()
            && tcx.is_async_drop_raw(typing_env.as_query_input(self))
    }

    /// Fast path helper for testing if a type is `AsyncDrop`.
    ///
    /// Returning true means the type is known to be `!AsyncDrop`. Returning
    /// `false` means nothing -- could be `AsyncDrop`, might not be.
    fn is_trivially_not_async_drop(self) -> bool {
        match self.kind() {
            ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Bool
            | ty::Char
            | ty::Str
            | ty::Never
            | ty::Ref(..)
            | ty::RawPtr(..)
            | ty::FnDef(..)
            | ty::Error(_)
            | ty::FnPtr(..) => true,
            // FIXME(unsafe_binders):
            ty::UnsafeBinder(_) => todo!(),
            ty::Tuple(fields) => fields.iter().all(Self::is_trivially_not_async_drop),
            ty::Pat(elem_ty, _) | ty::Slice(elem_ty) | ty::Array(elem_ty, _) => {
                elem_ty.is_trivially_not_async_drop()
            }
            ty::Adt(..)
            | ty::Bound(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Dynamic(..)
            | ty::Foreign(_)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Infer(_)
            | ty::Alias(..)
            | ty::Param(_)
            | ty::Placeholder(_) => false,
        }
    }

    /// If `ty.needs_drop(...)` returns `true`, then `ty` is definitely
    /// non-copy and *might* have a destructor attached; if it returns
    /// `false`, then `ty` definitely has no destructor (i.e., no drop glue).
    ///
    /// (Note that this implies that if `ty` has a destructor attached,
    /// then `needs_drop` will definitely return `true` for `ty`.)
    ///
    /// Note that this method is used to check eligible types in unions.
    #[inline]
    pub fn needs_drop(self, tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> bool {
        // Avoid querying in simple cases.
        match needs_drop_components(tcx, self) {
            Err(AlwaysRequiresDrop) => true,
            Ok(components) => {
                let query_ty = match *components {
                    [] => return false,
                    // If we've got a single component, call the query with that
                    // to increase the chance that we hit the query cache.
                    [component_ty] => component_ty,
                    _ => self,
                };

                // This doesn't depend on regions, so try to minimize distinct
                // query keys used. If normalization fails, we just use `query_ty`.
                debug_assert!(!typing_env.param_env.has_infer());
                let query_ty = tcx
                    .try_normalize_erasing_regions(typing_env, query_ty)
                    .unwrap_or_else(|_| tcx.erase_regions(query_ty));

                tcx.needs_drop_raw(typing_env.as_query_input(query_ty))
            }
        }
    }

    /// If `ty.needs_async_drop(...)` returns `true`, then `ty` is definitely
    /// non-copy and *might* have a async destructor attached; if it returns
    /// `false`, then `ty` definitely has no async destructor (i.e., no async
    /// drop glue).
    ///
    /// (Note that this implies that if `ty` has an async destructor attached,
    /// then `needs_async_drop` will definitely return `true` for `ty`.)
    ///
    // FIXME(zetanumbers): Note that this method is used to check eligible types
    // in unions.
    #[inline]
    pub fn needs_async_drop(self, tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> bool {
        // Avoid querying in simple cases.
        match needs_drop_components(tcx, self) {
            Err(AlwaysRequiresDrop) => true,
            Ok(components) => {
                let query_ty = match *components {
                    [] => return false,
                    // If we've got a single component, call the query with that
                    // to increase the chance that we hit the query cache.
                    [component_ty] => component_ty,
                    _ => self,
                };

                // This doesn't depend on regions, so try to minimize distinct
                // query keys used.
                // If normalization fails, we just use `query_ty`.
                debug_assert!(!typing_env.has_infer());
                let query_ty = tcx
                    .try_normalize_erasing_regions(typing_env, query_ty)
                    .unwrap_or_else(|_| tcx.erase_regions(query_ty));

                tcx.needs_async_drop_raw(typing_env.as_query_input(query_ty))
            }
        }
    }

    /// Checks if `ty` has a significant drop.
    ///
    /// Note that this method can return false even if `ty` has a destructor
    /// attached; even if that is the case then the adt has been marked with
    /// the attribute `rustc_insignificant_dtor`.
    ///
    /// Note that this method is used to check for change in drop order for
    /// 2229 drop reorder migration analysis.
    #[inline]
    pub fn has_significant_drop(self, tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> bool {
        // Avoid querying in simple cases.
        match needs_drop_components(tcx, self) {
            Err(AlwaysRequiresDrop) => true,
            Ok(components) => {
                let query_ty = match *components {
                    [] => return false,
                    // If we've got a single component, call the query with that
                    // to increase the chance that we hit the query cache.
                    [component_ty] => component_ty,
                    _ => self,
                };

                // FIXME(#86868): We should be canonicalizing, or else moving this to a method of inference
                // context, or *something* like that, but for now just avoid passing inference
                // variables to queries that can't cope with them. Instead, conservatively
                // return "true" (may change drop order).
                if query_ty.has_infer() {
                    return true;
                }

                // This doesn't depend on regions, so try to minimize distinct
                // query keys used.
                let erased = tcx.normalize_erasing_regions(typing_env, query_ty);
                tcx.has_significant_drop_raw(typing_env.as_query_input(erased))
            }
        }
    }

    /// Returns `true` if equality for this type is both reflexive and structural.
    ///
    /// Reflexive equality for a type is indicated by an `Eq` impl for that type.
    ///
    /// Primitive types (`u32`, `str`) have structural equality by definition. For composite data
    /// types, equality for the type as a whole is structural when it is the same as equality
    /// between all components (fields, array elements, etc.) of that type. For ADTs, structural
    /// equality is indicated by an implementation of `StructuralPartialEq` for that type.
    ///
    /// This function is "shallow" because it may return `true` for a composite type whose fields
    /// are not `StructuralPartialEq`. For example, `[T; 4]` has structural equality regardless of `T`
    /// because equality for arrays is determined by the equality of each array element. If you
    /// want to know whether a given call to `PartialEq::eq` will proceed structurally all the way
    /// down, you will need to use a type visitor.
    #[inline]
    pub fn is_structural_eq_shallow(self, tcx: TyCtxt<'tcx>) -> bool {
        match self.kind() {
            // Look for an impl of `StructuralPartialEq`.
            ty::Adt(..) => tcx.has_structural_eq_impl(self),

            // Primitive types that satisfy `Eq`.
            ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Str | ty::Never => true,

            // Composite types that satisfy `Eq` when all of their fields do.
            //
            // Because this function is "shallow", we return `true` for these composites regardless
            // of the type(s) contained within.
            ty::Pat(..) | ty::Ref(..) | ty::Array(..) | ty::Slice(_) | ty::Tuple(..) => true,

            // Raw pointers use bitwise comparison.
            ty::RawPtr(_, _) | ty::FnPtr(..) => true,

            // Floating point numbers are not `Eq`.
            ty::Float(_) => false,

            // Conservatively return `false` for all others...

            // Anonymous function types
            ty::FnDef(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Dynamic(..)
            | ty::Coroutine(..) => false,

            // Generic or inferred types
            //
            // FIXME(ecstaticmorse): Maybe we should `bug` here? This should probably only be
            // called for known, fully-monomorphized types.
            ty::Alias(..) | ty::Param(_) | ty::Bound(..) | ty::Placeholder(_) | ty::Infer(_) => {
                false
            }

            ty::Foreign(_) | ty::CoroutineWitness(..) | ty::Error(_) | ty::UnsafeBinder(_) => false,
        }
    }

    /// Peel off all reference types in this type until there are none left.
    ///
    /// This method is idempotent, i.e. `ty.peel_refs().peel_refs() == ty.peel_refs()`.
    ///
    /// # Examples
    ///
    /// - `u8` -> `u8`
    /// - `&'a mut u8` -> `u8`
    /// - `&'a &'b u8` -> `u8`
    /// - `&'a *const &'b u8 -> *const &'b u8`
    pub fn peel_refs(self) -> Ty<'tcx> {
        let mut ty = self;
        while let ty::Ref(_, inner_ty, _) = ty.kind() {
            ty = *inner_ty;
        }
        ty
    }

    // FIXME(compiler-errors): Think about removing this.
    #[inline]
    pub fn outer_exclusive_binder(self) -> ty::DebruijnIndex {
        self.0.outer_exclusive_binder
    }
}

/// Returns a list of types such that the given type needs drop if and only if
/// *any* of the returned types need drop. Returns `Err(AlwaysRequiresDrop)` if
/// this type always needs drop.
//
// FIXME(zetanumbers): consider replacing this with only
// `needs_drop_components_with_async`
#[inline]
pub fn needs_drop_components<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Result<SmallVec<[Ty<'tcx>; 2]>, AlwaysRequiresDrop> {
    needs_drop_components_with_async(tcx, ty, Asyncness::No)
}

/// Returns a list of types such that the given type needs drop if and only if
/// *any* of the returned types need drop. Returns `Err(AlwaysRequiresDrop)` if
/// this type always needs drop.
pub fn needs_drop_components_with_async<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    asyncness: Asyncness,
) -> Result<SmallVec<[Ty<'tcx>; 2]>, AlwaysRequiresDrop> {
    match *ty.kind() {
        ty::Infer(ty::FreshIntTy(_))
        | ty::Infer(ty::FreshFloatTy(_))
        | ty::Bool
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Never
        | ty::FnDef(..)
        | ty::FnPtr(..)
        | ty::Char
        | ty::RawPtr(_, _)
        | ty::Ref(..)
        | ty::Str => Ok(SmallVec::new()),

        // Foreign types can never have destructors.
        ty::Foreign(..) => Ok(SmallVec::new()),

        // FIXME(zetanumbers): Temporary workaround for async drop of dynamic types
        ty::Dynamic(..) | ty::Error(_) => {
            if asyncness.is_async() {
                Ok(SmallVec::new())
            } else {
                Err(AlwaysRequiresDrop)
            }
        }

        ty::Pat(ty, _) | ty::Slice(ty) => needs_drop_components_with_async(tcx, ty, asyncness),
        ty::Array(elem_ty, size) => {
            match needs_drop_components_with_async(tcx, elem_ty, asyncness) {
                Ok(v) if v.is_empty() => Ok(v),
                res => match size.try_to_target_usize(tcx) {
                    // Arrays of size zero don't need drop, even if their element
                    // type does.
                    Some(0) => Ok(SmallVec::new()),
                    Some(_) => res,
                    // We don't know which of the cases above we are in, so
                    // return the whole type and let the caller decide what to
                    // do.
                    None => Ok(smallvec![ty]),
                },
            }
        }
        // If any field needs drop, then the whole tuple does.
        ty::Tuple(fields) => fields.iter().try_fold(SmallVec::new(), move |mut acc, elem| {
            acc.extend(needs_drop_components_with_async(tcx, elem, asyncness)?);
            Ok(acc)
        }),

        // These require checking for `Copy` bounds or `Adt` destructors.
        ty::Adt(..)
        | ty::Alias(..)
        | ty::Param(_)
        | ty::Bound(..)
        | ty::Placeholder(..)
        | ty::Infer(_)
        | ty::Closure(..)
        | ty::CoroutineClosure(..)
        | ty::Coroutine(..)
        | ty::CoroutineWitness(..)
        | ty::UnsafeBinder(_) => Ok(smallvec![ty]),
    }
}

/// Does the equivalent of
/// ```ignore (illustrative)
/// let v = self.iter().map(|p| p.fold_with(folder)).collect::<SmallVec<[_; 8]>>();
/// folder.tcx().intern_*(&v)
/// ```
pub fn fold_list<'tcx, F, L, T>(
    list: L,
    folder: &mut F,
    intern: impl FnOnce(TyCtxt<'tcx>, &[T]) -> L,
) -> L
where
    F: TypeFolder<TyCtxt<'tcx>>,
    L: AsRef<[T]>,
    T: TypeFoldable<TyCtxt<'tcx>> + PartialEq + Copy,
{
    let slice = list.as_ref();
    let mut iter = slice.iter().copied();
    // Look for the first element that changed
    match iter.by_ref().enumerate().find_map(|(i, t)| {
        let new_t = t.fold_with(folder);
        if new_t != t { Some((i, new_t)) } else { None }
    }) {
        Some((i, new_t)) => {
            // An element changed, prepare to intern the resulting list
            let mut new_list = SmallVec::<[_; 8]>::with_capacity(slice.len());
            new_list.extend_from_slice(&slice[..i]);
            new_list.push(new_t);
            for t in iter {
                new_list.push(t.fold_with(folder))
            }
            intern(folder.cx(), &new_list)
        }
        None => list,
    }
}

/// Does the equivalent of
/// ```ignore (illustrative)
/// let v = self.iter().map(|p| p.try_fold_with(folder)).collect::<SmallVec<[_; 8]>>();
/// folder.tcx().intern_*(&v)
/// ```
pub fn try_fold_list<'tcx, F, L, T>(
    list: L,
    folder: &mut F,
    intern: impl FnOnce(TyCtxt<'tcx>, &[T]) -> L,
) -> Result<L, F::Error>
where
    F: FallibleTypeFolder<TyCtxt<'tcx>>,
    L: AsRef<[T]>,
    T: TypeFoldable<TyCtxt<'tcx>> + PartialEq + Copy,
{
    let slice = list.as_ref();
    let mut iter = slice.iter().copied();
    // Look for the first element that changed
    match iter.by_ref().enumerate().find_map(|(i, t)| match t.try_fold_with(folder) {
        Ok(new_t) if new_t == t => None,
        new_t => Some((i, new_t)),
    }) {
        Some((i, Ok(new_t))) => {
            // An element changed, prepare to intern the resulting list
            let mut new_list = SmallVec::<[_; 8]>::with_capacity(slice.len());
            new_list.extend_from_slice(&slice[..i]);
            new_list.push(new_t);
            for t in iter {
                new_list.push(t.try_fold_with(folder)?)
            }
            Ok(intern(folder.cx(), &new_list))
        }
        Some((_, Err(err))) => {
            return Err(err);
        }
        None => Ok(list),
    }
}

#[derive(Copy, Clone, Debug, HashStable, TyEncodable, TyDecodable)]
pub struct AlwaysRequiresDrop;

/// Reveals all opaque types in the given value, replacing them
/// with their underlying types.
pub fn reveal_opaque_types_in_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    val: ty::Clauses<'tcx>,
) -> ty::Clauses<'tcx> {
    assert!(!tcx.next_trait_solver_globally());
    let mut visitor = OpaqueTypeExpander {
        seen_opaque_tys: FxHashSet::default(),
        expanded_cache: FxHashMap::default(),
        primary_def_id: None,
        found_recursion: false,
        found_any_recursion: false,
        check_recursion: false,
        tcx,
    };
    val.fold_with(&mut visitor)
}

/// Determines whether an item is directly annotated with `doc(hidden)`.
fn is_doc_hidden(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    tcx.get_attrs(def_id, sym::doc)
        .filter_map(|attr| attr.meta_item_list())
        .any(|items| items.iter().any(|item| item.has_name(sym::hidden)))
}

/// Determines whether an item is annotated with `doc(notable_trait)`.
pub fn is_doc_notable_trait(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.get_attrs(def_id, sym::doc)
        .filter_map(|attr| attr.meta_item_list())
        .any(|items| items.iter().any(|item| item.has_name(sym::notable_trait)))
}

/// Determines whether an item is an intrinsic (which may be via Abi or via the `rustc_intrinsic` attribute).
///
/// We double check the feature gate here because whether a function may be defined as an intrinsic causes
/// the compiler to make some assumptions about its shape; if the user doesn't use a feature gate, they may
/// cause an ICE that we otherwise may want to prevent.
pub fn intrinsic_raw(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<ty::IntrinsicDef> {
    if tcx.features().intrinsics() && tcx.has_attr(def_id, sym::rustc_intrinsic) {
        let must_be_overridden = match tcx.hir_node_by_def_id(def_id) {
            hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn { has_body, .. }, .. }) => {
                !has_body
            }
            _ => true,
        };
        Some(ty::IntrinsicDef {
            name: tcx.item_name(def_id.into()),
            must_be_overridden,
            const_stable: tcx.has_attr(def_id, sym::rustc_intrinsic_const_stable_indirect),
        })
    } else {
        None
    }
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        reveal_opaque_types_in_bounds,
        is_doc_hidden,
        is_doc_notable_trait,
        intrinsic_raw,
        ..*providers
    }
}
