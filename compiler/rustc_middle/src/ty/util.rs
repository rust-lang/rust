//! Miscellaneous type-system utilities that are too small to deserve their own modules.

use crate::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use crate::mir;
use crate::ty::fast_reject::TreatProjections;
use crate::ty::layout::IntegerExt;
use crate::ty::{
    self, FallibleTypeFolder, ToPredicate, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeVisitableExt,
};
use crate::ty::{GenericArgKind, SubstsRef};
use rustc_apfloat::Float as _;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::GrowableBitSet;
use rustc_index::vec::{Idx, IndexVec};
use rustc_macros::HashStable;
use rustc_span::{sym, DUMMY_SP};
use rustc_target::abi::{Integer, IntegerType, Size, TargetDataLayout};
use rustc_target::spec::abi::Abi;
use smallvec::SmallVec;
use std::{fmt, iter};

#[derive(Copy, Clone, Debug)]
pub struct Discr<'tcx> {
    /// Bit representation of the discriminant (e.g., `-128i8` is `0xFF_u128`).
    pub val: u128,
    pub ty: Ty<'tcx>,
}

/// Used as an input to [`TyCtxt::uses_unique_generic_params`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IgnoreRegions {
    Yes,
    No,
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
                write!(fmt, "{}", x)
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
            let val = size.sign_extend(self.val) as i128;
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

pub trait IntTypeExt {
    fn to_ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx>;
    fn disr_incr<'tcx>(&self, tcx: TyCtxt<'tcx>, val: Option<Discr<'tcx>>) -> Option<Discr<'tcx>>;
    fn initial_discriminant<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Discr<'tcx>;
}

impl IntTypeExt for IntegerType {
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
    pub fn type_id_hash(self, ty: Ty<'tcx>) -> u64 {
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

    pub fn has_error_field(self, ty: Ty<'tcx>) -> bool {
        if let ty::Adt(def, substs) = *ty.kind() {
            for field in def.all_fields() {
                let field_ty = field.ty(self, substs);
                if let ty::Error(_) = field_ty.kind() {
                    return true;
                }
            }
        }
        false
    }

    /// Attempts to returns the deeply last field of nested structures, but
    /// does not apply any normalization in its search. Returns the same type
    /// if input `ty` is not a structure at all.
    pub fn struct_tail_without_normalization(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let tcx = self;
        tcx.struct_tail_with_normalize(ty, |ty| ty, || {})
    }

    /// Returns the deeply last field of nested structures, or the same type if
    /// not a structure at all. Corresponds to the only possible unsized field,
    /// and its type can be used to determine unsizing strategy.
    ///
    /// Should only be called if `ty` has no inference variables and does not
    /// need its lifetimes preserved (e.g. as part of codegen); otherwise
    /// normalization attempt may cause compiler bugs.
    pub fn struct_tail_erasing_lifetimes(
        self,
        ty: Ty<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self;
        tcx.struct_tail_with_normalize(ty, |ty| tcx.normalize_erasing_regions(param_env, ty), || {})
    }

    /// Returns the deeply last field of nested structures, or the same type if
    /// not a structure at all. Corresponds to the only possible unsized field,
    /// and its type can be used to determine unsizing strategy.
    ///
    /// This is parameterized over the normalization strategy (i.e. how to
    /// handle `<T as Trait>::Assoc` and `impl Trait`); pass the identity
    /// function to indicate no normalization should take place.
    ///
    /// See also `struct_tail_erasing_lifetimes`, which is suitable for use
    /// during codegen.
    pub fn struct_tail_with_normalize(
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
                return self.ty_error_with_message(
                    DUMMY_SP,
                    &format!("reached the recursion limit finding the struct tail for {}", ty),
                );
            }
            match *ty.kind() {
                ty::Adt(def, substs) => {
                    if !def.is_struct() {
                        break;
                    }
                    match def.non_enum_variant().fields.last() {
                        Some(field) => {
                            f();
                            ty = field.ty(self, substs);
                        }
                        None => break,
                    }
                }

                ty::Tuple(tys) if let Some((&last_ty, _)) = tys.split_last() => {
                    f();
                    ty = last_ty;
                }

                ty::Tuple(_) => break,

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
    /// For `(Foo<Foo<T>>, Foo<dyn Trait>)`, the result will be `(Foo<T>, Trait)`,
    /// whereas struct_tail produces `T`, and `Trait`, respectively.
    ///
    /// Should only be called if the types have no inference variables and do
    /// not need their lifetimes preserved (e.g., as part of codegen); otherwise,
    /// normalization attempt may cause compiler bugs.
    pub fn struct_lockstep_tails_erasing_lifetimes(
        self,
        source: Ty<'tcx>,
        target: Ty<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> (Ty<'tcx>, Ty<'tcx>) {
        let tcx = self;
        tcx.struct_lockstep_tails_with_normalize(source, target, |ty| {
            tcx.normalize_erasing_regions(param_env, ty)
        })
    }

    /// Same as applying `struct_tail` on `source` and `target`, but only
    /// keeps going as long as the two types are instances of the same
    /// structure definitions.
    /// For `(Foo<Foo<T>>, Foo<dyn Trait>)`, the result will be `(Foo<T>, Trait)`,
    /// whereas struct_tail produces `T`, and `Trait`, respectively.
    ///
    /// See also `struct_lockstep_tails_erasing_lifetimes`, which is suitable for use
    /// during codegen.
    pub fn struct_lockstep_tails_with_normalize(
        self,
        source: Ty<'tcx>,
        target: Ty<'tcx>,
        normalize: impl Fn(Ty<'tcx>) -> Ty<'tcx>,
    ) -> (Ty<'tcx>, Ty<'tcx>) {
        let (mut a, mut b) = (source, target);
        loop {
            match (&a.kind(), &b.kind()) {
                (&ty::Adt(a_def, a_substs), &ty::Adt(b_def, b_substs))
                    if a_def == b_def && a_def.is_struct() =>
                {
                    if let Some(f) = a_def.non_enum_variant().fields.last() {
                        a = f.ty(self, a_substs);
                        b = f.ty(self, b_substs);
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
        adt_did: DefId,
        validate: impl Fn(Self, DefId) -> Result<(), ErrorGuaranteed>,
    ) -> Option<ty::Destructor> {
        let drop_trait = self.lang_items().drop_trait()?;
        self.ensure().coherent_trait(drop_trait);

        let ty = self.type_of(adt_did).subst_identity();
        let (did, constness) = self.find_map_relevant_impl(
            drop_trait,
            ty,
            // FIXME: This could also be some other mode, like "unexpected"
            TreatProjections::ForLookup,
            |impl_did| {
                if let Some(item_id) = self.associated_item_def_ids(impl_did).first() {
                    if validate(self, impl_did).is_ok() {
                        return Some((*item_id, self.constness(impl_did)));
                    }
                }
                None
            },
        )?;

        Some(ty::Destructor { did, constness })
    }

    /// Returns the set of types that are required to be alive in
    /// order to run the destructor of `def` (see RFCs 769 and
    /// 1238).
    ///
    /// Note that this returns only the constraints for the
    /// destructor of `def` itself. For the destructors of the
    /// contents, you need `adt_dtorck_constraint`.
    pub fn destructor_constraints(self, def: ty::AdtDef<'tcx>) -> Vec<ty::subst::GenericArg<'tcx>> {
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

        let impl_substs = match *self.type_of(impl_def_id).subst_identity().kind() {
            ty::Adt(def_, substs) if def_ == def => substs,
            _ => bug!(),
        };

        let item_substs = match *self.type_of(def.did()).subst_identity().kind() {
            ty::Adt(def_, substs) if def_ == def => substs,
            _ => bug!(),
        };

        let result = iter::zip(item_substs, impl_substs)
            .filter(|&(_, k)| {
                match k.unpack() {
                    GenericArgKind::Lifetime(region) => match region.kind() {
                        ty::ReEarlyBound(ref ebr) => {
                            !impl_generics.region_param(ebr, self).pure_wrt_drop
                        }
                        // Error: not a region param
                        _ => false,
                    },
                    GenericArgKind::Type(ty) => match ty.kind() {
                        ty::Param(ref pt) => !impl_generics.type_param(pt, self).pure_wrt_drop,
                        // Error: not a type param
                        _ => false,
                    },
                    GenericArgKind::Const(ct) => match ct.kind() {
                        ty::ConstKind::Param(ref pc) => {
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
        substs: SubstsRef<'tcx>,
        ignore_regions: IgnoreRegions,
    ) -> Result<(), NotUniqueParam<'tcx>> {
        let mut seen = GrowableBitSet::default();
        for arg in substs {
            match arg.unpack() {
                GenericArgKind::Lifetime(lt) => {
                    if ignore_regions == IgnoreRegions::No {
                        let ty::ReEarlyBound(p) = lt.kind() else {
                            return Err(NotUniqueParam::NotParam(lt.into()))
                        };
                        if !seen.insert(p.index) {
                            return Err(NotUniqueParam::DuplicateParam(lt.into()));
                        }
                    }
                }
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

    /// Returns `true` if `def_id` refers to a closure (e.g., `|x| x * 2`). Note
    /// that closures have a `DefId`, but the closure *expression* also
    /// has a `HirId` that is located within the context where the
    /// closure appears (and, sadly, a corresponding `NodeId`, since
    /// those are not yet phased out). The parent of the closure's
    /// `DefId` will also be the context where it appears.
    pub fn is_closure(self, def_id: DefId) -> bool {
        matches!(self.def_kind(def_id), DefKind::Closure | DefKind::Generator)
    }

    /// Returns `true` if `def_id` refers to a definition that does not have its own
    /// type-checking context, i.e. closure, generator or inline const.
    pub fn is_typeck_child(self, def_id: DefId) -> bool {
        matches!(
            self.def_kind(def_id),
            DefKind::Closure | DefKind::Generator | DefKind::InlineConst
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
    pub fn closure_env_ty(
        self,
        closure_def_id: DefId,
        closure_substs: SubstsRef<'tcx>,
        env_region: ty::Region<'tcx>,
    ) -> Option<Ty<'tcx>> {
        let closure_ty = self.mk_closure(closure_def_id, closure_substs);
        let closure_kind_ty = closure_substs.as_closure().kind_ty();
        let closure_kind = closure_kind_ty.to_opt_closure_kind()?;
        let env_ty = match closure_kind {
            ty::ClosureKind::Fn => self.mk_imm_ref(env_region, closure_ty),
            ty::ClosureKind::FnMut => self.mk_mut_ref(env_region, closure_ty),
            ty::ClosureKind::FnOnce => closure_ty,
        };
        Some(env_ty)
    }

    /// Returns `true` if the node pointed to by `def_id` is a `static` item.
    #[inline]
    pub fn is_static(self, def_id: DefId) -> bool {
        matches!(self.def_kind(def_id), DefKind::Static(_))
    }

    #[inline]
    pub fn static_mutability(self, def_id: DefId) -> Option<hir::Mutability> {
        if let DefKind::Static(mt) = self.def_kind(def_id) { Some(mt) } else { None }
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

    /// Get the type of the pointer to the static that we use in MIR.
    pub fn static_ptr_ty(self, def_id: DefId) -> Ty<'tcx> {
        // Make sure that any constants in the static's type are evaluated.
        let static_ty = self.normalize_erasing_regions(
            ty::ParamEnv::empty(),
            self.type_of(def_id).subst_identity(),
        );

        // Make sure that accesses to unsafe statics end up using raw pointers.
        // For thread-locals, this needs to be kept in sync with `Rvalue::ty`.
        if self.is_mutable_static(def_id) {
            self.mk_mut_ptr(static_ty)
        } else if self.is_foreign_item(def_id) {
            self.mk_imm_ptr(static_ty)
        } else {
            self.mk_imm_ref(self.lifetimes.re_erased, static_ty)
        }
    }

    /// Return the set of types that should be taken into accound when checking
    /// trait bounds on a generator's internal state.
    pub fn generator_hidden_types(
        self,
        def_id: DefId,
    ) -> impl Iterator<Item = ty::EarlyBinder<Ty<'tcx>>> {
        let generator_layout = &self.mir_generator_witnesses(def_id);
        generator_layout
            .field_tys
            .iter()
            .filter(|decl| !decl.ignore_for_traits)
            .map(|decl| ty::EarlyBinder(decl.ty))
    }

    /// Normalizes all opaque types in the given value, replacing them
    /// with their underlying types.
    pub fn expand_opaque_types(self, val: Ty<'tcx>) -> Ty<'tcx> {
        let mut visitor = OpaqueTypeExpander {
            seen_opaque_tys: FxHashSet::default(),
            expanded_cache: FxHashMap::default(),
            primary_def_id: None,
            found_recursion: false,
            found_any_recursion: false,
            check_recursion: false,
            expand_generators: false,
            tcx: self,
        };
        val.fold_with(&mut visitor)
    }

    /// Expands the given impl trait type, stopping if the type is recursive.
    #[instrument(skip(self), level = "debug", ret)]
    pub fn try_expand_impl_trait_type(
        self,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Result<Ty<'tcx>, Ty<'tcx>> {
        let mut visitor = OpaqueTypeExpander {
            seen_opaque_tys: FxHashSet::default(),
            expanded_cache: FxHashMap::default(),
            primary_def_id: Some(def_id),
            found_recursion: false,
            found_any_recursion: false,
            check_recursion: true,
            expand_generators: true,
            tcx: self,
        };

        let expanded_type = visitor.expand_opaque_ty(def_id, substs).unwrap();
        if visitor.found_recursion { Err(expanded_type) } else { Ok(expanded_type) }
    }

    pub fn bound_return_position_impl_trait_in_trait_tys(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<Result<&'tcx FxHashMap<DefId, Ty<'tcx>>, ErrorGuaranteed>> {
        ty::EarlyBinder(self.collect_return_position_impl_trait_in_trait_tys(def_id))
    }

    pub fn bound_explicit_item_bounds(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<&'tcx [(ty::Predicate<'tcx>, rustc_span::Span)]> {
        ty::EarlyBinder(self.explicit_item_bounds(def_id))
    }

    pub fn bound_impl_subject(self, def_id: DefId) -> ty::EarlyBinder<ty::ImplSubject<'tcx>> {
        ty::EarlyBinder(self.impl_subject(def_id))
    }

    /// Returns names of captured upvars for closures and generators.
    ///
    /// Here are some examples:
    ///  - `name__field1__field2` when the upvar is captured by value.
    ///  - `_ref__name__field` when the upvar is captured by reference.
    ///
    /// For generators this only contains upvars that are shared by all states.
    pub fn closure_saved_names_of_captured_variables(
        self,
        def_id: DefId,
    ) -> SmallVec<[String; 16]> {
        let body = self.optimized_mir(def_id);

        body.var_debug_info
            .iter()
            .filter_map(|var| {
                let is_ref = match var.value {
                    mir::VarDebugInfoContents::Place(place)
                        if place.local == mir::Local::new(1) =>
                    {
                        // The projection is either `[.., Field, Deref]` or `[.., Field]`. It
                        // implies whether the variable is captured by value or by reference.
                        matches!(place.projection.last().unwrap(), mir::ProjectionElem::Deref)
                    }
                    _ => return None,
                };
                let prefix = if is_ref { "_ref__" } else { "" };
                Some(prefix.to_owned() + var.name.as_str())
            })
            .collect()
    }

    // FIXME(eddyb) maybe precompute this? Right now it's computed once
    // per generator monomorphization, but it doesn't depend on substs.
    pub fn generator_layout_and_saved_local_names(
        self,
        def_id: DefId,
    ) -> (
        &'tcx ty::GeneratorLayout<'tcx>,
        IndexVec<mir::GeneratorSavedLocal, Option<rustc_span::Symbol>>,
    ) {
        let tcx = self;
        let body = tcx.optimized_mir(def_id);
        let generator_layout = body.generator_layout().unwrap();
        let mut generator_saved_local_names =
            IndexVec::from_elem(None, &generator_layout.field_tys);

        let state_arg = mir::Local::new(1);
        for var in &body.var_debug_info {
            let mir::VarDebugInfoContents::Place(place) = &var.value else { continue };
            if place.local != state_arg {
                continue;
            }
            match place.projection[..] {
                [
                    // Deref of the `Pin<&mut Self>` state argument.
                    mir::ProjectionElem::Field(..),
                    mir::ProjectionElem::Deref,
                    // Field of a variant of the state.
                    mir::ProjectionElem::Downcast(_, variant),
                    mir::ProjectionElem::Field(field, _),
                ] => {
                    let name = &mut generator_saved_local_names
                        [generator_layout.variant_fields[variant][field]];
                    if name.is_none() {
                        name.replace(var.name);
                    }
                }
                _ => {}
            }
        }
        (generator_layout, generator_saved_local_names)
    }

    /// Query and get an English description for the item's kind.
    pub fn def_descr(self, def_id: DefId) -> &'static str {
        self.def_kind_descr(self.def_kind(def_id), def_id)
    }

    /// Get an English description for the item's kind.
    pub fn def_kind_descr(self, def_kind: DefKind, def_id: DefId) -> &'static str {
        match def_kind {
            DefKind::AssocFn if self.associated_item(def_id).fn_has_self_parameter => "method",
            DefKind::Generator => match self.generator_kind(def_id).unwrap() {
                rustc_hir::GeneratorKind::Async(..) => "async closure",
                rustc_hir::GeneratorKind::Gen => "generator",
            },
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
            DefKind::AssocFn if self.associated_item(def_id).fn_has_self_parameter => "a",
            DefKind::Generator => match self.generator_kind(def_id).unwrap() {
                rustc_hir::GeneratorKind::Async(..) => "an",
                rustc_hir::GeneratorKind::Gen => "a",
            },
            _ => def_kind.article(),
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
    expanded_cache: FxHashMap<(DefId, SubstsRef<'tcx>), Ty<'tcx>>,
    primary_def_id: Option<DefId>,
    found_recursion: bool,
    found_any_recursion: bool,
    expand_generators: bool,
    /// Whether or not to check for recursive opaque types.
    /// This is `true` when we're explicitly checking for opaque type
    /// recursion, and 'false' otherwise to avoid unnecessary work.
    check_recursion: bool,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> OpaqueTypeExpander<'tcx> {
    fn expand_opaque_ty(&mut self, def_id: DefId, substs: SubstsRef<'tcx>) -> Option<Ty<'tcx>> {
        if self.found_any_recursion {
            return None;
        }
        let substs = substs.fold_with(self);
        if !self.check_recursion || self.seen_opaque_tys.insert(def_id) {
            let expanded_ty = match self.expanded_cache.get(&(def_id, substs)) {
                Some(expanded_ty) => *expanded_ty,
                None => {
                    let generic_ty = self.tcx.type_of(def_id);
                    let concrete_ty = generic_ty.subst(self.tcx, substs);
                    let expanded_ty = self.fold_ty(concrete_ty);
                    self.expanded_cache.insert((def_id, substs), expanded_ty);
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

    fn expand_generator(&mut self, def_id: DefId, substs: SubstsRef<'tcx>) -> Option<Ty<'tcx>> {
        if self.found_any_recursion {
            return None;
        }
        let substs = substs.fold_with(self);
        if !self.check_recursion || self.seen_opaque_tys.insert(def_id) {
            let expanded_ty = match self.expanded_cache.get(&(def_id, substs)) {
                Some(expanded_ty) => *expanded_ty,
                None => {
                    for bty in self.tcx.generator_hidden_types(def_id) {
                        let hidden_ty = bty.subst(self.tcx, substs);
                        self.fold_ty(hidden_ty);
                    }
                    let expanded_ty = self.tcx.mk_generator_witness_mir(def_id, substs);
                    self.expanded_cache.insert((def_id, substs), expanded_ty);
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
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        let mut t = if let ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) = *t.kind() {
            self.expand_opaque_ty(def_id, substs).unwrap_or(t)
        } else if t.has_opaque_types() || t.has_generators() {
            t.super_fold_with(self)
        } else {
            t
        };
        if self.expand_generators {
            if let ty::GeneratorWitnessMIR(def_id, substs) = *t.kind() {
                t = self.expand_generator(def_id, substs).unwrap_or(t);
            }
        }
        t
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if let ty::PredicateKind::Clause(clause) = p.kind().skip_binder()
            && let ty::Clause::Projection(projection_pred) = clause
        {
            p.kind()
                .rebind(ty::ProjectionPredicate {
                    projection_ty: projection_pred.projection_ty.fold_with(self),
                    // Don't fold the term on the RHS of the projection predicate.
                    // This is because for default trait methods with RPITITs, we
                    // install a `NormalizesTo(Projection(RPITIT) -> Opaque(RPITIT))`
                    // predicate, which would trivially cause a cycle when we do
                    // anything that requires `ParamEnv::with_reveal_all_normalized`.
                    term: projection_pred.term,
                })
                .to_predicate(self.tcx)
        } else {
            p.super_fold_with(self)
        }
    }
}

impl<'tcx> Ty<'tcx> {
    pub fn int_size_and_signed(self, tcx: TyCtxt<'tcx>) -> (Size, bool) {
        let (int, signed) = match *self.kind() {
            ty::Int(ity) => (Integer::from_int_ty(&tcx, ity), true),
            ty::Uint(uty) => (Integer::from_uint_ty(&tcx, uty), false),
            _ => bug!("non integer discriminant"),
        };
        (int.size(), signed)
    }

    /// Returns the maximum value for the given numeric type (including `char`s)
    /// or returns `None` if the type is not numeric.
    pub fn numeric_max_val(self, tcx: TyCtxt<'tcx>) -> Option<ty::Const<'tcx>> {
        let val = match self.kind() {
            ty::Int(_) | ty::Uint(_) => {
                let (size, signed) = self.int_size_and_signed(tcx);
                let val =
                    if signed { size.signed_int_max() as u128 } else { size.unsigned_int_max() };
                Some(val)
            }
            ty::Char => Some(std::char::MAX as u128),
            ty::Float(fty) => Some(match fty {
                ty::FloatTy::F32 => rustc_apfloat::ieee::Single::INFINITY.to_bits(),
                ty::FloatTy::F64 => rustc_apfloat::ieee::Double::INFINITY.to_bits(),
            }),
            _ => None,
        };

        val.map(|v| ty::Const::from_bits(tcx, v, ty::ParamEnv::empty().and(self)))
    }

    /// Returns the minimum value for the given numeric type (including `char`s)
    /// or returns `None` if the type is not numeric.
    pub fn numeric_min_val(self, tcx: TyCtxt<'tcx>) -> Option<ty::Const<'tcx>> {
        let val = match self.kind() {
            ty::Int(_) | ty::Uint(_) => {
                let (size, signed) = self.int_size_and_signed(tcx);
                let val = if signed { size.truncate(size.signed_int_min() as u128) } else { 0 };
                Some(val)
            }
            ty::Char => Some(0),
            ty::Float(fty) => Some(match fty {
                ty::FloatTy::F32 => (-::rustc_apfloat::ieee::Single::INFINITY).to_bits(),
                ty::FloatTy::F64 => (-::rustc_apfloat::ieee::Double::INFINITY).to_bits(),
            }),
            _ => None,
        };

        val.map(|v| ty::Const::from_bits(tcx, v, ty::ParamEnv::empty().and(self)))
    }

    /// Checks whether values of this type `T` are *moved* or *copied*
    /// when referenced -- this amounts to a check for whether `T:
    /// Copy`, but note that we **don't** consider lifetimes when
    /// doing this check. This means that we may generate MIR which
    /// does copies even when the type actually doesn't satisfy the
    /// full requirements for the `Copy` trait (cc #29149) -- this
    /// winds up being reported as an error during NLL borrow check.
    pub fn is_copy_modulo_regions(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> bool {
        self.is_trivially_pure_clone_copy() || tcx.is_copy_raw(param_env.and(self))
    }

    /// Checks whether values of this type `T` have a size known at
    /// compile time (i.e., whether `T: Sized`). Lifetimes are ignored
    /// for the purposes of this check, so it can be an
    /// over-approximation in generic contexts, where one can have
    /// strange rules like `<T as Foo<'static>>::Bar: Sized` that
    /// actually carry lifetime requirements.
    pub fn is_sized(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> bool {
        self.is_trivially_sized(tcx) || tcx.is_sized_raw(param_env.and(self))
    }

    /// Checks whether values of this type `T` implement the `Freeze`
    /// trait -- frozen types are those that do not contain an
    /// `UnsafeCell` anywhere. This is a language concept used to
    /// distinguish "true immutability", which is relevant to
    /// optimization as well as the rules around static values. Note
    /// that the `Freeze` trait is not exposed to end users and is
    /// effectively an implementation detail.
    pub fn is_freeze(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> bool {
        self.is_trivially_freeze() || tcx.is_freeze_raw(param_env.and(self))
    }

    /// Fast path helper for testing if a type is `Freeze`.
    ///
    /// Returning true means the type is known to be `Freeze`. Returning
    /// `false` means nothing -- could be `Freeze`, might not be.
    fn is_trivially_freeze(self) -> bool {
        match self.kind() {
            ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Bool
            | ty::Char
            | ty::Str
            | ty::Never
            | ty::Ref(..)
            | ty::RawPtr(_)
            | ty::FnDef(..)
            | ty::Error(_)
            | ty::FnPtr(_) => true,
            ty::Tuple(fields) => fields.iter().all(Self::is_trivially_freeze),
            ty::Slice(elem_ty) | ty::Array(elem_ty, _) => elem_ty.is_trivially_freeze(),
            ty::Adt(..)
            | ty::Bound(..)
            | ty::Closure(..)
            | ty::Dynamic(..)
            | ty::Foreign(_)
            | ty::Generator(..)
            | ty::GeneratorWitness(_)
            | ty::GeneratorWitnessMIR(..)
            | ty::Infer(_)
            | ty::Alias(..)
            | ty::Param(_)
            | ty::Placeholder(_) => false,
        }
    }

    /// Checks whether values of this type `T` implement the `Unpin` trait.
    pub fn is_unpin(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> bool {
        self.is_trivially_unpin() || tcx.is_unpin_raw(param_env.and(self))
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
            | ty::RawPtr(_)
            | ty::FnDef(..)
            | ty::Error(_)
            | ty::FnPtr(_) => true,
            ty::Tuple(fields) => fields.iter().all(Self::is_trivially_unpin),
            ty::Slice(elem_ty) | ty::Array(elem_ty, _) => elem_ty.is_trivially_unpin(),
            ty::Adt(..)
            | ty::Bound(..)
            | ty::Closure(..)
            | ty::Dynamic(..)
            | ty::Foreign(_)
            | ty::Generator(..)
            | ty::GeneratorWitness(_)
            | ty::GeneratorWitnessMIR(..)
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
    pub fn needs_drop(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> bool {
        // Avoid querying in simple cases.
        match needs_drop_components(self, &tcx.data_layout) {
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
                let query_ty =
                    tcx.try_normalize_erasing_regions(param_env, query_ty).unwrap_or(query_ty);

                tcx.needs_drop_raw(param_env.and(query_ty))
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
    pub fn has_significant_drop(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> bool {
        // Avoid querying in simple cases.
        match needs_drop_components(self, &tcx.data_layout) {
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
                if query_ty.needs_infer() {
                    return true;
                }

                // This doesn't depend on regions, so try to minimize distinct
                // query keys used.
                let erased = tcx.normalize_erasing_regions(param_env, query_ty);
                tcx.has_significant_drop_raw(param_env.and(erased))
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
    /// equality is indicated by an implementation of `PartialStructuralEq` and `StructuralEq` for
    /// that type.
    ///
    /// This function is "shallow" because it may return `true` for a composite type whose fields
    /// are not `StructuralEq`. For example, `[T; 4]` has structural equality regardless of `T`
    /// because equality for arrays is determined by the equality of each array element. If you
    /// want to know whether a given call to `PartialEq::eq` will proceed structurally all the way
    /// down, you will need to use a type visitor.
    #[inline]
    pub fn is_structural_eq_shallow(self, tcx: TyCtxt<'tcx>) -> bool {
        match self.kind() {
            // Look for an impl of both `PartialStructuralEq` and `StructuralEq`.
            ty::Adt(..) => tcx.has_structural_eq_impls(self),

            // Primitive types that satisfy `Eq`.
            ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Str | ty::Never => true,

            // Composite types that satisfy `Eq` when all of their fields do.
            //
            // Because this function is "shallow", we return `true` for these composites regardless
            // of the type(s) contained within.
            ty::Ref(..) | ty::Array(..) | ty::Slice(_) | ty::Tuple(..) => true,

            // Raw pointers use bitwise comparison.
            ty::RawPtr(_) | ty::FnPtr(_) => true,

            // Floating point numbers are not `Eq`.
            ty::Float(_) => false,

            // Conservatively return `false` for all others...

            // Anonymous function types
            ty::FnDef(..) | ty::Closure(..) | ty::Dynamic(..) | ty::Generator(..) => false,

            // Generic or inferred types
            //
            // FIXME(ecstaticmorse): Maybe we should `bug` here? This should probably only be
            // called for known, fully-monomorphized types.
            ty::Alias(..) | ty::Param(_) | ty::Bound(..) | ty::Placeholder(_) | ty::Infer(_) => {
                false
            }

            ty::Foreign(_)
            | ty::GeneratorWitness(..)
            | ty::GeneratorWitnessMIR(..)
            | ty::Error(_) => false,
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

    #[inline]
    pub fn outer_exclusive_binder(self) -> ty::DebruijnIndex {
        self.0.outer_exclusive_binder
    }
}

pub enum ExplicitSelf<'tcx> {
    ByValue,
    ByReference(ty::Region<'tcx>, hir::Mutability),
    ByRawPointer(hir::Mutability),
    ByBox,
    Other,
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
    /// ```ignore (illustrative)
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
    pub fn determine<P>(self_arg_ty: Ty<'tcx>, is_self_ty: P) -> ExplicitSelf<'tcx>
    where
        P: Fn(Ty<'tcx>) -> bool,
    {
        use self::ExplicitSelf::*;

        match *self_arg_ty.kind() {
            _ if is_self_ty(self_arg_ty) => ByValue,
            ty::Ref(region, ty, mutbl) if is_self_ty(ty) => ByReference(region, mutbl),
            ty::RawPtr(ty::TypeAndMut { ty, mutbl }) if is_self_ty(ty) => ByRawPointer(mutbl),
            ty::Adt(def, _) if def.is_box() && is_self_ty(self_arg_ty.boxed_ty()) => ByBox,
            _ => Other,
        }
    }
}

/// Returns a list of types such that the given type needs drop if and only if
/// *any* of the returned types need drop. Returns `Err(AlwaysRequiresDrop)` if
/// this type always needs drop.
pub fn needs_drop_components<'tcx>(
    ty: Ty<'tcx>,
    target_layout: &TargetDataLayout,
) -> Result<SmallVec<[Ty<'tcx>; 2]>, AlwaysRequiresDrop> {
    match ty.kind() {
        ty::Infer(ty::FreshIntTy(_))
        | ty::Infer(ty::FreshFloatTy(_))
        | ty::Bool
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Never
        | ty::FnDef(..)
        | ty::FnPtr(_)
        | ty::Char
        | ty::GeneratorWitness(..)
        | ty::GeneratorWitnessMIR(..)
        | ty::RawPtr(_)
        | ty::Ref(..)
        | ty::Str => Ok(SmallVec::new()),

        // Foreign types can never have destructors.
        ty::Foreign(..) => Ok(SmallVec::new()),

        ty::Dynamic(..) | ty::Error(_) => Err(AlwaysRequiresDrop),

        ty::Slice(ty) => needs_drop_components(*ty, target_layout),
        ty::Array(elem_ty, size) => {
            match needs_drop_components(*elem_ty, target_layout) {
                Ok(v) if v.is_empty() => Ok(v),
                res => match size.kind().try_to_bits(target_layout.pointer_size) {
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
            acc.extend(needs_drop_components(elem, target_layout)?);
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
        | ty::Generator(..) => Ok(smallvec![ty]),
    }
}

pub fn is_trivially_const_drop(ty: Ty<'_>) -> bool {
    match *ty.kind() {
        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Infer(ty::IntVar(_))
        | ty::Infer(ty::FloatVar(_))
        | ty::Str
        | ty::RawPtr(_)
        | ty::Ref(..)
        | ty::FnDef(..)
        | ty::FnPtr(_)
        | ty::Never
        | ty::Foreign(_) => true,

        ty::Alias(..)
        | ty::Dynamic(..)
        | ty::Error(_)
        | ty::Bound(..)
        | ty::Param(_)
        | ty::Placeholder(_)
        | ty::Infer(_) => false,

        // Not trivial because they have components, and instead of looking inside,
        // we'll just perform trait selection.
        ty::Closure(..)
        | ty::Generator(..)
        | ty::GeneratorWitness(_)
        | ty::GeneratorWitnessMIR(..)
        | ty::Adt(..) => false,

        ty::Array(ty, _) | ty::Slice(ty) => is_trivially_const_drop(ty),

        ty::Tuple(tys) => tys.iter().all(|ty| is_trivially_const_drop(ty)),
    }
}

/// Does the equivalent of
/// ```ignore (ilustrative)
/// let v = self.iter().map(|p| p.fold_with(folder)).collect::<SmallVec<[_; 8]>>();
/// folder.tcx().intern_*(&v)
/// ```
pub fn fold_list<'tcx, F, T>(
    list: &'tcx ty::List<T>,
    folder: &mut F,
    intern: impl FnOnce(TyCtxt<'tcx>, &[T]) -> &'tcx ty::List<T>,
) -> Result<&'tcx ty::List<T>, F::Error>
where
    F: FallibleTypeFolder<TyCtxt<'tcx>>,
    T: TypeFoldable<TyCtxt<'tcx>> + PartialEq + Copy,
{
    let mut iter = list.iter();
    // Look for the first element that changed
    match iter.by_ref().enumerate().find_map(|(i, t)| match t.try_fold_with(folder) {
        Ok(new_t) if new_t == t => None,
        new_t => Some((i, new_t)),
    }) {
        Some((i, Ok(new_t))) => {
            // An element changed, prepare to intern the resulting list
            let mut new_list = SmallVec::<[_; 8]>::with_capacity(list.len());
            new_list.extend_from_slice(&list[..i]);
            new_list.push(new_t);
            for t in iter {
                new_list.push(t.try_fold_with(folder)?)
            }
            Ok(intern(folder.interner(), &new_list))
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
    val: &'tcx ty::List<ty::Predicate<'tcx>>,
) -> &'tcx ty::List<ty::Predicate<'tcx>> {
    let mut visitor = OpaqueTypeExpander {
        seen_opaque_tys: FxHashSet::default(),
        expanded_cache: FxHashMap::default(),
        primary_def_id: None,
        found_recursion: false,
        found_any_recursion: false,
        check_recursion: false,
        expand_generators: false,
        tcx,
    };
    val.fold_with(&mut visitor)
}

/// Determines whether an item is annotated with `doc(hidden)`.
fn is_doc_hidden(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    assert!(def_id.is_local());
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

/// Determines whether an item is an intrinsic by Abi.
pub fn is_intrinsic(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    matches!(tcx.fn_sig(def_id).skip_binder().abi(), Abi::RustIntrinsic | Abi::PlatformIntrinsic)
}

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        reveal_opaque_types_in_bounds,
        is_doc_hidden,
        is_doc_notable_trait,
        is_intrinsic,
        ..*providers
    }
}
