//! Defines how the compiler represents types internally.
//!
//! Two important entities in this module are:
//!
//! - [`rustc_middle::ty::Ty`], used to represent the semantics of a type.
//! - [`rustc_middle::ty::TyCtxt`], the central data structure in the compiler.
//!
//! For more information, see ["The `ty` module: representing types"] in the rustc-dev-guide.
//!
//! ["The `ty` module: representing types"]: https://rustc-dev-guide.rust-lang.org/ty.html

#![allow(rustc::usage_of_ty_tykind)]

use rustc_ast as ast;
use rustc_attr as attr;
use rustc_data_structures::intern::Interned;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::Node;
use rustc_serialize::{Decodable, Encodable};
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::{ExpnKind, Span};
use rustc_target::abi::{Align, FieldIdx, Integer, IntegerType, VariantIdx};
use rustc_type_ir::WithCachedTypeInfo;

use self::fast_reject::SimplifiedType;
use self::util::Discr;
use crate::error::TypeMismatchReason;
use crate::mir::{Body, GeneratorLayout};
use crate::ty;

pub mod _match;
pub mod abstract_const;
pub mod adjustment;
pub mod binding;
pub mod cast;
pub mod codec;
pub mod error;
pub mod fast_reject;
pub mod flags;
pub mod fold;
pub mod inhabitedness;
pub mod layout;
pub mod normalize_erasing_regions;
pub mod print;
pub mod query;
pub mod relate;
pub mod subst;
pub mod trait_def;
pub mod util;
pub mod visit;
pub mod vtable;
pub mod walk;

mod adt;
mod alias_relation_direction;
mod assoc;
mod c_reader_cache_key;
mod closure;
mod closure_size_profile_data;
mod coherence;
mod consts;
mod context;
mod crate_variances_map;
mod destructor;
mod destructured_const;
mod diagnostics;
mod erase_regions;
mod generics;
mod impl_trait_in_trait_data;
mod impls_ty;
mod infer_var_info;
mod instance;
mod list;
mod opaque;
mod param_env;
mod parameterized;
mod placeholder;
mod predicate;
mod rvalue_scopes;
mod structural_impls;
mod sty;
mod symbol_name;
mod term;
mod ty_; // FIXME: rename to `ty` once we don't import `crate::ty` here
mod typeck_results;
mod visibility;

pub use rustc_session::lint::RegisteredTools;
pub use rustc_target::abi::{ReprFlags, ReprOptions};
pub use rustc_type_ir::AliasKind::*;
pub use rustc_type_ir::DynKind::*;
pub use rustc_type_ir::InferTy::*;
pub use rustc_type_ir::RegionKind::*;
pub use rustc_type_ir::TyKind::*;
pub use rustc_type_ir::*;

pub use self::adt::{
    AdtDef, AdtDefData, AdtFlags, AdtKind, FieldDef, Representability, VariantDef, VariantDiscr,
    VariantFlags,
};
pub use self::alias_relation_direction::AliasRelationDirection;
pub use self::assoc::*;
pub use self::binding::BindingMode;
pub use self::binding::BindingMode::*;
pub use self::c_reader_cache_key::CReaderCacheKey;
pub use self::closure::{
    is_ancestor_or_same_capture, place_to_string_for_capture, BorrowKind, CaptureInfo,
    CapturedPlace, ClosureKind, ClosureTypeInfo, MinCaptureInformationMap, MinCaptureList,
    RootVariableMinCaptureList, UpvarCapture, UpvarCaptureMap, UpvarId, UpvarListMap, UpvarPath,
    CAPTURE_STRUCT_LOCAL,
};
pub use self::closure_size_profile_data::ClosureSizeProfileData;
pub use self::coherence::{
    CrateInherentImpls, ImplHeader, ImplOverlapKind, ImplPolarity, ImplSubject,
};
pub use self::consts::{
    BoundConst, Const, ConstData, ConstInt, ConstKind, Expr, InferConst, ScalarInt,
    UnevaluatedConst, ValTree,
};
pub use self::context::{
    tls, CtxtInterners, DeducedParamAttrs, FreeRegionInfo, GlobalCtxt, Lift, TyCtxt, TyCtxtFeed,
};
pub use self::crate_variances_map::CrateVariancesMap;
pub use self::destructor::Destructor;
pub use self::destructured_const::DestructuredConst;
pub use self::diagnostics::*;
pub use self::fold::{FallibleTypeFolder, TypeFoldable, TypeFolder, TypeSuperFoldable};
pub use self::generics::*;
pub use self::impl_trait_in_trait_data::ImplTraitInTraitData;
pub use self::infer_var_info::InferVarInfo;
pub use self::instance::{Instance, InstanceDef, ShortInstance, UnusedGenericParams};
pub use self::list::List;
pub use self::opaque::{OpaqueHiddenType, OpaqueTypeKey};
pub use self::param_env::{ParamEnv, ParamEnvAnd};
pub use self::parameterized::ParameterizedOverTcx;
pub use self::placeholder::{Placeholder, PlaceholderConst, PlaceholderRegion, PlaceholderType};
pub use self::predicate::{
    BoundConstness, Clause, CoercePredicate, CratePredicatesMap, InstantiatedPredicates,
    OutlivesPredicate, PolyCoercePredicate, PolyProjectionPredicate, PolyRegionOutlivesPredicate,
    PolySubtypePredicate, PolyTraitPredicate, PolyTypeOutlivesPredicate, Predicate, PredicateKind,
    ProjectionPredicate, RegionOutlivesPredicate, SubtypePredicate, ToPredicate, TraitPredicate,
    TypeOutlivesPredicate,
};
pub use self::rvalue_scopes::RvalueScopes;
pub use self::sty::BoundRegionKind::*;
pub use self::sty::{
    AliasTy, Article, Binder, BoundRegion, BoundRegionKind, BoundTy, BoundTyKind, BoundVar,
    BoundVariableKind, CanonicalPolyFnSig, ClosureSubsts, ClosureSubstsParts, ConstVid,
    EarlyBoundRegion, ExistentialPredicate, ExistentialProjection, ExistentialTraitRef, FnSig,
    FreeRegion, GenSig, GeneratorSubsts, GeneratorSubstsParts, InlineConstSubsts,
    InlineConstSubstsParts, ParamConst, ParamTy, PolyExistentialPredicate,
    PolyExistentialProjection, PolyExistentialTraitRef, PolyFnSig, PolyGenSig, PolyTraitRef,
    Region, RegionKind, RegionVid, TraitRef, TyKind, TypeAndMut, UpvarSubsts, VarianceDiagInfo,
};
pub use self::subst::*;
pub use self::symbol_name::SymbolName;
pub use self::term::{ParamTerm, Term, TermKind};
pub use self::trait_def::TraitDef;
pub use self::ty_::Ty;
pub use self::typeck_results::{
    CanonicalUserType, CanonicalUserTypeAnnotation, CanonicalUserTypeAnnotations,
    GeneratorDiagnosticData, GeneratorInteriorTypeCause, TypeckResults, UserType,
    UserTypeAnnotationIndex,
};
pub use self::visibility::Visibility;
pub use self::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};
pub use self::vtable::*;
pub use self::AssocItemContainer::*;
pub use self::BorrowKind::*;
pub use self::IntVarValue::*;
pub use self::Variance::*;

impl TyCtxt<'_> {
    #[inline]
    pub fn opt_parent(self, id: DefId) -> Option<DefId> {
        self.def_key(id).parent.map(|index| DefId { index, ..id })
    }

    #[inline]
    #[track_caller]
    pub fn parent(self, id: DefId) -> DefId {
        match self.opt_parent(id) {
            Some(id) => id,
            // not `unwrap_or_else` to avoid breaking caller tracking
            None => bug!("{id:?} doesn't have a parent"),
        }
    }

    #[inline]
    #[track_caller]
    pub fn opt_local_parent(self, id: LocalDefId) -> Option<LocalDefId> {
        self.opt_parent(id.to_def_id()).map(DefId::expect_local)
    }

    #[inline]
    #[track_caller]
    pub fn local_parent(self, id: LocalDefId) -> LocalDefId {
        self.parent(id.to_def_id()).expect_local()
    }

    pub fn is_descendant_of(self, mut descendant: DefId, ancestor: DefId) -> bool {
        if descendant.krate != ancestor.krate {
            return false;
        }

        while descendant != ancestor {
            match self.opt_parent(descendant) {
                Some(parent) => descendant = parent,
                None => return false,
            }
        }
        true
    }
}

const TAG_MASK: usize = 0b11;
const TYPE_TAG: usize = 0b00;
const CONST_TAG: usize = 0b01;

pub trait ToPolyTraitRef<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx>;
}

impl<'tcx> ToPolyTraitRef<'tcx> for PolyTraitPredicate<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        self.map_bound_ref(|trait_pred| trait_pred.trait_ref)
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn typeck_body(self, body: hir::BodyId) -> &'tcx TypeckResults<'tcx> {
        self.typeck(self.hir().body_owner_def_id(body))
    }

    pub fn provided_trait_methods(self, id: DefId) -> impl 'tcx + Iterator<Item = &'tcx AssocItem> {
        self.associated_items(id)
            .in_definition_order()
            .filter(move |item| item.kind == AssocKind::Fn && item.defaultness(self).has_value())
    }

    pub fn repr_options_of_def(self, did: DefId) -> ReprOptions {
        let mut flags = ReprFlags::empty();
        let mut size = None;
        let mut max_align: Option<Align> = None;
        let mut min_pack: Option<Align> = None;

        // Generate a deterministically-derived seed from the item's path hash
        // to allow for cross-crate compilation to actually work
        let mut field_shuffle_seed = self.def_path_hash(did).0.to_smaller_hash();

        // If the user defined a custom seed for layout randomization, xor the item's
        // path hash with the user defined seed, this will allowing determinism while
        // still allowing users to further randomize layout generation for e.g. fuzzing
        if let Some(user_seed) = self.sess.opts.unstable_opts.layout_seed {
            field_shuffle_seed ^= user_seed;
        }

        for attr in self.get_attrs(did, sym::repr) {
            for r in attr::parse_repr_attr(&self.sess, attr) {
                flags.insert(match r {
                    attr::ReprC => ReprFlags::IS_C,
                    attr::ReprPacked(pack) => {
                        let pack = Align::from_bytes(pack as u64).unwrap();
                        min_pack = Some(if let Some(min_pack) = min_pack {
                            min_pack.min(pack)
                        } else {
                            pack
                        });
                        ReprFlags::empty()
                    }
                    attr::ReprTransparent => ReprFlags::IS_TRANSPARENT,
                    attr::ReprSimd => ReprFlags::IS_SIMD,
                    attr::ReprInt(i) => {
                        size = Some(match i {
                            attr::IntType::SignedInt(x) => match x {
                                ast::IntTy::Isize => IntegerType::Pointer(true),
                                ast::IntTy::I8 => IntegerType::Fixed(Integer::I8, true),
                                ast::IntTy::I16 => IntegerType::Fixed(Integer::I16, true),
                                ast::IntTy::I32 => IntegerType::Fixed(Integer::I32, true),
                                ast::IntTy::I64 => IntegerType::Fixed(Integer::I64, true),
                                ast::IntTy::I128 => IntegerType::Fixed(Integer::I128, true),
                            },
                            attr::IntType::UnsignedInt(x) => match x {
                                ast::UintTy::Usize => IntegerType::Pointer(false),
                                ast::UintTy::U8 => IntegerType::Fixed(Integer::I8, false),
                                ast::UintTy::U16 => IntegerType::Fixed(Integer::I16, false),
                                ast::UintTy::U32 => IntegerType::Fixed(Integer::I32, false),
                                ast::UintTy::U64 => IntegerType::Fixed(Integer::I64, false),
                                ast::UintTy::U128 => IntegerType::Fixed(Integer::I128, false),
                            },
                        });
                        ReprFlags::empty()
                    }
                    attr::ReprAlign(align) => {
                        max_align = max_align.max(Some(Align::from_bytes(align as u64).unwrap()));
                        ReprFlags::empty()
                    }
                });
            }
        }

        // If `-Z randomize-layout` was enabled for the type definition then we can
        // consider performing layout randomization
        if self.sess.opts.unstable_opts.randomize_layout {
            flags.insert(ReprFlags::RANDOMIZE_LAYOUT);
        }

        // This is here instead of layout because the choice must make it into metadata.
        if !self.consider_optimizing(|| format!("Reorder fields of {:?}", self.def_path_str(did))) {
            flags.insert(ReprFlags::IS_LINEAR);
        }

        ReprOptions { int: size, align: max_align, pack: min_pack, flags, field_shuffle_seed }
    }

    /// Look up the name of a definition across crates. This does not look at HIR.
    pub fn opt_item_name(self, def_id: DefId) -> Option<Symbol> {
        if let Some(cnum) = def_id.as_crate_root() {
            Some(self.crate_name(cnum))
        } else {
            let def_key = self.def_key(def_id);
            match def_key.disambiguated_data.data {
                // The name of a constructor is that of its parent.
                rustc_hir::definitions::DefPathData::Ctor => self
                    .opt_item_name(DefId { krate: def_id.krate, index: def_key.parent.unwrap() }),
                // The name of opaque types only exists in HIR.
                rustc_hir::definitions::DefPathData::ImplTrait
                    if let Some(def_id) = def_id.as_local() =>
                    self.hir().opt_name(self.hir().local_def_id_to_hir_id(def_id)),
                _ => def_key.get_opt_name(),
            }
        }
    }

    /// Look up the name of a definition across crates. This does not look at HIR.
    ///
    /// This method will ICE if the corresponding item does not have a name. In these cases, use
    /// [`opt_item_name`] instead.
    ///
    /// [`opt_item_name`]: Self::opt_item_name
    pub fn item_name(self, id: DefId) -> Symbol {
        self.opt_item_name(id).unwrap_or_else(|| {
            bug!("item_name: no name for {:?}", self.def_path(id));
        })
    }

    /// Look up the name and span of a definition.
    ///
    /// See [`item_name`][Self::item_name] for more information.
    pub fn opt_item_ident(self, def_id: DefId) -> Option<Ident> {
        let def = self.opt_item_name(def_id)?;
        let span = self
            .def_ident_span(def_id)
            .unwrap_or_else(|| bug!("missing ident span for {def_id:?}"));
        Some(Ident::new(def, span))
    }

    pub fn opt_associated_item(self, def_id: DefId) -> Option<AssocItem> {
        if let DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy = self.def_kind(def_id) {
            Some(self.associated_item(def_id))
        } else {
            None
        }
    }

    /// If the def-id is an associated type that was desugared from a
    /// return-position `impl Trait` from a trait, then provide the source info
    /// about where that RPITIT came from.
    pub fn opt_rpitit_info(self, def_id: DefId) -> Option<ImplTraitInTraitData> {
        if let DefKind::AssocTy = self.def_kind(def_id) {
            self.associated_item(def_id).opt_rpitit_info
        } else {
            None
        }
    }

    pub fn find_field_index(self, ident: Ident, variant: &VariantDef) -> Option<FieldIdx> {
        variant.fields.iter_enumerated().find_map(|(i, field)| {
            self.hygienic_eq(ident, field.ident(self), variant.def_id).then_some(i)
        })
    }

    /// Returns `true` if the impls are the same polarity and the trait either
    /// has no items or is annotated `#[marker]` and prevents item overrides.
    #[instrument(level = "debug", skip(self), ret)]
    pub fn impls_are_allowed_to_overlap(
        self,
        def_id1: DefId,
        def_id2: DefId,
    ) -> Option<ImplOverlapKind> {
        let impl_trait_ref1 = self.impl_trait_ref(def_id1);
        let impl_trait_ref2 = self.impl_trait_ref(def_id2);
        // If either trait impl references an error, they're allowed to overlap,
        // as one of them essentially doesn't exist.
        if impl_trait_ref1.map_or(false, |tr| tr.subst_identity().references_error())
            || impl_trait_ref2.map_or(false, |tr| tr.subst_identity().references_error())
        {
            return Some(ImplOverlapKind::Permitted { marker: false });
        }

        match (self.impl_polarity(def_id1), self.impl_polarity(def_id2)) {
            (ImplPolarity::Reservation, _) | (_, ImplPolarity::Reservation) => {
                // `#[rustc_reservation_impl]` impls don't overlap with anything
                return Some(ImplOverlapKind::Permitted { marker: false });
            }
            (ImplPolarity::Positive, ImplPolarity::Negative)
            | (ImplPolarity::Negative, ImplPolarity::Positive) => {
                // `impl AutoTrait for Type` + `impl !AutoTrait for Type`
                return None;
            }
            (ImplPolarity::Positive, ImplPolarity::Positive)
            | (ImplPolarity::Negative, ImplPolarity::Negative) => {}
        };

        let is_marker_overlap = {
            let is_marker_impl = |trait_ref: Option<EarlyBinder<TraitRef<'_>>>| -> bool {
                trait_ref.map_or(false, |tr| self.trait_def(tr.skip_binder().def_id).is_marker)
            };
            is_marker_impl(impl_trait_ref1) && is_marker_impl(impl_trait_ref2)
        };

        if is_marker_overlap {
            Some(ImplOverlapKind::Permitted { marker: true })
        } else {
            if let Some(self_ty1) = self.issue33140_self_ty(def_id1) {
                if let Some(self_ty2) = self.issue33140_self_ty(def_id2) {
                    if self_ty1 == self_ty2 {
                        return Some(ImplOverlapKind::Issue33140);
                    } else {
                        debug!("found {self_ty1:?} != {self_ty2:?}");
                    }
                }
            }

            None
        }
    }

    /// Returns `ty::VariantDef` if `res` refers to a struct,
    /// or variant or their constructors, panics otherwise.
    pub fn expect_variant_res(self, res: Res) -> &'tcx VariantDef {
        match res {
            Res::Def(DefKind::Variant, did) => {
                let enum_did = self.parent(did);
                self.adt_def(enum_did).variant_with_id(did)
            }
            Res::Def(DefKind::Struct | DefKind::Union, did) => self.adt_def(did).non_enum_variant(),
            Res::Def(DefKind::Ctor(CtorOf::Variant, ..), variant_ctor_did) => {
                let variant_did = self.parent(variant_ctor_did);
                let enum_did = self.parent(variant_did);
                self.adt_def(enum_did).variant_with_ctor_id(variant_ctor_did)
            }
            Res::Def(DefKind::Ctor(CtorOf::Struct, ..), ctor_did) => {
                let struct_did = self.parent(ctor_did);
                self.adt_def(struct_did).non_enum_variant()
            }
            _ => bug!("expect_variant_res used with unexpected res {:?}", res),
        }
    }

    /// Returns the possibly-auto-generated MIR of a `(DefId, Subst)` pair.
    #[instrument(skip(self), level = "debug")]
    pub fn instance_mir(self, instance: ty::InstanceDef<'tcx>) -> &'tcx Body<'tcx> {
        match instance {
            ty::InstanceDef::Item(def) => {
                debug!("calling def_kind on def: {:?}", def);
                let def_kind = self.def_kind(def);
                debug!("returned from def_kind: {:?}", def_kind);
                match def_kind {
                    DefKind::Const
                    | DefKind::Static(..)
                    | DefKind::AssocConst
                    | DefKind::Ctor(..)
                    | DefKind::AnonConst
                    | DefKind::InlineConst => self.mir_for_ctfe(def),
                    // If the caller wants `mir_for_ctfe` of a function they should not be using
                    // `instance_mir`, so we'll assume const fn also wants the optimized version.
                    _ => self.optimized_mir(def),
                }
            }
            ty::InstanceDef::VTableShim(..)
            | ty::InstanceDef::ReifyShim(..)
            | ty::InstanceDef::Intrinsic(..)
            | ty::InstanceDef::FnPtrShim(..)
            | ty::InstanceDef::Virtual(..)
            | ty::InstanceDef::ClosureOnceShim { .. }
            | ty::InstanceDef::DropGlue(..)
            | ty::InstanceDef::CloneShim(..)
            | ty::InstanceDef::ThreadLocalShim(..)
            | ty::InstanceDef::FnPtrAddrShim(..) => self.mir_shims(instance),
        }
    }

    // FIXME(@lcnr): Remove this function.
    pub fn get_attrs_unchecked(self, did: DefId) -> &'tcx [ast::Attribute] {
        if let Some(did) = did.as_local() {
            self.hir().attrs(self.hir().local_def_id_to_hir_id(did))
        } else {
            self.item_attrs(did)
        }
    }

    /// Gets all attributes with the given name.
    pub fn get_attrs(
        self,
        did: impl Into<DefId>,
        attr: Symbol,
    ) -> impl Iterator<Item = &'tcx ast::Attribute> {
        let did: DefId = did.into();
        let filter_fn = move |a: &&ast::Attribute| a.has_name(attr);
        if let Some(did) = did.as_local() {
            self.hir().attrs(self.hir().local_def_id_to_hir_id(did)).iter().filter(filter_fn)
        } else if cfg!(debug_assertions) && rustc_feature::is_builtin_only_local(attr) {
            bug!("tried to access the `only_local` attribute `{}` from an extern crate", attr);
        } else {
            self.item_attrs(did).iter().filter(filter_fn)
        }
    }

    pub fn get_attr(self, did: impl Into<DefId>, attr: Symbol) -> Option<&'tcx ast::Attribute> {
        if cfg!(debug_assertions) && !rustc_feature::is_valid_for_get_attr(attr) {
            let did: DefId = did.into();
            bug!("get_attr: unexpected called with DefId `{:?}`, attr `{:?}`", did, attr);
        } else {
            self.get_attrs(did, attr).next()
        }
    }

    /// Determines whether an item is annotated with an attribute.
    pub fn has_attr(self, did: impl Into<DefId>, attr: Symbol) -> bool {
        let did: DefId = did.into();
        if cfg!(debug_assertions) && !did.is_local() && rustc_feature::is_builtin_only_local(attr) {
            bug!("tried to access the `only_local` attribute `{}` from an extern crate", attr);
        } else {
            self.get_attrs(did, attr).next().is_some()
        }
    }

    /// Returns `true` if this is an `auto trait`.
    pub fn trait_is_auto(self, trait_def_id: DefId) -> bool {
        self.trait_def(trait_def_id).has_auto_impl
    }

    /// Returns `true` if this is coinductive, either because it is
    /// an auto trait or because it has the `#[rustc_coinductive]` attribute.
    pub fn trait_is_coinductive(self, trait_def_id: DefId) -> bool {
        self.trait_def(trait_def_id).is_coinductive
    }

    /// Returns `true` if this is a trait alias.
    pub fn trait_is_alias(self, trait_def_id: DefId) -> bool {
        self.def_kind(trait_def_id) == DefKind::TraitAlias
    }

    /// Returns layout of a generator. Layout might be unavailable if the
    /// generator is tainted by errors.
    pub fn generator_layout(self, def_id: DefId) -> Option<&'tcx GeneratorLayout<'tcx>> {
        self.optimized_mir(def_id).generator_layout()
    }

    /// Given the `DefId` of an impl, returns the `DefId` of the trait it implements.
    /// If it implements no trait, returns `None`.
    pub fn trait_id_of_impl(self, def_id: DefId) -> Option<DefId> {
        self.impl_trait_ref(def_id).map(|tr| tr.skip_binder().def_id)
    }

    /// If the given `DefId` describes an item belonging to a trait,
    /// returns the `DefId` of the trait that the trait item belongs to;
    /// otherwise, returns `None`.
    pub fn trait_of_item(self, def_id: DefId) -> Option<DefId> {
        if let DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy = self.def_kind(def_id) {
            let parent = self.parent(def_id);
            if let DefKind::Trait | DefKind::TraitAlias = self.def_kind(parent) {
                return Some(parent);
            }
        }
        None
    }

    /// If the given `DefId` describes a method belonging to an impl, returns the
    /// `DefId` of the impl that the method belongs to; otherwise, returns `None`.
    pub fn impl_of_method(self, def_id: DefId) -> Option<DefId> {
        if let DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy = self.def_kind(def_id) {
            let parent = self.parent(def_id);
            if let DefKind::Impl { .. } = self.def_kind(parent) {
                return Some(parent);
            }
        }
        None
    }

    /// Check if the given `DefId` is `#\[automatically_derived\]`, *and*
    /// whether it was produced by expanding a builtin derive macro.
    pub fn is_builtin_derived(self, def_id: DefId) -> bool {
        if self.is_automatically_derived(def_id)
            && let Some(def_id) = def_id.as_local()
            && let outer = self.def_span(def_id).ctxt().outer_expn_data()
            && matches!(outer.kind, ExpnKind::Macro(MacroKind::Derive, _))
            && self.has_attr(outer.macro_def_id.unwrap(), sym::rustc_builtin_macro)
        {
            true
        } else {
            false
        }
    }

    /// Check if the given `DefId` is `#\[automatically_derived\]`.
    pub fn is_automatically_derived(self, def_id: DefId) -> bool {
        self.has_attr(def_id, sym::automatically_derived)
    }

    /// Looks up the span of `impl_did` if the impl is local; otherwise returns `Err`
    /// with the name of the crate containing the impl.
    pub fn span_of_impl(self, impl_def_id: DefId) -> Result<Span, Symbol> {
        if let Some(impl_def_id) = impl_def_id.as_local() {
            Ok(self.def_span(impl_def_id))
        } else {
            Err(self.crate_name(impl_def_id.krate))
        }
    }

    /// Hygienically compares a use-site name (`use_name`) for a field or an associated item with
    /// its supposed definition name (`def_name`). The method also needs `DefId` of the supposed
    /// definition's parent/scope to perform comparison.
    pub fn hygienic_eq(self, use_name: Ident, def_name: Ident, def_parent_def_id: DefId) -> bool {
        // We could use `Ident::eq` here, but we deliberately don't. The name
        // comparison fails frequently, and we want to avoid the expensive
        // `normalize_to_macros_2_0()` calls required for the span comparison whenever possible.
        use_name.name == def_name.name
            && use_name
                .span
                .ctxt()
                .hygienic_eq(def_name.span.ctxt(), self.expn_that_defined(def_parent_def_id))
    }

    pub fn adjust_ident(self, mut ident: Ident, scope: DefId) -> Ident {
        ident.span.normalize_to_macros_2_0_and_adjust(self.expn_that_defined(scope));
        ident
    }

    // FIXME(vincenzopalazzo): move the HirId to a LocalDefId
    pub fn adjust_ident_and_get_scope(
        self,
        mut ident: Ident,
        scope: DefId,
        block: hir::HirId,
    ) -> (Ident, DefId) {
        let scope = ident
            .span
            .normalize_to_macros_2_0_and_adjust(self.expn_that_defined(scope))
            .and_then(|actual_expansion| actual_expansion.expn_data().parent_module)
            .unwrap_or_else(|| self.parent_module(block).to_def_id());
        (ident, scope)
    }

    /// Returns `true` if the debuginfo for `span` should be collapsed to the outermost expansion
    /// site. Only applies when `Span` is the result of macro expansion.
    ///
    /// - If the `collapse_debuginfo` feature is enabled then debuginfo is not collapsed by default
    ///   and only when a macro definition is annotated with `#[collapse_debuginfo]`.
    /// - If `collapse_debuginfo` is not enabled, then debuginfo is collapsed by default.
    ///
    /// When `-Zdebug-macros` is provided then debuginfo will never be collapsed.
    pub fn should_collapse_debuginfo(self, span: Span) -> bool {
        !self.sess.opts.unstable_opts.debug_macros
            && if self.features().collapse_debuginfo {
                span.in_macro_expansion_with_collapse_debuginfo()
            } else {
                // Inlined spans should not be collapsed as that leads to all of the
                // inlined code being attributed to the inline callsite.
                span.from_expansion() && !span.is_inlined()
            }
    }

    #[inline]
    pub fn is_const_fn_raw(self, def_id: DefId) -> bool {
        matches!(
            self.def_kind(def_id),
            DefKind::Fn | DefKind::AssocFn | DefKind::Ctor(..) | DefKind::Closure
        ) && self.constness(def_id) == hir::Constness::Const
    }

    #[inline]
    pub fn is_const_default_method(self, def_id: DefId) -> bool {
        matches!(self.trait_of_item(def_id), Some(trait_id) if self.has_attr(trait_id, sym::const_trait))
    }

    pub fn impl_trait_in_trait_parent_fn(self, mut def_id: DefId) -> DefId {
        match self.opt_rpitit_info(def_id) {
            Some(ImplTraitInTraitData::Trait { fn_def_id, .. })
            | Some(ImplTraitInTraitData::Impl { fn_def_id, .. }) => fn_def_id,
            None => {
                while let def_kind = self.def_kind(def_id) && def_kind != DefKind::AssocFn {
                    debug_assert_eq!(def_kind, DefKind::ImplTraitPlaceholder);
                    def_id = self.parent(def_id);
                }
                def_id
            }
        }
    }

    pub fn impl_method_has_trait_impl_trait_tys(self, def_id: DefId) -> bool {
        if self.def_kind(def_id) != DefKind::AssocFn {
            return false;
        }

        let Some(item) = self.opt_associated_item(def_id) else { return false; };
        if item.container != ty::AssocItemContainer::ImplContainer {
            return false;
        }

        let Some(trait_item_def_id) = item.trait_item_def_id else { return false; };

        if self.lower_impl_trait_in_trait_to_assoc_ty() {
            return !self
                .associated_types_for_impl_traits_in_associated_fn(trait_item_def_id)
                .is_empty();
        }

        // FIXME(RPITIT): This does a somewhat manual walk through the signature
        // of the trait fn to look for any RPITITs, but that's kinda doing a lot
        // of work. We can probably remove this when we refactor RPITITs to be
        // associated types.
        self.fn_sig(trait_item_def_id).subst_identity().skip_binder().output().walk().any(|arg| {
            if let ty::GenericArgKind::Type(ty) = arg.unpack()
                && let ty::Alias(ty::Projection, data) = ty.kind()
                && self.def_kind(data.def_id) == DefKind::ImplTraitPlaceholder
            {
                true
            } else {
                false
            }
        })
    }
}

/// Yields the parent function's `LocalDefId` if `def_id` is an `impl Trait` definition.
pub fn is_impl_trait_defn(tcx: TyCtxt<'_>, def_id: DefId) -> Option<LocalDefId> {
    let def_id = def_id.as_local()?;
    if let Node::Item(item) = tcx.hir().get_by_def_id(def_id) {
        if let hir::ItemKind::OpaqueTy(ref opaque_ty) = item.kind {
            return match opaque_ty.origin {
                hir::OpaqueTyOrigin::FnReturn(parent) | hir::OpaqueTyOrigin::AsyncFn(parent) => {
                    Some(parent)
                }
                hir::OpaqueTyOrigin::TyAlias => None,
            };
        }
    }
    None
}

pub fn int_ty(ity: ast::IntTy) -> IntTy {
    match ity {
        ast::IntTy::Isize => IntTy::Isize,
        ast::IntTy::I8 => IntTy::I8,
        ast::IntTy::I16 => IntTy::I16,
        ast::IntTy::I32 => IntTy::I32,
        ast::IntTy::I64 => IntTy::I64,
        ast::IntTy::I128 => IntTy::I128,
    }
}

pub fn uint_ty(uty: ast::UintTy) -> UintTy {
    match uty {
        ast::UintTy::Usize => UintTy::Usize,
        ast::UintTy::U8 => UintTy::U8,
        ast::UintTy::U16 => UintTy::U16,
        ast::UintTy::U32 => UintTy::U32,
        ast::UintTy::U64 => UintTy::U64,
        ast::UintTy::U128 => UintTy::U128,
    }
}

pub fn float_ty(fty: ast::FloatTy) -> FloatTy {
    match fty {
        ast::FloatTy::F32 => FloatTy::F32,
        ast::FloatTy::F64 => FloatTy::F64,
    }
}

pub fn ast_int_ty(ity: IntTy) -> ast::IntTy {
    match ity {
        IntTy::Isize => ast::IntTy::Isize,
        IntTy::I8 => ast::IntTy::I8,
        IntTy::I16 => ast::IntTy::I16,
        IntTy::I32 => ast::IntTy::I32,
        IntTy::I64 => ast::IntTy::I64,
        IntTy::I128 => ast::IntTy::I128,
    }
}

pub fn ast_uint_ty(uty: UintTy) -> ast::UintTy {
    match uty {
        UintTy::Usize => ast::UintTy::Usize,
        UintTy::U8 => ast::UintTy::U8,
        UintTy::U16 => ast::UintTy::U16,
        UintTy::U32 => ast::UintTy::U32,
        UintTy::U64 => ast::UintTy::U64,
        UintTy::U128 => ast::UintTy::U128,
    }
}

pub fn provide(providers: &mut ty::query::Providers) {
    closure::provide(providers);
    context::provide(providers);
    erase_regions::provide(providers);
    inhabitedness::provide(providers);
    util::provide(providers);
    print::provide(providers);
    super::util::bug::provide(providers);
    super::middle::provide(providers);
    *providers = ty::query::Providers {
        trait_impls_of: trait_def::trait_impls_of_provider,
        incoherent_impls: trait_def::incoherent_impls_provider,
        const_param_default: consts::const_param_default,
        vtable_allocation: vtable::vtable_allocation_provider,
        ..*providers
    };
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    // tidy-alphabetical-start
    static_assert_size!(PredicateKind<'_>, 32);
    static_assert_size!(WithCachedTypeInfo<TyKind<'_>>, 56);
    // tidy-alphabetical-end
}
