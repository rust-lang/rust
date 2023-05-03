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

pub use self::fold::{FallibleTypeFolder, TypeFoldable, TypeFolder, TypeSuperFoldable};
pub use self::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};
pub use self::AssocItemContainer::*;
pub use self::BorrowKind::*;
pub use self::IntVarValue::*;
pub use self::Variance::*;
use crate::error::TypeMismatchReason;
use crate::mir::{Body, GeneratorLayout};
use crate::ty;
use crate::ty::fast_reject::SimplifiedType;
use crate::ty::util::Discr;
pub use adt::*;
pub use assoc::*;
pub use generics::*;
use rustc_ast as ast;
use rustc_attr as attr;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::intern::Interned;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, LocalDefId, LocalDefIdMap};
use rustc_hir::Node;
use rustc_macros::HashStable;
use rustc_serialize::{Decodable, Encodable};
pub use rustc_session::lint::RegisteredTools;
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::{ExpnKind, Span};
use rustc_target::abi::{Align, FieldIdx, Integer, IntegerType, VariantIdx};
pub use rustc_target::abi::{ReprFlags, ReprOptions};
use rustc_type_ir::WithCachedTypeInfo;
pub use subst::*;
pub use vtable::*;

use std::fmt::Debug;
use std::hash::Hash;

pub use crate::ty::diagnostics::*;
pub use rustc_type_ir::AliasKind::*;
pub use rustc_type_ir::DynKind::*;
pub use rustc_type_ir::InferTy::*;
pub use rustc_type_ir::RegionKind::*;
pub use rustc_type_ir::TyKind::*;
pub use rustc_type_ir::*;

pub use self::binding::BindingMode;
pub use self::binding::BindingMode::*;
pub use self::closure::{
    is_ancestor_or_same_capture, place_to_string_for_capture, BorrowKind, CaptureInfo,
    CapturedPlace, ClosureKind, ClosureTypeInfo, MinCaptureInformationMap, MinCaptureList,
    RootVariableMinCaptureList, UpvarCapture, UpvarCaptureMap, UpvarId, UpvarListMap, UpvarPath,
    CAPTURE_STRUCT_LOCAL,
};
pub use self::consts::{
    Const, ConstData, ConstInt, ConstKind, Expr, InferConst, ScalarInt, UnevaluatedConst, ValTree,
};
pub use self::context::{
    tls, CtxtInterners, DeducedParamAttrs, FreeRegionInfo, GlobalCtxt, Lift, TyCtxt, TyCtxtFeed,
};
pub use self::instance::{Instance, InstanceDef, ShortInstance, UnusedGenericParams};
pub use self::list::List;
pub use self::parameterized::ParameterizedOverTcx;
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
pub use self::trait_def::TraitDef;
pub use self::typeck_results::{
    CanonicalUserType, CanonicalUserTypeAnnotation, CanonicalUserTypeAnnotations,
    GeneratorDiagnosticData, GeneratorInteriorTypeCause, TypeckResults, UserType,
    UserTypeAnnotationIndex,
};

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
mod assoc;
mod closure;
mod consts;
mod context;
mod diagnostics;
mod erase_regions;
mod generics;
mod impls_ty;
mod instance;
mod list;
mod opaque_types;
mod parameterized;
mod rvalue_scopes;
mod structural_impls;
mod sty;
mod typeck_results;

mod alias_relation_direction;
mod bound_constness;
mod field_def;
mod impl_polarity;
mod opaque_hidden_type;
mod param_env;
mod placeholder;
mod predicate;
mod resolver_outputs;
mod symbol_name;
mod term;
mod ty_; // FIXME: rename to `ty` once we don't import `crate::ty` here
mod variant_def;
mod visibility;

pub use alias_relation_direction::AliasRelationDirection;
pub use bound_constness::BoundConstness;
pub use field_def::FieldDef;
pub use impl_polarity::ImplPolarity;
pub use opaque_hidden_type::OpaqueHiddenType;
pub use param_env::{ParamEnv, ParamEnvAnd};
pub use placeholder::{Placeholder, PlaceholderConst, PlaceholderRegion, PlaceholderType};
pub use predicate::{
    CoercePredicate, InstantiatedPredicates, OutlivesPredicate, PolyCoercePredicate,
    PolyProjectionPredicate, PolyRegionOutlivesPredicate, PolySubtypePredicate, PolyTraitPredicate,
    PolyTypeOutlivesPredicate, Predicate, PredicateKind, ProjectionPredicate,
    RegionOutlivesPredicate, SubtypePredicate, ToPredicate, TraitPredicate, TypeOutlivesPredicate,
};
pub use resolver_outputs::{ResolverAstLowering, ResolverGlobalCtxt, ResolverOutputs};
pub use symbol_name::SymbolName;
pub use term::{Term, TermKind};
pub use ty_::Ty;
pub use variant_def::VariantDef;
pub use visibility::Visibility;

#[derive(Clone, Copy, Debug)]
pub struct MainDefinition {
    pub res: Res<ast::NodeId>,
    pub is_import: bool,
    pub span: Span,
}

impl MainDefinition {
    pub fn opt_fn_def_id(self) -> Option<DefId> {
        if let Res::Def(DefKind::Fn, def_id) = self.res { Some(def_id) } else { None }
    }
}

/// The "header" of an impl is everything outside the body: a Self type, a trait
/// ref (in the case of a trait impl), and a set of predicates (from the
/// bounds / where-clauses).
#[derive(Clone, Debug, TypeFoldable, TypeVisitable)]
pub struct ImplHeader<'tcx> {
    pub impl_def_id: DefId,
    pub self_ty: Ty<'tcx>,
    pub trait_ref: Option<TraitRef<'tcx>>,
    pub predicates: Vec<Predicate<'tcx>>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, TypeFoldable, TypeVisitable)]
pub enum ImplSubject<'tcx> {
    Trait(TraitRef<'tcx>),
    Inherent(Ty<'tcx>),
}

#[derive(Clone, Debug, PartialEq, Eq, Copy, Hash, TyEncodable, TyDecodable, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ClosureSizeProfileData<'tcx> {
    /// Tuple containing the types of closure captures before the feature `capture_disjoint_fields`
    pub before_feature_tys: Ty<'tcx>,
    /// Tuple containing the types of closure captures after the feature `capture_disjoint_fields`
    pub after_feature_tys: Ty<'tcx>,
}

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
/// The crate variances map is computed during typeck and contains the
/// variance of every item in the local crate. You should not use it
/// directly, because to do so will make your pass dependent on the
/// HIR of every item in the local crate. Instead, use
/// `tcx.variances_of()` to get the variance for a *particular*
/// item.
#[derive(HashStable, Debug)]
pub struct CrateVariancesMap<'tcx> {
    /// For each item with generics, maps to a vector of the variance
    /// of its generics. If an item has no generics, it will have no
    /// entry.
    pub variances: DefIdMap<&'tcx [ty::Variance]>,
}

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct CReaderCacheKey {
    pub cnum: Option<CrateNum>,
    pub pos: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
/// A clause is something that can appear in where bounds or be inferred
/// by implied bounds.
pub enum Clause<'tcx> {
    /// Corresponds to `where Foo: Bar<A, B, C>`. `Foo` here would be
    /// the `Self` type of the trait reference and `A`, `B`, and `C`
    /// would be the type parameters.
    Trait(TraitPredicate<'tcx>),

    /// `where 'a: 'b`
    RegionOutlives(RegionOutlivesPredicate<'tcx>),

    /// `where T: 'a`
    TypeOutlives(TypeOutlivesPredicate<'tcx>),

    /// `where <T as TraitRef>::Name == X`, approximately.
    /// See the `ProjectionPredicate` struct for details.
    Projection(ProjectionPredicate<'tcx>),

    /// Ensures that a const generic argument to a parameter `const N: u8`
    /// is of type `u8`.
    ConstArgHasType(Const<'tcx>, Ty<'tcx>),
}

/// The crate outlives map is computed during typeck and contains the
/// outlives of every item in the local crate. You should not use it
/// directly, because to do so will make your pass dependent on the
/// HIR of every item in the local crate. Instead, use
/// `tcx.inferred_outlives_of()` to get the outlives for a *particular*
/// item.
#[derive(HashStable, Debug)]
pub struct CratePredicatesMap<'tcx> {
    /// For each struct with outlive bounds, maps to a vector of the
    /// predicate of its outlive bounds. If an item has no outlives
    /// bounds, it will have no entry.
    pub predicates: FxHashMap<DefId, &'tcx [(Clause<'tcx>, Span)]>,
}

const TAG_MASK: usize = 0b11;
const TYPE_TAG: usize = 0b00;
const CONST_TAG: usize = 0b01;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ParamTerm {
    Ty(ParamTy),
    Const(ParamConst),
}

impl ParamTerm {
    pub fn index(self) -> usize {
        match self {
            ParamTerm::Ty(ty) => ty.index as usize,
            ParamTerm::Const(ct) => ct.index as usize,
        }
    }
}
pub trait ToPolyTraitRef<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx>;
}

impl<'tcx> ToPolyTraitRef<'tcx> for PolyTraitPredicate<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        self.map_bound_ref(|trait_pred| trait_pred.trait_ref)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable, TyEncodable, TyDecodable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct OpaqueTypeKey<'tcx> {
    pub def_id: LocalDefId,
    pub substs: SubstsRef<'tcx>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable)]
#[derive(TyEncodable, TyDecodable, PartialOrd, Ord)]
pub struct BoundConst<'tcx> {
    pub var: BoundVar,
    pub ty: Ty<'tcx>,
}

#[derive(Copy, Clone, Debug, HashStable, Encodable, Decodable)]
pub struct Destructor {
    /// The `DefId` of the destructor method
    pub did: DefId,
    /// The constness of the destructor method
    pub constness: hir::Constness,
}

bitflags! {
    #[derive(HashStable, TyEncodable, TyDecodable)]
    pub struct VariantFlags: u8 {
        const NO_VARIANT_FLAGS        = 0;
        /// Indicates whether the field list of this variant is `#[non_exhaustive]`.
        const IS_FIELD_LIST_NON_EXHAUSTIVE = 1 << 0;
        /// Indicates whether this variant was obtained as part of recovering from
        /// a syntactic error. May be incomplete or bogus.
        const IS_RECOVERED = 1 << 1;
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, TyEncodable, TyDecodable, HashStable)]
pub enum VariantDiscr {
    /// Explicit value for this variant, i.e., `X = 123`.
    /// The `DefId` corresponds to the embedded constant.
    Explicit(DefId),

    /// The previous variant's discriminant plus one.
    /// For efficiency reasons, the distance from the
    /// last `Explicit` discriminant is being stored,
    /// or `0` for the first variant, if it has none.
    Relative(u32),
}

#[derive(Debug, PartialEq, Eq)]
pub enum ImplOverlapKind {
    /// These impls are always allowed to overlap.
    Permitted {
        /// Whether or not the impl is permitted due to the trait being a `#[marker]` trait
        marker: bool,
    },
    /// These impls are allowed to overlap, but that raises
    /// an issue #33140 future-compatibility warning.
    ///
    /// Some background: in Rust 1.0, the trait-object types `Send + Sync` (today's
    /// `dyn Send + Sync`) and `Sync + Send` (now `dyn Sync + Send`) were different.
    ///
    /// The widely-used version 0.1.0 of the crate `traitobject` had accidentally relied
    /// that difference, making what reduces to the following set of impls:
    ///
    /// ```compile_fail,(E0119)
    /// trait Trait {}
    /// impl Trait for dyn Send + Sync {}
    /// impl Trait for dyn Sync + Send {}
    /// ```
    ///
    /// Obviously, once we made these types be identical, that code causes a coherence
    /// error and a fairly big headache for us. However, luckily for us, the trait
    /// `Trait` used in this case is basically a marker trait, and therefore having
    /// overlapping impls for it is sound.
    ///
    /// To handle this, we basically regard the trait as a marker trait, with an additional
    /// future-compatibility warning. To avoid accidentally "stabilizing" this feature,
    /// it has the following restrictions:
    ///
    /// 1. The trait must indeed be a marker-like trait (i.e., no items), and must be
    /// positive impls.
    /// 2. The trait-ref of both impls must be equal.
    /// 3. The trait-ref of both impls must be a trait object type consisting only of
    /// marker traits.
    /// 4. Neither of the impls can have any where-clauses.
    ///
    /// Once `traitobject` 0.1.0 is no longer an active concern, this hack can be removed.
    Issue33140,
}

/// Useful source information about where a desugared associated type for an
/// RPITIT originated from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encodable, Decodable, HashStable)]
pub enum ImplTraitInTraitData {
    Trait { fn_def_id: DefId, opaque_def_id: DefId },
    Impl { fn_def_id: DefId },
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

/// A map for the local crate mapping each type to a vector of its
/// inherent impls. This is not meant to be used outside of coherence;
/// rather, you should request the vector for a specific type via
/// `tcx.inherent_impls(def_id)` so as to minimize your dependencies
/// (constructing this map requires touching the entire crate).
#[derive(Clone, Debug, Default, HashStable)]
pub struct CrateInherentImpls {
    pub inherent_impls: LocalDefIdMap<Vec<DefId>>,
    pub incoherent_impls: FxHashMap<SimplifiedType, Vec<LocalDefId>>,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct InferVarInfo {
    /// This is true if we identified that this Ty (`?T`) is found in a `?T: Foo`
    /// obligation, where:
    ///
    ///  * `Foo` is not `Sized`
    ///  * `(): Foo` may be satisfied
    pub self_in_trait: bool,
    /// This is true if we identified that this Ty (`?T`) is found in a `<_ as
    /// _>::AssocType = ?T`
    pub output: bool,
}

/// The constituent parts of a type level constant of kind ADT or array.
#[derive(Copy, Clone, Debug, HashStable)]
pub struct DestructuredConst<'tcx> {
    pub variant: Option<VariantIdx>,
    pub fields: &'tcx [ty::Const<'tcx>],
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
