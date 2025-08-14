use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::str;

use rustc_abi::{FIRST_VARIANT, ReprOptions, VariantIdx};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::intern::Interned;
use rustc_data_structures::stable_hasher::{HashStable, HashingControls, StableHasher};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, LangItem, find_attr};
use rustc_index::{IndexSlice, IndexVec};
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use rustc_query_system::ich::StableHashingContext;
use rustc_session::DataTypeKind;
use rustc_type_ir::solve::AdtDestructorKind;
use tracing::{debug, info, trace};

use super::{
    AsyncDestructor, Destructor, FieldDef, GenericPredicates, Ty, TyCtxt, VariantDef, VariantDiscr,
};
use crate::mir::interpret::ErrorHandled;
use crate::ty;
use crate::ty::util::{Discr, IntTypeExt};

#[derive(Clone, Copy, PartialEq, Eq, Hash, HashStable, TyEncodable, TyDecodable)]
pub struct AdtFlags(u16);
bitflags::bitflags! {
    impl AdtFlags: u16 {
        const NO_ADT_FLAGS        = 0;
        /// Indicates whether the ADT is an enum.
        const IS_ENUM             = 1 << 0;
        /// Indicates whether the ADT is a union.
        const IS_UNION            = 1 << 1;
        /// Indicates whether the ADT is a struct.
        const IS_STRUCT           = 1 << 2;
        /// Indicates whether the ADT is a struct and has a constructor.
        const HAS_CTOR            = 1 << 3;
        /// Indicates whether the type is `PhantomData`.
        const IS_PHANTOM_DATA     = 1 << 4;
        /// Indicates whether the type has a `#[fundamental]` attribute.
        const IS_FUNDAMENTAL      = 1 << 5;
        /// Indicates whether the type is `Box`.
        const IS_BOX              = 1 << 6;
        /// Indicates whether the type is `ManuallyDrop`.
        const IS_MANUALLY_DROP    = 1 << 7;
        /// Indicates whether the variant list of this ADT is `#[non_exhaustive]`.
        /// (i.e., this flag is never set unless this ADT is an enum).
        const IS_VARIANT_LIST_NON_EXHAUSTIVE = 1 << 8;
        /// Indicates whether the type is `UnsafeCell`.
        const IS_UNSAFE_CELL              = 1 << 9;
        /// Indicates whether the type is `UnsafePinned`.
        const IS_UNSAFE_PINNED              = 1 << 10;
    }
}
rustc_data_structures::external_bitflags_debug! { AdtFlags }

/// The definition of a user-defined type, e.g., a `struct`, `enum`, or `union`.
///
/// These are all interned (by `mk_adt_def`) into the global arena.
///
/// The initialism *ADT* stands for an [*algebraic data type (ADT)*][adt].
/// This is slightly wrong because `union`s are not ADTs.
/// Moreover, Rust only allows recursive data types through indirection.
///
/// [adt]: https://en.wikipedia.org/wiki/Algebraic_data_type
///
/// # Recursive types
///
/// It may seem impossible to represent recursive types using [`Ty`],
/// since [`TyKind::Adt`] includes [`AdtDef`], which includes its fields,
/// creating a cycle. However, `AdtDef` does not actually include the *types*
/// of its fields; it includes just their [`DefId`]s.
///
/// [`TyKind::Adt`]: ty::TyKind::Adt
///
/// For example, the following type:
///
/// ```
/// struct S { x: Box<S> }
/// ```
///
/// is essentially represented with [`Ty`] as the following pseudocode:
///
/// ```ignore (illustrative)
/// struct S { x }
/// ```
///
/// where `x` here represents the `DefId` of `S.x`. Then, the `DefId`
/// can be used with [`TyCtxt::type_of()`] to get the type of the field.
#[derive(TyEncodable, TyDecodable)]
pub struct AdtDefData {
    /// The `DefId` of the struct, enum or union item.
    pub did: DefId,
    /// Variants of the ADT. If this is a struct or union, then there will be a single variant.
    variants: IndexVec<VariantIdx, VariantDef>,
    /// Flags of the ADT (e.g., is this a struct? is this non-exhaustive?).
    flags: AdtFlags,
    /// Repr options provided by the user.
    repr: ReprOptions,
}

impl PartialEq for AdtDefData {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // There should be only one `AdtDefData` for each `def_id`, therefore
        // it is fine to implement `PartialEq` only based on `def_id`.
        //
        // Below, we exhaustively destructure `self` and `other` so that if the
        // definition of `AdtDefData` changes, a compile-error will be produced,
        // reminding us to revisit this assumption.

        let Self { did: self_def_id, variants: _, flags: _, repr: _ } = self;
        let Self { did: other_def_id, variants: _, flags: _, repr: _ } = other;

        let res = self_def_id == other_def_id;

        // Double check that implicit assumption detailed above.
        if cfg!(debug_assertions) && res {
            let deep = self.flags == other.flags
                && self.repr == other.repr
                && self.variants == other.variants;
            assert!(deep, "AdtDefData for the same def-id has differing data");
        }

        res
    }
}

impl Eq for AdtDefData {}

/// There should be only one AdtDef for each `did`, therefore
/// it is fine to implement `Hash` only based on `did`.
impl Hash for AdtDefData {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.did.hash(s)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for AdtDefData {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        thread_local! {
            static CACHE: RefCell<FxHashMap<(usize, HashingControls), Fingerprint>> = Default::default();
        }

        let hash: Fingerprint = CACHE.with(|cache| {
            let addr = self as *const AdtDefData as usize;
            let hashing_controls = hcx.hashing_controls();
            *cache.borrow_mut().entry((addr, hashing_controls)).or_insert_with(|| {
                let ty::AdtDefData { did, ref variants, ref flags, ref repr } = *self;

                let mut hasher = StableHasher::new();
                did.hash_stable(hcx, &mut hasher);
                variants.hash_stable(hcx, &mut hasher);
                flags.hash_stable(hcx, &mut hasher);
                repr.hash_stable(hcx, &mut hasher);

                hasher.finish()
            })
        });

        hash.hash_stable(hcx, hasher);
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct AdtDef<'tcx>(pub Interned<'tcx, AdtDefData>);

impl<'tcx> AdtDef<'tcx> {
    #[inline]
    pub fn did(self) -> DefId {
        self.0.0.did
    }

    #[inline]
    pub fn variants(self) -> &'tcx IndexSlice<VariantIdx, VariantDef> {
        &self.0.0.variants
    }

    #[inline]
    pub fn variant(self, idx: VariantIdx) -> &'tcx VariantDef {
        &self.0.0.variants[idx]
    }

    #[inline]
    pub fn flags(self) -> AdtFlags {
        self.0.0.flags
    }

    #[inline]
    pub fn repr(self) -> ReprOptions {
        self.0.0.repr
    }
}

impl<'tcx> rustc_type_ir::inherent::AdtDef<TyCtxt<'tcx>> for AdtDef<'tcx> {
    fn def_id(self) -> DefId {
        self.did()
    }

    fn is_struct(self) -> bool {
        self.is_struct()
    }

    fn struct_tail_ty(self, interner: TyCtxt<'tcx>) -> Option<ty::EarlyBinder<'tcx, Ty<'tcx>>> {
        Some(interner.type_of(self.non_enum_variant().tail_opt()?.did))
    }

    fn is_phantom_data(self) -> bool {
        self.is_phantom_data()
    }

    fn is_manually_drop(self) -> bool {
        self.is_manually_drop()
    }

    fn all_field_tys(
        self,
        tcx: TyCtxt<'tcx>,
    ) -> ty::EarlyBinder<'tcx, impl IntoIterator<Item = Ty<'tcx>>> {
        ty::EarlyBinder::bind(
            self.all_fields().map(move |field| tcx.type_of(field.did).skip_binder()),
        )
    }

    fn sizedness_constraint(
        self,
        tcx: TyCtxt<'tcx>,
        sizedness: ty::SizedTraitKind,
    ) -> Option<ty::EarlyBinder<'tcx, Ty<'tcx>>> {
        self.sizedness_constraint(tcx, sizedness)
    }

    fn is_fundamental(self) -> bool {
        self.is_fundamental()
    }

    fn destructor(self, tcx: TyCtxt<'tcx>) -> Option<AdtDestructorKind> {
        Some(match tcx.constness(self.destructor(tcx)?.did) {
            hir::Constness::Const => AdtDestructorKind::Const,
            hir::Constness::NotConst => AdtDestructorKind::NotConst,
        })
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, HashStable, TyEncodable, TyDecodable)]
pub enum AdtKind {
    Struct,
    Union,
    Enum,
}

impl From<AdtKind> for DataTypeKind {
    fn from(val: AdtKind) -> Self {
        match val {
            AdtKind::Struct => DataTypeKind::Struct,
            AdtKind::Union => DataTypeKind::Union,
            AdtKind::Enum => DataTypeKind::Enum,
        }
    }
}

impl AdtDefData {
    /// Creates a new `AdtDefData`.
    pub(super) fn new(
        tcx: TyCtxt<'_>,
        did: DefId,
        kind: AdtKind,
        variants: IndexVec<VariantIdx, VariantDef>,
        repr: ReprOptions,
    ) -> Self {
        debug!("AdtDef::new({:?}, {:?}, {:?}, {:?})", did, kind, variants, repr);
        let mut flags = AdtFlags::NO_ADT_FLAGS;

        if kind == AdtKind::Enum
            && find_attr!(tcx.get_all_attrs(did), AttributeKind::NonExhaustive(..))
        {
            debug!("found non-exhaustive variant list for {:?}", did);
            flags = flags | AdtFlags::IS_VARIANT_LIST_NON_EXHAUSTIVE;
        }

        flags |= match kind {
            AdtKind::Enum => AdtFlags::IS_ENUM,
            AdtKind::Union => AdtFlags::IS_UNION,
            AdtKind::Struct => AdtFlags::IS_STRUCT,
        };

        if kind == AdtKind::Struct && variants[FIRST_VARIANT].ctor.is_some() {
            flags |= AdtFlags::HAS_CTOR;
        }

        if find_attr!(tcx.get_all_attrs(did), AttributeKind::Fundamental) {
            flags |= AdtFlags::IS_FUNDAMENTAL;
        }
        if tcx.is_lang_item(did, LangItem::PhantomData) {
            flags |= AdtFlags::IS_PHANTOM_DATA;
        }
        if tcx.is_lang_item(did, LangItem::OwnedBox) {
            flags |= AdtFlags::IS_BOX;
        }
        if tcx.is_lang_item(did, LangItem::ManuallyDrop) {
            flags |= AdtFlags::IS_MANUALLY_DROP;
        }
        if tcx.is_lang_item(did, LangItem::UnsafeCell) {
            flags |= AdtFlags::IS_UNSAFE_CELL;
        }
        if tcx.is_lang_item(did, LangItem::UnsafePinned) {
            flags |= AdtFlags::IS_UNSAFE_PINNED;
        }

        AdtDefData { did, variants, flags, repr }
    }
}

impl<'tcx> AdtDef<'tcx> {
    /// Returns `true` if this is a struct.
    #[inline]
    pub fn is_struct(self) -> bool {
        self.flags().contains(AdtFlags::IS_STRUCT)
    }

    /// Returns `true` if this is a union.
    #[inline]
    pub fn is_union(self) -> bool {
        self.flags().contains(AdtFlags::IS_UNION)
    }

    /// Returns `true` if this is an enum.
    #[inline]
    pub fn is_enum(self) -> bool {
        self.flags().contains(AdtFlags::IS_ENUM)
    }

    /// Returns `true` if the variant list of this ADT is `#[non_exhaustive]`.
    ///
    /// Note that this function will return `true` even if the ADT has been
    /// defined in the crate currently being compiled. If that's not what you
    /// want, see [`Self::variant_list_has_applicable_non_exhaustive`].
    #[inline]
    pub fn is_variant_list_non_exhaustive(self) -> bool {
        self.flags().contains(AdtFlags::IS_VARIANT_LIST_NON_EXHAUSTIVE)
    }

    /// Returns `true` if the variant list of this ADT is `#[non_exhaustive]`
    /// and has been defined in another crate.
    #[inline]
    pub fn variant_list_has_applicable_non_exhaustive(self) -> bool {
        self.is_variant_list_non_exhaustive() && !self.did().is_local()
    }

    /// Returns the kind of the ADT.
    #[inline]
    pub fn adt_kind(self) -> AdtKind {
        if self.is_enum() {
            AdtKind::Enum
        } else if self.is_union() {
            AdtKind::Union
        } else {
            AdtKind::Struct
        }
    }

    /// Returns a description of this abstract data type.
    pub fn descr(self) -> &'static str {
        match self.adt_kind() {
            AdtKind::Struct => "struct",
            AdtKind::Union => "union",
            AdtKind::Enum => "enum",
        }
    }

    /// Returns a description of a variant of this abstract data type.
    #[inline]
    pub fn variant_descr(self) -> &'static str {
        match self.adt_kind() {
            AdtKind::Struct => "struct",
            AdtKind::Union => "union",
            AdtKind::Enum => "variant",
        }
    }

    /// If this function returns `true`, it implies that `is_struct` must return `true`.
    #[inline]
    pub fn has_ctor(self) -> bool {
        self.flags().contains(AdtFlags::HAS_CTOR)
    }

    /// Returns `true` if this type is `#[fundamental]` for the purposes
    /// of coherence checking.
    #[inline]
    pub fn is_fundamental(self) -> bool {
        self.flags().contains(AdtFlags::IS_FUNDAMENTAL)
    }

    /// Returns `true` if this is `PhantomData<T>`.
    #[inline]
    pub fn is_phantom_data(self) -> bool {
        self.flags().contains(AdtFlags::IS_PHANTOM_DATA)
    }

    /// Returns `true` if this is `Box<T>`.
    #[inline]
    pub fn is_box(self) -> bool {
        self.flags().contains(AdtFlags::IS_BOX)
    }

    /// Returns `true` if this is `UnsafeCell<T>`.
    #[inline]
    pub fn is_unsafe_cell(self) -> bool {
        self.flags().contains(AdtFlags::IS_UNSAFE_CELL)
    }

    /// Returns `true` if this is `UnsafePinned<T>`.
    #[inline]
    pub fn is_unsafe_pinned(self) -> bool {
        self.flags().contains(AdtFlags::IS_UNSAFE_PINNED)
    }

    /// Returns `true` if this is `ManuallyDrop<T>`.
    #[inline]
    pub fn is_manually_drop(self) -> bool {
        self.flags().contains(AdtFlags::IS_MANUALLY_DROP)
    }

    /// Returns `true` if this type has a destructor.
    pub fn has_dtor(self, tcx: TyCtxt<'tcx>) -> bool {
        self.destructor(tcx).is_some()
    }

    /// Asserts this is a struct or union and returns its unique variant.
    pub fn non_enum_variant(self) -> &'tcx VariantDef {
        assert!(self.is_struct() || self.is_union());
        self.variant(FIRST_VARIANT)
    }

    #[inline]
    pub fn predicates(self, tcx: TyCtxt<'tcx>) -> GenericPredicates<'tcx> {
        tcx.predicates_of(self.did())
    }

    /// Returns an iterator over all fields contained
    /// by this ADT (nested unnamed fields are not expanded).
    #[inline]
    pub fn all_fields(self) -> impl Iterator<Item = &'tcx FieldDef> + Clone {
        self.variants().iter().flat_map(|v| v.fields.iter())
    }

    /// Whether the ADT lacks fields. Note that this includes uninhabited enums,
    /// e.g., `enum Void {}` is considered payload free as well.
    pub fn is_payloadfree(self) -> bool {
        // Treat the ADT as not payload-free if arbitrary_enum_discriminant is used (#88621).
        // This would disallow the following kind of enum from being casted into integer.
        // ```
        // enum Enum {
        //    Foo() = 1,
        //    Bar{} = 2,
        //    Baz = 3,
        // }
        // ```
        if self.variants().iter().any(|v| {
            matches!(v.discr, VariantDiscr::Explicit(_)) && v.ctor_kind() != Some(CtorKind::Const)
        }) {
            return false;
        }
        self.variants().iter().all(|v| v.fields.is_empty())
    }

    /// Return a `VariantDef` given a variant id.
    pub fn variant_with_id(self, vid: DefId) -> &'tcx VariantDef {
        self.variants().iter().find(|v| v.def_id == vid).expect("variant_with_id: unknown variant")
    }

    /// Return a `VariantDef` given a constructor id.
    pub fn variant_with_ctor_id(self, cid: DefId) -> &'tcx VariantDef {
        self.variants()
            .iter()
            .find(|v| v.ctor_def_id() == Some(cid))
            .expect("variant_with_ctor_id: unknown variant")
    }

    /// Return the index of `VariantDef` given a variant id.
    #[inline]
    pub fn variant_index_with_id(self, vid: DefId) -> VariantIdx {
        self.variants()
            .iter_enumerated()
            .find(|(_, v)| v.def_id == vid)
            .expect("variant_index_with_id: unknown variant")
            .0
    }

    /// Return the index of `VariantDef` given a constructor id.
    pub fn variant_index_with_ctor_id(self, cid: DefId) -> VariantIdx {
        self.variants()
            .iter_enumerated()
            .find(|(_, v)| v.ctor_def_id() == Some(cid))
            .expect("variant_index_with_ctor_id: unknown variant")
            .0
    }

    pub fn variant_of_res(self, res: Res) -> &'tcx VariantDef {
        match res {
            Res::Def(DefKind::Variant, vid) => self.variant_with_id(vid),
            Res::Def(DefKind::Ctor(..), cid) => self.variant_with_ctor_id(cid),
            Res::Def(DefKind::Struct, _)
            | Res::Def(DefKind::Union, _)
            | Res::Def(DefKind::TyAlias, _)
            | Res::Def(DefKind::AssocTy, _)
            | Res::SelfTyParam { .. }
            | Res::SelfTyAlias { .. }
            | Res::SelfCtor(..) => self.non_enum_variant(),
            _ => bug!("unexpected res {:?} in variant_of_res", res),
        }
    }

    #[inline]
    pub fn eval_explicit_discr(
        self,
        tcx: TyCtxt<'tcx>,
        expr_did: DefId,
    ) -> Result<Discr<'tcx>, ErrorGuaranteed> {
        assert!(self.is_enum());

        let repr_type = self.repr().discr_type();
        match tcx.const_eval_poly(expr_did) {
            Ok(val) => {
                let typing_env = ty::TypingEnv::post_analysis(tcx, expr_did);
                let ty = repr_type.to_ty(tcx);
                if let Some(b) = val.try_to_bits_for_ty(tcx, typing_env, ty) {
                    trace!("discriminants: {} ({:?})", b, repr_type);
                    Ok(Discr { val: b, ty })
                } else {
                    info!("invalid enum discriminant: {:#?}", val);
                    let guar = tcx.dcx().emit_err(crate::error::ConstEvalNonIntError {
                        span: tcx.def_span(expr_did),
                    });
                    Err(guar)
                }
            }
            Err(err) => {
                let guar = match err {
                    ErrorHandled::Reported(info, _) => info.into(),
                    ErrorHandled::TooGeneric(..) => tcx.dcx().span_delayed_bug(
                        tcx.def_span(expr_did),
                        "enum discriminant depends on generics",
                    ),
                };
                Err(guar)
            }
        }
    }

    #[inline]
    pub fn discriminants(
        self,
        tcx: TyCtxt<'tcx>,
    ) -> impl Iterator<Item = (VariantIdx, Discr<'tcx>)> {
        assert!(self.is_enum());
        let repr_type = self.repr().discr_type();
        let initial = repr_type.initial_discriminant(tcx);
        let mut prev_discr = None::<Discr<'tcx>>;
        self.variants().iter_enumerated().map(move |(i, v)| {
            let mut discr = prev_discr.map_or(initial, |d| d.wrap_incr(tcx));
            if let VariantDiscr::Explicit(expr_did) = v.discr
                && let Ok(new_discr) = self.eval_explicit_discr(tcx, expr_did)
            {
                discr = new_discr;
            }
            prev_discr = Some(discr);

            (i, discr)
        })
    }

    #[inline]
    pub fn variant_range(self) -> Range<VariantIdx> {
        FIRST_VARIANT..self.variants().next_index()
    }

    /// Computes the discriminant value used by a specific variant.
    /// Unlike `discriminants`, this is (amortized) constant-time,
    /// only doing at most one query for evaluating an explicit
    /// discriminant (the last one before the requested variant),
    /// assuming there are no constant-evaluation errors there.
    #[inline]
    pub fn discriminant_for_variant(
        self,
        tcx: TyCtxt<'tcx>,
        variant_index: VariantIdx,
    ) -> Discr<'tcx> {
        assert!(self.is_enum());
        let (val, offset) = self.discriminant_def_for_variant(variant_index);
        let explicit_value = if let Some(expr_did) = val
            && let Ok(val) = self.eval_explicit_discr(tcx, expr_did)
        {
            val
        } else {
            self.repr().discr_type().initial_discriminant(tcx)
        };
        explicit_value.checked_add(tcx, offset as u128).0
    }

    /// Yields a `DefId` for the discriminant and an offset to add to it
    /// Alternatively, if there is no explicit discriminant, returns the
    /// inferred discriminant directly.
    pub fn discriminant_def_for_variant(self, variant_index: VariantIdx) -> (Option<DefId>, u32) {
        assert!(!self.variants().is_empty());
        let mut explicit_index = variant_index.as_u32();
        let expr_did;
        loop {
            match self.variant(VariantIdx::from_u32(explicit_index)).discr {
                ty::VariantDiscr::Relative(0) => {
                    expr_did = None;
                    break;
                }
                ty::VariantDiscr::Relative(distance) => {
                    explicit_index -= distance;
                }
                ty::VariantDiscr::Explicit(did) => {
                    expr_did = Some(did);
                    break;
                }
            }
        }
        (expr_did, variant_index.as_u32() - explicit_index)
    }

    pub fn destructor(self, tcx: TyCtxt<'tcx>) -> Option<Destructor> {
        tcx.adt_destructor(self.did())
    }

    // FIXME: consider combining this method with `AdtDef::destructor` and removing
    // this version
    pub fn async_destructor(self, tcx: TyCtxt<'tcx>) -> Option<AsyncDestructor> {
        tcx.adt_async_destructor(self.did())
    }

    /// If this ADT is a struct, returns a type such that `Self: {Meta,Pointee,}Sized` if and only
    /// if that type is `{Meta,Pointee,}Sized`, or `None` if this ADT is always
    /// `{Meta,Pointee,}Sized`.
    pub fn sizedness_constraint(
        self,
        tcx: TyCtxt<'tcx>,
        sizedness: ty::SizedTraitKind,
    ) -> Option<ty::EarlyBinder<'tcx, Ty<'tcx>>> {
        if self.is_struct() { tcx.adt_sizedness_constraint((self.did(), sizedness)) } else { None }
    }
}

#[derive(Clone, Copy, Debug, HashStable)]
pub enum Representability {
    Representable,
    Infinite(ErrorGuaranteed),
}
