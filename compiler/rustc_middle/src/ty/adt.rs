use crate::mir::interpret::ErrorHandled;
use crate::ty;
use crate::ty::util::{Discr, IntTypeExt};
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_errors::ErrorReported;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_index::vec::{Idx, IndexVec};
use rustc_query_system::ich::StableHashingContext;
use rustc_session::DataTypeKind;
use rustc_span::symbol::sym;
use rustc_target::abi::VariantIdx;

use std::cell::RefCell;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::str;

use super::{
    Destructor, FieldDef, GenericPredicates, ReprOptions, Ty, TyCtxt, VariantDef, VariantDiscr,
};

#[derive(Clone, HashStable, Debug)]
pub struct AdtSizedConstraint<'tcx>(pub &'tcx [Ty<'tcx>]);

bitflags! {
    #[derive(HashStable, TyEncodable, TyDecodable)]
    pub struct AdtFlags: u32 {
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
    }
}

/// The definition of a user-defined type, e.g., a `struct`, `enum`, or `union`.
///
/// These are all interned (by `alloc_adt_def`) into the global arena.
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
/// ```
/// struct S { x }
/// ```
///
/// where `x` here represents the `DefId` of `S.x`. Then, the `DefId`
/// can be used with [`TyCtxt::type_of()`] to get the type of the field.
#[derive(TyEncodable, TyDecodable)]
pub struct AdtDef {
    /// The `DefId` of the struct, enum or union item.
    pub did: DefId,
    /// Variants of the ADT. If this is a struct or union, then there will be a single variant.
    pub variants: IndexVec<VariantIdx, VariantDef>,
    /// Flags of the ADT (e.g., is this a struct? is this non-exhaustive?).
    flags: AdtFlags,
    /// Repr options provided by the user.
    pub repr: ReprOptions,
}

impl PartialOrd for AdtDef {
    fn partial_cmp(&self, other: &AdtDef) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

/// There should be only one AdtDef for each `did`, therefore
/// it is fine to implement `Ord` only based on `did`.
impl Ord for AdtDef {
    fn cmp(&self, other: &AdtDef) -> Ordering {
        self.did.cmp(&other.did)
    }
}

/// There should be only one AdtDef for each `did`, therefore
/// it is fine to implement `PartialEq` only based on `did`.
impl PartialEq for AdtDef {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.did == other.did
    }
}

impl Eq for AdtDef {}

/// There should be only one AdtDef for each `did`, therefore
/// it is fine to implement `Hash` only based on `did`.
impl Hash for AdtDef {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.did.hash(s)
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for AdtDef {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        thread_local! {
            static CACHE: RefCell<FxHashMap<usize, Fingerprint>> = Default::default();
        }

        let hash: Fingerprint = CACHE.with(|cache| {
            let addr = self as *const AdtDef as usize;
            *cache.borrow_mut().entry(addr).or_insert_with(|| {
                let ty::AdtDef { did, ref variants, ref flags, ref repr } = *self;

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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, TyEncodable, TyDecodable)]
pub enum AdtKind {
    Struct,
    Union,
    Enum,
}

impl Into<DataTypeKind> for AdtKind {
    fn into(self) -> DataTypeKind {
        match self {
            AdtKind::Struct => DataTypeKind::Struct,
            AdtKind::Union => DataTypeKind::Union,
            AdtKind::Enum => DataTypeKind::Enum,
        }
    }
}

impl<'tcx> AdtDef {
    /// Creates a new `AdtDef`.
    pub(super) fn new(
        tcx: TyCtxt<'_>,
        did: DefId,
        kind: AdtKind,
        variants: IndexVec<VariantIdx, VariantDef>,
        repr: ReprOptions,
    ) -> Self {
        debug!("AdtDef::new({:?}, {:?}, {:?}, {:?})", did, kind, variants, repr);
        let mut flags = AdtFlags::NO_ADT_FLAGS;

        if kind == AdtKind::Enum && tcx.has_attr(did, sym::non_exhaustive) {
            debug!("found non-exhaustive variant list for {:?}", did);
            flags = flags | AdtFlags::IS_VARIANT_LIST_NON_EXHAUSTIVE;
        }

        flags |= match kind {
            AdtKind::Enum => AdtFlags::IS_ENUM,
            AdtKind::Union => AdtFlags::IS_UNION,
            AdtKind::Struct => AdtFlags::IS_STRUCT,
        };

        if kind == AdtKind::Struct && variants[VariantIdx::new(0)].ctor_def_id.is_some() {
            flags |= AdtFlags::HAS_CTOR;
        }

        let attrs = tcx.get_attrs(did);
        if tcx.sess.contains_name(&attrs, sym::fundamental) {
            flags |= AdtFlags::IS_FUNDAMENTAL;
        }
        if Some(did) == tcx.lang_items().phantom_data() {
            flags |= AdtFlags::IS_PHANTOM_DATA;
        }
        if Some(did) == tcx.lang_items().owned_box() {
            flags |= AdtFlags::IS_BOX;
        }
        if Some(did) == tcx.lang_items().manually_drop() {
            flags |= AdtFlags::IS_MANUALLY_DROP;
        }

        AdtDef { did, variants, flags, repr }
    }

    /// Returns `true` if this is a struct.
    #[inline]
    pub fn is_struct(&self) -> bool {
        self.flags.contains(AdtFlags::IS_STRUCT)
    }

    /// Returns `true` if this is a union.
    #[inline]
    pub fn is_union(&self) -> bool {
        self.flags.contains(AdtFlags::IS_UNION)
    }

    /// Returns `true` if this is an enum.
    #[inline]
    pub fn is_enum(&self) -> bool {
        self.flags.contains(AdtFlags::IS_ENUM)
    }

    /// Returns `true` if the variant list of this ADT is `#[non_exhaustive]`.
    #[inline]
    pub fn is_variant_list_non_exhaustive(&self) -> bool {
        self.flags.contains(AdtFlags::IS_VARIANT_LIST_NON_EXHAUSTIVE)
    }

    /// Returns the kind of the ADT.
    #[inline]
    pub fn adt_kind(&self) -> AdtKind {
        if self.is_enum() {
            AdtKind::Enum
        } else if self.is_union() {
            AdtKind::Union
        } else {
            AdtKind::Struct
        }
    }

    /// Returns a description of this abstract data type.
    pub fn descr(&self) -> &'static str {
        match self.adt_kind() {
            AdtKind::Struct => "struct",
            AdtKind::Union => "union",
            AdtKind::Enum => "enum",
        }
    }

    /// Returns a description of a variant of this abstract data type.
    #[inline]
    pub fn variant_descr(&self) -> &'static str {
        match self.adt_kind() {
            AdtKind::Struct => "struct",
            AdtKind::Union => "union",
            AdtKind::Enum => "variant",
        }
    }

    /// If this function returns `true`, it implies that `is_struct` must return `true`.
    #[inline]
    pub fn has_ctor(&self) -> bool {
        self.flags.contains(AdtFlags::HAS_CTOR)
    }

    /// Returns `true` if this type is `#[fundamental]` for the purposes
    /// of coherence checking.
    #[inline]
    pub fn is_fundamental(&self) -> bool {
        self.flags.contains(AdtFlags::IS_FUNDAMENTAL)
    }

    /// Returns `true` if this is `PhantomData<T>`.
    #[inline]
    pub fn is_phantom_data(&self) -> bool {
        self.flags.contains(AdtFlags::IS_PHANTOM_DATA)
    }

    /// Returns `true` if this is Box<T>.
    #[inline]
    pub fn is_box(&self) -> bool {
        self.flags.contains(AdtFlags::IS_BOX)
    }

    /// Returns `true` if this is `ManuallyDrop<T>`.
    #[inline]
    pub fn is_manually_drop(&self) -> bool {
        self.flags.contains(AdtFlags::IS_MANUALLY_DROP)
    }

    /// Returns `true` if this type has a destructor.
    pub fn has_dtor(&self, tcx: TyCtxt<'tcx>) -> bool {
        self.destructor(tcx).is_some()
    }

    pub fn has_non_const_dtor(&self, tcx: TyCtxt<'tcx>) -> bool {
        matches!(self.destructor(tcx), Some(Destructor { constness: hir::Constness::NotConst, .. }))
    }

    /// Asserts this is a struct or union and returns its unique variant.
    pub fn non_enum_variant(&self) -> &VariantDef {
        assert!(self.is_struct() || self.is_union());
        &self.variants[VariantIdx::new(0)]
    }

    #[inline]
    pub fn predicates(&self, tcx: TyCtxt<'tcx>) -> GenericPredicates<'tcx> {
        tcx.predicates_of(self.did)
    }

    /// Returns an iterator over all fields contained
    /// by this ADT.
    #[inline]
    pub fn all_fields(&self) -> impl Iterator<Item = &FieldDef> + Clone {
        self.variants.iter().flat_map(|v| v.fields.iter())
    }

    /// Whether the ADT lacks fields. Note that this includes uninhabited enums,
    /// e.g., `enum Void {}` is considered payload free as well.
    pub fn is_payloadfree(&self) -> bool {
        // Treat the ADT as not payload-free if arbitrary_enum_discriminant is used (#88621).
        // This would disallow the following kind of enum from being casted into integer.
        // ```
        // enum Enum {
        //    Foo() = 1,
        //    Bar{} = 2,
        //    Baz = 3,
        // }
        // ```
        if self
            .variants
            .iter()
            .any(|v| matches!(v.discr, VariantDiscr::Explicit(_)) && v.ctor_kind != CtorKind::Const)
        {
            return false;
        }
        self.variants.iter().all(|v| v.fields.is_empty())
    }

    /// Return a `VariantDef` given a variant id.
    pub fn variant_with_id(&self, vid: DefId) -> &VariantDef {
        self.variants.iter().find(|v| v.def_id == vid).expect("variant_with_id: unknown variant")
    }

    /// Return a `VariantDef` given a constructor id.
    pub fn variant_with_ctor_id(&self, cid: DefId) -> &VariantDef {
        self.variants
            .iter()
            .find(|v| v.ctor_def_id == Some(cid))
            .expect("variant_with_ctor_id: unknown variant")
    }

    /// Return the index of `VariantDef` given a variant id.
    pub fn variant_index_with_id(&self, vid: DefId) -> VariantIdx {
        self.variants
            .iter_enumerated()
            .find(|(_, v)| v.def_id == vid)
            .expect("variant_index_with_id: unknown variant")
            .0
    }

    /// Return the index of `VariantDef` given a constructor id.
    pub fn variant_index_with_ctor_id(&self, cid: DefId) -> VariantIdx {
        self.variants
            .iter_enumerated()
            .find(|(_, v)| v.ctor_def_id == Some(cid))
            .expect("variant_index_with_ctor_id: unknown variant")
            .0
    }

    pub fn variant_of_res(&self, res: Res) -> &VariantDef {
        match res {
            Res::Def(DefKind::Variant, vid) => self.variant_with_id(vid),
            Res::Def(DefKind::Ctor(..), cid) => self.variant_with_ctor_id(cid),
            Res::Def(DefKind::Struct, _)
            | Res::Def(DefKind::Union, _)
            | Res::Def(DefKind::TyAlias, _)
            | Res::Def(DefKind::AssocTy, _)
            | Res::SelfTy(..)
            | Res::SelfCtor(..) => self.non_enum_variant(),
            _ => bug!("unexpected res {:?} in variant_of_res", res),
        }
    }

    #[inline]
    pub fn eval_explicit_discr(&self, tcx: TyCtxt<'tcx>, expr_did: DefId) -> Option<Discr<'tcx>> {
        assert!(self.is_enum());
        let param_env = tcx.param_env(expr_did);
        let repr_type = self.repr.discr_type();
        match tcx.const_eval_poly(expr_did) {
            Ok(val) => {
                let ty = repr_type.to_ty(tcx);
                if let Some(b) = val.try_to_bits_for_ty(tcx, param_env, ty) {
                    trace!("discriminants: {} ({:?})", b, repr_type);
                    Some(Discr { val: b, ty })
                } else {
                    info!("invalid enum discriminant: {:#?}", val);
                    crate::mir::interpret::struct_error(
                        tcx.at(tcx.def_span(expr_did)),
                        "constant evaluation of enum discriminant resulted in non-integer",
                    )
                    .emit();
                    None
                }
            }
            Err(err) => {
                let msg = match err {
                    ErrorHandled::Reported(ErrorReported) | ErrorHandled::Linted => {
                        "enum discriminant evaluation failed"
                    }
                    ErrorHandled::TooGeneric => "enum discriminant depends on generics",
                };
                tcx.sess.delay_span_bug(tcx.def_span(expr_did), msg);
                None
            }
        }
    }

    #[inline]
    pub fn discriminants(
        &'tcx self,
        tcx: TyCtxt<'tcx>,
    ) -> impl Iterator<Item = (VariantIdx, Discr<'tcx>)> + Captures<'tcx> {
        assert!(self.is_enum());
        let repr_type = self.repr.discr_type();
        let initial = repr_type.initial_discriminant(tcx);
        let mut prev_discr = None::<Discr<'tcx>>;
        self.variants.iter_enumerated().map(move |(i, v)| {
            let mut discr = prev_discr.map_or(initial, |d| d.wrap_incr(tcx));
            if let VariantDiscr::Explicit(expr_did) = v.discr {
                if let Some(new_discr) = self.eval_explicit_discr(tcx, expr_did) {
                    discr = new_discr;
                }
            }
            prev_discr = Some(discr);

            (i, discr)
        })
    }

    #[inline]
    pub fn variant_range(&self) -> Range<VariantIdx> {
        VariantIdx::new(0)..VariantIdx::new(self.variants.len())
    }

    /// Computes the discriminant value used by a specific variant.
    /// Unlike `discriminants`, this is (amortized) constant-time,
    /// only doing at most one query for evaluating an explicit
    /// discriminant (the last one before the requested variant),
    /// assuming there are no constant-evaluation errors there.
    #[inline]
    pub fn discriminant_for_variant(
        &self,
        tcx: TyCtxt<'tcx>,
        variant_index: VariantIdx,
    ) -> Discr<'tcx> {
        assert!(self.is_enum());
        let (val, offset) = self.discriminant_def_for_variant(variant_index);
        let explicit_value = val
            .and_then(|expr_did| self.eval_explicit_discr(tcx, expr_did))
            .unwrap_or_else(|| self.repr.discr_type().initial_discriminant(tcx));
        explicit_value.checked_add(tcx, offset as u128).0
    }

    /// Yields a `DefId` for the discriminant and an offset to add to it
    /// Alternatively, if there is no explicit discriminant, returns the
    /// inferred discriminant directly.
    pub fn discriminant_def_for_variant(&self, variant_index: VariantIdx) -> (Option<DefId>, u32) {
        assert!(!self.variants.is_empty());
        let mut explicit_index = variant_index.as_u32();
        let expr_did;
        loop {
            match self.variants[VariantIdx::from_u32(explicit_index)].discr {
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

    pub fn destructor(&self, tcx: TyCtxt<'tcx>) -> Option<Destructor> {
        tcx.adt_destructor(self.did)
    }

    /// Returns a list of types such that `Self: Sized` if and only
    /// if that type is `Sized`, or `TyErr` if this type is recursive.
    ///
    /// Oddly enough, checking that the sized-constraint is `Sized` is
    /// actually more expressive than checking all members:
    /// the `Sized` trait is inductive, so an associated type that references
    /// `Self` would prevent its containing ADT from being `Sized`.
    ///
    /// Due to normalization being eager, this applies even if
    /// the associated type is behind a pointer (e.g., issue #31299).
    pub fn sized_constraint(&self, tcx: TyCtxt<'tcx>) -> &'tcx [Ty<'tcx>] {
        tcx.adt_sized_constraint(self.did).0
    }
}
