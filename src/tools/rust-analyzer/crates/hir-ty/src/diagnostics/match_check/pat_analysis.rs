//! Interface with `rustc_pattern_analysis`.

use std::cell::LazyCell;
use std::fmt;

use hir_def::{DefWithBodyId, EnumId, EnumVariantId, HasModule, LocalFieldId, ModuleId, VariantId};
use intern::sym;
use rustc_pattern_analysis::{
    IndexVec, PatCx, PrivateUninhabitedField,
    constructor::{Constructor, ConstructorSet, VariantVisibility},
    usefulness::{PlaceValidity, UsefulnessReport, compute_match_usefulness},
};
use smallvec::{SmallVec, smallvec};
use stdx::never;
use triomphe::Arc;

use crate::{
    AdtId, Interner, Scalar, TraitEnvironment, Ty, TyExt, TyKind,
    db::HirDatabase,
    infer::normalize,
    inhabitedness::{is_enum_variant_uninhabited_from, is_ty_uninhabited_from},
};

use super::{FieldPat, Pat, PatKind, is_box};

use Constructor::*;

// Re-export r-a-specific versions of all these types.
pub(crate) type DeconstructedPat<'db> =
    rustc_pattern_analysis::pat::DeconstructedPat<MatchCheckCtx<'db>>;
pub(crate) type MatchArm<'db> = rustc_pattern_analysis::MatchArm<'db, MatchCheckCtx<'db>>;
pub(crate) type WitnessPat<'db> = rustc_pattern_analysis::pat::WitnessPat<MatchCheckCtx<'db>>;

/// [Constructor] uses this in unimplemented variants.
/// It allows porting match expressions from upstream algorithm without losing semantics.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Void {}

/// An index type for enum variants. This ranges from 0 to `variants.len()`, whereas `EnumVariantId`
/// can take arbitrary large values (and hence mustn't be used with `IndexVec`/`BitSet`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EnumVariantContiguousIndex(usize);

impl EnumVariantContiguousIndex {
    fn from_enum_variant_id(db: &dyn HirDatabase, target_evid: EnumVariantId) -> Self {
        // Find the index of this variant in the list of variants.
        use hir_def::Lookup;
        let i = target_evid.lookup(db).index as usize;
        EnumVariantContiguousIndex(i)
    }

    fn to_enum_variant_id(self, db: &dyn HirDatabase, eid: EnumId) -> EnumVariantId {
        eid.enum_variants(db).variants[self.0].0
    }
}

impl rustc_pattern_analysis::Idx for EnumVariantContiguousIndex {
    fn new(idx: usize) -> Self {
        EnumVariantContiguousIndex(idx)
    }

    fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone)]
pub(crate) struct MatchCheckCtx<'db> {
    module: ModuleId,
    body: DefWithBodyId,
    pub(crate) db: &'db dyn HirDatabase,
    exhaustive_patterns: bool,
    env: Arc<TraitEnvironment>,
}

impl<'db> MatchCheckCtx<'db> {
    pub(crate) fn new(
        module: ModuleId,
        body: DefWithBodyId,
        db: &'db dyn HirDatabase,
        env: Arc<TraitEnvironment>,
    ) -> Self {
        let def_map = module.crate_def_map(db);
        let exhaustive_patterns = def_map.is_unstable_feature_enabled(&sym::exhaustive_patterns);
        Self { module, body, db, exhaustive_patterns, env }
    }

    pub(crate) fn compute_match_usefulness(
        &self,
        arms: &[MatchArm<'db>],
        scrut_ty: Ty,
        known_valid_scrutinee: Option<bool>,
    ) -> Result<UsefulnessReport<'db, Self>, ()> {
        if scrut_ty.contains_unknown() {
            return Err(());
        }
        for arm in arms {
            if arm.pat.ty().contains_unknown() {
                return Err(());
            }
        }

        let place_validity = PlaceValidity::from_bool(known_valid_scrutinee.unwrap_or(true));
        // Measured to take ~100ms on modern hardware.
        let complexity_limit = 500000;
        compute_match_usefulness(self, arms, scrut_ty, place_validity, complexity_limit)
    }

    fn is_uninhabited(&self, ty: &Ty) -> bool {
        is_ty_uninhabited_from(self.db, ty, self.module, self.env.clone())
    }

    /// Returns whether the given ADT is from another crate declared `#[non_exhaustive]`.
    fn is_foreign_non_exhaustive(&self, adt: hir_def::AdtId) -> bool {
        let is_local = adt.krate(self.db) == self.module.krate();
        !is_local && self.db.attrs(adt.into()).by_key(sym::non_exhaustive).exists()
    }

    fn variant_id_for_adt(
        db: &'db dyn HirDatabase,
        ctor: &Constructor<Self>,
        adt: hir_def::AdtId,
    ) -> Option<VariantId> {
        match ctor {
            Variant(id) => {
                let hir_def::AdtId::EnumId(eid) = adt else {
                    panic!("bad constructor {ctor:?} for adt {adt:?}")
                };
                Some(id.to_enum_variant_id(db, eid).into())
            }
            Struct | UnionField => match adt {
                hir_def::AdtId::EnumId(_) => None,
                hir_def::AdtId::StructId(id) => Some(id.into()),
                hir_def::AdtId::UnionId(id) => Some(id.into()),
            },
            _ => panic!("bad constructor {ctor:?} for adt {adt:?}"),
        }
    }

    // This lists the fields of a variant along with their types.
    fn list_variant_fields(
        &self,
        ty: &Ty,
        variant: VariantId,
    ) -> impl Iterator<Item = (LocalFieldId, Ty)> {
        let (_, substs) = ty.as_adt().unwrap();

        let field_tys = self.db.field_types(variant);
        let fields_len = variant.fields(self.db).fields().len() as u32;

        (0..fields_len).map(|idx| LocalFieldId::from_raw(idx.into())).map(move |fid| {
            let ty = field_tys[fid].clone().substitute(Interner, substs);
            let ty = normalize(self.db, self.db.trait_environment_for_body(self.body), ty);
            (fid, ty)
        })
    }

    pub(crate) fn lower_pat(&self, pat: &Pat) -> DeconstructedPat<'db> {
        let singleton = |pat: DeconstructedPat<'db>| vec![pat.at_index(0)];
        let ctor;
        let mut fields: Vec<_>;
        let arity;

        match pat.kind.as_ref() {
            PatKind::Binding { subpattern: Some(subpat), .. } => return self.lower_pat(subpat),
            PatKind::Binding { subpattern: None, .. } | PatKind::Wild => {
                ctor = Wildcard;
                fields = Vec::new();
                arity = 0;
            }
            PatKind::Deref { subpattern } => {
                ctor = match pat.ty.kind(Interner) {
                    // This is a box pattern.
                    TyKind::Adt(adt, _) if is_box(self.db, adt.0) => Struct,
                    TyKind::Ref(..) => Ref,
                    _ => {
                        never!("pattern has unexpected type: pat: {:?}, ty: {:?}", pat, &pat.ty);
                        Wildcard
                    }
                };
                fields = singleton(self.lower_pat(subpattern));
                arity = 1;
            }
            PatKind::Leaf { subpatterns } | PatKind::Variant { subpatterns, .. } => {
                fields = subpatterns
                    .iter()
                    .map(|pat| {
                        let idx: u32 = pat.field.into_raw().into();
                        self.lower_pat(&pat.pattern).at_index(idx as usize)
                    })
                    .collect();
                match pat.ty.kind(Interner) {
                    TyKind::Tuple(_, substs) => {
                        ctor = Struct;
                        arity = substs.len(Interner);
                    }
                    TyKind::Adt(adt, _) if is_box(self.db, adt.0) => {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        // FIXME(Nadrieril): A `Box` can in theory be matched either with `Box(_,
                        // _)` or a box pattern. As a hack to avoid an ICE with the former, we
                        // ignore other fields than the first one. This will trigger an error later
                        // anyway.
                        // See https://github.com/rust-lang/rust/issues/82772 ,
                        // explanation: https://github.com/rust-lang/rust/pull/82789#issuecomment-796921977
                        // The problem is that we can't know from the type whether we'll match
                        // normally or through box-patterns. We'll have to figure out a proper
                        // solution when we introduce generalized deref patterns. Also need to
                        // prevent mixing of those two options.
                        fields.retain(|ipat| ipat.idx == 0);
                        ctor = Struct;
                        arity = 1;
                    }
                    &TyKind::Adt(AdtId(adt), _) => {
                        ctor = match pat.kind.as_ref() {
                            PatKind::Leaf { .. } if matches!(adt, hir_def::AdtId::UnionId(_)) => {
                                UnionField
                            }
                            PatKind::Leaf { .. } => Struct,
                            PatKind::Variant { enum_variant, .. } => {
                                Variant(EnumVariantContiguousIndex::from_enum_variant_id(
                                    self.db,
                                    *enum_variant,
                                ))
                            }
                            _ => {
                                never!();
                                Wildcard
                            }
                        };
                        let variant = Self::variant_id_for_adt(self.db, &ctor, adt).unwrap();
                        arity = variant.fields(self.db).fields().len();
                    }
                    _ => {
                        never!("pattern has unexpected type: pat: {:?}, ty: {:?}", pat, &pat.ty);
                        ctor = Wildcard;
                        fields.clear();
                        arity = 0;
                    }
                }
            }
            &PatKind::LiteralBool { value } => {
                ctor = Bool(value);
                fields = Vec::new();
                arity = 0;
            }
            PatKind::Never => {
                ctor = Never;
                fields = Vec::new();
                arity = 0;
            }
            PatKind::Or { pats } => {
                ctor = Or;
                fields = pats
                    .iter()
                    .enumerate()
                    .map(|(i, pat)| self.lower_pat(pat).at_index(i))
                    .collect();
                arity = pats.len();
            }
        }
        DeconstructedPat::new(ctor, fields, arity, pat.ty.clone(), ())
    }

    pub(crate) fn hoist_witness_pat(&self, pat: &WitnessPat<'db>) -> Pat {
        let mut subpatterns = pat.iter_fields().map(|p| self.hoist_witness_pat(p));
        let kind = match pat.ctor() {
            &Bool(value) => PatKind::LiteralBool { value },
            IntRange(_) => unimplemented!(),
            Struct | Variant(_) | UnionField => match pat.ty().kind(Interner) {
                TyKind::Tuple(..) => PatKind::Leaf {
                    subpatterns: subpatterns
                        .zip(0u32..)
                        .map(|(p, i)| FieldPat {
                            field: LocalFieldId::from_raw(i.into()),
                            pattern: p,
                        })
                        .collect(),
                },
                TyKind::Adt(adt, _) if is_box(self.db, adt.0) => {
                    // Without `box_patterns`, the only legal pattern of type `Box` is `_` (outside
                    // of `std`). So this branch is only reachable when the feature is enabled and
                    // the pattern is a box pattern.
                    PatKind::Deref { subpattern: subpatterns.next().unwrap() }
                }
                TyKind::Adt(adt, substs) => {
                    let variant = Self::variant_id_for_adt(self.db, pat.ctor(), adt.0).unwrap();
                    let subpatterns = self
                        .list_variant_fields(pat.ty(), variant)
                        .zip(subpatterns)
                        .map(|((field, _ty), pattern)| FieldPat { field, pattern })
                        .collect();

                    if let VariantId::EnumVariantId(enum_variant) = variant {
                        PatKind::Variant { substs: substs.clone(), enum_variant, subpatterns }
                    } else {
                        PatKind::Leaf { subpatterns }
                    }
                }
                _ => {
                    never!("unexpected ctor for type {:?} {:?}", pat.ctor(), pat.ty());
                    PatKind::Wild
                }
            },
            // Note: given the expansion of `&str` patterns done in `expand_pattern`, we should
            // be careful to reconstruct the correct constant pattern here. However a string
            // literal pattern will never be reported as a non-exhaustiveness witness, so we
            // ignore this issue.
            Ref => PatKind::Deref { subpattern: subpatterns.next().unwrap() },
            Slice(_) => unimplemented!(),
            DerefPattern(_) => unimplemented!(),
            &Str(void) => match void {},
            Wildcard | NonExhaustive | Hidden | PrivateUninhabited => PatKind::Wild,
            Never => PatKind::Never,
            Missing | F16Range(..) | F32Range(..) | F64Range(..) | F128Range(..) | Opaque(..)
            | Or => {
                never!("can't convert to pattern: {:?}", pat.ctor());
                PatKind::Wild
            }
        };
        Pat { ty: pat.ty().clone(), kind: Box::new(kind) }
    }
}

impl PatCx for MatchCheckCtx<'_> {
    type Error = ();
    type Ty = Ty;
    type VariantIdx = EnumVariantContiguousIndex;
    type StrLit = Void;
    type ArmData = ();
    type PatData = ();

    fn is_exhaustive_patterns_feature_on(&self) -> bool {
        self.exhaustive_patterns
    }

    fn ctor_arity(
        &self,
        ctor: &rustc_pattern_analysis::constructor::Constructor<Self>,
        ty: &Self::Ty,
    ) -> usize {
        match ctor {
            Struct | Variant(_) | UnionField => match *ty.kind(Interner) {
                TyKind::Tuple(arity, ..) => arity,
                TyKind::Adt(AdtId(adt), ..) => {
                    if is_box(self.db, adt) {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        1
                    } else {
                        let variant = Self::variant_id_for_adt(self.db, ctor, adt).unwrap();
                        variant.fields(self.db).fields().len()
                    }
                }
                _ => {
                    never!("Unexpected type for `Single` constructor: {:?}", ty);
                    0
                }
            },
            Ref => 1,
            Slice(..) => unimplemented!(),
            DerefPattern(..) => unimplemented!(),
            Never | Bool(..) | IntRange(..) | F16Range(..) | F32Range(..) | F64Range(..)
            | F128Range(..) | Str(..) | Opaque(..) | NonExhaustive | PrivateUninhabited
            | Hidden | Missing | Wildcard => 0,
            Or => {
                never!("The `Or` constructor doesn't have a fixed arity");
                0
            }
        }
    }

    fn ctor_sub_tys(
        &self,
        ctor: &rustc_pattern_analysis::constructor::Constructor<Self>,
        ty: &Self::Ty,
    ) -> impl ExactSizeIterator<Item = (Self::Ty, PrivateUninhabitedField)> {
        let single = |ty| smallvec![(ty, PrivateUninhabitedField(false))];
        let tys: SmallVec<[_; 2]> = match ctor {
            Struct | Variant(_) | UnionField => match ty.kind(Interner) {
                TyKind::Tuple(_, substs) => {
                    let tys = substs.iter(Interner).map(|ty| ty.assert_ty_ref(Interner));
                    tys.cloned().map(|ty| (ty, PrivateUninhabitedField(false))).collect()
                }
                TyKind::Ref(.., rty) => single(rty.clone()),
                &TyKind::Adt(AdtId(adt), ref substs) => {
                    if is_box(self.db, adt) {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        let subst_ty = substs.at(Interner, 0).assert_ty_ref(Interner).clone();
                        single(subst_ty)
                    } else {
                        let variant = Self::variant_id_for_adt(self.db, ctor, adt).unwrap();

                        let visibilities = LazyCell::new(|| self.db.field_visibilities(variant));

                        self.list_variant_fields(ty, variant)
                            .map(move |(fid, ty)| {
                                let is_visible = || {
                                    matches!(adt, hir_def::AdtId::EnumId(..))
                                        || visibilities[fid].is_visible_from(self.db, self.module)
                                };
                                let is_uninhabited = self.is_uninhabited(&ty);
                                let private_uninhabited = is_uninhabited && !is_visible();
                                (ty, PrivateUninhabitedField(private_uninhabited))
                            })
                            .collect()
                    }
                }
                ty_kind => {
                    never!("Unexpected type for `{:?}` constructor: {:?}", ctor, ty_kind);
                    single(ty.clone())
                }
            },
            Ref => match ty.kind(Interner) {
                TyKind::Ref(.., rty) => single(rty.clone()),
                ty_kind => {
                    never!("Unexpected type for `{:?}` constructor: {:?}", ctor, ty_kind);
                    single(ty.clone())
                }
            },
            Slice(_) => unreachable!("Found a `Slice` constructor in match checking"),
            DerefPattern(_) => unreachable!("Found a `DerefPattern` constructor in match checking"),
            Never | Bool(..) | IntRange(..) | F16Range(..) | F32Range(..) | F64Range(..)
            | F128Range(..) | Str(..) | Opaque(..) | NonExhaustive | PrivateUninhabited
            | Hidden | Missing | Wildcard => {
                smallvec![]
            }
            Or => {
                never!("called `Fields::wildcards` on an `Or` ctor");
                smallvec![]
            }
        };
        tys.into_iter()
    }

    fn ctors_for_ty(
        &self,
        ty: &Self::Ty,
    ) -> Result<rustc_pattern_analysis::constructor::ConstructorSet<Self>, Self::Error> {
        let cx = self;

        // Unhandled types are treated as non-exhaustive. Being explicit here instead of falling
        // to catchall arm to ease further implementation.
        let unhandled = || ConstructorSet::Unlistable;

        // This determines the set of all possible constructors for the type `ty`. For numbers,
        // arrays and slices we use ranges and variable-length slices when appropriate.
        //
        // If the `exhaustive_patterns` feature is enabled, we make sure to omit constructors that
        // are statically impossible. E.g., for `Option<!>`, we do not include `Some(_)` in the
        // returned list of constructors.
        // Invariant: this is empty if and only if the type is uninhabited (as determined by
        // `cx.is_uninhabited()`).
        Ok(match ty.kind(Interner) {
            TyKind::Scalar(Scalar::Bool) => ConstructorSet::Bool,
            TyKind::Scalar(Scalar::Char) => unhandled(),
            TyKind::Scalar(Scalar::Int(..) | Scalar::Uint(..)) => unhandled(),
            TyKind::Array(..) | TyKind::Slice(..) => unhandled(),
            &TyKind::Adt(AdtId(adt @ hir_def::AdtId::EnumId(enum_id)), ref subst) => {
                let enum_data = enum_id.enum_variants(cx.db);
                let is_declared_nonexhaustive = cx.is_foreign_non_exhaustive(adt);

                if enum_data.variants.is_empty() && !is_declared_nonexhaustive {
                    ConstructorSet::NoConstructors
                } else {
                    let mut variants = IndexVec::with_capacity(enum_data.variants.len());
                    for &(variant, _, _) in enum_data.variants.iter() {
                        let is_uninhabited = is_enum_variant_uninhabited_from(
                            cx.db,
                            variant,
                            subst,
                            cx.module,
                            self.env.clone(),
                        );
                        let visibility = if is_uninhabited {
                            VariantVisibility::Empty
                        } else {
                            VariantVisibility::Visible
                        };
                        variants.push(visibility);
                    }

                    ConstructorSet::Variants { variants, non_exhaustive: is_declared_nonexhaustive }
                }
            }
            TyKind::Adt(AdtId(hir_def::AdtId::UnionId(_)), _) => ConstructorSet::Union,
            TyKind::Adt(..) | TyKind::Tuple(..) => {
                ConstructorSet::Struct { empty: cx.is_uninhabited(ty) }
            }
            TyKind::Ref(..) => ConstructorSet::Ref,
            TyKind::Never => ConstructorSet::NoConstructors,
            // This type is one for which we cannot list constructors, like `str` or `f64`.
            _ => ConstructorSet::Unlistable,
        })
    }

    fn write_variant_name(
        f: &mut fmt::Formatter<'_>,
        _ctor: &Constructor<Self>,
        _ty: &Self::Ty,
    ) -> fmt::Result {
        write!(f, "<write_variant_name unsupported>")
        // We lack the database here ...
        // let variant = ty.as_adt().and_then(|(adt, _)| Self::variant_id_for_adt(db, ctor, adt));

        // if let Some(variant) = variant {
        //     match variant {
        //         VariantId::EnumVariantId(v) => {
        //             write!(f, "{}", db.enum_variant_data(v).name.display(db))?;
        //         }
        //         VariantId::StructId(s) => {
        //             write!(f, "{}", db.struct_data(s).name.display(db))?
        //         }
        //         VariantId::UnionId(u) => {
        //             write!(f, "{}", db.union_data(u).name.display(db))?
        //         }
        //     }
        // }
        // Ok(())
    }

    fn bug(&self, fmt: fmt::Arguments<'_>) {
        never!("{}", fmt)
    }

    fn complexity_exceeded(&self) -> Result<(), Self::Error> {
        Err(())
    }
}

impl fmt::Debug for MatchCheckCtx<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatchCheckCtx").finish()
    }
}
