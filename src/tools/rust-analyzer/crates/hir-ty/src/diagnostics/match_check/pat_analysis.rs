//! Interface with `rustc_pattern_analysis`.

use std::fmt;

use hir_def::{DefWithBodyId, EnumVariantId, HasModule, LocalFieldId, ModuleId, VariantId};
use rustc_hash::FxHashMap;
use rustc_pattern_analysis::{
    constructor::{Constructor, ConstructorSet, VariantVisibility},
    index::IdxContainer,
    Captures, TypeCx,
};
use smallvec::SmallVec;
use stdx::never;
use typed_arena::Arena;

use crate::{
    db::HirDatabase,
    infer::normalize,
    inhabitedness::{is_enum_variant_uninhabited_from, is_ty_uninhabited_from},
    AdtId, Interner, Scalar, Ty, TyExt, TyKind,
};

use super::{is_box, FieldPat, Pat, PatKind};

use Constructor::*;

// Re-export r-a-specific versions of all these types.
pub(crate) type DeconstructedPat<'p> =
    rustc_pattern_analysis::pat::DeconstructedPat<'p, MatchCheckCtx<'p>>;
pub(crate) type MatchArm<'p> = rustc_pattern_analysis::MatchArm<'p, MatchCheckCtx<'p>>;
pub(crate) type WitnessPat<'p> = rustc_pattern_analysis::pat::WitnessPat<MatchCheckCtx<'p>>;

/// [Constructor] uses this in unimplemented variants.
/// It allows porting match expressions from upstream algorithm without losing semantics.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Void {}

#[derive(Clone)]
pub(crate) struct MatchCheckCtx<'p> {
    module: ModuleId,
    body: DefWithBodyId,
    pub(crate) db: &'p dyn HirDatabase,
    pub(crate) pattern_arena: &'p Arena<DeconstructedPat<'p>>,
    ty_arena: &'p Arena<Ty>,
    exhaustive_patterns: bool,
}

impl<'p> MatchCheckCtx<'p> {
    pub(crate) fn new(
        module: ModuleId,
        body: DefWithBodyId,
        db: &'p dyn HirDatabase,
        pattern_arena: &'p Arena<DeconstructedPat<'p>>,
        ty_arena: &'p Arena<Ty>,
    ) -> Self {
        let def_map = db.crate_def_map(module.krate());
        let exhaustive_patterns = def_map.is_unstable_feature_enabled("exhaustive_patterns");
        Self { module, body, db, pattern_arena, exhaustive_patterns, ty_arena }
    }

    fn is_uninhabited(&self, ty: &Ty) -> bool {
        is_ty_uninhabited_from(ty, self.module, self.db)
    }

    /// Returns whether the given type is an enum from another crate declared `#[non_exhaustive]`.
    fn is_foreign_non_exhaustive_enum(&self, ty: &Ty) -> bool {
        match ty.as_adt() {
            Some((adt @ hir_def::AdtId::EnumId(_), _)) => {
                let has_non_exhaustive_attr =
                    self.db.attrs(adt.into()).by_key("non_exhaustive").exists();
                let is_local = adt.module(self.db.upcast()).krate() == self.module.krate();
                has_non_exhaustive_attr && !is_local
            }
            _ => false,
        }
    }

    fn variant_id_for_adt(&self, ctor: &Constructor<Self>, adt: hir_def::AdtId) -> VariantId {
        match ctor {
            &Variant(id) => id.into(),
            Struct | UnionField => {
                assert!(!matches!(adt, hir_def::AdtId::EnumId(_)));
                match adt {
                    hir_def::AdtId::EnumId(_) => unreachable!(),
                    hir_def::AdtId::StructId(id) => id.into(),
                    hir_def::AdtId::UnionId(id) => id.into(),
                }
            }
            _ => panic!("bad constructor {self:?} for adt {adt:?}"),
        }
    }

    // In the cases of either a `#[non_exhaustive]` field list or a non-public field, we hide
    // uninhabited fields in order not to reveal the uninhabitedness of the whole variant.
    // This lists the fields we keep along with their types.
    fn list_variant_nonhidden_fields<'a>(
        &'a self,
        ty: &'a Ty,
        variant: VariantId,
    ) -> impl Iterator<Item = (LocalFieldId, Ty)> + Captures<'a> + Captures<'p> {
        let cx = self;
        let (adt, substs) = ty.as_adt().unwrap();

        let adt_is_local = variant.module(cx.db.upcast()).krate() == cx.module.krate();

        // Whether we must not match the fields of this variant exhaustively.
        let is_non_exhaustive =
            cx.db.attrs(variant.into()).by_key("non_exhaustive").exists() && !adt_is_local;

        let visibility = cx.db.field_visibilities(variant);
        let field_ty = cx.db.field_types(variant);
        let fields_len = variant.variant_data(cx.db.upcast()).fields().len() as u32;

        (0..fields_len).map(|idx| LocalFieldId::from_raw(idx.into())).filter_map(move |fid| {
            let ty = field_ty[fid].clone().substitute(Interner, substs);
            let ty = normalize(cx.db, cx.db.trait_environment_for_body(cx.body), ty);
            let is_visible = matches!(adt, hir_def::AdtId::EnumId(..))
                || visibility[fid].is_visible_from(cx.db.upcast(), cx.module);
            let is_uninhabited = cx.is_uninhabited(&ty);

            if is_uninhabited && (!is_visible || is_non_exhaustive) {
                None
            } else {
                Some((fid, ty))
            }
        })
    }

    pub(crate) fn lower_pat(&self, pat: &Pat) -> DeconstructedPat<'p> {
        let singleton = |pat| std::slice::from_ref(self.pattern_arena.alloc(pat));
        let ctor;
        let fields: &[_];

        match pat.kind.as_ref() {
            PatKind::Binding { subpattern: Some(subpat), .. } => return self.lower_pat(subpat),
            PatKind::Binding { subpattern: None, .. } | PatKind::Wild => {
                ctor = Wildcard;
                fields = &[];
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
            }
            PatKind::Leaf { subpatterns } | PatKind::Variant { subpatterns, .. } => {
                match pat.ty.kind(Interner) {
                    TyKind::Tuple(_, substs) => {
                        ctor = Struct;
                        let mut wilds: SmallVec<[_; 2]> = substs
                            .iter(Interner)
                            .map(|arg| arg.assert_ty_ref(Interner).clone())
                            .map(DeconstructedPat::wildcard)
                            .collect();
                        for pat in subpatterns {
                            let idx: u32 = pat.field.into_raw().into();
                            wilds[idx as usize] = self.lower_pat(&pat.pattern);
                        }
                        fields = self.pattern_arena.alloc_extend(wilds)
                    }
                    TyKind::Adt(adt, substs) if is_box(self.db, adt.0) => {
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
                        let pat =
                            subpatterns.iter().find(|pat| pat.field.into_raw() == 0u32.into());
                        let field = if let Some(pat) = pat {
                            self.lower_pat(&pat.pattern)
                        } else {
                            let ty = substs.at(Interner, 0).assert_ty_ref(Interner).clone();
                            DeconstructedPat::wildcard(ty)
                        };
                        ctor = Struct;
                        fields = singleton(field);
                    }
                    &TyKind::Adt(adt, _) => {
                        ctor = match pat.kind.as_ref() {
                            PatKind::Leaf { .. } if matches!(adt.0, hir_def::AdtId::UnionId(_)) => {
                                UnionField
                            }
                            PatKind::Leaf { .. } => Struct,
                            PatKind::Variant { enum_variant, .. } => Variant(*enum_variant),
                            _ => {
                                never!();
                                Wildcard
                            }
                        };
                        let variant = self.variant_id_for_adt(&ctor, adt.0);
                        let fields_len = variant.variant_data(self.db.upcast()).fields().len();
                        // For each field in the variant, we store the relevant index into `self.fields` if any.
                        let mut field_id_to_id: Vec<Option<usize>> = vec![None; fields_len];
                        let tys = self
                            .list_variant_nonhidden_fields(&pat.ty, variant)
                            .enumerate()
                            .map(|(i, (fid, ty))| {
                                let field_idx: u32 = fid.into_raw().into();
                                field_id_to_id[field_idx as usize] = Some(i);
                                ty
                            });
                        let mut wilds: SmallVec<[_; 2]> =
                            tys.map(DeconstructedPat::wildcard).collect();
                        for pat in subpatterns {
                            let field_idx: u32 = pat.field.into_raw().into();
                            if let Some(i) = field_id_to_id[field_idx as usize] {
                                wilds[i] = self.lower_pat(&pat.pattern);
                            }
                        }
                        fields = self.pattern_arena.alloc_extend(wilds);
                    }
                    _ => {
                        never!("pattern has unexpected type: pat: {:?}, ty: {:?}", pat, &pat.ty);
                        ctor = Wildcard;
                        fields = &[];
                    }
                }
            }
            &PatKind::LiteralBool { value } => {
                ctor = Bool(value);
                fields = &[];
            }
            PatKind::Or { pats } => {
                ctor = Or;
                // Collect here because `Arena::alloc_extend` panics on reentrancy.
                let subpats: SmallVec<[_; 2]> =
                    pats.iter().map(|pat| self.lower_pat(pat)).collect();
                fields = self.pattern_arena.alloc_extend(subpats);
            }
        }
        DeconstructedPat::new(ctor, fields, pat.ty.clone(), ())
    }

    pub(crate) fn hoist_witness_pat(&self, pat: &WitnessPat<'p>) -> Pat {
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
                    let variant = self.variant_id_for_adt(pat.ctor(), adt.0);
                    let subpatterns = self
                        .list_variant_nonhidden_fields(pat.ty(), variant)
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
            &Str(void) => match void {},
            Wildcard | NonExhaustive | Hidden => PatKind::Wild,
            Missing | F32Range(..) | F64Range(..) | Opaque(..) | Or => {
                never!("can't convert to pattern: {:?}", pat.ctor());
                PatKind::Wild
            }
        };
        Pat { ty: pat.ty().clone(), kind: Box::new(kind) }
    }
}

impl<'p> TypeCx for MatchCheckCtx<'p> {
    type Error = Void;
    type Ty = Ty;
    type VariantIdx = EnumVariantId;
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
                        let variant = self.variant_id_for_adt(ctor, adt);
                        self.list_variant_nonhidden_fields(ty, variant).count()
                    }
                }
                _ => {
                    never!("Unexpected type for `Single` constructor: {:?}", ty);
                    0
                }
            },
            Ref => 1,
            Slice(..) => unimplemented!(),
            Bool(..) | IntRange(..) | F32Range(..) | F64Range(..) | Str(..) | Opaque(..)
            | NonExhaustive | Hidden | Missing | Wildcard => 0,
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
    ) -> &[Self::Ty] {
        use std::iter::once;
        fn alloc<'a>(cx: &'a MatchCheckCtx<'_>, iter: impl Iterator<Item = Ty>) -> &'a [Ty] {
            cx.ty_arena.alloc_extend(iter)
        }
        match ctor {
            Struct | Variant(_) | UnionField => match ty.kind(Interner) {
                TyKind::Tuple(_, substs) => {
                    let tys = substs.iter(Interner).map(|ty| ty.assert_ty_ref(Interner));
                    alloc(self, tys.cloned())
                }
                TyKind::Ref(.., rty) => alloc(self, once(rty.clone())),
                &TyKind::Adt(AdtId(adt), ref substs) => {
                    if is_box(self.db, adt) {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        let subst_ty = substs.at(Interner, 0).assert_ty_ref(Interner).clone();
                        alloc(self, once(subst_ty))
                    } else {
                        let variant = self.variant_id_for_adt(ctor, adt);
                        let tys = self.list_variant_nonhidden_fields(ty, variant).map(|(_, ty)| ty);
                        alloc(self, tys)
                    }
                }
                ty_kind => {
                    never!("Unexpected type for `{:?}` constructor: {:?}", ctor, ty_kind);
                    alloc(self, once(ty.clone()))
                }
            },
            Ref => match ty.kind(Interner) {
                TyKind::Ref(.., rty) => alloc(self, once(rty.clone())),
                ty_kind => {
                    never!("Unexpected type for `{:?}` constructor: {:?}", ctor, ty_kind);
                    alloc(self, once(ty.clone()))
                }
            },
            Slice(_) => unreachable!("Found a `Slice` constructor in match checking"),
            Bool(..) | IntRange(..) | F32Range(..) | F64Range(..) | Str(..) | Opaque(..)
            | NonExhaustive | Hidden | Missing | Wildcard => &[],
            Or => {
                never!("called `Fields::wildcards` on an `Or` ctor");
                &[]
            }
        }
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
            TyKind::Adt(AdtId(hir_def::AdtId::EnumId(enum_id)), subst) => {
                let enum_data = cx.db.enum_data(*enum_id);
                let is_declared_nonexhaustive = cx.is_foreign_non_exhaustive_enum(ty);

                if enum_data.variants.is_empty() && !is_declared_nonexhaustive {
                    ConstructorSet::NoConstructors
                } else {
                    let mut variants = FxHashMap::default();
                    for &(variant, _) in enum_data.variants.iter() {
                        let is_uninhabited =
                            is_enum_variant_uninhabited_from(variant, subst, cx.module, cx.db);
                        let visibility = if is_uninhabited {
                            VariantVisibility::Empty
                        } else {
                            VariantVisibility::Visible
                        };
                        variants.insert(variant, visibility);
                    }

                    ConstructorSet::Variants {
                        variants: IdxContainer(variants),
                        non_exhaustive: is_declared_nonexhaustive,
                    }
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

    fn debug_pat(
        _f: &mut fmt::Formatter<'_>,
        _pat: &rustc_pattern_analysis::pat::DeconstructedPat<'_, Self>,
    ) -> fmt::Result {
        // FIXME: implement this, as using `unimplemented!()` causes panics in `tracing`.
        Ok(())
    }

    fn bug(&self, fmt: fmt::Arguments<'_>) -> ! {
        panic!("{}", fmt)
    }
}

impl<'p> fmt::Debug for MatchCheckCtx<'p> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatchCheckCtx").finish()
    }
}
