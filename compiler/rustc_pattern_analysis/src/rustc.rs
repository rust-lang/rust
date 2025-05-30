use std::fmt;
use std::iter::once;

use rustc_abi::{FIRST_VARIANT, FieldIdx, Integer, VariantIdx};
use rustc_arena::DroplessArena;
use rustc_hir::HirId;
use rustc_hir::def_id::DefId;
use rustc_index::{Idx, IndexVec};
use rustc_middle::middle::stability::EvalResult;
use rustc_middle::mir::{self, Const};
use rustc_middle::thir::{self, Pat, PatKind, PatRange, PatRangeBoundary};
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::{
    self, FieldDef, OpaqueTypeKey, ScalarInt, Ty, TyCtxt, TypeVisitableExt, VariantDef,
};
use rustc_middle::{bug, span_bug};
use rustc_session::lint;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span, sym};

use crate::constructor::Constructor::*;
use crate::constructor::{
    IntRange, MaybeInfiniteInt, OpaqueId, RangeEnd, Slice, SliceKind, VariantVisibility,
};
use crate::lints::lint_nonexhaustive_missing_variants;
use crate::pat_column::PatternColumn;
use crate::rustc::print::EnumInfo;
use crate::usefulness::{PlaceValidity, compute_match_usefulness};
use crate::{PatCx, PrivateUninhabitedField, errors};

mod print;

// Re-export rustc-specific versions of all these types.
pub type Constructor<'p, 'tcx> = crate::constructor::Constructor<RustcPatCtxt<'p, 'tcx>>;
pub type ConstructorSet<'p, 'tcx> = crate::constructor::ConstructorSet<RustcPatCtxt<'p, 'tcx>>;
pub type DeconstructedPat<'p, 'tcx> = crate::pat::DeconstructedPat<RustcPatCtxt<'p, 'tcx>>;
pub type MatchArm<'p, 'tcx> = crate::MatchArm<'p, RustcPatCtxt<'p, 'tcx>>;
pub type RedundancyExplanation<'p, 'tcx> =
    crate::usefulness::RedundancyExplanation<'p, RustcPatCtxt<'p, 'tcx>>;
pub type Usefulness<'p, 'tcx> = crate::usefulness::Usefulness<'p, RustcPatCtxt<'p, 'tcx>>;
pub type UsefulnessReport<'p, 'tcx> =
    crate::usefulness::UsefulnessReport<'p, RustcPatCtxt<'p, 'tcx>>;
pub type WitnessPat<'p, 'tcx> = crate::pat::WitnessPat<RustcPatCtxt<'p, 'tcx>>;

/// A type which has gone through `cx.reveal_opaque_ty`, i.e. if it was opaque it was replaced by
/// the hidden type if allowed in the current body. This ensures we consistently inspect the hidden
/// types when we should.
///
/// Use `.inner()` or deref to get to the `Ty<'tcx>`.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RevealedTy<'tcx>(Ty<'tcx>);

impl<'tcx> fmt::Display for RevealedTy<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(fmt)
    }
}

impl<'tcx> fmt::Debug for RevealedTy<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(fmt)
    }
}

impl<'tcx> std::ops::Deref for RevealedTy<'tcx> {
    type Target = Ty<'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'tcx> RevealedTy<'tcx> {
    pub fn inner(self) -> Ty<'tcx> {
        self.0
    }
}

#[derive(Clone)]
pub struct RustcPatCtxt<'p, 'tcx: 'p> {
    pub tcx: TyCtxt<'tcx>,
    pub typeck_results: &'tcx ty::TypeckResults<'tcx>,
    /// The module in which the match occurs. This is necessary for
    /// checking inhabited-ness of types because whether a type is (visibly)
    /// inhabited can depend on whether it was defined in the current module or
    /// not. E.g., `struct Foo { _private: ! }` cannot be seen to be empty
    /// outside its module and should not be matchable with an empty match statement.
    pub module: DefId,
    pub typing_env: ty::TypingEnv<'tcx>,
    /// To allocate the result of `self.ctor_sub_tys()`
    pub dropless_arena: &'p DroplessArena,
    /// Lint level at the match.
    pub match_lint_level: HirId,
    /// The span of the whole match, if applicable.
    pub whole_match_span: Option<Span>,
    /// Span of the scrutinee.
    pub scrut_span: Span,
    /// Only produce `NON_EXHAUSTIVE_OMITTED_PATTERNS` lint on refutable patterns.
    pub refutable: bool,
    /// Whether the data at the scrutinee is known to be valid. This is false if the scrutinee comes
    /// from a union field, a pointer deref, or a reference deref (pending opsem decisions).
    pub known_valid_scrutinee: bool,
}

impl<'p, 'tcx: 'p> fmt::Debug for RustcPatCtxt<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RustcPatCtxt").finish()
    }
}

impl<'p, 'tcx: 'p> RustcPatCtxt<'p, 'tcx> {
    /// Type inference occasionally gives us opaque types in places where corresponding patterns
    /// have more specific types. To avoid inconsistencies as well as detect opaque uninhabited
    /// types, we use the corresponding concrete type if possible.
    // FIXME(#132279): This will be unnecessary once we have a TypingMode which supports revealing
    // opaque types defined in a body.
    #[inline]
    pub fn reveal_opaque_ty(&self, ty: Ty<'tcx>) -> RevealedTy<'tcx> {
        fn reveal_inner<'tcx>(cx: &RustcPatCtxt<'_, 'tcx>, ty: Ty<'tcx>) -> RevealedTy<'tcx> {
            let ty::Alias(ty::Opaque, alias_ty) = *ty.kind() else { bug!() };
            if let Some(local_def_id) = alias_ty.def_id.as_local() {
                let key = ty::OpaqueTypeKey { def_id: local_def_id, args: alias_ty.args };
                if let Some(ty) = cx.reveal_opaque_key(key) {
                    return RevealedTy(ty);
                }
            }
            RevealedTy(ty)
        }
        if let ty::Alias(ty::Opaque, _) = ty.kind() {
            reveal_inner(self, ty)
        } else {
            RevealedTy(ty)
        }
    }

    /// Returns the hidden type corresponding to this key if the body under analysis is allowed to
    /// know it.
    fn reveal_opaque_key(&self, key: OpaqueTypeKey<'tcx>) -> Option<Ty<'tcx>> {
        self.typeck_results
            .concrete_opaque_types
            .get(&key.def_id)
            .map(|x| ty::EarlyBinder::bind(x.ty).instantiate(self.tcx, key.args))
    }
    // This can take a non-revealed `Ty` because it reveals opaques itself.
    pub fn is_uninhabited(&self, ty: Ty<'tcx>) -> bool {
        !ty.inhabited_predicate(self.tcx).apply_revealing_opaque(
            self.tcx,
            self.typing_env,
            self.module,
            &|key| self.reveal_opaque_key(key),
        )
    }

    /// Returns whether the given type is an enum from another crate declared `#[non_exhaustive]`.
    pub fn is_foreign_non_exhaustive_enum(&self, ty: RevealedTy<'tcx>) -> bool {
        match ty.kind() {
            ty::Adt(def, ..) => def.variant_list_has_applicable_non_exhaustive(),
            _ => false,
        }
    }

    /// Whether the range denotes the fictitious values before `isize::MIN` or after
    /// `usize::MAX`/`isize::MAX` (see doc of [`IntRange::split`] for why these exist).
    pub fn is_range_beyond_boundaries(&self, range: &IntRange, ty: RevealedTy<'tcx>) -> bool {
        ty.is_ptr_sized_integral() && {
            // The two invalid ranges are `NegInfinity..isize::MIN` (represented as
            // `NegInfinity..0`), and `{u,i}size::MAX+1..PosInfinity`. `hoist_pat_range_bdy`
            // converts `MAX+1` to `PosInfinity`, and we couldn't have `PosInfinity` in `range.lo`
            // otherwise.
            let lo = self.hoist_pat_range_bdy(range.lo, ty);
            matches!(lo, PatRangeBoundary::PosInfinity)
                || matches!(range.hi, MaybeInfiniteInt::Finite(0))
        }
    }

    pub(crate) fn variant_sub_tys(
        &self,
        ty: RevealedTy<'tcx>,
        variant: &'tcx VariantDef,
    ) -> impl Iterator<Item = (&'tcx FieldDef, RevealedTy<'tcx>)> {
        let ty::Adt(_, args) = ty.kind() else { bug!() };
        variant.fields.iter().map(move |field| {
            let ty = field.ty(self.tcx, args);
            // `field.ty()` doesn't normalize after instantiating.
            let ty = self.tcx.normalize_erasing_regions(self.typing_env, ty);
            let ty = self.reveal_opaque_ty(ty);
            (field, ty)
        })
    }

    pub(crate) fn variant_index_for_adt(
        ctor: &Constructor<'p, 'tcx>,
        adt: ty::AdtDef<'tcx>,
    ) -> VariantIdx {
        match *ctor {
            Variant(idx) => idx,
            Struct | UnionField => {
                assert!(!adt.is_enum());
                FIRST_VARIANT
            }
            _ => bug!("bad constructor {:?} for adt {:?}", ctor, adt),
        }
    }

    /// Returns the types of the fields for a given constructor. The result must have a length of
    /// `ctor.arity()`.
    pub(crate) fn ctor_sub_tys(
        &self,
        ctor: &Constructor<'p, 'tcx>,
        ty: RevealedTy<'tcx>,
    ) -> impl Iterator<Item = (RevealedTy<'tcx>, PrivateUninhabitedField)> + ExactSizeIterator {
        fn reveal_and_alloc<'a, 'tcx>(
            cx: &'a RustcPatCtxt<'_, 'tcx>,
            iter: impl Iterator<Item = Ty<'tcx>>,
        ) -> &'a [(RevealedTy<'tcx>, PrivateUninhabitedField)] {
            cx.dropless_arena.alloc_from_iter(
                iter.map(|ty| cx.reveal_opaque_ty(ty))
                    .map(|ty| (ty, PrivateUninhabitedField(false))),
            )
        }
        let cx = self;
        let slice = match ctor {
            Struct | Variant(_) | UnionField => match ty.kind() {
                ty::Tuple(fs) => reveal_and_alloc(cx, fs.iter()),
                ty::Adt(adt, args) => {
                    if adt.is_box() {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        reveal_and_alloc(cx, once(args.type_at(0)))
                    } else {
                        let variant =
                            &adt.variant(RustcPatCtxt::variant_index_for_adt(&ctor, *adt));
                        let tys = cx.variant_sub_tys(ty, variant).map(|(field, ty)| {
                            let is_visible =
                                adt.is_enum() || field.vis.is_accessible_from(cx.module, cx.tcx);
                            let is_uninhabited = cx.is_uninhabited(*ty);
                            let is_unstable =
                                cx.tcx.lookup_stability(field.did).is_some_and(|stab| {
                                    stab.is_unstable() && stab.feature != sym::rustc_private
                                });
                            let skip = is_uninhabited && (!is_visible || is_unstable);
                            (ty, PrivateUninhabitedField(skip))
                        });
                        cx.dropless_arena.alloc_from_iter(tys)
                    }
                }
                _ => bug!("Unexpected type for constructor `{ctor:?}`: {ty:?}"),
            },
            Ref => match ty.kind() {
                ty::Ref(_, rty, _) => reveal_and_alloc(cx, once(*rty)),
                _ => bug!("Unexpected type for `Ref` constructor: {ty:?}"),
            },
            Slice(slice) => match ty.builtin_index() {
                Some(ty) => {
                    let arity = slice.arity();
                    reveal_and_alloc(cx, (0..arity).map(|_| ty))
                }
                None => bug!("bad slice pattern {:?} {:?}", ctor, ty),
            },
            DerefPattern(pointee_ty) => reveal_and_alloc(cx, once(pointee_ty.inner())),
            Bool(..) | IntRange(..) | F16Range(..) | F32Range(..) | F64Range(..)
            | F128Range(..) | Str(..) | Opaque(..) | Never | NonExhaustive | Hidden | Missing
            | PrivateUninhabited | Wildcard => &[],
            Or => {
                bug!("called `Fields::wildcards` on an `Or` ctor")
            }
        };
        slice.iter().copied()
    }

    /// The number of fields for this constructor.
    pub(crate) fn ctor_arity(&self, ctor: &Constructor<'p, 'tcx>, ty: RevealedTy<'tcx>) -> usize {
        match ctor {
            Struct | Variant(_) | UnionField => match ty.kind() {
                ty::Tuple(fs) => fs.len(),
                ty::Adt(adt, ..) => {
                    if adt.is_box() {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        1
                    } else {
                        let variant_idx = RustcPatCtxt::variant_index_for_adt(&ctor, *adt);
                        adt.variant(variant_idx).fields.len()
                    }
                }
                _ => bug!("Unexpected type for constructor `{ctor:?}`: {ty:?}"),
            },
            Ref | DerefPattern(_) => 1,
            Slice(slice) => slice.arity(),
            Bool(..) | IntRange(..) | F16Range(..) | F32Range(..) | F64Range(..)
            | F128Range(..) | Str(..) | Opaque(..) | Never | NonExhaustive | Hidden | Missing
            | PrivateUninhabited | Wildcard => 0,
            Or => bug!("The `Or` constructor doesn't have a fixed arity"),
        }
    }

    /// Creates a set that represents all the constructors of `ty`.
    ///
    /// See [`crate::constructor`] for considerations of emptiness.
    pub fn ctors_for_ty(
        &self,
        ty: RevealedTy<'tcx>,
    ) -> Result<ConstructorSet<'p, 'tcx>, ErrorGuaranteed> {
        let cx = self;
        let make_uint_range = |start, end| {
            IntRange::from_range(
                MaybeInfiniteInt::new_finite_uint(start),
                MaybeInfiniteInt::new_finite_uint(end),
                RangeEnd::Included,
            )
        };
        // Abort on type error.
        ty.error_reported()?;
        // This determines the set of all possible constructors for the type `ty`. For numbers,
        // arrays and slices we use ranges and variable-length slices when appropriate.
        Ok(match ty.kind() {
            ty::Bool => ConstructorSet::Bool,
            ty::Char => {
                // The valid Unicode Scalar Value ranges.
                ConstructorSet::Integers {
                    range_1: make_uint_range('\u{0000}' as u128, '\u{D7FF}' as u128),
                    range_2: Some(make_uint_range('\u{E000}' as u128, '\u{10FFFF}' as u128)),
                }
            }
            &ty::Int(ity) => {
                let range = if ty.is_ptr_sized_integral() {
                    // The min/max values of `isize` are not allowed to be observed.
                    IntRange {
                        lo: MaybeInfiniteInt::NegInfinity,
                        hi: MaybeInfiniteInt::PosInfinity,
                    }
                } else {
                    let size = Integer::from_int_ty(&cx.tcx, ity).size().bits();
                    let min = 1u128 << (size - 1);
                    let max = min - 1;
                    let min = MaybeInfiniteInt::new_finite_int(min, size);
                    let max = MaybeInfiniteInt::new_finite_int(max, size);
                    IntRange::from_range(min, max, RangeEnd::Included)
                };
                ConstructorSet::Integers { range_1: range, range_2: None }
            }
            &ty::Uint(uty) => {
                let range = if ty.is_ptr_sized_integral() {
                    // The max value of `usize` is not allowed to be observed.
                    let lo = MaybeInfiniteInt::new_finite_uint(0);
                    IntRange { lo, hi: MaybeInfiniteInt::PosInfinity }
                } else {
                    let size = Integer::from_uint_ty(&cx.tcx, uty).size();
                    let max = size.truncate(u128::MAX);
                    make_uint_range(0, max)
                };
                ConstructorSet::Integers { range_1: range, range_2: None }
            }
            ty::Slice(sub_ty) => ConstructorSet::Slice {
                array_len: None,
                subtype_is_empty: cx.is_uninhabited(*sub_ty),
            },
            ty::Array(sub_ty, len) => {
                // We treat arrays of a constant but unknown length like slices.
                ConstructorSet::Slice {
                    array_len: len.try_to_target_usize(cx.tcx).map(|l| l as usize),
                    subtype_is_empty: cx.is_uninhabited(*sub_ty),
                }
            }
            ty::Adt(def, args) if def.is_enum() => {
                let is_declared_nonexhaustive = cx.is_foreign_non_exhaustive_enum(ty);
                if def.variants().is_empty() && !is_declared_nonexhaustive {
                    ConstructorSet::NoConstructors
                } else {
                    let mut variants =
                        IndexVec::from_elem(VariantVisibility::Visible, def.variants());
                    for (idx, v) in def.variants().iter_enumerated() {
                        let variant_def_id = def.variant(idx).def_id;
                        // Visibly uninhabited variants.
                        let is_inhabited = v
                            .inhabited_predicate(cx.tcx, *def)
                            .instantiate(cx.tcx, args)
                            .apply_revealing_opaque(cx.tcx, cx.typing_env, cx.module, &|key| {
                                cx.reveal_opaque_key(key)
                            });
                        // Variants that depend on a disabled unstable feature.
                        let is_unstable = matches!(
                            cx.tcx.eval_stability(variant_def_id, None, DUMMY_SP, None),
                            EvalResult::Deny { .. }
                        );
                        // Foreign `#[doc(hidden)]` variants.
                        let is_doc_hidden =
                            cx.tcx.is_doc_hidden(variant_def_id) && !variant_def_id.is_local();
                        let visibility = if !is_inhabited {
                            // FIXME: handle empty+hidden
                            VariantVisibility::Empty
                        } else if is_unstable || is_doc_hidden {
                            VariantVisibility::Hidden
                        } else {
                            VariantVisibility::Visible
                        };
                        variants[idx] = visibility;
                    }

                    ConstructorSet::Variants { variants, non_exhaustive: is_declared_nonexhaustive }
                }
            }
            ty::Adt(def, _) if def.is_union() => ConstructorSet::Union,
            ty::Adt(..) | ty::Tuple(..) => {
                ConstructorSet::Struct { empty: cx.is_uninhabited(ty.inner()) }
            }
            ty::Ref(..) => ConstructorSet::Ref,
            ty::Never => ConstructorSet::NoConstructors,
            // This type is one for which we cannot list constructors, like `str` or `f64`.
            // FIXME(Nadrieril): which of these are actually allowed?
            ty::Float(_)
            | ty::Str
            | ty::Foreign(_)
            | ty::RawPtr(_, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(..)
            | ty::Pat(_, _)
            | ty::Dynamic(_, _, _)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(_, _)
            | ty::UnsafeBinder(_)
            | ty::Alias(_, _)
            | ty::Param(_)
            | ty::Error(_) => ConstructorSet::Unlistable,
            ty::CoroutineWitness(_, _) | ty::Bound(_, _) | ty::Placeholder(_) | ty::Infer(_) => {
                bug!("Encountered unexpected type in `ConstructorSet::for_ty`: {ty:?}")
            }
        })
    }

    pub(crate) fn lower_pat_range_bdy(
        &self,
        bdy: PatRangeBoundary<'tcx>,
        ty: RevealedTy<'tcx>,
    ) -> MaybeInfiniteInt {
        match bdy {
            PatRangeBoundary::NegInfinity => MaybeInfiniteInt::NegInfinity,
            PatRangeBoundary::Finite(value) => {
                let bits = value.eval_bits(self.tcx, self.typing_env);
                match *ty.kind() {
                    ty::Int(ity) => {
                        let size = Integer::from_int_ty(&self.tcx, ity).size().bits();
                        MaybeInfiniteInt::new_finite_int(bits, size)
                    }
                    _ => MaybeInfiniteInt::new_finite_uint(bits),
                }
            }
            PatRangeBoundary::PosInfinity => MaybeInfiniteInt::PosInfinity,
        }
    }

    /// Note: the input patterns must have been lowered through
    /// `rustc_mir_build::thir::pattern::check_match::MatchVisitor::lower_pattern`.
    pub fn lower_pat(&self, pat: &'p Pat<'tcx>) -> DeconstructedPat<'p, 'tcx> {
        let cx = self;
        let ty = cx.reveal_opaque_ty(pat.ty);
        let ctor;
        let arity;
        let fields: Vec<_>;
        match &pat.kind {
            PatKind::AscribeUserType { subpattern, .. }
            | PatKind::ExpandedConstant { subpattern, .. } => return self.lower_pat(subpattern),
            PatKind::Binding { subpattern: Some(subpat), .. } => return self.lower_pat(subpat),
            PatKind::Missing | PatKind::Binding { subpattern: None, .. } | PatKind::Wild => {
                ctor = Wildcard;
                fields = vec![];
                arity = 0;
            }
            PatKind::Deref { subpattern } => {
                fields = vec![self.lower_pat(subpattern).at_index(0)];
                arity = 1;
                ctor = match ty.kind() {
                    // This is a box pattern.
                    ty::Adt(adt, ..) if adt.is_box() => Struct,
                    ty::Ref(..) => Ref,
                    _ => span_bug!(
                        pat.span,
                        "pattern has unexpected type: pat: {:?}, ty: {:?}",
                        pat.kind,
                        ty.inner()
                    ),
                };
            }
            PatKind::DerefPattern { subpattern, .. } => {
                // NB(deref_patterns): This assumes the deref pattern is matching on a trusted
                // `DerefPure` type. If the `Deref` impl isn't trusted, exhaustiveness must take
                // into account that multiple calls to deref may return different results. Hence
                // multiple deref! patterns cannot be exhaustive together unless each is exhaustive
                // by itself.
                fields = vec![self.lower_pat(subpattern).at_index(0)];
                arity = 1;
                ctor = DerefPattern(cx.reveal_opaque_ty(subpattern.ty));
            }
            PatKind::Leaf { subpatterns } | PatKind::Variant { subpatterns, .. } => {
                match ty.kind() {
                    ty::Tuple(fs) => {
                        ctor = Struct;
                        arity = fs.len();
                        fields = subpatterns
                            .iter()
                            .map(|ipat| self.lower_pat(&ipat.pattern).at_index(ipat.field.index()))
                            .collect();
                    }
                    ty::Adt(adt, _) if adt.is_box() => {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        // FIXME(Nadrieril): A `Box` can in theory be matched either with `Box(_,
                        // _)` or a box pattern. As a hack to avoid an ICE with the former, we
                        // ignore other fields than the first one. This will trigger an error later
                        // anyway.
                        // See https://github.com/rust-lang/rust/issues/82772,
                        // explanation: https://github.com/rust-lang/rust/pull/82789#issuecomment-796921977
                        // The problem is that we can't know from the type whether we'll match
                        // normally or through box-patterns. We'll have to figure out a proper
                        // solution when we introduce generalized deref patterns. Also need to
                        // prevent mixing of those two options.
                        let pattern = subpatterns.into_iter().find(|pat| pat.field.index() == 0);
                        if let Some(pat) = pattern {
                            fields = vec![self.lower_pat(&pat.pattern).at_index(0)];
                        } else {
                            fields = vec![];
                        }
                        ctor = Struct;
                        arity = 1;
                    }
                    ty::Adt(adt, _) => {
                        ctor = match pat.kind {
                            PatKind::Leaf { .. } if adt.is_union() => UnionField,
                            PatKind::Leaf { .. } => Struct,
                            PatKind::Variant { variant_index, .. } => Variant(variant_index),
                            _ => bug!(),
                        };
                        let variant =
                            &adt.variant(RustcPatCtxt::variant_index_for_adt(&ctor, *adt));
                        arity = variant.fields.len();
                        fields = subpatterns
                            .iter()
                            .map(|ipat| self.lower_pat(&ipat.pattern).at_index(ipat.field.index()))
                            .collect();
                    }
                    _ => span_bug!(
                        pat.span,
                        "pattern has unexpected type: pat: {:?}, ty: {}",
                        pat.kind,
                        ty.inner()
                    ),
                }
            }
            PatKind::Constant { value } => {
                match ty.kind() {
                    ty::Bool => {
                        ctor = match value.try_eval_bool(cx.tcx, cx.typing_env) {
                            Some(b) => Bool(b),
                            None => Opaque(OpaqueId::new()),
                        };
                        fields = vec![];
                        arity = 0;
                    }
                    ty::Char | ty::Int(_) | ty::Uint(_) => {
                        ctor = match value.try_eval_bits(cx.tcx, cx.typing_env) {
                            Some(bits) => {
                                let x = match *ty.kind() {
                                    ty::Int(ity) => {
                                        let size = Integer::from_int_ty(&cx.tcx, ity).size().bits();
                                        MaybeInfiniteInt::new_finite_int(bits, size)
                                    }
                                    _ => MaybeInfiniteInt::new_finite_uint(bits),
                                };
                                IntRange(IntRange::from_singleton(x))
                            }
                            None => Opaque(OpaqueId::new()),
                        };
                        fields = vec![];
                        arity = 0;
                    }
                    ty::Float(ty::FloatTy::F16) => {
                        ctor = match value.try_eval_bits(cx.tcx, cx.typing_env) {
                            Some(bits) => {
                                use rustc_apfloat::Float;
                                let value = rustc_apfloat::ieee::Half::from_bits(bits);
                                F16Range(value, value, RangeEnd::Included)
                            }
                            None => Opaque(OpaqueId::new()),
                        };
                        fields = vec![];
                        arity = 0;
                    }
                    ty::Float(ty::FloatTy::F32) => {
                        ctor = match value.try_eval_bits(cx.tcx, cx.typing_env) {
                            Some(bits) => {
                                use rustc_apfloat::Float;
                                let value = rustc_apfloat::ieee::Single::from_bits(bits);
                                F32Range(value, value, RangeEnd::Included)
                            }
                            None => Opaque(OpaqueId::new()),
                        };
                        fields = vec![];
                        arity = 0;
                    }
                    ty::Float(ty::FloatTy::F64) => {
                        ctor = match value.try_eval_bits(cx.tcx, cx.typing_env) {
                            Some(bits) => {
                                use rustc_apfloat::Float;
                                let value = rustc_apfloat::ieee::Double::from_bits(bits);
                                F64Range(value, value, RangeEnd::Included)
                            }
                            None => Opaque(OpaqueId::new()),
                        };
                        fields = vec![];
                        arity = 0;
                    }
                    ty::Float(ty::FloatTy::F128) => {
                        ctor = match value.try_eval_bits(cx.tcx, cx.typing_env) {
                            Some(bits) => {
                                use rustc_apfloat::Float;
                                let value = rustc_apfloat::ieee::Quad::from_bits(bits);
                                F128Range(value, value, RangeEnd::Included)
                            }
                            None => Opaque(OpaqueId::new()),
                        };
                        fields = vec![];
                        arity = 0;
                    }
                    ty::Ref(_, t, _) if t.is_str() => {
                        // We want a `&str` constant to behave like a `Deref` pattern, to be compatible
                        // with other `Deref` patterns. This could have been done in `const_to_pat`,
                        // but that causes issues with the rest of the matching code.
                        // So here, the constructor for a `"foo"` pattern is `&` (represented by
                        // `Ref`), and has one field. That field has constructor `Str(value)` and no
                        // subfields.
                        // Note: `t` is `str`, not `&str`.
                        let ty = self.reveal_opaque_ty(*t);
                        let subpattern = DeconstructedPat::new(Str(*value), Vec::new(), 0, ty, pat);
                        ctor = Ref;
                        fields = vec![subpattern.at_index(0)];
                        arity = 1;
                    }
                    // All constants that can be structurally matched have already been expanded
                    // into the corresponding `Pat`s by `const_to_pat`. Constants that remain are
                    // opaque.
                    _ => {
                        ctor = Opaque(OpaqueId::new());
                        fields = vec![];
                        arity = 0;
                    }
                }
            }
            PatKind::Range(patrange) => {
                let PatRange { lo, hi, end, .. } = patrange.as_ref();
                let end = match end {
                    rustc_hir::RangeEnd::Included => RangeEnd::Included,
                    rustc_hir::RangeEnd::Excluded => RangeEnd::Excluded,
                };
                ctor = match ty.kind() {
                    ty::Char | ty::Int(_) | ty::Uint(_) => {
                        let lo = cx.lower_pat_range_bdy(*lo, ty);
                        let hi = cx.lower_pat_range_bdy(*hi, ty);
                        IntRange(IntRange::from_range(lo, hi, end))
                    }
                    ty::Float(fty) => {
                        use rustc_apfloat::Float;
                        let lo = lo.as_finite().map(|c| c.eval_bits(cx.tcx, cx.typing_env));
                        let hi = hi.as_finite().map(|c| c.eval_bits(cx.tcx, cx.typing_env));
                        match fty {
                            ty::FloatTy::F16 => {
                                use rustc_apfloat::ieee::Half;
                                let lo = lo.map(Half::from_bits).unwrap_or(-Half::INFINITY);
                                let hi = hi.map(Half::from_bits).unwrap_or(Half::INFINITY);
                                F16Range(lo, hi, end)
                            }
                            ty::FloatTy::F32 => {
                                use rustc_apfloat::ieee::Single;
                                let lo = lo.map(Single::from_bits).unwrap_or(-Single::INFINITY);
                                let hi = hi.map(Single::from_bits).unwrap_or(Single::INFINITY);
                                F32Range(lo, hi, end)
                            }
                            ty::FloatTy::F64 => {
                                use rustc_apfloat::ieee::Double;
                                let lo = lo.map(Double::from_bits).unwrap_or(-Double::INFINITY);
                                let hi = hi.map(Double::from_bits).unwrap_or(Double::INFINITY);
                                F64Range(lo, hi, end)
                            }
                            ty::FloatTy::F128 => {
                                use rustc_apfloat::ieee::Quad;
                                let lo = lo.map(Quad::from_bits).unwrap_or(-Quad::INFINITY);
                                let hi = hi.map(Quad::from_bits).unwrap_or(Quad::INFINITY);
                                F128Range(lo, hi, end)
                            }
                        }
                    }
                    _ => span_bug!(pat.span, "invalid type for range pattern: {}", ty.inner()),
                };
                fields = vec![];
                arity = 0;
            }
            PatKind::Array { prefix, slice, suffix } | PatKind::Slice { prefix, slice, suffix } => {
                let array_len = match ty.kind() {
                    ty::Array(_, length) => Some(
                        length
                            .try_to_target_usize(cx.tcx)
                            .expect("expected len of array pat to be definite")
                            as usize,
                    ),
                    ty::Slice(_) => None,
                    _ => span_bug!(pat.span, "bad ty {} for slice pattern", ty.inner()),
                };
                let kind = if slice.is_some() {
                    SliceKind::VarLen(prefix.len(), suffix.len())
                } else {
                    SliceKind::FixedLen(prefix.len() + suffix.len())
                };
                ctor = Slice(Slice::new(array_len, kind));
                fields = prefix
                    .iter()
                    .chain(suffix.iter())
                    .map(|p| self.lower_pat(&*p))
                    .enumerate()
                    .map(|(i, p)| p.at_index(i))
                    .collect();
                arity = kind.arity();
            }
            PatKind::Or { .. } => {
                ctor = Or;
                let pats = expand_or_pat(pat);
                fields = pats
                    .into_iter()
                    .map(|p| self.lower_pat(p))
                    .enumerate()
                    .map(|(i, p)| p.at_index(i))
                    .collect();
                arity = fields.len();
            }
            PatKind::Never => {
                // A never pattern matches all the values of its type (namely none). Moreover it
                // must be compatible with other constructors, since we can use `!` on a type like
                // `Result<!, !>` which has other constructors. Hence we lower it as a wildcard.
                ctor = Wildcard;
                fields = vec![];
                arity = 0;
            }
            PatKind::Error(_) => {
                ctor = Opaque(OpaqueId::new());
                fields = vec![];
                arity = 0;
            }
        }
        DeconstructedPat::new(ctor, fields, arity, ty, pat)
    }

    /// Convert back to a `thir::PatRangeBoundary` for diagnostic purposes.
    /// Note: it is possible to get `isize/usize::MAX+1` here, as explained in the doc for
    /// [`IntRange::split`]. This cannot be represented as a `Const`, so we represent it with
    /// `PosInfinity`.
    fn hoist_pat_range_bdy(
        &self,
        miint: MaybeInfiniteInt,
        ty: RevealedTy<'tcx>,
    ) -> PatRangeBoundary<'tcx> {
        use MaybeInfiniteInt::*;
        let tcx = self.tcx;
        match miint {
            NegInfinity => PatRangeBoundary::NegInfinity,
            Finite(_) => {
                let size = ty.primitive_size(tcx);
                let bits = match *ty.kind() {
                    ty::Int(_) => miint.as_finite_int(size.bits()).unwrap(),
                    _ => miint.as_finite_uint().unwrap(),
                };
                match ScalarInt::try_from_uint(bits, size) {
                    Some(scalar) => {
                        let value = mir::Const::from_scalar(tcx, scalar.into(), ty.inner());
                        PatRangeBoundary::Finite(value)
                    }
                    // The value doesn't fit. Since `x >= 0` and 0 always encodes the minimum value
                    // for a type, the problem isn't that the value is too small. So it must be too
                    // large.
                    None => PatRangeBoundary::PosInfinity,
                }
            }
            PosInfinity => PatRangeBoundary::PosInfinity,
        }
    }

    /// Prints an [`IntRange`] to a string for diagnostic purposes.
    fn print_pat_range(&self, range: &IntRange, ty: RevealedTy<'tcx>) -> String {
        use MaybeInfiniteInt::*;
        let cx = self;
        if matches!((range.lo, range.hi), (NegInfinity, PosInfinity)) {
            "_".to_string()
        } else if range.is_singleton() {
            let lo = cx.hoist_pat_range_bdy(range.lo, ty);
            let value = lo.as_finite().unwrap();
            value.to_string()
        } else {
            // We convert to an inclusive range for diagnostics.
            let mut end = rustc_hir::RangeEnd::Included;
            let mut lo = cx.hoist_pat_range_bdy(range.lo, ty);
            if matches!(lo, PatRangeBoundary::PosInfinity) {
                // The only reason to get `PosInfinity` here is the special case where
                // `hoist_pat_range_bdy` found `{u,i}size::MAX+1`. So the range denotes the
                // fictitious values after `{u,i}size::MAX` (see [`IntRange::split`] for why we do
                // this). We show this to the user as `usize::MAX..` which is slightly incorrect but
                // probably clear enough.
                lo = PatRangeBoundary::Finite(ty.numeric_max_val(cx.tcx).unwrap());
            }
            let hi = if let Some(hi) = range.hi.minus_one() {
                hi
            } else {
                // The range encodes `..ty::MIN`, so we can't convert it to an inclusive range.
                end = rustc_hir::RangeEnd::Excluded;
                range.hi
            };
            let hi = cx.hoist_pat_range_bdy(hi, ty);
            PatRange { lo, hi, end, ty: ty.inner() }.to_string()
        }
    }

    /// Prints a [`WitnessPat`] to an owned string, for diagnostic purposes.
    ///
    /// This panics for patterns that don't appear in diagnostics, like float ranges.
    pub fn print_witness_pat(&self, pat: &WitnessPat<'p, 'tcx>) -> String {
        let cx = self;
        let print = |p| cx.print_witness_pat(p);
        match pat.ctor() {
            Bool(b) => b.to_string(),
            Str(s) => s.to_string(),
            IntRange(range) => return self.print_pat_range(range, *pat.ty()),
            Struct if pat.ty().is_box() => {
                // Outside of the `alloc` crate, the only way to create a struct pattern
                // of type `Box` is to use a `box` pattern via #[feature(box_patterns)].
                format!("box {}", print(&pat.fields[0]))
            }
            Struct | Variant(_) | UnionField => {
                let enum_info = match *pat.ty().kind() {
                    ty::Adt(adt_def, _) if adt_def.is_enum() => EnumInfo::Enum {
                        adt_def,
                        variant_index: RustcPatCtxt::variant_index_for_adt(pat.ctor(), adt_def),
                    },
                    ty::Adt(..) | ty::Tuple(..) => EnumInfo::NotEnum,
                    _ => bug!("unexpected ctor for type {:?} {:?}", pat.ctor(), *pat.ty()),
                };

                let subpatterns = pat
                    .iter_fields()
                    .enumerate()
                    .map(|(i, pat)| print::FieldPat {
                        field: FieldIdx::new(i),
                        pattern: print(pat),
                        is_wildcard: would_print_as_wildcard(cx.tcx, pat),
                    })
                    .collect::<Vec<_>>();

                let mut s = String::new();
                print::write_struct_like(
                    &mut s,
                    self.tcx,
                    pat.ty().inner(),
                    &enum_info,
                    &subpatterns,
                )
                .unwrap();
                s
            }
            Ref => {
                let mut s = String::new();
                print::write_ref_like(&mut s, pat.ty().inner(), &print(&pat.fields[0])).unwrap();
                s
            }
            DerefPattern(_) => format!("deref!({})", print(&pat.fields[0])),
            Slice(slice) => {
                let (prefix_len, has_dot_dot) = match slice.kind {
                    SliceKind::FixedLen(len) => (len, false),
                    SliceKind::VarLen(prefix_len, _) => (prefix_len, true),
                };

                let (mut prefix, mut suffix) = pat.fields.split_at(prefix_len);

                // If the pattern contains a `..`, but is applied to values of statically-known
                // length (arrays), then we can slightly simplify diagnostics by merging any
                // adjacent wildcard patterns into the `..`: `[x, _, .., _, y]` => `[x, .., y]`.
                // (This simplification isn't allowed for slice values, because in that case
                // `[x, .., y]` would match some slices that `[x, _, .., _, y]` would not.)
                if has_dot_dot && slice.array_len.is_some() {
                    while let [rest @ .., last] = prefix
                        && would_print_as_wildcard(cx.tcx, last)
                    {
                        prefix = rest;
                    }
                    while let [first, rest @ ..] = suffix
                        && would_print_as_wildcard(cx.tcx, first)
                    {
                        suffix = rest;
                    }
                }

                let prefix = prefix.iter().map(print).collect::<Vec<_>>();
                let suffix = suffix.iter().map(print).collect::<Vec<_>>();

                let mut s = String::new();
                print::write_slice_like(&mut s, &prefix, has_dot_dot, &suffix).unwrap();
                s
            }
            Never if self.tcx.features().never_patterns() => "!".to_string(),
            Never | Wildcard | NonExhaustive | Hidden | PrivateUninhabited => "_".to_string(),
            Missing { .. } => bug!(
                "trying to convert a `Missing` constructor into a `Pat`; this is probably a bug,
                `Missing` should have been processed in `apply_constructors`"
            ),
            F16Range(..) | F32Range(..) | F64Range(..) | F128Range(..) | Opaque(..) | Or => {
                bug!("can't convert to pattern: {:?}", pat)
            }
        }
    }
}

/// Returns `true` if the given pattern would be printed as a wildcard (`_`).
fn would_print_as_wildcard(tcx: TyCtxt<'_>, p: &WitnessPat<'_, '_>) -> bool {
    match p.ctor() {
        Constructor::IntRange(IntRange {
            lo: MaybeInfiniteInt::NegInfinity,
            hi: MaybeInfiniteInt::PosInfinity,
        })
        | Constructor::Wildcard
        | Constructor::NonExhaustive
        | Constructor::Hidden
        | Constructor::PrivateUninhabited => true,
        Constructor::Never if !tcx.features().never_patterns() => true,
        _ => false,
    }
}

impl<'p, 'tcx: 'p> PatCx for RustcPatCtxt<'p, 'tcx> {
    type Ty = RevealedTy<'tcx>;
    type Error = ErrorGuaranteed;
    type VariantIdx = VariantIdx;
    type StrLit = Const<'tcx>;
    type ArmData = HirId;
    type PatData = &'p Pat<'tcx>;

    fn is_exhaustive_patterns_feature_on(&self) -> bool {
        self.tcx.features().exhaustive_patterns()
    }

    fn ctor_arity(&self, ctor: &crate::constructor::Constructor<Self>, ty: &Self::Ty) -> usize {
        self.ctor_arity(ctor, *ty)
    }
    fn ctor_sub_tys(
        &self,
        ctor: &crate::constructor::Constructor<Self>,
        ty: &Self::Ty,
    ) -> impl Iterator<Item = (Self::Ty, PrivateUninhabitedField)> + ExactSizeIterator {
        self.ctor_sub_tys(ctor, *ty)
    }
    fn ctors_for_ty(
        &self,
        ty: &Self::Ty,
    ) -> Result<crate::constructor::ConstructorSet<Self>, Self::Error> {
        self.ctors_for_ty(*ty)
    }

    fn write_variant_name(
        f: &mut fmt::Formatter<'_>,
        ctor: &crate::constructor::Constructor<Self>,
        ty: &Self::Ty,
    ) -> fmt::Result {
        if let ty::Adt(adt, _) = ty.kind() {
            if adt.is_box() {
                write!(f, "Box")?
            } else {
                let variant = adt.variant(Self::variant_index_for_adt(ctor, *adt));
                write!(f, "{}", variant.name)?;
            }
        }
        Ok(())
    }

    fn bug(&self, fmt: fmt::Arguments<'_>) -> Self::Error {
        span_bug!(self.scrut_span, "{}", fmt)
    }

    fn lint_overlapping_range_endpoints(
        &self,
        pat: &crate::pat::DeconstructedPat<Self>,
        overlaps_on: IntRange,
        overlaps_with: &[&crate::pat::DeconstructedPat<Self>],
    ) {
        let overlap_as_pat = self.print_pat_range(&overlaps_on, *pat.ty());
        let overlaps: Vec<_> = overlaps_with
            .iter()
            .map(|pat| pat.data().span)
            .map(|span| errors::Overlap { range: overlap_as_pat.to_string(), span })
            .collect();
        let pat_span = pat.data().span;
        self.tcx.emit_node_span_lint(
            lint::builtin::OVERLAPPING_RANGE_ENDPOINTS,
            self.match_lint_level,
            pat_span,
            errors::OverlappingRangeEndpoints { overlap: overlaps, range: pat_span },
        );
    }

    fn complexity_exceeded(&self) -> Result<(), Self::Error> {
        let span = self.whole_match_span.unwrap_or(self.scrut_span);
        Err(self.tcx.dcx().span_err(span, "reached pattern complexity limit"))
    }

    fn lint_non_contiguous_range_endpoints(
        &self,
        pat: &crate::pat::DeconstructedPat<Self>,
        gap: IntRange,
        gapped_with: &[&crate::pat::DeconstructedPat<Self>],
    ) {
        let &thir_pat = pat.data();
        let thir::PatKind::Range(range) = &thir_pat.kind else { return };
        // Only lint when the left range is an exclusive range.
        if range.end != rustc_hir::RangeEnd::Excluded {
            return;
        }
        // `pat` is an exclusive range like `lo..gap`. `gapped_with` contains ranges that start with
        // `gap+1`.
        let suggested_range: String = {
            // Suggest `lo..=gap` instead.
            let mut suggested_range = PatRange::clone(range);
            suggested_range.end = rustc_hir::RangeEnd::Included;
            suggested_range.to_string()
        };
        let gap_as_pat = self.print_pat_range(&gap, *pat.ty());
        if gapped_with.is_empty() {
            // If `gapped_with` is empty, `gap == T::MAX`.
            self.tcx.emit_node_span_lint(
                lint::builtin::NON_CONTIGUOUS_RANGE_ENDPOINTS,
                self.match_lint_level,
                thir_pat.span,
                errors::ExclusiveRangeMissingMax {
                    // Point at this range.
                    first_range: thir_pat.span,
                    // That's the gap that isn't covered.
                    max: gap_as_pat,
                    // Suggest `lo..=max` instead.
                    suggestion: suggested_range,
                },
            );
        } else {
            self.tcx.emit_node_span_lint(
                lint::builtin::NON_CONTIGUOUS_RANGE_ENDPOINTS,
                self.match_lint_level,
                thir_pat.span,
                errors::ExclusiveRangeMissingGap {
                    // Point at this range.
                    first_range: thir_pat.span,
                    // That's the gap that isn't covered.
                    gap: gap_as_pat.to_string(),
                    // Suggest `lo..=gap` instead.
                    suggestion: suggested_range,
                    // All these ranges skipped over `gap` which we think is probably a
                    // mistake.
                    gap_with: gapped_with
                        .iter()
                        .map(|pat| errors::GappedRange {
                            span: pat.data().span,
                            gap: gap_as_pat.to_string(),
                            first_range: range.to_string(),
                        })
                        .collect(),
                },
            );
        }
    }
}

/// Recursively expand this pattern into its subpatterns. Only useful for or-patterns.
fn expand_or_pat<'p, 'tcx>(pat: &'p Pat<'tcx>) -> Vec<&'p Pat<'tcx>> {
    fn expand<'p, 'tcx>(pat: &'p Pat<'tcx>, vec: &mut Vec<&'p Pat<'tcx>>) {
        if let PatKind::Or { pats } = &pat.kind {
            for pat in pats.iter() {
                expand(pat, vec);
            }
        } else {
            vec.push(pat)
        }
    }

    let mut pats = Vec::new();
    expand(pat, &mut pats);
    pats
}

/// The entrypoint for this crate. Computes whether a match is exhaustive and which of its arms are
/// useful, and runs some lints.
pub fn analyze_match<'p, 'tcx>(
    tycx: &RustcPatCtxt<'p, 'tcx>,
    arms: &[MatchArm<'p, 'tcx>],
    scrut_ty: Ty<'tcx>,
) -> Result<UsefulnessReport<'p, 'tcx>, ErrorGuaranteed> {
    let scrut_ty = tycx.reveal_opaque_ty(scrut_ty);

    // The analysis doesn't support deref patterns mixed with normal constructors; error if present.
    // FIXME(deref_patterns): This only needs to run when a deref pattern was found during lowering.
    if tycx.tcx.features().deref_patterns() {
        let pat_column = PatternColumn::new(arms);
        detect_mixed_deref_pat_ctors(tycx, &pat_column)?;
    }

    let scrut_validity = PlaceValidity::from_bool(tycx.known_valid_scrutinee);
    let report = compute_match_usefulness(
        tycx,
        arms,
        scrut_ty,
        scrut_validity,
        tycx.tcx.pattern_complexity_limit().0,
    )?;

    // Run the non_exhaustive_omitted_patterns lint. Only run on refutable patterns to avoid hitting
    // `if let`s. Only run if the match is exhaustive otherwise the error is redundant.
    if tycx.refutable && report.non_exhaustiveness_witnesses.is_empty() {
        let pat_column = PatternColumn::new(arms);
        lint_nonexhaustive_missing_variants(tycx, arms, &pat_column, scrut_ty)?;
    }

    Ok(report)
}

// FIXME(deref_patterns): Currently it's the responsibility of the frontend (rustc or rust-analyzer)
// to ensure that deref patterns don't appear in the same column as normal constructors. Deref
// patterns aren't currently implemented in rust-analyzer, but should they be, the columnwise check
// here could be made generic and shared between frontends.
fn detect_mixed_deref_pat_ctors<'p, 'tcx>(
    cx: &RustcPatCtxt<'p, 'tcx>,
    column: &PatternColumn<'p, RustcPatCtxt<'p, 'tcx>>,
) -> Result<(), ErrorGuaranteed> {
    let Some(&ty) = column.head_ty() else {
        return Ok(());
    };

    // Check for a mix of deref patterns and normal constructors.
    let mut normal_ctor_span = None;
    let mut deref_pat_span = None;
    for pat in column.iter() {
        match pat.ctor() {
            // The analysis can handle mixing deref patterns with wildcards and opaque patterns.
            Wildcard | Opaque(_) => {}
            DerefPattern(_) => deref_pat_span = Some(pat.data().span),
            // Nothing else can be compared to deref patterns in `Constructor::is_covered_by`.
            _ => normal_ctor_span = Some(pat.data().span),
        }
    }
    if let Some(normal_constructor_label) = normal_ctor_span
        && let Some(deref_pattern_label) = deref_pat_span
    {
        return Err(cx.tcx.dcx().emit_err(errors::MixedDerefPatternConstructors {
            spans: vec![deref_pattern_label, normal_constructor_label],
            smart_pointer_ty: ty.inner(),
            deref_pattern_label,
            normal_constructor_label,
        }));
    }

    // Specialize and recurse into the patterns' fields.
    let set = column.analyze_ctors(cx, &ty)?;
    for ctor in set.present {
        for specialized_column in column.specialize(cx, &ty, &ctor).iter() {
            detect_mixed_deref_pat_ctors(cx, specialized_column)?;
        }
    }
    Ok(())
}
