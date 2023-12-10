//! As explained in [`crate::usefulness`], values and patterns are made from constructors applied to
//! fields. This file defines types that represent patterns in this way.
use std::cell::Cell;
use std::fmt;
use std::iter::once;

use smallvec::{smallvec, SmallVec};

use rustc_data_structures::captures::Captures;
use rustc_hir::RangeEnd;
use rustc_index::Idx;
use rustc_middle::mir;
use rustc_middle::thir::{FieldPat, Pat, PatKind, PatRange};
use rustc_middle::ty::{self, Ty, VariantDef};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::FieldIdx;

use self::Constructor::*;
use self::SliceKind::*;

use crate::constructor::{Constructor, IntRange, MaybeInfiniteInt, OpaqueId, Slice, SliceKind};
use crate::usefulness::{MatchCheckCtxt, PatCtxt};

/// A value can be decomposed into a constructor applied to some fields. This struct represents
/// those fields, generalized to allow patterns in each field. See also `Constructor`.
///
/// This is constructed for a constructor using [`Fields::wildcards()`]. The idea is that
/// [`Fields::wildcards()`] constructs a list of fields where all entries are wildcards, and then
/// given a pattern we fill some of the fields with its subpatterns.
/// In the following example `Fields::wildcards` returns `[_, _, _, _]`. Then in
/// `extract_pattern_arguments` we fill some of the entries, and the result is
/// `[Some(0), _, _, _]`.
/// ```compile_fail,E0004
/// # fn foo() -> [Option<u8>; 4] { [None; 4] }
/// let x: [Option<u8>; 4] = foo();
/// match x {
///     [Some(0), ..] => {}
/// }
/// ```
///
/// Note that the number of fields of a constructor may not match the fields declared in the
/// original struct/variant. This happens if a private or `non_exhaustive` field is uninhabited,
/// because the code mustn't observe that it is uninhabited. In that case that field is not
/// included in `fields`. For that reason, when you have a `FieldIdx` you must use
/// `index_with_declared_idx`.
#[derive(Debug, Clone, Copy)]
pub struct Fields<'p, 'tcx> {
    fields: &'p [DeconstructedPat<'p, 'tcx>],
}

impl<'p, 'tcx> Fields<'p, 'tcx> {
    fn empty() -> Self {
        Fields { fields: &[] }
    }

    fn singleton(cx: &MatchCheckCtxt<'p, 'tcx>, field: DeconstructedPat<'p, 'tcx>) -> Self {
        let field: &_ = cx.pattern_arena.alloc(field);
        Fields { fields: std::slice::from_ref(field) }
    }

    pub fn from_iter(
        cx: &MatchCheckCtxt<'p, 'tcx>,
        fields: impl IntoIterator<Item = DeconstructedPat<'p, 'tcx>>,
    ) -> Self {
        let fields: &[_] = cx.pattern_arena.alloc_from_iter(fields);
        Fields { fields }
    }

    fn wildcards_from_tys(
        cx: &MatchCheckCtxt<'p, 'tcx>,
        tys: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> Self {
        Fields::from_iter(cx, tys.into_iter().map(|ty| DeconstructedPat::wildcard(ty, DUMMY_SP)))
    }

    // In the cases of either a `#[non_exhaustive]` field list or a non-public field, we hide
    // uninhabited fields in order not to reveal the uninhabitedness of the whole variant.
    // This lists the fields we keep along with their types.
    pub(crate) fn list_variant_nonhidden_fields<'a>(
        cx: &'a MatchCheckCtxt<'p, 'tcx>,
        ty: Ty<'tcx>,
        variant: &'a VariantDef,
    ) -> impl Iterator<Item = (FieldIdx, Ty<'tcx>)> + Captures<'a> + Captures<'p> {
        let ty::Adt(adt, args) = ty.kind() else { bug!() };
        // Whether we must not match the fields of this variant exhaustively.
        let is_non_exhaustive = variant.is_field_list_non_exhaustive() && !adt.did().is_local();

        variant.fields.iter().enumerate().filter_map(move |(i, field)| {
            let ty = field.ty(cx.tcx, args);
            // `field.ty()` doesn't normalize after substituting.
            let ty = cx.tcx.normalize_erasing_regions(cx.param_env, ty);
            let is_visible = adt.is_enum() || field.vis.is_accessible_from(cx.module, cx.tcx);
            let is_uninhabited = cx.tcx.features().exhaustive_patterns && cx.is_uninhabited(ty);

            if is_uninhabited && (!is_visible || is_non_exhaustive) {
                None
            } else {
                Some((FieldIdx::new(i), ty))
            }
        })
    }

    /// Creates a new list of wildcard fields for a given constructor. The result must have a
    /// length of `constructor.arity()`.
    #[instrument(level = "trace")]
    pub(super) fn wildcards(pcx: &PatCtxt<'_, 'p, 'tcx>, constructor: &Constructor<'tcx>) -> Self {
        let ret = match constructor {
            Single | Variant(_) => match pcx.ty.kind() {
                ty::Tuple(fs) => Fields::wildcards_from_tys(pcx.cx, fs.iter()),
                ty::Ref(_, rty, _) => Fields::wildcards_from_tys(pcx.cx, once(*rty)),
                ty::Adt(adt, args) => {
                    if adt.is_box() {
                        // The only legal patterns of type `Box` (outside `std`) are `_` and box
                        // patterns. If we're here we can assume this is a box pattern.
                        Fields::wildcards_from_tys(pcx.cx, once(args.type_at(0)))
                    } else {
                        let variant = &adt.variant(constructor.variant_index_for_adt(*adt));
                        let tys = Fields::list_variant_nonhidden_fields(pcx.cx, pcx.ty, variant)
                            .map(|(_, ty)| ty);
                        Fields::wildcards_from_tys(pcx.cx, tys)
                    }
                }
                _ => bug!("Unexpected type for `Single` constructor: {:?}", pcx),
            },
            Slice(slice) => match *pcx.ty.kind() {
                ty::Slice(ty) | ty::Array(ty, _) => {
                    let arity = slice.arity();
                    Fields::wildcards_from_tys(pcx.cx, (0..arity).map(|_| ty))
                }
                _ => bug!("bad slice pattern {:?} {:?}", constructor, pcx),
            },
            Bool(..)
            | IntRange(..)
            | F32Range(..)
            | F64Range(..)
            | Str(..)
            | Opaque(..)
            | NonExhaustive
            | Hidden
            | Missing { .. }
            | Wildcard => Fields::empty(),
            Or => {
                bug!("called `Fields::wildcards` on an `Or` ctor")
            }
        };
        debug!(?ret);
        ret
    }

    /// Returns the list of patterns.
    pub(super) fn iter_patterns<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'p DeconstructedPat<'p, 'tcx>> + Captures<'a> {
        self.fields.iter()
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

/// Values and patterns can be represented as a constructor applied to some fields. This represents
/// a pattern in this form.
/// This also uses interior mutability to keep track of whether the pattern has been found reachable
/// during analysis. For this reason they cannot be cloned.
/// A `DeconstructedPat` will almost always come from user input; the only exception are some
/// `Wildcard`s introduced during specialization.
pub struct DeconstructedPat<'p, 'tcx> {
    ctor: Constructor<'tcx>,
    fields: Fields<'p, 'tcx>,
    ty: Ty<'tcx>,
    span: Span,
    /// Whether removing this arm would change the behavior of the match expression.
    useful: Cell<bool>,
}

impl<'p, 'tcx> DeconstructedPat<'p, 'tcx> {
    pub(super) fn wildcard(ty: Ty<'tcx>, span: Span) -> Self {
        Self::new(Wildcard, Fields::empty(), ty, span)
    }

    pub(super) fn new(
        ctor: Constructor<'tcx>,
        fields: Fields<'p, 'tcx>,
        ty: Ty<'tcx>,
        span: Span,
    ) -> Self {
        DeconstructedPat { ctor, fields, ty, span, useful: Cell::new(false) }
    }

    /// Note: the input patterns must have been lowered through
    /// `rustc_mir_build::thir::pattern::check_match::MatchVisitor::lower_pattern`.
    pub fn from_pat(cx: &MatchCheckCtxt<'p, 'tcx>, pat: &Pat<'tcx>) -> Self {
        let mkpat = |pat| DeconstructedPat::from_pat(cx, pat);
        let ctor;
        let fields;
        match &pat.kind {
            PatKind::AscribeUserType { subpattern, .. }
            | PatKind::InlineConstant { subpattern, .. } => return mkpat(subpattern),
            PatKind::Binding { subpattern: Some(subpat), .. } => return mkpat(subpat),
            PatKind::Binding { subpattern: None, .. } | PatKind::Wild => {
                ctor = Wildcard;
                fields = Fields::empty();
            }
            PatKind::Deref { subpattern } => {
                ctor = Single;
                fields = Fields::singleton(cx, mkpat(subpattern));
            }
            PatKind::Leaf { subpatterns } | PatKind::Variant { subpatterns, .. } => {
                match pat.ty.kind() {
                    ty::Tuple(fs) => {
                        ctor = Single;
                        let mut wilds: SmallVec<[_; 2]> =
                            fs.iter().map(|ty| DeconstructedPat::wildcard(ty, pat.span)).collect();
                        for pat in subpatterns {
                            wilds[pat.field.index()] = mkpat(&pat.pattern);
                        }
                        fields = Fields::from_iter(cx, wilds);
                    }
                    ty::Adt(adt, args) if adt.is_box() => {
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
                        let pattern = subpatterns.into_iter().find(|pat| pat.field.index() == 0);
                        let pat = if let Some(pat) = pattern {
                            mkpat(&pat.pattern)
                        } else {
                            DeconstructedPat::wildcard(args.type_at(0), pat.span)
                        };
                        ctor = Single;
                        fields = Fields::singleton(cx, pat);
                    }
                    ty::Adt(adt, _) => {
                        ctor = match pat.kind {
                            PatKind::Leaf { .. } => Single,
                            PatKind::Variant { variant_index, .. } => Variant(variant_index),
                            _ => bug!(),
                        };
                        let variant = &adt.variant(ctor.variant_index_for_adt(*adt));
                        // For each field in the variant, we store the relevant index into `self.fields` if any.
                        let mut field_id_to_id: Vec<Option<usize>> =
                            (0..variant.fields.len()).map(|_| None).collect();
                        let tys = Fields::list_variant_nonhidden_fields(cx, pat.ty, variant)
                            .enumerate()
                            .map(|(i, (field, ty))| {
                                field_id_to_id[field.index()] = Some(i);
                                ty
                            });
                        let mut wilds: SmallVec<[_; 2]> =
                            tys.map(|ty| DeconstructedPat::wildcard(ty, pat.span)).collect();
                        for pat in subpatterns {
                            if let Some(i) = field_id_to_id[pat.field.index()] {
                                wilds[i] = mkpat(&pat.pattern);
                            }
                        }
                        fields = Fields::from_iter(cx, wilds);
                    }
                    _ => bug!("pattern has unexpected type: pat: {:?}, ty: {:?}", pat, pat.ty),
                }
            }
            PatKind::Constant { value } => {
                match pat.ty.kind() {
                    ty::Bool => {
                        ctor = match value.try_eval_bool(cx.tcx, cx.param_env) {
                            Some(b) => Bool(b),
                            None => Opaque(OpaqueId::new()),
                        };
                        fields = Fields::empty();
                    }
                    ty::Char | ty::Int(_) | ty::Uint(_) => {
                        ctor = match value.try_eval_bits(cx.tcx, cx.param_env) {
                            Some(bits) => IntRange(IntRange::from_bits(cx.tcx, pat.ty, bits)),
                            None => Opaque(OpaqueId::new()),
                        };
                        fields = Fields::empty();
                    }
                    ty::Float(ty::FloatTy::F32) => {
                        ctor = match value.try_eval_bits(cx.tcx, cx.param_env) {
                            Some(bits) => {
                                use rustc_apfloat::Float;
                                let value = rustc_apfloat::ieee::Single::from_bits(bits);
                                F32Range(value, value, RangeEnd::Included)
                            }
                            None => Opaque(OpaqueId::new()),
                        };
                        fields = Fields::empty();
                    }
                    ty::Float(ty::FloatTy::F64) => {
                        ctor = match value.try_eval_bits(cx.tcx, cx.param_env) {
                            Some(bits) => {
                                use rustc_apfloat::Float;
                                let value = rustc_apfloat::ieee::Double::from_bits(bits);
                                F64Range(value, value, RangeEnd::Included)
                            }
                            None => Opaque(OpaqueId::new()),
                        };
                        fields = Fields::empty();
                    }
                    ty::Ref(_, t, _) if t.is_str() => {
                        // We want a `&str` constant to behave like a `Deref` pattern, to be compatible
                        // with other `Deref` patterns. This could have been done in `const_to_pat`,
                        // but that causes issues with the rest of the matching code.
                        // So here, the constructor for a `"foo"` pattern is `&` (represented by
                        // `Single`), and has one field. That field has constructor `Str(value)` and no
                        // fields.
                        // Note: `t` is `str`, not `&str`.
                        let subpattern =
                            DeconstructedPat::new(Str(*value), Fields::empty(), *t, pat.span);
                        ctor = Single;
                        fields = Fields::singleton(cx, subpattern)
                    }
                    // All constants that can be structurally matched have already been expanded
                    // into the corresponding `Pat`s by `const_to_pat`. Constants that remain are
                    // opaque.
                    _ => {
                        ctor = Opaque(OpaqueId::new());
                        fields = Fields::empty();
                    }
                }
            }
            PatKind::Range(patrange) => {
                let PatRange { lo, hi, end, .. } = patrange.as_ref();
                let ty = pat.ty;
                ctor = match ty.kind() {
                    ty::Char | ty::Int(_) | ty::Uint(_) => {
                        let lo =
                            MaybeInfiniteInt::from_pat_range_bdy(*lo, ty, cx.tcx, cx.param_env);
                        let hi =
                            MaybeInfiniteInt::from_pat_range_bdy(*hi, ty, cx.tcx, cx.param_env);
                        IntRange(IntRange::from_range(lo, hi, *end))
                    }
                    ty::Float(fty) => {
                        use rustc_apfloat::Float;
                        let lo = lo.as_finite().map(|c| c.eval_bits(cx.tcx, cx.param_env));
                        let hi = hi.as_finite().map(|c| c.eval_bits(cx.tcx, cx.param_env));
                        match fty {
                            ty::FloatTy::F32 => {
                                use rustc_apfloat::ieee::Single;
                                let lo = lo.map(Single::from_bits).unwrap_or(-Single::INFINITY);
                                let hi = hi.map(Single::from_bits).unwrap_or(Single::INFINITY);
                                F32Range(lo, hi, *end)
                            }
                            ty::FloatTy::F64 => {
                                use rustc_apfloat::ieee::Double;
                                let lo = lo.map(Double::from_bits).unwrap_or(-Double::INFINITY);
                                let hi = hi.map(Double::from_bits).unwrap_or(Double::INFINITY);
                                F64Range(lo, hi, *end)
                            }
                        }
                    }
                    _ => bug!("invalid type for range pattern: {}", ty),
                };
                fields = Fields::empty();
            }
            PatKind::Array { prefix, slice, suffix } | PatKind::Slice { prefix, slice, suffix } => {
                let array_len = match pat.ty.kind() {
                    ty::Array(_, length) => {
                        Some(length.eval_target_usize(cx.tcx, cx.param_env) as usize)
                    }
                    ty::Slice(_) => None,
                    _ => span_bug!(pat.span, "bad ty {:?} for slice pattern", pat.ty),
                };
                let kind = if slice.is_some() {
                    VarLen(prefix.len(), suffix.len())
                } else {
                    FixedLen(prefix.len() + suffix.len())
                };
                ctor = Slice(Slice::new(array_len, kind));
                fields =
                    Fields::from_iter(cx, prefix.iter().chain(suffix.iter()).map(|p| mkpat(&*p)));
            }
            PatKind::Or { .. } => {
                ctor = Or;
                let pats = expand_or_pat(pat);
                fields = Fields::from_iter(cx, pats.into_iter().map(mkpat));
            }
            PatKind::Never => {
                // FIXME(never_patterns): handle `!` in exhaustiveness. This is a sane default
                // in the meantime.
                ctor = Wildcard;
                fields = Fields::empty();
            }
            PatKind::Error(_) => {
                ctor = Opaque(OpaqueId::new());
                fields = Fields::empty();
            }
        }
        DeconstructedPat::new(ctor, fields, pat.ty, pat.span)
    }

    pub(super) fn is_or_pat(&self) -> bool {
        matches!(self.ctor, Or)
    }
    /// Expand this (possibly-nested) or-pattern into its alternatives.
    pub(super) fn flatten_or_pat(&'p self) -> SmallVec<[&'p Self; 1]> {
        if self.is_or_pat() {
            self.iter_fields().flat_map(|p| p.flatten_or_pat()).collect()
        } else {
            smallvec![self]
        }
    }

    pub fn ctor(&self) -> &Constructor<'tcx> {
        &self.ctor
    }
    pub fn ty(&self) -> Ty<'tcx> {
        self.ty
    }
    pub fn span(&self) -> Span {
        self.span
    }

    pub fn iter_fields<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'p DeconstructedPat<'p, 'tcx>> + Captures<'a> {
        self.fields.iter_patterns()
    }

    /// Specialize this pattern with a constructor.
    /// `other_ctor` can be different from `self.ctor`, but must be covered by it.
    pub(super) fn specialize<'a>(
        &'a self,
        pcx: &PatCtxt<'_, 'p, 'tcx>,
        other_ctor: &Constructor<'tcx>,
    ) -> SmallVec<[&'p DeconstructedPat<'p, 'tcx>; 2]> {
        match (&self.ctor, other_ctor) {
            (Wildcard, _) => {
                // We return a wildcard for each field of `other_ctor`.
                Fields::wildcards(pcx, other_ctor).iter_patterns().collect()
            }
            (Slice(self_slice), Slice(other_slice))
                if self_slice.arity() != other_slice.arity() =>
            {
                // The only tricky case: two slices of different arity. Since `self_slice` covers
                // `other_slice`, `self_slice` must be `VarLen`, i.e. of the form
                // `[prefix, .., suffix]`. Moreover `other_slice` is guaranteed to have a larger
                // arity. So we fill the middle part with enough wildcards to reach the length of
                // the new, larger slice.
                match self_slice.kind {
                    FixedLen(_) => bug!("{:?} doesn't cover {:?}", self_slice, other_slice),
                    VarLen(prefix, suffix) => {
                        let (ty::Slice(inner_ty) | ty::Array(inner_ty, _)) = *self.ty.kind() else {
                            bug!("bad slice pattern {:?} {:?}", self.ctor, self.ty);
                        };
                        let prefix = &self.fields.fields[..prefix];
                        let suffix = &self.fields.fields[self_slice.arity() - suffix..];
                        let wildcard: &_ = pcx
                            .cx
                            .pattern_arena
                            .alloc(DeconstructedPat::wildcard(inner_ty, DUMMY_SP));
                        let extra_wildcards = other_slice.arity() - self_slice.arity();
                        let extra_wildcards = (0..extra_wildcards).map(|_| wildcard);
                        prefix.iter().chain(extra_wildcards).chain(suffix).collect()
                    }
                }
            }
            _ => self.fields.iter_patterns().collect(),
        }
    }

    /// We keep track for each pattern if it was ever useful during the analysis. This is used
    /// with `redundant_spans` to report redundant subpatterns arising from or patterns.
    pub(super) fn set_useful(&self) {
        self.useful.set(true)
    }
    pub(super) fn is_useful(&self) -> bool {
        if self.useful.get() {
            true
        } else if self.is_or_pat() && self.iter_fields().any(|f| f.is_useful()) {
            // We always expand or patterns in the matrix, so we will never see the actual
            // or-pattern (the one with constructor `Or`) in the column. As such, it will not be
            // marked as useful itself, only its children will. We recover this information here.
            self.set_useful();
            true
        } else {
            false
        }
    }

    /// Report the spans of subpatterns that were not useful, if any.
    pub(super) fn redundant_spans(&self) -> Vec<Span> {
        let mut spans = Vec::new();
        self.collect_redundant_spans(&mut spans);
        spans
    }
    fn collect_redundant_spans(&self, spans: &mut Vec<Span>) {
        // We don't look at subpatterns if we already reported the whole pattern as redundant.
        if !self.is_useful() {
            spans.push(self.span);
        } else {
            for p in self.iter_fields() {
                p.collect_redundant_spans(spans);
            }
        }
    }
}

/// This is mostly copied from the `Pat` impl. This is best effort and not good enough for a
/// `Display` impl.
impl<'p, 'tcx> fmt::Debug for DeconstructedPat<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Printing lists is a chore.
        let mut first = true;
        let mut start_or_continue = |s| {
            if first {
                first = false;
                ""
            } else {
                s
            }
        };
        let mut start_or_comma = || start_or_continue(", ");

        match &self.ctor {
            Single | Variant(_) => match self.ty.kind() {
                ty::Adt(def, _) if def.is_box() => {
                    // Without `box_patterns`, the only legal pattern of type `Box` is `_` (outside
                    // of `std`). So this branch is only reachable when the feature is enabled and
                    // the pattern is a box pattern.
                    let subpattern = self.iter_fields().next().unwrap();
                    write!(f, "box {subpattern:?}")
                }
                ty::Adt(..) | ty::Tuple(..) => {
                    let variant = match self.ty.kind() {
                        ty::Adt(adt, _) => Some(adt.variant(self.ctor.variant_index_for_adt(*adt))),
                        ty::Tuple(_) => None,
                        _ => unreachable!(),
                    };

                    if let Some(variant) = variant {
                        write!(f, "{}", variant.name)?;
                    }

                    // Without `cx`, we can't know which field corresponds to which, so we can't
                    // get the names of the fields. Instead we just display everything as a tuple
                    // struct, which should be good enough.
                    write!(f, "(")?;
                    for p in self.iter_fields() {
                        write!(f, "{}", start_or_comma())?;
                        write!(f, "{p:?}")?;
                    }
                    write!(f, ")")
                }
                // Note: given the expansion of `&str` patterns done in `expand_pattern`, we should
                // be careful to detect strings here. However a string literal pattern will never
                // be reported as a non-exhaustiveness witness, so we can ignore this issue.
                ty::Ref(_, _, mutbl) => {
                    let subpattern = self.iter_fields().next().unwrap();
                    write!(f, "&{}{:?}", mutbl.prefix_str(), subpattern)
                }
                _ => write!(f, "_"),
            },
            Slice(slice) => {
                let mut subpatterns = self.fields.iter_patterns();
                write!(f, "[")?;
                match slice.kind {
                    FixedLen(_) => {
                        for p in subpatterns {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                    }
                    VarLen(prefix_len, _) => {
                        for p in subpatterns.by_ref().take(prefix_len) {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                        write!(f, "{}", start_or_comma())?;
                        write!(f, "..")?;
                        for p in subpatterns {
                            write!(f, "{}{:?}", start_or_comma(), p)?;
                        }
                    }
                }
                write!(f, "]")
            }
            Bool(b) => write!(f, "{b}"),
            // Best-effort, will render signed ranges incorrectly
            IntRange(range) => write!(f, "{range:?}"),
            F32Range(lo, hi, end) => write!(f, "{lo}{end}{hi}"),
            F64Range(lo, hi, end) => write!(f, "{lo}{end}{hi}"),
            Str(value) => write!(f, "{value}"),
            Opaque(..) => write!(f, "<constant pattern>"),
            Or => {
                for pat in self.iter_fields() {
                    write!(f, "{}{:?}", start_or_continue(" | "), pat)?;
                }
                Ok(())
            }
            Wildcard | Missing { .. } | NonExhaustive | Hidden => write!(f, "_ : {:?}", self.ty),
        }
    }
}

/// Same idea as `DeconstructedPat`, except this is a fictitious pattern built up for diagnostics
/// purposes. As such they don't use interning and can be cloned.
#[derive(Debug, Clone)]
pub struct WitnessPat<'tcx> {
    ctor: Constructor<'tcx>,
    pub(crate) fields: Vec<WitnessPat<'tcx>>,
    ty: Ty<'tcx>,
}

impl<'tcx> WitnessPat<'tcx> {
    pub(super) fn new(ctor: Constructor<'tcx>, fields: Vec<Self>, ty: Ty<'tcx>) -> Self {
        Self { ctor, fields, ty }
    }
    pub(super) fn wildcard(ty: Ty<'tcx>) -> Self {
        Self::new(Wildcard, Vec::new(), ty)
    }

    /// Construct a pattern that matches everything that starts with this constructor.
    /// For example, if `ctor` is a `Constructor::Variant` for `Option::Some`, we get the pattern
    /// `Some(_)`.
    pub(super) fn wild_from_ctor(pcx: &PatCtxt<'_, '_, 'tcx>, ctor: Constructor<'tcx>) -> Self {
        // Reuse `Fields::wildcards` to get the types.
        let fields = Fields::wildcards(pcx, &ctor)
            .iter_patterns()
            .map(|deco_pat| Self::wildcard(deco_pat.ty()))
            .collect();
        Self::new(ctor, fields, pcx.ty)
    }

    pub fn ctor(&self) -> &Constructor<'tcx> {
        &self.ctor
    }
    pub fn ty(&self) -> Ty<'tcx> {
        self.ty
    }

    /// Convert back to a `thir::Pat` for diagnostic purposes. This panics for patterns that don't
    /// appear in diagnostics, like float ranges.
    pub fn to_diagnostic_pat(&self, cx: &MatchCheckCtxt<'_, 'tcx>) -> Pat<'tcx> {
        let is_wildcard = |pat: &Pat<'_>| matches!(pat.kind, PatKind::Wild);
        let mut subpatterns = self.iter_fields().map(|p| Box::new(p.to_diagnostic_pat(cx)));
        let kind = match &self.ctor {
            Bool(b) => PatKind::Constant { value: mir::Const::from_bool(cx.tcx, *b) },
            IntRange(range) => return range.to_diagnostic_pat(self.ty, cx.tcx),
            Single | Variant(_) => match self.ty.kind() {
                ty::Tuple(..) => PatKind::Leaf {
                    subpatterns: subpatterns
                        .enumerate()
                        .map(|(i, pattern)| FieldPat { field: FieldIdx::new(i), pattern })
                        .collect(),
                },
                ty::Adt(adt_def, _) if adt_def.is_box() => {
                    // Without `box_patterns`, the only legal pattern of type `Box` is `_` (outside
                    // of `std`). So this branch is only reachable when the feature is enabled and
                    // the pattern is a box pattern.
                    PatKind::Deref { subpattern: subpatterns.next().unwrap() }
                }
                ty::Adt(adt_def, args) => {
                    let variant_index = self.ctor.variant_index_for_adt(*adt_def);
                    let variant = &adt_def.variant(variant_index);
                    let subpatterns = Fields::list_variant_nonhidden_fields(cx, self.ty, variant)
                        .zip(subpatterns)
                        .map(|((field, _ty), pattern)| FieldPat { field, pattern })
                        .collect();

                    if adt_def.is_enum() {
                        PatKind::Variant { adt_def: *adt_def, args, variant_index, subpatterns }
                    } else {
                        PatKind::Leaf { subpatterns }
                    }
                }
                // Note: given the expansion of `&str` patterns done in `expand_pattern`, we should
                // be careful to reconstruct the correct constant pattern here. However a string
                // literal pattern will never be reported as a non-exhaustiveness witness, so we
                // ignore this issue.
                ty::Ref(..) => PatKind::Deref { subpattern: subpatterns.next().unwrap() },
                _ => bug!("unexpected ctor for type {:?} {:?}", self.ctor, self.ty),
            },
            Slice(slice) => {
                match slice.kind {
                    FixedLen(_) => PatKind::Slice {
                        prefix: subpatterns.collect(),
                        slice: None,
                        suffix: Box::new([]),
                    },
                    VarLen(prefix, _) => {
                        let mut subpatterns = subpatterns.peekable();
                        let mut prefix: Vec<_> = subpatterns.by_ref().take(prefix).collect();
                        if slice.array_len.is_some() {
                            // Improves diagnostics a bit: if the type is a known-size array, instead
                            // of reporting `[x, _, .., _, y]`, we prefer to report `[x, .., y]`.
                            // This is incorrect if the size is not known, since `[_, ..]` captures
                            // arrays of lengths `>= 1` whereas `[..]` captures any length.
                            while !prefix.is_empty() && is_wildcard(prefix.last().unwrap()) {
                                prefix.pop();
                            }
                            while subpatterns.peek().is_some()
                                && is_wildcard(subpatterns.peek().unwrap())
                            {
                                subpatterns.next();
                            }
                        }
                        let suffix: Box<[_]> = subpatterns.collect();
                        let wild = Pat::wildcard_from_ty(self.ty);
                        PatKind::Slice {
                            prefix: prefix.into_boxed_slice(),
                            slice: Some(Box::new(wild)),
                            suffix,
                        }
                    }
                }
            }
            &Str(value) => PatKind::Constant { value },
            Wildcard | NonExhaustive | Hidden => PatKind::Wild,
            Missing { .. } => bug!(
                "trying to convert a `Missing` constructor into a `Pat`; this is probably a bug,
                `Missing` should have been processed in `apply_constructors`"
            ),
            F32Range(..) | F64Range(..) | Opaque(..) | Or => {
                bug!("can't convert to pattern: {:?}", self)
            }
        };

        Pat { ty: self.ty, span: DUMMY_SP, kind }
    }

    pub fn iter_fields<'a>(&'a self) -> impl Iterator<Item = &'a WitnessPat<'tcx>> {
        self.fields.iter()
    }
}
