use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::{is_lint_allowed, path_to_local, search_same, SpanlessEq, SpanlessHash};
use core::cmp::Ordering;
use core::iter;
use core::slice;
use rustc_arena::DroplessArena;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{Arm, Expr, ExprKind, HirId, HirIdMap, HirIdMapEntry, HirIdSet, Pat, PatKind, RangeEnd};
use rustc_lint::builtin::NON_EXHAUSTIVE_OMITTED_PATTERNS;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::Symbol;

use super::MATCH_SAME_ARMS;

#[expect(clippy::too_many_lines)]
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, arms: &'tcx [Arm<'_>]) {
    let hash = |&(_, arm): &(usize, &Arm<'_>)| -> u64 {
        let mut h = SpanlessHash::new(cx);
        h.hash_expr(arm.body);
        h.finish()
    };

    let arena = DroplessArena::default();
    let normalized_pats: Vec<_> = arms
        .iter()
        .map(|a| NormalizedPat::from_pat(cx, &arena, a.pat))
        .collect();

    // The furthest forwards a pattern can move without semantic changes
    let forwards_blocking_idxs: Vec<_> = normalized_pats
        .iter()
        .enumerate()
        .map(|(i, pat)| {
            normalized_pats[i + 1..]
                .iter()
                .enumerate()
                .find_map(|(j, other)| pat.has_overlapping_values(other).then_some(i + 1 + j))
                .unwrap_or(normalized_pats.len())
        })
        .collect();

    // The furthest backwards a pattern can move without semantic changes
    let backwards_blocking_idxs: Vec<_> = normalized_pats
        .iter()
        .enumerate()
        .map(|(i, pat)| {
            normalized_pats[..i]
                .iter()
                .enumerate()
                .rev()
                .zip(forwards_blocking_idxs[..i].iter().copied().rev())
                .skip_while(|&(_, forward_block)| forward_block > i)
                .find_map(|((j, other), forward_block)| {
                    (forward_block == i || pat.has_overlapping_values(other)).then_some(j)
                })
                .unwrap_or(0)
        })
        .collect();

    let eq = |&(lindex, lhs): &(usize, &Arm<'_>), &(rindex, rhs): &(usize, &Arm<'_>)| -> bool {
        let min_index = usize::min(lindex, rindex);
        let max_index = usize::max(lindex, rindex);

        let mut local_map: HirIdMap<HirId> = HirIdMap::default();
        let eq_fallback = |a: &Expr<'_>, b: &Expr<'_>| {
            if_chain! {
                if let Some(a_id) = path_to_local(a);
                if let Some(b_id) = path_to_local(b);
                let entry = match local_map.entry(a_id) {
                    HirIdMapEntry::Vacant(entry) => entry,
                    // check if using the same bindings as before
                    HirIdMapEntry::Occupied(entry) => return *entry.get() == b_id,
                };
                // the names technically don't have to match; this makes the lint more conservative
                if cx.tcx.hir().name(a_id) == cx.tcx.hir().name(b_id);
                if cx.typeck_results().expr_ty(a) == cx.typeck_results().expr_ty(b);
                if pat_contains_local(lhs.pat, a_id);
                if pat_contains_local(rhs.pat, b_id);
                then {
                    entry.insert(b_id);
                    true
                } else {
                    false
                }
            }
        };
        // Arms with a guard are ignored, those can’t always be merged together
        // If both arms overlap with an arm in between then these can't be merged either.
        !(backwards_blocking_idxs[max_index] > min_index && forwards_blocking_idxs[min_index] < max_index)
                && lhs.guard.is_none()
                && rhs.guard.is_none()
                && SpanlessEq::new(cx)
                    .expr_fallback(eq_fallback)
                    .eq_expr(lhs.body, rhs.body)
                // these checks could be removed to allow unused bindings
                && bindings_eq(lhs.pat, local_map.keys().copied().collect())
                && bindings_eq(rhs.pat, local_map.values().copied().collect())
    };

    let indexed_arms: Vec<(usize, &Arm<'_>)> = arms.iter().enumerate().collect();
    for (&(i, arm1), &(j, arm2)) in search_same(&indexed_arms, hash, eq) {
        if matches!(arm2.pat.kind, PatKind::Wild) {
            if !cx.tcx.features().non_exhaustive_omitted_patterns_lint
                || is_lint_allowed(cx, NON_EXHAUSTIVE_OMITTED_PATTERNS, arm2.hir_id)
            {
                span_lint_and_then(
                    cx,
                    MATCH_SAME_ARMS,
                    arm1.span,
                    "this match arm has an identical body to the `_` wildcard arm",
                    |diag| {
                        diag.span_suggestion(arm1.span, "try removing the arm", "", Applicability::MaybeIncorrect)
                            .help("or try changing either arm body")
                            .span_note(arm2.span, "`_` wildcard arm here");
                    },
                );
            }
        } else {
            let back_block = backwards_blocking_idxs[j];
            let (keep_arm, move_arm) = if back_block < i || (back_block == 0 && forwards_blocking_idxs[i] <= j) {
                (arm1, arm2)
            } else {
                (arm2, arm1)
            };

            span_lint_and_then(
                cx,
                MATCH_SAME_ARMS,
                keep_arm.span,
                "this match arm has an identical body to another arm",
                |diag| {
                    let move_pat_snip = snippet(cx, move_arm.pat.span, "<pat2>");
                    let keep_pat_snip = snippet(cx, keep_arm.pat.span, "<pat1>");

                    diag.span_suggestion(
                        keep_arm.pat.span,
                        "try merging the arm patterns",
                        format!("{keep_pat_snip} | {move_pat_snip}"),
                        Applicability::MaybeIncorrect,
                    )
                    .help("or try changing either arm body")
                    .span_note(move_arm.span, "other arm here");
                },
            );
        }
    }
}

#[derive(Clone, Copy)]
enum NormalizedPat<'a> {
    Wild,
    Struct(Option<DefId>, &'a [(Symbol, Self)]),
    Tuple(Option<DefId>, &'a [Self]),
    Or(&'a [Self]),
    Path(Option<DefId>),
    LitStr(Symbol),
    LitBytes(&'a [u8]),
    LitInt(u128),
    LitBool(bool),
    Range(PatRange),
    /// A slice pattern. If the second value is `None`, then this matches an exact size. Otherwise
    /// the first value contains everything before the `..` wildcard pattern, and the second value
    /// contains everything afterwards. Note that either side, or both sides, may contain zero
    /// patterns.
    Slice(&'a [Self], Option<&'a [Self]>),
}

#[derive(Clone, Copy)]
struct PatRange {
    start: u128,
    end: u128,
    bounds: RangeEnd,
}
impl PatRange {
    fn contains(&self, x: u128) -> bool {
        x >= self.start
            && match self.bounds {
                RangeEnd::Included => x <= self.end,
                RangeEnd::Excluded => x < self.end,
            }
    }

    fn overlaps(&self, other: &Self) -> bool {
        // Note: Empty ranges are impossible, so this is correct even though it would return true if an
        // empty exclusive range were to reside within an inclusive range.
        (match self.bounds {
            RangeEnd::Included => self.end >= other.start,
            RangeEnd::Excluded => self.end > other.start,
        } && match other.bounds {
            RangeEnd::Included => self.start <= other.end,
            RangeEnd::Excluded => self.start < other.end,
        })
    }
}

/// Iterates over the pairs of fields with matching names.
fn iter_matching_struct_fields<'a>(
    left: &'a [(Symbol, NormalizedPat<'a>)],
    right: &'a [(Symbol, NormalizedPat<'a>)],
) -> impl Iterator<Item = (&'a NormalizedPat<'a>, &'a NormalizedPat<'a>)> + 'a {
    struct Iter<'a>(
        slice::Iter<'a, (Symbol, NormalizedPat<'a>)>,
        slice::Iter<'a, (Symbol, NormalizedPat<'a>)>,
    );
    impl<'a> Iterator for Iter<'a> {
        type Item = (&'a NormalizedPat<'a>, &'a NormalizedPat<'a>);
        fn next(&mut self) -> Option<Self::Item> {
            // Note: all the fields in each slice are sorted by symbol value.
            let mut left = self.0.next()?;
            let mut right = self.1.next()?;
            loop {
                match left.0.cmp(&right.0) {
                    Ordering::Equal => return Some((&left.1, &right.1)),
                    Ordering::Less => left = self.0.next()?,
                    Ordering::Greater => right = self.1.next()?,
                }
            }
        }
    }
    Iter(left.iter(), right.iter())
}

#[expect(clippy::similar_names)]
impl<'a> NormalizedPat<'a> {
    fn from_pat(cx: &LateContext<'_>, arena: &'a DroplessArena, pat: &'a Pat<'_>) -> Self {
        match pat.kind {
            PatKind::Wild | PatKind::Binding(.., None) => Self::Wild,
            PatKind::Binding(.., Some(pat)) | PatKind::Box(pat) | PatKind::Ref(pat, _) => {
                Self::from_pat(cx, arena, pat)
            },
            PatKind::Struct(ref path, fields, _) => {
                let fields =
                    arena.alloc_from_iter(fields.iter().map(|f| (f.ident.name, Self::from_pat(cx, arena, f.pat))));
                fields.sort_by_key(|&(name, _)| name);
                Self::Struct(cx.qpath_res(path, pat.hir_id).opt_def_id(), fields)
            },
            PatKind::TupleStruct(ref path, pats, wild_idx) => {
                let Some(adt) = cx.typeck_results().pat_ty(pat).ty_adt_def() else {
                    return Self::Wild
                };
                let (var_id, variant) = if adt.is_enum() {
                    match cx.qpath_res(path, pat.hir_id).opt_def_id() {
                        Some(x) => (Some(x), adt.variant_with_ctor_id(x)),
                        None => return Self::Wild,
                    }
                } else {
                    (None, adt.non_enum_variant())
                };
                let (front, back) = match wild_idx.as_opt_usize() {
                    Some(i) => pats.split_at(i),
                    None => (pats, [].as_slice()),
                };
                let pats = arena.alloc_from_iter(
                    front
                        .iter()
                        .map(|pat| Self::from_pat(cx, arena, pat))
                        .chain(iter::repeat_with(|| Self::Wild).take(variant.fields.len() - pats.len()))
                        .chain(back.iter().map(|pat| Self::from_pat(cx, arena, pat))),
                );
                Self::Tuple(var_id, pats)
            },
            PatKind::Or(pats) => Self::Or(arena.alloc_from_iter(pats.iter().map(|pat| Self::from_pat(cx, arena, pat)))),
            PatKind::Path(ref path) => Self::Path(cx.qpath_res(path, pat.hir_id).opt_def_id()),
            PatKind::Tuple(pats, wild_idx) => {
                let field_count = match cx.typeck_results().pat_ty(pat).kind() {
                    ty::Tuple(subs) => subs.len(),
                    _ => return Self::Wild,
                };
                let (front, back) = match wild_idx.as_opt_usize() {
                    Some(i) => pats.split_at(i),
                    None => (pats, [].as_slice()),
                };
                let pats = arena.alloc_from_iter(
                    front
                        .iter()
                        .map(|pat| Self::from_pat(cx, arena, pat))
                        .chain(iter::repeat_with(|| Self::Wild).take(field_count - pats.len()))
                        .chain(back.iter().map(|pat| Self::from_pat(cx, arena, pat))),
                );
                Self::Tuple(None, pats)
            },
            PatKind::Lit(e) => match &e.kind {
                // TODO: Handle negative integers. They're currently treated as a wild match.
                ExprKind::Lit(lit) => match lit.node {
                    LitKind::Str(sym, _) => Self::LitStr(sym),
                    LitKind::ByteStr(ref bytes, _) | LitKind::CStr(ref bytes, _) => Self::LitBytes(bytes),
                    LitKind::Byte(val) => Self::LitInt(val.into()),
                    LitKind::Char(val) => Self::LitInt(val.into()),
                    LitKind::Int(val, _) => Self::LitInt(val),
                    LitKind::Bool(val) => Self::LitBool(val),
                    LitKind::Float(..) | LitKind::Err => Self::Wild,
                },
                _ => Self::Wild,
            },
            PatKind::Range(start, end, bounds) => {
                // TODO: Handle negative integers. They're currently treated as a wild match.
                let start = match start {
                    None => 0,
                    Some(e) => match &e.kind {
                        ExprKind::Lit(lit) => match lit.node {
                            LitKind::Int(val, _) => val,
                            LitKind::Char(val) => val.into(),
                            LitKind::Byte(val) => val.into(),
                            _ => return Self::Wild,
                        },
                        _ => return Self::Wild,
                    },
                };
                let (end, bounds) = match end {
                    None => (u128::MAX, RangeEnd::Included),
                    Some(e) => match &e.kind {
                        ExprKind::Lit(lit) => match lit.node {
                            LitKind::Int(val, _) => (val, bounds),
                            LitKind::Char(val) => (val.into(), bounds),
                            LitKind::Byte(val) => (val.into(), bounds),
                            _ => return Self::Wild,
                        },
                        _ => return Self::Wild,
                    },
                };
                Self::Range(PatRange { start, end, bounds })
            },
            PatKind::Slice(front, wild_pat, back) => Self::Slice(
                arena.alloc_from_iter(front.iter().map(|pat| Self::from_pat(cx, arena, pat))),
                wild_pat.map(|_| &*arena.alloc_from_iter(back.iter().map(|pat| Self::from_pat(cx, arena, pat)))),
            ),
        }
    }

    /// Checks if two patterns overlap in the values they can match assuming they are for the same
    /// type.
    fn has_overlapping_values(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Wild, _) | (_, Self::Wild) => true,
            (Self::Or(pats), ref other) | (ref other, Self::Or(pats)) => {
                pats.iter().any(|pat| pat.has_overlapping_values(other))
            },
            (Self::Struct(lpath, lfields), Self::Struct(rpath, rfields)) => {
                if lpath != rpath {
                    return false;
                }
                iter_matching_struct_fields(lfields, rfields).all(|(lpat, rpat)| lpat.has_overlapping_values(rpat))
            },
            (Self::Tuple(lpath, lpats), Self::Tuple(rpath, rpats)) => {
                if lpath != rpath {
                    return false;
                }
                lpats
                    .iter()
                    .zip(rpats.iter())
                    .all(|(lpat, rpat)| lpat.has_overlapping_values(rpat))
            },
            (Self::Path(x), Self::Path(y)) => x == y,
            (Self::LitStr(x), Self::LitStr(y)) => x == y,
            (Self::LitBytes(x), Self::LitBytes(y)) => x == y,
            (Self::LitInt(x), Self::LitInt(y)) => x == y,
            (Self::LitBool(x), Self::LitBool(y)) => x == y,
            (Self::Range(ref x), Self::Range(ref y)) => x.overlaps(y),
            (Self::Range(ref range), Self::LitInt(x)) | (Self::LitInt(x), Self::Range(ref range)) => range.contains(x),
            (Self::Slice(lpats, None), Self::Slice(rpats, None)) => {
                lpats.len() == rpats.len() && lpats.iter().zip(rpats.iter()).all(|(x, y)| x.has_overlapping_values(y))
            },
            (Self::Slice(pats, None), Self::Slice(front, Some(back)))
            | (Self::Slice(front, Some(back)), Self::Slice(pats, None)) => {
                // Here `pats` is an exact size match. If the combined lengths of `front` and `back` are greater
                // then the minimum length required will be greater than the length of `pats`.
                if pats.len() < front.len() + back.len() {
                    return false;
                }
                pats[..front.len()]
                    .iter()
                    .zip(front.iter())
                    .chain(pats[pats.len() - back.len()..].iter().zip(back.iter()))
                    .all(|(x, y)| x.has_overlapping_values(y))
            },
            (Self::Slice(lfront, Some(lback)), Self::Slice(rfront, Some(rback))) => lfront
                .iter()
                .zip(rfront.iter())
                .chain(lback.iter().rev().zip(rback.iter().rev()))
                .all(|(x, y)| x.has_overlapping_values(y)),

            // Enums can mix unit variants with tuple/struct variants. These can never overlap.
            (Self::Path(_), Self::Tuple(..) | Self::Struct(..))
            | (Self::Tuple(..) | Self::Struct(..), Self::Path(_)) => false,

            // Tuples can be matched like a struct.
            (Self::Tuple(x, _), Self::Struct(y, _)) | (Self::Struct(x, _), Self::Tuple(y, _)) => {
                // TODO: check fields here.
                x == y
            },

            // TODO: Lit* with Path, Range with Path, LitBytes with Slice
            _ => true,
        }
    }
}

fn pat_contains_local(pat: &Pat<'_>, id: HirId) -> bool {
    let mut result = false;
    pat.walk_short(|p| {
        result |= matches!(p.kind, PatKind::Binding(_, binding_id, ..) if binding_id == id);
        !result
    });
    result
}

/// Returns true if all the bindings in the `Pat` are in `ids` and vice versa
fn bindings_eq(pat: &Pat<'_>, mut ids: HirIdSet) -> bool {
    let mut result = true;
    pat.each_binding_or_first(&mut |_, id, _, _| result &= ids.remove(&id));
    result && ids.is_empty()
}
