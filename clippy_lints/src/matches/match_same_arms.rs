use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::{path_to_local, search_same, SpanlessEq, SpanlessHash};
use core::iter;
use rustc_arena::DroplessArena;
use rustc_ast::ast::LitKind;
use rustc_hir::def_id::DefId;
use rustc_hir::{Arm, Expr, ExprKind, HirId, HirIdMap, HirIdSet, Pat, PatKind, RangeEnd};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::Symbol;
use std::collections::hash_map::Entry;

use super::MATCH_SAME_ARMS;

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

    // The furthast forwards a pattern can move without semantic changes
    let forwards_blocking_idxs: Vec<_> = normalized_pats
        .iter()
        .enumerate()
        .map(|(i, pat)| {
            normalized_pats[i + 1..]
                .iter()
                .enumerate()
                .find_map(|(j, other)| pat.can_also_match(other).then(|| i + 1 + j))
                .unwrap_or(normalized_pats.len())
        })
        .collect();

    // The furthast backwards a pattern can move without semantic changes
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
                .find_map(|((j, other), forward_block)| (forward_block == i || pat.can_also_match(other)).then(|| j))
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
                    Entry::Vacant(entry) => entry,
                    // check if using the same bindings as before
                    Entry::Occupied(entry) => return *entry.get() == b_id,
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
    for (&(_, i), &(_, j)) in search_same(&indexed_arms, hash, eq) {
        span_lint_and_then(
            cx,
            MATCH_SAME_ARMS,
            j.body.span,
            "this `match` has identical arm bodies",
            |diag| {
                diag.span_note(i.body.span, "same as this");

                // Note: this does not use `span_suggestion` on purpose:
                // there is no clean way
                // to remove the other arm. Building a span and suggest to replace it to ""
                // makes an even more confusing error message. Also in order not to make up a
                // span for the whole pattern, the suggestion is only shown when there is only
                // one pattern. The user should know about `|` if they are already using it…

                let lhs = snippet(cx, i.pat.span, "<pat1>");
                let rhs = snippet(cx, j.pat.span, "<pat2>");

                if let PatKind::Wild = j.pat.kind {
                    // if the last arm is _, then i could be integrated into _
                    // note that i.pat cannot be _, because that would mean that we're
                    // hiding all the subsequent arms, and rust won't compile
                    diag.span_note(
                        i.body.span,
                        &format!(
                            "`{}` has the same arm body as the `_` wildcard, consider removing it",
                            lhs
                        ),
                    );
                } else {
                    diag.span_help(i.pat.span, &format!("consider refactoring into `{} | {}`", lhs, rhs,))
                        .help("...or consider changing the match arm bodies");
                }
            },
        );
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
        !(self.is_empty() || other.is_empty())
            && match self.bounds {
                RangeEnd::Included => self.end >= other.start,
                RangeEnd::Excluded => self.end > other.start,
            }
            && match other.bounds {
                RangeEnd::Included => self.start <= other.end,
                RangeEnd::Excluded => self.start < other.end,
            }
    }

    fn is_empty(&self) -> bool {
        match self.bounds {
            RangeEnd::Included => false,
            RangeEnd::Excluded => self.start == self.end,
        }
    }
}

#[allow(clippy::similar_names)]
impl<'a> NormalizedPat<'a> {
    #[allow(clippy::too_many_lines)]
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
                let adt = match cx.typeck_results().pat_ty(pat).ty_adt_def() {
                    Some(x) => x,
                    None => return Self::Wild,
                };
                let (var_id, variant) = if adt.is_enum() {
                    match cx.qpath_res(path, pat.hir_id).opt_def_id() {
                        Some(x) => (Some(x), adt.variant_with_ctor_id(x)),
                        None => return Self::Wild,
                    }
                } else {
                    (None, adt.non_enum_variant())
                };
                let (front, back) = match wild_idx {
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
                let (front, back) = match wild_idx {
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
                ExprKind::Lit(lit) => match lit.node {
                    LitKind::Str(sym, _) => Self::LitStr(sym),
                    LitKind::ByteStr(ref bytes) => Self::LitBytes(&**bytes),
                    LitKind::Byte(val) => Self::LitInt(val.into()),
                    LitKind::Char(val) => Self::LitInt(val.into()),
                    LitKind::Int(val, _) => Self::LitInt(val),
                    LitKind::Bool(val) => Self::LitBool(val),
                    LitKind::Float(..) | LitKind::Err(_) => Self::Wild,
                },
                _ => Self::Wild,
            },
            PatKind::Range(start, end, bounds) => {
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
    fn can_also_match(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Wild, _) | (_, Self::Wild) => true,
            (Self::Or(pats), ref other) | (ref other, Self::Or(pats)) => {
                pats.iter().any(|pat| pat.can_also_match(other))
            },
            (Self::Struct(lpath, lfields), Self::Struct(rpath, rfields)) => {
                if lpath != rpath {
                    return false;
                }
                let mut rfields = rfields.iter();
                let mut rfield = match rfields.next() {
                    Some(x) => x,
                    None => return true,
                };
                'outer: for lfield in lfields {
                    loop {
                        if lfield.0 < rfield.0 {
                            continue 'outer;
                        } else if lfield.0 > rfield.0 {
                            rfield = match rfields.next() {
                                Some(x) => x,
                                None => return true,
                            };
                        } else if !lfield.1.can_also_match(&rfield.1) {
                            return false;
                        } else {
                            rfield = match rfields.next() {
                                Some(x) => x,
                                None => return true,
                            };
                            continue 'outer;
                        }
                    }
                }
                true
            },
            (Self::Tuple(lpath, lpats), Self::Tuple(rpath, rpats)) => {
                if lpath != rpath {
                    return false;
                }
                lpats
                    .iter()
                    .zip(rpats.iter())
                    .all(|(lpat, rpat)| lpat.can_also_match(rpat))
            },
            (Self::Path(x), Self::Path(y)) => x == y,
            (Self::LitStr(x), Self::LitStr(y)) => x == y,
            (Self::LitBytes(x), Self::LitBytes(y)) => x == y,
            (Self::LitInt(x), Self::LitInt(y)) => x == y,
            (Self::LitBool(x), Self::LitBool(y)) => x == y,
            (Self::Range(ref x), Self::Range(ref y)) => x.overlaps(y),
            (Self::Range(ref range), Self::LitInt(x)) | (Self::LitInt(x), Self::Range(ref range)) => range.contains(x),
            (Self::Slice(lpats, None), Self::Slice(rpats, None)) => {
                lpats.len() == rpats.len() && lpats.iter().zip(rpats.iter()).all(|(x, y)| x.can_also_match(y))
            },
            (Self::Slice(pats, None), Self::Slice(front, Some(back)))
            | (Self::Slice(front, Some(back)), Self::Slice(pats, None)) => {
                if pats.len() < front.len() + back.len() {
                    return false;
                }
                pats[..front.len()]
                    .iter()
                    .zip(front.iter())
                    .chain(pats[pats.len() - back.len()..].iter().zip(back.iter()))
                    .all(|(x, y)| x.can_also_match(y))
            },
            (Self::Slice(lfront, Some(lback)), Self::Slice(rfront, Some(rback))) => lfront
                .iter()
                .zip(rfront.iter())
                .chain(lback.iter().rev().zip(rback.iter().rev()))
                .all(|(x, y)| x.can_also_match(y)),

            // Todo: Lit* with Path, Range with Path, LitBytes with Slice, Slice with Slice
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
