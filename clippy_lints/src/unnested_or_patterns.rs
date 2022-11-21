#![allow(clippy::wildcard_imports, clippy::enum_glob_use)]

use clippy_utils::ast_utils::{eq_field_pat, eq_id, eq_maybe_qself, eq_pat, eq_path};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::over;
use rustc_ast::mut_visit::*;
use rustc_ast::ptr::P;
use rustc_ast::{self as ast, Mutability, Pat, PatKind, PatKind::*, DUMMY_NODE_ID};
use rustc_ast_pretty::pprust;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::DUMMY_SP;

use std::cell::Cell;
use std::mem;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnested or-patterns, e.g., `Some(0) | Some(2)` and
    /// suggests replacing the pattern with a nested one, `Some(0 | 2)`.
    ///
    /// Another way to think of this is that it rewrites patterns in
    /// *disjunctive normal form (DNF)* into *conjunctive normal form (CNF)*.
    ///
    /// ### Why is this bad?
    /// In the example above, `Some` is repeated, which unnecessarily complicates the pattern.
    ///
    /// ### Example
    /// ```rust
    /// fn main() {
    ///     if let Some(0) | Some(2) = Some(0) {}
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn main() {
    ///     if let Some(0 | 2) = Some(0) {}
    /// }
    /// ```
    #[clippy::version = "1.46.0"]
    pub UNNESTED_OR_PATTERNS,
    pedantic,
    "unnested or-patterns, e.g., `Foo(Bar) | Foo(Baz) instead of `Foo(Bar | Baz)`"
}

pub struct UnnestedOrPatterns {
    msrv: Msrv,
}

impl UnnestedOrPatterns {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(UnnestedOrPatterns => [UNNESTED_OR_PATTERNS]);

impl EarlyLintPass for UnnestedOrPatterns {
    fn check_arm(&mut self, cx: &EarlyContext<'_>, a: &ast::Arm) {
        if self.msrv.meets(msrvs::OR_PATTERNS) {
            lint_unnested_or_patterns(cx, &a.pat);
        }
    }

    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        if self.msrv.meets(msrvs::OR_PATTERNS) {
            if let ast::ExprKind::Let(pat, _, _) = &e.kind {
                lint_unnested_or_patterns(cx, pat);
            }
        }
    }

    fn check_param(&mut self, cx: &EarlyContext<'_>, p: &ast::Param) {
        if self.msrv.meets(msrvs::OR_PATTERNS) {
            lint_unnested_or_patterns(cx, &p.pat);
        }
    }

    fn check_local(&mut self, cx: &EarlyContext<'_>, l: &ast::Local) {
        if self.msrv.meets(msrvs::OR_PATTERNS) {
            lint_unnested_or_patterns(cx, &l.pat);
        }
    }

    extract_msrv_attr!(EarlyContext);
}

fn lint_unnested_or_patterns(cx: &EarlyContext<'_>, pat: &Pat) {
    if let Ident(.., None) | Lit(_) | Wild | Path(..) | Range(..) | Rest | MacCall(_) = pat.kind {
        // This is a leaf pattern, so cloning is unprofitable.
        return;
    }

    let mut pat = P(pat.clone());

    // Nix all the paren patterns everywhere so that they aren't in our way.
    remove_all_parens(&mut pat);

    // Transform all unnested or-patterns into nested ones, and if there were none, quit.
    if !unnest_or_patterns(&mut pat) {
        return;
    }

    span_lint_and_then(cx, UNNESTED_OR_PATTERNS, pat.span, "unnested or-patterns", |db| {
        insert_necessary_parens(&mut pat);
        db.span_suggestion_verbose(
            pat.span,
            "nest the patterns",
            pprust::pat_to_string(&pat),
            Applicability::MachineApplicable,
        );
    });
}

/// Remove all `(p)` patterns in `pat`.
fn remove_all_parens(pat: &mut P<Pat>) {
    struct Visitor;
    impl MutVisitor for Visitor {
        fn visit_pat(&mut self, pat: &mut P<Pat>) {
            noop_visit_pat(pat, self);
            let inner = match &mut pat.kind {
                Paren(i) => mem::replace(&mut i.kind, Wild),
                _ => return,
            };
            pat.kind = inner;
        }
    }
    Visitor.visit_pat(pat);
}

/// Insert parens where necessary according to Rust's precedence rules for patterns.
fn insert_necessary_parens(pat: &mut P<Pat>) {
    struct Visitor;
    impl MutVisitor for Visitor {
        fn visit_pat(&mut self, pat: &mut P<Pat>) {
            use ast::BindingAnnotation;
            noop_visit_pat(pat, self);
            let target = match &mut pat.kind {
                // `i @ a | b`, `box a | b`, and `& mut? a | b`.
                Ident(.., Some(p)) | Box(p) | Ref(p, _) if matches!(&p.kind, Or(ps) if ps.len() > 1) => p,
                Ref(p, Mutability::Not) if matches!(p.kind, Ident(BindingAnnotation::MUT, ..)) => p, // `&(mut x)`
                _ => return,
            };
            target.kind = Paren(P(take_pat(target)));
        }
    }
    Visitor.visit_pat(pat);
}

/// Unnest or-patterns `p0 | ... | p1` in the pattern `pat`.
/// For example, this would transform `Some(0) | FOO | Some(2)` into `Some(0 | 2) | FOO`.
fn unnest_or_patterns(pat: &mut P<Pat>) -> bool {
    struct Visitor {
        changed: bool,
    }
    impl MutVisitor for Visitor {
        fn visit_pat(&mut self, p: &mut P<Pat>) {
            // This is a bottom up transformation, so recurse first.
            noop_visit_pat(p, self);

            // Don't have an or-pattern? Just quit early on.
            let Or(alternatives) = &mut p.kind else {
                return
            };

            // Collapse or-patterns directly nested in or-patterns.
            let mut idx = 0;
            let mut this_level_changed = false;
            while idx < alternatives.len() {
                let inner = if let Or(ps) = &mut alternatives[idx].kind {
                    mem::take(ps)
                } else {
                    idx += 1;
                    continue;
                };
                this_level_changed = true;
                alternatives.splice(idx..=idx, inner);
            }

            // Focus on `p_n` and then try to transform all `p_i` where `i > n`.
            let mut focus_idx = 0;
            while focus_idx < alternatives.len() {
                this_level_changed |= transform_with_focus_on_idx(alternatives, focus_idx);
                focus_idx += 1;
            }
            self.changed |= this_level_changed;

            // Deal with `Some(Some(0)) | Some(Some(1))`.
            if this_level_changed {
                noop_visit_pat(p, self);
            }
        }
    }

    let mut visitor = Visitor { changed: false };
    visitor.visit_pat(pat);
    visitor.changed
}

/// Match `$scrutinee` against `$pat` and extract `$then` from it.
/// Panics if there is no match.
macro_rules! always_pat {
    ($scrutinee:expr, $pat:pat => $then:expr) => {
        match $scrutinee {
            $pat => $then,
            _ => unreachable!(),
        }
    };
}

/// Focus on `focus_idx` in `alternatives`,
/// attempting to extend it with elements of the same constructor `C`
/// in `alternatives[focus_idx + 1..]`.
fn transform_with_focus_on_idx(alternatives: &mut Vec<P<Pat>>, focus_idx: usize) -> bool {
    // Extract the kind; we'll need to make some changes in it.
    let mut focus_kind = mem::replace(&mut alternatives[focus_idx].kind, PatKind::Wild);
    // We'll focus on `alternatives[focus_idx]`,
    // so we're draining from `alternatives[focus_idx + 1..]`.
    let start = focus_idx + 1;

    // We're trying to find whatever kind (~"constructor") we found in `alternatives[start..]`.
    let changed = match &mut focus_kind {
        // These pattern forms are "leafs" and do not have sub-patterns.
        // Therefore they are not some form of constructor `C`,
        // with which a pattern `C(p_0)` may be formed,
        // which we would want to join with other `C(p_j)`s.
        Ident(.., None) | Lit(_) | Wild | Path(..) | Range(..) | Rest | MacCall(_)
        // Skip immutable refs, as grouping them saves few characters,
        // and almost always requires adding parens (increasing noisiness).
        // In the case of only two patterns, replacement adds net characters.
        | Ref(_, Mutability::Not)
        // Dealt with elsewhere.
        | Or(_) | Paren(_) => false,
        // Transform `box x | ... | box y` into `box (x | y)`.
        //
        // The cases below until `Slice(...)` deal with *singleton* products.
        // These patterns have the shape `C(p)`, and not e.g., `C(p0, ..., pn)`.
        Box(target) => extend_with_matching(
            target, start, alternatives,
            |k| matches!(k, Box(_)),
            |k| always_pat!(k, Box(p) => p),
        ),
        // Transform `&mut x | ... | &mut y` into `&mut (x | y)`.
        Ref(target, Mutability::Mut) => extend_with_matching(
            target, start, alternatives,
            |k| matches!(k, Ref(_, Mutability::Mut)),
            |k| always_pat!(k, Ref(p, _) => p),
        ),
        // Transform `b @ p0 | ... b @ p1` into `b @ (p0 | p1)`.
        Ident(b1, i1, Some(target)) => extend_with_matching(
            target, start, alternatives,
            // Binding names must match.
            |k| matches!(k, Ident(b2, i2, Some(_)) if b1 == b2 && eq_id(*i1, *i2)),
            |k| always_pat!(k, Ident(_, _, Some(p)) => p),
        ),
        // Transform `[pre, x, post] | ... | [pre, y, post]` into `[pre, x | y, post]`.
        Slice(ps1) => extend_with_matching_product(
            ps1, start, alternatives,
            |k, ps1, idx| matches!(k, Slice(ps2) if eq_pre_post(ps1, ps2, idx)),
            |k| always_pat!(k, Slice(ps) => ps),
        ),
        // Transform `(pre, x, post) | ... | (pre, y, post)` into `(pre, x | y, post)`.
        Tuple(ps1) => extend_with_matching_product(
            ps1, start, alternatives,
            |k, ps1, idx| matches!(k, Tuple(ps2) if eq_pre_post(ps1, ps2, idx)),
            |k| always_pat!(k, Tuple(ps) => ps),
        ),
        // Transform `S(pre, x, post) | ... | S(pre, y, post)` into `S(pre, x | y, post)`.
        TupleStruct(qself1, path1, ps1) => extend_with_matching_product(
            ps1, start, alternatives,
            |k, ps1, idx| matches!(
                k,
                TupleStruct(qself2, path2, ps2)
                    if eq_maybe_qself(qself1, qself2) && eq_path(path1, path2) && eq_pre_post(ps1, ps2, idx)
            ),
            |k| always_pat!(k, TupleStruct(_, _, ps) => ps),
        ),
        // Transform a record pattern `S { fp_0, ..., fp_n }`.
        Struct(qself1, path1, fps1, rest1) => extend_with_struct_pat(qself1, path1, fps1, *rest1, start, alternatives),
    };

    alternatives[focus_idx].kind = focus_kind;
    changed
}

/// Here we focusing on a record pattern `S { fp_0, ..., fp_n }`.
/// In particular, for a record pattern, the order in which the field patterns is irrelevant.
/// So when we fixate on some `ident_k: pat_k`, we try to find `ident_k` in the other pattern
/// and check that all `fp_i` where `i ∈ ((0...n) \ k)` between two patterns are equal.
fn extend_with_struct_pat(
    qself1: &Option<ast::QSelf>,
    path1: &ast::Path,
    fps1: &mut [ast::PatField],
    rest1: bool,
    start: usize,
    alternatives: &mut Vec<P<Pat>>,
) -> bool {
    (0..fps1.len()).any(|idx| {
        let pos_in_2 = Cell::new(None); // The element `k`.
        let tail_or = drain_matching(
            start,
            alternatives,
            |k| {
                matches!(k, Struct(qself2, path2, fps2, rest2)
                if rest1 == *rest2 // If one struct pattern has `..` so must the other.
                && eq_maybe_qself(qself1, qself2)
                && eq_path(path1, path2)
                && fps1.len() == fps2.len()
                && fps1.iter().enumerate().all(|(idx_1, fp1)| {
                    if idx_1 == idx {
                        // In the case of `k`, we merely require identical field names
                        // so that we will transform into `ident_k: p1_k | p2_k`.
                        let pos = fps2.iter().position(|fp2| eq_id(fp1.ident, fp2.ident));
                        pos_in_2.set(pos);
                        pos.is_some()
                    } else {
                        fps2.iter().any(|fp2| eq_field_pat(fp1, fp2))
                    }
                }))
            },
            // Extract `p2_k`.
            |k| always_pat!(k, Struct(_, _, mut fps, _) => fps.swap_remove(pos_in_2.take().unwrap()).pat),
        );
        extend_with_tail_or(&mut fps1[idx].pat, tail_or)
    })
}

/// Like `extend_with_matching` but for products with > 1 factor, e.g., `C(p_0, ..., p_n)`.
/// Here, the idea is that we fixate on some `p_k` in `C`,
/// allowing it to vary between two `targets` and `ps2` (returned by `extract`),
/// while also requiring `ps1[..n] ~ ps2[..n]` (pre) and `ps1[n + 1..] ~ ps2[n + 1..]` (post),
/// where `~` denotes semantic equality.
fn extend_with_matching_product(
    targets: &mut [P<Pat>],
    start: usize,
    alternatives: &mut Vec<P<Pat>>,
    predicate: impl Fn(&PatKind, &[P<Pat>], usize) -> bool,
    extract: impl Fn(PatKind) -> Vec<P<Pat>>,
) -> bool {
    (0..targets.len()).any(|idx| {
        let tail_or = drain_matching(
            start,
            alternatives,
            |k| predicate(k, targets, idx),
            |k| extract(k).swap_remove(idx),
        );
        extend_with_tail_or(&mut targets[idx], tail_or)
    })
}

/// Extract the pattern from the given one and replace it with `Wild`.
/// This is meant for temporarily swapping out the pattern for manipulation.
fn take_pat(from: &mut Pat) -> Pat {
    let dummy = Pat {
        id: DUMMY_NODE_ID,
        kind: Wild,
        span: DUMMY_SP,
        tokens: None,
    };
    mem::replace(from, dummy)
}

/// Extend `target` as an or-pattern with the alternatives
/// in `tail_or` if there are any and return if there were.
fn extend_with_tail_or(target: &mut Pat, tail_or: Vec<P<Pat>>) -> bool {
    fn extend(target: &mut Pat, mut tail_or: Vec<P<Pat>>) {
        match target {
            // On an existing or-pattern in the target, append to it.
            Pat { kind: Or(ps), .. } => ps.append(&mut tail_or),
            // Otherwise convert the target to an or-pattern.
            target => {
                let mut init_or = vec![P(take_pat(target))];
                init_or.append(&mut tail_or);
                target.kind = Or(init_or);
            },
        }
    }

    let changed = !tail_or.is_empty();
    if changed {
        // Extend the target.
        extend(target, tail_or);
    }
    changed
}

// Extract all inner patterns in `alternatives` matching our `predicate`.
// Only elements beginning with `start` are considered for extraction.
fn drain_matching(
    start: usize,
    alternatives: &mut Vec<P<Pat>>,
    predicate: impl Fn(&PatKind) -> bool,
    extract: impl Fn(PatKind) -> P<Pat>,
) -> Vec<P<Pat>> {
    let mut tail_or = vec![];
    let mut idx = 0;
    for pat in alternatives.drain_filter(|p| {
        // Check if we should extract, but only if `idx >= start`.
        idx += 1;
        idx > start && predicate(&p.kind)
    }) {
        tail_or.push(extract(pat.into_inner().kind));
    }
    tail_or
}

fn extend_with_matching(
    target: &mut Pat,
    start: usize,
    alternatives: &mut Vec<P<Pat>>,
    predicate: impl Fn(&PatKind) -> bool,
    extract: impl Fn(PatKind) -> P<Pat>,
) -> bool {
    extend_with_tail_or(target, drain_matching(start, alternatives, predicate, extract))
}

/// Are the patterns in `ps1` and `ps2` equal save for `ps1[idx]` compared to `ps2[idx]`?
fn eq_pre_post(ps1: &[P<Pat>], ps2: &[P<Pat>], idx: usize) -> bool {
    ps1.len() == ps2.len()
        && ps1[idx].is_rest() == ps2[idx].is_rest() // Avoid `[x, ..] | [x, 0]` => `[x, .. | 0]`.
        && over(&ps1[..idx], &ps2[..idx], |l, r| eq_pat(l, r))
        && over(&ps1[idx + 1..], &ps2[idx + 1..], |l, r| eq_pat(l, r))
}
