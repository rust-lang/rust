// ignore-tidy-filelength :(

mod ambiguity;
mod infer_ctxt_ext;
pub mod on_unimplemented;
pub mod suggestions;
mod type_err_ctxt_ext;

use super::{Obligation, ObligationCause, ObligationCauseCode, PredicateObligation};
use crate::infer::InferCtxt;
use crate::solve::{GenerateProofTree, InferCtxtEvalExt, UseGlobalCache};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::Visitor;
use rustc_middle::traits::solve::Goal;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;
use std::io::Write;
use std::ops::ControlFlow;

pub use self::infer_ctxt_ext::*;
pub use self::type_err_ctxt_ext::*;

// When outputting impl candidates, prefer showing those that are more similar.
//
// We also compare candidates after skipping lifetimes, which has a lower
// priority than exact matches.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CandidateSimilarity {
    Exact { ignoring_lifetimes: bool },
    Fuzzy { ignoring_lifetimes: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImplCandidate<'tcx> {
    pub trait_ref: ty::TraitRef<'tcx>,
    pub similarity: CandidateSimilarity,
    impl_def_id: DefId,
}

enum GetSafeTransmuteErrorAndReason {
    Silent,
    Error { err_msg: String, safe_transmute_explanation: String },
}

struct UnsatisfiedConst(pub bool);

/// Crude way of getting back an `Expr` from a `Span`.
pub struct FindExprBySpan<'hir> {
    pub span: Span,
    pub result: Option<&'hir hir::Expr<'hir>>,
    pub ty_result: Option<&'hir hir::Ty<'hir>>,
}

impl<'hir> FindExprBySpan<'hir> {
    pub fn new(span: Span) -> Self {
        Self { span, result: None, ty_result: None }
    }
}

impl<'v> Visitor<'v> for FindExprBySpan<'v> {
    fn visit_expr(&mut self, ex: &'v hir::Expr<'v>) {
        if self.span == ex.span {
            self.result = Some(ex);
        } else {
            hir::intravisit::walk_expr(self, ex);
        }
    }
    fn visit_ty(&mut self, ty: &'v hir::Ty<'v>) {
        if self.span == ty.span {
            self.ty_result = Some(ty);
        } else {
            hir::intravisit::walk_ty(self, ty);
        }
    }
}

/// Look for type `param` in an ADT being used only through a reference to confirm that suggesting
/// `param: ?Sized` would be a valid constraint.
struct FindTypeParam {
    param: rustc_span::Symbol,
    invalid_spans: Vec<Span>,
    nested: bool,
}

impl<'v> Visitor<'v> for FindTypeParam {
    fn visit_where_predicate(&mut self, _: &'v hir::WherePredicate<'v>) {
        // Skip where-clauses, to avoid suggesting indirection for type parameters found there.
    }

    fn visit_ty(&mut self, ty: &hir::Ty<'_>) {
        // We collect the spans of all uses of the "bare" type param, like in `field: T` or
        // `field: (T, T)` where we could make `T: ?Sized` while skipping cases that are known to be
        // valid like `field: &'a T` or `field: *mut T` and cases that *might* have further `Sized`
        // obligations like `Box<T>` and `Vec<T>`, but we perform no extra analysis for those cases
        // and suggest `T: ?Sized` regardless of their obligations. This is fine because the errors
        // in that case should make what happened clear enough.
        match ty.kind {
            hir::TyKind::Ptr(_) | hir::TyKind::Ref(..) | hir::TyKind::TraitObject(..) => {}
            hir::TyKind::Path(hir::QPath::Resolved(None, path))
                if path.segments.len() == 1 && path.segments[0].ident.name == self.param =>
            {
                if !self.nested {
                    debug!(?ty, "FindTypeParam::visit_ty");
                    self.invalid_spans.push(ty.span);
                }
            }
            hir::TyKind::Path(_) => {
                let prev = self.nested;
                self.nested = true;
                hir::intravisit::walk_ty(self, ty);
                self.nested = prev;
            }
            _ => {
                hir::intravisit::walk_ty(self, ty);
            }
        }
    }
}

/// Summarizes information
#[derive(Clone)]
pub enum ArgKind {
    /// An argument of non-tuple type. Parameters are (name, ty)
    Arg(String, String),

    /// An argument of tuple type. For a "found" argument, the span is
    /// the location in the source of the pattern. For an "expected"
    /// argument, it will be None. The vector is a list of (name, ty)
    /// strings for the components of the tuple.
    Tuple(Option<Span>, Vec<(String, String)>),
}

impl ArgKind {
    fn empty() -> ArgKind {
        ArgKind::Arg("_".to_owned(), "_".to_owned())
    }

    /// Creates an `ArgKind` from the expected type of an
    /// argument. It has no name (`_`) and an optional source span.
    pub fn from_expected_ty(t: Ty<'_>, span: Option<Span>) -> ArgKind {
        match t.kind() {
            ty::Tuple(tys) => ArgKind::Tuple(
                span,
                tys.iter().map(|ty| ("_".to_owned(), ty.to_string())).collect::<Vec<_>>(),
            ),
            _ => ArgKind::Arg("_".to_owned(), t.to_string()),
        }
    }
}

struct HasNumericInferVisitor;

impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for HasNumericInferVisitor {
    type BreakTy = ();

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        if matches!(ty.kind(), ty::Infer(ty::FloatVar(_) | ty::IntVar(_))) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

#[derive(Copy, Clone)]
pub enum DefIdOrName {
    DefId(DefId),
    Name(&'static str),
}

pub fn dump_proof_tree<'tcx>(o: &Obligation<'tcx, ty::Predicate<'tcx>>, infcx: &InferCtxt<'tcx>) {
    infcx.probe(|_| {
        let goal = Goal { predicate: o.predicate, param_env: o.param_env };
        let tree = infcx
            .evaluate_root_goal(goal, GenerateProofTree::Yes(UseGlobalCache::No))
            .1
            .expect("proof tree should have been generated");
        let mut lock = std::io::stdout().lock();
        let _ = lock.write_fmt(format_args!("{tree:?}\n"));
        let _ = lock.flush();
    });
}
