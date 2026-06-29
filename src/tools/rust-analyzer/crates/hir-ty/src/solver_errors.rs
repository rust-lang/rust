//! Handling of trait solver errors and converting them to errors `hir` can pass to `ide-diagnostics`.
//!
//! Note that we also have [`crate::next_solver::infer::errors`], which takes the raw [`NextSolverError`],
//! and converts it into [`FulfillmentError`] that contains more details.
//!
//! [`NextSolverError`]: crate::next_solver::fulfill::NextSolverError

use macros::{TypeFoldable, TypeVisitable};
use rustc_type_ir::{PredicatePolarity, inherent::IntoKind};

use crate::{
    Span,
    next_solver::{
        ClauseKind, DbInterner, PredicateKind, StoredTraitRef, TraitPredicate,
        infer::{
            errors::{FulfillmentError, FulfillmentErrorCode},
            select::SelectionError,
        },
    },
};

#[derive(Debug, Clone, PartialEq, Eq, TypeVisitable, TypeFoldable)]
pub struct SolverDiagnostic {
    pub span: Span,
    pub kind: SolverDiagnosticKind,
}

#[derive(Debug, Clone, PartialEq, Eq, TypeVisitable, TypeFoldable)]
pub enum SolverDiagnosticKind {
    TraitUnimplemented {
        trait_predicate: StoredTraitPredicate,
        root_trait_predicate: Option<StoredTraitPredicate>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, TypeVisitable, TypeFoldable)]
pub struct StoredTraitPredicate {
    pub trait_ref: StoredTraitRef,
    pub polarity: PredicatePolarity,
}

impl StoredTraitPredicate {
    #[inline]
    pub fn get<'db>(&'db self, interner: DbInterner<'db>) -> TraitPredicate<'db> {
        TraitPredicate { polarity: self.polarity, trait_ref: self.trait_ref.get(interner) }
    }
}

impl SolverDiagnostic {
    pub fn from_fulfillment_error(error: &FulfillmentError<'_>) -> Option<Self> {
        let span = error.obligation.cause.span();
        if span.is_dummy() {
            return None;
        }

        // FIXME: Handle more error kinds.
        let kind = match &error.code {
            FulfillmentErrorCode::Select(SelectionError::Unimplemented) => {
                match error.obligation.predicate.kind().skip_binder() {
                    PredicateKind::Clause(ClauseKind::Trait(trait_pred)) => {
                        handle_trait_unimplemented(error, trait_pred)?
                    }
                    _ => return None,
                }
            }
            _ => return None,
        };
        Some(SolverDiagnostic { span, kind })
    }
}

fn handle_trait_unimplemented<'db>(
    error: &FulfillmentError<'db>,
    trait_pred: TraitPredicate<'db>,
) -> Option<SolverDiagnosticKind> {
    let trait_predicate = StoredTraitPredicate {
        trait_ref: StoredTraitRef::new(trait_pred.trait_ref),
        polarity: trait_pred.polarity,
    };

    let root_trait_predicate = match error.root_obligation.predicate.kind().skip_binder() {
        PredicateKind::Clause(ClauseKind::Trait(trait_pred)) => Some(StoredTraitPredicate {
            trait_ref: StoredTraitRef::new(trait_pred.trait_ref),
            polarity: trait_pred.polarity,
        }),
        _ => None,
    };

    Some(SolverDiagnosticKind::TraitUnimplemented { trait_predicate, root_trait_predicate })
}
