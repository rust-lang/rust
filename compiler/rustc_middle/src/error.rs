use std::io;
use std::path::Path;

use rustc_errors::codes::*;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

use crate::ty::{Instance, Ty};

#[derive(Diagnostic)]
#[diag("overflow while adding drop-check rules for `{$ty}`", code = E0320)]
#[note("overflowed on `{$overflow_ty}`")]
pub(crate) struct DropCheckOverflow<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub overflow_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("failed to write file {$path}: {$error}\"")]
pub(crate) struct FailedWritingFile<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}

#[derive(Diagnostic)]
#[diag("concrete type differs from previous defining opaque type use")]
pub(crate) struct OpaqueHiddenTypeMismatch<'tcx> {
    pub self_ty: Ty<'tcx>,
    pub other_ty: Ty<'tcx>,
    #[primary_span]
    #[label("expected `{$self_ty}`, got `{$other_ty}`")]
    pub other_span: Span,
    #[subdiagnostic]
    pub sub: TypeMismatchReason,
}

#[derive(Subdiagnostic)]
pub(crate) enum TypeMismatchReason {
    #[label("this expression supplies two conflicting concrete types for the same opaque type")]
    ConflictType {
        #[primary_span]
        span: Span,
    },
    #[note("previous use here")]
    PreviousUse {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("reached the recursion limit finding the struct tail for `{$ty}`")]
#[help(
    "consider increasing the recursion limit by adding a `#![recursion_limit = \"{$suggested_limit}\"]`"
)]
pub(crate) struct RecursionLimitReached<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub suggested_limit: rustc_hir::limit::Limit,
}

#[derive(Diagnostic)]
#[diag("reached the recursion limit while computing the size of `{$ty}`")]
#[help(
    "consider increasing the recursion limit by adding a `#![recursion_limit = \"{$suggested_limit}\"]`"
)]
pub(crate) struct RecursionLimitReachedSizeSkeleton<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub suggested_limit: rustc_hir::limit::Limit,
}

#[derive(Diagnostic)]
#[diag("constant evaluation of enum discriminant resulted in non-integer")]
pub(crate) struct ConstEvalNonIntError {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "to use `strict_coherence` on this trait, the `with_negative_coherence` feature must be enabled"
)]
pub(crate) struct StrictCoherenceNeedsNegativeCoherence {
    #[primary_span]
    pub span: Span,
    #[label("due to this attribute")]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag("requires `{$name}` lang_item")]
pub(crate) struct RequiresLangItem {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "const parameter `{$ct}` is part of concrete type but not used in parameter list for the `impl Trait` type alias"
)]
pub(super) struct ConstNotUsedTraitAlias {
    pub ct: String,
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("erroneous constant encountered")]
pub(crate) struct ErroneousConstant {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("reached the type-length limit while instantiating `{$instance}`")]
#[help("consider adding a `#![type_length_limit=\"{$type_length}\"]` attribute to your crate")]
pub(crate) struct TypeLengthLimit<'tcx> {
    #[primary_span]
    pub span: Span,
    pub instance: Instance<'tcx>,
    pub type_length: usize,
}

#[derive(Diagnostic)]
#[diag("maximum number of nodes exceeded in constant {$global_const_id}")]
pub(crate) struct MaxNumNodesInValtree {
    #[primary_span]
    pub span: Span,
    pub global_const_id: String,
}

#[derive(Diagnostic)]
#[diag("constant {$global_const_id} cannot be used as pattern")]
#[note("constants that reference mutable or external memory cannot be used as patterns")]
pub(crate) struct InvalidConstInValtree {
    #[primary_span]
    pub span: Span,
    pub global_const_id: String,
}

#[derive(Diagnostic)]
#[diag("internal compiler error: reentrant incremental verify failure, suppressing message")]
pub(crate) struct Reentrant;

#[derive(Diagnostic)]
#[diag("internal compiler error: encountered incremental compilation error with {$dep_node}")]
#[note("please follow the instructions below to create a bug report with the provided information")]
#[note("for incremental compilation bugs, having a reproduction is vital")]
#[note(
    "an ideal reproduction consists of the code before and some patch that then triggers the bug when applied and compiled again"
)]
#[note("as a workaround, you can {$run_cmd} to allow your project to compile")]
pub(crate) struct IncrementCompilation {
    pub run_cmd: String,
    pub dep_node: String,
}

pub mod lint_lints {

    use rustc_macros::{Diagnostic, Subdiagnostic};
    use rustc_span::{Span, Symbol};

    #[derive(Diagnostic)]
    #[diag("lint `{$name}` has been renamed to `{$replace}`")]
    pub struct RenamedLint {
        pub name: Symbol,
        pub replace: Symbol,
        #[subdiagnostic]
        pub suggestion: RenamedLintSuggestion,
    }

    #[derive(Diagnostic)]
    #[diag("lint name `{$name}` is deprecated and may not have an effect in the future")]
    pub struct DeprecatedLintName {
        pub name: Symbol,
        #[suggestion("change it to", code = "{replace}", applicability = "machine-applicable")]
        pub suggestion: Span,
        pub replace: Symbol,
    }

    #[derive(Subdiagnostic)]
    pub enum RenamedLintSuggestion {
        #[suggestion("use the new name", code = "{replace}", applicability = "machine-applicable")]
        WithSpan {
            #[primary_span]
            suggestion: Span,
            replace: Symbol,
        },
        #[help("use the new name `{$replace}`")]
        WithoutSpan { replace: Symbol },
    }

    #[derive(Diagnostic)]
    #[diag("lint `{$name}` has been removed: {$reason}")]
    pub struct RemovedLint {
        pub name: Symbol,
        pub reason: String,
    }

    #[derive(Diagnostic)]
    #[diag("unknown lint: `{$name}`")]
    pub struct UnknownLint {
        pub name: Symbol,
        #[subdiagnostic]
        pub suggestion: Option<UnknownLintSuggestion>,
    }
    #[derive(Subdiagnostic)]
    pub enum UnknownLintSuggestion {
        #[suggestion(
            "{$from_rustc ->
            [true] a lint with a similar name exists in `rustc` lints
            *[false] did you mean
        }",
            code = "{replace}",
            applicability = "maybe-incorrect"
        )]
        WithSpan {
            #[primary_span]
            suggestion: Span,
            replace: Symbol,
            from_rustc: bool,
        },
        #[help(
            "{$from_rustc ->
            [true] a lint with a similar name exists in `rustc` lints: `{$replace}`
            *[false] did you mean: `{$replace}`
        }"
        )]
        WithoutSpan { replace: Symbol, from_rustc: bool },
    }

    #[derive(Diagnostic)]
    #[diag("{$level}({$name}) is ignored unless specified at crate level")]
    pub struct IgnoredUnlessCrateSpecified {
        pub level: Symbol,
        pub name: Symbol,
    }
}
