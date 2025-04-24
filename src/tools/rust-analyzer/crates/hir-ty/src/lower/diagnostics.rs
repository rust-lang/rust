//! This files contains the declaration of diagnostics kinds for ty and path lowering.

use hir_def::type_ref::TypeRefId;
use hir_def::{GenericDefId, GenericParamId};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TyLoweringDiagnostic {
    pub source: TypeRefId,
    pub kind: TyLoweringDiagnosticKind,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TyLoweringDiagnosticKind {
    PathDiagnostic(PathLoweringDiagnostic),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenericArgsProhibitedReason {
    Module,
    TyParam,
    SelfTy,
    PrimitiveTy,
    Const,
    Static,
    LocalVariable,
    /// When there is a generic enum, within the expression `Enum::Variant`,
    /// either `Enum` or `Variant` are allowed to have generic arguments, but not both.
    EnumVariant,
}

/// A path can have many generic arguments: each segment may have one associated with the
/// segment, and in addition, each associated type binding may have generic arguments. This
/// enum abstracts over both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathGenericsSource {
    /// Generic arguments directly on the segment.
    Segment(u32),
    /// Generic arguments on an associated type, e.g. `Foo<Assoc<A, B> = C>` or `Foo<Assoc<A, B>: Bound>`.
    AssocType { segment: u32, assoc_type: u32 },
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum PathLoweringDiagnostic {
    GenericArgsProhibited {
        segment: u32,
        reason: GenericArgsProhibitedReason,
    },
    ParenthesizedGenericArgsWithoutFnTrait {
        segment: u32,
    },
    /// The expected lifetimes & types and consts counts can be found by inspecting the `GenericDefId`.
    IncorrectGenericsLen {
        generics_source: PathGenericsSource,
        provided_count: u32,
        expected_count: u32,
        kind: IncorrectGenericsLenKind,
        def: GenericDefId,
    },
    IncorrectGenericsOrder {
        generics_source: PathGenericsSource,
        param_id: GenericParamId,
        arg_idx: u32,
        /// Whether the `GenericArgs` contains a `Self` arg.
        has_self_arg: bool,
    },
    ElidedLifetimesInPath {
        generics_source: PathGenericsSource,
        def: GenericDefId,
        expected_count: u32,
        hard_error: bool,
    },
    /// An elided lifetimes was used (either implicitly, by not specifying lifetimes, or explicitly, by using `'_`),
    /// but lifetime elision could not find a lifetime to replace it with.
    ElisionFailure {
        generics_source: PathGenericsSource,
        def: GenericDefId,
        expected_count: u32,
    },
    MissingLifetime {
        generics_source: PathGenericsSource,
        def: GenericDefId,
        expected_count: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncorrectGenericsLenKind {
    Lifetimes,
    TypesAndConsts,
}
