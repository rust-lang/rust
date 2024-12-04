use either::Either;
use hir_def::type_ref::TypeRefId;

type TypeSource = Either<TypeRefId, hir_def::type_ref::TypeSource>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TyLoweringDiagnostic {
    pub source: TypeSource,
    pub kind: TyLoweringDiagnosticKind,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TyLoweringDiagnosticKind {
    GenericArgsProhibited { segment: u32, reason: GenericArgsProhibitedReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenericArgsProhibitedReason {
    Module,
    TyParam,
    SelfTy,
    PrimitiveTy,
    /// When there is a generic enum, within the expression `Enum::Variant`,
    /// either `Enum` or `Variant` are allowed to have generic arguments, but not both.
    // FIXME: This is not used now but it should be.
    EnumVariant,
}
