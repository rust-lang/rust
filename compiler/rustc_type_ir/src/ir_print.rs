use std::fmt;

#[cfg(feature = "nightly")]
use crate::{AliasConst, ClosureKind};
use crate::{
    AliasTerm, AliasTy, Binder, CoercePredicate, Const, ExistentialProjection, ExistentialTraitRef,
    FnSig, HostEffectClause, Interner, NormalizesTo, OutlivesClause, PatternKind, Placeholder,
    ProjectionClause, Region, SubtypePredicate, TraitClause, TraitRef,
};

pub trait IrPrint<T> {
    fn print(t: &T, fmt: &mut fmt::Formatter<'_>) -> fmt::Result;
    fn print_debug(t: &T, fmt: &mut fmt::Formatter<'_>) -> fmt::Result;
}

macro_rules! define_display_via_print {
    ($($ty:ident),+ $(,)?) => {
        $(
            impl<I: Interner> fmt::Display for $ty<I> {
                fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                    <I as IrPrint<$ty<I>>>::print(self, fmt)
                }
            }
        )*
    }
}

macro_rules! define_debug_via_print {
    ($($ty:ident),+ $(,)?) => {
        $(
            impl<I: Interner> fmt::Debug for $ty<I> {
                fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                    <I as IrPrint<$ty<I>>>::print_debug(self, fmt)
                }
            }
        )*
    }
}

define_display_via_print!(
    TraitRef,
    TraitClause,
    ExistentialTraitRef,
    ExistentialProjection,
    ProjectionClause,
    NormalizesTo,
    SubtypePredicate,
    CoercePredicate,
    HostEffectClause,
    AliasTy,
    AliasTerm,
    FnSig,
    PatternKind,
);

define_debug_via_print!(TraitRef, ExistentialTraitRef, PatternKind);

impl<I: Interner> fmt::Display for Region<I>
where
    I: IrPrint<Region<I>>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        <I as IrPrint<Region<I>>>::print(self, fmt)
    }
}

// Display is implemented where the representation is defined. Each frontend
// provides the actual formatting through IrPrint.
impl<I: Interner + IrPrint<crate::predicates::Predicate<I>>> fmt::Display
    for crate::predicates::Predicate<I>
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        <I as IrPrint<Self>>::print(self, fmt)
    }
}

impl<I: Interner + IrPrint<crate::predicates::Clause<I>>> fmt::Display
    for crate::predicates::Clause<I>
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        <I as IrPrint<Self>>::print(self, fmt)
    }
}

impl<I: Interner> fmt::Display for Const<I>
where
    I: IrPrint<Const<I>>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        <I as IrPrint<Const<I>>>::print(self, fmt)
    }
}

impl<I: Interner, T> fmt::Display for OutlivesClause<I, T>
where
    I: IrPrint<OutlivesClause<I, T>>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        <I as IrPrint<OutlivesClause<I, T>>>::print(self, fmt)
    }
}

impl<I: Interner, T> fmt::Display for Binder<I, T>
where
    I: IrPrint<Binder<I, T>>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        <I as IrPrint<Binder<I, T>>>::print(self, fmt)
    }
}

impl<I: Interner, T> fmt::Display for Placeholder<I, T>
where
    I: IrPrint<Placeholder<I, T>>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        <I as IrPrint<Placeholder<I, T>>>::print(self, fmt)
    }
}

/// Provides frontend-specific diagnostic formatting for predicates.
///
/// The frontend owns the type-printing context and long-type path handling.
#[cfg(feature = "nightly")]
pub trait PredicateDiagFormatter: Interner {
    fn predicate_diag_string(
        predicate: crate::predicates::Predicate<Self>,
        path: &mut Option<std::path::PathBuf>,
    ) -> String;

    fn clause_diag_string(
        clause: crate::predicates::Clause<Self>,
        path: &mut Option<std::path::PathBuf>,
    ) -> String;
}

#[cfg(feature = "nightly")]
mod into_diag_arg_impls {
    use rustc_error_messages::{DiagArgValue, IntoDiagArg};

    use super::*;

    impl<I: PredicateDiagFormatter> IntoDiagArg for crate::predicates::Predicate<I> {
        fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
            DiagArgValue::Str(I::predicate_diag_string(self, path).into())
        }
    }

    impl<I: PredicateDiagFormatter> IntoDiagArg for crate::predicates::Clause<I> {
        fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
            DiagArgValue::Str(I::clause_diag_string(self, path).into())
        }
    }

    impl<I: Interner> IntoDiagArg for TraitRef<I> {
        fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
            self.to_string().into_diag_arg(path)
        }
    }

    impl<I: Interner> IntoDiagArg for ExistentialTraitRef<I> {
        fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
            self.to_string().into_diag_arg(path)
        }
    }

    impl<I: Interner + IrPrint<Region<I>>> IntoDiagArg for Region<I> {
        fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
            self.to_string().into_diag_arg(path)
        }
    }

    impl<I: Interner> IntoDiagArg for AliasConst<I> {
        fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
            format!("{self:?}").into_diag_arg(path)
        }
    }

    impl<I: Interner> IntoDiagArg for FnSig<I> {
        fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
            format!("{self:?}").into_diag_arg(path)
        }
    }

    impl<I: Interner, T: IntoDiagArg> IntoDiagArg for Binder<I, T> {
        fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
            self.skip_binder().into_diag_arg(path)
        }
    }

    impl IntoDiagArg for ClosureKind {
        fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
            DiagArgValue::Str(self.as_str().into())
        }
    }
}
