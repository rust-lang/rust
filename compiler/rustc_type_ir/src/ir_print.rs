use std::fmt;

use crate::{
    AliasTerm, AliasTy, Binder, ClosureKind, CoercePredicate, ExistentialProjection,
    ExistentialTraitRef, FnSig, HostEffectPredicate, Interner, NormalizesTo, OutlivesPredicate,
    PatternKind, ProjectionPredicate, SubtypePredicate, TraitPredicate, TraitRef, UnevaluatedConst,
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

impl<I: Interner, T> fmt::Display for Binder<I, T>
where
    I: IrPrint<Binder<I, T>>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        <I as IrPrint<Binder<I, T>>>::print(self, fmt)
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
    TraitPredicate,
    ExistentialTraitRef,
    ExistentialProjection,
    ProjectionPredicate,
    NormalizesTo,
    SubtypePredicate,
    CoercePredicate,
    HostEffectPredicate,
    AliasTy,
    AliasTerm,
    FnSig,
    PatternKind,
);

define_debug_via_print!(TraitRef, ExistentialTraitRef, PatternKind);

impl<I: Interner, T> fmt::Display for OutlivesPredicate<I, T>
where
    I: IrPrint<OutlivesPredicate<I, T>>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        <I as IrPrint<OutlivesPredicate<I, T>>>::print(self, fmt)
    }
}

#[cfg(feature = "nightly")]
mod into_diag_arg_impls {
    use rustc_error_messages::{DiagArgValue, IntoDiagArg};

    use super::*;

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

    impl<I: Interner> IntoDiagArg for UnevaluatedConst<I> {
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
