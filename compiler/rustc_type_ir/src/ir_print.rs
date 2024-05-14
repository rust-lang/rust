use std::fmt;

use crate::{
    AliasTerm, AliasTy, CoercePredicate, ExistentialProjection, ExistentialTraitRef, Interner,
    NormalizesTo, ProjectionPredicate, SubtypePredicate, TraitPredicate, TraitRef,
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
    TraitPredicate,
    ExistentialTraitRef,
    ExistentialProjection,
    ProjectionPredicate,
    NormalizesTo,
    SubtypePredicate,
    CoercePredicate,
    AliasTy,
    AliasTerm,
);

define_debug_via_print!(TraitRef, ExistentialTraitRef, ExistentialProjection);
