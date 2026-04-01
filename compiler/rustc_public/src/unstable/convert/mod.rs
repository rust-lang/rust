//! This module holds the logic to convert rustc internal ADTs into rustc_public ADTs.
//!
//! The conversion from stable to internal is not meant to be complete,
//! and it should be added as when needed to be passed as input to rustc_public_bridge functions.
//!
//! For contributors, please make sure to avoid calling rustc's internal functions and queries.
//! These should be done via `rustc_public_bridge` APIs, but it's possible to access ADT fields directly.

use std::ops::RangeInclusive;

use rustc_public_bridge::Tables;
use rustc_public_bridge::context::CompilerCtxt;

use super::Stable;
use crate::compiler_interface::BridgeTys;

mod internal;
mod stable;

impl<'tcx, T> Stable<'tcx> for &T
where
    T: Stable<'tcx>,
{
    type T = T::T;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        (*self).stable(tables, cx)
    }
}

impl<'tcx, T> Stable<'tcx> for Option<T>
where
    T: Stable<'tcx>,
{
    type T = Option<T::T>;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        self.as_ref().map(|value| value.stable(tables, cx))
    }
}

impl<'tcx, T, E> Stable<'tcx> for Result<T, E>
where
    T: Stable<'tcx>,
    E: Stable<'tcx>,
{
    type T = Result<T::T, E::T>;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            Ok(val) => Ok(val.stable(tables, cx)),
            Err(error) => Err(error.stable(tables, cx)),
        }
    }
}

impl<'tcx, T> Stable<'tcx> for &[T]
where
    T: Stable<'tcx>,
{
    type T = Vec<T::T>;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        self.iter().map(|e| e.stable(tables, cx)).collect()
    }
}

impl<'tcx, T, U> Stable<'tcx> for (T, U)
where
    T: Stable<'tcx>,
    U: Stable<'tcx>,
{
    type T = (T::T, U::T);
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        (self.0.stable(tables, cx), self.1.stable(tables, cx))
    }
}

impl<'tcx, T> Stable<'tcx> for RangeInclusive<T>
where
    T: Stable<'tcx>,
{
    type T = RangeInclusive<T::T>;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        RangeInclusive::new(self.start().stable(tables, cx), self.end().stable(tables, cx))
    }
}
