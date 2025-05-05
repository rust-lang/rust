use std::ops::RangeInclusive;

use rustc_smir::Tables;
use rustc_smir::context::SmirCtxt;

use super::compiler_interface::BridgeTys;
use crate::rustc_smir;

mod internal;
mod stable;

impl<'tcx, T> Stable<'tcx> for &T
where
    T: Stable<'tcx>,
{
    type T = T::T;

    fn stable(
        &self,
        tables: &mut Tables<'tcx, BridgeTys>,
        cx: &SmirCtxt<'tcx, BridgeTys>,
    ) -> Self::T {
        (*self).stable(tables, cx)
    }
}

impl<'tcx, T> Stable<'tcx> for Option<T>
where
    T: Stable<'tcx>,
{
    type T = Option<T::T>;

    fn stable(
        &self,
        tables: &mut Tables<'tcx, BridgeTys>,
        cx: &SmirCtxt<'tcx, BridgeTys>,
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

    fn stable(
        &self,
        tables: &mut Tables<'tcx, BridgeTys>,
        cx: &SmirCtxt<'tcx, BridgeTys>,
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
    fn stable(
        &self,
        tables: &mut Tables<'tcx, BridgeTys>,
        cx: &SmirCtxt<'tcx, BridgeTys>,
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
    fn stable(
        &self,
        tables: &mut Tables<'tcx, BridgeTys>,
        cx: &SmirCtxt<'tcx, BridgeTys>,
    ) -> Self::T {
        (self.0.stable(tables, cx), self.1.stable(tables, cx))
    }
}

impl<'tcx, T> Stable<'tcx> for RangeInclusive<T>
where
    T: Stable<'tcx>,
{
    type T = RangeInclusive<T::T>;
    fn stable(
        &self,
        tables: &mut Tables<'tcx, BridgeTys>,
        cx: &SmirCtxt<'tcx, BridgeTys>,
    ) -> Self::T {
        RangeInclusive::new(self.start().stable(tables, cx), self.end().stable(tables, cx))
    }
}

/// Trait used to convert between an internal MIR type to a Stable MIR type.
pub trait Stable<'tcx> {
    /// The stable representation of the type implementing Stable.
    type T;
    /// Converts an object to the equivalent Stable MIR representation.
    fn stable(
        &self,
        tables: &mut Tables<'tcx, BridgeTys>,
        cx: &SmirCtxt<'tcx, BridgeTys>,
    ) -> Self::T;
}

/// Trait used to translate a stable construct to its rustc counterpart.
///
/// This is basically a mirror of [Stable].
pub trait RustcInternal {
    type T<'tcx>;
    fn internal<'tcx>(
        &self,
        tables: &mut Tables<'_, BridgeTys>,
        cx: &SmirCtxt<'tcx, BridgeTys>,
    ) -> Self::T<'tcx>;
}
