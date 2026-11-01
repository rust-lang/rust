use crate::layout;

/// Context necessary to answer the question "Are these types transmutable?".
pub(crate) trait QueryContext {
    type Def: layout::Def;
    type Region: layout::Region;
    type Type: layout::Type;
}

#[cfg(test)]
pub(crate) mod test {
    use std::marker::PhantomData;

    use super::QueryContext;

    pub(crate) struct UltraMinimal<R = !, T = !>(PhantomData<(R, T)>);

    impl<R, T> Default for UltraMinimal<R, T> {
        fn default() -> Self {
            Self(PhantomData)
        }
    }

    #[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
    pub(crate) enum Def {
        HasSafetyInvariants,
        NoSafetyInvariants,
    }

    impl crate::layout::Def for Def {
        fn has_safety_invariants(&self) -> bool {
            self == &Self::HasSafetyInvariants
        }
    }

    impl<R, T> QueryContext for UltraMinimal<R, T>
    where
        R: crate::layout::Region,
        T: crate::layout::Type,
    {
        type Def = Def;
        type Region = R;
        type Type = T;
    }
}

#[cfg(feature = "rustc")]
mod rustc {
    use rustc_middle::ty::{Region, Ty, TyCtxt};

    use super::*;

    impl<'tcx> super::QueryContext for TyCtxt<'tcx> {
        type Def = layout::rustc::Def<'tcx>;
        type Region = Region<'tcx>;
        type Type = Ty<'tcx>;
    }
}
