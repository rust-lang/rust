use crate::layout;

/// Context necessary to answer the question "Are these types transmutable?".
pub(crate) trait QueryContext {
    type Def: layout::Def;
    type Ref: layout::Ref;
}

#[cfg(test)]
pub(crate) mod test {
    use std::marker::PhantomData;

    use super::QueryContext;

    pub(crate) struct UltraMinimal<R = !>(PhantomData<R>);

    impl<R> Default for UltraMinimal<R> {
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

    impl<R: crate::layout::Ref> QueryContext for UltraMinimal<R> {
        type Def = Def;
        type Ref = R;
    }
}

#[cfg(feature = "rustc")]
mod rustc {
    use rustc_middle::ty::TyCtxt;

    use super::*;

    impl<'tcx> super::QueryContext for TyCtxt<'tcx> {
        type Def = layout::rustc::Def<'tcx>;
        type Ref = layout::rustc::Ref<'tcx>;
    }
}
