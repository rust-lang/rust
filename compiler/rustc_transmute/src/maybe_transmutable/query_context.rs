use crate::layout;

/// Context necessary to answer the question "Are these types transmutable?".
pub(crate) trait QueryContext {
    type Def: layout::Def;
    type Ref: layout::Ref;
}

#[cfg(test)]
pub(crate) mod test {
    use super::QueryContext;

    pub(crate) struct UltraMinimal;

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

    impl QueryContext for UltraMinimal {
        type Def = Def;
        type Ref = !;
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
