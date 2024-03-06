use crate::layout;

/// Context necessary to answer the question "Are these types transmutable?".
pub(crate) trait QueryContext {
    type Def: layout::Def;
    type Ref: layout::Ref;
    type Scope: Copy;

    fn min_align(&self, reference: Self::Ref) -> usize;
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
        type Scope = ();

        fn min_align(&self, reference: !) -> usize {
            unimplemented!()
        }
    }
}

#[cfg(feature = "rustc")]
mod rustc {
    use super::*;
    use rustc_middle::ty::{Ty, TyCtxt};

    impl<'tcx> super::QueryContext for TyCtxt<'tcx> {
        type Def = layout::rustc::Def<'tcx>;
        type Ref = layout::rustc::Ref<'tcx>;

        type Scope = Ty<'tcx>;

        fn min_align(&self, reference: Self::Ref) -> usize {
            unimplemented!()
        }
    }
}
