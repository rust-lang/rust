use rustc_hir::OwnerId;
use rustc_hir::def_id::{DefId, LocalDefId, LocalModDefId, ModDefId};

/// An analogue of the `Into` trait that's intended only for query parameters.
///
/// This exists to allow queries to accept either `DefId` or `LocalDefId` while requiring that the
/// user call `to_def_id` to convert between them everywhere else.
pub trait IntoQueryParam<P> {
    fn into_query_param(self) -> P;
}

impl<P> IntoQueryParam<P> for P {
    #[inline(always)]
    fn into_query_param(self) -> P {
        self
    }
}

impl IntoQueryParam<LocalDefId> for OwnerId {
    #[inline(always)]
    fn into_query_param(self) -> LocalDefId {
        self.def_id
    }
}

impl IntoQueryParam<DefId> for LocalDefId {
    #[inline(always)]
    fn into_query_param(self) -> DefId {
        self.to_def_id()
    }
}

impl IntoQueryParam<DefId> for OwnerId {
    #[inline(always)]
    fn into_query_param(self) -> DefId {
        self.to_def_id()
    }
}

impl IntoQueryParam<DefId> for ModDefId {
    #[inline(always)]
    fn into_query_param(self) -> DefId {
        self.to_def_id()
    }
}

impl IntoQueryParam<DefId> for LocalModDefId {
    #[inline(always)]
    fn into_query_param(self) -> DefId {
        self.to_def_id()
    }
}

impl IntoQueryParam<LocalDefId> for LocalModDefId {
    #[inline(always)]
    fn into_query_param(self) -> LocalDefId {
        self.into()
    }
}
