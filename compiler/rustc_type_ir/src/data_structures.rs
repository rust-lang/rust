#[cfg(feature = "nightly")]
mod impl_ {
    pub use rustc_data_structures::fx::{
        FxHashMap as HashMap, FxHashSet as HashSet, FxIndexMap as IndexMap, FxIndexSet as IndexSet,
    };
    pub use rustc_data_structures::sso::{SsoHashMap, SsoHashSet};
    pub use rustc_data_structures::stack::ensure_sufficient_stack;
    pub use rustc_data_structures::sync::Lrc;
}

#[cfg(not(feature = "nightly"))]
mod impl_ {
    pub use std::collections::{HashMap, HashMap as SsoHashMap, HashSet, HashSet as SsoHashSet};
    pub use std::sync::Arc as Lrc;

    pub use indexmap::{IndexMap, IndexSet};

    #[inline]
    pub fn ensure_sufficient_stack<R>(f: impl FnOnce() -> R) -> R {
        f()
    }
}

pub use impl_::*;
