#[cfg(feature = "nightly")]
mod impl_ {
    pub use rustc_data_structures::gx::GxHashMap as HashMap;
    pub use rustc_data_structures::gx::GxHashSet as HashSet;
    pub use rustc_data_structures::gx::GxIndexMap as IndexMap;
    pub use rustc_data_structures::gx::GxIndexSet as IndexSet;
    pub use rustc_data_structures::sso::SsoHashMap;
    pub use rustc_data_structures::sso::SsoHashSet;
    pub use rustc_data_structures::stack::ensure_sufficient_stack;
    pub use rustc_data_structures::sync::Lrc;
}

#[cfg(not(feature = "nightly"))]
mod impl_ {
    pub use indexmap::IndexMap;
    pub use indexmap::IndexSet;
    pub use std::collections::HashMap;
    pub use std::collections::HashMap as SsoHashMap;
    pub use std::collections::HashSet;
    pub use std::collections::HashSet as SsoHashSet;
    pub use std::sync::Arc as Lrc;

    #[inline]
    pub fn ensure_sufficient_stack<R>(f: impl FnOnce() -> R) -> R {
        f()
    }
}

pub use impl_::*;
