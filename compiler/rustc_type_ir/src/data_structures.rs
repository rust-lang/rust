#[cfg(feature = "nightly")]
mod impl_ {
    pub use rustc_data_structures::fx::FxHashMap as HashMap;
    pub use rustc_data_structures::fx::FxHashSet as HashSet;
    pub use rustc_data_structures::sso::SsoHashMap as SsoHashMap;
    pub use rustc_data_structures::sso::SsoHashSet as SsoHashSet;
    pub use rustc_data_structures::sync::Lrc;
}

#[cfg(not(feature = "nightly"))]
mod impl_ {
    pub use std::collections::HashMap;
    pub use std::collections::HashSet;
    pub use std::collections::HashMap as SsoHashMap;
    pub use std::collections::HashSet as SsoHashSet;
    pub use std::sync::Arc as Lrc;
}

pub use impl_::*;