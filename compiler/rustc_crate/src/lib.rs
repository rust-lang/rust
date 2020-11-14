#![feature(int_error_matching)]
#![feature(once_cell)]
#![feature(or_patterns)]

#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate rustc_macros;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_span::symbol::Symbol;

pub mod codegen_fn_attrs;
pub mod cstore;
pub mod dependency_format;

#[derive(HashStable_Generic)]
pub struct LibFeatures {
    // A map from feature to stabilisation version.
    pub stable: FxHashMap<Symbol, Symbol>,
    pub unstable: FxHashSet<Symbol>,
}

impl LibFeatures {
    pub fn to_vec(&self) -> Vec<(Symbol, Option<Symbol>)> {
        let mut all_features: Vec<_> = self
            .stable
            .iter()
            .map(|(f, s)| (*f, Some(*s)))
            .chain(self.unstable.iter().map(|f| (*f, None)))
            .collect();
        all_features.sort_unstable_by_key(|f| f.0.as_str());
        all_features
    }
}

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in librustc_middle.
pub trait HashStableContext: rustc_ast::HashStableContext + rustc_hir::HashStableContext {}
