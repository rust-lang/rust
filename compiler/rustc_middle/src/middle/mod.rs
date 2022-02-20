pub mod codegen_fn_attrs;
pub mod dependency_format;
pub mod exported_symbols;
pub mod lang_items;
pub mod lib_features {
    use rustc_data_structures::{fx::FxHashMap, stable_set::StableSet};
    use rustc_span::symbol::Symbol;

    use crate::ty::TyCtxt;

    #[derive(HashStable, Debug)]
    pub struct LibFeatures {
        // A map from feature to stabilisation version.
        pub stable: FxHashMap<Symbol, Symbol>,
        pub unstable: StableSet<Symbol>,
    }

    impl LibFeatures {
        pub fn to_vec(&self, tcx: TyCtxt<'_>) -> Vec<(Symbol, Option<Symbol>)> {
            let hcx = tcx.create_stable_hashing_context();
            let mut all_features: Vec<_> = self
                .stable
                .iter()
                .map(|(f, s)| (*f, Some(*s)))
                .chain(self.unstable.sorted_vector(&hcx).into_iter().map(|f| (*f, None)))
                .collect();
            all_features.sort_unstable_by(|a, b| a.0.as_str().partial_cmp(b.0.as_str()).unwrap());
            all_features
        }
    }
}
pub mod limits;
pub mod privacy;
pub mod region;
pub mod resolve_lifetime;
pub mod stability;

pub fn provide(providers: &mut crate::ty::query::Providers) {
    limits::provide(providers);
}
