pub mod codegen_fn_attrs;
pub mod debugger_visualizer;
pub mod dependency_format;
pub mod exported_symbols;
pub mod lang_items;
pub mod lib_features {
    use rustc_data_structures::fx::FxHashMap;
    use rustc_span::{symbol::Symbol, Span};

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[derive(HashStable, TyEncodable, TyDecodable)]
    pub enum FeatureStability {
        AcceptedSince(Symbol),
        Unstable(bool),
    }

    #[derive(HashStable, Debug, Default)]
    pub struct LibFeatures {
        /// A map from feature to stabilisation version.
        pub stable: FxHashMap<Symbol, (Symbol, Span)>,
        pub unstable: FxHashMap<Symbol, (bool, Span)>,
    }

    impl LibFeatures {
        pub fn to_vec(&self) -> Vec<(Symbol, FeatureStability)> {
            let mut all_features: Vec<_> = self
                .stable
                .iter()
                .map(|(f, (s, _))| (*f, FeatureStability::AcceptedSince(*s)))
                .chain(
                    self.unstable
                        .iter()
                        .map(|(f, (internal, _))| (*f, FeatureStability::Unstable(*internal))),
                )
                .collect();
            all_features.sort_unstable_by(|a, b| a.0.as_str().partial_cmp(b.0.as_str()).unwrap());
            all_features
        }
    }
}
pub mod limits;
pub mod privacy;
pub mod region;
pub mod resolve_bound_vars;
pub mod stability;

pub fn provide(providers: &mut crate::query::Providers) {
    limits::provide(providers);
}
