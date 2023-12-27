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
        Unstable,
    }

    #[derive(HashStable, Debug, Default)]
    pub struct LibFeatures {
        pub stability: FxHashMap<Symbol, (FeatureStability, Span)>,
    }

    impl LibFeatures {
        pub fn to_vec(&self) -> Vec<(Symbol, FeatureStability)> {
            let mut all_features: Vec<_> =
                self.stability.iter().map(|(&sym, &(stab, _))| (sym, stab)).collect();
            all_features.sort_unstable_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));
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
