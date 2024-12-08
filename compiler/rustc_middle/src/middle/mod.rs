pub mod codegen_fn_attrs;
pub mod debugger_visualizer;
pub mod dependency_format;
pub mod exported_symbols;
pub mod lang_items;
pub mod lib_features {
    use rustc_data_structures::unord::UnordMap;
    use rustc_macros::{HashStable, TyDecodable, TyEncodable};
    use rustc_span::Span;
    use rustc_span::symbol::Symbol;

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    #[derive(HashStable, TyEncodable, TyDecodable)]
    pub enum FeatureStability {
        AcceptedSince(Symbol),
        Unstable,
    }

    #[derive(HashStable, Debug, Default)]
    pub struct LibFeatures {
        pub stability: UnordMap<Symbol, (FeatureStability, Span)>,
    }

    impl LibFeatures {
        pub fn to_sorted_vec(&self) -> Vec<(Symbol, FeatureStability)> {
            self.stability
                .to_sorted_stable_ord()
                .iter()
                .map(|(&sym, &(stab, _))| (sym, stab))
                .collect()
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
