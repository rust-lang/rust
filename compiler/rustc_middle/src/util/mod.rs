pub mod bug;

#[derive(Default, Copy, Clone)]
pub struct Providers {
    pub queries: crate::query::Providers,
    pub extern_queries: crate::query::ExternProviders,
    pub fallback_queries: crate::query::FallbackProviders,
    pub hooks: crate::hooks::Providers,
}
