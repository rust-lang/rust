pub mod bug;

#[derive(Default, Copy, Clone)]
pub struct Providers {
    pub queries: crate::query::Providers,
    pub extern_queries: crate::query::ExternProviders,
    pub hooks: crate::hooks::Providers,
}
