pub mod bug;

#[derive(Default, Copy, Clone)]
pub struct Providers {
    pub queries: crate::queries::Providers,
    pub extern_queries: crate::queries::ExternProviders,
    pub hooks: crate::hooks::Providers,
}
