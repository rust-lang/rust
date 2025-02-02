pub mod bug;

#[derive(Default, Copy, Clone)]
pub struct Providers {
    pub queries: rustc_middle::query::Providers,
    pub extern_queries: rustc_middle::query::ExternProviders,
    pub hooks: rustc_middle::hooks::Providers,
}

/// Backwards compatibility hack to keep the diff small. This
/// gives direct access to the `queries` field's fields, which
/// are what almost everything wants access to.
impl std::ops::DerefMut for Providers {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.queries
    }
}

impl std::ops::Deref for Providers {
    type Target = rustc_middle::query::Providers;

    fn deref(&self) -> &Self::Target {
        &self.queries
    }
}
