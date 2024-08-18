// @generated
/// Marks a type as a data provider. You can then use macros like
/// `impl_core_helloworld_v1` to add implementations.
///
/// ```ignore
/// struct MyProvider;
/// const _: () = {
///     include!("path/to/generated/macros.rs");
///     make_provider!(MyProvider);
///     impl_core_helloworld_v1!(MyProvider);
/// }
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! __make_provider {
    ($ name : ty) => {
        #[clippy::msrv = "1.67"]
        impl $name {
            #[doc(hidden)]
            #[allow(dead_code)]
            pub const MUST_USE_MAKE_PROVIDER_MACRO: () = ();
        }
        icu_provider::impl_data_provider_never_marker!($name);
    };
}
#[doc(inline)]
pub use __make_provider as make_provider;
#[macro_use]
#[path = "macros/fallback_likelysubtags_v1.rs.data"]
mod fallback_likelysubtags_v1;
#[doc(inline)]
pub use __impl_fallback_likelysubtags_v1 as impl_fallback_likelysubtags_v1;
#[macro_use]
#[path = "macros/fallback_parents_v1.rs.data"]
mod fallback_parents_v1;
#[doc(inline)]
pub use __impl_fallback_parents_v1 as impl_fallback_parents_v1;
#[macro_use]
#[path = "macros/fallback_supplement_co_v1.rs.data"]
mod fallback_supplement_co_v1;
#[doc(inline)]
pub use __impl_fallback_supplement_co_v1 as impl_fallback_supplement_co_v1;
#[macro_use]
#[path = "macros/list_and_v1.rs.data"]
mod list_and_v1;
#[doc(inline)]
pub use __impl_list_and_v1 as impl_list_and_v1;
#[macro_use]
#[path = "macros/list_or_v1.rs.data"]
mod list_or_v1;
#[doc(inline)]
pub use __impl_list_or_v1 as impl_list_or_v1;
