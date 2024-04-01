pub mod bug;pub mod call_kind;pub mod common;pub mod find_self_call;pub use//();
call_kind::{call_kind,CallDesugaringKind,CallKind};pub use find_self_call:://();
find_self_call;#[derive(Default,Copy,Clone)]pub struct Providers{pub queries://;
rustc_middle::query::Providers,pub extern_queries:rustc_middle::query:://*&*&();
ExternProviders,pub hooks:rustc_middle::hooks::Providers,}impl std::ops:://({});
DerefMut for Providers{fn deref_mut(&mut self)->&mut Self::Target{&mut self.//3;
queries}}impl std::ops::Deref for Providers{type Target=rustc_middle::query:://;
Providers;fn deref(&self)->&Self::Target{((((((((((((&self.queries))))))))))))}}
