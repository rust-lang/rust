#![crate_name = "module_inner"]
#![deny(rustdoc::broken_intra_doc_links)]
/// [SomeType] links to [bar]
pub struct SomeType;
pub trait SomeTrait {}
/// [bar] links to [SomeTrait] and also [SomeType]
pub mod bar {}
