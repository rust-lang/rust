#![crate_name = "module_inner"]
#![deny(intra_doc_link_resolution_failure)]
/// [SomeType] links to [bar]
pub struct SomeType;
pub trait SomeTrait {}
/// [bar] links to [SomeTrait] and also [SomeType]
pub mod bar {}
