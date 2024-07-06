pub use rustc_type_ir::relate::*;

pub mod combine;

/// Whether aliases should be related structurally or not. Used
/// to adjust the behavior of generalization and combine.
///
/// This should always be `No` unless in a few special-cases when
/// instantiating canonical responses and in the new solver. Each
/// such case should have a comment explaining why it is used.
#[derive(Debug, Copy, Clone)]
pub enum StructurallyRelateAliases {
    Yes,
    No,
}
