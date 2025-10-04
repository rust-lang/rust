use crate::llvm;
use crate::llvm::debuginfo::DIBuilder;

/// Extension trait for defining safe wrappers and helper methods on
/// `&DIBuilder<'ll>`, without requiring it to be defined in the same crate.
pub(crate) trait DIBuilderExt<'ll> {
    fn as_di_builder(&self) -> &DIBuilder<'ll>;

    fn create_expression(&self, addr_ops: &[u64]) -> &'ll llvm::Metadata {
        let this = self.as_di_builder();
        unsafe { llvm::LLVMDIBuilderCreateExpression(this, addr_ops.as_ptr(), addr_ops.len()) }
    }
}

impl<'ll> DIBuilderExt<'ll> for &DIBuilder<'ll> {
    fn as_di_builder(&self) -> &DIBuilder<'ll> {
        self
    }

    // All other methods have default bodies that rely on `as_di_builder`.
}
