use libc::c_uint;
use rustc_abi::Align;

use crate::common::AsCCharPtr;
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

    fn create_static_variable(
        &self,
        scope: Option<&'ll llvm::Metadata>,
        name: &str,
        linkage_name: &str,
        file: &'ll llvm::Metadata,
        line_number: c_uint,
        ty: &'ll llvm::Metadata,
        is_local_to_unit: bool,
        val: &'ll llvm::Value,
        decl: Option<&'ll llvm::Metadata>,
        align: Option<Align>,
    ) -> &'ll llvm::Metadata {
        let this = self.as_di_builder();
        let align_in_bits = align.map_or(0, |align| align.bits() as u32);

        unsafe {
            llvm::LLVMRustDIBuilderCreateStaticVariable(
                this,
                scope,
                name.as_c_char_ptr(),
                name.len(),
                linkage_name.as_c_char_ptr(),
                linkage_name.len(),
                file,
                line_number,
                ty,
                is_local_to_unit,
                val,
                decl,
                align_in_bits,
            )
        }
    }
}

impl<'ll> DIBuilderExt<'ll> for &DIBuilder<'ll> {
    fn as_di_builder(&self) -> &DIBuilder<'ll> {
        self
    }

    // All other methods have default bodies that rely on `as_di_builder`.
}
