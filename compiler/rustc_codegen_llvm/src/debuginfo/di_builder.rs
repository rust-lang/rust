use libc::c_uint;
use rustc_abi::Align;

use crate::llvm::debuginfo::DIBuilder;
use crate::llvm::{self, ToLlvmBool};

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

        // `LLVMDIBuilderCreateGlobalVariableExpression` would assert if we
        // gave it a null `Expr` pointer, so give it an empty expression
        // instead, which is what the C++ `createGlobalVariableExpression`
        // method would do if given a null `DIExpression` pointer.
        let expr = self.create_expression(&[]);

        let global_var_expr = unsafe {
            llvm::LLVMDIBuilderCreateGlobalVariableExpression(
                this,
                scope,
                name.as_ptr(),
                name.len(),
                linkage_name.as_ptr(),
                linkage_name.len(),
                file,
                line_number,
                ty,
                is_local_to_unit.to_llvm_bool(),
                expr,
                decl,
                align_in_bits,
            )
        };

        unsafe { llvm::LLVMGlobalSetMetadata(val, llvm::MD_dbg, global_var_expr) };

        global_var_expr
    }
}

impl<'ll> DIBuilderExt<'ll> for &DIBuilder<'ll> {
    fn as_di_builder(&self) -> &DIBuilder<'ll> {
        self
    }

    // All other methods have default bodies that rely on `as_di_builder`.
}
