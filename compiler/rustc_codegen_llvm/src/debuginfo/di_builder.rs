//! Safe wrappers for [`DIBuilder`] FFI functions.

use libc::c_uint;

use crate::llvm::debuginfo::{DIBuilder, DIFlags};
use crate::llvm::{self, Metadata};

impl<'ll> DIBuilder<'ll> {
    pub(crate) fn create_subroutine_type(
        &self,
        parameter_types: &[Option<&'ll Metadata>],
        flags: DIFlags,
    ) -> &'ll Metadata {
        unsafe {
            llvm::LLVMDIBuilderCreateSubroutineType(
                self,
                None, // ("File"; unused)
                parameter_types.as_ptr(),
                parameter_types.len() as c_uint,
                flags,
            )
        }
    }
}
