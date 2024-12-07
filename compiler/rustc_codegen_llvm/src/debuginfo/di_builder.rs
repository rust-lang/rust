//! Safe wrappers for [`DIBuilder`] FFI functions.

use libc::c_uint;
use rustc_abi::{Align, Size};

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

    pub(crate) fn create_union_type(
        &self,
        scope: Option<&'ll Metadata>,
        name: &str,
        file_metadata: &'ll Metadata,
        line_number: c_uint,
        size: Size,
        align: Align,
        flags: DIFlags,
        elements: &[&'ll Metadata],
        unique_id: &str,
    ) -> &'ll Metadata {
        unsafe {
            llvm::LLVMDIBuilderCreateUnionType(
                self,
                scope,
                name.as_ptr(),
                name.len(),
                file_metadata,
                line_number,
                size.bits(),
                align.bits() as u32,
                flags,
                elements.as_ptr(),
                elements.len() as c_uint,
                0, // ("Objective-C runtime version"; default is 0)
                unique_id.as_ptr(),
                unique_id.len(),
            )
        }
    }
}
