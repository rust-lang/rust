use std::ffi::CStr;
use std::ptr::NonNull;

use rustc_data_structures::small_c_str::SmallCStr;

use crate::diagnostics::LlvmError;
use crate::llvm;

/// Responsible for safely creating and disposing llvm::MCSubtargetInfo via ffi functions.
/// Not cloneable as there is no clone function for llvm::MCSubtargetInfo.
pub(crate) struct OwnedMCSubtargetInfo {
    info_unique: NonNull<llvm::MCSubtargetInfo>,
}

impl OwnedMCSubtargetInfo {
    pub(crate) fn new(
        triple: &CStr,
        cpu: &CStr,
        features: &CStr,
    ) -> Result<Self, LlvmError<'static>> {
        // SAFETY: llvm::LLVMRustCreateMCSubtargetInfo copies pointed-to data.
        let info_ptr = unsafe {
            llvm::LLVMRustCreateMCSubtargetInfo(triple.as_ptr(), cpu.as_ptr(), features.as_ptr())
        };

        NonNull::new(info_ptr)
            .map(|info_unique| Self { info_unique })
            .ok_or_else(|| LlvmError::CreateMCSubtargetInfo { triple: SmallCStr::from(triple) })
    }

    pub(crate) fn has_feature(&self, feature: &CStr) -> bool {
        // SAFETY: `new` ensures we have a valid pointer created by
        // `llvm::LLVMRustCreateMCSubtargetInfo`.
        unsafe {
            llvm::LLVMRustMCSubtargetInfoHasFeature(self.info_unique.as_ref(), feature.as_ptr())
        }
    }
}

impl Drop for OwnedMCSubtargetInfo {
    fn drop(&mut self) {
        // SAFETY: `new` ensures we have a valid pointer created by
        // `llvm::LLVMRustCreateMCSubtargetInfo` and `OwnedMCSubtargetInfo` is not copyable so
        // there is no double free or use after free.
        unsafe {
            llvm::LLVMRustDisposeMCSubtargetInfo(self.info_unique);
        }
    }
}
