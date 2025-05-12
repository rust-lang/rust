//! Safe wrappers for coverage-specific FFI functions.

use std::ffi::CString;

use crate::common::AsCCharPtr;
use crate::coverageinfo::ffi;
use crate::llvm;

pub(crate) fn covmap_var_name() -> CString {
    CString::new(llvm::build_byte_buffer(|s| unsafe {
        llvm::LLVMRustCoverageWriteCovmapVarNameToString(s);
    }))
    .expect("covmap variable name should not contain NUL")
}

pub(crate) fn covmap_section_name(llmod: &llvm::Module) -> CString {
    CString::new(llvm::build_byte_buffer(|s| unsafe {
        llvm::LLVMRustCoverageWriteCovmapSectionNameToString(llmod, s);
    }))
    .expect("covmap section name should not contain NUL")
}

pub(crate) fn covfun_section_name(llmod: &llvm::Module) -> CString {
    CString::new(llvm::build_byte_buffer(|s| unsafe {
        llvm::LLVMRustCoverageWriteCovfunSectionNameToString(llmod, s);
    }))
    .expect("covfun section name should not contain NUL")
}

pub(crate) fn create_pgo_func_name_var<'ll>(
    llfn: &'ll llvm::Value,
    mangled_fn_name: &str,
) -> &'ll llvm::Value {
    unsafe {
        llvm::LLVMRustCoverageCreatePGOFuncNameVar(
            llfn,
            mangled_fn_name.as_c_char_ptr(),
            mangled_fn_name.len(),
        )
    }
}

pub(crate) fn write_filenames_to_buffer(filenames: &[impl AsRef<str>]) -> Vec<u8> {
    let (pointers, lengths) = filenames
        .into_iter()
        .map(AsRef::as_ref)
        .map(|s: &str| (s.as_c_char_ptr(), s.len()))
        .unzip::<_, _, Vec<_>, Vec<_>>();

    llvm::build_byte_buffer(|buffer| unsafe {
        llvm::LLVMRustCoverageWriteFilenamesToBuffer(
            pointers.as_ptr(),
            pointers.len(),
            lengths.as_ptr(),
            lengths.len(),
            buffer,
        );
    })
}

pub(crate) fn write_function_mappings_to_buffer(
    virtual_file_mapping: &[u32],
    expressions: &[ffi::CounterExpression],
    regions: &ffi::Regions,
) -> Vec<u8> {
    let ffi::Regions {
        code_regions,
        expansion_regions,
        branch_regions,
        mcdc_branch_regions,
        mcdc_decision_regions,
    } = regions;

    // SAFETY:
    // - All types are FFI-compatible and have matching representations in Rust/C++.
    // - For pointer/length pairs, the pointer and length come from the same vector or slice.
    // - C++ code does not retain any pointers after the call returns.
    llvm::build_byte_buffer(|buffer| unsafe {
        llvm::LLVMRustCoverageWriteFunctionMappingsToBuffer(
            virtual_file_mapping.as_ptr(),
            virtual_file_mapping.len(),
            expressions.as_ptr(),
            expressions.len(),
            code_regions.as_ptr(),
            code_regions.len(),
            expansion_regions.as_ptr(),
            expansion_regions.len(),
            branch_regions.as_ptr(),
            branch_regions.len(),
            mcdc_branch_regions.as_ptr(),
            mcdc_branch_regions.len(),
            mcdc_decision_regions.as_ptr(),
            mcdc_decision_regions.len(),
            buffer,
        )
    })
}

/// Hashes some bytes into a 64-bit hash, via LLVM's `IndexedInstrProf::ComputeHash`,
/// as required for parts of the LLVM coverage mapping format.
pub(crate) fn hash_bytes(bytes: &[u8]) -> u64 {
    unsafe { llvm::LLVMRustCoverageHashBytes(bytes.as_c_char_ptr(), bytes.len()) }
}

/// Returns LLVM's `coverage::CovMapVersion::CurrentVersion` (CoverageMapping.h)
/// as a raw numeric value. For historical reasons, the numeric value is 1 less
/// than the number in the version's name, so `Version7` is actually `6u32`.
pub(crate) fn mapping_version() -> u32 {
    unsafe { llvm::LLVMRustCoverageMappingVersion() }
}
