//! Safe wrappers for coverage-specific FFI functions.

use std::ffi::CString;

use crate::coverageinfo::ffi;
use crate::llvm;

pub(crate) fn covmap_var_name() -> CString {
    CString::new(llvm::build_byte_buffer(|s| {
        llvm::LLVMRustCoverageWriteCovmapVarNameToString(s);
    }))
    .expect("covmap variable name should not contain NUL")
}

pub(crate) fn covmap_section_name(llmod: &llvm::Module) -> CString {
    CString::new(llvm::build_byte_buffer(|s| {
        llvm::LLVMRustCoverageWriteCovmapSectionNameToString(llmod, s);
    }))
    .expect("covmap section name should not contain NUL")
}

pub(crate) fn covfun_section_name(llmod: &llvm::Module) -> CString {
    CString::new(llvm::build_byte_buffer(|s| {
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
            mangled_fn_name.as_ptr(),
            mangled_fn_name.len(),
        )
    }
}

pub(crate) fn write_filenames_to_buffer(filenames: &[impl AsRef<str>]) -> Vec<u8> {
    let (pointers, lengths) = filenames
        .into_iter()
        .map(AsRef::as_ref)
        .map(|s: &str| (s.as_ptr(), s.len()))
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

/// Holds tables of the various region types in one struct.
///
/// Don't pass this struct across FFI; pass the individual region tables as
/// pointer/length pairs instead.
///
/// Each field name has a `_regions` suffix for improved readability after
/// exhaustive destructing, which ensures that all region types are handled.
#[derive(Clone, Debug, Default)]
pub(crate) struct Regions {
    pub(crate) code_regions: Vec<ffi::CodeRegion>,
    pub(crate) expansion_regions: Vec<ffi::ExpansionRegion>,
    pub(crate) branch_regions: Vec<ffi::BranchRegion>,
}

impl Regions {
    /// Returns true if none of this structure's tables contain any regions.
    pub(crate) fn has_no_regions(&self) -> bool {
        let Self { code_regions, expansion_regions, branch_regions } = self;

        code_regions.is_empty() && expansion_regions.is_empty() && branch_regions.is_empty()
    }
}

pub(crate) fn write_function_mappings_to_buffer(
    virtual_file_mapping: &[u32],
    expressions: &[ffi::CounterExpression],
    regions: &Regions,
) -> Vec<u8> {
    let Regions { code_regions, expansion_regions, branch_regions } = regions;

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
            buffer,
        )
    })
}

/// Hashes some bytes into a 64-bit hash, via LLVM's `IndexedInstrProf::ComputeHash`,
/// as required for parts of the LLVM coverage mapping format.
pub(crate) fn hash_bytes(bytes: &[u8]) -> u64 {
    unsafe { llvm::LLVMRustCoverageHashBytes(bytes.as_ptr(), bytes.len()) }
}

/// Returns LLVM's `coverage::CovMapVersion::CurrentVersion` (CoverageMapping.h)
/// as a raw numeric value. For historical reasons, the numeric value is 1 less
/// than the number in the version's name, so `Version7` is actually `6u32`.
pub(crate) fn mapping_version() -> u32 {
    llvm::LLVMRustCoverageMappingVersion()
}
