use crate::llvm;
use crate::llvm::archive_ro::ArchiveRO;
use crate::llvm::{mk_section_iter, False, ObjectFile};
use rustc_middle::middle::cstore::MetadataLoader;
use rustc_target::spec::Target;

use rustc_codegen_ssa::METADATA_FILENAME;
use rustc_data_structures::owning_ref::OwningRef;
use rustc_data_structures::rustc_erase_owner;
use tracing::debug;

use rustc_fs_util::path_to_c_string;
use std::path::Path;
use std::slice;

pub use rustc_data_structures::sync::MetadataRef;

pub struct LlvmMetadataLoader;

impl MetadataLoader for LlvmMetadataLoader {
    fn get_rlib_metadata(&self, _: &Target, filename: &Path) -> Result<MetadataRef, String> {
        // Use ArchiveRO for speed here, it's backed by LLVM and uses mmap
        // internally to read the file. We also avoid even using a memcpy by
        // just keeping the archive along while the metadata is in use.
        let archive =
            ArchiveRO::open(filename).map(|ar| OwningRef::new(Box::new(ar))).map_err(|e| {
                debug!("llvm didn't like `{}`: {}", filename.display(), e);
                format!("failed to read rlib metadata in '{}': {}", filename.display(), e)
            })?;
        let buf: OwningRef<_, [u8]> = archive.try_map(|ar| {
            ar.iter()
                .filter_map(|s| s.ok())
                .find(|sect| sect.name() == Some(METADATA_FILENAME))
                .map(|s| s.data())
                .ok_or_else(|| {
                    debug!("didn't find '{}' in the archive", METADATA_FILENAME);
                    format!("failed to read rlib metadata: '{}'", filename.display())
                })
        })?;
        Ok(rustc_erase_owner!(buf))
    }

    fn get_dylib_metadata(&self, target: &Target, filename: &Path) -> Result<MetadataRef, String> {
        unsafe {
            let buf = path_to_c_string(filename);
            let mb = llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(buf.as_ptr())
                .ok_or_else(|| format!("error reading library: '{}'", filename.display()))?;
            let of =
                ObjectFile::new(mb).map(|of| OwningRef::new(Box::new(of))).ok_or_else(|| {
                    format!("provided path not an object file: '{}'", filename.display())
                })?;
            let buf = of.try_map(|of| search_meta_section(of, target, filename))?;
            Ok(rustc_erase_owner!(buf))
        }
    }
}

fn search_meta_section<'a>(
    of: &'a ObjectFile,
    target: &Target,
    filename: &Path,
) -> Result<&'a [u8], String> {
    unsafe {
        let si = mk_section_iter(of.llof);
        while llvm::LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) == False {
            let mut name_buf = None;
            let name_len = llvm::LLVMRustGetSectionName(si.llsi, &mut name_buf);
            let name = name_buf.map_or(
                String::new(), // We got a NULL ptr, ignore `name_len`.
                |buf| {
                    String::from_utf8(
                        slice::from_raw_parts(buf.as_ptr() as *const u8, name_len as usize)
                            .to_vec(),
                    )
                    .unwrap()
                },
            );
            debug!("get_metadata_section: name {}", name);
            if read_metadata_section_name(target) == name {
                let cbuf = llvm::LLVMGetSectionContents(si.llsi);
                let csz = llvm::LLVMGetSectionSize(si.llsi) as usize;
                // The buffer is valid while the object file is around
                let buf: &'a [u8] = slice::from_raw_parts(cbuf as *const u8, csz);
                return Ok(buf);
            }
            llvm::LLVMMoveToNextSection(si.llsi);
        }
    }
    Err(format!("metadata not found: '{}'", filename.display()))
}

pub fn metadata_section_name(target: &Target) -> &'static str {
    // Historical note:
    //
    // When using link.exe it was seen that the section name `.note.rustc`
    // was getting shortened to `.note.ru`, and according to the PE and COFF
    // specification:
    //
    // > Executable images do not use a string table and do not support
    // > section names longer than 8 characters
    //
    // https://docs.microsoft.com/en-us/windows/win32/debug/pe-format
    //
    // As a result, we choose a slightly shorter name! As to why
    // `.note.rustc` works on MinGW, that's another good question...

    if target.is_like_osx { "__DATA,.rustc" } else { ".rustc" }
}

fn read_metadata_section_name(_target: &Target) -> &'static str {
    ".rustc"
}
