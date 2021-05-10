//! Reading of the rustc metadata for rlibs and dylibs

use std::fs::File;
use std::path::Path;

use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::owning_ref::OwningRef;
use rustc_data_structures::rustc_erase_owner;
use rustc_data_structures::sync::MetadataRef;
use rustc_middle::middle::cstore::MetadataLoader;
use rustc_target::spec::Target;

use crate::METADATA_FILENAME;

/// The default metadata loader. This is used by cg_llvm and cg_clif.
///
/// # Metadata location
///
/// <dl>
/// <dt>rlib</dt>
/// <dd>The metadata can be found in the `lib.rmeta` file inside of the ar archive.</dd>
/// <dt>dylib</dt>
/// <dd>The metadata can be found in the `.rustc` section of the shared library.</dd>
/// </dl>
pub struct DefaultMetadataLoader;

fn load_metadata_with(
    path: &Path,
    f: impl for<'a> FnOnce(&'a [u8]) -> Result<&'a [u8], String>,
) -> Result<MetadataRef, String> {
    let file =
        File::open(path).map_err(|e| format!("failed to open file '{}': {}", path.display(), e))?;
    let data = unsafe { Mmap::map(file) }
        .map_err(|e| format!("failed to mmap file '{}': {}", path.display(), e))?;
    let metadata = OwningRef::new(data).try_map(f)?;
    return Ok(rustc_erase_owner!(metadata.map_owner_box()));
}

impl MetadataLoader for DefaultMetadataLoader {
    fn get_rlib_metadata(&self, _target: &Target, path: &Path) -> Result<MetadataRef, String> {
        load_metadata_with(path, |data| {
            let archive = object::read::archive::ArchiveFile::parse(&*data)
                .map_err(|e| format!("failed to parse rlib '{}': {}", path.display(), e))?;

            for entry_result in archive.members() {
                let entry = entry_result
                    .map_err(|e| format!("failed to parse rlib '{}': {}", path.display(), e))?;
                if entry.name() == METADATA_FILENAME.as_bytes() {
                    return Ok(entry.data());
                }
            }

            Err(format!("metadata not found in rlib '{}'", path.display()))
        })
    }

    fn get_dylib_metadata(&self, _target: &Target, path: &Path) -> Result<MetadataRef, String> {
        use object::{Object, ObjectSection};

        load_metadata_with(path, |data| {
            let file = object::File::parse(&data)
                .map_err(|e| format!("failed to parse dylib '{}': {}", path.display(), e))?;
            file.section_by_name(".rustc")
                .ok_or_else(|| format!("no .rustc section in '{}'", path.display()))?
                .data()
                .map_err(|e| {
                    format!("failed to read .rustc section in '{}': {}", path.display(), e)
                })
        })
    }
}
