//! Reading and writing of the rustc metadata for rlibs and dylibs

use std::fs::File;
use std::ops::Deref;
use std::path::Path;

use rustc_codegen_ssa::METADATA_FILENAME;
use rustc_data_structures::owning_ref::{OwningRef, StableAddress};
use rustc_data_structures::rustc_erase_owner;
use rustc_data_structures::sync::MetadataRef;
use rustc_middle::middle::cstore::{EncodedMetadata, MetadataLoader};
use rustc_middle::ty::TyCtxt;
use rustc_session::config;
use rustc_target::spec::Target;

use crate::backend::WriteMetadata;

pub(crate) struct CraneliftMetadataLoader;

struct StableMmap(memmap2::Mmap);

impl Deref for StableMmap {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &*self.0
    }
}

unsafe impl StableAddress for StableMmap {}

fn load_metadata_with(
    path: &Path,
    f: impl for<'a> FnOnce(&'a [u8]) -> Result<&'a [u8], String>,
) -> Result<MetadataRef, String> {
    let file = File::open(path).map_err(|e| format!("{:?}", e))?;
    let data = unsafe { memmap2::MmapOptions::new().map_copy_read_only(&file) }
        .map_err(|e| format!("{:?}", e))?;
    let metadata = OwningRef::new(StableMmap(data)).try_map(f)?;
    return Ok(rustc_erase_owner!(metadata.map_owner_box()));
}

impl MetadataLoader for CraneliftMetadataLoader {
    fn get_rlib_metadata(&self, _target: &Target, path: &Path) -> Result<MetadataRef, String> {
        load_metadata_with(path, |data| {
            let archive = object::read::archive::ArchiveFile::parse(&*data)
                .map_err(|e| format!("{:?}", e))?;

            for entry_result in archive.members() {
                let entry = entry_result.map_err(|e| format!("{:?}", e))?;
                if entry.name() == METADATA_FILENAME.as_bytes() {
                    return Ok(entry.data());
                }
            }

            Err("couldn't find metadata entry".to_string())
        })
    }

    fn get_dylib_metadata(&self, _target: &Target, path: &Path) -> Result<MetadataRef, String> {
        use object::{Object, ObjectSection};

        load_metadata_with(path, |data| {
            let file = object::File::parse(&data).map_err(|e| format!("parse: {:?}", e))?;
            file.section_by_name(".rustc")
                .ok_or("no .rustc section")?
                .data()
                .map_err(|e| format!("failed to read .rustc section: {:?}", e))
        })
    }
}

// Adapted from https://github.com/rust-lang/rust/blob/da573206f87b5510de4b0ee1a9c044127e409bd3/src/librustc_codegen_llvm/base.rs#L47-L112
pub(crate) fn write_metadata<P: WriteMetadata>(
    tcx: TyCtxt<'_>,
    product: &mut P,
) -> EncodedMetadata {
    use snap::write::FrameEncoder;
    use std::io::Write;

    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    enum MetadataKind {
        None,
        Uncompressed,
        Compressed,
    }

    let kind = tcx
        .sess
        .crate_types()
        .iter()
        .map(|ty| match *ty {
            config::CrateType::Executable
            | config::CrateType::Staticlib
            | config::CrateType::Cdylib => MetadataKind::None,

            config::CrateType::Rlib => MetadataKind::Uncompressed,

            config::CrateType::Dylib | config::CrateType::ProcMacro => MetadataKind::Compressed,
        })
        .max()
        .unwrap_or(MetadataKind::None);

    if kind == MetadataKind::None {
        return EncodedMetadata::new();
    }

    let metadata = tcx.encode_metadata();
    if kind == MetadataKind::Uncompressed {
        return metadata;
    }

    assert!(kind == MetadataKind::Compressed);
    let mut compressed = tcx.metadata_encoding_version();
    FrameEncoder::new(&mut compressed).write_all(&metadata.raw_data).unwrap();

    product.add_rustc_section(
        rustc_middle::middle::exported_symbols::metadata_symbol_name(tcx),
        compressed,
        tcx.sess.target.is_like_osx,
    );

    metadata
}
