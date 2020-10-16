//! Reading and writing of the rustc metadata for rlibs and dylibs

use std::convert::TryFrom;
use std::fs::File;
use std::path::Path;

use rustc_codegen_ssa::METADATA_FILENAME;
use rustc_data_structures::owning_ref::OwningRef;
use rustc_data_structures::rustc_erase_owner;
use rustc_data_structures::sync::MetadataRef;
use rustc_middle::middle::cstore::{EncodedMetadata, MetadataLoader};
use rustc_middle::ty::TyCtxt;
use rustc_session::config;
use rustc_target::spec::Target;

use crate::backend::WriteMetadata;

pub(crate) struct CraneliftMetadataLoader;

impl MetadataLoader for CraneliftMetadataLoader {
    fn get_rlib_metadata(&self, _target: &Target, path: &Path) -> Result<MetadataRef, String> {
        let mut archive = ar::Archive::new(File::open(path).map_err(|e| format!("{:?}", e))?);
        // Iterate over all entries in the archive:
        while let Some(entry_result) = archive.next_entry() {
            let mut entry = entry_result.map_err(|e| format!("{:?}", e))?;
            if entry.header().identifier() == METADATA_FILENAME.as_bytes() {
                let mut buf = Vec::with_capacity(
                    usize::try_from(entry.header().size())
                        .expect("Rlib metadata file too big to load into memory."),
                );
                ::std::io::copy(&mut entry, &mut buf).map_err(|e| format!("{:?}", e))?;
                let buf: OwningRef<Vec<u8>, [u8]> = OwningRef::new(buf).into();
                return Ok(rustc_erase_owner!(buf.map_owner_box()));
            }
        }

        Err("couldn't find metadata entry".to_string())
    }

    fn get_dylib_metadata(&self, _target: &Target, path: &Path) -> Result<MetadataRef, String> {
        use object::{Object, ObjectSection};
        let file = std::fs::read(path).map_err(|e| format!("read:{:?}", e))?;
        let file = object::File::parse(&file).map_err(|e| format!("parse: {:?}", e))?;
        let buf = file
            .section_by_name(".rustc")
            .ok_or("no .rustc section")?
            .data()
            .map_err(|e| format!("failed to read .rustc section: {:?}", e))?
            .to_owned();
        let buf: OwningRef<Vec<u8>, [u8]> = OwningRef::new(buf).into();
        Ok(rustc_erase_owner!(buf.map_owner_box()))
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
    FrameEncoder::new(&mut compressed)
        .write_all(&metadata.raw_data)
        .unwrap();

    product.add_rustc_section(
        rustc_middle::middle::exported_symbols::metadata_symbol_name(tcx),
        compressed,
        tcx.sess.target.options.is_like_osx,
    );

    metadata
}
