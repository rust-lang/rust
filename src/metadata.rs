use std::fs::File;
use std::path::Path;

use rustc::session::config;
use rustc::ty::TyCtxt;
use rustc::middle::cstore::{EncodedMetadata, MetadataLoader};
use rustc_codegen_ssa::METADATA_FILENAME;
use rustc_data_structures::owning_ref::{self, OwningRef};
use rustc_data_structures::rustc_erase_owner;
use rustc_target::spec::Target;

pub struct CraneliftMetadataLoader;

impl MetadataLoader for CraneliftMetadataLoader {
    fn get_rlib_metadata(
        &self,
        _target: &Target,
        path: &Path,
    ) -> Result<owning_ref::ErasedBoxRef<[u8]>, String> {
        let mut archive = ar::Archive::new(File::open(path).map_err(|e| format!("{:?}", e))?);
        // Iterate over all entries in the archive:
        while let Some(entry_result) = archive.next_entry() {
            let mut entry = entry_result.map_err(|e| format!("{:?}", e))?;
            if entry.header().identifier() == METADATA_FILENAME.as_bytes() {
                let mut buf = Vec::new();
                ::std::io::copy(&mut entry, &mut buf).map_err(|e| format!("{:?}", e))?;
                let buf: OwningRef<Vec<u8>, [u8]> = OwningRef::new(buf).into();
                return Ok(rustc_erase_owner!(buf.map_owner_box()));
            }
        }

        Err("couldn't find metadata entry".to_string())
        //self.get_dylib_metadata(target, path)
    }

    fn get_dylib_metadata(
        &self,
        _target: &Target,
        path: &Path,
    ) -> Result<owning_ref::ErasedBoxRef<[u8]>, String> {
        use object::Object;
        let file = std::fs::read(path).map_err(|e| format!("read:{:?}", e))?;
        let file = object::File::parse(&file).map_err(|e| format!("parse: {:?}", e))?;
        let buf = file.section_data_by_name(".rustc").ok_or("no .rustc section")?.into_owned();
        let buf: OwningRef<Vec<u8>, [u8]> = OwningRef::new(buf).into();
        Ok(rustc_erase_owner!(buf.map_owner_box()))
    }
}

// Adapted from https://github.com/rust-lang/rust/blob/da573206f87b5510de4b0ee1a9c044127e409bd3/src/librustc_codegen_llvm/base.rs#L47-L112
pub fn write_metadata<'a, 'gcx>(
    tcx: TyCtxt<'a, 'gcx, 'gcx>,
    artifact: &mut faerie::Artifact
) -> EncodedMetadata {
    use std::io::Write;
    use flate2::Compression;
    use flate2::write::DeflateEncoder;

    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    enum MetadataKind {
        None,
        Uncompressed,
        Compressed
    }

    let kind = tcx.sess.crate_types.borrow().iter().map(|ty| {
        match *ty {
            config::CrateType::Executable |
            config::CrateType::Staticlib |
            config::CrateType::Cdylib => MetadataKind::None,

            config::CrateType::Rlib => MetadataKind::Uncompressed,

            config::CrateType::Dylib |
            config::CrateType::ProcMacro => MetadataKind::Compressed,
        }
    }).max().unwrap_or(MetadataKind::None);

    if kind == MetadataKind::None {
        return EncodedMetadata::new();
    }

    let metadata = tcx.encode_metadata();
    if kind == MetadataKind::Uncompressed {
        return metadata;
    }

    assert!(kind == MetadataKind::Compressed);
    let mut compressed = tcx.metadata_encoding_version();
    DeflateEncoder::new(&mut compressed, Compression::fast())
        .write_all(&metadata.raw_data).unwrap();

    artifact.declare(".rustc", faerie::Decl::section(faerie::SectionKind::Data)).unwrap();
    artifact.define_with_symbols(".rustc", compressed, {
        let mut map = std::collections::BTreeMap::new();
        map.insert(rustc::middle::exported_symbols::metadata_symbol_name(tcx), 0);
        map
    }).unwrap();

    metadata
}
