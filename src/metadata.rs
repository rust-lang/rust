use rustc::middle::cstore::MetadataLoader;
use rustc_data_structures::owning_ref::{self, OwningRef};
use std::fs::File;
use std::path::Path;

pub const METADATA_FILENAME: &'static [u8] = b"rust.metadata.bin" as &[u8];

pub struct CraneliftMetadataLoader;

impl MetadataLoader for CraneliftMetadataLoader {
    fn get_rlib_metadata(
        &self,
        _target: &crate::rustc_target::spec::Target,
        path: &Path,
    ) -> Result<owning_ref::ErasedBoxRef<[u8]>, String> {
        let mut archive = ar::Archive::new(File::open(path).map_err(|e| format!("{:?}", e))?);
        // Iterate over all entries in the archive:
        while let Some(entry_result) = archive.next_entry() {
            let mut entry = entry_result.map_err(|e| format!("{:?}", e))?;
            if entry.header().identifier() == METADATA_FILENAME {
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
        _target: &crate::rustc_target::spec::Target,
        _path: &Path,
    ) -> Result<owning_ref::ErasedBoxRef<[u8]>, String> {
        Err("dylib metadata loading is not yet supported".to_string())
    }
}
