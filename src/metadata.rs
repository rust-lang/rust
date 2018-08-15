use std::path::Path;
use std::fs::File;
use rustc_data_structures::owning_ref::{self, OwningRef};
use rustc::middle::cstore::MetadataLoader;

pub struct CraneliftMetadataLoader;

impl MetadataLoader for CraneliftMetadataLoader {
    fn get_rlib_metadata(
        &self,
        _target: &::rustc_target::spec::Target,
        path: &Path,
    ) -> Result<owning_ref::ErasedBoxRef<[u8]>, String> {
        let mut archive = ar::Archive::new(File::open(path).map_err(|e| format!("{:?}", e))?);
        // Iterate over all entries in the archive:
        while let Some(entry_result) = archive.next_entry() {
            let mut entry = entry_result.map_err(|e| format!("{:?}", e))?;
            if entry.header().identifier().starts_with(b".rustc.clif_metadata") {
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
        _target: &::rustc_target::spec::Target,
        _path: &Path,
    ) -> Result<owning_ref::ErasedBoxRef<[u8]>, String> {
        //use goblin::Object;

        //let buffer = ::std::fs::read(path).map_err(|e|format!("{:?}", e))?;
        /*match Object::parse(&buffer).map_err(|e|format!("{:?}", e))? {
            Object::Elf(elf) => {
                println!("elf: {:#?}", &elf);
            },
            Object::PE(pe) => {
                println!("pe: {:#?}", &pe);
            },
            Object::Mach(mach) => {
                println!("mach: {:#?}", &mach);
            },
            Object::Archive(archive) => {
                return Err(format!("archive: {:#?}", &archive));
            },
            Object::Unknown(magic) => {
                return Err(format!("unknown magic: {:#x}", magic))
            }
        }*/
        Err("dylib metadata loading is not yet supported".to_string())
    }
}
