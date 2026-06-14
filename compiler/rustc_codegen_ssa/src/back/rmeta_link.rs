//! Late-metadata archive member that lists which rlib entries are Rust object files,
//! and potentially other data collected and used when building or linking a rlib.
//! See <https://github.com/rust-lang/rust/issues/138243>.

use std::fs::File;
use std::path::Path;

use object::read::archive::ArchiveFile;
use rustc_data_structures::memmap::Mmap;
use rustc_serialize::opaque::mem_encoder::MemEncoder;
use rustc_serialize::opaque::{MAGIC_END_BYTES, MemDecoder};
use rustc_serialize::{Decodable, Encodable};
use rustc_target::spec::Target;
use tracing::debug;

use super::metadata::{get_metadata_xcoff, search_for_section};

pub(crate) const FILENAME: &str = "lib.rmeta-link";
pub(crate) const SECTION: &str = ".rmeta-link";

pub struct RmetaLink {
    pub rust_object_files: Vec<String>,
    /// Positionally aligned with `native_libraries` in regular metadata: index `i` is the
    /// bundled filename for native library `i`, or `None` if that library needs no bundling.
    pub native_lib_filenames: Vec<Option<String>>,
}

impl RmetaLink {
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut encoder = MemEncoder::new();
        self.rust_object_files.encode(&mut encoder);
        self.native_lib_filenames.encode(&mut encoder);
        let mut data = encoder.finish();
        data.extend_from_slice(MAGIC_END_BYTES);
        data
    }

    pub(crate) fn decode(data: &[u8]) -> Option<RmetaLink> {
        let mut decoder = MemDecoder::new(data, 0).ok()?;
        let rust_object_files = Vec::<String>::decode(&mut decoder);
        let native_lib_filenames = Vec::<Option<String>>::decode(&mut decoder);
        Some(RmetaLink { rust_object_files, native_lib_filenames })
    }
}

/// Reads the link-time metadata from an already-parsed archive.
pub fn read(archive: &ArchiveFile<'_>, archive_data: &[u8], rlib_path: &Path) -> Option<RmetaLink> {
    for entry in archive.members() {
        let entry = entry.ok()?;
        if entry.name() == FILENAME.as_bytes() {
            let data = entry.data(archive_data).ok()?;
            let section_data = search_for_section(rlib_path, data, SECTION).ok()?;
            return RmetaLink::decode(section_data);
        }
    }
    None
}

/// Like [`read`], but parses the archive from raw bytes.
///
/// Use this when the caller's `ArchiveFile` comes from a different version of the `object` crate.
pub fn read_from_data(archive_data: &[u8], rlib_path: &Path) -> Option<RmetaLink> {
    let archive = ArchiveFile::parse(archive_data).ok()?;
    read(&archive, archive_data, rlib_path)
}

pub fn read_from_path(target: &Target, path: &Path) -> Option<RmetaLink> {
    let Ok(file) = File::open(path) else {
        debug!("failed to open rlib for rmeta-link: {}", path.display());
        return None;
    };
    let Ok(mmap) = (unsafe { Mmap::map(file) }) else {
        debug!("failed to mmap rlib for rmeta-link: {}", path.display());
        return None;
    };

    if target.is_like_aix {
        let archive = ArchiveFile::parse(&*mmap).ok()?;
        for entry in archive.members() {
            let entry = entry.ok()?;
            if entry.name() == FILENAME.as_bytes() {
                let member_data = entry.data(&*mmap).ok()?;
                let section_data = get_metadata_xcoff(path, member_data).ok()?;
                return RmetaLink::decode(section_data);
            }
        }
        return None;
    }

    read_from_data(&mmap, path)
}
