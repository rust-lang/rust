//! Late-metadata archive member that lists which rlib entries are Rust object files,
//! and potentially other data collected and used when building or linking a rlib.
//! See <https://github.com/rust-lang/rust/issues/138243>.

use std::path::Path;

use object::read::archive::ArchiveFile;
use rustc_serialize::opaque::mem_encoder::MemEncoder;
use rustc_serialize::opaque::{MAGIC_END_BYTES, MemDecoder};
use rustc_serialize::{Decodable, Encodable};

use super::metadata::search_for_section;

pub(crate) const FILENAME: &str = "lib.rmeta-link";
pub(crate) const SECTION: &str = ".rmeta-link";

pub struct RmetaLink {
    pub rust_object_files: Vec<String>,
}

impl RmetaLink {
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut encoder = MemEncoder::new();
        self.rust_object_files.encode(&mut encoder);
        let mut data = encoder.finish();
        data.extend_from_slice(MAGIC_END_BYTES);
        data
    }

    pub(crate) fn decode(data: &[u8]) -> Option<RmetaLink> {
        let mut decoder = MemDecoder::new(data, 0).ok()?;
        let rust_object_files = Vec::<String>::decode(&mut decoder);
        Some(RmetaLink { rust_object_files })
    }
}

/// Reads the digest from an already-parsed archive.
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
