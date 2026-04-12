//! Late-metadata archive member that lists which rlib entries are Rust object files,
//! replacing the `looks_like_rust_object_file` filename heuristic.
//! See <https://github.com/rust-lang/rust/issues/138243>.

use std::path::Path;

use object::read::archive::ArchiveFile;
use rustc_serialize::opaque::mem_encoder::MemEncoder;
use rustc_serialize::opaque::{MAGIC_END_BYTES, MemDecoder};
use rustc_serialize::{Decodable, Encodable};

use super::metadata::search_for_section;

pub(crate) const FILENAME: &str = "lib.rmeta-link";
pub(crate) const SECTION: &str = ".rmeta-link";

pub(crate) struct RmetaLink {
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

/// Reads the digest from already-mapped archive data.
pub(crate) fn read(archive_data: &[u8], rlib_path: &Path) -> Option<RmetaLink> {
    let archive = ArchiveFile::parse(archive_data).ok()?;

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
