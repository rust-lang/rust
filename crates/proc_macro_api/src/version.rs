//! Reading proc-macro rustc version information from binary data

use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

use memmap2::Mmap;
use object::read::{File as BinaryFile, Object, ObjectSection};
use snap::read::FrameDecoder as SnapDecoder;

#[derive(Debug)]
pub struct RustCInfo {
    pub version: (usize, usize, usize),
    pub channel: String,
    pub commit: String,
    pub date: String,
}

/// Read rustc dylib information
pub fn read_dylib_info(dylib_path: &Path) -> io::Result<RustCInfo> {
    macro_rules! err {
        ($e:literal) => {
            io::Error::new(io::ErrorKind::InvalidData, $e)
        };
    }

    let ver_str = read_version(dylib_path)?;
    let mut items = ver_str.split_whitespace();
    let tag = items.next().ok_or(err!("version format error"))?;
    if tag != "rustc" {
        return Err(err!("version format error (No rustc tag)"));
    }

    let version_part = items.next().ok_or(err!("no version string"))?;
    let mut version_parts = version_part.split('-');
    let version = version_parts.next().ok_or(err!("no version"))?;
    let channel = version_parts.next().unwrap_or_default().to_string();

    let commit = items.next().ok_or(err!("no commit info"))?;
    // remove (
    if commit.len() == 0 {
        return Err(err!("commit format error"));
    }
    let commit = commit[1..].to_string();
    let date = items.next().ok_or(err!("no date info"))?;
    // remove )
    if date.len() == 0 {
        return Err(err!("date format error"));
    }
    let date = date[0..date.len() - 2].to_string();

    let version_numbers = version
        .split('.')
        .map(|it| it.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| err!("version number error"))?;

    if version_numbers.len() != 3 {
        return Err(err!("version number format error"));
    }
    let version = (version_numbers[0], version_numbers[1], version_numbers[2]);

    Ok(RustCInfo { version, channel, commit, date })
}

/// This is used inside read_version() to locate the ".rustc" section
/// from a proc macro crate's binary file.
fn read_section<'a>(dylib_binary: &'a [u8], section_name: &str) -> io::Result<&'a [u8]> {
    BinaryFile::parse(dylib_binary)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        .section_by_name(section_name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "section read error"))?
        .data()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Check the version of rustc that was used to compile a proc macro crate's
///
/// binary file.
/// A proc macro crate binary's ".rustc" section has following byte layout:
/// * [b'r',b'u',b's',b't',0,0,0,5] is the first 8 bytes
/// * ff060000 734e6150 is followed, it's the snappy format magic bytes,
///   means bytes from here(including this sequence) are compressed in
///   snappy compression format. Version info is inside here, so decompress
///   this.
/// The bytes you get after decompressing the snappy format portion has
/// following layout:
/// * [b'r',b'u',b's',b't',0,0,0,5] is the first 8 bytes(again)
/// * [crate root bytes] next 4 bytes is to store crate root position,
///   according to rustc's source code comment
/// * [length byte] next 1 byte tells us how many bytes we should read next
///   for the version string's utf8 bytes
/// * [version string bytes encoded in utf8] <- GET THIS BOI
/// * [some more bytes that we don really care but still there] :-)
/// Check this issue for more about the bytes layout:
/// https://github.com/rust-analyzer/rust-analyzer/issues/6174
fn read_version(dylib_path: &Path) -> io::Result<String> {
    let dylib_file = File::open(dylib_path)?;
    let dylib_mmaped = unsafe { Mmap::map(&dylib_file) }?;

    let dot_rustc = read_section(&dylib_mmaped, ".rustc")?;

    let header = &dot_rustc[..8];
    const EXPECTED_HEADER: [u8; 8] = [b'r', b'u', b's', b't', 0, 0, 0, 5];
    // check if header is valid
    if header != EXPECTED_HEADER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("only metadata version 5 is supported, section header was: {:?}", header),
        ));
    }

    let snappy_portion = &dot_rustc[8..];

    let mut snappy_decoder = SnapDecoder::new(snappy_portion);

    // the bytes before version string bytes, so this basically is:
    // 8 bytes for [b'r',b'u',b's',b't',0,0,0,5]
    // 4 bytes for [crate root bytes]
    // 1 byte for length of version string
    // so 13 bytes in total, and we should check the 13th byte
    // to know the length
    let mut bytes_before_version = [0u8; 13];
    snappy_decoder.read_exact(&mut bytes_before_version)?;
    let length = bytes_before_version[12]; // what? can't use -1 indexing?

    let mut version_string_utf8 = vec![0u8; length as usize];
    snappy_decoder.read_exact(&mut version_string_utf8)?;
    let version_string = String::from_utf8(version_string_utf8);
    version_string.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
