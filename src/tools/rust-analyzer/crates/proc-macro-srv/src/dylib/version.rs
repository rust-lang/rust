//! Reading proc-macro rustc version information from binary data

use std::io::{self, Read};

use object::read::{Object, ObjectSection};

#[derive(Debug)]
#[allow(dead_code)]
pub struct RustCInfo {
    pub version: (usize, usize, usize),
    pub channel: String,
    pub commit: Option<String>,
    pub date: Option<String>,
    // something like "rustc 1.58.1 (db9d1b20b 2022-01-20)"
    pub version_string: String,
}

/// Read rustc dylib information
pub fn read_dylib_info(obj: &object::File<'_>) -> io::Result<RustCInfo> {
    macro_rules! err {
        ($e:literal) => {
            io::Error::new(io::ErrorKind::InvalidData, $e)
        };
    }

    let ver_str = read_version(obj)?;
    let mut items = ver_str.split_whitespace();
    let tag = items.next().ok_or_else(|| err!("version format error"))?;
    if tag != "rustc" {
        return Err(err!("version format error (No rustc tag)"));
    }

    let version_part = items.next().ok_or_else(|| err!("no version string"))?;
    let mut version_parts = version_part.split('-');
    let version = version_parts.next().ok_or_else(|| err!("no version"))?;
    let channel = version_parts.next().unwrap_or_default().to_owned();

    let commit = match items.next() {
        Some(commit) => {
            match commit.len() {
                0 => None,
                _ => Some(commit[1..].to_string() /* remove ( */),
            }
        }
        None => None,
    };
    let date = match items.next() {
        Some(date) => {
            match date.len() {
                0 => None,
                _ => Some(date[0..date.len() - 2].to_string() /* remove ) */),
            }
        }
        None => None,
    };

    let version_numbers = version
        .split('.')
        .map(|it| it.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| err!("version number error"))?;

    if version_numbers.len() != 3 {
        return Err(err!("version number format error"));
    }
    let version = (version_numbers[0], version_numbers[1], version_numbers[2]);

    Ok(RustCInfo { version, channel, commit, date, version_string: ver_str })
}

/// This is used inside read_version() to locate the ".rustc" section
/// from a proc macro crate's binary file.
fn read_section<'a>(obj: &object::File<'a>, section_name: &str) -> io::Result<&'a [u8]> {
    obj.section_by_name(section_name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "section read error"))?
        .data()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Check the version of rustc that was used to compile a proc macro crate's
/// binary file.
///
/// A proc macro crate binary's ".rustc" section has following byte layout:
/// * [b'r',b'u',b's',b't',0,0,0,5] is the first 8 bytes
/// * ff060000 734e6150 is followed, it's the snappy format magic bytes,
///   means bytes from here(including this sequence) are compressed in
///   snappy compression format. Version info is inside here, so decompress
///   this.
///
/// The bytes you get after decompressing the snappy format portion has
/// following layout:
/// * [b'r',b'u',b's',b't',0,0,0,5] is the first 8 bytes(again)
/// * [crate root bytes] next 8 bytes (4 in old versions) is to store
///   crate root position, according to rustc's source code comment
/// * [length byte] next 1 byte tells us how many bytes we should read next
///   for the version string's utf8 bytes
/// * [version string bytes encoded in utf8] <- GET THIS BOI
/// * [some more bytes that we don't really care but about still there] :-)
///
/// Check this issue for more about the bytes layout:
/// <https://github.com/rust-lang/rust-analyzer/issues/6174>
pub fn read_version(obj: &object::File<'_>) -> io::Result<String> {
    let dot_rustc = read_section(obj, ".rustc")?;

    // check if magic is valid
    if &dot_rustc[0..4] != b"rust" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown metadata magic, expected `rust`, found `{:?}`", &dot_rustc[0..4]),
        ));
    }
    let version = u32::from_be_bytes([dot_rustc[4], dot_rustc[5], dot_rustc[6], dot_rustc[7]]);
    // Last breaking version change is:
    // https://github.com/rust-lang/rust/commit/b94cfefc860715fb2adf72a6955423d384c69318
    let (mut metadata_portion, bytes_before_version) = match version {
        8 => {
            let len_bytes = &dot_rustc[8..12];
            let data_len = u32::from_be_bytes(len_bytes.try_into().unwrap()) as usize;
            (&dot_rustc[12..data_len + 12], 13)
        }
        9 | 10 => {
            let len_bytes = &dot_rustc[8..16];
            let data_len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
            (&dot_rustc[16..data_len + 12], 17)
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported metadata version {version}"),
            ));
        }
    };

    // We're going to skip over the bytes before the version string, so basically:
    // 8 bytes for [b'r',b'u',b's',b't',0,0,0,5]
    // 4 or 8 bytes for [crate root bytes]
    // 1 byte for length of version string
    // so 13 or 17 bytes in total, and we should check the last of those bytes
    // to know the length
    let mut bytes = [0u8; 17];
    metadata_portion.read_exact(&mut bytes[..bytes_before_version])?;
    let length = bytes[bytes_before_version - 1];

    let mut version_string_utf8 = vec![0u8; length as usize];
    metadata_portion.read_exact(&mut version_string_utf8)?;
    let version_string = String::from_utf8(version_string_utf8);
    version_string.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[test]
fn test_version_check() {
    let info = read_dylib_info(
        &object::File::parse(&*std::fs::read(crate::proc_macro_test_dylib_path()).unwrap())
            .unwrap(),
    )
    .unwrap();

    assert_eq!(
        info.version_string,
        crate::RUSTC_VERSION_STRING,
        "sysroot ABI mismatch: dylib rustc version (read from .rustc section): {:?} != proc-macro-srv version (read from 'rustc --version'): {:?}",
        info.version_string,
        crate::RUSTC_VERSION_STRING,
    );
}
