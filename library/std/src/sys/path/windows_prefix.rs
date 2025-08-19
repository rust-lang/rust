//! Parse Windows prefixes, for both Windows and Cygwin.

use super::{is_sep_byte, is_verbatim_sep};
use crate::ffi::OsStr;
use crate::path::Prefix;

struct PrefixParser<'a, const LEN: usize> {
    path: &'a OsStr,
    prefix: [u8; LEN],
}

impl<'a, const LEN: usize> PrefixParser<'a, LEN> {
    #[inline]
    fn get_prefix(path: &OsStr) -> [u8; LEN] {
        let mut prefix = [0; LEN];
        // SAFETY: Only ASCII characters are modified.
        for (i, &ch) in path.as_encoded_bytes().iter().take(LEN).enumerate() {
            prefix[i] = if ch == b'/' { b'\\' } else { ch };
        }
        prefix
    }

    fn new(path: &'a OsStr) -> Self {
        Self { path, prefix: Self::get_prefix(path) }
    }

    fn as_slice(&self) -> PrefixParserSlice<'a, '_> {
        PrefixParserSlice {
            path: self.path,
            prefix: &self.prefix[..LEN.min(self.path.len())],
            index: 0,
        }
    }
}

struct PrefixParserSlice<'a, 'b> {
    path: &'a OsStr,
    prefix: &'b [u8],
    index: usize,
}

impl<'a> PrefixParserSlice<'a, '_> {
    fn strip_prefix(&self, prefix: &str) -> Option<Self> {
        self.prefix[self.index..]
            .starts_with(prefix.as_bytes())
            .then_some(Self { index: self.index + prefix.len(), ..*self })
    }

    fn prefix_bytes(&self) -> &'a [u8] {
        &self.path.as_encoded_bytes()[..self.index]
    }

    fn finish(self) -> &'a OsStr {
        // SAFETY: The unsafety here stems from converting between &OsStr and
        // &[u8] and back. This is safe to do because (1) we only look at ASCII
        // contents of the encoding and (2) new &OsStr values are produced only
        // from ASCII-bounded slices of existing &OsStr values.
        unsafe { OsStr::from_encoded_bytes_unchecked(&self.path.as_encoded_bytes()[self.index..]) }
    }
}

pub fn parse_prefix(path: &OsStr) -> Option<Prefix<'_>> {
    use Prefix::{DeviceNS, Disk, UNC, Verbatim, VerbatimDisk, VerbatimUNC};

    let parser = PrefixParser::<8>::new(path);
    let parser = parser.as_slice();
    if let Some(parser) = parser.strip_prefix(r"\\") {
        // \\

        // It's a POSIX path.
        if cfg!(target_os = "cygwin") && !path.as_encoded_bytes().iter().any(|&x| x == b'\\') {
            return None;
        }

        // The meaning of verbatim paths can change when they use a different
        // separator.
        if let Some(parser) = parser.strip_prefix(r"?\")
            // Cygwin allows `/` in verbatim paths.
            && (cfg!(target_os = "cygwin") || !parser.prefix_bytes().iter().any(|&x| x == b'/'))
        {
            // \\?\
            if let Some(parser) = parser.strip_prefix(r"UNC\") {
                // \\?\UNC\server\share

                let path = parser.finish();
                let (server, path) = parse_next_component(path, true);
                let (share, _) = parse_next_component(path, true);

                Some(VerbatimUNC(server, share))
            } else {
                let path = parser.finish();

                // in verbatim paths only recognize an exact drive prefix
                if let Some(drive) = parse_drive_exact(path) {
                    // \\?\C:
                    Some(VerbatimDisk(drive))
                } else {
                    // \\?\prefix
                    let (prefix, _) = parse_next_component(path, true);
                    Some(Verbatim(prefix))
                }
            }
        } else if let Some(parser) = parser.strip_prefix(r".\") {
            // \\.\COM42
            let path = parser.finish();
            let (prefix, _) = parse_next_component(path, false);
            Some(DeviceNS(prefix))
        } else {
            let path = parser.finish();
            let (server, path) = parse_next_component(path, false);
            let (share, _) = parse_next_component(path, false);

            if !server.is_empty() && !share.is_empty() {
                // \\server\share
                Some(UNC(server, share))
            } else {
                // no valid prefix beginning with "\\" recognized
                None
            }
        }
    } else {
        // If it has a drive like `C:` then it's a disk.
        // Otherwise there is no prefix.
        Some(Disk(parse_drive(path)?))
    }
}

// Parses a drive prefix, e.g. "C:" and "C:\whatever"
fn parse_drive(path: &OsStr) -> Option<u8> {
    // In most DOS systems, it is not possible to have more than 26 drive letters.
    // See <https://en.wikipedia.org/wiki/Drive_letter_assignment#Common_assignments>.
    fn is_valid_drive_letter(drive: &u8) -> bool {
        drive.is_ascii_alphabetic()
    }

    match path.as_encoded_bytes() {
        [drive, b':', ..] if is_valid_drive_letter(drive) => Some(drive.to_ascii_uppercase()),
        _ => None,
    }
}

// Parses a drive prefix exactly, e.g. "C:"
fn parse_drive_exact(path: &OsStr) -> Option<u8> {
    // only parse two bytes: the drive letter and the drive separator
    if path.as_encoded_bytes().get(2).map(|&x| is_sep_byte(x)).unwrap_or(true) {
        parse_drive(path)
    } else {
        None
    }
}

// Parse the next path component.
//
// Returns the next component and the rest of the path excluding the component and separator.
// Does not recognize `/` as a separator character on Windows if `verbatim` is true.
pub(crate) fn parse_next_component(path: &OsStr, verbatim: bool) -> (&OsStr, &OsStr) {
    let separator = if verbatim { is_verbatim_sep } else { is_sep_byte };

    match path.as_encoded_bytes().iter().position(|&x| separator(x)) {
        Some(separator_start) => {
            let separator_end = separator_start + 1;

            let component = &path.as_encoded_bytes()[..separator_start];

            // Panic safe
            // The max `separator_end` is `bytes.len()` and `bytes[bytes.len()..]` is a valid index.
            let path = &path.as_encoded_bytes()[separator_end..];

            // SAFETY: `path` is a valid wtf8 encoded slice and each of the separators ('/', '\')
            // is encoded in a single byte, therefore `bytes[separator_start]` and
            // `bytes[separator_end]` must be code point boundaries and thus
            // `bytes[..separator_start]` and `bytes[separator_end..]` are valid wtf8 slices.
            unsafe {
                (
                    OsStr::from_encoded_bytes_unchecked(component),
                    OsStr::from_encoded_bytes_unchecked(path),
                )
            }
        }
        None => (path, OsStr::new("")),
    }
}
