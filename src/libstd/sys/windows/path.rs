use crate::ffi::OsStr;
use crate::mem;
use crate::path::Prefix;

#[cfg(test)]
mod tests;

pub const MAIN_SEP_STR: &str = "\\";
pub const MAIN_SEP: char = '\\';

// The unsafety here stems from converting between `&OsStr` and `&[u8]`
// and back. This is safe to do because (1) we only look at ASCII
// contents of the encoding and (2) new &OsStr values are produced
// only from ASCII-bounded slices of existing &OsStr values.
fn os_str_as_u8_slice(s: &OsStr) -> &[u8] {
    unsafe { mem::transmute(s) }
}
unsafe fn u8_slice_as_os_str(s: &[u8]) -> &OsStr {
    mem::transmute(s)
}

#[inline]
pub fn is_sep_byte(b: u8) -> bool {
    b == b'/' || b == b'\\'
}

#[inline]
pub fn is_verbatim_sep(b: u8) -> bool {
    b == b'\\'
}

// In most DOS systems, it is not possible to have more than 26 drive letters.
// See <https://en.wikipedia.org/wiki/Drive_letter_assignment#Common_assignments>.
pub fn is_valid_drive_letter(disk: u8) -> bool {
    disk.is_ascii_alphabetic()
}

pub fn parse_prefix(path: &OsStr) -> Option<Prefix<'_>> {
    use Prefix::{DeviceNS, Disk, Verbatim, VerbatimDisk, VerbatimUNC, UNC};

    let path = os_str_as_u8_slice(path);

    // \\
    if let Some(path) = path.strip_prefix(br"\\") {
        // \\?\
        if let Some(path) = path.strip_prefix(br"?\") {
            // \\?\UNC\server\share
            if let Some(path) = path.strip_prefix(br"UNC\") {
                let (server, share) = match get_first_two_components(path, is_verbatim_sep) {
                    Some((server, share)) => unsafe {
                        (u8_slice_as_os_str(server), u8_slice_as_os_str(share))
                    },
                    None => (unsafe { u8_slice_as_os_str(path) }, OsStr::new("")),
                };
                return Some(VerbatimUNC(server, share));
            } else {
                // \\?\path
                match path {
                    // \\?\C:\path
                    [c, b':', b'\\', ..] if is_valid_drive_letter(*c) => {
                        return Some(VerbatimDisk(c.to_ascii_uppercase()));
                    }
                    // \\?\cat_pics
                    _ => {
                        let idx = path.iter().position(|&b| b == b'\\').unwrap_or(path.len());
                        let slice = &path[..idx];
                        return Some(Verbatim(unsafe { u8_slice_as_os_str(slice) }));
                    }
                }
            }
        } else if let Some(path) = path.strip_prefix(b".\\") {
            // \\.\COM42
            let idx = path.iter().position(|&b| b == b'\\').unwrap_or(path.len());
            let slice = &path[..idx];
            return Some(DeviceNS(unsafe { u8_slice_as_os_str(slice) }));
        }
        match get_first_two_components(path, is_sep_byte) {
            Some((server, share)) if !server.is_empty() && !share.is_empty() => {
                // \\server\share
                return Some(unsafe { UNC(u8_slice_as_os_str(server), u8_slice_as_os_str(share)) });
            }
            _ => {}
        }
    } else if let [c, b':', ..] = path {
        // C:
        if is_valid_drive_letter(*c) {
            return Some(Disk(c.to_ascii_uppercase()));
        }
    }
    None
}

/// Returns the first two path components with predicate `f`.
///
/// The two components returned will be use by caller
/// to construct `VerbatimUNC` or `UNC` Windows path prefix.
///
/// Returns [`None`] if there are no separators in path.
fn get_first_two_components(path: &[u8], f: fn(u8) -> bool) -> Option<(&[u8], &[u8])> {
    let idx = path.iter().position(|&x| f(x))?;
    // Panic safe
    // The max `idx+1` is `path.len()` and `path[path.len()..]` is a valid index.
    let (first, path) = (&path[..idx], &path[idx + 1..]);
    let idx = path.iter().position(|&x| f(x)).unwrap_or(path.len());
    let second = &path[..idx];
    Some((first, second))
}
