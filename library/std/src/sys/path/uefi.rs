#![forbid(unsafe_op_in_unsafe_fn)]
use crate::ffi::OsStr;
use crate::io;
use crate::path::{Path, PathBuf, Prefix};
use crate::sys::{helpers, unsupported_err};

const FORWARD_SLASH: u8 = b'/';
const COLON: u8 = b':';

#[inline]
pub fn is_sep_byte(b: u8) -> bool {
    b == b'\\'
}

#[inline]
pub fn is_verbatim_sep(b: u8) -> bool {
    b == b'\\'
}

pub fn parse_prefix(_: &OsStr) -> Option<Prefix<'_>> {
    None
}

pub const MAIN_SEP_STR: &str = "\\";
pub const MAIN_SEP: char = '\\';

/// UEFI paths can be of 4 types:
///
/// 1. Absolute Shell Path: Uses shell mappings (eg: `FS0:`). Does not exist if UEFI shell not present.
///    It can be identified with `:`.
///    Eg: FS0:\abc\run.efi
///
/// 2. Absolute Device Path: this is what we want
///    It can be identified with `/`.
///    Eg: PciRoot(0x0)/Pci(0x1,0x1)/Ata(Secondary,Slave,0x0)/\abc\run.efi
///
/// 3: Relative root: path relative to the current volume.
///    It will start with `\`.
///    Eg: \abc\run.efi
///
/// 4: Relative
///    Eg: run.efi
///
/// The algorithm is mostly taken from edk2 UEFI shell implementation and is
/// somewhat simple. Check for the path type in order.
///
/// The volume mapping in Absolute Shell Path (not the rest of the path) can be converted to Device
/// Path Protocol using `EFI_SHELL->GetDevicePathFromMap`. The rest of the path (Relative root
/// path), can just be appended to the remaining path.
///
/// For Relative root, we get the current volume (either in Shell Mapping, or Device Path Protocol
/// form) and join it with the relative root path. We then recurse the function to resolve the Shell
/// Mapping if present.
///
/// For Relative paths, we use the current working directory to construct
/// the new path and recurse the function to resolve the Shell mapping if present.
///
/// Finally, at the end, we get the 2nd form, i.e. Absolute Device Path, which can be used in the
/// normal UEFI APIs such as file, process, etc.
/// Eg: PciRoot(0x0)/Pci(0x1,0x1)/Ata(Secondary,Slave,0x0)/\abc\run.efi
pub(crate) fn absolute(path: &Path) -> io::Result<PathBuf> {
    // Absolute Shell Path
    if path.as_os_str().as_encoded_bytes().contains(&COLON) {
        let mut path_components = path.components();
        // Since path is not empty, it has at least one Component
        let prefix = path_components.next().unwrap();

        let dev_path = helpers::get_device_path_from_map(prefix.as_ref())?;
        let mut dev_path_text = dev_path.to_text().map_err(|_| unsupported_err())?;

        // UEFI Shell does not seem to end device path with `/`
        if *dev_path_text.as_encoded_bytes().last().unwrap() != FORWARD_SLASH {
            dev_path_text.push("/");
        }

        let mut ans = PathBuf::from(dev_path_text);
        ans.push(path_components);

        return Ok(ans);
    }

    // Absolute Device Path
    if path.as_os_str().as_encoded_bytes().contains(&FORWARD_SLASH) {
        return Ok(path.to_path_buf());
    }

    // cur_dir() always returns something
    let cur_dir = crate::env::current_dir().unwrap();
    let mut path_components = path.components();

    // Relative Root
    if path_components.next().unwrap() == crate::path::Component::RootDir {
        let mut ans = PathBuf::new();
        ans.push(cur_dir.components().next().unwrap());
        ans.push(path_components);
        return absolute(&ans);
    }

    absolute(&cur_dir.join(path))
}

pub(crate) fn is_absolute(path: &Path) -> bool {
    let temp = path.as_os_str().as_encoded_bytes();
    temp.contains(&COLON) || temp.contains(&FORWARD_SLASH)
}
