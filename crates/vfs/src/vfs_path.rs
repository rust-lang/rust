//! Abstract-ish representation of paths for VFS.
use std::fmt;

use paths::{AbsPath, AbsPathBuf};

/// Long-term, we want to support files which do not reside in the file-system,
/// so we treat VfsPaths as opaque identifiers.
#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct VfsPath(VfsPathRepr);

impl VfsPath {
    /// Creates an "in-memory" path from `/`-separates string.
    /// This is most useful for testing, to avoid windows/linux differences
    pub fn new_virtual_path(path: String) -> VfsPath {
        assert!(path.starts_with('/'));
        VfsPath(VfsPathRepr::VirtualPath(VirtualPath(path)))
    }

    pub fn as_path(&self) -> Option<&AbsPath> {
        match &self.0 {
            VfsPathRepr::PathBuf(it) => Some(it.as_path()),
            VfsPathRepr::VirtualPath(_) => None,
        }
    }
    pub fn join(&self, path: &str) -> Option<VfsPath> {
        match &self.0 {
            VfsPathRepr::PathBuf(it) => {
                let res = it.join(path).normalize();
                Some(VfsPath(VfsPathRepr::PathBuf(res)))
            }
            VfsPathRepr::VirtualPath(it) => {
                let res = it.join(path)?;
                Some(VfsPath(VfsPathRepr::VirtualPath(res)))
            }
        }
    }
    pub fn pop(&mut self) -> bool {
        match &mut self.0 {
            VfsPathRepr::PathBuf(it) => it.pop(),
            VfsPathRepr::VirtualPath(it) => it.pop(),
        }
    }
    pub fn starts_with(&self, other: &VfsPath) -> bool {
        match (&self.0, &other.0) {
            (VfsPathRepr::PathBuf(lhs), VfsPathRepr::PathBuf(rhs)) => lhs.starts_with(rhs),
            (VfsPathRepr::PathBuf(_), _) => false,
            (VfsPathRepr::VirtualPath(lhs), VfsPathRepr::VirtualPath(rhs)) => lhs.starts_with(rhs),
            (VfsPathRepr::VirtualPath(_), _) => false,
        }
    }

    // Don't make this `pub`
    pub(crate) fn encode(&self, buf: &mut Vec<u8>) {
        let tag = match &self.0 {
            VfsPathRepr::PathBuf(_) => 0,
            VfsPathRepr::VirtualPath(_) => 1,
        };
        buf.push(tag);
        match &self.0 {
            VfsPathRepr::PathBuf(path) => {
                #[cfg(windows)]
                {
                    use windows_paths::Encode;
                    let components = path.components();
                    let mut add_sep = false;
                    for component in components {
                        if add_sep {
                            windows_paths::SEP.encode(buf);
                        }
                        let len_before = buf.len();
                        match component {
                            std::path::Component::Prefix(prefix) => {
                                // kind() returns a normalized and comparable path prefix.
                                prefix.kind().encode(buf);
                            }
                            std::path::Component::RootDir => {
                                if !add_sep {
                                    component.as_os_str().encode(buf);
                                }
                            }
                            _ => component.as_os_str().encode(buf),
                        }

                        // some components may be encoded empty
                        add_sep = len_before != buf.len();
                    }
                }
                #[cfg(unix)]
                {
                    use std::os::unix::ffi::OsStrExt;
                    buf.extend(path.as_os_str().as_bytes());
                }
                #[cfg(not(any(windows, unix)))]
                {
                    buf.extend(path.as_os_str().to_string_lossy().as_bytes());
                }
            }
            VfsPathRepr::VirtualPath(VirtualPath(s)) => buf.extend(s.as_bytes()),
        }
    }
}

#[cfg(windows)]
mod windows_paths {
    pub trait Encode {
        fn encode(&self, buf: &mut Vec<u8>);
    }

    impl Encode for std::ffi::OsStr {
        fn encode(&self, buf: &mut Vec<u8>) {
            use std::os::windows::ffi::OsStrExt;
            for wchar in self.encode_wide() {
                buf.extend(wchar.to_le_bytes().iter().copied());
            }
        }
    }

    impl Encode for u8 {
        fn encode(&self, buf: &mut Vec<u8>) {
            let wide = *self as u16;
            buf.extend(wide.to_le_bytes().iter().copied())
        }
    }

    impl Encode for &str {
        fn encode(&self, buf: &mut Vec<u8>) {
            debug_assert!(self.is_ascii());
            for b in self.as_bytes() {
                b.encode(buf)
            }
        }
    }

    pub const SEP: &str = "\\";
    const VERBATIM: &str = "\\\\?\\";
    const UNC: &str = "UNC";
    const DEVICE: &str = "\\\\.\\";
    const COLON: &str = ":";

    impl Encode for std::path::Prefix<'_> {
        fn encode(&self, buf: &mut Vec<u8>) {
            match self {
                std::path::Prefix::Verbatim(c) => {
                    VERBATIM.encode(buf);
                    c.encode(buf);
                }
                std::path::Prefix::VerbatimUNC(server, share) => {
                    VERBATIM.encode(buf);
                    UNC.encode(buf);
                    SEP.encode(buf);
                    server.encode(buf);
                    SEP.encode(buf);
                    share.encode(buf);
                }
                std::path::Prefix::VerbatimDisk(d) => {
                    VERBATIM.encode(buf);
                    d.encode(buf);
                    COLON.encode(buf);
                }
                std::path::Prefix::DeviceNS(device) => {
                    DEVICE.encode(buf);
                    device.encode(buf);
                }
                std::path::Prefix::UNC(server, share) => {
                    SEP.encode(buf);
                    SEP.encode(buf);
                    server.encode(buf);
                    SEP.encode(buf);
                    share.encode(buf);
                }
                std::path::Prefix::Disk(d) => {
                    d.encode(buf);
                    COLON.encode(buf);
                }
            }
        }
    }
    #[test]
    fn paths_encoding() {
        // drive letter casing agnostic
        test_eq("C:/x.rs", "c:/x.rs");
        // separator agnostic
        test_eq("C:/x/y.rs", "C:\\x\\y.rs");

        fn test_eq(a: &str, b: &str) {
            let mut b1 = Vec::new();
            let mut b2 = Vec::new();
            vfs(a).encode(&mut b1);
            vfs(b).encode(&mut b2);
            assert_eq!(b1, b2);
        }
    }

    #[test]
    fn test_sep_root_dir_encoding() {
        let mut buf = Vec::new();
        vfs("C:/x/y").encode(&mut buf);
        assert_eq!(&buf, &[0, 67, 0, 58, 0, 92, 0, 120, 0, 92, 0, 121, 0])
    }

    #[cfg(test)]
    fn vfs(str: &str) -> super::VfsPath {
        use super::{AbsPathBuf, VfsPath};
        use std::convert::TryFrom;
        VfsPath::from(AbsPathBuf::try_from(str).unwrap())
    }
}

#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
enum VfsPathRepr {
    PathBuf(AbsPathBuf),
    VirtualPath(VirtualPath),
}

impl From<AbsPathBuf> for VfsPath {
    fn from(v: AbsPathBuf) -> Self {
        VfsPath(VfsPathRepr::PathBuf(v.normalize()))
    }
}

impl fmt::Display for VfsPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            VfsPathRepr::PathBuf(it) => fmt::Display::fmt(&it.display(), f),
            VfsPathRepr::VirtualPath(VirtualPath(it)) => fmt::Display::fmt(it, f),
        }
    }
}

impl fmt::Debug for VfsPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Debug for VfsPathRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            VfsPathRepr::PathBuf(it) => fmt::Debug::fmt(&it.display(), f),
            VfsPathRepr::VirtualPath(VirtualPath(it)) => fmt::Debug::fmt(&it, f),
        }
    }
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
struct VirtualPath(String);

impl VirtualPath {
    fn starts_with(&self, other: &VirtualPath) -> bool {
        self.0.starts_with(&other.0)
    }
    fn pop(&mut self) -> bool {
        let pos = match self.0.rfind('/') {
            Some(pos) => pos,
            None => return false,
        };
        self.0 = self.0[..pos].to_string();
        true
    }
    fn join(&self, mut path: &str) -> Option<VirtualPath> {
        let mut res = self.clone();
        while path.starts_with("../") {
            if !res.pop() {
                return None;
            }
            path = &path["../".len()..]
        }
        res.0 = format!("{}/{}", res.0, path);
        Some(res)
    }
}
