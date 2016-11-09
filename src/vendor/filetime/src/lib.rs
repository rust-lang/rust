//! Timestamps for files in Rust
//!
//! This library provides platform-agnostic inspection of the various timestamps
//! present in the standard `fs::Metadata` structure.
//!
//! # Installation
//!
//! Add this to you `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! filetime = "0.1"
//! ```
//!
//! # Usage
//!
//! ```no_run
//! use std::fs;
//! use filetime::FileTime;
//!
//! let metadata = fs::metadata("foo.txt").unwrap();
//!
//! let mtime = FileTime::from_last_modification_time(&metadata);
//! println!("{}", mtime);
//!
//! let atime = FileTime::from_last_access_time(&metadata);
//! assert!(mtime < atime);
//!
//! // Inspect values that can be interpreted across platforms
//! println!("{}", mtime.seconds_relative_to_1970());
//! println!("{}", mtime.nanoseconds());
//!
//! // Print the platform-specific value of seconds
//! println!("{}", mtime.seconds());
//! ```

extern crate libc;

#[cfg(unix)] use std::os::unix::prelude::*;
#[cfg(windows)] use std::os::windows::prelude::*;

use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

/// A helper structure to represent a timestamp for a file.
///
/// The actual value contined within is platform-specific and does not have the
/// same meaning across platforms, but comparisons and stringification can be
/// significant among the same platform.
#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Copy, Clone, Hash)]
pub struct FileTime {
    seconds: u64,
    nanos: u32,
}

impl FileTime {
    /// Creates a new timestamp representing a 0 time.
    ///
    /// Useful for creating the base of a cmp::max chain of times.
    pub fn zero() -> FileTime {
        FileTime { seconds: 0, nanos: 0 }
    }

    /// Creates a new instance of `FileTime` with a number of seconds and
    /// nanoseconds relative to January 1, 1970.
    ///
    /// Note that this is typically the relative point that Unix time stamps are
    /// from, but on Windows the native time stamp is relative to January 1,
    /// 1601 so the return value of `seconds` from the returned `FileTime`
    /// instance may not be the same as that passed in.
    pub fn from_seconds_since_1970(seconds: u64, nanos: u32) -> FileTime {
        FileTime {
            seconds: seconds + if cfg!(windows) {11644473600} else {0},
            nanos: nanos,
        }
    }

    /// Creates a new timestamp from the last modification time listed in the
    /// specified metadata.
    ///
    /// The returned value corresponds to the `mtime` field of `stat` on Unix
    /// platforms and the `ftLastWriteTime` field on Windows platforms.
    pub fn from_last_modification_time(meta: &fs::Metadata) -> FileTime {
        #[cfg(unix)]
        fn imp(meta: &fs::Metadata) -> FileTime {
            FileTime::from_os_repr(meta.mtime() as u64, meta.mtime_nsec() as u32)
        }
        #[cfg(windows)]
        fn imp(meta: &fs::Metadata) -> FileTime {
            FileTime::from_os_repr(meta.last_write_time())
        }
        imp(meta)
    }

    /// Creates a new timestamp from the last access time listed in the
    /// specified metadata.
    ///
    /// The returned value corresponds to the `atime` field of `stat` on Unix
    /// platforms and the `ftLastAccessTime` field on Windows platforms.
    pub fn from_last_access_time(meta: &fs::Metadata) -> FileTime {
        #[cfg(unix)]
        fn imp(meta: &fs::Metadata) -> FileTime {
            FileTime::from_os_repr(meta.atime() as u64, meta.atime_nsec() as u32)
        }
        #[cfg(windows)]
        fn imp(meta: &fs::Metadata) -> FileTime {
            FileTime::from_os_repr(meta.last_access_time())
        }
        imp(meta)
    }

    /// Creates a new timestamp from the creation time listed in the specified
    /// metadata.
    ///
    /// The returned value corresponds to the `birthtime` field of `stat` on
    /// Unix platforms and the `ftCreationTime` field on Windows platforms. Note
    /// that not all Unix platforms have this field available and may return
    /// `None` in some circumstances.
    pub fn from_creation_time(meta: &fs::Metadata) -> Option<FileTime> {
        macro_rules! birthtim {
            ($(($e:expr, $i:ident)),*) => {
                #[cfg(any($(target_os = $e),*))]
                fn imp(meta: &fs::Metadata) -> Option<FileTime> {
                    $(
                        #[cfg(target_os = $e)]
                        use std::os::$i::fs::MetadataExt;
                    )*
                    let raw = meta.as_raw_stat();
                    Some(FileTime::from_os_repr(raw.st_birthtime as u64,
                                                raw.st_birthtime_nsec as u32))
                }

                #[cfg(all(not(windows),
                          $(not(target_os = $e)),*))]
                fn imp(_meta: &fs::Metadata) -> Option<FileTime> {
                    None
                }
            }
        }

        birthtim! {
            ("bitrig", bitrig),
            ("freebsd", freebsd),
            ("ios", ios),
            ("macos", macos),
            ("openbsd", openbsd)
        }

        #[cfg(windows)]
        fn imp(meta: &fs::Metadata) -> Option<FileTime> {
            Some(FileTime::from_os_repr(meta.last_access_time()))
        }
        imp(meta)
    }

    #[cfg(windows)]
    fn from_os_repr(time: u64) -> FileTime {
        // Windows write times are in 100ns intervals, so do a little math to
        // get it into the right representation.
        FileTime {
            seconds: time / (1_000_000_000 / 100),
            nanos: ((time % (1_000_000_000 / 100)) * 100) as u32,
        }
    }

    #[cfg(unix)]
    fn from_os_repr(seconds: u64, nanos: u32) -> FileTime {
        FileTime { seconds: seconds, nanos: nanos }
    }

    /// Returns the whole number of seconds represented by this timestamp.
    ///
    /// Note that this value's meaning is **platform specific**. On Unix
    /// platform time stamps are typically relative to January 1, 1970, but on
    /// Windows platforms time stamps are relative to January 1, 1601.
    pub fn seconds(&self) -> u64 { self.seconds }

    /// Returns the whole number of seconds represented by this timestamp,
    /// relative to the Unix epoch start of January 1, 1970.
    ///
    /// Note that this does not return the same value as `seconds` for Windows
    /// platforms as seconds are relative to a different date there.
    pub fn seconds_relative_to_1970(&self) -> u64 {
        self.seconds - if cfg!(windows) {11644473600} else {0}
    }

    /// Returns the nanosecond precision of this timestamp.
    ///
    /// The returned value is always less than one billion and represents a
    /// portion of a second forward from the seconds returned by the `seconds`
    /// method.
    pub fn nanoseconds(&self) -> u32 { self.nanos }
}

impl fmt::Display for FileTime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{:09}s", self.seconds, self.nanos)
    }
}

/// Set the last access and modification times for a file on the filesystem.
///
/// This function will set the `atime` and `mtime` metadata fields for a file
/// on the local filesystem, returning any error encountered.
pub fn set_file_times<P>(p: P, atime: FileTime, mtime: FileTime)
                         -> io::Result<()> where P: AsRef<Path> {
    set_file_times_(p.as_ref(), atime, mtime)
}

#[cfg(unix)]
fn set_file_times_(p: &Path, atime: FileTime, mtime: FileTime) -> io::Result<()> {
    use std::ffi::CString;
    use libc::{timeval, time_t, suseconds_t, utimes};

    let times = [to_timeval(&atime), to_timeval(&mtime)];
    let p = try!(CString::new(p.as_os_str().as_bytes()));
    return unsafe {
        if utimes(p.as_ptr() as *const _, times.as_ptr()) == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    };

    fn to_timeval(ft: &FileTime) -> timeval {
        timeval {
            tv_sec: ft.seconds() as time_t,
            tv_usec: (ft.nanoseconds() / 1000) as suseconds_t,
        }
    }
}

#[cfg(windows)]
#[allow(bad_style)]
fn set_file_times_(p: &Path, atime: FileTime, mtime: FileTime) -> io::Result<()> {
    use std::fs::OpenOptions;

    type BOOL = i32;
    type HANDLE = *mut u8;
    type DWORD = u32;
    #[repr(C)]
    struct FILETIME {
        dwLowDateTime: u32,
        dwHighDateTime: u32,
    }
    extern "system" {
        fn SetFileTime(hFile: HANDLE,
                       lpCreationTime: *const FILETIME,
                       lpLastAccessTime: *const FILETIME,
                       lpLastWriteTime: *const FILETIME) -> BOOL;
    }

    let f = try!(OpenOptions::new().write(true).open(p));
    let atime = to_filetime(&atime);
    let mtime = to_filetime(&mtime);
    return unsafe {
        let ret = SetFileTime(f.as_raw_handle() as *mut _,
                              0 as *const _,
                              &atime, &mtime);
        if ret != 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    };

    fn to_filetime(ft: &FileTime) -> FILETIME {
        let intervals = ft.seconds() * (1_000_000_000 / 100) +
                        ((ft.nanoseconds() as u64) / 100);
        FILETIME {
            dwLowDateTime: intervals as DWORD,
            dwHighDateTime: (intervals >> 32) as DWORD,
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate tempdir;

    use std::fs::{self, File};
    use self::tempdir::TempDir;
    use super::{FileTime, set_file_times};

    #[test]
    fn set_file_times_test() {
        let td = TempDir::new("filetime").unwrap();
        let path = td.path().join("foo.txt");
        File::create(&path).unwrap();

        let metadata = fs::metadata(&path).unwrap();
        let mtime = FileTime::from_last_modification_time(&metadata);
        let atime = FileTime::from_last_access_time(&metadata);
        set_file_times(&path, atime, mtime).unwrap();

        let new_mtime = FileTime::from_seconds_since_1970(10_000, 0);
        set_file_times(&path, atime, new_mtime).unwrap();

        let metadata = fs::metadata(&path).unwrap();
        let mtime = FileTime::from_last_modification_time(&metadata);
        assert_eq!(mtime, new_mtime);
    }
}
