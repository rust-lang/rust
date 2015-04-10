// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use std::io;
use std::path::{Path, PathBuf};

#[cfg(windows)]
pub fn realpath(original: &Path) -> io::Result<PathBuf> {
    use std::fs::File;
    use std::ffi::OsString;
    use std::os::windows::prelude::*;

    extern "system" {
        fn GetFinalPathNameByHandleW(hFile: libc::HANDLE,
                                     lpszFilePath: libc::LPCWSTR,
                                     cchFilePath: libc::DWORD,
                                     dwFlags: libc::DWORD) -> libc::DWORD;
    }

    let mut v = Vec::with_capacity(16 * 1024);
    let f = try!(File::open(original));
    unsafe {
        let ret = GetFinalPathNameByHandleW(f.as_raw_handle(),
                                            v.as_mut_ptr(),
                                            v.capacity() as libc::DWORD,
                                            libc::VOLUME_NAME_DOS);
        if ret == 0 {
            return Err(io::Error::last_os_error())
        }
        assert!((ret as usize) < v.capacity());
        v.set_len(ret);
    }
    Ok(PathBuf::from(OsString::from_wide(&v)))
}

#[cfg(unix)]
pub fn realpath(original: &Path) -> io::Result<PathBuf> {
    use std::os::unix::prelude::*;
    use std::ffi::{OsString, CString};

    extern {
        fn realpath(pathname: *const libc::c_char, resolved: *mut libc::c_char)
                    -> *mut libc::c_char;
    }

    let path = try!(CString::new(original.as_os_str().as_bytes()));
    let mut buf = vec![0u8; 16 * 1024];
    unsafe {
        let r = realpath(path.as_ptr(), buf.as_mut_ptr() as *mut _);
        if r.is_null() {
            return Err(io::Error::last_os_error())
        }
    }
    let p = buf.iter().position(|i| *i == 0).unwrap();
    buf.truncate(p);
    Ok(PathBuf::from(OsString::from_vec(buf)))
}

#[cfg(all(not(windows), test))]
mod test {
    use tempdir::TempDir;
    use std::fs::{self, File};
    use super::realpath;

    #[test]
    fn realpath_works() {
        let tmpdir = TempDir::new("rustc-fs").unwrap();
        let tmpdir = realpath(tmpdir.path()).unwrap();
        let file = tmpdir.join("test");
        let dir = tmpdir.join("test2");
        let link = dir.join("link");
        let linkdir = tmpdir.join("test3");

        File::create(&file).unwrap();
        fs::create_dir(&dir).unwrap();
        fs::soft_link(&file, &link).unwrap();
        fs::soft_link(&dir, &linkdir).unwrap();

        assert_eq!(realpath(&tmpdir).unwrap(), tmpdir);
        assert_eq!(realpath(&file).unwrap(), file);
        assert_eq!(realpath(&link).unwrap(), file);
        assert_eq!(realpath(&linkdir).unwrap(), dir);
        assert_eq!(realpath(&linkdir.join("link")).unwrap(), file);
    }

    #[test]
    fn realpath_works_tricky() {
        let tmpdir = TempDir::new("rustc-fs").unwrap();
        let tmpdir = realpath(tmpdir.path()).unwrap();

        let a = tmpdir.join("a");
        let b = a.join("b");
        let c = b.join("c");
        let d = a.join("d");
        let e = d.join("e");
        let f = a.join("f");

        fs::create_dir_all(&b).unwrap();
        fs::create_dir_all(&d).unwrap();
        File::create(&f).unwrap();
        fs::soft_link("../d/e", &c).unwrap();
        fs::soft_link("../f", &e).unwrap();

        assert_eq!(realpath(&c).unwrap(), f);
        assert_eq!(realpath(&e).unwrap(), f);
    }
}
