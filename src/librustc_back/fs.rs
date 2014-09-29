// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;
use std::io::fs;
use std::os;

/// Returns an absolute path in the filesystem that `path` points to. The
/// returned path does not contain any symlinks in its hierarchy.
pub fn realpath(original: &Path) -> io::IoResult<Path> {
    static MAX_LINKS_FOLLOWED: uint = 256;
    let original = os::make_absolute(original);

    // Right now lstat on windows doesn't work quite well
    if cfg!(windows) {
        return Ok(original)
    }

    let result = original.root_path();
    let mut result = result.expect("make_absolute has no root_path");
    let mut followed = 0;

    for part in original.components() {
        result.push(part);

        loop {
            if followed == MAX_LINKS_FOLLOWED {
                return Err(io::standard_error(io::InvalidInput))
            }

            match fs::lstat(&result) {
                Err(..) => break,
                Ok(ref stat) if stat.kind != io::TypeSymlink => break,
                Ok(..) => {
                    followed += 1;
                    let path = try!(fs::readlink(&result));
                    result.pop();
                    result.push(path);
                }
            }
        }
    }

    return Ok(result);
}

#[cfg(all(not(windows), test))]
mod test {
    use std::io;
    use std::io::fs::{File, symlink, mkdir, mkdir_recursive};
    use super::realpath;
    use std::io::TempDir;

    #[test]
    fn realpath_works() {
        let tmpdir = TempDir::new("rustc-fs").unwrap();
        let tmpdir = realpath(tmpdir.path()).unwrap();
        let file = tmpdir.join("test");
        let dir = tmpdir.join("test2");
        let link = dir.join("link");
        let linkdir = tmpdir.join("test3");

        File::create(&file).unwrap();
        mkdir(&dir, io::UserRWX).unwrap();
        symlink(&file, &link).unwrap();
        symlink(&dir, &linkdir).unwrap();

        assert!(realpath(&tmpdir).unwrap() == tmpdir);
        assert!(realpath(&file).unwrap() == file);
        assert!(realpath(&link).unwrap() == file);
        assert!(realpath(&linkdir).unwrap() == dir);
        assert!(realpath(&linkdir.join("link")).unwrap() == file);
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

        mkdir_recursive(&b, io::UserRWX).unwrap();
        mkdir_recursive(&d, io::UserRWX).unwrap();
        File::create(&f).unwrap();
        symlink(&Path::new("../d/e"), &c).unwrap();
        symlink(&Path::new("../f"), &e).unwrap();

        assert!(realpath(&c).unwrap() == f);
        assert!(realpath(&e).unwrap() == f);
    }
}
