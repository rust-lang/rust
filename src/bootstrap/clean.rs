//! Implementation of `make clean` in rustbuild.
//!
//! Responsible for cleaning out a build directory of all old and stale
//! artifacts to prepare for a fresh build. Currently doesn't remove the
//! `build/cache` directory (download cache) or the `build/$target/llvm`
//! directory unless the `--all` flag is present.

use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

use build_helper::t;

use crate::Build;

pub fn clean(build: &Build, all: bool) {
    rm_rf("tmp".as_ref());

    if all {
        rm_rf(&build.out);
    } else {
        rm_rf(&build.out.join("tmp"));
        rm_rf(&build.out.join("dist"));
        rm_rf(&build.out.join("bootstrap"));

        for host in &build.hosts {
            let entries = match build.out.join(host.triple).read_dir() {
                Ok(iter) => iter,
                Err(_) => continue,
            };

            for entry in entries {
                let entry = t!(entry);
                if entry.file_name().to_str() == Some("llvm") {
                    continue;
                }
                let path = t!(entry.path().canonicalize());
                rm_rf(&path);
            }
        }
    }
}

fn rm_rf(path: &Path) {
    match path.symlink_metadata() {
        Err(e) => {
            if e.kind() == ErrorKind::NotFound {
                return;
            }
            panic!("failed to get metadata for file {}: {}", path.display(), e);
        }
        Ok(metadata) => {
            if metadata.file_type().is_file() || metadata.file_type().is_symlink() {
                do_op(path, "remove file", |p| fs::remove_file(p));
                return;
            }

            for file in t!(fs::read_dir(path)) {
                rm_rf(&t!(file).path());
            }
            do_op(path, "remove dir", |p| fs::remove_dir(p));
        }
    };
}

fn do_op<F>(path: &Path, desc: &str, mut f: F)
where
    F: FnMut(&Path) -> io::Result<()>,
{
    match f(path) {
        Ok(()) => {}
        // On windows we can't remove a readonly file, and git will often clone files as readonly.
        // As a result, we have some special logic to remove readonly files on windows.
        // This is also the reason that we can't use things like fs::remove_dir_all().
        Err(ref e) if cfg!(windows) && e.kind() == ErrorKind::PermissionDenied => {
            let mut p = t!(path.symlink_metadata()).permissions();
            p.set_readonly(false);
            t!(fs::set_permissions(path, p));
            f(path).unwrap_or_else(|e| {
                panic!("failed to {} {}: {}", desc, path.display(), e);
            })
        }
        Err(e) => {
            panic!("failed to {} {}: {}", desc, path.display(), e);
        }
    }
}
