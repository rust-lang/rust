// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A helper class for dealing with static archives

use driver::session::Session;
use metadata::filesearch;

use std::io::fs;
use std::os;
use std::run::{ProcessOptions, Process, ProcessOutput};
use std::str;
use extra::tempfile::TempDir;
use syntax::abi;

pub struct Archive {
    priv sess: Session,
    priv dst: Path,
}

fn run_ar(sess: Session, args: &str, cwd: Option<&Path>,
        paths: &[&Path]) -> ProcessOutput {
    let ar = sess.opts.ar.clone().unwrap_or_else(|| ~"ar");
    let mut args = ~[args.to_owned()];
    let mut paths = paths.iter().map(|p| p.as_str().unwrap().to_owned());
    args.extend(&mut paths);
    let mut opts = ProcessOptions::new();
    opts.dir = cwd;
    debug!("{} {}", ar, args.connect(" "));
    match cwd {
        Some(p) => { debug!("inside {}", p.display()); }
        None => {}
    }
    let o = Process::new(ar, args.as_slice(), opts).finish_with_output();
    if !o.status.success() {
        sess.err(format!("{} failed with: {}", ar, o.status));
        sess.note(format!("stdout ---\n{}", str::from_utf8(o.output)));
        sess.note(format!("stderr ---\n{}", str::from_utf8(o.error)));
        sess.abort_if_errors();
    }
    o
}

impl Archive {
    /// Initializes a new static archive with the given object file
    pub fn create<'a>(sess: Session, dst: &'a Path,
                      initial_object: &'a Path) -> Archive {
        run_ar(sess, "crus", None, [dst, initial_object]);
        Archive { sess: sess, dst: dst.clone() }
    }

    /// Opens an existing static archive
    pub fn open(sess: Session, dst: Path) -> Archive {
        assert!(dst.exists());
        Archive { sess: sess, dst: dst }
    }

    /// Read a file in the archive
    pub fn read(&self, file: &str) -> ~[u8] {
        // Apparently if "ar p" is used on windows, it generates a corrupt file
        // which has bad headers and LLVM will immediately choke on it
        if cfg!(windows) && cfg!(windows) { // FIXME(#10734) double-and
            let loc = TempDir::new("rsar").unwrap();
            let archive = os::make_absolute(&self.dst);
            run_ar(self.sess, "x", Some(loc.path()), [&archive,
                                                      &Path::new(file)]);
            fs::File::open(&loc.path().join(file)).read_to_end()
        } else {
            run_ar(self.sess, "p", None, [&self.dst, &Path::new(file)]).output
        }
    }

    /// Adds all of the contents of a native library to this archive. This will
    /// search in the relevant locations for a library named `name`.
    pub fn add_native_library(&mut self, name: &str) {
        let location = self.find_library(name);
        self.add_archive(&location, name);
    }

    /// Adds all of the contents of the rlib at the specified path to this
    /// archive.
    pub fn add_rlib(&mut self, rlib: &Path) {
        let name = rlib.filename_str().unwrap().split('-').next().unwrap();
        self.add_archive(rlib, name);
    }

    fn add_archive(&mut self, archive: &Path, name: &str) {
        let loc = TempDir::new("rsar").unwrap();

        // First, extract the contents of the archive to a temporary directory
        let archive = os::make_absolute(archive);
        run_ar(self.sess, "x", Some(loc.path()), [&archive]);

        // Next, we must rename all of the inputs to "guaranteed unique names".
        // The reason for this is that archives are keyed off the name of the
        // files, so if two files have the same name they will override one
        // another in the archive (bad).
        let files = fs::readdir(loc.path());
        let mut inputs = ~[];
        for file in files.iter() {
            let filename = file.filename_str().unwrap();
            let filename = format!("r-{}-{}", name, filename);
            let new_filename = file.with_filename(filename);
            fs::rename(file, &new_filename);
            inputs.push(new_filename);
        }

        // Finally, add all the renamed files to this archive
        let mut args = ~[&self.dst];
        args.extend(&mut inputs.iter());
        run_ar(self.sess, "r", None, args.as_slice());
    }

    fn find_library(&self, name: &str) -> Path {
        let (osprefix, osext) = match self.sess.targ_cfg.os {
            abi::OsWin32 => ("", "lib"), _ => ("lib", "a"),
        };
        // On windows, static libraries sometimes show up as libfoo.a and other
        // times show up as foo.lib
        let oslibname = format!("{}{}.{}", osprefix, name, osext);
        let unixlibname = format!("lib{}.a", name);

        let mut rustpath = filesearch::rust_path();
        rustpath.push(self.sess.filesearch.get_target_lib_path());
        let path = self.sess.opts.addl_lib_search_paths.iter();
        for path in path.chain(rustpath.iter()) {
            debug!("looking for {} inside {}", name, path.display());
            let test = path.join(oslibname.as_slice());
            if test.exists() { return test }
            if oslibname != unixlibname {
                let test = path.join(unixlibname.as_slice());
                if test.exists() { return test }
            }
        }
        self.sess.fatal(format!("could not find native static library `{}`, \
                                 perhaps an -L flag is missing?", name));
    }
}
