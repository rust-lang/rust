// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A helper class for dealing with static archives

use std::io::process::{Command, ProcessOutput};
use std::io::{fs, TempDir};
use std::io;
use std::os;
use std::str;
use syntax::abi;
use ErrorHandler = syntax::diagnostic::Handler;

pub static METADATA_FILENAME: &'static str = "rust.metadata.bin";

pub struct ArchiveConfig<'a> {
    pub handler: &'a ErrorHandler,
    pub dst: Path,
    pub lib_search_paths: Vec<Path>,
    pub os: abi::Os,
    pub maybe_ar_prog: Option<String>
}

pub struct Archive<'a> {
    handler: &'a ErrorHandler,
    dst: Path,
    lib_search_paths: Vec<Path>,
    os: abi::Os,
    maybe_ar_prog: Option<String>
}

fn run_ar(handler: &ErrorHandler, maybe_ar_prog: &Option<String>,
          args: &str, cwd: Option<&Path>,
          paths: &[&Path]) -> ProcessOutput {
    let ar = match *maybe_ar_prog {
        Some(ref ar) => ar.as_slice(),
        None => "ar"
    };
    let mut cmd = Command::new(ar);

    cmd.arg(args).args(paths);
    debug!("{}", cmd);

    match cwd {
        Some(p) => {
            cmd.cwd(p);
            debug!("inside {}", p.display());
        }
        None => {}
    }

    match cmd.spawn() {
        Ok(prog) => {
            let o = prog.wait_with_output().unwrap();
            if !o.status.success() {
                handler.err(format!("{} failed with: {}",
                                 cmd,
                                 o.status).as_slice());
                handler.note(format!("stdout ---\n{}",
                                  str::from_utf8(o.output
                                                  .as_slice()).unwrap())
                          .as_slice());
                handler.note(format!("stderr ---\n{}",
                                  str::from_utf8(o.error
                                                  .as_slice()).unwrap())
                          .as_slice());
                handler.abort_if_errors();
            }
            o
        },
        Err(e) => {
            handler.err(format!("could not exec `{}`: {}", ar.as_slice(),
                             e).as_slice());
            handler.abort_if_errors();
            fail!("rustc::back::archive::run_ar() should not reach this point");
        }
    }
}

impl<'a> Archive<'a> {
    /// Initializes a new static archive with the given object file
    pub fn create<'b>(config: ArchiveConfig<'a>, initial_object: &'b Path) -> Archive<'a> {
        let ArchiveConfig { handler, dst, lib_search_paths, os, maybe_ar_prog } = config;
        run_ar(handler, &maybe_ar_prog, "crus", None, [&dst, initial_object]);
        Archive {
            handler: handler,
            dst: dst,
            lib_search_paths: lib_search_paths,
            os: os,
            maybe_ar_prog: maybe_ar_prog
        }
    }

    /// Opens an existing static archive
    pub fn open(config: ArchiveConfig<'a>) -> Archive<'a> {
        let ArchiveConfig { handler, dst, lib_search_paths, os, maybe_ar_prog } = config;
        assert!(dst.exists());
        Archive {
            handler: handler,
            dst: dst,
            lib_search_paths: lib_search_paths,
            os: os,
            maybe_ar_prog: maybe_ar_prog
        }
    }

    /// Adds all of the contents of a native library to this archive. This will
    /// search in the relevant locations for a library named `name`.
    pub fn add_native_library(&mut self, name: &str) -> io::IoResult<()> {
        let location = self.find_library(name);
        self.add_archive(&location, name, [])
    }

    /// Adds all of the contents of the rlib at the specified path to this
    /// archive.
    ///
    /// This ignores adding the bytecode from the rlib, and if LTO is enabled
    /// then the object file also isn't added.
    pub fn add_rlib(&mut self, rlib: &Path, name: &str,
                    lto: bool) -> io::IoResult<()> {
        let object = format!("{}.o", name);
        let bytecode = format!("{}.bytecode.deflate", name);
        let mut ignore = vec!(bytecode.as_slice(), METADATA_FILENAME);
        if lto {
            ignore.push(object.as_slice());
        }
        self.add_archive(rlib, name, ignore.as_slice())
    }

    /// Adds an arbitrary file to this archive
    pub fn add_file(&mut self, file: &Path, has_symbols: bool) {
        let cmd = if has_symbols {"r"} else {"rS"};
        run_ar(self.handler, &self.maybe_ar_prog, cmd, None, [&self.dst, file]);
    }

    /// Removes a file from this archive
    pub fn remove_file(&mut self, file: &str) {
        run_ar(self.handler, &self.maybe_ar_prog, "d", None, [&self.dst, &Path::new(file)]);
    }

    /// Updates all symbols in the archive (runs 'ar s' over it)
    pub fn update_symbols(&mut self) {
        run_ar(self.handler, &self.maybe_ar_prog, "s", None, [&self.dst]);
    }

    /// Lists all files in an archive
    pub fn files(&self) -> Vec<String> {
        let output = run_ar(self.handler, &self.maybe_ar_prog, "t", None, [&self.dst]);
        let output = str::from_utf8(output.output.as_slice()).unwrap();
        // use lines_any because windows delimits output with `\r\n` instead of
        // just `\n`
        output.lines_any().map(|s| s.to_string()).collect()
    }

    fn add_archive(&mut self, archive: &Path, name: &str,
                   skip: &[&str]) -> io::IoResult<()> {
        let loc = TempDir::new("rsar").unwrap();

        // First, extract the contents of the archive to a temporary directory
        let archive = os::make_absolute(archive);
        run_ar(self.handler, &self.maybe_ar_prog, "x", Some(loc.path()), [&archive]);

        // Next, we must rename all of the inputs to "guaranteed unique names".
        // The reason for this is that archives are keyed off the name of the
        // files, so if two files have the same name they will override one
        // another in the archive (bad).
        //
        // We skip any files explicitly desired for skipping, and we also skip
        // all SYMDEF files as these are just magical placeholders which get
        // re-created when we make a new archive anyway.
        let files = try!(fs::readdir(loc.path()));
        let mut inputs = Vec::new();
        for file in files.iter() {
            let filename = file.filename_str().unwrap();
            if skip.iter().any(|s| *s == filename) { continue }
            if filename.contains(".SYMDEF") { continue }

            let filename = format!("r-{}-{}", name, filename);
            // LLDB (as mentioned in back::link) crashes on filenames of exactly
            // 16 bytes in length. If we're including an object file with
            // exactly 16-bytes of characters, give it some prefix so that it's
            // not 16 bytes.
            let filename = if filename.len() == 16 {
                format!("lldb-fix-{}", filename)
            } else {
                filename
            };
            let new_filename = file.with_filename(filename);
            try!(fs::rename(file, &new_filename));
            inputs.push(new_filename);
        }
        if inputs.len() == 0 { return Ok(()) }

        // Finally, add all the renamed files to this archive
        let mut args = vec!(&self.dst);
        args.extend(inputs.iter());
        run_ar(self.handler, &self.maybe_ar_prog, "r", None, args.as_slice());
        Ok(())
    }

    fn find_library(&self, name: &str) -> Path {
        let (osprefix, osext) = match self.os {
            abi::OsWin32 => ("", "lib"), _ => ("lib", "a"),
        };
        // On Windows, static libraries sometimes show up as libfoo.a and other
        // times show up as foo.lib
        let oslibname = format!("{}{}.{}", osprefix, name, osext);
        let unixlibname = format!("lib{}.a", name);

        for path in self.lib_search_paths.iter() {
            debug!("looking for {} inside {}", name, path.display());
            let test = path.join(oslibname.as_slice());
            if test.exists() { return test }
            if oslibname != unixlibname {
                let test = path.join(unixlibname.as_slice());
                if test.exists() { return test }
            }
        }
        self.handler.fatal(format!("could not find native static library `{}`, \
                                 perhaps an -L flag is missing?",
                                name).as_slice());
    }
}

