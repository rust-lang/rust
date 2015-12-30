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

use std::env;
use std::ffi::{CString, CStr, OsString};
use std::fs::{self, File};
use std::io::prelude::*;
use std::io;
use std::mem;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::ptr;
use std::str;

use middle::cstore::CrateStore;

use libc;
use llvm::archive_ro::{ArchiveRO, Child};
use llvm::{self, ArchiveKind};
use rustc::session::Session;
use rustc_back::tempdir::TempDir;

pub struct ArchiveConfig<'a> {
    pub sess: &'a Session,
    pub dst: PathBuf,
    pub src: Option<PathBuf>,
    pub lib_search_paths: Vec<PathBuf>,
    pub ar_prog: String,
    pub command_path: OsString,
}

/// Helper for adding many files to an archive with a single invocation of
/// `ar`.
#[must_use = "must call build() to finish building the archive"]
pub struct ArchiveBuilder<'a> {
    config: ArchiveConfig<'a>,
    work_dir: TempDir,
    removals: Vec<String>,
    additions: Vec<Addition>,
    should_update_symbols: bool,
    src_archive: Option<Option<ArchiveRO>>,
}

enum Addition {
    File {
        path: PathBuf,
        name_in_archive: String,
    },
    Archive {
        archive: ArchiveRO,
        archive_name: String,
        skip: Box<FnMut(&str) -> bool>,
    },
}

enum Action<'a> {
    Remove(&'a [String]),
    AddObjects(&'a [&'a PathBuf], bool),
    UpdateSymbols,
}

pub fn find_library(name: &str, search_paths: &[PathBuf], sess: &Session)
                    -> PathBuf {
    // On Windows, static libraries sometimes show up as libfoo.a and other
    // times show up as foo.lib
    let oslibname = format!("{}{}{}",
                            sess.target.target.options.staticlib_prefix,
                            name,
                            sess.target.target.options.staticlib_suffix);
    let unixlibname = format!("lib{}.a", name);

    for path in search_paths {
        debug!("looking for {} inside {:?}", name, path);
        let test = path.join(&oslibname[..]);
        if test.exists() { return test }
        if oslibname != unixlibname {
            let test = path.join(&unixlibname[..]);
            if test.exists() { return test }
        }
    }
    sess.fatal(&format!("could not find native static library `{}`, \
                         perhaps an -L flag is missing?", name));
}

fn is_relevant_child(c: &Child) -> bool {
    match c.name() {
        Some(name) => !name.contains("SYMDEF"),
        None => false,
    }
}

impl<'a> ArchiveBuilder<'a> {
    /// Create a new static archive, ready for modifying the archive specified
    /// by `config`.
    pub fn new(config: ArchiveConfig<'a>) -> ArchiveBuilder<'a> {
        ArchiveBuilder {
            config: config,
            work_dir: TempDir::new("rsar").unwrap(),
            removals: Vec::new(),
            additions: Vec::new(),
            should_update_symbols: false,
            src_archive: None,
        }
    }

    /// Removes a file from this archive
    pub fn remove_file(&mut self, file: &str) {
        self.removals.push(file.to_string());
    }

    /// Lists all files in an archive
    pub fn src_files(&mut self) -> Vec<String> {
        if self.src_archive().is_none() {
            return Vec::new()
        }
        let archive = self.src_archive.as_ref().unwrap().as_ref().unwrap();
        let ret = archive.iter()
                         .filter(is_relevant_child)
                         .filter_map(|child| child.name())
                         .filter(|name| !self.removals.iter().any(|x| x == name))
                         .map(|name| name.to_string())
                         .collect();
        return ret;
    }

    fn src_archive(&mut self) -> Option<&ArchiveRO> {
        if let Some(ref a) = self.src_archive {
            return a.as_ref()
        }
        let src = match self.config.src {
            Some(ref src) => src,
            None => return None,
        };
        self.src_archive = Some(ArchiveRO::open(src));
        self.src_archive.as_ref().unwrap().as_ref()
    }

    /// Adds all of the contents of a native library to this archive. This will
    /// search in the relevant locations for a library named `name`.
    pub fn add_native_library(&mut self, name: &str) {
        let location = find_library(name, &self.config.lib_search_paths,
                                    self.config.sess);
        self.add_archive(&location, name, |_| false).unwrap_or_else(|e| {
            self.config.sess.fatal(&format!("failed to add native library {}: {}",
                                            location.to_string_lossy(), e));
        });
    }

    /// Adds all of the contents of the rlib at the specified path to this
    /// archive.
    ///
    /// This ignores adding the bytecode from the rlib, and if LTO is enabled
    /// then the object file also isn't added.
    pub fn add_rlib(&mut self, rlib: &Path, name: &str, lto: bool)
                    -> io::Result<()> {
        // Ignoring obj file starting with the crate name
        // as simple comparison is not enough - there
        // might be also an extra name suffix
        let obj_start = format!("{}", name);

        // Ignoring all bytecode files, no matter of
        // name
        let bc_ext = ".bytecode.deflate";
        let metadata_filename =
            self.config.sess.cstore.metadata_filename().to_owned();

        self.add_archive(rlib, &name[..], move |fname: &str| {
            let skip_obj = lto && fname.starts_with(&obj_start)
                && fname.ends_with(".o");
            skip_obj || fname.ends_with(bc_ext) || fname == metadata_filename
        })
    }

    fn add_archive<F>(&mut self, archive: &Path, name: &str, skip: F)
                      -> io::Result<()>
        where F: FnMut(&str) -> bool + 'static
    {
        let archive = match ArchiveRO::open(archive) {
            Some(ar) => ar,
            None => return Err(io::Error::new(io::ErrorKind::Other,
                                              "failed to open archive")),
        };
        self.additions.push(Addition::Archive {
            archive: archive,
            archive_name: name.to_string(),
            skip: Box::new(skip),
        });
        Ok(())
    }

    /// Adds an arbitrary file to this archive
    pub fn add_file(&mut self, file: &Path) {
        let name = file.file_name().unwrap().to_str().unwrap();
        self.additions.push(Addition::File {
            path: file.to_path_buf(),
            name_in_archive: name.to_string(),
        });
    }

    /// Indicate that the next call to `build` should updates all symbols in
    /// the archive (run 'ar s' over it).
    pub fn update_symbols(&mut self) {
        self.should_update_symbols = true;
    }

    /// Combine the provided files, rlibs, and native libraries into a single
    /// `Archive`.
    pub fn build(&mut self) {
        let res = match self.llvm_archive_kind() {
            Some(kind) => self.build_with_llvm(kind),
            None => self.build_with_ar_cmd(),
        };
        if let Err(e) = res {
            self.config.sess.fatal(&format!("failed to build archive: {}", e));
        }
    }

    pub fn llvm_archive_kind(&self) -> Option<ArchiveKind> {
        if unsafe { llvm::LLVMVersionMinor() < 7 } {
            return None
        }

        // Currently LLVM only supports writing archives in the 'gnu' format.
        match &self.config.sess.target.target.options.archive_format[..] {
            "gnu" => Some(ArchiveKind::K_GNU),
            "mips64" => Some(ArchiveKind::K_MIPS64),
            "bsd" => Some(ArchiveKind::K_BSD),
            "coff" => Some(ArchiveKind::K_COFF),
            _ => None,
        }
    }

    pub fn using_llvm(&self) -> bool {
        self.llvm_archive_kind().is_some()
    }

    fn build_with_ar_cmd(&mut self) -> io::Result<()> {
        let removals = mem::replace(&mut self.removals, Vec::new());
        let additions = mem::replace(&mut self.additions, Vec::new());
        let should_update_symbols = mem::replace(&mut self.should_update_symbols,
                                                 false);

        // Don't use fs::copy because libs may be installed as read-only and we
        // want to modify this archive, so we use `io::copy` to not preserve
        // permission bits.
        if let Some(ref s) = self.config.src {
            try!(io::copy(&mut try!(File::open(s)),
                          &mut try!(File::create(&self.config.dst))));
        }

        if removals.len() > 0 {
            self.run(None, Action::Remove(&removals));
        }

        let mut members = Vec::new();
        for addition in additions {
            match addition {
                Addition::File { path, name_in_archive } => {
                    let dst = self.work_dir.path().join(&name_in_archive);
                    try!(fs::copy(&path, &dst));
                    members.push(PathBuf::from(name_in_archive));
                }
                Addition::Archive { archive, archive_name, mut skip } => {
                    try!(self.add_archive_members(&mut members, archive,
                                                  &archive_name, &mut *skip));
                }
            }
        }

        // Get an absolute path to the destination, so `ar` will work even
        // though we run it from `self.work_dir`.
        let mut objects = Vec::new();
        let mut total_len = self.config.dst.to_string_lossy().len();

        if members.is_empty() {
            if should_update_symbols {
                self.run(Some(self.work_dir.path()), Action::UpdateSymbols);
            }
            return Ok(())
        }

        // Don't allow the total size of `args` to grow beyond 32,000 bytes.
        // Windows will raise an error if the argument string is longer than
        // 32,768, and we leave a bit of extra space for the program name.
        const ARG_LENGTH_LIMIT: usize = 32_000;

        for member_name in &members {
            let len = member_name.to_string_lossy().len();

            // `len + 1` to account for the space that's inserted before each
            // argument.  (Windows passes command-line arguments as a single
            // string, not an array of strings.)
            if total_len + len + 1 > ARG_LENGTH_LIMIT {
                // Add the archive members seen so far, without updating the
                // symbol table.
                self.run(Some(self.work_dir.path()),
                         Action::AddObjects(&objects, false));

                objects.clear();
                total_len = self.config.dst.to_string_lossy().len();
            }

            objects.push(member_name);
            total_len += len + 1;
        }

        // Add the remaining archive members, and update the symbol table if
        // necessary.
        self.run(Some(self.work_dir.path()),
                         Action::AddObjects(&objects, should_update_symbols));
        Ok(())
    }

    fn add_archive_members(&mut self, members: &mut Vec<PathBuf>,
                           archive: ArchiveRO, name: &str,
                           skip: &mut FnMut(&str) -> bool) -> io::Result<()> {
        // Next, we must rename all of the inputs to "guaranteed unique names".
        // We write each file into `self.work_dir` under its new unique name.
        // The reason for this renaming is that archives are keyed off the name
        // of the files, so if two files have the same name they will override
        // one another in the archive (bad).
        //
        // We skip any files explicitly desired for skipping, and we also skip
        // all SYMDEF files as these are just magical placeholders which get
        // re-created when we make a new archive anyway.
        for file in archive.iter().filter(is_relevant_child) {
            let filename = file.name().unwrap();
            if skip(filename) { continue }
            let filename = Path::new(filename).file_name().unwrap()
                                              .to_str().unwrap();

            // Archives on unix systems typically do not have slashes in
            // filenames as the `ar` utility generally only uses the last
            // component of a path for the filename list in the archive. On
            // Windows, however, archives assembled with `lib.exe` will preserve
            // the full path to the file that was placed in the archive,
            // including path separators.
            //
            // The code below is munging paths so it'll go wrong pretty quickly
            // if there's some unexpected slashes in the filename, so here we
            // just chop off everything but the filename component. Note that
            // this can cause duplicate filenames, but that's also handled below
            // as well.
            let filename = Path::new(filename).file_name().unwrap()
                                              .to_str().unwrap();

            // An archive can contain files of the same name multiple times, so
            // we need to be sure to not have them overwrite one another when we
            // extract them. Consequently we need to find a truly unique file
            // name for us!
            let mut new_filename = String::new();
            for n in 0.. {
                let n = if n == 0 {String::new()} else {format!("-{}", n)};
                new_filename = format!("r{}-{}-{}", n, name, filename);

                // LLDB (as mentioned in back::link) crashes on filenames of
                // exactly
                // 16 bytes in length. If we're including an object file with
                //    exactly 16-bytes of characters, give it some prefix so
                //    that it's not 16 bytes.
                new_filename = if new_filename.len() == 16 {
                    format!("lldb-fix-{}", new_filename)
                } else {
                    new_filename
                };

                let present = members.iter().filter_map(|p| {
                    p.file_name().and_then(|f| f.to_str())
                }).any(|s| s == new_filename);
                if !present {
                    break
                }
            }
            let dst = self.work_dir.path().join(&new_filename);
            try!(try!(File::create(&dst)).write_all(file.data()));
            members.push(PathBuf::from(new_filename));
        }
        Ok(())
    }

    fn run(&self, cwd: Option<&Path>, action: Action) -> Output {
        let abs_dst = env::current_dir().unwrap().join(&self.config.dst);
        let ar = &self.config.ar_prog;
        let mut cmd = Command::new(ar);
        cmd.env("PATH", &self.config.command_path);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        self.prepare_ar_action(&mut cmd, &abs_dst, action);
        info!("{:?}", cmd);

        if let Some(p) = cwd {
            cmd.current_dir(p);
            info!("inside {:?}", p.display());
        }

        let sess = &self.config.sess;
        match cmd.spawn() {
            Ok(prog) => {
                let o = prog.wait_with_output().unwrap();
                if !o.status.success() {
                    sess.struct_err(&format!("{:?} failed with: {}", cmd, o.status))
                        .note(&format!("stdout ---\n{}",
                                       str::from_utf8(&o.stdout).unwrap()))
                        .note(&format!("stderr ---\n{}",
                                       str::from_utf8(&o.stderr).unwrap()))
                        .emit();
                    sess.abort_if_errors();
                }
                o
            },
            Err(e) => {
                sess.fatal(&format!("could not exec `{}`: {}",
                                    self.config.ar_prog, e));
            }
        }
    }

    fn prepare_ar_action(&self, cmd: &mut Command, dst: &Path, action: Action) {
        match action {
            Action::Remove(files) => {
                cmd.arg("d").arg(dst).args(files);
            }
            Action::AddObjects(objs, update_symbols) => {
                cmd.arg(if update_symbols {"crs"} else {"crS"})
                   .arg(dst)
                   .args(objs);
            }
            Action::UpdateSymbols => {
                cmd.arg("s").arg(dst);
            }
        }
    }

    fn build_with_llvm(&mut self, kind: ArchiveKind) -> io::Result<()> {
        let mut archives = Vec::new();
        let mut strings = Vec::new();
        let mut members = Vec::new();
        let removals = mem::replace(&mut self.removals, Vec::new());

        unsafe {
            if let Some(archive) = self.src_archive() {
                for child in archive.iter() {
                    let child_name = match child.name() {
                        Some(s) => s,
                        None => continue,
                    };
                    if removals.iter().any(|r| r == child_name) {
                        continue
                    }

                    let name = try!(CString::new(child_name));
                    members.push(llvm::LLVMRustArchiveMemberNew(ptr::null(),
                                                                name.as_ptr(),
                                                                child.raw()));
                    strings.push(name);
                }
            }
            for addition in mem::replace(&mut self.additions, Vec::new()) {
                match addition {
                    Addition::File { path, name_in_archive } => {
                        let path = try!(CString::new(path.to_str().unwrap()));
                        let name = try!(CString::new(name_in_archive));
                        members.push(llvm::LLVMRustArchiveMemberNew(path.as_ptr(),
                                                                    name.as_ptr(),
                                                                    ptr::null_mut()));
                        strings.push(path);
                        strings.push(name);
                    }
                    Addition::Archive { archive, archive_name: _, mut skip } => {
                        for child in archive.iter().filter(is_relevant_child) {
                            let child_name = child.name().unwrap();
                            if skip(child_name) { continue }

                            let name = try!(CString::new(child_name));
                            let m = llvm::LLVMRustArchiveMemberNew(ptr::null(),
                                                                   name.as_ptr(),
                                                                   child.raw());
                            members.push(m);
                            strings.push(name);
                        }
                        archives.push(archive);
                    }
                }
            }

            let dst = self.config.dst.to_str().unwrap().as_bytes();
            let dst = try!(CString::new(dst));
            let r = llvm::LLVMRustWriteArchive(dst.as_ptr(),
                                               members.len() as libc::size_t,
                                               members.as_ptr(),
                                               self.should_update_symbols,
                                               kind);
            let ret = if r != 0 {
                let err = llvm::LLVMRustGetLastError();
                let msg = if err.is_null() {
                    "failed to write archive".to_string()
                } else {
                    String::from_utf8_lossy(CStr::from_ptr(err).to_bytes())
                            .into_owned()
                };
                Err(io::Error::new(io::ErrorKind::Other, msg))
            } else {
                Ok(())
            };
            for member in members {
                llvm::LLVMRustArchiveMemberFree(member);
            }
            return ret
        }
    }
}
