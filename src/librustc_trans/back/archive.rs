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

use std::ffi::{CString, CStr, OsString};
use std::io;
use std::mem;
use std::path::{Path, PathBuf};
use std::ptr;
use std::str;

use libc;
use llvm::archive_ro::{ArchiveRO, Child};
use llvm::{self, ArchiveKind};
use rustc::session::Session;

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
        skip: Box<FnMut(&str) -> bool>,
    },
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
                         .filter_map(|child| child.ok())
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
        self.add_archive(&location, |_| false).unwrap_or_else(|e| {
            self.config.sess.fatal(&format!("failed to add native library {}: {}",
                                            location.to_string_lossy(), e));
        });
    }

    /// Adds all of the contents of the rlib at the specified path to this
    /// archive.
    ///
    /// This ignores adding the bytecode from the rlib, and if LTO is enabled
    /// then the object file also isn't added.
    pub fn add_rlib(&mut self,
                    rlib: &Path,
                    name: &str,
                    lto: bool,
                    skip_objects: bool) -> io::Result<()> {
        // Ignoring obj file starting with the crate name
        // as simple comparison is not enough - there
        // might be also an extra name suffix
        let obj_start = format!("{}", name);

        // Ignoring all bytecode files, no matter of
        // name
        let bc_ext = ".bytecode.deflate";
        let metadata_filename =
            self.config.sess.cstore.metadata_filename().to_owned();

        self.add_archive(rlib, move |fname: &str| {
            if fname.ends_with(bc_ext) || fname == metadata_filename {
                return true
            }

            // Don't include Rust objects if LTO is enabled
            if lto && fname.starts_with(&obj_start) && fname.ends_with(".o") {
                return true
            }

            // Otherwise if this is *not* a rust object and we're skipping
            // objects then skip this file
            if skip_objects && (!fname.starts_with(&obj_start) || !fname.ends_with(".o")) {
                return true
            }

            // ok, don't skip this
            return false
        })
    }

    fn add_archive<F>(&mut self, archive: &Path, skip: F)
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
        let kind = match self.llvm_archive_kind() {
            Ok(kind) => kind,
            Err(kind) => {
                self.config.sess.fatal(&format!("Don't know how to build archive of type: {}",
                                                kind));
            }
        };

        if let Err(e) = self.build_with_llvm(kind) {
            self.config.sess.fatal(&format!("failed to build archive: {}", e));
        }

    }

    fn llvm_archive_kind(&self) -> Result<ArchiveKind, &str> {
        let kind = &*self.config.sess.target.target.options.archive_format;
        kind.parse().map_err(|_| kind)
    }

    fn build_with_llvm(&mut self, kind: ArchiveKind) -> io::Result<()> {
        let mut archives = Vec::new();
        let mut strings = Vec::new();
        let mut members = Vec::new();
        let removals = mem::replace(&mut self.removals, Vec::new());

        unsafe {
            if let Some(archive) = self.src_archive() {
                for child in archive.iter() {
                    let child = child.map_err(string_to_io_error)?;
                    let child_name = match child.name() {
                        Some(s) => s,
                        None => continue,
                    };
                    if removals.iter().any(|r| r == child_name) {
                        continue
                    }

                    let name = CString::new(child_name)?;
                    members.push(llvm::LLVMRustArchiveMemberNew(ptr::null(),
                                                                name.as_ptr(),
                                                                child.raw()));
                    strings.push(name);
                }
            }
            for addition in mem::replace(&mut self.additions, Vec::new()) {
                match addition {
                    Addition::File { path, name_in_archive } => {
                        let path = CString::new(path.to_str().unwrap())?;
                        let name = CString::new(name_in_archive)?;
                        members.push(llvm::LLVMRustArchiveMemberNew(path.as_ptr(),
                                                                    name.as_ptr(),
                                                                    ptr::null_mut()));
                        strings.push(path);
                        strings.push(name);
                    }
                    Addition::Archive { archive, mut skip } => {
                        for child in archive.iter() {
                            let child = child.map_err(string_to_io_error)?;
                            if !is_relevant_child(&child) {
                                continue
                            }
                            let child_name = child.name().unwrap();
                            if skip(child_name) {
                                continue
                            }

                            // It appears that LLVM's archive writer is a little
                            // buggy if the name we pass down isn't just the
                            // filename component, so chop that off here and
                            // pass it in.
                            //
                            // See LLVM bug 25877 for more info.
                            let child_name = Path::new(child_name)
                                                  .file_name().unwrap()
                                                  .to_str().unwrap();
                            let name = CString::new(child_name)?;
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
            let dst = CString::new(dst)?;
            let r = llvm::LLVMRustWriteArchive(dst.as_ptr(),
                                               members.len() as libc::size_t,
                                               members.as_ptr(),
                                               self.should_update_symbols,
                                               kind);
            let ret = if r.into_result().is_err() {
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

fn string_to_io_error(s: String) -> io::Error {
    io::Error::new(io::ErrorKind::Other, format!("bad archive: {}", s))
}
