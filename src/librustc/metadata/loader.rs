// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Finds crate binaries and loads their metadata

use back::archive::{ArchiveRO, METADATA_FILENAME};
use driver::session::Session;
use lib::llvm::{False, llvm, ObjectFile, mk_section_iter};
use metadata::cstore::{MetadataBlob, MetadataVec, MetadataArchive};
use metadata::decoder;
use metadata::encoder;
use metadata::filesearch::{FileMatches, FileDoesntMatch};
use metadata::filesearch;
use syntax::codemap::Span;
use syntax::diagnostic::span_handler;
use syntax::parse::token::ident_interner;
use syntax::pkgid::PkgId;
use syntax::attr;
use syntax::attr::AttrMetaMethods;

use std::c_str::ToCStr;
use std::cast;
use std::io;
use std::num;
use std::option;
use std::os::consts::{macos, freebsd, linux, android, win32};
use std::ptr;
use std::str;
use std::vec;
use extra::flate;

pub enum Os {
    OsMacos,
    OsWin32,
    OsLinux,
    OsAndroid,
    OsFreebsd
}

pub struct Context {
    sess: Session,
    span: Span,
    ident: @str,
    name: @str,
    version: @str,
    hash: @str,
    os: Os,
    intr: @ident_interner
}

pub struct Library {
    dylib: Option<Path>,
    rlib: Option<Path>,
    metadata: MetadataBlob,
}

pub struct ArchiveMetadata {
    priv archive: ArchiveRO,
    // See comments in ArchiveMetadata::new for why this is static
    priv data: &'static [u8],
}

impl Context {
    pub fn load_library_crate(&self) -> Library {
        match self.find_library_crate() {
            Some(t) => t,
            None => {
                self.sess.span_fatal(self.span,
                                     format!("can't find crate for `{}`",
                                             self.ident));
            }
        }
    }

    fn find_library_crate(&self) -> Option<Library> {
        let filesearch = self.sess.filesearch;
        let crate_name = self.name;
        let (dyprefix, dysuffix) = self.dylibname();

        // want: crate_name.dir_part() + prefix + crate_name.file_part + "-"
        let dylib_prefix = format!("{}{}-", dyprefix, crate_name);
        let rlib_prefix = format!("lib{}-", crate_name);

        let mut matches = ~[];
        filesearch::search(filesearch, |path| {
            match path.filename_str() {
                None => FileDoesntMatch,
                Some(file) => {
                    let (candidate, existing) = if file.starts_with(rlib_prefix) &&
                                                   file.ends_with(".rlib") {
                        debug!("{} is an rlib candidate", path.display());
                        (true, self.add_existing_rlib(matches, path, file))
                    } else if file.starts_with(dylib_prefix) &&
                              file.ends_with(dysuffix) {
                        debug!("{} is a dylib candidate", path.display());
                        (true, self.add_existing_dylib(matches, path, file))
                    } else {
                        (false, false)
                    };

                    if candidate && existing {
                        FileMatches
                    } else if candidate {
                        match get_metadata_section(self.os, path) {
                            Some(cvec) =>
                                if crate_matches(cvec.as_slice(), self.name,
                                                 self.version, self.hash) {
                                    debug!("found {} with matching pkgid",
                                           path.display());
                                    let (rlib, dylib) = if file.ends_with(".rlib") {
                                        (Some(path.clone()), None)
                                    } else {
                                        (None, Some(path.clone()))
                                    };
                                    matches.push(Library {
                                        rlib: rlib,
                                        dylib: dylib,
                                        metadata: cvec,
                                    });
                                    FileMatches
                                } else {
                                    debug!("skipping {}, pkgid doesn't match",
                                           path.display());
                                    FileDoesntMatch
                                },
                                _ => {
                                    debug!("could not load metadata for {}",
                                           path.display());
                                    FileDoesntMatch
                                }
                        }
                    } else {
                        FileDoesntMatch
                    }
                }
            }
        });

        match matches.len() {
            0 => None,
            1 => Some(matches[0]),
            _ => {
                self.sess.span_err(self.span,
                    format!("multiple matching crates for `{}`", crate_name));
                self.sess.note("candidates:");
                for lib in matches.iter() {
                    match lib.dylib {
                        Some(ref p) => {
                            self.sess.note(format!("path: {}", p.display()));
                        }
                        None => {}
                    }
                    match lib.rlib {
                        Some(ref p) => {
                            self.sess.note(format!("path: {}", p.display()));
                        }
                        None => {}
                    }
                    let data = lib.metadata.as_slice();
                    let attrs = decoder::get_crate_attributes(data);
                    match attr::find_pkgid(attrs) {
                        None => {}
                        Some(pkgid) => {
                            note_pkgid_attr(self.sess.diagnostic(), &pkgid);
                        }
                    }
                }
                self.sess.abort_if_errors();
                None
            }
        }
    }

    fn add_existing_rlib(&self, libs: &mut [Library],
                         path: &Path, file: &str) -> bool {
        let (prefix, suffix) = self.dylibname();
        let file = file.slice_from(3); // chop off 'lib'
        let file = file.slice_to(file.len() - 5); // chop off '.rlib'
        let file = format!("{}{}{}", prefix, file, suffix);

        for lib in libs.mut_iter() {
            match lib.dylib {
                Some(ref p) if p.filename_str() == Some(file.as_slice()) => {
                    assert!(lib.rlib.is_none()); // XXX: legit compiler error
                    lib.rlib = Some(path.clone());
                    return true;
                }
                Some(..) | None => {}
            }
        }
        return false;
    }

    fn add_existing_dylib(&self, libs: &mut [Library],
                          path: &Path, file: &str) -> bool {
        let (prefix, suffix) = self.dylibname();
        let file = file.slice_from(prefix.len());
        let file = file.slice_to(file.len() - suffix.len());
        let file = format!("lib{}.rlib", file);

        for lib in libs.mut_iter() {
            match lib.rlib {
                Some(ref p) if p.filename_str() == Some(file.as_slice()) => {
                    assert!(lib.dylib.is_none()); // XXX: legit compiler error
                    lib.dylib = Some(path.clone());
                    return true;
                }
                Some(..) | None => {}
            }
        }
        return false;
    }

    // Returns the corresponding (prefix, suffix) that files need to have for
    // dynamic libraries
    fn dylibname(&self) -> (&'static str, &'static str) {
        match self.os {
            OsWin32 => (win32::DLL_PREFIX, win32::DLL_SUFFIX),
            OsMacos => (macos::DLL_PREFIX, macos::DLL_SUFFIX),
            OsLinux => (linux::DLL_PREFIX, linux::DLL_SUFFIX),
            OsAndroid => (android::DLL_PREFIX, android::DLL_SUFFIX),
            OsFreebsd => (freebsd::DLL_PREFIX, freebsd::DLL_SUFFIX),
        }
    }
}

pub fn note_pkgid_attr(diag: @mut span_handler,
                       pkgid: &PkgId) {
    diag.handler().note(format!("pkgid: {}", pkgid.to_str()));
}

fn crate_matches(crate_data: &[u8],
                 name: @str,
                 version: @str,
                 hash: @str) -> bool {
    let attrs = decoder::get_crate_attributes(crate_data);
    match attr::find_pkgid(attrs) {
        None => false,
        Some(pkgid) => {
            if !hash.is_empty() {
                let chash = decoder::get_crate_hash(crate_data);
                if chash != hash { return false; }
            }
            name == pkgid.name.to_managed() &&
                (version.is_empty() || version == pkgid.version_or_default().to_managed())
        }
    }
}

impl ArchiveMetadata {
    fn new(ar: ArchiveRO) -> Option<ArchiveMetadata> {
        let data: &'static [u8] = {
            let data = match ar.read(METADATA_FILENAME) {
                Some(data) => data,
                None => {
                    debug!("didn't find '{}' in the archive", METADATA_FILENAME);
                    return None;
                }
            };
            // This data is actually a pointer inside of the archive itself, but
            // we essentially want to cache it because the lookup inside the
            // archive is a fairly expensive operation (and it's queried for
            // *very* frequently). For this reason, we transmute it to the
            // static lifetime to put into the struct. Note that the buffer is
            // never actually handed out with a static lifetime, but rather the
            // buffer is loaned with the lifetime of this containing object.
            // Hence, we're guaranteed that the buffer will never be used after
            // this object is dead, so this is a safe operation to transmute and
            // store the data as a static buffer.
            unsafe { cast::transmute(data) }
        };
        Some(ArchiveMetadata {
            archive: ar,
            data: data,
        })
    }

    pub fn as_slice<'a>(&'a self) -> &'a [u8] { self.data }
}

// Just a small wrapper to time how long reading metadata takes.
fn get_metadata_section(os: Os, filename: &Path) -> Option<MetadataBlob> {
    use extra::time;
    let start = time::precise_time_ns();
    let ret = get_metadata_section_imp(os, filename);
    info!("reading {} => {}ms", filename.filename_display(),
           (time::precise_time_ns() - start) / 1000000);
    return ret;
}

fn get_metadata_section_imp(os: Os, filename: &Path) -> Option<MetadataBlob> {
    if filename.filename_str().unwrap().ends_with(".rlib") {
        // Use ArchiveRO for speed here, it's backed by LLVM and uses mmap
        // internally to read the file. We also avoid even using a memcpy by
        // just keeping the archive along while the metadata is in use.
        let archive = match ArchiveRO::open(filename) {
            Some(ar) => ar,
            None => {
                debug!("llvm didn't like `{}`", filename.display());
                return None;
            }
        };
        return ArchiveMetadata::new(archive).map(|ar| MetadataArchive(ar));
    }
    unsafe {
        let mb = filename.with_c_str(|buf| {
            llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(buf)
        });
        if mb as int == 0 { return None }
        let of = match ObjectFile::new(mb) {
            Some(of) => of,
            _ => return None
        };
        let si = mk_section_iter(of.llof);
        while llvm::LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) == False {
            let name_buf = llvm::LLVMGetSectionName(si.llsi);
            let name = str::raw::from_c_str(name_buf);
            debug!("get_metadata_section: name {}", name);
            if read_meta_section_name(os) == name {
                let cbuf = llvm::LLVMGetSectionContents(si.llsi);
                let csz = llvm::LLVMGetSectionSize(si.llsi) as uint;
                let mut found = None;
                let cvbuf: *u8 = cast::transmute(cbuf);
                let vlen = encoder::metadata_encoding_version.len();
                debug!("checking {} bytes of metadata-version stamp",
                       vlen);
                let minsz = num::min(vlen, csz);
                let mut version_ok = false;
                vec::raw::buf_as_slice(cvbuf, minsz, |buf0| {
                    version_ok = (buf0 ==
                                  encoder::metadata_encoding_version);
                });
                if !version_ok { return None; }

                let cvbuf1 = ptr::offset(cvbuf, vlen as int);
                debug!("inflating {} bytes of compressed metadata",
                       csz - vlen);
                vec::raw::buf_as_slice(cvbuf1, csz-vlen, |bytes| {
                    let inflated = flate::inflate_bytes(bytes);
                    found = Some(MetadataVec(inflated));
                });
                if found.is_some() {
                    return found;
                }
            }
            llvm::LLVMMoveToNextSection(si.llsi);
        }
        return None;
    }
}

pub fn meta_section_name(os: Os) -> &'static str {
    match os {
        OsMacos => "__DATA,__note.rustc",
        OsWin32 => ".note.rustc",
        OsLinux => ".note.rustc",
        OsAndroid => ".note.rustc",
        OsFreebsd => ".note.rustc"
    }
}

pub fn read_meta_section_name(os: Os) -> &'static str {
    match os {
        OsMacos => "__note.rustc",
        OsWin32 => ".note.rustc",
        OsLinux => ".note.rustc",
        OsAndroid => ".note.rustc",
        OsFreebsd => ".note.rustc"
    }
}

// A diagnostic function for dumping crate metadata to an output stream
pub fn list_file_metadata(intr: @ident_interner,
                          os: Os,
                          path: &Path,
                          out: @mut io::Writer) {
    match get_metadata_section(os, path) {
      option::Some(bytes) => decoder::list_crate_metadata(intr,
                                                          bytes.as_slice(),
                                                          out),
      option::None => {
        write!(out, "could not find metadata in {}.\n", path.display())
      }
    }
}
