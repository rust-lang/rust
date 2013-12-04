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

use back::archive::{Archive, METADATA_FILENAME};
use driver::session::Session;
use lib::llvm::{False, llvm, ObjectFile, mk_section_iter};
use metadata::decoder;
use metadata::encoder;
use metadata::filesearch::{FileMatches, FileDoesntMatch};
use metadata::filesearch;
use syntax::codemap::Span;
use syntax::diagnostic::span_handler;
use syntax::parse::token::ident_interner;
use syntax::print::pprust;
use syntax::{ast, attr};
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
    metas: ~[@ast::MetaItem],
    hash: @str,
    os: Os,
    intr: @ident_interner
}

pub struct Library {
    dylib: Option<Path>,
    rlib: Option<Path>,
    metadata: @~[u8],
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
        attr::require_unique_names(self.sess.diagnostic(), self.metas);
        let filesearch = self.sess.filesearch;
        let crate_name = crate_name_from_metas(self.metas);
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
                        match get_metadata_section(self.sess, self.os, path) {
                            Some(cvec) =>
                                if crate_matches(cvec, self.metas, self.hash) {
                                    debug!("found {} with matching metadata",
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
                                    debug!("skipping {}, metadata doesn't match",
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
                    let attrs = decoder::get_crate_attributes(lib.metadata);
                    note_linkage_attrs(self.intr, self.sess.diagnostic(), attrs);
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

pub fn crate_name_from_metas(metas: &[@ast::MetaItem]) -> @str {
    for m in metas.iter() {
        match m.name_str_pair() {
            Some((name, s)) if "name" == name => { return s; }
            _ => {}
        }
    }
    fail!("expected to find the crate name")
}

pub fn package_id_from_metas(metas: &[@ast::MetaItem]) -> Option<@str> {
    for m in metas.iter() {
        match m.name_str_pair() {
            Some((name, s)) if "package_id" == name => { return Some(s); }
            _ => {}
        }
    }
    None
}

pub fn note_linkage_attrs(intr: @ident_interner,
                          diag: @mut span_handler,
                          attrs: ~[ast::Attribute]) {
    let r = attr::find_linkage_metas(attrs);
    for mi in r.iter() {
        diag.handler().note(format!("meta: {}", pprust::meta_item_to_str(*mi,intr)));
    }
}

fn crate_matches(crate_data: @~[u8],
                 metas: &[@ast::MetaItem],
                 hash: @str) -> bool {
    let attrs = decoder::get_crate_attributes(crate_data);
    let linkage_metas = attr::find_linkage_metas(attrs);
    if !hash.is_empty() {
        let chash = decoder::get_crate_hash(crate_data);
        if chash != hash { return false; }
    }
    metadata_matches(linkage_metas, metas)
}

pub fn metadata_matches(extern_metas: &[@ast::MetaItem],
                        local_metas: &[@ast::MetaItem]) -> bool {

// extern_metas: metas we read from the crate
// local_metas: metas we're looking for
    debug!("matching {} metadata requirements against {} items",
           local_metas.len(), extern_metas.len());

    local_metas.iter().all(|needed| attr::contains(extern_metas, *needed))
}

fn get_metadata_section(sess: Session, os: Os, filename: &Path) -> Option<@~[u8]> {
    if filename.filename_str().unwrap().ends_with(".rlib") {
        let archive = Archive::open(sess, filename.clone());
        return Some(@archive.read(METADATA_FILENAME));
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
                    found = Some(@(inflated));
                });
                if found != None {
                    return found;
                }
            }
            llvm::LLVMMoveToNextSection(si.llsi);
        }
        return option::None::<@~[u8]>;
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
pub fn list_file_metadata(sess: Session,
                          intr: @ident_interner,
                          os: Os,
                          path: &Path,
                          out: @mut io::Writer) {
    match get_metadata_section(sess, os, path) {
      option::Some(bytes) => decoder::list_crate_metadata(intr, bytes, out),
      option::None => {
        write!(out, "could not find metadata in {}.\n", path.display())
      }
    }
}
