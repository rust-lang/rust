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
use syntax::codemap::Span;
use syntax::diagnostic::SpanHandler;
use syntax::parse::token::IdentInterner;
use syntax::crateid::CrateId;
use syntax::attr;
use syntax::attr::AttrMetaMethods;

use std::c_str::ToCStr;
use std::cast;
use std::hashmap::{HashMap, HashSet};
use std::cmp;
use std::io;
use std::os::consts::{macos, freebsd, linux, android, win32};
use std::str;
use std::vec;

use flate;
use time;

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
    ident: ~str,
    name: ~str,
    version: ~str,
    hash: ~str,
    os: Os,
    intr: @IdentInterner
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
    pub fn load_library_crate(&self, root_ident: Option<~str>) -> Library {
        match self.find_library_crate() {
            Some(t) => t,
            None => {
                self.sess.abort_if_errors();
                let message = match root_ident {
                    None => format!("can't find crate for `{}`", self.ident),
                    Some(c) => format!("can't find crate for `{}` which `{}` depends on",
                                       self.ident,
                                       c)
                };
                self.sess.span_fatal(self.span, message);
            }
        }
    }

    fn find_library_crate(&self) -> Option<Library> {
        let filesearch = self.sess.filesearch;
        let (dyprefix, dysuffix) = self.dylibname();

        // want: crate_name.dir_part() + prefix + crate_name.file_part + "-"
        let dylib_prefix = format!("{}{}-", dyprefix, self.name);
        let rlib_prefix = format!("lib{}-", self.name);

        let mut candidates = HashMap::new();

        // First, find all possible candidate rlibs and dylibs purely based on
        // the name of the files themselves. We're trying to match against an
        // exact crate_id and a possibly an exact hash.
        //
        // During this step, we can filter all found libraries based on the
        // name and id found in the crate id (we ignore the path portion for
        // filename matching), as well as the exact hash (if specified). If we
        // end up having many candidates, we must look at the metadata to
        // perform exact matches against hashes/crate ids. Note that opening up
        // the metadata is where we do an exact match against the full contents
        // of the crate id (path/name/id).
        //
        // The goal of this step is to look at as little metadata as possible.
        filesearch.search(|path| {
            let file = match path.filename_str() {
                None => return FileDoesntMatch,
                Some(file) => file,
            };
            if file.starts_with(rlib_prefix) && file.ends_with(".rlib") {
                info!("rlib candidate: {}", path.display());
                match self.try_match(file, rlib_prefix, ".rlib") {
                    Some(hash) => {
                        info!("rlib accepted, hash: {}", hash);
                        let slot = candidates.find_or_insert_with(hash, |_| {
                            (HashSet::new(), HashSet::new())
                        });
                        let (ref mut rlibs, _) = *slot;
                        rlibs.insert(path.clone());
                        FileMatches
                    }
                    None => {
                        info!("rlib rejected");
                        FileDoesntMatch
                    }
                }
            } else if file.starts_with(dylib_prefix) && file.ends_with(dysuffix){
                info!("dylib candidate: {}", path.display());
                match self.try_match(file, dylib_prefix, dysuffix) {
                    Some(hash) => {
                        info!("dylib accepted, hash: {}", hash);
                        let slot = candidates.find_or_insert_with(hash, |_| {
                            (HashSet::new(), HashSet::new())
                        });
                        let (_, ref mut dylibs) = *slot;
                        dylibs.insert(path.clone());
                        FileMatches
                    }
                    None => {
                        info!("dylib rejected");
                        FileDoesntMatch
                    }
                }
            } else {
                FileDoesntMatch
            }
        });

        // We have now collected all known libraries into a set of candidates
        // keyed of the filename hash listed. For each filename, we also have a
        // list of rlibs/dylibs that apply. Here, we map each of these lists
        // (per hash), to a Library candidate for returning.
        //
        // A Library candidate is created if the metadata for the set of
        // libraries corresponds to the crate id and hash criteria that this
        // serach is being performed for.
        let mut libraries = ~[];
        for (_hash, (rlibs, dylibs)) in candidates.move_iter() {
            let mut metadata = None;
            let rlib = self.extract_one(rlibs, "rlib", &mut metadata);
            let dylib = self.extract_one(dylibs, "dylib", &mut metadata);
            match metadata {
                Some(metadata) => {
                    libraries.push(Library {
                        dylib: dylib,
                        rlib: rlib,
                        metadata: metadata,
                    })
                }
                None => {}
            }
        }

        // Having now translated all relevant found hashes into libraries, see
        // what we've got and figure out if we found multiple candidates for
        // libraries or not.
        match libraries.len() {
            0 => None,
            1 => Some(libraries[0]),
            _ => {
                self.sess.span_err(self.span,
                    format!("multiple matching crates for `{}`", self.name));
                self.sess.note("candidates:");
                for lib in libraries.iter() {
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
                    match attr::find_crateid(attrs) {
                        None => {}
                        Some(crateid) => {
                            note_crateid_attr(self.sess.diagnostic(), &crateid);
                        }
                    }
                }
                None
            }
        }
    }

    // Attempts to match the requested version of a library against the file
    // specified. The prefix/suffix are specified (disambiguates between
    // rlib/dylib).
    //
    // The return value is `None` if `file` doesn't look like a rust-generated
    // library, or if a specific version was requested and it doens't match the
    // apparent file's version.
    //
    // If everything checks out, then `Some(hash)` is returned where `hash` is
    // the listed hash in the filename itself.
    fn try_match(&self, file: &str, prefix: &str, suffix: &str) -> Option<~str>{
        let middle = file.slice(prefix.len(), file.len() - suffix.len());
        debug!("matching -- {}, middle: {}", file, middle);
        let mut parts = middle.splitn('-', 1);
        let hash = match parts.next() { Some(h) => h, None => return None };
        debug!("matching -- {}, hash: {}", file, hash);
        let vers = match parts.next() { Some(v) => v, None => return None };
        debug!("matching -- {}, vers: {}", file, vers);
        if !self.version.is_empty() && self.version.as_slice() != vers {
            return None
        }
        debug!("matching -- {}, vers ok (requested {})", file,
               self.version);
        // hashes in filenames are prefixes of the "true hash"
        if self.hash.is_empty() || self.hash.starts_with(hash) {
            debug!("matching -- {}, hash ok (requested {})", file, self.hash);
            Some(hash.to_owned())
        } else {
            None
        }
    }

    // Attempts to extract *one* library from the set `m`. If the set has no
    // elements, `None` is returned. If the set has more than one element, then
    // the errors and notes are emitted about the set of libraries.
    //
    // With only one library in the set, this function will extract it, and then
    // read the metadata from it if `*slot` is `None`. If the metadata couldn't
    // be read, it is assumed that the file isn't a valid rust library (no
    // errors are emitted).
    //
    // FIXME(#10786): for an optimization, we only read one of the library's
    //                metadata sections. In theory we should read both, but
    //                reading dylib metadata is quite slow.
    fn extract_one(&self, m: HashSet<Path>, flavor: &str,
                   slot: &mut Option<MetadataBlob>) -> Option<Path> {
        if m.len() == 0 { return None }
        if m.len() > 1 {
            self.sess.span_err(self.span,
                               format!("multiple {} candidates for `{}` \
                                        found", flavor, self.name));
            for (i, path) in m.iter().enumerate() {
                self.sess.span_note(self.span,
                                    format!(r"candidate \#{}: {}", i + 1,
                                            path.display()));
            }
            return None
        }

        let lib = m.move_iter().next().unwrap();
        if slot.is_none() {
            info!("{} reading meatadata from: {}", flavor, lib.display());
            match get_metadata_section(self.os, &lib) {
                Some(blob) => {
                    if crate_matches(blob.as_slice(), self.name,
                                     self.version, self.hash) {
                        *slot = Some(blob);
                    } else {
                        info!("metadata mismatch");
                        return None;
                    }
                }
                None => {
                    info!("no metadata found");
                    return None
                }
            }
        }
        return Some(lib);
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

pub fn note_crateid_attr(diag: @SpanHandler, crateid: &CrateId) {
    diag.handler().note(format!("crate_id: {}", crateid.to_str()));
}

fn crate_matches(crate_data: &[u8],
                 name: &str,
                 version: &str,
                 hash: &str) -> bool {
    let attrs = decoder::get_crate_attributes(crate_data);
    match attr::find_crateid(attrs) {
        None => false,
        Some(crateid) => {
            if !hash.is_empty() {
                let chash = decoder::get_crate_hash(crate_data);
                if chash.as_slice() != hash { return false; }
            }
            name == crateid.name &&
                (version.is_empty() ||
                 crateid.version_or_default() == version)
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
                let minsz = cmp::min(vlen, csz);
                let mut version_ok = false;
                vec::raw::buf_as_slice(cvbuf, minsz, |buf0| {
                    version_ok = (buf0 ==
                                  encoder::metadata_encoding_version);
                });
                if !version_ok { return None; }

                let cvbuf1 = cvbuf.offset(vlen as int);
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
pub fn list_file_metadata(os: Os, path: &Path,
                          out: &mut io::Writer) -> io::IoResult<()> {
    match get_metadata_section(os, path) {
        Some(bytes) => decoder::list_crate_metadata(bytes.as_slice(), out),
        None => {
            write!(out, "could not find metadata in {}.\n", path.display())
        }
    }
}
