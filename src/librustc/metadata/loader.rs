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
use back::svh::Svh;
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
use syntax::attr::AttrMetaMethods;

use std::c_str::ToCStr;
use std::cast;
use std::cmp;
use std::io;
use std::os::consts::{macos, freebsd, linux, android, win32};
use std::str;
use std::vec;
use std::vec_ng::Vec;

use collections::{HashMap, HashSet};
use flate;
use time;

pub enum Os {
    OsMacos,
    OsWin32,
    OsLinux,
    OsAndroid,
    OsFreebsd
}

pub struct Context<'a> {
    sess: &'a Session,
    span: Span,
    ident: &'a str,
    crate_id: &'a CrateId,
    id_hash: &'a str,
    hash: Option<&'a Svh>,
    os: Os,
    intr: @IdentInterner,
    rejected_via_hash: bool,
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

// FIXME(#11857) this should be a "real" realpath
fn realpath(p: &Path) -> Path {
    use std::os;
    use std::io::fs;

    let path = os::make_absolute(p);
    match fs::readlink(&path) {
        Ok(p) => p,
        Err(..) => path
    }
}

impl<'a> Context<'a> {
    pub fn load_library_crate(&mut self, root_ident: Option<&str>) -> Library {
        match self.find_library_crate() {
            Some(t) => t,
            None => {
                self.sess.abort_if_errors();
                let message = if self.rejected_via_hash {
                    format!("found possibly newer version of crate `{}`",
                            self.ident)
                } else {
                    format!("can't find crate for `{}`", self.ident)
                };
                let message = match root_ident {
                    None => message,
                    Some(c) => format!("{} which `{}` depends on", message, c),
                };
                self.sess.span_err(self.span, message);

                if self.rejected_via_hash {
                    self.sess.span_note(self.span, "perhaps this crate needs \
                                                    to be recompiled?");
                }
                self.sess.abort_if_errors();
                unreachable!()
            }
        }
    }

    fn find_library_crate(&mut self) -> Option<Library> {
        let filesearch = self.sess.filesearch();
        let (dyprefix, dysuffix) = self.dylibname();

        // want: crate_name.dir_part() + prefix + crate_name.file_part + "-"
        let dylib_prefix = format!("{}{}-", dyprefix, self.crate_id.name);
        let rlib_prefix = format!("lib{}-", self.crate_id.name);

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
                        rlibs.insert(realpath(path));
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
                        dylibs.insert(realpath(path));
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
        let mut libraries = Vec::new();
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
            1 => Some(libraries.move_iter().next().unwrap()),
            _ => {
                self.sess.span_err(self.span,
                    format!("multiple matching crates for `{}`",
                            self.crate_id.name));
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
                    let crate_id = decoder::get_crate_id(data);
                    note_crateid_attr(self.sess.diagnostic(), &crate_id);
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
        debug!("matching -- {}, hash: {} (want {})", file, hash, self.id_hash);
        let vers = match parts.next() { Some(v) => v, None => return None };
        debug!("matching -- {}, vers: {} (want {})", file, vers,
               self.crate_id.version);
        match self.crate_id.version {
            Some(ref version) if version.as_slice() != vers => return None,
            Some(..) => {} // check the hash

            // hash is irrelevant, no version specified
            None => return Some(hash.to_owned())
        }
        debug!("matching -- {}, vers ok", file);
        // hashes in filenames are prefixes of the "true hash"
        if self.id_hash == hash.as_slice() {
            debug!("matching -- {}, hash ok", file);
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
    fn extract_one(&mut self, m: HashSet<Path>, flavor: &str,
                   slot: &mut Option<MetadataBlob>) -> Option<Path> {
        if m.len() == 0 { return None }
        if m.len() > 1 {
            self.sess.span_err(self.span,
                               format!("multiple {} candidates for `{}` \
                                        found", flavor, self.crate_id.name));
            for (i, path) in m.iter().enumerate() {
                self.sess.span_note(self.span,
                                    format!(r"candidate \#{}: {}", i + 1,
                                            path.display()));
            }
            return None
        }

        let lib = m.move_iter().next().unwrap();
        if slot.is_none() {
            info!("{} reading metadata from: {}", flavor, lib.display());
            match get_metadata_section(self.os, &lib) {
                Ok(blob) => {
                    if self.crate_matches(blob.as_slice()) {
                        *slot = Some(blob);
                    } else {
                        info!("metadata mismatch");
                        return None;
                    }
                }
                Err(_) => {
                    info!("no metadata found");
                    return None
                }
            }
        }
        return Some(lib);
    }

    fn crate_matches(&mut self, crate_data: &[u8]) -> bool {
        match decoder::maybe_get_crate_id(crate_data) {
            Some(ref id) if self.crate_id.matches(id) => {}
            _ => return false
        }
        let hash = match decoder::maybe_get_crate_hash(crate_data) {
            Some(hash) => hash, None => return false
        };
        match self.hash {
            None => true,
            Some(myhash) => {
                if *myhash != hash {
                    self.rejected_via_hash = true;
                    false
                } else {
                    true
                }
            }
        }
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

pub fn note_crateid_attr(diag: &SpanHandler, crateid: &CrateId) {
    diag.handler().note(format!("crate_id: {}", crateid.to_str()));
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
fn get_metadata_section(os: Os, filename: &Path) -> Result<MetadataBlob, ~str> {
    let start = time::precise_time_ns();
    let ret = get_metadata_section_imp(os, filename);
    info!("reading {} => {}ms", filename.filename_display(),
           (time::precise_time_ns() - start) / 1000000);
    return ret;
}

fn get_metadata_section_imp(os: Os, filename: &Path) -> Result<MetadataBlob, ~str> {
    if !filename.exists() {
        return Err(format!("no such file: '{}'", filename.display()));
    }
    if filename.filename_str().unwrap().ends_with(".rlib") {
        // Use ArchiveRO for speed here, it's backed by LLVM and uses mmap
        // internally to read the file. We also avoid even using a memcpy by
        // just keeping the archive along while the metadata is in use.
        let archive = match ArchiveRO::open(filename) {
            Some(ar) => ar,
            None => {
                debug!("llvm didn't like `{}`", filename.display());
                return Err(format!("failed to read rlib metadata: '{}'",
                                   filename.display()));
            }
        };
        return match ArchiveMetadata::new(archive).map(|ar| MetadataArchive(ar)) {
            None => return Err(format!("failed to read rlib metadata: '{}'",
                                       filename.display())),
            Some(blob) => return Ok(blob)
        }
    }
    unsafe {
        let mb = filename.with_c_str(|buf| {
            llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(buf)
        });
        if mb as int == 0 {
            return Err(format!("error reading library: '{}'",filename.display()))
        }
        let of = match ObjectFile::new(mb) {
            Some(of) => of,
            _ => return Err(format!("provided path not an object file: '{}'", filename.display()))
        };
        let si = mk_section_iter(of.llof);
        while llvm::LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) == False {
            let name_buf = llvm::LLVMGetSectionName(si.llsi);
            let name = str::raw::from_c_str(name_buf);
            debug!("get_metadata_section: name {}", name);
            if read_meta_section_name(os) == name {
                let cbuf = llvm::LLVMGetSectionContents(si.llsi);
                let csz = llvm::LLVMGetSectionSize(si.llsi) as uint;
                let mut found = Err(format!("metadata not found: '{}'", filename.display()));
                let cvbuf: *u8 = cast::transmute(cbuf);
                let vlen = encoder::metadata_encoding_version.len();
                debug!("checking {} bytes of metadata-version stamp",
                       vlen);
                let minsz = cmp::min(vlen, csz);
                let version_ok = vec::raw::buf_as_slice(cvbuf, minsz,
                    |buf0| buf0 == encoder::metadata_encoding_version);
                if !version_ok { return Err(format!("incompatible metadata version found: '{}'",
                                                    filename.display()));}

                let cvbuf1 = cvbuf.offset(vlen as int);
                debug!("inflating {} bytes of compressed metadata",
                       csz - vlen);
                vec::raw::buf_as_slice(cvbuf1, csz-vlen, |bytes| {
                    let inflated = flate::inflate_bytes(bytes);
                    found = Ok(MetadataVec(inflated));
                });
                if found.is_ok() {
                    return found;
                }
            }
            llvm::LLVMMoveToNextSection(si.llsi);
        }
        return Err(format!("metadata not found: '{}'", filename.display()));
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
        Ok(bytes) => decoder::list_crate_metadata(bytes.as_slice(), out),
        Err(msg) => {
            write!(out, "{}\n", msg)
        }
    }
}
