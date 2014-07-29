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
//!
//! Might I be the first to welcome you to a world of platform differences,
//! version requirements, dependency graphs, conficting desires, and fun! This
//! is the major guts (along with metadata::creader) of the compiler for loading
//! crates and resolving dependencies. Let's take a tour!
//!
//! # The problem
//!
//! Each invocation of the compiler is immediately concerned with one primary
//! problem, to connect a set of crates to resolved crates on the filesystem.
//! Concretely speaking, the compiler follows roughly these steps to get here:
//!
//! 1. Discover a set of `extern crate` statements.
//! 2. Transform these directives into crate names. If the directive does not
//!    have an explicit name, then the identifier is the name.
//! 3. For each of these crate names, find a corresponding crate on the
//!    filesystem.
//!
//! Sounds easy, right? Let's walk into some of the nuances.
//!
//! ## Transitive Dependencies
//!
//! Let's say we've got three crates: A, B, and C. A depends on B, and B depends
//! on C. When we're compiling A, we primarily need to find and locate B, but we
//! also end up needing to find and locate C as well.
//!
//! The reason for this is that any of B's types could be composed of C's types,
//! any function in B could return a type from C, etc. To be able to guarantee
//! that we can always typecheck/translate any function, we have to have
//! complete knowledge of the whole ecosystem, not just our immediate
//! dependencies.
//!
//! So now as part of the "find a corresponding crate on the filesystem" step
//! above, this involves also finding all crates for *all upstream
//! dependencies*. This includes all dependencies transitively.
//!
//! ## Rlibs and Dylibs
//!
//! The compiler has two forms of intermediate dependencies. These are dubbed
//! rlibs and dylibs for the static and dynamic variants, respectively. An rlib
//! is a rustc-defined file format (currently just an ar archive) while a dylib
//! is a platform-defined dynamic library. Each library has a metadata somewhere
//! inside of it.
//!
//! When translating a crate name to a crate on the filesystem, we all of a
//! sudden need to take into account both rlibs and dylibs! Linkage later on may
//! use either one of these files, as each has their pros/cons. The job of crate
//! loading is to discover what's possible by finding all candidates.
//!
//! Most parts of this loading systems keep the dylib/rlib as just separate
//! variables.
//!
//! ## Where to look?
//!
//! We can't exactly scan your whole hard drive when looking for dependencies,
//! so we need to places to look. Currently the compiler will implicitly add the
//! target lib search path ($prefix/lib/rustlib/$target/lib) to any compilation,
//! and otherwise all -L flags are added to the search paths.
//!
//! ## What criterion to select on?
//!
//! This a pretty tricky area of loading crates. Given a file, how do we know
//! whether it's the right crate? Currently, the rules look along these lines:
//!
//! 1. Does the filename match an rlib/dylib pattern? That is to say, does the
//!    filename have the right prefix/suffix?
//! 2. Does the filename have the right prefix for the crate name being queried?
//!    This is filtering for files like `libfoo*.rlib` and such.
//! 3. Is the file an actual rust library? This is done by loading the metadata
//!    from the library and making sure it's actually there.
//! 4. Does the name in the metadata agree with the name of the library?
//! 5. Does the target in the metadata agree with the current target?
//! 6. Does the SVH match? (more on this later)
//!
//! If the file answeres `yes` to all these questions, then the file is
//! considered as being *candidate* for being accepted. It is illegal to have
//! more than two candidates as the compiler has no method by which to resolve
//! this conflict. Additionally, rlib/dylib candidates are considered
//! separately.
//!
//! After all this has happened, we have 1 or two files as candidates. These
//! represent the rlib/dylib file found for a library, and they're returned as
//! being found.
//!
//! ### What about versions?
//!
//! A lot of effort has been put forth to remove versioning from the compiler.
//! There have been forays in the past to have versioning baked in, but it was
//! largely always deemed insufficient to the point that it was recognized that
//! it's probably something the compiler shouldn't do anyway due to its
//! complicated nature and the state of the half-baked solutions.
//!
//! With a departure from versioning, the primary criterion for loading crates
//! is just the name of a crate. If we stopped here, it would imply that you
//! could never link two crates of the same name from different sources
//! together, which is clearly a bad state to be in.
//!
//! To resolve this problem, we come to the next section!
//!
//! # Expert Mode
//!
//! A number of flags have been added to the compiler to solve the "version
//! problem" in the previous section, as well as generally enabling more
//! powerful usage of the crate loading system of the compiler. The goal of
//! these flags and options are to enable third-party tools to drive the
//! compiler with prior knowledge about how the world should look.
//!
//! ## The `--extern` flag
//!
//! The compiler accepts a flag of this form a number of times:
//!
//! ```notrust
//! --extern crate-name=path/to/the/crate.rlib
//! ```
//!
//! This flag is basically the following letter to the compiler:
//!
//! > Dear rustc,
//! >
//! > When you are attempting to load the immediate dependency `crate-name`, I
//! > would like you too assume that the library is located at
//! > `path/to/the/crate.rlib`, and look nowhere else. Also, please do not
//! > assume that the path I specified has the name `crate-name`.
//!
//! This flag basically overrides most matching logic except for validating that
//! the file is indeed a rust library. The same `crate-name` can be specified
//! twice to specify the rlib/dylib pair.
//!
//! ## Enabling "multiple versions"
//!
//! This basically boils down to the ability to specify arbitrary packages to
//! the compiler. For example, if crate A wanted to use Bv1 and Bv2, then it
//! would look something like:
//!
//! ```ignore
//! extern crate b1;
//! extern crate b2;
//!
//! fn main() {}
//! ```
//!
//! and the compiler would be invoked as:
//!
//! ```notrust
//! rustc a.rs --extern b1=path/to/libb1.rlib --extern b2=path/to/libb2.rlib
//! ```
//!
//! In this scenario there are two crates named `b` and the compiler must be
//! manually driven to be informed where each crate is.
//!
//! ## Frobbing symbols
//!
//! One of the immediate problems with linking the same library together twice
//! in the same problem is dealing with duplicate symbols. The primary way to
//! deal with this in rustc is to add hashes to the end of each symbol.
//!
//! In order to force hashes to change between versions of a library, if
//! desired, the compiler exposes an option `-C metadata=foo`, which is used to
//! initially seed each symbol hash. The string `foo` is prepended to each
//! string-to-hash to ensure that symbols change over time.
//!
//! ## Loading transitive dependencies
//!
//! Dealing with same-named-but-distinct crates is not just a local problem, but
//! one that also needs to be dealt with for transitive dependences. Note that
//! in the letter above `--extern` flags only apply to the *local* set of
//! dependencies, not the upstream transitive dependencies. Consider this
//! dependency graph:
//!
//! ```notrust
//! A.1   A.2
//! |     |
//! |     |
//! B     C
//!  \   /
//!   \ /
//!    D
//! ```
//!
//! In this scenario, when we compile `D`, we need to be able to distinctly
//! resolve `A.1` and `A.2`, but an `--extern` flag cannot apply to these
//! transitive dependencies.
//!
//! Note that the key idea here is that `B` and `C` are both *already compiled*.
//! That is, they have already resolved their dependencies. Due to unrelated
//! technical reasons, when a library is compiled, it is only compatible with
//! the *exact same* version of the upstream libraries it was compiled against.
//! We use the "Strict Version Hash" to identify the exact copy of an upstream
//! library.
//!
//! With this knowledge, we know that `B` and `C` will depend on `A` with
//! different SVH values, so we crawl the normal `-L` paths looking for
//! `liba*.rlib` and filter based on the contained SVH.
//!
//! In the end, this ends up not needing `--extern` to specify upstream
//! transitive dependencies.
//!
//! # Wrapping up
//!
//! That's the general overview of loading crates in the compiler, but it's by
//! no means all of the necessary details. Take a look at the rest of
//! metadata::loader or metadata::creader for all the juicy details!

use back::archive::{METADATA_FILENAME};
use back::svh::Svh;
use driver::session::Session;
use llvm;
use llvm::{False, ObjectFile, mk_section_iter};
use llvm::archive_ro::ArchiveRO;
use metadata::cstore::{MetadataBlob, MetadataVec, MetadataArchive};
use metadata::decoder;
use metadata::encoder;
use metadata::filesearch::{FileSearch, FileMatches, FileDoesntMatch};
use syntax::abi;
use syntax::codemap::Span;
use syntax::diagnostic::SpanHandler;
use util::fs;

use std::c_str::ToCStr;
use std::cmp;
use std::io;
use std::mem;
use std::ptr;
use std::slice;
use std::string;

use std::collections::{HashMap, HashSet};
use flate;
use time;

pub static MACOS_DLL_PREFIX: &'static str = "lib";
pub static MACOS_DLL_SUFFIX: &'static str = ".dylib";

pub static WIN32_DLL_PREFIX: &'static str = "";
pub static WIN32_DLL_SUFFIX: &'static str = ".dll";

pub static LINUX_DLL_PREFIX: &'static str = "lib";
pub static LINUX_DLL_SUFFIX: &'static str = ".so";

pub static FREEBSD_DLL_PREFIX: &'static str = "lib";
pub static FREEBSD_DLL_SUFFIX: &'static str = ".so";

pub static DRAGONFLY_DLL_PREFIX: &'static str = "lib";
pub static DRAGONFLY_DLL_SUFFIX: &'static str = ".so";

pub static ANDROID_DLL_PREFIX: &'static str = "lib";
pub static ANDROID_DLL_SUFFIX: &'static str = ".so";

pub struct CrateMismatch {
    path: Path,
    got: String,
}

pub struct Context<'a> {
    pub sess: &'a Session,
    pub span: Span,
    pub ident: &'a str,
    pub crate_name: &'a str,
    pub hash: Option<&'a Svh>,
    pub triple: &'a str,
    pub os: abi::Os,
    pub filesearch: FileSearch<'a>,
    pub root: &'a Option<CratePaths>,
    pub rejected_via_hash: Vec<CrateMismatch>,
    pub rejected_via_triple: Vec<CrateMismatch>,
    pub should_match_name: bool,
}

pub struct Library {
    pub dylib: Option<Path>,
    pub rlib: Option<Path>,
    pub metadata: MetadataBlob,
}

pub struct ArchiveMetadata {
    _archive: ArchiveRO,
    // See comments in ArchiveMetadata::new for why this is static
    data: &'static [u8],
}

pub struct CratePaths {
    pub ident: String,
    pub dylib: Option<Path>,
    pub rlib: Option<Path>
}

impl CratePaths {
    fn paths(&self) -> Vec<Path> {
        match (&self.dylib, &self.rlib) {
            (&None,    &None)              => vec!(),
            (&Some(ref p), &None) |
            (&None, &Some(ref p))          => vec!(p.clone()),
            (&Some(ref p1), &Some(ref p2)) => vec!(p1.clone(), p2.clone()),
        }
    }
}

impl<'a> Context<'a> {
    pub fn maybe_load_library_crate(&mut self) -> Option<Library> {
        self.find_library_crate()
    }

    pub fn load_library_crate(&mut self) -> Library {
        match self.find_library_crate() {
            Some(t) => t,
            None => {
                self.report_load_errs();
                unreachable!()
            }
        }
    }

    pub fn report_load_errs(&mut self) {
        let message = if self.rejected_via_hash.len() > 0 {
            format!("found possibly newer version of crate `{}`",
                    self.ident)
        } else if self.rejected_via_triple.len() > 0 {
            format!("found incorrect triple for crate `{}`", self.ident)
        } else {
            format!("can't find crate for `{}`", self.ident)
        };
        let message = match self.root {
            &None => message,
            &Some(ref r) => format!("{} which `{}` depends on",
                                    message, r.ident)
        };
        self.sess.span_err(self.span, message.as_slice());

        let mismatches = self.rejected_via_triple.iter();
        if self.rejected_via_triple.len() > 0 {
            self.sess.span_note(self.span,
                                format!("expected triple of {}",
                                        self.triple).as_slice());
            for (i, &CrateMismatch{ ref path, ref got }) in mismatches.enumerate() {
                self.sess.fileline_note(self.span,
                    format!("crate `{}` path {}{}, triple {}: {}",
                            self.ident, "#", i+1, got, path.display()).as_slice());
            }
        }
        if self.rejected_via_hash.len() > 0 {
            self.sess.span_note(self.span, "perhaps this crate needs \
                                            to be recompiled?");
            let mismatches = self.rejected_via_hash.iter();
            for (i, &CrateMismatch{ ref path, .. }) in mismatches.enumerate() {
                self.sess.fileline_note(self.span,
                    format!("crate `{}` path {}{}: {}",
                            self.ident, "#", i+1, path.display()).as_slice());
            }
            match self.root {
                &None => {}
                &Some(ref r) => {
                    for (i, path) in r.paths().iter().enumerate() {
                        self.sess.fileline_note(self.span,
                            format!("crate `{}` path #{}: {}",
                                    r.ident, i+1, path.display()).as_slice());
                    }
                }
            }
        }
        self.sess.abort_if_errors();
    }

    fn find_library_crate(&mut self) -> Option<Library> {
        // If an SVH is specified, then this is a transitive dependency that
        // must be loaded via -L plus some filtering.
        if self.hash.is_none() {
            self.should_match_name = false;
            match self.find_commandline_library() {
                Some(l) => return Some(l),
                None => {}
            }
            self.should_match_name = true;
        }

        let dypair = self.dylibname();

        // want: crate_name.dir_part() + prefix + crate_name.file_part + "-"
        let dylib_prefix = dypair.map(|(prefix, _)| {
            format!("{}{}", prefix, self.crate_name)
        });
        let rlib_prefix = format!("lib{}", self.crate_name);

        let mut candidates = HashMap::new();

        // First, find all possible candidate rlibs and dylibs purely based on
        // the name of the files themselves. We're trying to match against an
        // exact crate name and a possibly an exact hash.
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
        self.filesearch.search(|path| {
            let file = match path.filename_str() {
                None => return FileDoesntMatch,
                Some(file) => file,
            };
            let (hash, rlib) = if file.starts_with(rlib_prefix.as_slice()) &&
                    file.ends_with(".rlib") {
                (file.slice(rlib_prefix.len(), file.len() - ".rlib".len()),
                 true)
            } else if dypair.map_or(false, |(_, suffix)| {
                file.starts_with(dylib_prefix.get_ref().as_slice()) &&
                file.ends_with(suffix)
            }) {
                let (_, suffix) = dypair.unwrap();
                let dylib_prefix = dylib_prefix.get_ref().as_slice();
                (file.slice(dylib_prefix.len(), file.len() - suffix.len()),
                 false)
            } else {
                return FileDoesntMatch
            };
            info!("lib candidate: {}", path.display());
            let slot = candidates.find_or_insert_with(hash.to_string(), |_| {
                (HashSet::new(), HashSet::new())
            });
            let (ref mut rlibs, ref mut dylibs) = *slot;
            if rlib {
                rlibs.insert(fs::realpath(path).unwrap());
            } else {
                dylibs.insert(fs::realpath(path).unwrap());
            }
            FileMatches
        });

        // We have now collected all known libraries into a set of candidates
        // keyed of the filename hash listed. For each filename, we also have a
        // list of rlibs/dylibs that apply. Here, we map each of these lists
        // (per hash), to a Library candidate for returning.
        //
        // A Library candidate is created if the metadata for the set of
        // libraries corresponds to the crate id and hash criteria that this
        // search is being performed for.
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
                            self.crate_name).as_slice());
                self.sess.note("candidates:");
                for lib in libraries.iter() {
                    match lib.dylib {
                        Some(ref p) => {
                            self.sess.note(format!("path: {}",
                                                   p.display()).as_slice());
                        }
                        None => {}
                    }
                    match lib.rlib {
                        Some(ref p) => {
                            self.sess.note(format!("path: {}",
                                                   p.display()).as_slice());
                        }
                        None => {}
                    }
                    let data = lib.metadata.as_slice();
                    let name = decoder::get_crate_name(data);
                    note_crate_name(self.sess.diagnostic(), name.as_slice());
                }
                None
            }
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
    fn extract_one(&mut self, m: HashSet<Path>, flavor: &str,
                   slot: &mut Option<MetadataBlob>) -> Option<Path> {
        let mut ret = None::<Path>;
        let mut error = 0u;

        if slot.is_some() {
            // FIXME(#10786): for an optimization, we only read one of the
            //                library's metadata sections. In theory we should
            //                read both, but reading dylib metadata is quite
            //                slow.
            if m.len() == 0 {
                return None
            } else if m.len() == 1 {
                return Some(m.move_iter().next().unwrap())
            }
        }

        for lib in m.move_iter() {
            info!("{} reading metadata from: {}", flavor, lib.display());
            let metadata = match get_metadata_section(self.os, &lib) {
                Ok(blob) => {
                    if self.crate_matches(blob.as_slice(), &lib) {
                        blob
                    } else {
                        info!("metadata mismatch");
                        continue
                    }
                }
                Err(_) => {
                    info!("no metadata found");
                    continue
                }
            };
            if ret.is_some() {
                self.sess.span_err(self.span,
                                   format!("multiple {} candidates for `{}` \
                                            found",
                                           flavor,
                                           self.crate_name).as_slice());
                self.sess.span_note(self.span,
                                    format!(r"candidate #1: {}",
                                            ret.get_ref()
                                               .display()).as_slice());
                error = 1;
                ret = None;
            }
            if error > 0 {
                error += 1;
                self.sess.span_note(self.span,
                                    format!(r"candidate #{}: {}", error,
                                            lib.display()).as_slice());
                continue
            }
            *slot = Some(metadata);
            ret = Some(lib);
        }
        return if error > 0 {None} else {ret}
    }

    fn crate_matches(&mut self, crate_data: &[u8], libpath: &Path) -> bool {
        if self.should_match_name {
            match decoder::maybe_get_crate_name(crate_data) {
                Some(ref name) if self.crate_name == name.as_slice() => {}
                _ => { info!("Rejecting via crate name"); return false }
            }
        }
        let hash = match decoder::maybe_get_crate_hash(crate_data) {
            Some(hash) => hash, None => {
                info!("Rejecting via lack of crate hash");
                return false;
            }
        };

        let triple = match decoder::get_crate_triple(crate_data) {
            None => { debug!("triple not present"); return false }
            Some(t) => t,
        };
        if triple.as_slice() != self.triple {
            info!("Rejecting via crate triple: expected {} got {}", self.triple, triple);
            self.rejected_via_triple.push(CrateMismatch {
                path: libpath.clone(),
                got: triple.to_string()
            });
            return false;
        }

        match self.hash {
            None => true,
            Some(myhash) => {
                if *myhash != hash {
                    info!("Rejecting via hash: expected {} got {}", *myhash, hash);
                    self.rejected_via_hash.push(CrateMismatch {
                        path: libpath.clone(),
                        got: myhash.as_str().to_string()
                    });
                    false
                } else {
                    true
                }
            }
        }
    }


    // Returns the corresponding (prefix, suffix) that files need to have for
    // dynamic libraries
    fn dylibname(&self) -> Option<(&'static str, &'static str)> {
        match self.os {
            abi::OsWin32 => Some((WIN32_DLL_PREFIX, WIN32_DLL_SUFFIX)),
            abi::OsMacos => Some((MACOS_DLL_PREFIX, MACOS_DLL_SUFFIX)),
            abi::OsLinux => Some((LINUX_DLL_PREFIX, LINUX_DLL_SUFFIX)),
            abi::OsAndroid => Some((ANDROID_DLL_PREFIX, ANDROID_DLL_SUFFIX)),
            abi::OsFreebsd => Some((FREEBSD_DLL_PREFIX, FREEBSD_DLL_SUFFIX)),
            abi::OsDragonfly => Some((DRAGONFLY_DLL_PREFIX, DRAGONFLY_DLL_SUFFIX)),
            abi::OsiOS => None,
        }
    }

    fn find_commandline_library(&mut self) -> Option<Library> {
        let locs = match self.sess.opts.externs.find_equiv(&self.crate_name) {
            Some(s) => s,
            None => return None,
        };

        // First, filter out all libraries that look suspicious. We only accept
        // files which actually exist that have the correct naming scheme for
        // rlibs/dylibs.
        let sess = self.sess;
        let dylibname = self.dylibname();
        let mut locs = locs.iter().map(|l| Path::new(l.as_slice())).filter(|loc| {
            if !loc.exists() {
                sess.err(format!("extern location does not exist: {}",
                                 loc.display()).as_slice());
                return false;
            }
            let file = loc.filename_str().unwrap();
            if file.starts_with("lib") && file.ends_with(".rlib") {
                return true
            } else {
                match dylibname {
                    Some((prefix, suffix)) => {
                        if file.starts_with(prefix) && file.ends_with(suffix) {
                            return true
                        }
                    }
                    None => {}
                }
            }
            sess.err(format!("extern location is of an unknown type: {}",
                             loc.display()).as_slice());
            false
        });

        // Now that we have an itertor of good candidates, make sure there's at
        // most one rlib and at most one dylib.
        let mut rlibs = HashSet::new();
        let mut dylibs = HashSet::new();
        for loc in locs {
            if loc.filename_str().unwrap().ends_with(".rlib") {
                rlibs.insert(loc.clone());
            } else {
                dylibs.insert(loc.clone());
            }
        }

        // Extract the rlib/dylib pair.
        let mut metadata = None;
        let rlib = self.extract_one(rlibs, "rlib", &mut metadata);
        let dylib = self.extract_one(dylibs, "dylib", &mut metadata);

        if rlib.is_none() && dylib.is_none() { return None }
        match metadata {
            Some(metadata) => Some(Library {
                dylib: dylib,
                rlib: rlib,
                metadata: metadata,
            }),
            None => None,
        }
    }
}

pub fn note_crate_name(diag: &SpanHandler, name: &str) {
    diag.handler().note(format!("crate name: {}", name).as_slice());
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
            unsafe { mem::transmute(data) }
        };
        Some(ArchiveMetadata {
            _archive: ar,
            data: data,
        })
    }

    pub fn as_slice<'a>(&'a self) -> &'a [u8] { self.data }
}

// Just a small wrapper to time how long reading metadata takes.
fn get_metadata_section(os: abi::Os, filename: &Path) -> Result<MetadataBlob, String> {
    let start = time::precise_time_ns();
    let ret = get_metadata_section_imp(os, filename);
    info!("reading {} => {}ms", filename.filename_display(),
           (time::precise_time_ns() - start) / 1000000);
    return ret;
}

fn get_metadata_section_imp(os: abi::Os, filename: &Path) -> Result<MetadataBlob, String> {
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
            None => {
                return Err((format!("failed to read rlib metadata: '{}'",
                                    filename.display())))
            }
            Some(blob) => return Ok(blob)
        }
    }
    unsafe {
        let mb = filename.with_c_str(|buf| {
            llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(buf)
        });
        if mb as int == 0 {
            return Err(format!("error reading library: '{}'",
                               filename.display()))
        }
        let of = match ObjectFile::new(mb) {
            Some(of) => of,
            _ => {
                return Err((format!("provided path not an object file: '{}'",
                                    filename.display())))
            }
        };
        let si = mk_section_iter(of.llof);
        while llvm::LLVMIsSectionIteratorAtEnd(of.llof, si.llsi) == False {
            let mut name_buf = ptr::null();
            let name_len = llvm::LLVMRustGetSectionName(si.llsi, &mut name_buf);
            let name = string::raw::from_buf_len(name_buf as *const u8,
                                              name_len as uint);
            debug!("get_metadata_section: name {}", name);
            if read_meta_section_name(os).as_slice() == name.as_slice() {
                let cbuf = llvm::LLVMGetSectionContents(si.llsi);
                let csz = llvm::LLVMGetSectionSize(si.llsi) as uint;
                let mut found =
                    Err(format!("metadata not found: '{}'", filename.display()));
                let cvbuf: *const u8 = mem::transmute(cbuf);
                let vlen = encoder::metadata_encoding_version.len();
                debug!("checking {} bytes of metadata-version stamp",
                       vlen);
                let minsz = cmp::min(vlen, csz);
                let version_ok = slice::raw::buf_as_slice(cvbuf, minsz,
                    |buf0| buf0 == encoder::metadata_encoding_version);
                if !version_ok {
                    return Err((format!("incompatible metadata version found: '{}'",
                                        filename.display())));
                }

                let cvbuf1 = cvbuf.offset(vlen as int);
                debug!("inflating {} bytes of compressed metadata",
                       csz - vlen);
                slice::raw::buf_as_slice(cvbuf1, csz-vlen, |bytes| {
                    match flate::inflate_bytes(bytes) {
                        Some(inflated) => found = Ok(MetadataVec(inflated)),
                        None => {
                            found =
                                Err(format!("failed to decompress \
                                             metadata for: '{}'",
                                            filename.display()))
                        }
                    }
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

pub fn meta_section_name(os: abi::Os) -> Option<&'static str> {
    match os {
        abi::OsMacos => Some("__DATA,__note.rustc"),
        abi::OsiOS => Some("__DATA,__note.rustc"),
        abi::OsWin32 => Some(".note.rustc"),
        abi::OsLinux => Some(".note.rustc"),
        abi::OsAndroid => Some(".note.rustc"),
        abi::OsFreebsd => Some(".note.rustc"),
        abi::OsDragonfly => Some(".note.rustc"),
    }
}

pub fn read_meta_section_name(os: abi::Os) -> &'static str {
    match os {
        abi::OsMacos => "__note.rustc",
        abi::OsiOS => unreachable!(),
        abi::OsWin32 => ".note.rustc",
        abi::OsLinux => ".note.rustc",
        abi::OsAndroid => ".note.rustc",
        abi::OsFreebsd => ".note.rustc",
        abi::OsDragonfly => ".note.rustc"
    }
}

// A diagnostic function for dumping crate metadata to an output stream
pub fn list_file_metadata(os: abi::Os, path: &Path,
                          out: &mut io::Writer) -> io::IoResult<()> {
    match get_metadata_section(os, path) {
        Ok(bytes) => decoder::list_crate_metadata(bytes.as_slice(), out),
        Err(msg) => {
            write!(out, "{}\n", msg)
        }
    }
}
