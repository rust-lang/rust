// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
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
//! version requirements, dependency graphs, conflicting desires, and fun! This
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
//! If the file answers `yes` to all these questions, then the file is
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
//! ```text
//! --extern crate-name=path/to/the/crate.rlib
//! ```
//!
//! This flag is basically the following letter to the compiler:
//!
//! > Dear rustc,
//! >
//! > When you are attempting to load the immediate dependency `crate-name`, I
//! > would like you to assume that the library is located at
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
//! ```text
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
//! one that also needs to be dealt with for transitive dependencies. Note that
//! in the letter above `--extern` flags only apply to the *local* set of
//! dependencies, not the upstream transitive dependencies. Consider this
//! dependency graph:
//!
//! ```text
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

use cstore::{MetadataBlob, MetadataVec, MetadataArchive};
use decoder;
use encoder;

use rustc::back::svh::Svh;
use rustc::session::Session;
use rustc::session::filesearch::{FileSearch, FileMatches, FileDoesntMatch};
use rustc::session::search_paths::PathKind;
use rustc::util::common;

use rustc_llvm as llvm;
use rustc_llvm::{False, ObjectFile, mk_section_iter};
use rustc_llvm::archive_ro::ArchiveRO;
use syntax::codemap::Span;
use syntax::errors::Handler;
use rustc_back::target::Target;

use std::cmp;
use std::collections::HashMap;
use std::fs;
use std::io::prelude::*;
use std::io;
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;
use std::time::Instant;

use flate;

pub struct CrateMismatch {
    path: PathBuf,
    got: String,
}

pub struct Context<'a> {
    pub sess: &'a Session,
    pub span: Span,
    pub ident: &'a str,
    pub crate_name: &'a str,
    pub hash: Option<&'a Svh>,
    // points to either self.sess.target.target or self.sess.host, must match triple
    pub target: &'a Target,
    pub triple: &'a str,
    pub filesearch: FileSearch<'a>,
    pub root: &'a Option<CratePaths>,
    pub rejected_via_hash: Vec<CrateMismatch>,
    pub rejected_via_triple: Vec<CrateMismatch>,
    pub rejected_via_kind: Vec<CrateMismatch>,
    pub should_match_name: bool,
}

pub struct Library {
    pub dylib: Option<(PathBuf, PathKind)>,
    pub rlib: Option<(PathBuf, PathKind)>,
    pub metadata: MetadataBlob,
}

pub struct ArchiveMetadata {
    _archive: ArchiveRO,
    // points into self._archive
    data: *const [u8],
}

pub struct CratePaths {
    pub ident: String,
    pub dylib: Option<PathBuf>,
    pub rlib: Option<PathBuf>
}

pub const METADATA_FILENAME: &'static str = "rust.metadata.bin";

impl CratePaths {
    fn paths(&self) -> Vec<PathBuf> {
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
        let add = match self.root {
            &None => String::new(),
            &Some(ref r) => format!(" which `{}` depends on",
                                    r.ident)
        };
        if !self.rejected_via_hash.is_empty() {
            span_err!(self.sess, self.span, E0460,
                      "found possibly newer version of crate `{}`{}",
                      self.ident, add);
        } else if !self.rejected_via_triple.is_empty() {
            span_err!(self.sess, self.span, E0461,
                      "couldn't find crate `{}` with expected target triple {}{}",
                      self.ident, self.triple, add);
        } else if !self.rejected_via_kind.is_empty() {
            span_err!(self.sess, self.span, E0462,
                      "found staticlib `{}` instead of rlib or dylib{}",
                      self.ident, add);
        } else {
            span_err!(self.sess, self.span, E0463,
                      "can't find crate for `{}`{}",
                      self.ident, add);
        }

        if !self.rejected_via_triple.is_empty() {
            let mismatches = self.rejected_via_triple.iter();
            for (i, &CrateMismatch{ ref path, ref got }) in mismatches.enumerate() {
                self.sess.fileline_note(self.span,
                    &format!("crate `{}`, path #{}, triple {}: {}",
                            self.ident, i+1, got, path.display()));
            }
        }
        if !self.rejected_via_hash.is_empty() {
            self.sess.span_note(self.span, "perhaps this crate needs \
                                            to be recompiled?");
            let mismatches = self.rejected_via_hash.iter();
            for (i, &CrateMismatch{ ref path, .. }) in mismatches.enumerate() {
                self.sess.fileline_note(self.span,
                    &format!("crate `{}` path #{}: {}",
                            self.ident, i+1, path.display()));
            }
            match self.root {
                &None => {}
                &Some(ref r) => {
                    for (i, path) in r.paths().iter().enumerate() {
                        self.sess.fileline_note(self.span,
                            &format!("crate `{}` path #{}: {}",
                                    r.ident, i+1, path.display()));
                    }
                }
            }
        }
        if !self.rejected_via_kind.is_empty() {
            self.sess.fileline_help(self.span, "please recompile this crate using \
                                            --crate-type lib");
            let mismatches = self.rejected_via_kind.iter();
            for (i, &CrateMismatch { ref path, .. }) in mismatches.enumerate() {
                self.sess.fileline_note(self.span,
                                        &format!("crate `{}` path #{}: {}",
                                                 self.ident, i+1, path.display()));
            }
        }
        self.sess.abort_if_errors();
    }

    fn find_library_crate(&mut self) -> Option<Library> {
        // If an SVH is specified, then this is a transitive dependency that
        // must be loaded via -L plus some filtering.
        if self.hash.is_none() {
            self.should_match_name = false;
            if let Some(s) = self.sess.opts.externs.get(self.crate_name) {
                return self.find_commandline_library(s);
            }
            self.should_match_name = true;
        }

        let dypair = self.dylibname();

        // want: crate_name.dir_part() + prefix + crate_name.file_part + "-"
        let dylib_prefix = format!("{}{}", dypair.0, self.crate_name);
        let rlib_prefix = format!("lib{}", self.crate_name);
        let staticlib_prefix = format!("lib{}", self.crate_name);

        let mut candidates = HashMap::new();
        let mut staticlibs = vec!();

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
        self.filesearch.search(|path, kind| {
            let file = match path.file_name().and_then(|s| s.to_str()) {
                None => return FileDoesntMatch,
                Some(file) => file,
            };
            let (hash, rlib) = if file.starts_with(&rlib_prefix[..]) &&
                                  file.ends_with(".rlib") {
                (&file[(rlib_prefix.len()) .. (file.len() - ".rlib".len())],
                 true)
            } else if file.starts_with(&dylib_prefix) &&
                      file.ends_with(&dypair.1) {
                (&file[(dylib_prefix.len()) .. (file.len() - dypair.1.len())],
                 false)
            } else {
                if file.starts_with(&staticlib_prefix[..]) &&
                   file.ends_with(".a") {
                    staticlibs.push(CrateMismatch {
                        path: path.to_path_buf(),
                        got: "static".to_string()
                    });
                }
                return FileDoesntMatch
            };
            info!("lib candidate: {}", path.display());

            let hash_str = hash.to_string();
            let slot = candidates.entry(hash_str)
                                 .or_insert_with(|| (HashMap::new(), HashMap::new()));
            let (ref mut rlibs, ref mut dylibs) = *slot;
            fs::canonicalize(path).map(|p| {
                if rlib {
                    rlibs.insert(p, kind);
                } else {
                    dylibs.insert(p, kind);
                }
                FileMatches
            }).unwrap_or(FileDoesntMatch)
        });
        self.rejected_via_kind.extend(staticlibs);

        // We have now collected all known libraries into a set of candidates
        // keyed of the filename hash listed. For each filename, we also have a
        // list of rlibs/dylibs that apply. Here, we map each of these lists
        // (per hash), to a Library candidate for returning.
        //
        // A Library candidate is created if the metadata for the set of
        // libraries corresponds to the crate id and hash criteria that this
        // search is being performed for.
        let mut libraries = Vec::new();
        for (_hash, (rlibs, dylibs)) in candidates {
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
            1 => Some(libraries.into_iter().next().unwrap()),
            _ => {
                span_err!(self.sess, self.span, E0464,
                          "multiple matching crates for `{}`",
                          self.crate_name);
                self.sess.note("candidates:");
                for lib in &libraries {
                    match lib.dylib {
                        Some((ref p, _)) => {
                            self.sess.note(&format!("path: {}",
                                                   p.display()));
                        }
                        None => {}
                    }
                    match lib.rlib {
                        Some((ref p, _)) => {
                            self.sess.note(&format!("path: {}",
                                                    p.display()));
                        }
                        None => {}
                    }
                    let data = lib.metadata.as_slice();
                    let name = decoder::get_crate_name(data);
                    note_crate_name(self.sess.diagnostic(), &name);
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
    fn extract_one(&mut self, m: HashMap<PathBuf, PathKind>, flavor: &str,
                   slot: &mut Option<MetadataBlob>) -> Option<(PathBuf, PathKind)> {
        let mut ret = None::<(PathBuf, PathKind)>;
        let mut error = 0;

        if slot.is_some() {
            // FIXME(#10786): for an optimization, we only read one of the
            //                library's metadata sections. In theory we should
            //                read both, but reading dylib metadata is quite
            //                slow.
            if m.is_empty() {
                return None
            } else if m.len() == 1 {
                return Some(m.into_iter().next().unwrap())
            }
        }

        for (lib, kind) in m {
            info!("{} reading metadata from: {}", flavor, lib.display());
            let metadata = match get_metadata_section(self.target, &lib) {
                Ok(blob) => {
                    if self.crate_matches(blob.as_slice(), &lib) {
                        blob
                    } else {
                        info!("metadata mismatch");
                        continue
                    }
                }
                Err(err) => {
                    info!("no metadata found: {}", err);
                    continue
                }
            };
            // If we've already found a candidate and we're not matching hashes,
            // emit an error about duplicate candidates found. If we're matching
            // based on a hash, however, then if we've gotten this far both
            // candidates have the same hash, so they're not actually
            // duplicates that we should warn about.
            if ret.is_some() && self.hash.is_none() {
                span_err!(self.sess, self.span, E0465,
                          "multiple {} candidates for `{}` found",
                          flavor, self.crate_name);
                self.sess.span_note(self.span,
                                    &format!(r"candidate #1: {}",
                                            ret.as_ref().unwrap().0
                                               .display()));
                error = 1;
                ret = None;
            }
            if error > 0 {
                error += 1;
                self.sess.span_note(self.span,
                                    &format!(r"candidate #{}: {}", error,
                                            lib.display()));
                continue
            }
            *slot = Some(metadata);
            ret = Some((lib, kind));
        }
        return if error > 0 {None} else {ret}
    }

    fn crate_matches(&mut self, crate_data: &[u8], libpath: &Path) -> bool {
        if self.should_match_name {
            match decoder::maybe_get_crate_name(crate_data) {
                Some(ref name) if self.crate_name == *name => {}
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
        if triple != self.triple {
            info!("Rejecting via crate triple: expected {} got {}", self.triple, triple);
            self.rejected_via_triple.push(CrateMismatch {
                path: libpath.to_path_buf(),
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
                        path: libpath.to_path_buf(),
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
    fn dylibname(&self) -> (String, String) {
        let t = &self.target;
        (t.options.dll_prefix.clone(), t.options.dll_suffix.clone())
    }

    fn find_commandline_library(&mut self, locs: &[String]) -> Option<Library> {
        // First, filter out all libraries that look suspicious. We only accept
        // files which actually exist that have the correct naming scheme for
        // rlibs/dylibs.
        let sess = self.sess;
        let dylibname = self.dylibname();
        let mut rlibs = HashMap::new();
        let mut dylibs = HashMap::new();
        {
            let locs = locs.iter().map(|l| PathBuf::from(l)).filter(|loc| {
                if !loc.exists() {
                    sess.err(&format!("extern location for {} does not exist: {}",
                                     self.crate_name, loc.display()));
                    return false;
                }
                let file = match loc.file_name().and_then(|s| s.to_str()) {
                    Some(file) => file,
                    None => {
                        sess.err(&format!("extern location for {} is not a file: {}",
                                         self.crate_name, loc.display()));
                        return false;
                    }
                };
                if file.starts_with("lib") && file.ends_with(".rlib") {
                    return true
                } else {
                    let (ref prefix, ref suffix) = dylibname;
                    if file.starts_with(&prefix[..]) &&
                       file.ends_with(&suffix[..]) {
                        return true
                    }
                }
                sess.err(&format!("extern location for {} is of an unknown type: {}",
                                 self.crate_name, loc.display()));
                sess.help(&format!("file name should be lib*.rlib or {}*.{}",
                                   dylibname.0, dylibname.1));
                false
            });

            // Now that we have an iterator of good candidates, make sure
            // there's at most one rlib and at most one dylib.
            for loc in locs {
                if loc.file_name().unwrap().to_str().unwrap().ends_with(".rlib") {
                    rlibs.insert(fs::canonicalize(&loc).unwrap(),
                                 PathKind::ExternFlag);
                } else {
                    dylibs.insert(fs::canonicalize(&loc).unwrap(),
                                  PathKind::ExternFlag);
                }
            }
        };

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

pub fn note_crate_name(diag: &Handler, name: &str) {
    diag.note(&format!("crate name: {}", name));
}

impl ArchiveMetadata {
    fn new(ar: ArchiveRO) -> Option<ArchiveMetadata> {
        let data = {
            let section = ar.iter().find(|sect| {
                sect.name() == Some(METADATA_FILENAME)
            });
            match section {
                Some(s) => s.data() as *const [u8],
                None => {
                    debug!("didn't find '{}' in the archive", METADATA_FILENAME);
                    return None;
                }
            }
        };

        Some(ArchiveMetadata {
            _archive: ar,
            data: data,
        })
    }

    pub fn as_slice<'a>(&'a self) -> &'a [u8] { unsafe { &*self.data } }
}

// Just a small wrapper to time how long reading metadata takes.
fn get_metadata_section(target: &Target, filename: &Path)
                        -> Result<MetadataBlob, String> {
    let start = Instant::now();
    let ret = get_metadata_section_imp(target, filename);
    info!("reading {:?} => {:?}", filename.file_name().unwrap(),
          start.elapsed());
    return ret
}

fn get_metadata_section_imp(target: &Target, filename: &Path)
                            -> Result<MetadataBlob, String> {
    if !filename.exists() {
        return Err(format!("no such file: '{}'", filename.display()));
    }
    if filename.file_name().unwrap().to_str().unwrap().ends_with(".rlib") {
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
            None => Err(format!("failed to read rlib metadata: '{}'",
                                filename.display())),
            Some(blob) => Ok(blob)
        };
    }
    unsafe {
        let buf = common::path2cstr(filename);
        let mb = llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(buf.as_ptr());
        if mb as isize == 0 {
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
            let name = slice::from_raw_parts(name_buf as *const u8,
                                             name_len as usize).to_vec();
            let name = String::from_utf8(name).unwrap();
            debug!("get_metadata_section: name {}", name);
            if read_meta_section_name(target) == name {
                let cbuf = llvm::LLVMGetSectionContents(si.llsi);
                let csz = llvm::LLVMGetSectionSize(si.llsi) as usize;
                let cvbuf: *const u8 = cbuf as *const u8;
                let vlen = encoder::metadata_encoding_version.len();
                debug!("checking {} bytes of metadata-version stamp",
                       vlen);
                let minsz = cmp::min(vlen, csz);
                let buf0 = slice::from_raw_parts(cvbuf, minsz);
                let version_ok = buf0 == encoder::metadata_encoding_version;
                if !version_ok {
                    return Err((format!("incompatible metadata version found: '{}'",
                                        filename.display())));
                }

                let cvbuf1 = cvbuf.offset(vlen as isize);
                debug!("inflating {} bytes of compressed metadata",
                       csz - vlen);
                let bytes = slice::from_raw_parts(cvbuf1, csz - vlen);
                match flate::inflate_bytes(bytes) {
                    Ok(inflated) => return Ok(MetadataVec(inflated)),
                    Err(_) => {}
                }
            }
            llvm::LLVMMoveToNextSection(si.llsi);
        }
        Err(format!("metadata not found: '{}'", filename.display()))
    }
}

pub fn meta_section_name(target: &Target) -> &'static str {
    if target.options.is_like_osx {
        "__DATA,__note.rustc"
    } else if target.options.is_like_msvc {
        // When using link.exe it was seen that the section name `.note.rustc`
        // was getting shortened to `.note.ru`, and according to the PE and COFF
        // specification:
        //
        // > Executable images do not use a string table and do not support
        // > section names longer than 8 characters
        //
        // https://msdn.microsoft.com/en-us/library/windows/hardware/gg463119.aspx
        //
        // As a result, we choose a slightly shorter name! As to why
        // `.note.rustc` works on MinGW, that's another good question...
        ".rustc"
    } else {
        ".note.rustc"
    }
}

pub fn read_meta_section_name(target: &Target) -> &'static str {
    if target.options.is_like_osx {
        "__note.rustc"
    } else if target.options.is_like_msvc {
        ".rustc"
    } else {
        ".note.rustc"
    }
}

// A diagnostic function for dumping crate metadata to an output stream
pub fn list_file_metadata(target: &Target, path: &Path,
                          out: &mut io::Write) -> io::Result<()> {
    match get_metadata_section(target, path) {
        Ok(bytes) => decoder::list_crate_metadata(bytes.as_slice(), out),
        Err(msg) => {
            write!(out, "{}\n", msg)
        }
    }
}
