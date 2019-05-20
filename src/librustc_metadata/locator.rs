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
//! that we can always type-check/translate any function, we have to have
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
//! A third kind of dependency is an rmeta file. These are metadata files and do
//! not contain any code, etc. To a first approximation, these are treated in the
//! same way as rlibs. Where there is both an rlib and an rmeta file, the rlib
//! gets priority (even if the rmeta file is newer). An rmeta file is only
//! useful for checking a downstream crate, attempting to link one will cause an
//! error.
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
//!    This is filtering for files like `libfoo*.rlib` and such. If the crate
//!    we're looking for was originally compiled with -C extra-filename, the
//!    extra filename will be included in this prefix to reduce reading
//!    metadata from crates that would otherwise share our prefix.
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
//! ```compile_fail,E0463
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
//! metadata::locator or metadata::creader for all the juicy details!

use crate::cstore::{MetadataRef, MetadataBlob};
use crate::creader::Library;
use crate::schema::{METADATA_HEADER, rustc_version};

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::svh::Svh;
use rustc::middle::cstore::MetadataLoader;
use rustc::session::{config, Session};
use rustc::session::filesearch::{FileSearch, FileMatches, FileDoesntMatch};
use rustc::session::search_paths::PathKind;
use rustc::util::nodemap::FxHashMap;

use errors::DiagnosticBuilder;
use syntax::symbol::{Symbol, sym};
use syntax::struct_span_err;
use syntax_pos::Span;
use rustc_target::spec::{Target, TargetTriple};

use std::cmp;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::time::Instant;

use flate2::read::DeflateDecoder;

use rustc_data_structures::owning_ref::OwningRef;

use log::{debug, info, warn};

#[derive(Clone)]
pub struct CrateMismatch {
    path: PathBuf,
    got: String,
}

#[derive(Clone)]
pub struct Context<'a> {
    pub sess: &'a Session,
    pub span: Span,
    pub ident: Symbol,
    pub crate_name: Symbol,
    pub hash: Option<&'a Svh>,
    pub extra_filename: Option<&'a str>,
    // points to either self.sess.target.target or self.sess.host, must match triple
    pub target: &'a Target,
    pub triple: TargetTriple,
    pub filesearch: FileSearch<'a>,
    pub root: &'a Option<CratePaths>,
    pub rejected_via_hash: Vec<CrateMismatch>,
    pub rejected_via_triple: Vec<CrateMismatch>,
    pub rejected_via_kind: Vec<CrateMismatch>,
    pub rejected_via_version: Vec<CrateMismatch>,
    pub rejected_via_filename: Vec<CrateMismatch>,
    pub should_match_name: bool,
    pub is_proc_macro: Option<bool>,
    pub metadata_loader: &'a dyn MetadataLoader,
}

pub struct CratePaths {
    pub ident: String,
    pub dylib: Option<PathBuf>,
    pub rlib: Option<PathBuf>,
    pub rmeta: Option<PathBuf>,
}

#[derive(Copy, Clone, PartialEq)]
enum CrateFlavor {
    Rlib,
    Rmeta,
    Dylib,
}

impl fmt::Display for CrateFlavor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            CrateFlavor::Rlib => "rlib",
            CrateFlavor::Rmeta => "rmeta",
            CrateFlavor::Dylib => "dylib",
        })
    }
}

impl CratePaths {
    fn paths(&self) -> Vec<PathBuf> {
        self.dylib.iter().chain(self.rlib.iter()).chain(self.rmeta.iter()).cloned().collect()
    }
}

impl<'a> Context<'a> {
    pub fn reset(&mut self) {
        self.rejected_via_hash.clear();
        self.rejected_via_triple.clear();
        self.rejected_via_kind.clear();
        self.rejected_via_version.clear();
        self.rejected_via_filename.clear();
    }

    pub fn maybe_load_library_crate(&mut self) -> Option<Library> {
        let mut seen_paths = FxHashSet::default();
        match self.extra_filename {
            Some(s) => self.find_library_crate(s, &mut seen_paths)
                .or_else(|| self.find_library_crate("", &mut seen_paths)),
            None => self.find_library_crate("", &mut seen_paths)
        }
    }

    pub fn report_errs(self) -> ! {
        let add = match self.root {
            &None => String::new(),
            &Some(ref r) => format!(" which `{}` depends on", r.ident),
        };
        let mut msg = "the following crate versions were found:".to_string();
        let mut err = if !self.rejected_via_hash.is_empty() {
            let mut err = struct_span_err!(self.sess,
                                           self.span,
                                           E0460,
                                           "found possibly newer version of crate `{}`{}",
                                           self.ident,
                                           add);
            err.note("perhaps that crate needs to be recompiled?");
            let mismatches = self.rejected_via_hash.iter();
            for &CrateMismatch { ref path, .. } in mismatches {
                msg.push_str(&format!("\ncrate `{}`: {}", self.ident, path.display()));
            }
            match self.root {
                &None => {}
                &Some(ref r) => {
                    for path in r.paths().iter() {
                        msg.push_str(&format!("\ncrate `{}`: {}", r.ident, path.display()));
                    }
                }
            }
            err.note(&msg);
            err
        } else if !self.rejected_via_triple.is_empty() {
            let mut err = struct_span_err!(self.sess,
                                           self.span,
                                           E0461,
                                           "couldn't find crate `{}` \
                                            with expected target triple {}{}",
                                           self.ident,
                                           self.triple,
                                           add);
            let mismatches = self.rejected_via_triple.iter();
            for &CrateMismatch { ref path, ref got } in mismatches {
                msg.push_str(&format!("\ncrate `{}`, target triple {}: {}",
                                      self.ident,
                                      got,
                                      path.display()));
            }
            err.note(&msg);
            err
        } else if !self.rejected_via_kind.is_empty() {
            let mut err = struct_span_err!(self.sess,
                                           self.span,
                                           E0462,
                                           "found staticlib `{}` instead of rlib or dylib{}",
                                           self.ident,
                                           add);
            err.help("please recompile that crate using --crate-type lib");
            let mismatches = self.rejected_via_kind.iter();
            for &CrateMismatch { ref path, .. } in mismatches {
                msg.push_str(&format!("\ncrate `{}`: {}", self.ident, path.display()));
            }
            err.note(&msg);
            err
        } else if !self.rejected_via_version.is_empty() {
            let mut err = struct_span_err!(self.sess,
                                           self.span,
                                           E0514,
                                           "found crate `{}` compiled by an incompatible version \
                                            of rustc{}",
                                           self.ident,
                                           add);
            err.help(&format!("please recompile that crate using this compiler ({})",
                              rustc_version()));
            let mismatches = self.rejected_via_version.iter();
            for &CrateMismatch { ref path, ref got } in mismatches {
                msg.push_str(&format!("\ncrate `{}` compiled by {}: {}",
                                      self.ident,
                                      got,
                                      path.display()));
            }
            err.note(&msg);
            err
        } else {
            let mut err = struct_span_err!(self.sess,
                                           self.span,
                                           E0463,
                                           "can't find crate for `{}`{}",
                                           self.ident,
                                           add);

            if (self.ident == sym::std || self.ident == sym::core)
                && self.triple != TargetTriple::from_triple(config::host_triple()) {
                err.note(&format!("the `{}` target may not be installed", self.triple));
            }
            err.span_label(self.span, "can't find crate");
            err
        };

        if !self.rejected_via_filename.is_empty() {
            let dylibname = self.dylibname();
            let mismatches = self.rejected_via_filename.iter();
            for &CrateMismatch { ref path, .. } in mismatches {
                err.note(&format!("extern location for {} is of an unknown type: {}",
                                  self.crate_name,
                                  path.display()))
                   .help(&format!("file name should be lib*.rlib or {}*.{}",
                                  dylibname.0,
                                  dylibname.1));
            }
        }

        err.emit();
        self.sess.abort_if_errors();
        unreachable!();
    }

    fn find_library_crate(&mut self,
                          extra_prefix: &str,
                          seen_paths: &mut FxHashSet<PathBuf>)
                          -> Option<Library> {
        // If an SVH is specified, then this is a transitive dependency that
        // must be loaded via -L plus some filtering.
        if self.hash.is_none() {
            self.should_match_name = false;
            if let Some(entry) = self.sess.opts.externs.get(&self.crate_name.as_str()) {
                // Only use `--extern crate_name=path` here, not `--extern crate_name`.
                if entry.locations.iter().any(|l| l.is_some()) {
                    return self.find_commandline_library(
                        entry.locations.iter().filter_map(|l| l.as_ref()),
                    );
                }
            }
            self.should_match_name = true;
        }

        let dypair = self.dylibname();
        let staticpair = self.staticlibname();

        // want: crate_name.dir_part() + prefix + crate_name.file_part + "-"
        let dylib_prefix = format!("{}{}{}", dypair.0, self.crate_name, extra_prefix);
        let rlib_prefix = format!("lib{}{}", self.crate_name, extra_prefix);
        let staticlib_prefix = format!("{}{}{}", staticpair.0, self.crate_name, extra_prefix);

        let mut candidates: FxHashMap<
            _,
            (FxHashMap<_, _>, FxHashMap<_, _>, FxHashMap<_, _>),
        > = Default::default();
        let mut staticlibs = vec![];

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
            let (hash, found_kind) =
                if file.starts_with(&rlib_prefix) && file.ends_with(".rlib") {
                    (&file[(rlib_prefix.len())..(file.len() - ".rlib".len())], CrateFlavor::Rlib)
                } else if file.starts_with(&rlib_prefix) && file.ends_with(".rmeta") {
                    (&file[(rlib_prefix.len())..(file.len() - ".rmeta".len())], CrateFlavor::Rmeta)
                } else if file.starts_with(&dylib_prefix) &&
                                             file.ends_with(&dypair.1) {
                    (&file[(dylib_prefix.len())..(file.len() - dypair.1.len())], CrateFlavor::Dylib)
                } else {
                    if file.starts_with(&staticlib_prefix) && file.ends_with(&staticpair.1) {
                        staticlibs.push(CrateMismatch {
                            path: path.to_path_buf(),
                            got: "static".to_string(),
                        });
                    }
                    return FileDoesntMatch;
                };

            info!("lib candidate: {}", path.display());

            let hash_str = hash.to_string();
            let slot = candidates.entry(hash_str).or_default();
            let (ref mut rlibs, ref mut rmetas, ref mut dylibs) = *slot;
            fs::canonicalize(path)
                .map(|p| {
                    if seen_paths.contains(&p) {
                        return FileDoesntMatch
                    };
                    seen_paths.insert(p.clone());
                    match found_kind {
                        CrateFlavor::Rlib => { rlibs.insert(p, kind); }
                        CrateFlavor::Rmeta => { rmetas.insert(p, kind); }
                        CrateFlavor::Dylib => { dylibs.insert(p, kind); }
                    }
                    FileMatches
                })
                .unwrap_or(FileDoesntMatch)
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
        let mut libraries = FxHashMap::default();
        for (_hash, (rlibs, rmetas, dylibs)) in candidates {
            let mut slot = None;
            let rlib = self.extract_one(rlibs, CrateFlavor::Rlib, &mut slot);
            let rmeta = self.extract_one(rmetas, CrateFlavor::Rmeta, &mut slot);
            let dylib = self.extract_one(dylibs, CrateFlavor::Dylib, &mut slot);
            if let Some((h, m)) = slot {
                libraries.insert(h,
                                 Library {
                                     dylib,
                                     rlib,
                                     rmeta,
                                     metadata: m,
                                 });
            }
        }

        // Having now translated all relevant found hashes into libraries, see
        // what we've got and figure out if we found multiple candidates for
        // libraries or not.
        match libraries.len() {
            0 => None,
            1 => Some(libraries.into_iter().next().unwrap().1),
            _ => {
                let mut err = struct_span_err!(self.sess,
                                               self.span,
                                               E0464,
                                               "multiple matching crates for `{}`",
                                               self.crate_name);
                let candidates = libraries.iter().filter_map(|(_, lib)| {
                    let crate_name = &lib.metadata.get_root().name.as_str();
                    match &(&lib.dylib, &lib.rlib) {
                        &(&Some((ref pd, _)), &Some((ref pr, _))) => {
                            Some(format!("\ncrate `{}`: {}\n{:>padding$}",
                                         crate_name,
                                         pd.display(),
                                         pr.display(),
                                         padding=8 + crate_name.len()))
                        }
                        &(&Some((ref p, _)), &None) | &(&None, &Some((ref p, _))) => {
                            Some(format!("\ncrate `{}`: {}", crate_name, p.display()))
                        }
                        &(&None, &None) => None,
                    }
                }).collect::<String>();
                err.note(&format!("candidates:{}", candidates));
                err.emit();
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
    fn extract_one(&mut self,
                   m: FxHashMap<PathBuf, PathKind>,
                   flavor: CrateFlavor,
                   slot: &mut Option<(Svh, MetadataBlob)>)
                   -> Option<(PathBuf, PathKind)> {
        let mut ret: Option<(PathBuf, PathKind)> = None;
        let mut error = 0;

        if slot.is_some() {
            // FIXME(#10786): for an optimization, we only read one of the
            //                libraries' metadata sections. In theory we should
            //                read both, but reading dylib metadata is quite
            //                slow.
            if m.is_empty() {
                return None;
            } else if m.len() == 1 {
                return Some(m.into_iter().next().unwrap());
            }
        }

        let mut err: Option<DiagnosticBuilder<'_>> = None;
        for (lib, kind) in m {
            info!("{} reading metadata from: {}", flavor, lib.display());
            let (hash, metadata) =
                match get_metadata_section(self.target, flavor, &lib, self.metadata_loader) {
                    Ok(blob) => {
                        if let Some(h) = self.crate_matches(&blob, &lib) {
                            (h, blob)
                        } else {
                            info!("metadata mismatch");
                            continue;
                        }
                    }
                    Err(err) => {
                        warn!("no metadata found: {}", err);
                        continue;
                    }
                };
            // If we see multiple hashes, emit an error about duplicate candidates.
            if slot.as_ref().map_or(false, |s| s.0 != hash) {
                let mut e = struct_span_err!(self.sess,
                                             self.span,
                                             E0465,
                                             "multiple {} candidates for `{}` found",
                                             flavor,
                                             self.crate_name);
                e.span_note(self.span,
                            &format!(r"candidate #1: {}",
                                     ret.as_ref()
                                         .unwrap()
                                         .0
                                         .display()));
                if let Some(ref mut e) = err {
                    e.emit();
                }
                err = Some(e);
                error = 1;
                *slot = None;
            }
            if error > 0 {
                error += 1;
                err.as_mut().unwrap().span_note(self.span,
                                                &format!(r"candidate #{}: {}",
                                                         error,
                                                         lib.display()));
                continue;
            }

            // Ok so at this point we've determined that `(lib, kind)` above is
            // a candidate crate to load, and that `slot` is either none (this
            // is the first crate of its kind) or if some the previous path has
            // the exact same hash (e.g., it's the exact same crate).
            //
            // In principle these two candidate crates are exactly the same so
            // we can choose either of them to link. As a stupidly gross hack,
            // however, we favor crate in the sysroot.
            //
            // You can find more info in rust-lang/rust#39518 and various linked
            // issues, but the general gist is that during testing libstd the
            // compilers has two candidates to choose from: one in the sysroot
            // and one in the deps folder. These two crates are the exact same
            // crate but if the compiler chooses the one in the deps folder
            // it'll cause spurious errors on Windows.
            //
            // As a result, we favor the sysroot crate here. Note that the
            // candidates are all canonicalized, so we canonicalize the sysroot
            // as well.
            if let Some((ref prev, _)) = ret {
                let sysroot = &self.sess.sysroot;
                let sysroot = sysroot.canonicalize()
                                     .unwrap_or_else(|_| sysroot.to_path_buf());
                if prev.starts_with(&sysroot) {
                    continue
                }
            }
            *slot = Some((hash, metadata));
            ret = Some((lib, kind));
        }

        if error > 0 {
            err.unwrap().emit();
            None
        } else {
            ret
        }
    }

    fn crate_matches(&mut self, metadata: &MetadataBlob, libpath: &Path) -> Option<Svh> {
        let rustc_version = rustc_version();
        let found_version = metadata.get_rustc_version();
        if found_version != rustc_version {
            info!("Rejecting via version: expected {} got {}",
                  rustc_version,
                  found_version);
            self.rejected_via_version.push(CrateMismatch {
                path: libpath.to_path_buf(),
                got: found_version,
            });
            return None;
        }

        let root = metadata.get_root();
        if let Some(is_proc_macro) = self.is_proc_macro {
            if root.proc_macro_decls_static.is_some() != is_proc_macro {
                return None;
            }
        }

        if self.should_match_name {
            if self.crate_name != root.name {
                info!("Rejecting via crate name");
                return None;
            }
        }

        if root.triple != self.triple {
            info!("Rejecting via crate triple: expected {} got {}",
                  self.triple,
                  root.triple);
            self.rejected_via_triple.push(CrateMismatch {
                path: libpath.to_path_buf(),
                got: root.triple.to_string(),
            });
            return None;
        }

        if let Some(myhash) = self.hash {
            if *myhash != root.hash {
                info!("Rejecting via hash: expected {} got {}", *myhash, root.hash);
                self.rejected_via_hash.push(CrateMismatch {
                    path: libpath.to_path_buf(),
                    got: myhash.to_string(),
                });
                return None;
            }
        }

        Some(root.hash)
    }


    // Returns the corresponding (prefix, suffix) that files need to have for
    // dynamic libraries
    fn dylibname(&self) -> (String, String) {
        let t = &self.target;
        (t.options.dll_prefix.clone(), t.options.dll_suffix.clone())
    }

    // Returns the corresponding (prefix, suffix) that files need to have for
    // static libraries
    fn staticlibname(&self) -> (String, String) {
        let t = &self.target;
        (t.options.staticlib_prefix.clone(), t.options.staticlib_suffix.clone())
    }

    fn find_commandline_library<'b, LOCS>(&mut self, locs: LOCS) -> Option<Library>
        where LOCS: Iterator<Item = &'b String>
    {
        // First, filter out all libraries that look suspicious. We only accept
        // files which actually exist that have the correct naming scheme for
        // rlibs/dylibs.
        let sess = self.sess;
        let dylibname = self.dylibname();
        let mut rlibs = FxHashMap::default();
        let mut rmetas = FxHashMap::default();
        let mut dylibs = FxHashMap::default();
        {
            let locs = locs.map(|l| PathBuf::from(l)).filter(|loc| {
                if !loc.exists() {
                    sess.err(&format!("extern location for {} does not exist: {}",
                                      self.crate_name,
                                      loc.display()));
                    return false;
                }
                let file = match loc.file_name().and_then(|s| s.to_str()) {
                    Some(file) => file,
                    None => {
                        sess.err(&format!("extern location for {} is not a file: {}",
                                          self.crate_name,
                                          loc.display()));
                        return false;
                    }
                };
                if file.starts_with("lib") &&
                   (file.ends_with(".rlib") || file.ends_with(".rmeta")) {
                    return true;
                } else {
                    let (ref prefix, ref suffix) = dylibname;
                    if file.starts_with(&prefix[..]) && file.ends_with(&suffix[..]) {
                        return true;
                    }
                }

                self.rejected_via_filename.push(CrateMismatch {
                    path: loc.clone(),
                    got: String::new(),
                });

                false
            });

            // Now that we have an iterator of good candidates, make sure
            // there's at most one rlib and at most one dylib.
            for loc in locs {
                if loc.file_name().unwrap().to_str().unwrap().ends_with(".rlib") {
                    rlibs.insert(fs::canonicalize(&loc).unwrap(), PathKind::ExternFlag);
                } else if loc.file_name().unwrap().to_str().unwrap().ends_with(".rmeta") {
                    rmetas.insert(fs::canonicalize(&loc).unwrap(), PathKind::ExternFlag);
                } else {
                    dylibs.insert(fs::canonicalize(&loc).unwrap(), PathKind::ExternFlag);
                }
            }
        };

        // Extract the rlib/dylib pair.
        let mut slot = None;
        let rlib = self.extract_one(rlibs, CrateFlavor::Rlib, &mut slot);
        let rmeta = self.extract_one(rmetas, CrateFlavor::Rmeta, &mut slot);
        let dylib = self.extract_one(dylibs, CrateFlavor::Dylib, &mut slot);

        if rlib.is_none() && rmeta.is_none() && dylib.is_none() {
            return None;
        }
        slot.map(|(_, metadata)|
            Library {
                dylib,
                rlib,
                rmeta,
                metadata,
            }
        )
    }
}

// Just a small wrapper to time how long reading metadata takes.
fn get_metadata_section(target: &Target,
                        flavor: CrateFlavor,
                        filename: &Path,
                        loader: &dyn MetadataLoader)
                        -> Result<MetadataBlob, String> {
    let start = Instant::now();
    let ret = get_metadata_section_imp(target, flavor, filename, loader);
    info!("reading {:?} => {:?}",
          filename.file_name().unwrap(),
          start.elapsed());
    return ret;
}

/// A trivial wrapper for `Mmap` that implements `StableDeref`.
struct StableDerefMmap(memmap::Mmap);

impl Deref for StableDerefMmap {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.0.deref()
    }
}

unsafe impl stable_deref_trait::StableDeref for StableDerefMmap {}

fn get_metadata_section_imp(target: &Target,
                            flavor: CrateFlavor,
                            filename: &Path,
                            loader: &dyn MetadataLoader)
                            -> Result<MetadataBlob, String> {
    if !filename.exists() {
        return Err(format!("no such file: '{}'", filename.display()));
    }
    let raw_bytes: MetadataRef = match flavor {
        CrateFlavor::Rlib => loader.get_rlib_metadata(target, filename)?,
        CrateFlavor::Dylib => {
            let buf = loader.get_dylib_metadata(target, filename)?;
            // The header is uncompressed
            let header_len = METADATA_HEADER.len();
            debug!("checking {} bytes of metadata-version stamp", header_len);
            let header = &buf[..cmp::min(header_len, buf.len())];
            if header != METADATA_HEADER {
                return Err(format!("incompatible metadata version found: '{}'",
                                   filename.display()));
            }

            // Header is okay -> inflate the actual metadata
            let compressed_bytes = &buf[header_len..];
            debug!("inflating {} bytes of compressed metadata", compressed_bytes.len());
            let mut inflated = Vec::new();
            match DeflateDecoder::new(compressed_bytes).read_to_end(&mut inflated) {
                Ok(_) => {
                    rustc_erase_owner!(OwningRef::new(inflated).map_owner_box())
                }
                Err(_) => {
                    return Err(format!("failed to decompress metadata: {}", filename.display()));
                }
            }
        }
        CrateFlavor::Rmeta => {
            // mmap the file, because only a small fraction of it is read.
            let file = std::fs::File::open(filename).map_err(|_|
                format!("failed to open rmeta metadata: '{}'", filename.display()))?;
            let mmap = unsafe { memmap::Mmap::map(&file) };
            let mmap = mmap.map_err(|_|
                format!("failed to mmap rmeta metadata: '{}'", filename.display()))?;

            rustc_erase_owner!(OwningRef::new(StableDerefMmap(mmap)).map_owner_box())
        }
    };
    let blob = MetadataBlob(raw_bytes);
    if blob.is_compatible() {
        Ok(blob)
    } else {
        Err(format!("incompatible metadata version found: '{}'", filename.display()))
    }
}

/// A diagnostic function for dumping crate metadata to an output stream.
pub fn list_file_metadata(target: &Target,
                          path: &Path,
                          loader: &dyn MetadataLoader,
                          out: &mut dyn io::Write)
                          -> io::Result<()> {
    let filename = path.file_name().unwrap().to_str().unwrap();
    let flavor = if filename.ends_with(".rlib") {
        CrateFlavor::Rlib
    } else if filename.ends_with(".rmeta") {
        CrateFlavor::Rmeta
    } else {
        CrateFlavor::Dylib
    };
    match get_metadata_section(target, flavor, path, loader) {
        Ok(metadata) => metadata.list_crate_metadata(out),
        Err(msg) => write!(out, "{}\n", msg),
    }
}
