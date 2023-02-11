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
//! This is a pretty tricky area of loading crates. Given a file, how do we know
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

use crate::creader::Library;
use crate::errors;
use crate::rmeta::{rustc_version, MetadataBlob, METADATA_HEADER};

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::owning_ref::OwningRef;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::MetadataRef;
use rustc_errors::{DiagnosticArgValue, FatalError, IntoDiagnosticArg};
use rustc_session::config::{self, CrateType};
use rustc_session::cstore::{CrateSource, MetadataLoader};
use rustc_session::filesearch::FileSearch;
use rustc_session::search_paths::PathKind;
use rustc_session::utils::CanonicalizedPath;
use rustc_session::Session;
use rustc_span::symbol::Symbol;
use rustc_span::Span;
use rustc_target::spec::{Target, TargetTriple};

use snap::read::FrameDecoder;
use std::borrow::Cow;
use std::io::{Read, Result as IoResult, Write};
use std::path::{Path, PathBuf};
use std::{cmp, fmt, fs};

#[derive(Clone)]
pub(crate) struct CrateLocator<'a> {
    // Immutable per-session configuration.
    only_needs_metadata: bool,
    sysroot: &'a Path,
    metadata_loader: &'a dyn MetadataLoader,

    // Immutable per-search configuration.
    crate_name: Symbol,
    exact_paths: Vec<CanonicalizedPath>,
    pub hash: Option<Svh>,
    extra_filename: Option<&'a str>,
    pub target: &'a Target,
    pub triple: TargetTriple,
    pub filesearch: FileSearch<'a>,
    pub is_proc_macro: bool,

    // Mutable in-progress state or output.
    crate_rejections: CrateRejections,
}

#[derive(Clone)]
pub(crate) struct CratePaths {
    name: Symbol,
    source: CrateSource,
}

impl CratePaths {
    pub(crate) fn new(name: Symbol, source: CrateSource) -> CratePaths {
        CratePaths { name, source }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum CrateFlavor {
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

impl IntoDiagnosticArg for CrateFlavor {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        match self {
            CrateFlavor::Rlib => DiagnosticArgValue::Str(Cow::Borrowed("rlib")),
            CrateFlavor::Rmeta => DiagnosticArgValue::Str(Cow::Borrowed("rmeta")),
            CrateFlavor::Dylib => DiagnosticArgValue::Str(Cow::Borrowed("dylib")),
        }
    }
}

impl<'a> CrateLocator<'a> {
    pub(crate) fn new(
        sess: &'a Session,
        metadata_loader: &'a dyn MetadataLoader,
        crate_name: Symbol,
        hash: Option<Svh>,
        extra_filename: Option<&'a str>,
        is_host: bool,
        path_kind: PathKind,
    ) -> CrateLocator<'a> {
        // The all loop is because `--crate-type=rlib --crate-type=rlib` is
        // legal and produces both inside this type.
        let is_rlib = sess.crate_types().iter().all(|c| *c == CrateType::Rlib);
        let needs_object_code = sess.opts.output_types.should_codegen();
        // If we're producing an rlib, then we don't need object code.
        // Or, if we're not producing object code, then we don't need it either
        // (e.g., if we're a cdylib but emitting just metadata).
        let only_needs_metadata = is_rlib || !needs_object_code;

        CrateLocator {
            only_needs_metadata,
            sysroot: &sess.sysroot,
            metadata_loader,
            crate_name,
            exact_paths: if hash.is_none() {
                sess.opts
                    .externs
                    .get(crate_name.as_str())
                    .into_iter()
                    .filter_map(|entry| entry.files())
                    .flatten()
                    .cloned()
                    .collect()
            } else {
                // SVH being specified means this is a transitive dependency,
                // so `--extern` options do not apply.
                Vec::new()
            },
            hash,
            extra_filename,
            target: if is_host { &sess.host } else { &sess.target },
            triple: if is_host {
                TargetTriple::from_triple(config::host_triple())
            } else {
                sess.opts.target_triple.clone()
            },
            filesearch: if is_host {
                sess.host_filesearch(path_kind)
            } else {
                sess.target_filesearch(path_kind)
            },
            is_proc_macro: false,
            crate_rejections: CrateRejections::default(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.crate_rejections.via_hash.clear();
        self.crate_rejections.via_triple.clear();
        self.crate_rejections.via_kind.clear();
        self.crate_rejections.via_version.clear();
        self.crate_rejections.via_filename.clear();
        self.crate_rejections.via_invalid.clear();
    }

    pub(crate) fn maybe_load_library_crate(&mut self) -> Result<Option<Library>, CrateError> {
        if !self.exact_paths.is_empty() {
            return self.find_commandline_library();
        }
        let mut seen_paths = FxHashSet::default();
        if let Some(extra_filename) = self.extra_filename {
            if let library @ Some(_) = self.find_library_crate(extra_filename, &mut seen_paths)? {
                return Ok(library);
            }
        }
        self.find_library_crate("", &mut seen_paths)
    }

    fn find_library_crate(
        &mut self,
        extra_prefix: &str,
        seen_paths: &mut FxHashSet<PathBuf>,
    ) -> Result<Option<Library>, CrateError> {
        let rmeta_prefix = &format!("lib{}{}", self.crate_name, extra_prefix);
        let rlib_prefix = rmeta_prefix;
        let dylib_prefix =
            &format!("{}{}{}", self.target.dll_prefix, self.crate_name, extra_prefix);
        let staticlib_prefix =
            &format!("{}{}{}", self.target.staticlib_prefix, self.crate_name, extra_prefix);

        let rmeta_suffix = ".rmeta";
        let rlib_suffix = ".rlib";
        let dylib_suffix = &self.target.dll_suffix;
        let staticlib_suffix = &self.target.staticlib_suffix;

        let mut candidates: FxHashMap<_, (FxHashMap<_, _>, FxHashMap<_, _>, FxHashMap<_, _>)> =
            Default::default();

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
        // Unfortunately, the prefix-based matching sometimes is over-eager.
        // E.g. if `rlib_suffix` is `libstd` it'll match the file
        // `libstd_detect-8d6701fb958915ad.rlib` (incorrect) as well as
        // `libstd-f3ab5b1dea981f17.rlib` (correct). But this is hard to avoid
        // given that `extra_filename` comes from the `-C extra-filename`
        // option and thus can be anything, and the incorrect match will be
        // handled safely in `extract_one`.
        for search_path in self.filesearch.search_paths() {
            debug!("searching {}", search_path.dir.display());
            for spf in search_path.files.iter() {
                debug!("testing {}", spf.path.display());

                let f = &spf.file_name_str;
                let (hash, kind) = if f.starts_with(rlib_prefix) && f.ends_with(rlib_suffix) {
                    (&f[rlib_prefix.len()..(f.len() - rlib_suffix.len())], CrateFlavor::Rlib)
                } else if f.starts_with(rmeta_prefix) && f.ends_with(rmeta_suffix) {
                    (&f[rmeta_prefix.len()..(f.len() - rmeta_suffix.len())], CrateFlavor::Rmeta)
                } else if f.starts_with(dylib_prefix) && f.ends_with(dylib_suffix.as_ref()) {
                    (&f[dylib_prefix.len()..(f.len() - dylib_suffix.len())], CrateFlavor::Dylib)
                } else {
                    if f.starts_with(staticlib_prefix) && f.ends_with(staticlib_suffix.as_ref()) {
                        self.crate_rejections.via_kind.push(CrateMismatch {
                            path: spf.path.clone(),
                            got: "static".to_string(),
                        });
                    }
                    continue;
                };

                info!("lib candidate: {}", spf.path.display());

                let (rlibs, rmetas, dylibs) = candidates.entry(hash.to_string()).or_default();
                let path = fs::canonicalize(&spf.path).unwrap_or_else(|_| spf.path.clone());
                if seen_paths.contains(&path) {
                    continue;
                };
                seen_paths.insert(path.clone());
                match kind {
                    CrateFlavor::Rlib => rlibs.insert(path, search_path.kind),
                    CrateFlavor::Rmeta => rmetas.insert(path, search_path.kind),
                    CrateFlavor::Dylib => dylibs.insert(path, search_path.kind),
                };
            }
        }

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
            if let Some((svh, lib)) = self.extract_lib(rlibs, rmetas, dylibs)? {
                libraries.insert(svh, lib);
            }
        }

        // Having now translated all relevant found hashes into libraries, see
        // what we've got and figure out if we found multiple candidates for
        // libraries or not.
        match libraries.len() {
            0 => Ok(None),
            1 => Ok(Some(libraries.into_iter().next().unwrap().1)),
            _ => {
                let mut libraries: Vec<_> = libraries.into_values().collect();

                libraries.sort_by_cached_key(|lib| lib.source.paths().next().unwrap().clone());
                let candidates = libraries
                    .iter()
                    .map(|lib| lib.source.paths().next().unwrap().clone())
                    .collect::<Vec<_>>();

                Err(CrateError::MultipleCandidates(
                    self.crate_name,
                    // these are the same for all candidates
                    get_flavor_from_path(candidates.first().unwrap()),
                    candidates,
                ))
            }
        }
    }

    fn extract_lib(
        &mut self,
        rlibs: FxHashMap<PathBuf, PathKind>,
        rmetas: FxHashMap<PathBuf, PathKind>,
        dylibs: FxHashMap<PathBuf, PathKind>,
    ) -> Result<Option<(Svh, Library)>, CrateError> {
        let mut slot = None;
        // Order here matters, rmeta should come first. See comment in
        // `extract_one` below.
        let source = CrateSource {
            rmeta: self.extract_one(rmetas, CrateFlavor::Rmeta, &mut slot)?,
            rlib: self.extract_one(rlibs, CrateFlavor::Rlib, &mut slot)?,
            dylib: self.extract_one(dylibs, CrateFlavor::Dylib, &mut slot)?,
        };
        Ok(slot.map(|(svh, metadata)| (svh, Library { source, metadata })))
    }

    fn needs_crate_flavor(&self, flavor: CrateFlavor) -> bool {
        if flavor == CrateFlavor::Dylib && self.is_proc_macro {
            return true;
        }

        if self.only_needs_metadata {
            flavor == CrateFlavor::Rmeta
        } else {
            // we need all flavors (perhaps not true, but what we do for now)
            true
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
    fn extract_one(
        &mut self,
        m: FxHashMap<PathBuf, PathKind>,
        flavor: CrateFlavor,
        slot: &mut Option<(Svh, MetadataBlob)>,
    ) -> Result<Option<(PathBuf, PathKind)>, CrateError> {
        // If we are producing an rlib, and we've already loaded metadata, then
        // we should not attempt to discover further crate sources (unless we're
        // locating a proc macro; exact logic is in needs_crate_flavor). This means
        // that under -Zbinary-dep-depinfo we will not emit a dependency edge on
        // the *unused* rlib, and by returning `None` here immediately we
        // guarantee that we do indeed not use it.
        //
        // See also #68149 which provides more detail on why emitting the
        // dependency on the rlib is a bad thing.
        //
        // We currently do not verify that these other sources are even in sync,
        // and this is arguably a bug (see #10786), but because reading metadata
        // is quite slow (especially from dylibs) we currently do not read it
        // from the other crate sources.
        if slot.is_some() {
            if m.is_empty() || !self.needs_crate_flavor(flavor) {
                return Ok(None);
            } else if m.len() == 1 {
                return Ok(Some(m.into_iter().next().unwrap()));
            }
        }

        let mut ret: Option<(PathBuf, PathKind)> = None;
        let mut err_data: Option<Vec<PathBuf>> = None;
        for (lib, kind) in m {
            info!("{} reading metadata from: {}", flavor, lib.display());
            if flavor == CrateFlavor::Rmeta && lib.metadata().map_or(false, |m| m.len() == 0) {
                // Empty files will cause get_metadata_section to fail. Rmeta
                // files can be empty, for example with binaries (which can
                // often appear with `cargo check` when checking a library as
                // a unittest). We don't want to emit a user-visible warning
                // in this case as it is not a real problem.
                debug!("skipping empty file");
                continue;
            }
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
                    Err(MetadataError::LoadFailure(err)) => {
                        info!("no metadata found: {}", err);
                        // The file was present and created by the same compiler version, but we
                        // couldn't load it for some reason. Give a hard error instead of silently
                        // ignoring it, but only if we would have given an error anyway.
                        self.crate_rejections
                            .via_invalid
                            .push(CrateMismatch { path: lib, got: err });
                        continue;
                    }
                    Err(err @ MetadataError::NotPresent(_)) => {
                        info!("no metadata found: {}", err);
                        continue;
                    }
                };
            // If we see multiple hashes, emit an error about duplicate candidates.
            if slot.as_ref().map_or(false, |s| s.0 != hash) {
                if let Some(candidates) = err_data {
                    return Err(CrateError::MultipleCandidates(
                        self.crate_name,
                        flavor,
                        candidates,
                    ));
                }
                err_data = Some(vec![ret.as_ref().unwrap().0.clone()]);
                *slot = None;
            }
            if let Some(candidates) = &mut err_data {
                candidates.push(lib);
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
            if let Some((prev, _)) = &ret {
                let sysroot = self.sysroot;
                let sysroot = sysroot.canonicalize().unwrap_or_else(|_| sysroot.to_path_buf());
                if prev.starts_with(&sysroot) {
                    continue;
                }
            }
            *slot = Some((hash, metadata));
            ret = Some((lib, kind));
        }

        if let Some(candidates) = err_data {
            Err(CrateError::MultipleCandidates(self.crate_name, flavor, candidates))
        } else {
            Ok(ret)
        }
    }

    fn crate_matches(&mut self, metadata: &MetadataBlob, libpath: &Path) -> Option<Svh> {
        let rustc_version = rustc_version();
        let found_version = metadata.get_rustc_version();
        if found_version != rustc_version {
            info!("Rejecting via version: expected {} got {}", rustc_version, found_version);
            self.crate_rejections
                .via_version
                .push(CrateMismatch { path: libpath.to_path_buf(), got: found_version });
            return None;
        }

        let root = metadata.get_root();
        if root.is_proc_macro_crate() != self.is_proc_macro {
            info!(
                "Rejecting via proc macro: expected {} got {}",
                self.is_proc_macro,
                root.is_proc_macro_crate(),
            );
            return None;
        }

        if self.exact_paths.is_empty() && self.crate_name != root.name() {
            info!("Rejecting via crate name");
            return None;
        }

        if root.triple() != &self.triple {
            info!("Rejecting via crate triple: expected {} got {}", self.triple, root.triple());
            self.crate_rejections.via_triple.push(CrateMismatch {
                path: libpath.to_path_buf(),
                got: root.triple().to_string(),
            });
            return None;
        }

        let hash = root.hash();
        if let Some(expected_hash) = self.hash {
            if hash != expected_hash {
                info!("Rejecting via hash: expected {} got {}", expected_hash, hash);
                self.crate_rejections
                    .via_hash
                    .push(CrateMismatch { path: libpath.to_path_buf(), got: hash.to_string() });
                return None;
            }
        }

        Some(hash)
    }

    fn find_commandline_library(&mut self) -> Result<Option<Library>, CrateError> {
        // First, filter out all libraries that look suspicious. We only accept
        // files which actually exist that have the correct naming scheme for
        // rlibs/dylibs.
        let mut rlibs = FxHashMap::default();
        let mut rmetas = FxHashMap::default();
        let mut dylibs = FxHashMap::default();
        for loc in &self.exact_paths {
            if !loc.canonicalized().exists() {
                return Err(CrateError::ExternLocationNotExist(
                    self.crate_name,
                    loc.original().clone(),
                ));
            }
            if !loc.original().is_file() {
                return Err(CrateError::ExternLocationNotFile(
                    self.crate_name,
                    loc.original().clone(),
                ));
            }
            let Some(file) = loc.original().file_name().and_then(|s| s.to_str()) else {
                return Err(CrateError::ExternLocationNotFile(
                    self.crate_name,
                    loc.original().clone(),
                ));
            };

            if file.starts_with("lib") && (file.ends_with(".rlib") || file.ends_with(".rmeta"))
                || file.starts_with(self.target.dll_prefix.as_ref())
                    && file.ends_with(self.target.dll_suffix.as_ref())
            {
                // Make sure there's at most one rlib and at most one dylib.
                // Note to take care and match against the non-canonicalized name:
                // some systems save build artifacts into content-addressed stores
                // that do not preserve extensions, and then link to them using
                // e.g. symbolic links. If we canonicalize too early, we resolve
                // the symlink, the file type is lost and we might treat rlibs and
                // rmetas as dylibs.
                let loc_canon = loc.canonicalized().clone();
                let loc = loc.original();
                if loc.file_name().unwrap().to_str().unwrap().ends_with(".rlib") {
                    rlibs.insert(loc_canon, PathKind::ExternFlag);
                } else if loc.file_name().unwrap().to_str().unwrap().ends_with(".rmeta") {
                    rmetas.insert(loc_canon, PathKind::ExternFlag);
                } else {
                    dylibs.insert(loc_canon, PathKind::ExternFlag);
                }
            } else {
                self.crate_rejections
                    .via_filename
                    .push(CrateMismatch { path: loc.original().clone(), got: String::new() });
            }
        }

        // Extract the dylib/rlib/rmeta triple.
        Ok(self.extract_lib(rlibs, rmetas, dylibs)?.map(|(_, lib)| lib))
    }

    pub(crate) fn into_error(self, root: Option<CratePaths>) -> CrateError {
        CrateError::LocatorCombined(CombinedLocatorError {
            crate_name: self.crate_name,
            root,
            triple: self.triple,
            dll_prefix: self.target.dll_prefix.to_string(),
            dll_suffix: self.target.dll_suffix.to_string(),
            crate_rejections: self.crate_rejections,
        })
    }
}

fn get_metadata_section<'p>(
    target: &Target,
    flavor: CrateFlavor,
    filename: &'p Path,
    loader: &dyn MetadataLoader,
) -> Result<MetadataBlob, MetadataError<'p>> {
    if !filename.exists() {
        return Err(MetadataError::NotPresent(filename));
    }
    let raw_bytes: MetadataRef = match flavor {
        CrateFlavor::Rlib => {
            loader.get_rlib_metadata(target, filename).map_err(MetadataError::LoadFailure)?
        }
        CrateFlavor::Dylib => {
            let buf =
                loader.get_dylib_metadata(target, filename).map_err(MetadataError::LoadFailure)?;
            // The header is uncompressed
            let header_len = METADATA_HEADER.len();
            debug!("checking {} bytes of metadata-version stamp", header_len);
            let header = &buf[..cmp::min(header_len, buf.len())];
            if header != METADATA_HEADER {
                return Err(MetadataError::LoadFailure(format!(
                    "invalid metadata version found: {}",
                    filename.display()
                )));
            }

            // Header is okay -> inflate the actual metadata
            let compressed_bytes = &buf[header_len..];
            debug!("inflating {} bytes of compressed metadata", compressed_bytes.len());
            // Assume the decompressed data will be at least the size of the compressed data, so we
            // don't have to grow the buffer as much.
            let mut inflated = Vec::with_capacity(compressed_bytes.len());
            match FrameDecoder::new(compressed_bytes).read_to_end(&mut inflated) {
                Ok(_) => rustc_erase_owner!(OwningRef::new(inflated).map_owner_box()),
                Err(_) => {
                    return Err(MetadataError::LoadFailure(format!(
                        "failed to decompress metadata: {}",
                        filename.display()
                    )));
                }
            }
        }
        CrateFlavor::Rmeta => {
            // mmap the file, because only a small fraction of it is read.
            let file = std::fs::File::open(filename).map_err(|_| {
                MetadataError::LoadFailure(format!(
                    "failed to open rmeta metadata: '{}'",
                    filename.display()
                ))
            })?;
            let mmap = unsafe { Mmap::map(file) };
            let mmap = mmap.map_err(|_| {
                MetadataError::LoadFailure(format!(
                    "failed to mmap rmeta metadata: '{}'",
                    filename.display()
                ))
            })?;

            rustc_erase_owner!(OwningRef::new(mmap).map_owner_box())
        }
    };
    let blob = MetadataBlob::new(raw_bytes);
    if blob.is_compatible() {
        Ok(blob)
    } else {
        Err(MetadataError::LoadFailure(format!(
            "invalid metadata version found: {}",
            filename.display()
        )))
    }
}

/// Look for a plugin registrar. Returns its library path and crate disambiguator.
pub fn find_plugin_registrar(
    sess: &Session,
    metadata_loader: &dyn MetadataLoader,
    span: Span,
    name: Symbol,
) -> PathBuf {
    find_plugin_registrar_impl(sess, metadata_loader, name).unwrap_or_else(|err| {
        // `core` is always available if we got as far as loading plugins.
        err.report(sess, span, false);
        FatalError.raise()
    })
}

fn find_plugin_registrar_impl<'a>(
    sess: &'a Session,
    metadata_loader: &dyn MetadataLoader,
    name: Symbol,
) -> Result<PathBuf, CrateError> {
    info!("find plugin registrar `{}`", name);
    let mut locator = CrateLocator::new(
        sess,
        metadata_loader,
        name,
        None, // hash
        None, // extra_filename
        true, // is_host
        PathKind::Crate,
    );

    match locator.maybe_load_library_crate()? {
        Some(library) => match library.source.dylib {
            Some(dylib) => Ok(dylib.0),
            None => Err(CrateError::NonDylibPlugin(name)),
        },
        None => Err(locator.into_error(None)),
    }
}

/// A diagnostic function for dumping crate metadata to an output stream.
pub fn list_file_metadata(
    target: &Target,
    path: &Path,
    metadata_loader: &dyn MetadataLoader,
    out: &mut dyn Write,
) -> IoResult<()> {
    let flavor = get_flavor_from_path(path);
    match get_metadata_section(target, flavor, path, metadata_loader) {
        Ok(metadata) => metadata.list_crate_metadata(out),
        Err(msg) => write!(out, "{}\n", msg),
    }
}

fn get_flavor_from_path(path: &Path) -> CrateFlavor {
    let filename = path.file_name().unwrap().to_str().unwrap();

    if filename.ends_with(".rlib") {
        CrateFlavor::Rlib
    } else if filename.ends_with(".rmeta") {
        CrateFlavor::Rmeta
    } else {
        CrateFlavor::Dylib
    }
}

// ------------------------------------------ Error reporting -------------------------------------

#[derive(Clone)]
struct CrateMismatch {
    path: PathBuf,
    got: String,
}

#[derive(Clone, Default)]
struct CrateRejections {
    via_hash: Vec<CrateMismatch>,
    via_triple: Vec<CrateMismatch>,
    via_kind: Vec<CrateMismatch>,
    via_version: Vec<CrateMismatch>,
    via_filename: Vec<CrateMismatch>,
    via_invalid: Vec<CrateMismatch>,
}

/// Candidate rejection reasons collected during crate search.
/// If no candidate is accepted, then these reasons are presented to the user,
/// otherwise they are ignored.
pub(crate) struct CombinedLocatorError {
    crate_name: Symbol,
    root: Option<CratePaths>,
    triple: TargetTriple,
    dll_prefix: String,
    dll_suffix: String,
    crate_rejections: CrateRejections,
}

pub(crate) enum CrateError {
    NonAsciiName(Symbol),
    ExternLocationNotExist(Symbol, PathBuf),
    ExternLocationNotFile(Symbol, PathBuf),
    MultipleCandidates(Symbol, CrateFlavor, Vec<PathBuf>),
    SymbolConflictsCurrent(Symbol),
    StableCrateIdCollision(Symbol, Symbol),
    DlOpen(String),
    DlSym(String),
    LocatorCombined(CombinedLocatorError),
    NonDylibPlugin(Symbol),
}

enum MetadataError<'a> {
    /// The file was missing.
    NotPresent(&'a Path),
    /// The file was present and invalid.
    LoadFailure(String),
}

impl fmt::Display for MetadataError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetadataError::NotPresent(filename) => {
                f.write_str(&format!("no such file: '{}'", filename.display()))
            }
            MetadataError::LoadFailure(msg) => f.write_str(msg),
        }
    }
}

impl CrateError {
    pub(crate) fn report(self, sess: &Session, span: Span, missing_core: bool) {
        match self {
            CrateError::NonAsciiName(crate_name) => {
                sess.emit_err(errors::NonAsciiName { span, crate_name });
            }
            CrateError::ExternLocationNotExist(crate_name, loc) => {
                sess.emit_err(errors::ExternLocationNotExist { span, crate_name, location: &loc });
            }
            CrateError::ExternLocationNotFile(crate_name, loc) => {
                sess.emit_err(errors::ExternLocationNotFile { span, crate_name, location: &loc });
            }
            CrateError::MultipleCandidates(crate_name, flavor, candidates) => {
                sess.emit_err(errors::MultipleCandidates { span, crate_name, flavor, candidates });
            }
            CrateError::SymbolConflictsCurrent(root_name) => {
                sess.emit_err(errors::SymbolConflictsCurrent { span, crate_name: root_name });
            }
            CrateError::StableCrateIdCollision(crate_name0, crate_name1) => {
                sess.emit_err(errors::StableCrateIdCollision { span, crate_name0, crate_name1 });
            }
            CrateError::DlOpen(s) | CrateError::DlSym(s) => {
                sess.emit_err(errors::DlError { span, err: s });
            }
            CrateError::LocatorCombined(locator) => {
                let crate_name = locator.crate_name;
                let add_info = match &locator.root {
                    None => String::new(),
                    Some(r) => format!(" which `{}` depends on", r.name),
                };
                if !locator.crate_rejections.via_filename.is_empty() {
                    let mismatches = locator.crate_rejections.via_filename.iter();
                    for CrateMismatch { path, .. } in mismatches {
                        sess.emit_err(errors::CrateLocationUnknownType {
                            span,
                            path: &path,
                            crate_name,
                        });
                        sess.emit_err(errors::LibFilenameForm {
                            span,
                            dll_prefix: &locator.dll_prefix,
                            dll_suffix: &locator.dll_suffix,
                        });
                    }
                }
                let mut found_crates = String::new();
                if !locator.crate_rejections.via_hash.is_empty() {
                    let mismatches = locator.crate_rejections.via_hash.iter();
                    for CrateMismatch { path, .. } in mismatches {
                        found_crates.push_str(&format!(
                            "\ncrate `{}`: {}",
                            crate_name,
                            path.display()
                        ));
                    }
                    if let Some(r) = locator.root {
                        for path in r.source.paths() {
                            found_crates.push_str(&format!(
                                "\ncrate `{}`: {}",
                                r.name,
                                path.display()
                            ));
                        }
                    }
                    sess.emit_err(errors::NewerCrateVersion {
                        span,
                        crate_name: crate_name,
                        add_info,
                        found_crates,
                    });
                } else if !locator.crate_rejections.via_triple.is_empty() {
                    let mismatches = locator.crate_rejections.via_triple.iter();
                    for CrateMismatch { path, got } in mismatches {
                        found_crates.push_str(&format!(
                            "\ncrate `{}`, target triple {}: {}",
                            crate_name,
                            got,
                            path.display(),
                        ));
                    }
                    sess.emit_err(errors::NoCrateWithTriple {
                        span,
                        crate_name,
                        locator_triple: locator.triple.triple(),
                        add_info,
                        found_crates,
                    });
                } else if !locator.crate_rejections.via_kind.is_empty() {
                    let mismatches = locator.crate_rejections.via_kind.iter();
                    for CrateMismatch { path, .. } in mismatches {
                        found_crates.push_str(&format!(
                            "\ncrate `{}`: {}",
                            crate_name,
                            path.display()
                        ));
                    }
                    sess.emit_err(errors::FoundStaticlib {
                        span,
                        crate_name,
                        add_info,
                        found_crates,
                    });
                } else if !locator.crate_rejections.via_version.is_empty() {
                    let mismatches = locator.crate_rejections.via_version.iter();
                    for CrateMismatch { path, got } in mismatches {
                        found_crates.push_str(&format!(
                            "\ncrate `{}` compiled by {}: {}",
                            crate_name,
                            got,
                            path.display(),
                        ));
                    }
                    sess.emit_err(errors::IncompatibleRustc {
                        span,
                        crate_name,
                        add_info,
                        found_crates,
                        rustc_version: rustc_version(),
                    });
                } else if !locator.crate_rejections.via_invalid.is_empty() {
                    let mut crate_rejections = Vec::new();
                    for CrateMismatch { path: _, got } in locator.crate_rejections.via_invalid {
                        crate_rejections.push(got);
                    }
                    sess.emit_err(errors::InvalidMetadataFiles {
                        span,
                        crate_name,
                        add_info,
                        crate_rejections,
                    });
                } else {
                    sess.emit_err(errors::CannotFindCrate {
                        span,
                        crate_name,
                        add_info,
                        missing_core,
                        current_crate: sess
                            .opts
                            .crate_name
                            .clone()
                            .unwrap_or("<unknown>".to_string()),
                        is_nightly_build: sess.is_nightly_build(),
                        profiler_runtime: Symbol::intern(&sess.opts.unstable_opts.profiler_runtime),
                        locator_triple: locator.triple,
                    });
                }
            }
            CrateError::NonDylibPlugin(crate_name) => {
                sess.emit_err(errors::NoDylibPlugin { span, crate_name });
            }
        }
    }
}
