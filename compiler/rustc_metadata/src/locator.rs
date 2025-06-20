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

use std::borrow::Cow;
use std::io::{Result as IoResult, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::{cmp, fmt};

use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::owned_slice::{OwnedSlice, slice_owned};
use rustc_data_structures::svh::Svh;
use rustc_errors::{DiagArgValue, IntoDiagArg};
use rustc_fs_util::try_canonicalize;
use rustc_session::Session;
use rustc_session::cstore::CrateSource;
use rustc_session::filesearch::FileSearch;
use rustc_session::search_paths::PathKind;
use rustc_session::utils::CanonicalizedPath;
use rustc_span::{Span, Symbol};
use rustc_target::spec::{Target, TargetTuple};
use tempfile::Builder as TempFileBuilder;
use tracing::{debug, info};

use crate::creader::{Library, MetadataLoader};
use crate::errors;
use crate::rmeta::{METADATA_HEADER, MetadataBlob, rustc_version};

#[derive(Clone)]
pub(crate) struct CrateLocator<'a> {
    // Immutable per-session configuration.
    only_needs_metadata: bool,
    sysroot: &'a Path,
    metadata_loader: &'a dyn MetadataLoader,
    cfg_version: &'static str,

    // Immutable per-search configuration.
    crate_name: Symbol,
    exact_paths: Vec<CanonicalizedPath>,
    pub hash: Option<Svh>,
    extra_filename: Option<&'a str>,
    pub target: &'a Target,
    pub tuple: TargetTuple,
    pub filesearch: &'a FileSearch,
    pub is_proc_macro: bool,

    pub path_kind: PathKind,
    // Mutable in-progress state or output.
    crate_rejections: CrateRejections,
}

#[derive(Clone, Debug)]
pub(crate) struct CratePaths {
    pub(crate) name: Symbol,
    source: CrateSource,
}

impl CratePaths {
    pub(crate) fn new(name: Symbol, source: CrateSource) -> CratePaths {
        CratePaths { name, source }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum CrateFlavor {
    Rlib,
    Rmeta,
    Dylib,
    SDylib,
}

impl fmt::Display for CrateFlavor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            CrateFlavor::Rlib => "rlib",
            CrateFlavor::Rmeta => "rmeta",
            CrateFlavor::Dylib => "dylib",
            CrateFlavor::SDylib => "sdylib",
        })
    }
}

impl IntoDiagArg for CrateFlavor {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        match self {
            CrateFlavor::Rlib => DiagArgValue::Str(Cow::Borrowed("rlib")),
            CrateFlavor::Rmeta => DiagArgValue::Str(Cow::Borrowed("rmeta")),
            CrateFlavor::Dylib => DiagArgValue::Str(Cow::Borrowed("dylib")),
            CrateFlavor::SDylib => DiagArgValue::Str(Cow::Borrowed("sdylib")),
        }
    }
}

impl<'a> CrateLocator<'a> {
    pub(crate) fn new(
        sess: &'a Session,
        metadata_loader: &'a dyn MetadataLoader,
        crate_name: Symbol,
        is_rlib: bool,
        hash: Option<Svh>,
        extra_filename: Option<&'a str>,
        path_kind: PathKind,
    ) -> CrateLocator<'a> {
        let needs_object_code = sess.opts.output_types.should_codegen();
        // If we're producing an rlib, then we don't need object code.
        // Or, if we're not producing object code, then we don't need it either
        // (e.g., if we're a cdylib but emitting just metadata).
        let only_needs_metadata = is_rlib || !needs_object_code;

        CrateLocator {
            only_needs_metadata,
            sysroot: sess.opts.sysroot.path(),
            metadata_loader,
            cfg_version: sess.cfg_version,
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
            target: &sess.target,
            tuple: sess.opts.target_triple.clone(),
            filesearch: sess.target_filesearch(),
            path_kind,
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
        let interface_prefix = rmeta_prefix;

        let rmeta_suffix = ".rmeta";
        let rlib_suffix = ".rlib";
        let dylib_suffix = &self.target.dll_suffix;
        let staticlib_suffix = &self.target.staticlib_suffix;
        let interface_suffix = ".rs";

        let mut candidates: FxIndexMap<
            _,
            (FxIndexMap<_, _>, FxIndexMap<_, _>, FxIndexMap<_, _>, FxIndexMap<_, _>),
        > = Default::default();

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
        for search_path in self.filesearch.search_paths(self.path_kind) {
            debug!("searching {}", search_path.dir.display());
            let spf = &search_path.files;

            let mut should_check_staticlibs = true;
            for (prefix, suffix, kind) in [
                (rlib_prefix.as_str(), rlib_suffix, CrateFlavor::Rlib),
                (rmeta_prefix.as_str(), rmeta_suffix, CrateFlavor::Rmeta),
                (dylib_prefix, dylib_suffix, CrateFlavor::Dylib),
                (interface_prefix, interface_suffix, CrateFlavor::SDylib),
            ] {
                if prefix == staticlib_prefix && suffix == staticlib_suffix {
                    should_check_staticlibs = false;
                }
                if let Some(matches) = spf.query(prefix, suffix) {
                    for (hash, spf) in matches {
                        info!("lib candidate: {}", spf.path.display());

                        let (rlibs, rmetas, dylibs, interfaces) =
                            candidates.entry(hash).or_default();
                        {
                            // As a perforamnce optimisation we canonicalize the path and skip
                            // ones we've already seeen. This allows us to ignore crates
                            // we know are exactual equal to ones we've already found.
                            // Going to the same crate through different symlinks does not change the result.
                            let path = try_canonicalize(&spf.path)
                                .unwrap_or_else(|_| spf.path.to_path_buf());
                            if seen_paths.contains(&path) {
                                continue;
                            };
                            seen_paths.insert(path);
                        }
                        // Use the original path (potentially with unresolved symlinks),
                        // filesystem code should not care, but this is nicer for diagnostics.
                        let path = spf.path.to_path_buf();
                        match kind {
                            CrateFlavor::Rlib => rlibs.insert(path, search_path.kind),
                            CrateFlavor::Rmeta => rmetas.insert(path, search_path.kind),
                            CrateFlavor::Dylib => dylibs.insert(path, search_path.kind),
                            CrateFlavor::SDylib => interfaces.insert(path, search_path.kind),
                        };
                    }
                }
            }
            if let Some(static_matches) = should_check_staticlibs
                .then(|| spf.query(staticlib_prefix, staticlib_suffix))
                .flatten()
            {
                for (_, spf) in static_matches {
                    self.crate_rejections.via_kind.push(CrateMismatch {
                        path: spf.path.to_path_buf(),
                        got: "static".to_string(),
                    });
                }
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
        let mut libraries = FxIndexMap::default();
        for (_hash, (rlibs, rmetas, dylibs, interfaces)) in candidates {
            if let Some((svh, lib)) = self.extract_lib(rlibs, rmetas, dylibs, interfaces)? {
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
        rlibs: FxIndexMap<PathBuf, PathKind>,
        rmetas: FxIndexMap<PathBuf, PathKind>,
        dylibs: FxIndexMap<PathBuf, PathKind>,
        interfaces: FxIndexMap<PathBuf, PathKind>,
    ) -> Result<Option<(Svh, Library)>, CrateError> {
        let mut slot = None;
        // Order here matters, rmeta should come first.
        //
        // Make sure there's at most one rlib and at most one dylib.
        //
        // See comment in `extract_one` below.
        let rmeta = self.extract_one(rmetas, CrateFlavor::Rmeta, &mut slot)?;
        let rlib = self.extract_one(rlibs, CrateFlavor::Rlib, &mut slot)?;
        let sdylib_interface = self.extract_one(interfaces, CrateFlavor::SDylib, &mut slot)?;
        let dylib = self.extract_one(dylibs, CrateFlavor::Dylib, &mut slot)?;

        if sdylib_interface.is_some() && dylib.is_none() {
            return Err(CrateError::FullMetadataNotFound(self.crate_name, CrateFlavor::SDylib));
        }

        let source = CrateSource { rmeta, rlib, dylib, sdylib_interface };
        Ok(slot.map(|(svh, metadata, _, _)| (svh, Library { source, metadata })))
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
    //
    // The `PathBuf` in `slot` will only be used for diagnostic purposes.
    fn extract_one(
        &mut self,
        m: FxIndexMap<PathBuf, PathKind>,
        flavor: CrateFlavor,
        slot: &mut Option<(Svh, MetadataBlob, PathBuf, CrateFlavor)>,
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
        if slot.is_some() {
            if m.is_empty() || !self.needs_crate_flavor(flavor) {
                return Ok(None);
            }
        }

        let mut ret: Option<(PathBuf, PathKind)> = None;
        let mut err_data: Option<Vec<PathBuf>> = None;
        for (lib, kind) in m {
            info!("{} reading metadata from: {}", flavor, lib.display());
            if flavor == CrateFlavor::Rmeta && lib.metadata().is_ok_and(|m| m.len() == 0) {
                // Empty files will cause get_metadata_section to fail. Rmeta
                // files can be empty, for example with binaries (which can
                // often appear with `cargo check` when checking a library as
                // a unittest). We don't want to emit a user-visible warning
                // in this case as it is not a real problem.
                debug!("skipping empty file");
                continue;
            }
            let (hash, metadata) = match get_metadata_section(
                self.target,
                flavor,
                &lib,
                self.metadata_loader,
                self.cfg_version,
                Some(self.crate_name),
            ) {
                Ok(blob) => {
                    if let Some(h) = self.crate_matches(&blob, &lib) {
                        (h, blob)
                    } else {
                        info!("metadata mismatch");
                        continue;
                    }
                }
                Err(MetadataError::VersionMismatch { expected_version, found_version }) => {
                    // The file was present and created by the same compiler version, but we
                    // couldn't load it for some reason. Give a hard error instead of silently
                    // ignoring it, but only if we would have given an error anyway.
                    info!(
                        "Rejecting via version: expected {} got {}",
                        expected_version, found_version
                    );
                    self.crate_rejections
                        .via_version
                        .push(CrateMismatch { path: lib, got: found_version });
                    continue;
                }
                Err(MetadataError::LoadFailure(err)) => {
                    info!("no metadata found: {}", err);
                    // Metadata was loaded from interface file earlier.
                    if let Some((.., CrateFlavor::SDylib)) = slot {
                        ret = Some((lib, kind));
                        continue;
                    }
                    // The file was present and created by the same compiler version, but we
                    // couldn't load it for some reason. Give a hard error instead of silently
                    // ignoring it, but only if we would have given an error anyway.
                    self.crate_rejections.via_invalid.push(CrateMismatch { path: lib, got: err });
                    continue;
                }
                Err(err @ MetadataError::NotPresent(_)) => {
                    info!("no metadata found: {}", err);
                    continue;
                }
            };
            // If we see multiple hashes, emit an error about duplicate candidates.
            if slot.as_ref().is_some_and(|s| s.0 != hash) {
                if let Some(candidates) = err_data {
                    return Err(CrateError::MultipleCandidates(
                        self.crate_name,
                        flavor,
                        candidates,
                    ));
                }
                err_data = Some(vec![slot.take().unwrap().2]);
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
                let sysroot = try_canonicalize(sysroot).unwrap_or_else(|_| sysroot.to_path_buf());
                if prev.starts_with(&sysroot) {
                    continue;
                }
            }

            // We error eagerly here. If we're locating a rlib, then in theory the full metadata
            // could still be in a (later resolved) dylib. In practice, if the rlib and dylib
            // were produced in a way where one has full metadata and the other hasn't, it would
            // mean that they were compiled using different compiler flags and probably also have
            // a different SVH value.
            if metadata.get_header().is_stub {
                // `is_stub` should never be true for .rmeta files.
                assert_ne!(flavor, CrateFlavor::Rmeta);

                // Because rmeta files are resolved before rlib/dylib files, if this is a stub and
                // we haven't found a slot already, it means that the full metadata is missing.
                if slot.is_none() {
                    return Err(CrateError::FullMetadataNotFound(self.crate_name, flavor));
                }
            } else {
                *slot = Some((hash, metadata, lib.clone(), flavor));
            }
            ret = Some((lib, kind));
        }

        if let Some(candidates) = err_data {
            Err(CrateError::MultipleCandidates(self.crate_name, flavor, candidates))
        } else {
            Ok(ret)
        }
    }

    fn crate_matches(&mut self, metadata: &MetadataBlob, libpath: &Path) -> Option<Svh> {
        let header = metadata.get_header();
        if header.is_proc_macro_crate != self.is_proc_macro {
            info!(
                "Rejecting via proc macro: expected {} got {}",
                self.is_proc_macro, header.is_proc_macro_crate,
            );
            return None;
        }

        if self.exact_paths.is_empty() && self.crate_name != header.name {
            info!("Rejecting via crate name");
            return None;
        }

        if header.triple != self.tuple {
            info!("Rejecting via crate triple: expected {} got {}", self.tuple, header.triple);
            self.crate_rejections.via_triple.push(CrateMismatch {
                path: libpath.to_path_buf(),
                got: header.triple.to_string(),
            });
            return None;
        }

        let hash = header.hash;
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
        let mut rlibs = FxIndexMap::default();
        let mut rmetas = FxIndexMap::default();
        let mut dylibs = FxIndexMap::default();
        let mut sdylib_interfaces = FxIndexMap::default();
        for loc in &self.exact_paths {
            let loc_canon = loc.canonicalized();
            let loc_orig = loc.original();
            if !loc_canon.exists() {
                return Err(CrateError::ExternLocationNotExist(self.crate_name, loc_orig.clone()));
            }
            if !loc_orig.is_file() {
                return Err(CrateError::ExternLocationNotFile(self.crate_name, loc_orig.clone()));
            }
            // Note to take care and match against the non-canonicalized name:
            // some systems save build artifacts into content-addressed stores
            // that do not preserve extensions, and then link to them using
            // e.g. symbolic links. If we canonicalize too early, we resolve
            // the symlink, the file type is lost and we might treat rlibs and
            // rmetas as dylibs.
            let Some(file) = loc_orig.file_name().and_then(|s| s.to_str()) else {
                return Err(CrateError::ExternLocationNotFile(self.crate_name, loc_orig.clone()));
            };
            if file.starts_with("lib") {
                if file.ends_with(".rlib") {
                    rlibs.insert(loc_canon.clone(), PathKind::ExternFlag);
                    continue;
                }
                if file.ends_with(".rmeta") {
                    rmetas.insert(loc_canon.clone(), PathKind::ExternFlag);
                    continue;
                }
                if file.ends_with(".rs") {
                    sdylib_interfaces.insert(loc_canon.clone(), PathKind::ExternFlag);
                }
            }
            let dll_prefix = self.target.dll_prefix.as_ref();
            let dll_suffix = self.target.dll_suffix.as_ref();
            if file.starts_with(dll_prefix) && file.ends_with(dll_suffix) {
                dylibs.insert(loc_canon.clone(), PathKind::ExternFlag);
                continue;
            }
            self.crate_rejections
                .via_filename
                .push(CrateMismatch { path: loc_orig.clone(), got: String::new() });
        }

        // Extract the dylib/rlib/rmeta triple.
        self.extract_lib(rlibs, rmetas, dylibs, sdylib_interfaces)
            .map(|opt| opt.map(|(_, lib)| lib))
    }

    pub(crate) fn into_error(self, dep_root: Option<CratePaths>) -> CrateError {
        CrateError::LocatorCombined(Box::new(CombinedLocatorError {
            crate_name: self.crate_name,
            dep_root,
            triple: self.tuple,
            dll_prefix: self.target.dll_prefix.to_string(),
            dll_suffix: self.target.dll_suffix.to_string(),
            crate_rejections: self.crate_rejections,
        }))
    }
}

fn get_metadata_section<'p>(
    target: &Target,
    flavor: CrateFlavor,
    filename: &'p Path,
    loader: &dyn MetadataLoader,
    cfg_version: &'static str,
    crate_name: Option<Symbol>,
) -> Result<MetadataBlob, MetadataError<'p>> {
    if !filename.exists() {
        return Err(MetadataError::NotPresent(filename));
    }
    let raw_bytes = match flavor {
        CrateFlavor::Rlib => {
            loader.get_rlib_metadata(target, filename).map_err(MetadataError::LoadFailure)?
        }
        CrateFlavor::SDylib => {
            let compiler = std::env::current_exe().map_err(|_err| {
                MetadataError::LoadFailure(
                    "couldn't obtain current compiler binary when loading sdylib interface"
                        .to_string(),
                )
            })?;

            let tmp_path = match TempFileBuilder::new().prefix("rustc").tempdir() {
                Ok(tmp_path) => tmp_path,
                Err(error) => {
                    return Err(MetadataError::LoadFailure(format!(
                        "couldn't create a temp dir: {}",
                        error
                    )));
                }
            };

            let crate_name = crate_name.unwrap();
            debug!("compiling {}", filename.display());
            // FIXME: This will need to be done either within the current compiler session or
            // as a separate compiler session in the same process.
            let res = std::process::Command::new(compiler)
                .arg(&filename)
                .arg("--emit=metadata")
                .arg(format!("--crate-name={}", crate_name))
                .arg(format!("--out-dir={}", tmp_path.path().display()))
                .arg("-Zbuild-sdylib-interface")
                .output()
                .map_err(|err| {
                    MetadataError::LoadFailure(format!("couldn't compile interface: {}", err))
                })?;

            if !res.status.success() {
                return Err(MetadataError::LoadFailure(format!(
                    "couldn't compile interface: {}",
                    std::str::from_utf8(&res.stderr).unwrap_or_default()
                )));
            }

            // Load interface metadata instead of crate metadata.
            let interface_metadata_name = format!("lib{}.rmeta", crate_name);
            let rmeta_file = tmp_path.path().join(interface_metadata_name);
            debug!("loading interface metadata from {}", rmeta_file.display());
            let rmeta = get_rmeta_metadata_section(&rmeta_file)?;
            let _ = std::fs::remove_file(rmeta_file);

            rmeta
        }
        CrateFlavor::Dylib => {
            let buf =
                loader.get_dylib_metadata(target, filename).map_err(MetadataError::LoadFailure)?;
            let header_len = METADATA_HEADER.len();
            // header + u64 length of data
            let data_start = header_len + 8;

            debug!("checking {} bytes of metadata-version stamp", header_len);
            let header = &buf[..cmp::min(header_len, buf.len())];
            if header != METADATA_HEADER {
                return Err(MetadataError::LoadFailure(format!(
                    "invalid metadata version found: {}",
                    filename.display()
                )));
            }

            // Length of the metadata - this allows linkers to pad the section if they want
            let Ok(len_bytes) =
                <[u8; 8]>::try_from(&buf[header_len..cmp::min(data_start, buf.len())])
            else {
                return Err(MetadataError::LoadFailure(
                    "invalid metadata length found".to_string(),
                ));
            };
            let metadata_len = u64::from_le_bytes(len_bytes) as usize;

            // Header is okay -> inflate the actual metadata
            buf.slice(|buf| &buf[data_start..(data_start + metadata_len)])
        }
        CrateFlavor::Rmeta => get_rmeta_metadata_section(filename)?,
    };
    let Ok(blob) = MetadataBlob::new(raw_bytes) else {
        return Err(MetadataError::LoadFailure(format!(
            "corrupt metadata encountered in {}",
            filename.display()
        )));
    };
    match blob.check_compatibility(cfg_version) {
        Ok(()) => {
            debug!("metadata blob read okay");
            Ok(blob)
        }
        Err(None) => Err(MetadataError::LoadFailure(format!(
            "invalid metadata version found: {}",
            filename.display()
        ))),
        Err(Some(found_version)) => {
            return Err(MetadataError::VersionMismatch {
                expected_version: rustc_version(cfg_version),
                found_version,
            });
        }
    }
}

fn get_rmeta_metadata_section<'a, 'p>(filename: &'p Path) -> Result<OwnedSlice, MetadataError<'a>> {
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

    Ok(slice_owned(mmap, Deref::deref))
}

/// A diagnostic function for dumping crate metadata to an output stream.
pub fn list_file_metadata(
    target: &Target,
    path: &Path,
    metadata_loader: &dyn MetadataLoader,
    out: &mut dyn Write,
    ls_kinds: &[String],
    cfg_version: &'static str,
) -> IoResult<()> {
    let flavor = get_flavor_from_path(path);
    match get_metadata_section(target, flavor, path, metadata_loader, cfg_version, None) {
        Ok(metadata) => metadata.list_crate_metadata(out, ls_kinds),
        Err(msg) => write!(out, "{msg}\n"),
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

#[derive(Clone, Debug)]
struct CrateMismatch {
    path: PathBuf,
    got: String,
}

#[derive(Clone, Debug, Default)]
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
#[derive(Debug)]
pub(crate) struct CombinedLocatorError {
    crate_name: Symbol,
    dep_root: Option<CratePaths>,
    triple: TargetTuple,
    dll_prefix: String,
    dll_suffix: String,
    crate_rejections: CrateRejections,
}

#[derive(Debug)]
pub(crate) enum CrateError {
    NonAsciiName(Symbol),
    ExternLocationNotExist(Symbol, PathBuf),
    ExternLocationNotFile(Symbol, PathBuf),
    MultipleCandidates(Symbol, CrateFlavor, Vec<PathBuf>),
    FullMetadataNotFound(Symbol, CrateFlavor),
    SymbolConflictsCurrent(Symbol),
    StableCrateIdCollision(Symbol, Symbol),
    DlOpen(String, String),
    DlSym(String, String),
    LocatorCombined(Box<CombinedLocatorError>),
    NotFound(Symbol),
}

enum MetadataError<'a> {
    /// The file was missing.
    NotPresent(&'a Path),
    /// The file was present and invalid.
    LoadFailure(String),
    /// The file was present, but compiled with a different rustc version.
    VersionMismatch { expected_version: String, found_version: String },
}

impl fmt::Display for MetadataError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetadataError::NotPresent(filename) => {
                f.write_str(&format!("no such file: '{}'", filename.display()))
            }
            MetadataError::LoadFailure(msg) => f.write_str(msg),
            MetadataError::VersionMismatch { expected_version, found_version } => {
                f.write_str(&format!(
                    "rustc version mismatch. expected {}, found {}",
                    expected_version, found_version,
                ))
            }
        }
    }
}

impl CrateError {
    pub(crate) fn report(self, sess: &Session, span: Span, missing_core: bool) {
        let dcx = sess.dcx();
        match self {
            CrateError::NonAsciiName(crate_name) => {
                dcx.emit_err(errors::NonAsciiName { span, crate_name });
            }
            CrateError::ExternLocationNotExist(crate_name, loc) => {
                dcx.emit_err(errors::ExternLocationNotExist { span, crate_name, location: &loc });
            }
            CrateError::ExternLocationNotFile(crate_name, loc) => {
                dcx.emit_err(errors::ExternLocationNotFile { span, crate_name, location: &loc });
            }
            CrateError::MultipleCandidates(crate_name, flavor, candidates) => {
                dcx.emit_err(errors::MultipleCandidates { span, crate_name, flavor, candidates });
            }
            CrateError::FullMetadataNotFound(crate_name, flavor) => {
                dcx.emit_err(errors::FullMetadataNotFound { span, crate_name, flavor });
            }
            CrateError::SymbolConflictsCurrent(root_name) => {
                dcx.emit_err(errors::SymbolConflictsCurrent { span, crate_name: root_name });
            }
            CrateError::StableCrateIdCollision(crate_name0, crate_name1) => {
                dcx.emit_err(errors::StableCrateIdCollision { span, crate_name0, crate_name1 });
            }
            CrateError::DlOpen(path, err) | CrateError::DlSym(path, err) => {
                dcx.emit_err(errors::DlError { span, path, err });
            }
            CrateError::LocatorCombined(locator) => {
                let crate_name = locator.crate_name;
                let add_info = match &locator.dep_root {
                    None => String::new(),
                    Some(r) => format!(" which `{}` depends on", r.name),
                };
                if !locator.crate_rejections.via_filename.is_empty() {
                    let mismatches = locator.crate_rejections.via_filename.iter();
                    for CrateMismatch { path, .. } in mismatches {
                        dcx.emit_err(errors::CrateLocationUnknownType { span, path, crate_name });
                        dcx.emit_err(errors::LibFilenameForm {
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
                    if let Some(r) = locator.dep_root {
                        for path in r.source.paths() {
                            found_crates.push_str(&format!(
                                "\ncrate `{}`: {}",
                                r.name,
                                path.display()
                            ));
                        }
                    }
                    dcx.emit_err(errors::NewerCrateVersion {
                        span,
                        crate_name,
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
                    dcx.emit_err(errors::NoCrateWithTriple {
                        span,
                        crate_name,
                        locator_triple: locator.triple.tuple(),
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
                    dcx.emit_err(errors::FoundStaticlib {
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
                    dcx.emit_err(errors::IncompatibleRustc {
                        span,
                        crate_name,
                        add_info,
                        found_crates,
                        rustc_version: rustc_version(sess.cfg_version),
                    });
                } else if !locator.crate_rejections.via_invalid.is_empty() {
                    let mut crate_rejections = Vec::new();
                    for CrateMismatch { path: _, got } in locator.crate_rejections.via_invalid {
                        crate_rejections.push(got);
                    }
                    dcx.emit_err(errors::InvalidMetadataFiles {
                        span,
                        crate_name,
                        add_info,
                        crate_rejections,
                    });
                } else {
                    let error = errors::CannotFindCrate {
                        span,
                        crate_name,
                        add_info,
                        missing_core,
                        current_crate: sess
                            .opts
                            .crate_name
                            .clone()
                            .unwrap_or_else(|| "<unknown>".to_string()),
                        is_nightly_build: sess.is_nightly_build(),
                        profiler_runtime: Symbol::intern(&sess.opts.unstable_opts.profiler_runtime),
                        locator_triple: locator.triple,
                        is_ui_testing: sess.opts.unstable_opts.ui_testing,
                    };
                    // The diagnostic for missing core is very good, but it is followed by a lot of
                    // other diagnostics that do not add information.
                    if missing_core {
                        dcx.emit_fatal(error);
                    } else {
                        dcx.emit_err(error);
                    }
                }
            }
            CrateError::NotFound(crate_name) => {
                let error = errors::CannotFindCrate {
                    span,
                    crate_name,
                    add_info: String::new(),
                    missing_core,
                    current_crate: sess
                        .opts
                        .crate_name
                        .clone()
                        .unwrap_or_else(|| "<unknown>".to_string()),
                    is_nightly_build: sess.is_nightly_build(),
                    profiler_runtime: Symbol::intern(&sess.opts.unstable_opts.profiler_runtime),
                    locator_triple: sess.opts.target_triple.clone(),
                    is_ui_testing: sess.opts.unstable_opts.ui_testing,
                };
                // The diagnostic for missing core is very good, but it is followed by a lot of
                // other diagnostics that do not add information.
                if missing_core {
                    dcx.emit_fatal(error);
                } else {
                    dcx.emit_err(error);
                }
            }
        }
    }
}
