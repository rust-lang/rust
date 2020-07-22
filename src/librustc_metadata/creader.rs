//! Validates all used crates and extern libraries and loads their metadata

use crate::locator::{CrateLocator, CratePaths};
use crate::rmeta::{CrateDep, CrateMetadata, CrateNumMap, CrateRoot, MetadataBlob};

use rustc_ast::expand::allocator::{global_allocator_spans, AllocatorKind};
use rustc_ast::{ast, attr};
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::Lrc;
use rustc_errors::struct_span_err;
use rustc_expand::base::SyntaxExtension;
use rustc_hir::def_id::{CrateNum, LocalDefId, LOCAL_CRATE};
use rustc_hir::definitions::Definitions;
use rustc_index::vec::IndexVec;
use rustc_middle::middle::cstore::DepKind;
use rustc_middle::middle::cstore::{
    CrateSource, ExternCrate, ExternCrateSource, MetadataLoaderDyn,
};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{self, CrateType, ExternLocation};
use rustc_session::lint;
use rustc_session::output::validate_crate_name;
use rustc_session::search_paths::PathKind;
use rustc_session::{CrateDisambiguator, Session};
use rustc_span::edition::Edition;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::spec::{PanicStrategy, TargetTriple};

use log::{debug, info, log_enabled};
use proc_macro::bridge::client::ProcMacro;
use std::path::Path;
use std::{cmp, fs};

#[derive(Clone)]
pub struct CStore {
    metas: IndexVec<CrateNum, Option<Lrc<CrateMetadata>>>,
    injected_panic_runtime: Option<CrateNum>,
    /// This crate needs an allocator and either provides it itself, or finds it in a dependency.
    /// If the above is true, then this field denotes the kind of the found allocator.
    allocator_kind: Option<AllocatorKind>,
    /// This crate has a `#[global_allocator]` item.
    has_global_allocator: bool,
}

pub struct CrateLoader<'a> {
    // Immutable configuration.
    sess: &'a Session,
    metadata_loader: &'a MetadataLoaderDyn,
    local_crate_name: Symbol,
    // Mutable output.
    cstore: CStore,
    used_extern_options: FxHashSet<Symbol>,
}

pub enum LoadedMacro {
    MacroDef(ast::Item, Edition),
    ProcMacro(SyntaxExtension),
}

crate struct Library {
    pub source: CrateSource,
    pub metadata: MetadataBlob,
}

enum LoadResult {
    Previous(CrateNum),
    Loaded(Library),
}

enum LoadError<'a> {
    LocatorError(CrateLocator<'a>),
}

impl<'a> LoadError<'a> {
    fn report(self) -> ! {
        match self {
            LoadError::LocatorError(locator) => locator.report_errs(),
        }
    }
}

/// A reference to `CrateMetadata` that can also give access to whole crate store when necessary.
#[derive(Clone, Copy)]
crate struct CrateMetadataRef<'a> {
    pub cdata: &'a CrateMetadata,
    pub cstore: &'a CStore,
}

impl std::ops::Deref for CrateMetadataRef<'_> {
    type Target = CrateMetadata;

    fn deref(&self) -> &Self::Target {
        self.cdata
    }
}

fn dump_crates(cstore: &CStore) {
    info!("resolved crates:");
    cstore.iter_crate_data(|cnum, data| {
        info!("  name: {}", data.name());
        info!("  cnum: {}", cnum);
        info!("  hash: {}", data.hash());
        info!("  reqd: {:?}", data.dep_kind());
        let CrateSource { dylib, rlib, rmeta } = data.source();
        if let Some(dylib) = dylib {
            info!("  dylib: {}", dylib.0.display());
        }
        if let Some(rlib) = rlib {
            info!("   rlib: {}", rlib.0.display());
        }
        if let Some(rmeta) = rmeta {
            info!("   rmeta: {}", rmeta.0.display());
        }
    });
}

impl CStore {
    crate fn from_tcx(tcx: TyCtxt<'_>) -> &CStore {
        tcx.cstore_as_any().downcast_ref::<CStore>().expect("`tcx.cstore` is not a `CStore`")
    }

    fn alloc_new_crate_num(&mut self) -> CrateNum {
        self.metas.push(None);
        CrateNum::new(self.metas.len() - 1)
    }

    crate fn get_crate_data(&self, cnum: CrateNum) -> CrateMetadataRef<'_> {
        let cdata = self.metas[cnum]
            .as_ref()
            .unwrap_or_else(|| panic!("Failed to get crate data for {:?}", cnum));
        CrateMetadataRef { cdata, cstore: self }
    }

    fn set_crate_data(&mut self, cnum: CrateNum, data: CrateMetadata) {
        assert!(self.metas[cnum].is_none(), "Overwriting crate metadata entry");
        self.metas[cnum] = Some(Lrc::new(data));
    }

    crate fn iter_crate_data(&self, mut f: impl FnMut(CrateNum, &CrateMetadata)) {
        for (cnum, data) in self.metas.iter_enumerated() {
            if let Some(data) = data {
                f(cnum, data);
            }
        }
    }

    fn push_dependencies_in_postorder(&self, deps: &mut Vec<CrateNum>, cnum: CrateNum) {
        if !deps.contains(&cnum) {
            let data = self.get_crate_data(cnum);
            for &dep in data.dependencies().iter() {
                if dep != cnum {
                    self.push_dependencies_in_postorder(deps, dep);
                }
            }

            deps.push(cnum);
        }
    }

    crate fn crate_dependencies_in_postorder(&self, cnum: CrateNum) -> Vec<CrateNum> {
        let mut deps = Vec::new();
        if cnum == LOCAL_CRATE {
            self.iter_crate_data(|cnum, _| self.push_dependencies_in_postorder(&mut deps, cnum));
        } else {
            self.push_dependencies_in_postorder(&mut deps, cnum);
        }
        deps
    }

    fn crate_dependencies_in_reverse_postorder(&self, cnum: CrateNum) -> Vec<CrateNum> {
        let mut deps = self.crate_dependencies_in_postorder(cnum);
        deps.reverse();
        deps
    }

    crate fn injected_panic_runtime(&self) -> Option<CrateNum> {
        self.injected_panic_runtime
    }

    crate fn allocator_kind(&self) -> Option<AllocatorKind> {
        self.allocator_kind
    }

    crate fn has_global_allocator(&self) -> bool {
        self.has_global_allocator
    }
}

impl<'a> CrateLoader<'a> {
    pub fn new(
        sess: &'a Session,
        metadata_loader: &'a MetadataLoaderDyn,
        local_crate_name: &str,
    ) -> Self {
        CrateLoader {
            sess,
            metadata_loader,
            local_crate_name: Symbol::intern(local_crate_name),
            cstore: CStore {
                // We add an empty entry for LOCAL_CRATE (which maps to zero) in
                // order to make array indices in `metas` match with the
                // corresponding `CrateNum`. This first entry will always remain
                // `None`.
                metas: IndexVec::from_elem_n(None, 1),
                injected_panic_runtime: None,
                allocator_kind: None,
                has_global_allocator: false,
            },
            used_extern_options: Default::default(),
        }
    }

    pub fn cstore(&self) -> &CStore {
        &self.cstore
    }

    pub fn into_cstore(self) -> CStore {
        self.cstore
    }

    fn existing_match(&self, name: Symbol, hash: Option<Svh>, kind: PathKind) -> Option<CrateNum> {
        let mut ret = None;
        self.cstore.iter_crate_data(|cnum, data| {
            if data.name() != name {
                return;
            }

            match hash {
                Some(hash) if hash == data.hash() => {
                    ret = Some(cnum);
                    return;
                }
                Some(..) => return,
                None => {}
            }

            // When the hash is None we're dealing with a top-level dependency
            // in which case we may have a specification on the command line for
            // this library. Even though an upstream library may have loaded
            // something of the same name, we have to make sure it was loaded
            // from the exact same location as well.
            //
            // We're also sure to compare *paths*, not actual byte slices. The
            // `source` stores paths which are normalized which may be different
            // from the strings on the command line.
            let source = self.cstore.get_crate_data(cnum).cdata.source();
            if let Some(entry) = self.sess.opts.externs.get(&name.as_str()) {
                // Only use `--extern crate_name=path` here, not `--extern crate_name`.
                if let Some(mut files) = entry.files() {
                    if files.any(|l| {
                        let l = fs::canonicalize(l).ok();
                        source.dylib.as_ref().map(|p| &p.0) == l.as_ref()
                            || source.rlib.as_ref().map(|p| &p.0) == l.as_ref()
                    }) {
                        ret = Some(cnum);
                    }
                }
                return;
            }

            // Alright, so we've gotten this far which means that `data` has the
            // right name, we don't have a hash, and we don't have a --extern
            // pointing for ourselves. We're still not quite yet done because we
            // have to make sure that this crate was found in the crate lookup
            // path (this is a top-level dependency) as we don't want to
            // implicitly load anything inside the dependency lookup path.
            let prev_kind = source
                .dylib
                .as_ref()
                .or(source.rlib.as_ref())
                .or(source.rmeta.as_ref())
                .expect("No sources for crate")
                .1;
            if kind.matches(prev_kind) {
                ret = Some(cnum);
            }
        });
        ret
    }

    fn verify_no_symbol_conflicts(&self, span: Span, root: &CrateRoot<'_>) {
        // Check for (potential) conflicts with the local crate
        if self.local_crate_name == root.name()
            && self.sess.local_crate_disambiguator() == root.disambiguator()
        {
            struct_span_err!(
                self.sess,
                span,
                E0519,
                "the current crate is indistinguishable from one of its \
                         dependencies: it has the same crate-name `{}` and was \
                         compiled with the same `-C metadata` arguments. This \
                         will result in symbol conflicts between the two.",
                root.name()
            )
            .emit()
        }

        // Check for conflicts with any crate loaded so far
        self.cstore.iter_crate_data(|_, other| {
            if other.name() == root.name() && // same crate-name
               other.disambiguator() == root.disambiguator() &&  // same crate-disambiguator
               other.hash() != root.hash()
            {
                // but different SVH
                struct_span_err!(
                    self.sess,
                    span,
                    E0523,
                    "found two different crates with name `{}` that are \
                         not distinguished by differing `-C metadata`. This \
                         will result in symbol conflicts between the two.",
                    root.name()
                )
                .emit();
            }
        });
    }

    fn register_crate(
        &mut self,
        host_lib: Option<Library>,
        root: Option<&CratePaths>,
        span: Span,
        lib: Library,
        dep_kind: DepKind,
        name: Symbol,
    ) -> CrateNum {
        let _prof_timer = self.sess.prof.generic_activity("metadata_register_crate");

        let Library { source, metadata } = lib;
        let crate_root = metadata.get_root();
        let host_hash = host_lib.as_ref().map(|lib| lib.metadata.get_root().hash());
        self.verify_no_symbol_conflicts(span, &crate_root);

        let private_dep =
            self.sess.opts.externs.get(&name.as_str()).map(|e| e.is_private_dep).unwrap_or(false);

        info!("register crate `{}` (private_dep = {})", crate_root.name(), private_dep);

        // Claim this crate number and cache it
        let cnum = self.cstore.alloc_new_crate_num();

        // Maintain a reference to the top most crate.
        // Stash paths for top-most crate locally if necessary.
        let crate_paths;
        let root = if let Some(root) = root {
            root
        } else {
            crate_paths = CratePaths::new(crate_root.name(), source.clone());
            &crate_paths
        };

        let cnum_map = self.resolve_crate_deps(root, &crate_root, &metadata, cnum, span, dep_kind);

        let raw_proc_macros = if crate_root.is_proc_macro_crate() {
            let temp_root;
            let (dlsym_source, dlsym_root) = match &host_lib {
                Some(host_lib) => (&host_lib.source, {
                    temp_root = host_lib.metadata.get_root();
                    &temp_root
                }),
                None => (&source, &crate_root),
            };
            let dlsym_dylib = dlsym_source.dylib.as_ref().expect("no dylib for a proc-macro crate");
            Some(self.dlsym_proc_macros(&dlsym_dylib.0, dlsym_root.disambiguator(), span))
        } else {
            None
        };

        self.cstore.set_crate_data(
            cnum,
            CrateMetadata::new(
                self.sess,
                metadata,
                crate_root,
                raw_proc_macros,
                cnum,
                cnum_map,
                dep_kind,
                source,
                private_dep,
                host_hash,
            ),
        );

        cnum
    }

    fn load_proc_macro<'b>(
        &self,
        locator: &mut CrateLocator<'b>,
        path_kind: PathKind,
    ) -> Option<(LoadResult, Option<Library>)>
    where
        'a: 'b,
    {
        // Use a new crate locator so trying to load a proc macro doesn't affect the error
        // message we emit
        let mut proc_macro_locator = locator.clone();

        // Try to load a proc macro
        proc_macro_locator.is_proc_macro = Some(true);

        // Load the proc macro crate for the target
        let (locator, target_result) = if self.sess.opts.debugging_opts.dual_proc_macros {
            proc_macro_locator.reset();
            let result = match self.load(&mut proc_macro_locator)? {
                LoadResult::Previous(cnum) => return Some((LoadResult::Previous(cnum), None)),
                LoadResult::Loaded(library) => Some(LoadResult::Loaded(library)),
            };
            locator.hash = locator.host_hash;
            // Use the locator when looking for the host proc macro crate, as that is required
            // so we want it to affect the error message
            (locator, result)
        } else {
            (&mut proc_macro_locator, None)
        };

        // Load the proc macro crate for the host

        locator.reset();
        locator.is_proc_macro = Some(true);
        locator.target = &self.sess.host;
        locator.triple = TargetTriple::from_triple(config::host_triple());
        locator.filesearch = self.sess.host_filesearch(path_kind);

        let host_result = self.load(locator)?;

        Some(if self.sess.opts.debugging_opts.dual_proc_macros {
            let host_result = match host_result {
                LoadResult::Previous(..) => {
                    panic!("host and target proc macros must be loaded in lock-step")
                }
                LoadResult::Loaded(library) => library,
            };
            (target_result.unwrap(), Some(host_result))
        } else {
            (host_result, None)
        })
    }

    fn resolve_crate<'b>(
        &'b mut self,
        name: Symbol,
        span: Span,
        dep_kind: DepKind,
        dep: Option<(&'b CratePaths, &'b CrateDep)>,
    ) -> CrateNum {
        if dep.is_none() {
            self.used_extern_options.insert(name);
        }
        if !name.as_str().is_ascii() {
            self.sess
                .struct_span_err(
                    span,
                    &format!("cannot load a crate with a non-ascii name `{}`", name,),
                )
                .emit();
        }
        self.maybe_resolve_crate(name, span, dep_kind, dep).unwrap_or_else(|err| err.report())
    }

    fn maybe_resolve_crate<'b>(
        &'b mut self,
        name: Symbol,
        span: Span,
        mut dep_kind: DepKind,
        dep: Option<(&'b CratePaths, &'b CrateDep)>,
    ) -> Result<CrateNum, LoadError<'b>> {
        info!("resolving crate `{}`", name);
        let (root, hash, host_hash, extra_filename, path_kind) = match dep {
            Some((root, dep)) => (
                Some(root),
                Some(dep.hash),
                dep.host_hash,
                Some(&dep.extra_filename[..]),
                PathKind::Dependency,
            ),
            None => (None, None, None, None, PathKind::Crate),
        };
        let result = if let Some(cnum) = self.existing_match(name, hash, path_kind) {
            (LoadResult::Previous(cnum), None)
        } else {
            info!("falling back to a load");
            let mut locator = CrateLocator::new(
                self.sess,
                self.metadata_loader,
                name,
                hash,
                host_hash,
                extra_filename,
                false, // is_host
                path_kind,
                span,
                root,
                Some(false), // is_proc_macro
            );

            self.load(&mut locator)
                .map(|r| (r, None))
                .or_else(|| {
                    dep_kind = DepKind::MacrosOnly;
                    self.load_proc_macro(&mut locator, path_kind)
                })
                .ok_or_else(move || LoadError::LocatorError(locator))?
        };

        match result {
            (LoadResult::Previous(cnum), None) => {
                let data = self.cstore.get_crate_data(cnum);
                if data.is_proc_macro_crate() {
                    dep_kind = DepKind::MacrosOnly;
                }
                data.update_dep_kind(|data_dep_kind| cmp::max(data_dep_kind, dep_kind));
                Ok(cnum)
            }
            (LoadResult::Loaded(library), host_library) => {
                Ok(self.register_crate(host_library, root, span, library, dep_kind, name))
            }
            _ => panic!(),
        }
    }

    fn load(&self, locator: &mut CrateLocator<'_>) -> Option<LoadResult> {
        let library = locator.maybe_load_library_crate()?;

        // In the case that we're loading a crate, but not matching
        // against a hash, we could load a crate which has the same hash
        // as an already loaded crate. If this is the case prevent
        // duplicates by just using the first crate.
        //
        // Note that we only do this for target triple crates, though, as we
        // don't want to match a host crate against an equivalent target one
        // already loaded.
        let root = library.metadata.get_root();
        if locator.triple == self.sess.opts.target_triple {
            let mut result = LoadResult::Loaded(library);
            self.cstore.iter_crate_data(|cnum, data| {
                if data.name() == root.name() && root.hash() == data.hash() {
                    assert!(locator.hash.is_none());
                    info!("load success, going to previous cnum: {}", cnum);
                    result = LoadResult::Previous(cnum);
                }
            });
            Some(result)
        } else {
            Some(LoadResult::Loaded(library))
        }
    }

    fn update_extern_crate(&self, cnum: CrateNum, extern_crate: ExternCrate) {
        let cmeta = self.cstore.get_crate_data(cnum);
        if cmeta.update_extern_crate(extern_crate) {
            // Propagate the extern crate info to dependencies if it was updated.
            let extern_crate = ExternCrate { dependency_of: cnum, ..extern_crate };
            for &dep_cnum in cmeta.dependencies().iter() {
                self.update_extern_crate(dep_cnum, extern_crate);
            }
        }
    }

    // Go through the crate metadata and load any crates that it references
    fn resolve_crate_deps(
        &mut self,
        root: &CratePaths,
        crate_root: &CrateRoot<'_>,
        metadata: &MetadataBlob,
        krate: CrateNum,
        span: Span,
        dep_kind: DepKind,
    ) -> CrateNumMap {
        debug!("resolving deps of external crate");
        if crate_root.is_proc_macro_crate() {
            return CrateNumMap::new();
        }

        // The map from crate numbers in the crate we're resolving to local crate numbers.
        // We map 0 and all other holes in the map to our parent crate. The "additional"
        // self-dependencies should be harmless.
        std::iter::once(krate)
            .chain(crate_root.decode_crate_deps(metadata).map(|dep| {
                info!(
                    "resolving dep crate {} hash: `{}` extra filename: `{}`",
                    dep.name, dep.hash, dep.extra_filename
                );
                let dep_kind = match dep_kind {
                    DepKind::MacrosOnly => DepKind::MacrosOnly,
                    _ => dep.kind,
                };
                self.resolve_crate(dep.name, span, dep_kind, Some((root, &dep)))
            }))
            .collect()
    }

    fn dlsym_proc_macros(
        &self,
        path: &Path,
        disambiguator: CrateDisambiguator,
        span: Span,
    ) -> &'static [ProcMacro] {
        use crate::dynamic_lib::DynamicLibrary;
        use std::env;

        // Make sure the path contains a / or the linker will search for it.
        let path = env::current_dir().unwrap().join(path);
        let lib = match DynamicLibrary::open(&path) {
            Ok(lib) => lib,
            Err(err) => self.sess.span_fatal(span, &err),
        };

        let sym = self.sess.generate_proc_macro_decls_symbol(disambiguator);
        let decls = unsafe {
            let sym = match lib.symbol(&sym) {
                Ok(f) => f,
                Err(err) => self.sess.span_fatal(span, &err),
            };
            *(sym as *const &[ProcMacro])
        };

        // Intentionally leak the dynamic library. We can't ever unload it
        // since the library can make things that will live arbitrarily long.
        std::mem::forget(lib);

        decls
    }

    fn inject_panic_runtime(&mut self, krate: &ast::Crate) {
        // If we're only compiling an rlib, then there's no need to select a
        // panic runtime, so we just skip this section entirely.
        let any_non_rlib = self.sess.crate_types().iter().any(|ct| *ct != CrateType::Rlib);
        if !any_non_rlib {
            info!("panic runtime injection skipped, only generating rlib");
            return;
        }

        // If we need a panic runtime, we try to find an existing one here. At
        // the same time we perform some general validation of the DAG we've got
        // going such as ensuring everything has a compatible panic strategy.
        //
        // The logic for finding the panic runtime here is pretty much the same
        // as the allocator case with the only addition that the panic strategy
        // compilation mode also comes into play.
        let desired_strategy = self.sess.panic_strategy();
        let mut runtime_found = false;
        let mut needs_panic_runtime = attr::contains_name(&krate.attrs, sym::needs_panic_runtime);

        self.cstore.iter_crate_data(|cnum, data| {
            needs_panic_runtime = needs_panic_runtime || data.needs_panic_runtime();
            if data.is_panic_runtime() {
                // Inject a dependency from all #![needs_panic_runtime] to this
                // #![panic_runtime] crate.
                self.inject_dependency_if(cnum, "a panic runtime", &|data| {
                    data.needs_panic_runtime()
                });
                runtime_found = runtime_found || data.dep_kind() == DepKind::Explicit;
            }
        });

        // If an explicitly linked and matching panic runtime was found, or if
        // we just don't need one at all, then we're done here and there's
        // nothing else to do.
        if !needs_panic_runtime || runtime_found {
            return;
        }

        // By this point we know that we (a) need a panic runtime and (b) no
        // panic runtime was explicitly linked. Here we just load an appropriate
        // default runtime for our panic strategy and then inject the
        // dependencies.
        //
        // We may resolve to an already loaded crate (as the crate may not have
        // been explicitly linked prior to this) and we may re-inject
        // dependencies again, but both of those situations are fine.
        //
        // Also note that we have yet to perform validation of the crate graph
        // in terms of everyone has a compatible panic runtime format, that's
        // performed later as part of the `dependency_format` module.
        let name = match desired_strategy {
            PanicStrategy::Unwind => sym::panic_unwind,
            PanicStrategy::Abort => sym::panic_abort,
        };
        info!("panic runtime not found -- loading {}", name);

        let cnum = self.resolve_crate(name, DUMMY_SP, DepKind::Implicit, None);
        let data = self.cstore.get_crate_data(cnum);

        // Sanity check the loaded crate to ensure it is indeed a panic runtime
        // and the panic strategy is indeed what we thought it was.
        if !data.is_panic_runtime() {
            self.sess.err(&format!("the crate `{}` is not a panic runtime", name));
        }
        if data.panic_strategy() != desired_strategy {
            self.sess.err(&format!(
                "the crate `{}` does not have the panic \
                                    strategy `{}`",
                name,
                desired_strategy.desc()
            ));
        }

        self.cstore.injected_panic_runtime = Some(cnum);
        self.inject_dependency_if(cnum, "a panic runtime", &|data| data.needs_panic_runtime());
    }

    fn inject_profiler_runtime(&mut self) {
        if (self.sess.opts.debugging_opts.instrument_coverage
            || self.sess.opts.debugging_opts.profile
            || self.sess.opts.cg.profile_generate.enabled())
            && !self.sess.opts.debugging_opts.no_profiler_runtime
        {
            info!("loading profiler");

            let name = sym::profiler_builtins;
            let cnum = self.resolve_crate(name, DUMMY_SP, DepKind::Implicit, None);
            let data = self.cstore.get_crate_data(cnum);

            // Sanity check the loaded crate to ensure it is indeed a profiler runtime
            if !data.is_profiler_runtime() {
                self.sess.err("the crate `profiler_builtins` is not a profiler runtime");
            }
        }
    }

    fn inject_allocator_crate(&mut self, krate: &ast::Crate) {
        self.cstore.has_global_allocator = match &*global_allocator_spans(krate) {
            [span1, span2, ..] => {
                self.sess
                    .struct_span_err(*span2, "cannot define multiple global allocators")
                    .span_label(*span2, "cannot define a new global allocator")
                    .span_label(*span1, "previous global allocator defined here")
                    .emit();
                true
            }
            spans => !spans.is_empty(),
        };

        // Check to see if we actually need an allocator. This desire comes
        // about through the `#![needs_allocator]` attribute and is typically
        // written down in liballoc.
        let mut needs_allocator = attr::contains_name(&krate.attrs, sym::needs_allocator);
        self.cstore.iter_crate_data(|_, data| {
            needs_allocator = needs_allocator || data.needs_allocator();
        });
        if !needs_allocator {
            return;
        }

        // At this point we've determined that we need an allocator. Let's see
        // if our compilation session actually needs an allocator based on what
        // we're emitting.
        let all_rlib = self.sess.crate_types().iter().all(|ct| match *ct {
            CrateType::Rlib => true,
            _ => false,
        });
        if all_rlib {
            return;
        }

        // Ok, we need an allocator. Not only that but we're actually going to
        // create an artifact that needs one linked in. Let's go find the one
        // that we're going to link in.
        //
        // First up we check for global allocators. Look at the crate graph here
        // and see what's a global allocator, including if we ourselves are a
        // global allocator.
        let mut global_allocator =
            self.cstore.has_global_allocator.then(|| Symbol::intern("this crate"));
        self.cstore.iter_crate_data(|_, data| {
            if !data.has_global_allocator() {
                return;
            }
            match global_allocator {
                Some(other_crate) => {
                    self.sess.err(&format!(
                        "the `#[global_allocator]` in {} \
                                            conflicts with global \
                                            allocator in: {}",
                        other_crate,
                        data.name()
                    ));
                }
                None => global_allocator = Some(data.name()),
            }
        });
        if global_allocator.is_some() {
            self.cstore.allocator_kind = Some(AllocatorKind::Global);
            return;
        }

        // Ok we haven't found a global allocator but we still need an
        // allocator. At this point our allocator request is typically fulfilled
        // by the standard library, denoted by the `#![default_lib_allocator]`
        // attribute.
        let mut has_default = attr::contains_name(&krate.attrs, sym::default_lib_allocator);
        self.cstore.iter_crate_data(|_, data| {
            if data.has_default_lib_allocator() {
                has_default = true;
            }
        });

        if !has_default {
            self.sess.err(
                "no global memory allocator found but one is \
                           required; link to std or \
                           add `#[global_allocator]` to a static item \
                           that implements the GlobalAlloc trait.",
            );
        }
        self.cstore.allocator_kind = Some(AllocatorKind::Default);
    }

    fn inject_dependency_if(
        &self,
        krate: CrateNum,
        what: &str,
        needs_dep: &dyn Fn(&CrateMetadata) -> bool,
    ) {
        // don't perform this validation if the session has errors, as one of
        // those errors may indicate a circular dependency which could cause
        // this to stack overflow.
        if self.sess.has_errors() {
            return;
        }

        // Before we inject any dependencies, make sure we don't inject a
        // circular dependency by validating that this crate doesn't
        // transitively depend on any crates satisfying `needs_dep`.
        for dep in self.cstore.crate_dependencies_in_reverse_postorder(krate) {
            let data = self.cstore.get_crate_data(dep);
            if needs_dep(&data) {
                self.sess.err(&format!(
                    "the crate `{}` cannot depend \
                                        on a crate that needs {}, but \
                                        it depends on `{}`",
                    self.cstore.get_crate_data(krate).name(),
                    what,
                    data.name()
                ));
            }
        }

        // All crates satisfying `needs_dep` do not explicitly depend on the
        // crate provided for this compile, but in order for this compilation to
        // be successfully linked we need to inject a dependency (to order the
        // crates on the command line correctly).
        self.cstore.iter_crate_data(|cnum, data| {
            if !needs_dep(data) {
                return;
            }

            info!("injecting a dep from {} to {}", cnum, krate);
            data.add_dependency(krate);
        });
    }

    fn report_unused_deps(&mut self, krate: &ast::Crate) {
        // Make a point span rather than covering the whole file
        let span = krate.span.shrink_to_lo();
        // Complain about anything left over
        for (name, entry) in self.sess.opts.externs.iter() {
            if let ExternLocation::FoundInLibrarySearchDirectories = entry.location {
                // Don't worry about pathless `--extern foo` sysroot references
                continue;
            }
            if !self.used_extern_options.contains(&Symbol::intern(name)) {
                self.sess.parse_sess.buffer_lint(
                    lint::builtin::UNUSED_CRATE_DEPENDENCIES,
                    span,
                    ast::CRATE_NODE_ID,
                    &format!(
                        "external crate `{}` unused in `{}`: remove the dependency or add `use {} as _;`",
                        name,
                        self.local_crate_name,
                        name),
                );
            }
        }
    }

    pub fn postprocess(&mut self, krate: &ast::Crate) {
        self.inject_profiler_runtime();
        self.inject_allocator_crate(krate);
        self.inject_panic_runtime(krate);

        if log_enabled!(log::Level::Info) {
            dump_crates(&self.cstore);
        }

        self.report_unused_deps(krate);
    }

    pub fn process_extern_crate(
        &mut self,
        item: &ast::Item,
        definitions: &Definitions,
        def_id: LocalDefId,
    ) -> CrateNum {
        match item.kind {
            ast::ItemKind::ExternCrate(orig_name) => {
                debug!(
                    "resolving extern crate stmt. ident: {} orig_name: {:?}",
                    item.ident, orig_name
                );
                let name = match orig_name {
                    Some(orig_name) => {
                        validate_crate_name(Some(self.sess), &orig_name.as_str(), Some(item.span));
                        orig_name
                    }
                    None => item.ident.name,
                };
                let dep_kind = if attr::contains_name(&item.attrs, sym::no_link) {
                    DepKind::MacrosOnly
                } else {
                    DepKind::Explicit
                };

                let cnum = self.resolve_crate(name, item.span, dep_kind, None);

                let path_len = definitions.def_path(def_id).data.len();
                self.update_extern_crate(
                    cnum,
                    ExternCrate {
                        src: ExternCrateSource::Extern(def_id.to_def_id()),
                        span: item.span,
                        path_len,
                        dependency_of: LOCAL_CRATE,
                    },
                );
                cnum
            }
            _ => bug!(),
        }
    }

    pub fn process_path_extern(&mut self, name: Symbol, span: Span) -> CrateNum {
        let cnum = self.resolve_crate(name, span, DepKind::Explicit, None);

        self.update_extern_crate(
            cnum,
            ExternCrate {
                src: ExternCrateSource::Path,
                span,
                // to have the least priority in `update_extern_crate`
                path_len: usize::MAX,
                dependency_of: LOCAL_CRATE,
            },
        );

        cnum
    }

    pub fn maybe_process_path_extern(&mut self, name: Symbol, span: Span) -> Option<CrateNum> {
        self.maybe_resolve_crate(name, span, DepKind::Explicit, None).ok()
    }
}
