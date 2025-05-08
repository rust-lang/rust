//! Validates all used crates and extern libraries and loads their metadata

use std::error::Error;
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;
use std::{cmp, env, iter};

use rustc_ast::expand::allocator::{AllocatorKind, alloc_error_handler_name, global_fn_name};
use rustc_ast::{self as ast, *};
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::owned_slice::OwnedSlice;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::{self, FreezeReadGuard, FreezeWriteGuard};
use rustc_errors::DiagCtxtHandle;
use rustc_expand::base::SyntaxExtension;
use rustc_fs_util::try_canonicalize;
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, LOCAL_CRATE, LocalDefId, StableCrateId};
use rustc_hir::definitions::Definitions;
use rustc_index::IndexVec;
use rustc_middle::bug;
use rustc_middle::ty::data_structures::IndexSet;
use rustc_middle::ty::{TyCtxt, TyCtxtFeed};
use rustc_proc_macro::bridge::client::ProcMacro;
use rustc_session::config::{
    self, CrateType, ExtendedTargetModifierInfo, ExternLocation, OptionsTargetModifiers,
    TargetModifier,
};
use rustc_session::cstore::{CrateDepKind, CrateSource, ExternCrate, ExternCrateSource};
use rustc_session::lint::{self, BuiltinLintDiag};
use rustc_session::output::validate_crate_name;
use rustc_session::search_paths::PathKind;
use rustc_span::edition::Edition;
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, sym};
use rustc_target::spec::{PanicStrategy, Target, TargetTuple};
use tracing::{debug, info, trace};

use crate::errors;
use crate::locator::{CrateError, CrateLocator, CratePaths};
use crate::rmeta::{
    CrateDep, CrateMetadata, CrateNumMap, CrateRoot, MetadataBlob, TargetModifiers,
};

/// The backend's way to give the crate store access to the metadata in a library.
/// Note that it returns the raw metadata bytes stored in the library file, whether
/// it is compressed, uncompressed, some weird mix, etc.
/// rmeta files are backend independent and not handled here.
pub trait MetadataLoader {
    fn get_rlib_metadata(&self, target: &Target, filename: &Path) -> Result<OwnedSlice, String>;
    fn get_dylib_metadata(&self, target: &Target, filename: &Path) -> Result<OwnedSlice, String>;
}

pub type MetadataLoaderDyn = dyn MetadataLoader + Send + Sync + sync::DynSend + sync::DynSync;

pub struct CStore {
    metadata_loader: Box<MetadataLoaderDyn>,

    metas: IndexVec<CrateNum, Option<Box<CrateMetadata>>>,
    injected_panic_runtime: Option<CrateNum>,
    /// This crate needs an allocator and either provides it itself, or finds it in a dependency.
    /// If the above is true, then this field denotes the kind of the found allocator.
    allocator_kind: Option<AllocatorKind>,
    /// This crate needs an allocation error handler and either provides it itself, or finds it in a dependency.
    /// If the above is true, then this field denotes the kind of the found allocator.
    alloc_error_handler_kind: Option<AllocatorKind>,
    /// This crate has a `#[global_allocator]` item.
    has_global_allocator: bool,
    /// This crate has a `#[alloc_error_handler]` item.
    has_alloc_error_handler: bool,

    /// Unused externs of the crate
    unused_externs: Vec<Symbol>,
}

impl std::fmt::Debug for CStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CStore").finish_non_exhaustive()
    }
}

pub struct CrateLoader<'a, 'tcx: 'a> {
    // Immutable configuration.
    tcx: TyCtxt<'tcx>,
    // Mutable output.
    cstore: &'a mut CStore,
    used_extern_options: &'a mut FxHashSet<Symbol>,
}

impl<'a, 'tcx> std::ops::Deref for CrateLoader<'a, 'tcx> {
    type Target = TyCtxt<'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.tcx
    }
}

impl<'a, 'tcx> CrateLoader<'a, 'tcx> {
    fn dcx(&self) -> DiagCtxtHandle<'tcx> {
        self.tcx.dcx()
    }
}

pub enum LoadedMacro {
    MacroDef {
        def: MacroDef,
        ident: Ident,
        attrs: Vec<hir::Attribute>,
        span: Span,
        edition: Edition,
    },
    ProcMacro(SyntaxExtension),
}

pub(crate) struct Library {
    pub source: CrateSource,
    pub metadata: MetadataBlob,
}

enum LoadResult {
    Previous(CrateNum),
    Loaded(Library),
}

/// A reference to `CrateMetadata` that can also give access to whole crate store when necessary.
#[derive(Clone, Copy)]
pub(crate) struct CrateMetadataRef<'a> {
    pub cdata: &'a CrateMetadata,
    pub cstore: &'a CStore,
}

impl std::ops::Deref for CrateMetadataRef<'_> {
    type Target = CrateMetadata;

    fn deref(&self) -> &Self::Target {
        self.cdata
    }
}

struct CrateDump<'a>(&'a CStore);

impl<'a> std::fmt::Debug for CrateDump<'a> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(fmt, "resolved crates:")?;
        for (cnum, data) in self.0.iter_crate_data() {
            writeln!(fmt, "  name: {}", data.name())?;
            writeln!(fmt, "  cnum: {cnum}")?;
            writeln!(fmt, "  hash: {}", data.hash())?;
            writeln!(fmt, "  reqd: {:?}", data.dep_kind())?;
            writeln!(fmt, "  priv: {:?}", data.is_private_dep())?;
            let CrateSource { dylib, rlib, rmeta, sdylib_interface } = data.source();
            if let Some(dylib) = dylib {
                writeln!(fmt, "  dylib: {}", dylib.0.display())?;
            }
            if let Some(rlib) = rlib {
                writeln!(fmt, "   rlib: {}", rlib.0.display())?;
            }
            if let Some(rmeta) = rmeta {
                writeln!(fmt, "   rmeta: {}", rmeta.0.display())?;
            }
            if let Some(sdylib_interface) = sdylib_interface {
                writeln!(fmt, "   sdylib interface: {}", sdylib_interface.0.display())?;
            }
        }
        Ok(())
    }
}

/// Reason that a crate is being sourced as a dependency.
#[derive(Clone, Copy)]
enum CrateOrigin<'a> {
    /// This crate was a dependency of another crate.
    IndirectDependency {
        /// Where this dependency was included from.
        dep_root: &'a CratePaths,
        /// True if the parent is private, meaning the dependent should also be private.
        parent_private: bool,
        /// Dependency info about this crate.
        dep: &'a CrateDep,
    },
    /// Injected by `rustc`.
    Injected,
    /// Provided by `extern crate foo` or as part of the extern prelude.
    Extern,
}

impl<'a> CrateOrigin<'a> {
    /// Return the dependency root, if any.
    fn dep_root(&self) -> Option<&'a CratePaths> {
        match self {
            CrateOrigin::IndirectDependency { dep_root, .. } => Some(dep_root),
            _ => None,
        }
    }

    /// Return dependency information, if any.
    fn dep(&self) -> Option<&'a CrateDep> {
        match self {
            CrateOrigin::IndirectDependency { dep, .. } => Some(dep),
            _ => None,
        }
    }

    /// `Some(true)` if the dependency is private or its parent is private, `Some(false)` if the
    /// dependency is not private, `None` if it could not be determined.
    fn private_dep(&self) -> Option<bool> {
        match self {
            CrateOrigin::IndirectDependency { parent_private, dep, .. } => {
                Some(dep.is_private || *parent_private)
            }
            _ => None,
        }
    }
}

impl CStore {
    pub fn from_tcx(tcx: TyCtxt<'_>) -> FreezeReadGuard<'_, CStore> {
        FreezeReadGuard::map(tcx.untracked().cstore.read(), |cstore| {
            cstore.as_any().downcast_ref::<CStore>().expect("`tcx.cstore` is not a `CStore`")
        })
    }

    pub fn from_tcx_mut(tcx: TyCtxt<'_>) -> FreezeWriteGuard<'_, CStore> {
        FreezeWriteGuard::map(tcx.untracked().cstore.write(), |cstore| {
            cstore.untracked_as_any().downcast_mut().expect("`tcx.cstore` is not a `CStore`")
        })
    }

    fn intern_stable_crate_id<'tcx>(
        &mut self,
        root: &CrateRoot,
        tcx: TyCtxt<'tcx>,
    ) -> Result<TyCtxtFeed<'tcx, CrateNum>, CrateError> {
        assert_eq!(self.metas.len(), tcx.untracked().stable_crate_ids.read().len());
        let num = tcx.create_crate_num(root.stable_crate_id()).map_err(|existing| {
            // Check for (potential) conflicts with the local crate
            if existing == LOCAL_CRATE {
                CrateError::SymbolConflictsCurrent(root.name())
            } else if let Some(crate_name1) = self.metas[existing].as_ref().map(|data| data.name())
            {
                let crate_name0 = root.name();
                CrateError::StableCrateIdCollision(crate_name0, crate_name1)
            } else {
                CrateError::NotFound(root.name())
            }
        })?;

        self.metas.push(None);
        Ok(num)
    }

    pub fn has_crate_data(&self, cnum: CrateNum) -> bool {
        self.metas[cnum].is_some()
    }

    pub(crate) fn get_crate_data(&self, cnum: CrateNum) -> CrateMetadataRef<'_> {
        let cdata = self.metas[cnum]
            .as_ref()
            .unwrap_or_else(|| panic!("Failed to get crate data for {cnum:?}"));
        CrateMetadataRef { cdata, cstore: self }
    }

    pub(crate) fn get_crate_data_mut(&mut self, cnum: CrateNum) -> &mut CrateMetadata {
        self.metas[cnum].as_mut().unwrap_or_else(|| panic!("Failed to get crate data for {cnum:?}"))
    }

    fn set_crate_data(&mut self, cnum: CrateNum, data: CrateMetadata) {
        assert!(self.metas[cnum].is_none(), "Overwriting crate metadata entry");
        self.metas[cnum] = Some(Box::new(data));
    }

    pub(crate) fn iter_crate_data(&self) -> impl Iterator<Item = (CrateNum, &CrateMetadata)> {
        self.metas
            .iter_enumerated()
            .filter_map(|(cnum, data)| data.as_deref().map(|data| (cnum, data)))
    }

    fn push_dependencies_in_postorder(&self, deps: &mut IndexSet<CrateNum>, cnum: CrateNum) {
        if !deps.contains(&cnum) {
            let data = self.get_crate_data(cnum);
            for dep in data.dependencies() {
                if dep != cnum {
                    self.push_dependencies_in_postorder(deps, dep);
                }
            }

            deps.insert(cnum);
        }
    }

    pub(crate) fn crate_dependencies_in_postorder(&self, cnum: CrateNum) -> IndexSet<CrateNum> {
        let mut deps = IndexSet::default();
        if cnum == LOCAL_CRATE {
            for (cnum, _) in self.iter_crate_data() {
                self.push_dependencies_in_postorder(&mut deps, cnum);
            }
        } else {
            self.push_dependencies_in_postorder(&mut deps, cnum);
        }
        deps
    }

    pub(crate) fn injected_panic_runtime(&self) -> Option<CrateNum> {
        self.injected_panic_runtime
    }

    pub(crate) fn allocator_kind(&self) -> Option<AllocatorKind> {
        self.allocator_kind
    }

    pub(crate) fn alloc_error_handler_kind(&self) -> Option<AllocatorKind> {
        self.alloc_error_handler_kind
    }

    pub(crate) fn has_global_allocator(&self) -> bool {
        self.has_global_allocator
    }

    pub(crate) fn has_alloc_error_handler(&self) -> bool {
        self.has_alloc_error_handler
    }

    pub fn report_unused_deps(&self, tcx: TyCtxt<'_>) {
        let json_unused_externs = tcx.sess.opts.json_unused_externs;

        // We put the check for the option before the lint_level_at_node call
        // because the call mutates internal state and introducing it
        // leads to some ui tests failing.
        if !json_unused_externs.is_enabled() {
            return;
        }
        let level = tcx
            .lint_level_at_node(lint::builtin::UNUSED_CRATE_DEPENDENCIES, rustc_hir::CRATE_HIR_ID)
            .level;
        if level != lint::Level::Allow {
            let unused_externs =
                self.unused_externs.iter().map(|ident| ident.to_ident_string()).collect::<Vec<_>>();
            let unused_externs = unused_externs.iter().map(String::as_str).collect::<Vec<&str>>();
            tcx.dcx().emit_unused_externs(level, json_unused_externs.is_loud(), &unused_externs);
        }
    }

    fn report_target_modifiers_extended(
        tcx: TyCtxt<'_>,
        krate: &Crate,
        mods: &TargetModifiers,
        dep_mods: &TargetModifiers,
        data: &CrateMetadata,
    ) {
        let span = krate.spans.inner_span.shrink_to_lo();
        let allowed_flag_mismatches = &tcx.sess.opts.cg.unsafe_allow_abi_mismatch;
        let local_crate = tcx.crate_name(LOCAL_CRATE);
        let tmod_extender = |tmod: &TargetModifier| (tmod.extend(), tmod.clone());
        let report_diff = |prefix: &String,
                           opt_name: &String,
                           flag_local_value: Option<&String>,
                           flag_extern_value: Option<&String>| {
            if allowed_flag_mismatches.contains(&opt_name) {
                return;
            }
            let extern_crate = data.name();
            let flag_name = opt_name.clone();
            let flag_name_prefixed = format!("-{}{}", prefix, opt_name);

            match (flag_local_value, flag_extern_value) {
                (Some(local_value), Some(extern_value)) => {
                    tcx.dcx().emit_err(errors::IncompatibleTargetModifiers {
                        span,
                        extern_crate,
                        local_crate,
                        flag_name,
                        flag_name_prefixed,
                        local_value: local_value.to_string(),
                        extern_value: extern_value.to_string(),
                    })
                }
                (None, Some(extern_value)) => {
                    tcx.dcx().emit_err(errors::IncompatibleTargetModifiersLMissed {
                        span,
                        extern_crate,
                        local_crate,
                        flag_name,
                        flag_name_prefixed,
                        extern_value: extern_value.to_string(),
                    })
                }
                (Some(local_value), None) => {
                    tcx.dcx().emit_err(errors::IncompatibleTargetModifiersRMissed {
                        span,
                        extern_crate,
                        local_crate,
                        flag_name,
                        flag_name_prefixed,
                        local_value: local_value.to_string(),
                    })
                }
                (None, None) => panic!("Incorrect target modifiers report_diff(None, None)"),
            };
        };
        let mut it1 = mods.iter().map(tmod_extender);
        let mut it2 = dep_mods.iter().map(tmod_extender);
        let mut left_name_val: Option<(ExtendedTargetModifierInfo, TargetModifier)> = None;
        let mut right_name_val: Option<(ExtendedTargetModifierInfo, TargetModifier)> = None;
        loop {
            left_name_val = left_name_val.or_else(|| it1.next());
            right_name_val = right_name_val.or_else(|| it2.next());
            match (&left_name_val, &right_name_val) {
                (Some(l), Some(r)) => match l.1.opt.cmp(&r.1.opt) {
                    cmp::Ordering::Equal => {
                        if l.0.tech_value != r.0.tech_value {
                            report_diff(
                                &l.0.prefix,
                                &l.0.name,
                                Some(&l.1.value_name),
                                Some(&r.1.value_name),
                            );
                        }
                        left_name_val = None;
                        right_name_val = None;
                    }
                    cmp::Ordering::Greater => {
                        report_diff(&r.0.prefix, &r.0.name, None, Some(&r.1.value_name));
                        right_name_val = None;
                    }
                    cmp::Ordering::Less => {
                        report_diff(&l.0.prefix, &l.0.name, Some(&l.1.value_name), None);
                        left_name_val = None;
                    }
                },
                (Some(l), None) => {
                    report_diff(&l.0.prefix, &l.0.name, Some(&l.1.value_name), None);
                    left_name_val = None;
                }
                (None, Some(r)) => {
                    report_diff(&r.0.prefix, &r.0.name, None, Some(&r.1.value_name));
                    right_name_val = None;
                }
                (None, None) => break,
            }
        }
    }

    pub fn report_incompatible_target_modifiers(&self, tcx: TyCtxt<'_>, krate: &Crate) {
        for flag_name in &tcx.sess.opts.cg.unsafe_allow_abi_mismatch {
            if !OptionsTargetModifiers::is_target_modifier(flag_name) {
                tcx.dcx().emit_err(errors::UnknownTargetModifierUnsafeAllowed {
                    span: krate.spans.inner_span.shrink_to_lo(),
                    flag_name: flag_name.clone(),
                });
            }
        }
        let mods = tcx.sess.opts.gather_target_modifiers();
        for (_cnum, data) in self.iter_crate_data() {
            if data.is_proc_macro_crate() {
                continue;
            }
            let dep_mods = data.target_modifiers();
            if mods != dep_mods {
                Self::report_target_modifiers_extended(tcx, krate, &mods, &dep_mods, data);
            }
        }
    }

    // Report about async drop types in dependency if async drop feature is disabled
    pub fn report_incompatible_async_drop_feature(&self, tcx: TyCtxt<'_>, krate: &Crate) {
        if tcx.features().async_drop() {
            return;
        }
        for (_cnum, data) in self.iter_crate_data() {
            if data.is_proc_macro_crate() {
                continue;
            }
            if data.has_async_drops() {
                let extern_crate = data.name();
                let local_crate = tcx.crate_name(LOCAL_CRATE);
                tcx.dcx().emit_warn(errors::AsyncDropTypesInDependency {
                    span: krate.spans.inner_span.shrink_to_lo(),
                    extern_crate,
                    local_crate,
                });
            }
        }
    }

    pub fn new(metadata_loader: Box<MetadataLoaderDyn>) -> CStore {
        CStore {
            metadata_loader,
            // We add an empty entry for LOCAL_CRATE (which maps to zero) in
            // order to make array indices in `metas` match with the
            // corresponding `CrateNum`. This first entry will always remain
            // `None`.
            metas: IndexVec::from_iter(iter::once(None)),
            injected_panic_runtime: None,
            allocator_kind: None,
            alloc_error_handler_kind: None,
            has_global_allocator: false,
            has_alloc_error_handler: false,
            unused_externs: Vec::new(),
        }
    }
}

impl<'a, 'tcx> CrateLoader<'a, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        cstore: &'a mut CStore,
        used_extern_options: &'a mut FxHashSet<Symbol>,
    ) -> Self {
        CrateLoader { tcx, cstore, used_extern_options }
    }

    fn existing_match(&self, name: Symbol, hash: Option<Svh>, kind: PathKind) -> Option<CrateNum> {
        for (cnum, data) in self.cstore.iter_crate_data() {
            if data.name() != name {
                trace!("{} did not match {}", data.name(), name);
                continue;
            }

            match hash {
                Some(hash) if hash == data.hash() => return Some(cnum),
                Some(hash) => {
                    debug!("actual hash {} did not match expected {}", hash, data.hash());
                    continue;
                }
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
            if let Some(entry) = self.sess.opts.externs.get(name.as_str()) {
                // Only use `--extern crate_name=path` here, not `--extern crate_name`.
                if let Some(mut files) = entry.files() {
                    if files.any(|l| {
                        let l = l.canonicalized();
                        source.dylib.as_ref().map(|(p, _)| p) == Some(l)
                            || source.rlib.as_ref().map(|(p, _)| p) == Some(l)
                            || source.rmeta.as_ref().map(|(p, _)| p) == Some(l)
                    }) {
                        return Some(cnum);
                    }
                }
                continue;
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
                return Some(cnum);
            } else {
                debug!(
                    "failed to load existing crate {}; kind {:?} did not match prev_kind {:?}",
                    name, kind, prev_kind
                );
            }
        }

        None
    }

    /// Determine whether a dependency should be considered private.
    ///
    /// Dependencies are private if they get extern option specified, e.g. `--extern priv:mycrate`.
    /// This is stored in metadata, so `private_dep`  can be correctly set during load. A `Some`
    /// value for `private_dep` indicates that the crate is known to be private or public (note
    /// that any `None` or `Some(false)` use of the same crate will make it public).
    ///
    /// Sometimes the directly dependent crate is not specified by `--extern`, in this case,
    /// `private-dep` is none during loading. This is equivalent to the scenario where the
    /// command parameter is set to `public-dependency`
    fn is_private_dep(
        &self,
        name: Symbol,
        private_dep: Option<bool>,
        origin: CrateOrigin<'_>,
    ) -> bool {
        if matches!(origin, CrateOrigin::Injected) {
            return true;
        }

        let extern_private = self.sess.opts.externs.get(name.as_str()).map(|e| e.is_private_dep);
        match (extern_private, private_dep) {
            // Explicit non-private via `--extern`, explicit non-private from metadata, or
            // unspecified with default to public.
            (Some(false), _) | (_, Some(false)) | (None, None) => false,
            // Marked private via `--extern priv:mycrate` or in metadata.
            (Some(true) | None, Some(true) | None) => true,
        }
    }

    fn register_crate(
        &mut self,
        host_lib: Option<Library>,
        origin: CrateOrigin<'_>,
        lib: Library,
        dep_kind: CrateDepKind,
        name: Symbol,
        private_dep: Option<bool>,
    ) -> Result<CrateNum, CrateError> {
        let _prof_timer =
            self.sess.prof.generic_activity_with_arg("metadata_register_crate", name.as_str());

        let Library { source, metadata } = lib;
        let crate_root = metadata.get_root();
        let host_hash = host_lib.as_ref().map(|lib| lib.metadata.get_root().hash());
        let private_dep = self.is_private_dep(name, private_dep, origin);

        // Claim this crate number and cache it
        let feed = self.cstore.intern_stable_crate_id(&crate_root, self.tcx)?;
        let cnum = feed.key();

        info!(
            "register crate `{}` (cnum = {}. private_dep = {})",
            crate_root.name(),
            cnum,
            private_dep
        );

        // Maintain a reference to the top most crate.
        // Stash paths for top-most crate locally if necessary.
        let crate_paths;
        let dep_root = if let Some(dep_root) = origin.dep_root() {
            dep_root
        } else {
            crate_paths = CratePaths::new(crate_root.name(), source.clone());
            &crate_paths
        };

        let cnum_map =
            self.resolve_crate_deps(dep_root, &crate_root, &metadata, cnum, dep_kind, private_dep)?;

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
            Some(self.dlsym_proc_macros(&dlsym_dylib.0, dlsym_root.stable_crate_id())?)
        } else {
            None
        };

        let crate_metadata = CrateMetadata::new(
            self.sess,
            self.cstore,
            metadata,
            crate_root,
            raw_proc_macros,
            cnum,
            cnum_map,
            dep_kind,
            source,
            private_dep,
            host_hash,
        );

        self.cstore.set_crate_data(cnum, crate_metadata);

        Ok(cnum)
    }

    fn load_proc_macro<'b>(
        &self,
        locator: &mut CrateLocator<'b>,
        path_kind: PathKind,
        host_hash: Option<Svh>,
    ) -> Result<Option<(LoadResult, Option<Library>)>, CrateError>
    where
        'a: 'b,
    {
        // Use a new crate locator so trying to load a proc macro doesn't affect the error
        // message we emit
        let mut proc_macro_locator = locator.clone();

        // Try to load a proc macro
        proc_macro_locator.is_proc_macro = true;

        // Load the proc macro crate for the target
        let (locator, target_result) = if self.sess.opts.unstable_opts.dual_proc_macros {
            proc_macro_locator.reset();
            let result = match self.load(&mut proc_macro_locator)? {
                Some(LoadResult::Previous(cnum)) => {
                    return Ok(Some((LoadResult::Previous(cnum), None)));
                }
                Some(LoadResult::Loaded(library)) => Some(LoadResult::Loaded(library)),
                None => return Ok(None),
            };
            locator.hash = host_hash;
            // Use the locator when looking for the host proc macro crate, as that is required
            // so we want it to affect the error message
            (locator, result)
        } else {
            (&mut proc_macro_locator, None)
        };

        // Load the proc macro crate for the host

        locator.reset();
        locator.is_proc_macro = true;
        locator.target = &self.sess.host;
        locator.tuple = TargetTuple::from_tuple(config::host_tuple());
        locator.filesearch = self.sess.host_filesearch();
        locator.path_kind = path_kind;

        let Some(host_result) = self.load(locator)? else {
            return Ok(None);
        };

        Ok(Some(if self.sess.opts.unstable_opts.dual_proc_macros {
            let host_result = match host_result {
                LoadResult::Previous(..) => {
                    panic!("host and target proc macros must be loaded in lock-step")
                }
                LoadResult::Loaded(library) => library,
            };
            (target_result.unwrap(), Some(host_result))
        } else {
            (host_result, None)
        }))
    }

    fn resolve_crate(
        &mut self,
        name: Symbol,
        span: Span,
        dep_kind: CrateDepKind,
        origin: CrateOrigin<'_>,
    ) -> Option<CrateNum> {
        self.used_extern_options.insert(name);
        match self.maybe_resolve_crate(name, dep_kind, origin) {
            Ok(cnum) => {
                self.cstore.set_used_recursively(cnum);
                Some(cnum)
            }
            Err(err) => {
                debug!("failed to resolve crate {} {:?}", name, dep_kind);
                let missing_core = self
                    .maybe_resolve_crate(sym::core, CrateDepKind::Explicit, CrateOrigin::Extern)
                    .is_err();
                err.report(self.sess, span, missing_core);
                None
            }
        }
    }

    fn maybe_resolve_crate<'b>(
        &'b mut self,
        name: Symbol,
        mut dep_kind: CrateDepKind,
        origin: CrateOrigin<'b>,
    ) -> Result<CrateNum, CrateError> {
        info!("resolving crate `{}`", name);
        if !name.as_str().is_ascii() {
            return Err(CrateError::NonAsciiName(name));
        }

        let dep_root = origin.dep_root();
        let dep = origin.dep();
        let hash = dep.map(|d| d.hash);
        let host_hash = dep.map(|d| d.host_hash).flatten();
        let extra_filename = dep.map(|d| &d.extra_filename[..]);
        let path_kind = if dep.is_some() { PathKind::Dependency } else { PathKind::Crate };
        let private_dep = origin.private_dep();

        let result = if let Some(cnum) = self.existing_match(name, hash, path_kind) {
            (LoadResult::Previous(cnum), None)
        } else {
            info!("falling back to a load");
            let mut locator = CrateLocator::new(
                self.sess,
                &*self.cstore.metadata_loader,
                name,
                // The all loop is because `--crate-type=rlib --crate-type=rlib` is
                // legal and produces both inside this type.
                self.tcx.crate_types().iter().all(|c| *c == CrateType::Rlib),
                hash,
                extra_filename,
                path_kind,
            );

            match self.load(&mut locator)? {
                Some(res) => (res, None),
                None => {
                    info!("falling back to loading proc_macro");
                    dep_kind = CrateDepKind::MacrosOnly;
                    match self.load_proc_macro(&mut locator, path_kind, host_hash)? {
                        Some(res) => res,
                        None => return Err(locator.into_error(dep_root.cloned())),
                    }
                }
            }
        };

        match result {
            (LoadResult::Previous(cnum), None) => {
                info!("library for `{}` was loaded previously, cnum {cnum}", name);
                // When `private_dep` is none, it indicates the directly dependent crate. If it is
                // not specified by `--extern` on command line parameters, it may be
                // `private-dependency` when `register_crate` is called for the first time. Then it must be updated to
                // `public-dependency` here.
                let private_dep = self.is_private_dep(name, private_dep, origin);
                let data = self.cstore.get_crate_data_mut(cnum);
                if data.is_proc_macro_crate() {
                    dep_kind = CrateDepKind::MacrosOnly;
                }
                data.set_dep_kind(cmp::max(data.dep_kind(), dep_kind));
                data.update_and_private_dep(private_dep);
                Ok(cnum)
            }
            (LoadResult::Loaded(library), host_library) => {
                info!("register newly loaded library for `{}`", name);
                self.register_crate(host_library, origin, library, dep_kind, name, private_dep)
            }
            _ => panic!(),
        }
    }

    fn load(&self, locator: &mut CrateLocator<'_>) -> Result<Option<LoadResult>, CrateError> {
        let Some(library) = locator.maybe_load_library_crate()? else {
            return Ok(None);
        };

        // In the case that we're loading a crate, but not matching
        // against a hash, we could load a crate which has the same hash
        // as an already loaded crate. If this is the case prevent
        // duplicates by just using the first crate.
        let root = library.metadata.get_root();
        let mut result = LoadResult::Loaded(library);
        for (cnum, data) in self.cstore.iter_crate_data() {
            if data.name() == root.name() && root.hash() == data.hash() {
                assert!(locator.hash.is_none());
                info!("load success, going to previous cnum: {}", cnum);
                result = LoadResult::Previous(cnum);
                break;
            }
        }
        Ok(Some(result))
    }

    /// Go through the crate metadata and load any crates that it references.
    fn resolve_crate_deps(
        &mut self,
        dep_root: &CratePaths,
        crate_root: &CrateRoot,
        metadata: &MetadataBlob,
        krate: CrateNum,
        dep_kind: CrateDepKind,
        parent_is_private: bool,
    ) -> Result<CrateNumMap, CrateError> {
        debug!(
            "resolving deps of external crate `{}` with dep root `{}`",
            crate_root.name(),
            dep_root.name
        );
        if crate_root.is_proc_macro_crate() {
            return Ok(CrateNumMap::new());
        }

        // The map from crate numbers in the crate we're resolving to local crate numbers.
        // We map 0 and all other holes in the map to our parent crate. The "additional"
        // self-dependencies should be harmless.
        let deps = crate_root.decode_crate_deps(metadata);
        let mut crate_num_map = CrateNumMap::with_capacity(1 + deps.len());
        crate_num_map.push(krate);
        for dep in deps {
            info!(
                "resolving dep `{}`->`{}` hash: `{}` extra filename: `{}` private {}",
                crate_root.name(),
                dep.name,
                dep.hash,
                dep.extra_filename,
                dep.is_private,
            );
            let dep_kind = match dep_kind {
                CrateDepKind::MacrosOnly => CrateDepKind::MacrosOnly,
                _ => dep.kind,
            };
            let cnum = self.maybe_resolve_crate(
                dep.name,
                dep_kind,
                CrateOrigin::IndirectDependency {
                    dep_root,
                    parent_private: parent_is_private,
                    dep: &dep,
                },
            )?;
            crate_num_map.push(cnum);
        }

        debug!("resolve_crate_deps: cnum_map for {:?} is {:?}", krate, crate_num_map);
        Ok(crate_num_map)
    }

    fn dlsym_proc_macros(
        &self,
        path: &Path,
        stable_crate_id: StableCrateId,
    ) -> Result<&'static [ProcMacro], CrateError> {
        let sym_name = self.sess.generate_proc_macro_decls_symbol(stable_crate_id);
        debug!("trying to dlsym proc_macros {} for symbol `{}`", path.display(), sym_name);

        unsafe {
            let result = load_symbol_from_dylib::<*const &[ProcMacro]>(path, &sym_name);
            match result {
                Ok(result) => {
                    debug!("loaded dlsym proc_macros {} for symbol `{}`", path.display(), sym_name);
                    Ok(*result)
                }
                Err(err) => {
                    debug!(
                        "failed to dlsym proc_macros {} for symbol `{}`",
                        path.display(),
                        sym_name
                    );
                    Err(err.into())
                }
            }
        }
    }

    fn inject_panic_runtime(&mut self, krate: &ast::Crate) {
        // If we're only compiling an rlib, then there's no need to select a
        // panic runtime, so we just skip this section entirely.
        let only_rlib = self.tcx.crate_types().iter().all(|ct| *ct == CrateType::Rlib);
        if only_rlib {
            info!("panic runtime injection skipped, only generating rlib");
            return;
        }

        // If we need a panic runtime, we try to find an existing one here. At
        // the same time we perform some general validation of the DAG we've got
        // going such as ensuring everything has a compatible panic strategy.
        let mut needs_panic_runtime = attr::contains_name(&krate.attrs, sym::needs_panic_runtime);
        for (_cnum, data) in self.cstore.iter_crate_data() {
            needs_panic_runtime |= data.needs_panic_runtime();
        }

        // If we just don't need a panic runtime at all, then we're done here
        // and there's nothing else to do.
        if !needs_panic_runtime {
            return;
        }

        // By this point we know that we need a panic runtime. Here we just load
        // an appropriate default runtime for our panic strategy.
        //
        // We may resolve to an already loaded crate (as the crate may not have
        // been explicitly linked prior to this), but this is fine.
        //
        // Also note that we have yet to perform validation of the crate graph
        // in terms of everyone has a compatible panic runtime format, that's
        // performed later as part of the `dependency_format` module.
        let desired_strategy = self.sess.panic_strategy();
        let name = match desired_strategy {
            PanicStrategy::Unwind => sym::panic_unwind,
            PanicStrategy::Abort => sym::panic_abort,
        };
        info!("panic runtime not found -- loading {}", name);

        let Some(cnum) =
            self.resolve_crate(name, DUMMY_SP, CrateDepKind::Implicit, CrateOrigin::Injected)
        else {
            return;
        };
        let data = self.cstore.get_crate_data(cnum);

        // Sanity check the loaded crate to ensure it is indeed a panic runtime
        // and the panic strategy is indeed what we thought it was.
        if !data.is_panic_runtime() {
            self.dcx().emit_err(errors::CrateNotPanicRuntime { crate_name: name });
        }
        if data.required_panic_strategy() != Some(desired_strategy) {
            self.dcx()
                .emit_err(errors::NoPanicStrategy { crate_name: name, strategy: desired_strategy });
        }

        self.cstore.injected_panic_runtime = Some(cnum);
    }

    fn inject_profiler_runtime(&mut self) {
        let needs_profiler_runtime =
            self.sess.instrument_coverage() || self.sess.opts.cg.profile_generate.enabled();
        if !needs_profiler_runtime || self.sess.opts.unstable_opts.no_profiler_runtime {
            return;
        }

        info!("loading profiler");

        let name = Symbol::intern(&self.sess.opts.unstable_opts.profiler_runtime);
        let Some(cnum) =
            self.resolve_crate(name, DUMMY_SP, CrateDepKind::Implicit, CrateOrigin::Injected)
        else {
            return;
        };
        let data = self.cstore.get_crate_data(cnum);

        // Sanity check the loaded crate to ensure it is indeed a profiler runtime
        if !data.is_profiler_runtime() {
            self.dcx().emit_err(errors::NotProfilerRuntime { crate_name: name });
        }
    }

    fn inject_allocator_crate(&mut self, krate: &ast::Crate) {
        self.cstore.has_global_allocator =
            match &*fn_spans(krate, Symbol::intern(&global_fn_name(sym::alloc))) {
                [span1, span2, ..] => {
                    self.dcx()
                        .emit_err(errors::NoMultipleGlobalAlloc { span2: *span2, span1: *span1 });
                    true
                }
                spans => !spans.is_empty(),
            };
        self.cstore.has_alloc_error_handler = match &*fn_spans(
            krate,
            Symbol::intern(alloc_error_handler_name(AllocatorKind::Global)),
        ) {
            [span1, span2, ..] => {
                self.dcx()
                    .emit_err(errors::NoMultipleAllocErrorHandler { span2: *span2, span1: *span1 });
                true
            }
            spans => !spans.is_empty(),
        };

        // Check to see if we actually need an allocator. This desire comes
        // about through the `#![needs_allocator]` attribute and is typically
        // written down in liballoc.
        if !attr::contains_name(&krate.attrs, sym::needs_allocator)
            && !self.cstore.iter_crate_data().any(|(_, data)| data.needs_allocator())
        {
            return;
        }

        // At this point we've determined that we need an allocator. Let's see
        // if our compilation session actually needs an allocator based on what
        // we're emitting.
        let all_rlib = self.tcx.crate_types().iter().all(|ct| matches!(*ct, CrateType::Rlib));
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
        #[allow(rustc::symbol_intern_string_literal)]
        let this_crate = Symbol::intern("this crate");

        let mut global_allocator = self.cstore.has_global_allocator.then_some(this_crate);
        for (_, data) in self.cstore.iter_crate_data() {
            if data.has_global_allocator() {
                match global_allocator {
                    Some(other_crate) => {
                        self.dcx().emit_err(errors::ConflictingGlobalAlloc {
                            crate_name: data.name(),
                            other_crate_name: other_crate,
                        });
                    }
                    None => global_allocator = Some(data.name()),
                }
            }
        }
        let mut alloc_error_handler = self.cstore.has_alloc_error_handler.then_some(this_crate);
        for (_, data) in self.cstore.iter_crate_data() {
            if data.has_alloc_error_handler() {
                match alloc_error_handler {
                    Some(other_crate) => {
                        self.dcx().emit_err(errors::ConflictingAllocErrorHandler {
                            crate_name: data.name(),
                            other_crate_name: other_crate,
                        });
                    }
                    None => alloc_error_handler = Some(data.name()),
                }
            }
        }

        if global_allocator.is_some() {
            self.cstore.allocator_kind = Some(AllocatorKind::Global);
        } else {
            // Ok we haven't found a global allocator but we still need an
            // allocator. At this point our allocator request is typically fulfilled
            // by the standard library, denoted by the `#![default_lib_allocator]`
            // attribute.
            if !attr::contains_name(&krate.attrs, sym::default_lib_allocator)
                && !self.cstore.iter_crate_data().any(|(_, data)| data.has_default_lib_allocator())
            {
                self.dcx().emit_err(errors::GlobalAllocRequired);
            }
            self.cstore.allocator_kind = Some(AllocatorKind::Default);
        }

        if alloc_error_handler.is_some() {
            self.cstore.alloc_error_handler_kind = Some(AllocatorKind::Global);
        } else {
            // The alloc crate provides a default allocation error handler if
            // one isn't specified.
            self.cstore.alloc_error_handler_kind = Some(AllocatorKind::Default);
        }
    }

    fn inject_forced_externs(&mut self) {
        for (name, entry) in self.sess.opts.externs.iter() {
            if entry.force {
                let name_interned = Symbol::intern(name);
                if !self.used_extern_options.contains(&name_interned) {
                    self.resolve_crate(
                        name_interned,
                        DUMMY_SP,
                        CrateDepKind::Explicit,
                        CrateOrigin::Extern,
                    );
                }
            }
        }
    }

    /// Inject the `compiler_builtins` crate if it is not already in the graph.
    fn inject_compiler_builtins(&mut self, krate: &ast::Crate) {
        // `compiler_builtins` does not get extern builtins, nor do `#![no_core]` crates
        if attr::contains_name(&krate.attrs, sym::compiler_builtins)
            || attr::contains_name(&krate.attrs, sym::no_core)
        {
            info!("`compiler_builtins` unneeded");
            return;
        }

        // If a `#![compiler_builtins]` crate already exists, avoid injecting it twice. This is
        // the common case since usually it appears as a dependency of `std` or `alloc`.
        for (cnum, cmeta) in self.cstore.iter_crate_data() {
            if cmeta.is_compiler_builtins() {
                info!("`compiler_builtins` already exists (cnum = {cnum}); skipping injection");
                return;
            }
        }

        // `compiler_builtins` is not yet in the graph; inject it. Error on resolution failure.
        let Some(cnum) = self.resolve_crate(
            sym::compiler_builtins,
            krate.spans.inner_span.shrink_to_lo(),
            CrateDepKind::Explicit,
            CrateOrigin::Injected,
        ) else {
            info!("`compiler_builtins` not resolved");
            return;
        };

        // Sanity check that the loaded crate is `#![compiler_builtins]`
        let cmeta = self.cstore.get_crate_data(cnum);
        if !cmeta.is_compiler_builtins() {
            self.dcx().emit_err(errors::CrateNotCompilerBuiltins { crate_name: cmeta.name() });
        }
    }

    fn report_unused_deps(&mut self, krate: &ast::Crate) {
        // Make a point span rather than covering the whole file
        let span = krate.spans.inner_span.shrink_to_lo();
        // Complain about anything left over
        for (name, entry) in self.sess.opts.externs.iter() {
            if let ExternLocation::FoundInLibrarySearchDirectories = entry.location {
                // Don't worry about pathless `--extern foo` sysroot references
                continue;
            }
            if entry.nounused_dep || entry.force {
                // We're not worried about this one
                continue;
            }
            let name_interned = Symbol::intern(name);
            if self.used_extern_options.contains(&name_interned) {
                continue;
            }

            // Got a real unused --extern
            if self.sess.opts.json_unused_externs.is_enabled() {
                self.cstore.unused_externs.push(name_interned);
                continue;
            }

            self.sess.psess.buffer_lint(
                lint::builtin::UNUSED_CRATE_DEPENDENCIES,
                span,
                ast::CRATE_NODE_ID,
                BuiltinLintDiag::UnusedCrateDependency {
                    extern_crate: name_interned,
                    local_crate: self.tcx.crate_name(LOCAL_CRATE),
                },
            );
        }
    }

    fn report_future_incompatible_deps(&self, krate: &ast::Crate) {
        let name = self.tcx.crate_name(LOCAL_CRATE);

        if name.as_str() == "wasm_bindgen" {
            let major = env::var("CARGO_PKG_VERSION_MAJOR")
                .ok()
                .and_then(|major| u64::from_str(&major).ok());
            let minor = env::var("CARGO_PKG_VERSION_MINOR")
                .ok()
                .and_then(|minor| u64::from_str(&minor).ok());
            let patch = env::var("CARGO_PKG_VERSION_PATCH")
                .ok()
                .and_then(|patch| u64::from_str(&patch).ok());

            match (major, minor, patch) {
                // v1 or bigger is valid.
                (Some(1..), _, _) => return,
                // v0.3 or bigger is valid.
                (Some(0), Some(3..), _) => return,
                // v0.2.88 or bigger is valid.
                (Some(0), Some(2), Some(88..)) => return,
                // Not using Cargo.
                (None, None, None) => return,
                _ => (),
            }

            // Make a point span rather than covering the whole file
            let span = krate.spans.inner_span.shrink_to_lo();

            self.sess.dcx().emit_err(errors::WasmCAbi { span });
        }
    }

    pub fn postprocess(&mut self, krate: &ast::Crate) {
        self.inject_compiler_builtins(krate);
        self.inject_forced_externs();
        self.inject_profiler_runtime();
        self.inject_allocator_crate(krate);
        self.inject_panic_runtime(krate);

        self.report_unused_deps(krate);
        self.report_future_incompatible_deps(krate);

        info!("{:?}", CrateDump(self.cstore));
    }

    /// Process an `extern crate foo` AST node.
    pub fn process_extern_crate(
        &mut self,
        item: &ast::Item,
        def_id: LocalDefId,
        definitions: &Definitions,
    ) -> Option<CrateNum> {
        match item.kind {
            ast::ItemKind::ExternCrate(orig_name, ident) => {
                debug!("resolving extern crate stmt. ident: {} orig_name: {:?}", ident, orig_name);
                let name = match orig_name {
                    Some(orig_name) => {
                        validate_crate_name(self.sess, orig_name, Some(item.span));
                        orig_name
                    }
                    None => ident.name,
                };
                let dep_kind = if attr::contains_name(&item.attrs, sym::no_link) {
                    CrateDepKind::MacrosOnly
                } else {
                    CrateDepKind::Explicit
                };

                let cnum = self.resolve_crate(name, item.span, dep_kind, CrateOrigin::Extern)?;

                let path_len = definitions.def_path(def_id).data.len();
                self.cstore.update_extern_crate(
                    cnum,
                    ExternCrate {
                        src: ExternCrateSource::Extern(def_id.to_def_id()),
                        span: item.span,
                        path_len,
                        dependency_of: LOCAL_CRATE,
                    },
                );
                Some(cnum)
            }
            _ => bug!(),
        }
    }

    pub fn process_path_extern(&mut self, name: Symbol, span: Span) -> Option<CrateNum> {
        let cnum = self.resolve_crate(name, span, CrateDepKind::Explicit, CrateOrigin::Extern)?;

        self.cstore.update_extern_crate(
            cnum,
            ExternCrate {
                src: ExternCrateSource::Path,
                span,
                // to have the least priority in `update_extern_crate`
                path_len: usize::MAX,
                dependency_of: LOCAL_CRATE,
            },
        );

        Some(cnum)
    }

    pub fn maybe_process_path_extern(&mut self, name: Symbol) -> Option<CrateNum> {
        self.maybe_resolve_crate(name, CrateDepKind::Explicit, CrateOrigin::Extern).ok()
    }
}

fn fn_spans(krate: &ast::Crate, name: Symbol) -> Vec<Span> {
    struct Finder {
        name: Symbol,
        spans: Vec<Span>,
    }
    impl<'ast> visit::Visitor<'ast> for Finder {
        fn visit_item(&mut self, item: &'ast ast::Item) {
            if let Some(ident) = item.kind.ident()
                && ident.name == self.name
                && attr::contains_name(&item.attrs, sym::rustc_std_internal_symbol)
            {
                self.spans.push(item.span);
            }
            visit::walk_item(self, item)
        }
    }

    let mut f = Finder { name, spans: Vec::new() };
    visit::walk_crate(&mut f, krate);
    f.spans
}

fn format_dlopen_err(e: &(dyn std::error::Error + 'static)) -> String {
    e.sources().map(|e| format!(": {e}")).collect()
}

fn attempt_load_dylib(path: &Path) -> Result<libloading::Library, libloading::Error> {
    #[cfg(target_os = "aix")]
    if let Some(ext) = path.extension()
        && ext.eq("a")
    {
        // On AIX, we ship all libraries as .a big_af archive
        // the expected format is lib<name>.a(libname.so) for the actual
        // dynamic library
        let library_name = path.file_stem().expect("expect a library name");
        let mut archive_member = std::ffi::OsString::from("a(");
        archive_member.push(library_name);
        archive_member.push(".so)");
        let new_path = path.with_extension(archive_member);

        // On AIX, we need RTLD_MEMBER to dlopen an archived shared
        let flags = libc::RTLD_LAZY | libc::RTLD_LOCAL | libc::RTLD_MEMBER;
        return unsafe { libloading::os::unix::Library::open(Some(&new_path), flags) }
            .map(|lib| lib.into());
    }

    unsafe { libloading::Library::new(&path) }
}

// On Windows the compiler would sometimes intermittently fail to open the
// proc-macro DLL with `Error::LoadLibraryExW`. It is suspected that something in the
// system still holds a lock on the file, so we retry a few times before calling it
// an error.
fn load_dylib(path: &Path, max_attempts: usize) -> Result<libloading::Library, String> {
    assert!(max_attempts > 0);

    let mut last_error = None;

    for attempt in 0..max_attempts {
        debug!("Attempt to load proc-macro `{}`.", path.display());
        match attempt_load_dylib(path) {
            Ok(lib) => {
                if attempt > 0 {
                    debug!(
                        "Loaded proc-macro `{}` after {} attempts.",
                        path.display(),
                        attempt + 1
                    );
                }
                return Ok(lib);
            }
            Err(err) => {
                // Only try to recover from this specific error.
                if !matches!(err, libloading::Error::LoadLibraryExW { .. }) {
                    debug!("Failed to load proc-macro `{}`. Not retrying", path.display());
                    let err = format_dlopen_err(&err);
                    // We include the path of the dylib in the error ourselves, so
                    // if it's in the error, we strip it.
                    if let Some(err) = err.strip_prefix(&format!(": {}", path.display())) {
                        return Err(err.to_string());
                    }
                    return Err(err);
                }

                last_error = Some(err);
                std::thread::sleep(Duration::from_millis(100));
                debug!("Failed to load proc-macro `{}`. Retrying.", path.display());
            }
        }
    }

    debug!("Failed to load proc-macro `{}` even after {} attempts.", path.display(), max_attempts);

    let last_error = last_error.unwrap();
    let message = if let Some(src) = last_error.source() {
        format!("{} ({src}) (retried {max_attempts} times)", format_dlopen_err(&last_error))
    } else {
        format!("{} (retried {max_attempts} times)", format_dlopen_err(&last_error))
    };
    Err(message)
}

pub enum DylibError {
    DlOpen(String, String),
    DlSym(String, String),
}

impl From<DylibError> for CrateError {
    fn from(err: DylibError) -> CrateError {
        match err {
            DylibError::DlOpen(path, err) => CrateError::DlOpen(path, err),
            DylibError::DlSym(path, err) => CrateError::DlSym(path, err),
        }
    }
}

pub unsafe fn load_symbol_from_dylib<T: Copy>(
    path: &Path,
    sym_name: &str,
) -> Result<T, DylibError> {
    // Make sure the path contains a / or the linker will search for it.
    let path = try_canonicalize(path).unwrap();
    let lib =
        load_dylib(&path, 5).map_err(|err| DylibError::DlOpen(path.display().to_string(), err))?;

    let sym = unsafe { lib.get::<T>(sym_name.as_bytes()) }
        .map_err(|err| DylibError::DlSym(path.display().to_string(), format_dlopen_err(&err)))?;

    // Intentionally leak the dynamic library. We can't ever unload it
    // since the library can make things that will live arbitrarily long.
    let sym = unsafe { sym.into_raw() };
    std::mem::forget(lib);

    Ok(*sym)
}
