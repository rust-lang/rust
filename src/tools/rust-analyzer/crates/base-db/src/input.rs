//! This module specifies the input to rust-analyzer. In some sense, this is
//! **the** most important module, because all other fancy stuff is strictly
//! derived from this input.
//!
//! Note that neither this module, nor any other part of the analyzer's core do
//! actual IO. See `vfs` and `project_model` in the `rust-analyzer` crate for how
//! actual IO is done and lowered to input.

use std::hash::BuildHasherDefault;
use std::{fmt, mem, ops};

use cfg::{CfgOptions, HashableCfgOptions};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use intern::Symbol;
use la_arena::{Arena, Idx, RawIdx};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet, FxHasher};
use salsa::{Durability, Setter};
use span::Edition;
use triomphe::Arc;
use vfs::{AbsPathBuf, AnchoredPath, FileId, VfsPath, file_set::FileSet};

use crate::{CrateWorkspaceData, EditionedFileId, FxIndexSet, RootQueryDb};

pub type ProcMacroPaths = FxHashMap<CrateBuilderId, Result<(String, AbsPathBuf), String>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceRootId(pub u32);

/// Files are grouped into source roots. A source root is a directory on the
/// file systems which is watched for changes. Typically it corresponds to a
/// Rust crate. Source roots *might* be nested: in this case, a file belongs to
/// the nearest enclosing source root. Paths to files are always relative to a
/// source root, and the analyzer does not know the root path of the source root at
/// all. So, a file from one source root can't refer to a file in another source
/// root by path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRoot {
    /// Sysroot or crates.io library.
    ///
    /// Libraries are considered mostly immutable, this assumption is used to
    /// optimize salsa's query structure
    pub is_library: bool,
    file_set: FileSet,
}

impl SourceRoot {
    pub fn new_local(file_set: FileSet) -> SourceRoot {
        SourceRoot { is_library: false, file_set }
    }

    pub fn new_library(file_set: FileSet) -> SourceRoot {
        SourceRoot { is_library: true, file_set }
    }

    pub fn path_for_file(&self, file: &FileId) -> Option<&VfsPath> {
        self.file_set.path_for_file(file)
    }

    pub fn file_for_path(&self, path: &VfsPath) -> Option<&FileId> {
        self.file_set.file_for_path(path)
    }

    pub fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId> {
        self.file_set.resolve_path(path)
    }

    pub fn iter(&self) -> impl Iterator<Item = FileId> + '_ {
        self.file_set.iter()
    }
}

#[derive(Default, Clone)]
pub struct CrateGraphBuilder {
    arena: Arena<CrateBuilder>,
}

pub type CrateBuilderId = Idx<CrateBuilder>;

impl ops::Index<CrateBuilderId> for CrateGraphBuilder {
    type Output = CrateBuilder;

    fn index(&self, index: CrateBuilderId) -> &Self::Output {
        &self.arena[index]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateBuilder {
    pub basic: CrateDataBuilder,
    pub extra: ExtraCrateData,
    pub cfg_options: CfgOptions,
    pub env: Env,
    ws_data: Arc<CrateWorkspaceData>,
}

impl fmt::Debug for CrateGraphBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.arena.iter().map(|(id, data)| (u32::from(id.into_raw()), data)))
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateName(Symbol);

impl CrateName {
    /// Creates a crate name, checking for dashes in the string provided.
    /// Dashes are not allowed in the crate names,
    /// hence the input string is returned as `Err` for those cases.
    pub fn new(name: &str) -> Result<CrateName, &str> {
        if name.contains('-') { Err(name) } else { Ok(Self(Symbol::intern(name))) }
    }

    /// Creates a crate name, unconditionally replacing the dashes with underscores.
    pub fn normalize_dashes(name: &str) -> CrateName {
        Self(Symbol::intern(&name.replace('-', "_")))
    }

    pub fn symbol(&self) -> &Symbol {
        &self.0
    }
}

impl fmt::Display for CrateName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl ops::Deref for CrateName {
    type Target = Symbol;
    fn deref(&self) -> &Symbol {
        &self.0
    }
}

/// Origin of the crates.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CrateOrigin {
    /// Crates that are from the rustc workspace.
    Rustc { name: Symbol },
    /// Crates that are workspace members.
    Local { repo: Option<String>, name: Option<Symbol> },
    /// Crates that are non member libraries.
    Library { repo: Option<String>, name: Symbol },
    /// Crates that are provided by the language, like std, core, proc-macro, ...
    Lang(LangCrateOrigin),
}

impl CrateOrigin {
    pub fn is_local(&self) -> bool {
        matches!(self, CrateOrigin::Local { .. })
    }

    pub fn is_lib(&self) -> bool {
        matches!(self, CrateOrigin::Library { .. })
    }

    pub fn is_lang(&self) -> bool {
        matches!(self, CrateOrigin::Lang { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LangCrateOrigin {
    Alloc,
    Core,
    ProcMacro,
    Std,
    Test,
    Other,
}

impl From<&str> for LangCrateOrigin {
    fn from(s: &str) -> Self {
        match s {
            "alloc" => LangCrateOrigin::Alloc,
            "core" => LangCrateOrigin::Core,
            "proc-macro" | "proc_macro" => LangCrateOrigin::ProcMacro,
            "std" => LangCrateOrigin::Std,
            "test" => LangCrateOrigin::Test,
            _ => LangCrateOrigin::Other,
        }
    }
}

impl fmt::Display for LangCrateOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            LangCrateOrigin::Alloc => "alloc",
            LangCrateOrigin::Core => "core",
            LangCrateOrigin::ProcMacro => "proc_macro",
            LangCrateOrigin::Std => "std",
            LangCrateOrigin::Test => "test",
            LangCrateOrigin::Other => "other",
        };
        f.write_str(text)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateDisplayName {
    // The name we use to display various paths (with `_`).
    crate_name: CrateName,
    // The name as specified in Cargo.toml (with `-`).
    canonical_name: Symbol,
}

impl CrateDisplayName {
    pub fn canonical_name(&self) -> &Symbol {
        &self.canonical_name
    }
    pub fn crate_name(&self) -> &CrateName {
        &self.crate_name
    }
}

impl From<CrateName> for CrateDisplayName {
    fn from(crate_name: CrateName) -> CrateDisplayName {
        let canonical_name = crate_name.0.clone();
        CrateDisplayName { crate_name, canonical_name }
    }
}

impl fmt::Display for CrateDisplayName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.crate_name.fmt(f)
    }
}

impl ops::Deref for CrateDisplayName {
    type Target = Symbol;
    fn deref(&self) -> &Symbol {
        &self.crate_name
    }
}

impl CrateDisplayName {
    pub fn from_canonical_name(canonical_name: &str) -> CrateDisplayName {
        let crate_name = CrateName::normalize_dashes(canonical_name);
        CrateDisplayName { crate_name, canonical_name: Symbol::intern(canonical_name) }
    }
}

pub type TargetLayoutLoadResult = Result<Arc<str>, Arc<str>>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ReleaseChannel {
    Stable,
    Beta,
    Nightly,
}

impl ReleaseChannel {
    pub fn as_str(self) -> &'static str {
        match self {
            ReleaseChannel::Stable => "stable",
            ReleaseChannel::Beta => "beta",
            ReleaseChannel::Nightly => "nightly",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(str: &str) -> Option<Self> {
        Some(match str {
            "" | "stable" => ReleaseChannel::Stable,
            "nightly" => ReleaseChannel::Nightly,
            _ if str.starts_with("beta") => ReleaseChannel::Beta,
            _ => return None,
        })
    }
}

/// The crate data from which we derive the `Crate`.
///
/// We want this to contain as little data as possible, because if it contains dependencies and
/// something changes, this crate and all of its dependencies ids are invalidated, which causes
/// pretty much everything to be recomputed. If the crate id is not invalidated, only this crate's
/// information needs to be recomputed.
///
/// *Most* different crates have different root files (actually, pretty much all of them).
/// Still, it is possible to have crates distinguished by other factors (e.g. dependencies).
/// So we store only the root file - unless we find that this crate has the same root file as
/// another crate, in which case we store all data for one of them (if one is a dependency of
/// the other, we store for it, because it has more dependencies to be invalidated).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UniqueCrateData {
    root_file_id: FileId,
    disambiguator: Option<Box<(BuiltCrateData, HashableCfgOptions)>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateData<Id> {
    pub root_file_id: FileId,
    pub edition: Edition,
    /// The dependencies of this crate.
    ///
    /// Note that this may contain more dependencies than the crate actually uses.
    /// A common example is the test crate which is included but only actually is active when
    /// declared in source via `extern crate test`.
    pub dependencies: Vec<Dependency<Id>>,
    pub origin: CrateOrigin,
    pub is_proc_macro: bool,
    /// The working directory to run proc-macros in invoked in the context of this crate.
    /// This is the workspace root of the cargo workspace for workspace members, the crate manifest
    /// dir otherwise.
    // FIXME: This ought to be a `VfsPath` or something opaque.
    pub proc_macro_cwd: Arc<AbsPathBuf>,
}

pub type CrateDataBuilder = CrateData<CrateBuilderId>;
pub type BuiltCrateData = CrateData<Crate>;

/// Crate data unrelated to analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtraCrateData {
    pub version: Option<String>,
    /// A name used in the package's project declaration: for Cargo projects,
    /// its `[package].name` can be different for other project types or even
    /// absent (a dummy crate for the code snippet, for example).
    ///
    /// For purposes of analysis, crates are anonymous (only names in
    /// `Dependency` matters), this name should only be used for UI.
    pub display_name: Option<CrateDisplayName>,
    /// The cfg options that could be used by the crate
    pub potential_cfg_options: Option<CfgOptions>,
}

#[derive(Default, Clone, PartialEq, Eq)]
pub struct Env {
    entries: FxHashMap<String, String>,
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct EnvDebug<'s>(Vec<(&'s String, &'s String)>);

        impl fmt::Debug for EnvDebug<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_map().entries(self.0.iter().copied()).finish()
            }
        }
        f.debug_struct("Env")
            .field("entries", &{
                let mut entries: Vec<_> = self.entries.iter().collect();
                entries.sort();
                EnvDebug(entries)
            })
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dependency<Id> {
    pub crate_id: Id,
    pub name: CrateName,
    prelude: bool,
    sysroot: bool,
}

pub type DependencyBuilder = Dependency<CrateBuilderId>;
pub type BuiltDependency = Dependency<Crate>;

impl DependencyBuilder {
    pub fn new(name: CrateName, crate_id: CrateBuilderId) -> Self {
        Self { name, crate_id, prelude: true, sysroot: false }
    }

    pub fn with_prelude(
        name: CrateName,
        crate_id: CrateBuilderId,
        prelude: bool,
        sysroot: bool,
    ) -> Self {
        Self { name, crate_id, prelude, sysroot }
    }
}

impl BuiltDependency {
    /// Whether this dependency is to be added to the depending crate's extern prelude.
    pub fn is_prelude(&self) -> bool {
        self.prelude
    }

    /// Whether this dependency is a sysroot injected one.
    pub fn is_sysroot(&self) -> bool {
        self.sysroot
    }
}

pub type CratesIdMap = FxHashMap<CrateBuilderId, Crate>;

#[salsa_macros::input]
#[derive(Debug, PartialOrd, Ord)]
pub struct Crate {
    #[returns(ref)]
    pub data: BuiltCrateData,
    /// Crate data that is not needed for analysis.
    ///
    /// This is split into a separate field to increase incrementality.
    #[returns(ref)]
    pub extra_data: ExtraCrateData,
    // This is in `Arc` because it is shared for all crates in a workspace.
    #[returns(ref)]
    pub workspace_data: Arc<CrateWorkspaceData>,
    #[returns(ref)]
    pub cfg_options: CfgOptions,
    #[returns(ref)]
    pub env: Env,
}

/// The mapping from [`UniqueCrateData`] to their [`Crate`] input.
#[derive(Debug, Default)]
pub struct CratesMap(DashMap<UniqueCrateData, Crate, BuildHasherDefault<FxHasher>>);

impl CrateGraphBuilder {
    pub fn add_crate_root(
        &mut self,
        root_file_id: FileId,
        edition: Edition,
        display_name: Option<CrateDisplayName>,
        version: Option<String>,
        mut cfg_options: CfgOptions,
        mut potential_cfg_options: Option<CfgOptions>,
        mut env: Env,
        origin: CrateOrigin,
        is_proc_macro: bool,
        proc_macro_cwd: Arc<AbsPathBuf>,
        ws_data: Arc<CrateWorkspaceData>,
    ) -> CrateBuilderId {
        env.entries.shrink_to_fit();
        cfg_options.shrink_to_fit();
        if let Some(potential_cfg_options) = &mut potential_cfg_options {
            potential_cfg_options.shrink_to_fit();
        }
        self.arena.alloc(CrateBuilder {
            basic: CrateData {
                root_file_id,
                edition,
                dependencies: Vec::new(),
                origin,
                is_proc_macro,
                proc_macro_cwd,
            },
            extra: ExtraCrateData { version, display_name, potential_cfg_options },
            cfg_options,
            env,
            ws_data,
        })
    }

    pub fn add_dep(
        &mut self,
        from: CrateBuilderId,
        dep: DependencyBuilder,
    ) -> Result<(), CyclicDependenciesError> {
        let _p = tracing::info_span!("add_dep").entered();

        // Check if adding a dep from `from` to `to` creates a cycle. To figure
        // that out, look for a  path in the *opposite* direction, from `to` to
        // `from`.
        if let Some(path) = self.find_path(&mut FxHashSet::default(), dep.crate_id, from) {
            let path =
                path.into_iter().map(|it| (it, self[it].extra.display_name.clone())).collect();
            let err = CyclicDependenciesError { path };
            assert!(err.from().0 == from && err.to().0 == dep.crate_id);
            return Err(err);
        }

        self.arena[from].basic.dependencies.push(dep);
        Ok(())
    }

    pub fn set_in_db(self, db: &mut dyn RootQueryDb) -> CratesIdMap {
        // For some reason in some repositories we have duplicate crates, so we use a set and not `Vec`.
        // We use an `IndexSet` because the list needs to be topologically sorted.
        let mut all_crates = FxIndexSet::with_capacity_and_hasher(self.arena.len(), FxBuildHasher);
        let mut visited = FxHashMap::default();
        let mut visited_root_files = FxHashSet::default();

        let old_all_crates = db.all_crates();

        let crates_map = db.crates_map();
        // salsa doesn't compare new input to old input to see if they are the same, so here we are doing all the work ourselves.
        for krate in self.iter() {
            go(
                &self,
                db,
                &crates_map,
                &mut visited,
                &mut visited_root_files,
                &mut all_crates,
                krate,
            );
        }

        if old_all_crates.len() != all_crates.len()
            || old_all_crates.iter().any(|&krate| !all_crates.contains(&krate))
        {
            db.set_all_crates_with_durability(
                Arc::new(Vec::from_iter(all_crates).into_boxed_slice()),
                Durability::MEDIUM,
            );
        }

        return visited;

        fn go(
            graph: &CrateGraphBuilder,
            db: &mut dyn RootQueryDb,
            crates_map: &CratesMap,
            visited: &mut FxHashMap<CrateBuilderId, Crate>,
            visited_root_files: &mut FxHashSet<FileId>,
            all_crates: &mut FxIndexSet<Crate>,
            source: CrateBuilderId,
        ) -> Crate {
            if let Some(&crate_id) = visited.get(&source) {
                return crate_id;
            }
            let krate = &graph[source];
            let dependencies = krate
                .basic
                .dependencies
                .iter()
                .map(|dep| BuiltDependency {
                    crate_id: go(
                        graph,
                        db,
                        crates_map,
                        visited,
                        visited_root_files,
                        all_crates,
                        dep.crate_id,
                    ),
                    name: dep.name.clone(),
                    prelude: dep.prelude,
                    sysroot: dep.sysroot,
                })
                .collect::<Vec<_>>();
            let crate_data = BuiltCrateData {
                dependencies,
                edition: krate.basic.edition,
                is_proc_macro: krate.basic.is_proc_macro,
                origin: krate.basic.origin.clone(),
                root_file_id: krate.basic.root_file_id,
                proc_macro_cwd: krate.basic.proc_macro_cwd.clone(),
            };
            let disambiguator = if visited_root_files.insert(krate.basic.root_file_id) {
                None
            } else {
                Some(Box::new((crate_data.clone(), krate.cfg_options.to_hashable())))
            };

            let unique_crate_data =
                UniqueCrateData { root_file_id: krate.basic.root_file_id, disambiguator };
            let crate_input = match crates_map.0.entry(unique_crate_data) {
                Entry::Occupied(entry) => {
                    let old_crate = *entry.get();
                    if crate_data != *old_crate.data(db) {
                        old_crate.set_data(db).with_durability(Durability::MEDIUM).to(crate_data);
                    }
                    if krate.extra != *old_crate.extra_data(db) {
                        old_crate
                            .set_extra_data(db)
                            .with_durability(Durability::MEDIUM)
                            .to(krate.extra.clone());
                    }
                    if krate.cfg_options != *old_crate.cfg_options(db) {
                        old_crate
                            .set_cfg_options(db)
                            .with_durability(Durability::MEDIUM)
                            .to(krate.cfg_options.clone());
                    }
                    if krate.env != *old_crate.env(db) {
                        old_crate
                            .set_env(db)
                            .with_durability(Durability::MEDIUM)
                            .to(krate.env.clone());
                    }
                    if krate.ws_data != *old_crate.workspace_data(db) {
                        old_crate
                            .set_workspace_data(db)
                            .with_durability(Durability::MEDIUM)
                            .to(krate.ws_data.clone());
                    }
                    old_crate
                }
                Entry::Vacant(entry) => {
                    let input = Crate::builder(
                        crate_data,
                        krate.extra.clone(),
                        krate.ws_data.clone(),
                        krate.cfg_options.clone(),
                        krate.env.clone(),
                    )
                    .durability(Durability::MEDIUM)
                    .new(db);
                    entry.insert(input);
                    input
                }
            };
            all_crates.insert(crate_input);
            visited.insert(source, crate_input);
            crate_input
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = CrateBuilderId> + '_ {
        self.arena.iter().map(|(idx, _)| idx)
    }

    /// Returns an iterator over all transitive dependencies of the given crate,
    /// including the crate itself.
    pub fn transitive_deps(&self, of: CrateBuilderId) -> impl Iterator<Item = CrateBuilderId> {
        let mut worklist = vec![of];
        let mut deps = FxHashSet::default();

        while let Some(krate) = worklist.pop() {
            if !deps.insert(krate) {
                continue;
            }

            worklist.extend(self[krate].basic.dependencies.iter().map(|dep| dep.crate_id));
        }

        deps.into_iter()
    }

    /// Returns all crates in the graph, sorted in topological order (ie. dependencies of a crate
    /// come before the crate itself).
    fn crates_in_topological_order(&self) -> Vec<CrateBuilderId> {
        let mut res = Vec::new();
        let mut visited = FxHashSet::default();

        for krate in self.iter() {
            go(self, &mut visited, &mut res, krate);
        }

        return res;

        fn go(
            graph: &CrateGraphBuilder,
            visited: &mut FxHashSet<CrateBuilderId>,
            res: &mut Vec<CrateBuilderId>,
            source: CrateBuilderId,
        ) {
            if !visited.insert(source) {
                return;
            }
            for dep in graph[source].basic.dependencies.iter() {
                go(graph, visited, res, dep.crate_id)
            }
            res.push(source)
        }
    }

    /// Extends this crate graph by adding a complete second crate
    /// graph and adjust the ids in the [`ProcMacroPaths`] accordingly.
    ///
    /// This will deduplicate the crates of the graph where possible.
    /// Furthermore dependencies are sorted by crate id to make deduplication easier.
    ///
    /// Returns a map mapping `other`'s IDs to the new IDs in `self`.
    pub fn extend(
        &mut self,
        mut other: CrateGraphBuilder,
        proc_macros: &mut ProcMacroPaths,
    ) -> FxHashMap<CrateBuilderId, CrateBuilderId> {
        // Sorting here is a bit pointless because the input is likely already sorted.
        // However, the overhead is small and it makes the `extend` method harder to misuse.
        self.arena
            .iter_mut()
            .for_each(|(_, data)| data.basic.dependencies.sort_by_key(|dep| dep.crate_id));

        let m = self.arena.len();
        let topo = other.crates_in_topological_order();
        let mut id_map: FxHashMap<CrateBuilderId, CrateBuilderId> = FxHashMap::default();
        for topo in topo {
            let crate_data = &mut other.arena[topo];

            crate_data
                .basic
                .dependencies
                .iter_mut()
                .for_each(|dep| dep.crate_id = id_map[&dep.crate_id]);
            crate_data.basic.dependencies.sort_by_key(|dep| dep.crate_id);

            let find = self.arena.iter().take(m).find_map(|(k, v)| (v == crate_data).then_some(k));
            let new_id = find.unwrap_or_else(|| self.arena.alloc(crate_data.clone()));
            id_map.insert(topo, new_id);
        }

        *proc_macros =
            mem::take(proc_macros).into_iter().map(|(id, macros)| (id_map[&id], macros)).collect();
        id_map
    }

    fn find_path(
        &self,
        visited: &mut FxHashSet<CrateBuilderId>,
        from: CrateBuilderId,
        to: CrateBuilderId,
    ) -> Option<Vec<CrateBuilderId>> {
        if !visited.insert(from) {
            return None;
        }

        if from == to {
            return Some(vec![to]);
        }

        for dep in &self[from].basic.dependencies {
            let crate_id = dep.crate_id;
            if let Some(mut path) = self.find_path(visited, crate_id, to) {
                path.push(from);
                return Some(path);
            }
        }

        None
    }

    /// Removes all crates from this crate graph except for the ones in `to_keep` and fixes up the dependencies.
    /// Returns a mapping from old crate ids to new crate ids.
    pub fn remove_crates_except(
        &mut self,
        to_keep: &[CrateBuilderId],
    ) -> Vec<Option<CrateBuilderId>> {
        let mut id_map = vec![None; self.arena.len()];
        self.arena = std::mem::take(&mut self.arena)
            .into_iter()
            .filter_map(|(id, data)| if to_keep.contains(&id) { Some((id, data)) } else { None })
            .enumerate()
            .map(|(new_id, (id, data))| {
                id_map[id.into_raw().into_u32() as usize] =
                    Some(CrateBuilderId::from_raw(RawIdx::from_u32(new_id as u32)));
                data
            })
            .collect();
        for (_, data) in self.arena.iter_mut() {
            data.basic.dependencies.iter_mut().for_each(|dep| {
                dep.crate_id =
                    id_map[dep.crate_id.into_raw().into_u32() as usize].expect("crate was filtered")
            });
        }
        id_map
    }

    pub fn shrink_to_fit(&mut self) {
        self.arena.shrink_to_fit();
    }
}

pub(crate) fn transitive_rev_deps(db: &dyn RootQueryDb, of: Crate) -> FxHashSet<Crate> {
    let mut worklist = vec![of];
    let mut rev_deps = FxHashSet::default();
    rev_deps.insert(of);

    let mut inverted_graph = FxHashMap::<_, Vec<_>>::default();
    db.all_crates().iter().for_each(|&krate| {
        krate
            .data(db)
            .dependencies
            .iter()
            .for_each(|dep| inverted_graph.entry(dep.crate_id).or_default().push(krate))
    });

    while let Some(krate) = worklist.pop() {
        if let Some(crate_rev_deps) = inverted_graph.get(&krate) {
            crate_rev_deps
                .iter()
                .copied()
                .filter(|&rev_dep| rev_deps.insert(rev_dep))
                .for_each(|rev_dep| worklist.push(rev_dep));
        }
    }

    rev_deps
}

impl BuiltCrateData {
    pub fn root_file_id(&self, db: &dyn salsa::Database) -> EditionedFileId {
        EditionedFileId::new(db, self.root_file_id, self.edition)
    }
}

impl Extend<(String, String)> for Env {
    fn extend<T: IntoIterator<Item = (String, String)>>(&mut self, iter: T) {
        self.entries.extend(iter);
    }
}

impl FromIterator<(String, String)> for Env {
    fn from_iter<T: IntoIterator<Item = (String, String)>>(iter: T) -> Self {
        Env { entries: FromIterator::from_iter(iter) }
    }
}

impl Env {
    pub fn set(&mut self, env: &str, value: impl Into<String>) {
        self.entries.insert(env.to_owned(), value.into());
    }

    pub fn get(&self, env: &str) -> Option<String> {
        self.entries.get(env).cloned()
    }

    pub fn extend_from_other(&mut self, other: &Env) {
        self.entries.extend(other.entries.iter().map(|(x, y)| (x.to_owned(), y.to_owned())));
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn insert(&mut self, k: impl Into<String>, v: impl Into<String>) -> Option<String> {
        self.entries.insert(k.into(), v.into())
    }
}

impl From<Env> for Vec<(String, String)> {
    fn from(env: Env) -> Vec<(String, String)> {
        let mut entries: Vec<_> = env.entries.into_iter().collect();
        entries.sort();
        entries
    }
}

impl<'a> IntoIterator for &'a Env {
    type Item = (&'a String, &'a String);
    type IntoIter = std::collections::hash_map::Iter<'a, String, String>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

#[derive(Debug)]
pub struct CyclicDependenciesError {
    path: Vec<(CrateBuilderId, Option<CrateDisplayName>)>,
}

impl CyclicDependenciesError {
    fn from(&self) -> &(CrateBuilderId, Option<CrateDisplayName>) {
        self.path.first().unwrap()
    }
    fn to(&self) -> &(CrateBuilderId, Option<CrateDisplayName>) {
        self.path.last().unwrap()
    }
}

impl fmt::Display for CyclicDependenciesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let render = |(id, name): &(CrateBuilderId, Option<CrateDisplayName>)| match name {
            Some(it) => format!("{it}({id:?})"),
            None => format!("{id:?}"),
        };
        let path = self.path.iter().rev().map(render).collect::<Vec<String>>().join(" -> ");
        write!(
            f,
            "cyclic deps: {} -> {}, alternative path: {}",
            render(self.from()),
            render(self.to()),
            path
        )
    }
}

#[cfg(test)]
mod tests {
    use triomphe::Arc;
    use vfs::AbsPathBuf;

    use crate::{CrateWorkspaceData, DependencyBuilder};

    use super::{CrateGraphBuilder, CrateName, CrateOrigin, Edition::Edition2018, Env, FileId};

    fn empty_ws_data() -> Arc<CrateWorkspaceData> {
        Arc::new(CrateWorkspaceData { data_layout: Err("".into()), toolchain: None })
    }

    #[test]
    fn detect_cyclic_dependency_indirect() {
        let mut graph = CrateGraphBuilder::default();
        let crate1 = graph.add_crate_root(
            FileId::from_raw(1u32),
            Edition2018,
            None,
            None,
            Default::default(),
            Default::default(),
            Env::default(),
            CrateOrigin::Local { repo: None, name: None },
            false,
            Arc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap())),
            empty_ws_data(),
        );
        let crate2 = graph.add_crate_root(
            FileId::from_raw(2u32),
            Edition2018,
            None,
            None,
            Default::default(),
            Default::default(),
            Env::default(),
            CrateOrigin::Local { repo: None, name: None },
            false,
            Arc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap())),
            empty_ws_data(),
        );
        let crate3 = graph.add_crate_root(
            FileId::from_raw(3u32),
            Edition2018,
            None,
            None,
            Default::default(),
            Default::default(),
            Env::default(),
            CrateOrigin::Local { repo: None, name: None },
            false,
            Arc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap())),
            empty_ws_data(),
        );
        assert!(
            graph
                .add_dep(crate1, DependencyBuilder::new(CrateName::new("crate2").unwrap(), crate2,))
                .is_ok()
        );
        assert!(
            graph
                .add_dep(crate2, DependencyBuilder::new(CrateName::new("crate3").unwrap(), crate3,))
                .is_ok()
        );
        assert!(
            graph
                .add_dep(crate3, DependencyBuilder::new(CrateName::new("crate1").unwrap(), crate1,))
                .is_err()
        );
    }

    #[test]
    fn detect_cyclic_dependency_direct() {
        let mut graph = CrateGraphBuilder::default();
        let crate1 = graph.add_crate_root(
            FileId::from_raw(1u32),
            Edition2018,
            None,
            None,
            Default::default(),
            Default::default(),
            Env::default(),
            CrateOrigin::Local { repo: None, name: None },
            false,
            Arc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap())),
            empty_ws_data(),
        );
        let crate2 = graph.add_crate_root(
            FileId::from_raw(2u32),
            Edition2018,
            None,
            None,
            Default::default(),
            Default::default(),
            Env::default(),
            CrateOrigin::Local { repo: None, name: None },
            false,
            Arc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap())),
            empty_ws_data(),
        );
        assert!(
            graph
                .add_dep(crate1, DependencyBuilder::new(CrateName::new("crate2").unwrap(), crate2,))
                .is_ok()
        );
        assert!(
            graph
                .add_dep(crate2, DependencyBuilder::new(CrateName::new("crate2").unwrap(), crate2,))
                .is_err()
        );
    }

    #[test]
    fn it_works() {
        let mut graph = CrateGraphBuilder::default();
        let crate1 = graph.add_crate_root(
            FileId::from_raw(1u32),
            Edition2018,
            None,
            None,
            Default::default(),
            Default::default(),
            Env::default(),
            CrateOrigin::Local { repo: None, name: None },
            false,
            Arc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap())),
            empty_ws_data(),
        );
        let crate2 = graph.add_crate_root(
            FileId::from_raw(2u32),
            Edition2018,
            None,
            None,
            Default::default(),
            Default::default(),
            Env::default(),
            CrateOrigin::Local { repo: None, name: None },
            false,
            Arc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap())),
            empty_ws_data(),
        );
        let crate3 = graph.add_crate_root(
            FileId::from_raw(3u32),
            Edition2018,
            None,
            None,
            Default::default(),
            Default::default(),
            Env::default(),
            CrateOrigin::Local { repo: None, name: None },
            false,
            Arc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap())),
            empty_ws_data(),
        );
        assert!(
            graph
                .add_dep(crate1, DependencyBuilder::new(CrateName::new("crate2").unwrap(), crate2,))
                .is_ok()
        );
        assert!(
            graph
                .add_dep(crate2, DependencyBuilder::new(CrateName::new("crate3").unwrap(), crate3,))
                .is_ok()
        );
    }

    #[test]
    fn dashes_are_normalized() {
        let mut graph = CrateGraphBuilder::default();
        let crate1 = graph.add_crate_root(
            FileId::from_raw(1u32),
            Edition2018,
            None,
            None,
            Default::default(),
            Default::default(),
            Env::default(),
            CrateOrigin::Local { repo: None, name: None },
            false,
            Arc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap())),
            empty_ws_data(),
        );
        let crate2 = graph.add_crate_root(
            FileId::from_raw(2u32),
            Edition2018,
            None,
            None,
            Default::default(),
            Default::default(),
            Env::default(),
            CrateOrigin::Local { repo: None, name: None },
            false,
            Arc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap())),
            empty_ws_data(),
        );
        assert!(
            graph
                .add_dep(
                    crate1,
                    DependencyBuilder::new(
                        CrateName::normalize_dashes("crate-name-with-dashes"),
                        crate2,
                    )
                )
                .is_ok()
        );
        assert_eq!(
            graph.arena[crate1].basic.dependencies,
            vec![
                DependencyBuilder::new(CrateName::new("crate_name_with_dashes").unwrap(), crate2,)
            ]
        );
    }
}
