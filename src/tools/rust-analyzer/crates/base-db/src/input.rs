//! This module specifies the input to rust-analyzer. In some sense, this is
//! **the** most important module, because all other fancy stuff is strictly
//! derived from this input.
//!
//! Note that neither this module, nor any other part of the analyzer's core do
//! actual IO. See `vfs` and `project_model` in the `rust-analyzer` crate for how
//! actual IO is done and lowered to input.

use std::{fmt, mem, ops};

use cfg::CfgOptions;
use intern::Symbol;
use la_arena::{Arena, Idx, RawIdx};
use rustc_hash::{FxHashMap, FxHashSet};
use span::{Edition, EditionedFileId};
use triomphe::Arc;
use vfs::{file_set::FileSet, AbsPathBuf, AnchoredPath, FileId, VfsPath};

pub type ProcMacroPaths = FxHashMap<CrateId, Result<(String, AbsPathBuf), String>>;

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

/// `CrateGraph` is a bit of information which turns a set of text files into a
/// number of Rust crates.
///
/// Each crate is defined by the `FileId` of its root module, the set of enabled
/// `cfg` flags and the set of dependencies.
///
/// Note that, due to cfg's, there might be several crates for a single `FileId`!
///
/// For the purposes of analysis, a crate does not have a name. Instead, names
/// are specified on dependency edges. That is, a crate might be known under
/// different names in different dependent crates.
///
/// Note that `CrateGraph` is build-system agnostic: it's a concept of the Rust
/// language proper, not a concept of the build system. In practice, we get
/// `CrateGraph` by lowering `cargo metadata` output.
///
/// `CrateGraph` is `!Serialize` by design, see
/// <https://github.com/rust-lang/rust-analyzer/blob/master/docs/dev/architecture.md#serialization>
#[derive(Clone, Default)]
pub struct CrateGraph {
    arena: Arena<CrateData>,
}

impl fmt::Debug for CrateGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.arena.iter().map(|(id, data)| (u32::from(id.into_raw()), data)))
            .finish()
    }
}

pub type CrateId = Idx<CrateData>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateName(Symbol);

impl CrateName {
    /// Creates a crate name, checking for dashes in the string provided.
    /// Dashes are not allowed in the crate names,
    /// hence the input string is returned as `Err` for those cases.
    pub fn new(name: &str) -> Result<CrateName, &str> {
        if name.contains('-') {
            Err(name)
        } else {
            Ok(Self(Symbol::intern(name)))
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateData {
    pub root_file_id: FileId,
    pub edition: Edition,
    pub version: Option<String>,
    /// A name used in the package's project declaration: for Cargo projects,
    /// its `[package].name` can be different for other project types or even
    /// absent (a dummy crate for the code snippet, for example).
    ///
    /// For purposes of analysis, crates are anonymous (only names in
    /// `Dependency` matters), this name should only be used for UI.
    pub display_name: Option<CrateDisplayName>,
    pub cfg_options: Arc<CfgOptions>,
    /// The cfg options that could be used by the crate
    pub potential_cfg_options: Option<Arc<CfgOptions>>,
    pub env: Env,
    /// The dependencies of this crate.
    ///
    /// Note that this may contain more dependencies than the crate actually uses.
    /// A common example is the test crate which is included but only actually is active when
    /// declared in source via `extern crate test`.
    pub dependencies: Vec<Dependency>,
    pub origin: CrateOrigin,
    pub is_proc_macro: bool,
    /// The working directory to run proc-macros in. This is the workspace root of the cargo workspace
    /// for workspace members, the crate manifest dir otherwise.
    pub proc_macro_cwd: Option<AbsPathBuf>,
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
pub struct Dependency {
    pub crate_id: CrateId,
    pub name: CrateName,
    prelude: bool,
    sysroot: bool,
}

impl Dependency {
    pub fn new(name: CrateName, crate_id: CrateId) -> Self {
        Self { name, crate_id, prelude: true, sysroot: false }
    }

    pub fn with_prelude(name: CrateName, crate_id: CrateId, prelude: bool, sysroot: bool) -> Self {
        Self { name, crate_id, prelude, sysroot }
    }

    /// Whether this dependency is to be added to the depending crate's extern prelude.
    pub fn is_prelude(&self) -> bool {
        self.prelude
    }

    /// Whether this dependency is a sysroot injected one.
    pub fn is_sysroot(&self) -> bool {
        self.sysroot
    }
}

impl CrateGraph {
    pub fn add_crate_root(
        &mut self,
        root_file_id: FileId,
        edition: Edition,
        display_name: Option<CrateDisplayName>,
        version: Option<String>,
        cfg_options: Arc<CfgOptions>,
        potential_cfg_options: Option<Arc<CfgOptions>>,
        mut env: Env,
        origin: CrateOrigin,
        is_proc_macro: bool,
        proc_macro_cwd: Option<AbsPathBuf>,
    ) -> CrateId {
        env.entries.shrink_to_fit();
        let data = CrateData {
            root_file_id,
            edition,
            version,
            display_name,
            cfg_options,
            potential_cfg_options,
            env,
            dependencies: Vec::new(),
            origin,
            is_proc_macro,
            proc_macro_cwd,
        };
        self.arena.alloc(data)
    }

    pub fn add_dep(
        &mut self,
        from: CrateId,
        dep: Dependency,
    ) -> Result<(), CyclicDependenciesError> {
        let _p = tracing::info_span!("add_dep").entered();

        // Check if adding a dep from `from` to `to` creates a cycle. To figure
        // that out, look for a  path in the *opposite* direction, from `to` to
        // `from`.
        if let Some(path) = self.find_path(&mut FxHashSet::default(), dep.crate_id, from) {
            let path = path.into_iter().map(|it| (it, self[it].display_name.clone())).collect();
            let err = CyclicDependenciesError { path };
            assert!(err.from().0 == from && err.to().0 == dep.crate_id);
            return Err(err);
        }

        self.arena[from].add_dep(dep);
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    pub fn len(&self) -> usize {
        self.arena.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = CrateId> + '_ {
        self.arena.iter().map(|(idx, _)| idx)
    }

    // FIXME: used for fixing up the toolchain sysroot, should be removed and done differently
    #[doc(hidden)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (CrateId, &mut CrateData)> + '_ {
        self.arena.iter_mut()
    }

    /// Returns an iterator over all transitive dependencies of the given crate,
    /// including the crate itself.
    pub fn transitive_deps(&self, of: CrateId) -> impl Iterator<Item = CrateId> {
        let mut worklist = vec![of];
        let mut deps = FxHashSet::default();

        while let Some(krate) = worklist.pop() {
            if !deps.insert(krate) {
                continue;
            }

            worklist.extend(self[krate].dependencies.iter().map(|dep| dep.crate_id));
        }

        deps.into_iter()
    }

    /// Returns all transitive reverse dependencies of the given crate,
    /// including the crate itself.
    pub fn transitive_rev_deps(&self, of: CrateId) -> impl Iterator<Item = CrateId> {
        let mut worklist = vec![of];
        let mut rev_deps = FxHashSet::default();
        rev_deps.insert(of);

        let mut inverted_graph = FxHashMap::<_, Vec<_>>::default();
        self.arena.iter().for_each(|(krate, data)| {
            data.dependencies
                .iter()
                .for_each(|dep| inverted_graph.entry(dep.crate_id).or_default().push(krate))
        });

        while let Some(krate) = worklist.pop() {
            if let Some(krate_rev_deps) = inverted_graph.get(&krate) {
                krate_rev_deps
                    .iter()
                    .copied()
                    .filter(|&rev_dep| rev_deps.insert(rev_dep))
                    .for_each(|rev_dep| worklist.push(rev_dep));
            }
        }

        rev_deps.into_iter()
    }

    /// Returns all crates in the graph, sorted in topological order (ie. dependencies of a crate
    /// come before the crate itself).
    pub fn crates_in_topological_order(&self) -> Vec<CrateId> {
        let mut res = Vec::new();
        let mut visited = FxHashSet::default();

        for krate in self.iter() {
            go(self, &mut visited, &mut res, krate);
        }

        return res;

        fn go(
            graph: &CrateGraph,
            visited: &mut FxHashSet<CrateId>,
            res: &mut Vec<CrateId>,
            source: CrateId,
        ) {
            if !visited.insert(source) {
                return;
            }
            for dep in graph[source].dependencies.iter() {
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
        mut other: CrateGraph,
        proc_macros: &mut ProcMacroPaths,
    ) -> FxHashMap<CrateId, CrateId> {
        // Sorting here is a bit pointless because the input is likely already sorted.
        // However, the overhead is small and it makes the `extend` method harder to misuse.
        self.arena
            .iter_mut()
            .for_each(|(_, data)| data.dependencies.sort_by_key(|dep| dep.crate_id));

        let m = self.len();
        let topo = other.crates_in_topological_order();
        let mut id_map: FxHashMap<CrateId, CrateId> = FxHashMap::default();
        for topo in topo {
            let crate_data = &mut other.arena[topo];

            crate_data.dependencies.iter_mut().for_each(|dep| dep.crate_id = id_map[&dep.crate_id]);
            crate_data.dependencies.sort_by_key(|dep| dep.crate_id);

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
        visited: &mut FxHashSet<CrateId>,
        from: CrateId,
        to: CrateId,
    ) -> Option<Vec<CrateId>> {
        if !visited.insert(from) {
            return None;
        }

        if from == to {
            return Some(vec![to]);
        }

        for dep in &self[from].dependencies {
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
    pub fn remove_crates_except(&mut self, to_keep: &[CrateId]) -> Vec<Option<CrateId>> {
        let mut id_map = vec![None; self.arena.len()];
        self.arena = std::mem::take(&mut self.arena)
            .into_iter()
            .filter_map(|(id, data)| if to_keep.contains(&id) { Some((id, data)) } else { None })
            .enumerate()
            .map(|(new_id, (id, data))| {
                id_map[id.into_raw().into_u32() as usize] =
                    Some(CrateId::from_raw(RawIdx::from_u32(new_id as u32)));
                data
            })
            .collect();
        for (_, data) in self.arena.iter_mut() {
            data.dependencies.iter_mut().for_each(|dep| {
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

impl ops::Index<CrateId> for CrateGraph {
    type Output = CrateData;
    fn index(&self, crate_id: CrateId) -> &CrateData {
        &self.arena[crate_id]
    }
}

impl CrateData {
    /// Add a dependency to `self` without checking if the dependency
    // is existent among `self.dependencies`.
    fn add_dep(&mut self, dep: Dependency) {
        self.dependencies.push(dep)
    }

    pub fn root_file_id(&self) -> EditionedFileId {
        EditionedFileId::new(self.root_file_id, self.edition)
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
    path: Vec<(CrateId, Option<CrateDisplayName>)>,
}

impl CyclicDependenciesError {
    fn from(&self) -> &(CrateId, Option<CrateDisplayName>) {
        self.path.first().unwrap()
    }
    fn to(&self) -> &(CrateId, Option<CrateDisplayName>) {
        self.path.last().unwrap()
    }
}

impl fmt::Display for CyclicDependenciesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let render = |(id, name): &(CrateId, Option<CrateDisplayName>)| match name {
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
    use crate::CrateOrigin;

    use super::{CrateGraph, CrateName, Dependency, Edition::Edition2018, Env, FileId};

    #[test]
    fn detect_cyclic_dependency_indirect() {
        let mut graph = CrateGraph::default();
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
            None,
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
            None,
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
            None,
        );
        assert!(graph
            .add_dep(crate1, Dependency::new(CrateName::new("crate2").unwrap(), crate2,))
            .is_ok());
        assert!(graph
            .add_dep(crate2, Dependency::new(CrateName::new("crate3").unwrap(), crate3,))
            .is_ok());
        assert!(graph
            .add_dep(crate3, Dependency::new(CrateName::new("crate1").unwrap(), crate1,))
            .is_err());
    }

    #[test]
    fn detect_cyclic_dependency_direct() {
        let mut graph = CrateGraph::default();
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
            None,
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
            None,
        );
        assert!(graph
            .add_dep(crate1, Dependency::new(CrateName::new("crate2").unwrap(), crate2,))
            .is_ok());
        assert!(graph
            .add_dep(crate2, Dependency::new(CrateName::new("crate2").unwrap(), crate2,))
            .is_err());
    }

    #[test]
    fn it_works() {
        let mut graph = CrateGraph::default();
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
            None,
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
            None,
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
            None,
        );
        assert!(graph
            .add_dep(crate1, Dependency::new(CrateName::new("crate2").unwrap(), crate2,))
            .is_ok());
        assert!(graph
            .add_dep(crate2, Dependency::new(CrateName::new("crate3").unwrap(), crate3,))
            .is_ok());
    }

    #[test]
    fn dashes_are_normalized() {
        let mut graph = CrateGraph::default();
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
            None,
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
            None,
        );
        assert!(graph
            .add_dep(
                crate1,
                Dependency::new(CrateName::normalize_dashes("crate-name-with-dashes"), crate2,)
            )
            .is_ok());
        assert_eq!(
            graph[crate1].dependencies,
            vec![Dependency::new(CrateName::new("crate_name_with_dashes").unwrap(), crate2,)]
        );
    }
}
