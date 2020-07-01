//! This module specifies the input to rust-analyzer. In some sense, this is
//! **the** most important module, because all other fancy stuff is strictly
//! derived from this input.
//!
//! Note that neither this module, nor any other part of the analyzer's core do
//! actual IO. See `vfs` and `project_model` in the `rust-analyzer` crate for how
//! actual IO is done and lowered to input.

use std::{fmt, ops, str::FromStr, sync::Arc};

use ra_cfg::CfgOptions;
use ra_syntax::SmolStr;
use ra_tt::TokenExpander;
use rustc_hash::{FxHashMap, FxHashSet};
use vfs::file_set::FileSet;

pub use vfs::FileId;

/// Files are grouped into source roots. A source root is a directory on the
/// file systems which is watched for changes. Typically it corresponds to a
/// Rust crate. Source roots *might* be nested: in this case, a file belongs to
/// the nearest enclosing source root. Paths to files are always relative to a
/// source root, and the analyzer does not know the root path of the source root at
/// all. So, a file from one source root can't refer to a file in another source
/// root by path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceRootId(pub u32);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRoot {
    /// Sysroot or crates.io library.
    ///
    /// Libraries are considered mostly immutable, this assumption is used to
    /// optimize salsa's query structure
    pub is_library: bool,
    pub(crate) file_set: FileSet,
}

impl SourceRoot {
    pub fn new_local(file_set: FileSet) -> SourceRoot {
        SourceRoot { is_library: false, file_set }
    }
    pub fn new_library(file_set: FileSet) -> SourceRoot {
        SourceRoot { is_library: true, file_set }
    }
    pub fn iter(&self) -> impl Iterator<Item = FileId> + '_ {
        self.file_set.iter()
    }
}

/// `CrateGraph` is a bit of information which turns a set of text files into a
/// number of Rust crates. Each crate is defined by the `FileId` of its root module,
/// the set of cfg flags (not yet implemented) and the set of dependencies. Note
/// that, due to cfg's, there might be several crates for a single `FileId`! As
/// in the rust-lang proper, a crate does not have a name. Instead, names are
/// specified on dependency edges. That is, a crate might be known under
/// different names in different dependent crates.
///
/// Note that `CrateGraph` is build-system agnostic: it's a concept of the Rust
/// language proper, not a concept of the build system. In practice, we get
/// `CrateGraph` by lowering `cargo metadata` output.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrateGraph {
    arena: FxHashMap<CrateId, CrateData>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CrateId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrateName(SmolStr);

impl CrateName {
    /// Creates a crate name, checking for dashes in the string provided.
    /// Dashes are not allowed in the crate names,
    /// hence the input string is returned as `Err` for those cases.
    pub fn new(name: &str) -> Result<CrateName, &str> {
        if name.contains('-') {
            Err(name)
        } else {
            Ok(Self(SmolStr::new(name)))
        }
    }

    /// Creates a crate name, unconditionally replacing the dashes with underscores.
    pub fn normalize_dashes(name: &str) -> CrateName {
        Self(SmolStr::new(name.replace('-', "_")))
    }
}

impl fmt::Display for CrateName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ops::Deref for CrateName {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ProcMacroId(pub u32);

#[derive(Debug, Clone)]
pub struct ProcMacro {
    pub name: SmolStr,
    pub expander: Arc<dyn TokenExpander>,
}

impl Eq for ProcMacro {}
impl PartialEq for ProcMacro {
    fn eq(&self, other: &ProcMacro) -> bool {
        self.name == other.name && Arc::ptr_eq(&self.expander, &other.expander)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateData {
    pub root_file_id: FileId,
    pub edition: Edition,
    /// The name to display to the end user.
    /// This actual crate name can be different in a particular dependent crate
    /// or may even be missing for some cases, such as a dummy crate for the code snippet.
    pub display_name: Option<String>,
    pub cfg_options: CfgOptions,
    pub env: Env,
    pub dependencies: Vec<Dependency>,
    pub proc_macro: Vec<ProcMacro>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Edition {
    Edition2018,
    Edition2015,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Env {
    entries: FxHashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dependency {
    pub crate_id: CrateId,
    pub name: CrateName,
}

impl CrateGraph {
    pub fn add_crate_root(
        &mut self,
        file_id: FileId,
        edition: Edition,
        display_name: Option<String>,
        cfg_options: CfgOptions,
        env: Env,
        proc_macro: Vec<(SmolStr, Arc<dyn ra_tt::TokenExpander>)>,
    ) -> CrateId {
        let proc_macro =
            proc_macro.into_iter().map(|(name, it)| ProcMacro { name, expander: it }).collect();

        let data = CrateData {
            root_file_id: file_id,
            edition,
            display_name,
            cfg_options,
            env,
            proc_macro,
            dependencies: Vec::new(),
        };
        let crate_id = CrateId(self.arena.len() as u32);
        let prev = self.arena.insert(crate_id, data);
        assert!(prev.is_none());
        crate_id
    }

    pub fn add_dep(
        &mut self,
        from: CrateId,
        name: CrateName,
        to: CrateId,
    ) -> Result<(), CyclicDependenciesError> {
        if self.dfs_find(from, to, &mut FxHashSet::default()) {
            return Err(CyclicDependenciesError);
        }
        self.arena.get_mut(&from).unwrap().add_dep(name, to);
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = CrateId> + '_ {
        self.arena.keys().copied()
    }

    /// Returns an iterator over all transitive dependencies of the given crate.
    pub fn transitive_deps(&self, of: CrateId) -> impl Iterator<Item = CrateId> + '_ {
        let mut worklist = vec![of];
        let mut deps = FxHashSet::default();

        while let Some(krate) = worklist.pop() {
            if !deps.insert(krate) {
                continue;
            }

            worklist.extend(self[krate].dependencies.iter().map(|dep| dep.crate_id));
        }

        deps.remove(&of);
        deps.into_iter()
    }

    // FIXME: this only finds one crate with the given root; we could have multiple
    pub fn crate_id_for_crate_root(&self, file_id: FileId) -> Option<CrateId> {
        let (&crate_id, _) =
            self.arena.iter().find(|(_crate_id, data)| data.root_file_id == file_id)?;
        Some(crate_id)
    }

    /// Extends this crate graph by adding a complete disjoint second crate
    /// graph.
    ///
    /// The ids of the crates in the `other` graph are shifted by the return
    /// amount.
    pub fn extend(&mut self, other: CrateGraph) -> u32 {
        let start = self.arena.len() as u32;
        self.arena.extend(other.arena.into_iter().map(|(id, mut data)| {
            let new_id = id.shift(start);
            for dep in &mut data.dependencies {
                dep.crate_id = dep.crate_id.shift(start);
            }
            (new_id, data)
        }));
        start
    }

    fn dfs_find(&self, target: CrateId, from: CrateId, visited: &mut FxHashSet<CrateId>) -> bool {
        if !visited.insert(from) {
            return false;
        }

        if target == from {
            return true;
        }

        for dep in &self[from].dependencies {
            let crate_id = dep.crate_id;
            if self.dfs_find(target, crate_id, visited) {
                return true;
            }
        }
        false
    }
}

impl ops::Index<CrateId> for CrateGraph {
    type Output = CrateData;
    fn index(&self, crate_id: CrateId) -> &CrateData {
        &self.arena[&crate_id]
    }
}

impl CrateId {
    pub fn shift(self, amount: u32) -> CrateId {
        CrateId(self.0 + amount)
    }
}

impl CrateData {
    fn add_dep(&mut self, name: CrateName, crate_id: CrateId) {
        self.dependencies.push(Dependency { name, crate_id })
    }
}

impl FromStr for Edition {
    type Err = ParseEditionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let res = match s {
            "2015" => Edition::Edition2015,
            "2018" => Edition::Edition2018,
            _ => return Err(ParseEditionError { invalid_input: s.to_string() }),
        };
        Ok(res)
    }
}

impl fmt::Display for Edition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Edition::Edition2015 => "2015",
            Edition::Edition2018 => "2018",
        })
    }
}

impl<'a, T> From<T> for Env
where
    T: Iterator<Item = (&'a String, &'a String)>,
{
    fn from(iter: T) -> Self {
        let mut result = Self::default();

        for (k, v) in iter {
            result.entries.insert(k.to_owned(), v.to_owned());
        }

        result
    }
}

impl Env {
    pub fn set(&mut self, env: &str, value: String) {
        self.entries.insert(env.to_owned(), value);
    }

    pub fn get(&self, env: &str) -> Option<String> {
        self.entries.get(env).cloned()
    }
}

#[derive(Debug)]
pub struct ParseEditionError {
    invalid_input: String,
}

impl fmt::Display for ParseEditionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid edition: {:?}", self.invalid_input)
    }
}

impl std::error::Error for ParseEditionError {}

#[derive(Debug)]
pub struct CyclicDependenciesError;

#[cfg(test)]
mod tests {
    use super::{CfgOptions, CrateGraph, CrateName, Dependency, Edition::Edition2018, Env, FileId};

    #[test]
    fn detect_cyclic_dependency_indirect() {
        let mut graph = CrateGraph::default();
        let crate1 = graph.add_crate_root(
            FileId(1u32),
            Edition2018,
            None,
            CfgOptions::default(),
            Env::default(),
            Default::default(),
        );
        let crate2 = graph.add_crate_root(
            FileId(2u32),
            Edition2018,
            None,
            CfgOptions::default(),
            Env::default(),
            Default::default(),
        );
        let crate3 = graph.add_crate_root(
            FileId(3u32),
            Edition2018,
            None,
            CfgOptions::default(),
            Env::default(),
            Default::default(),
        );
        assert!(graph.add_dep(crate1, CrateName::new("crate2").unwrap(), crate2).is_ok());
        assert!(graph.add_dep(crate2, CrateName::new("crate3").unwrap(), crate3).is_ok());
        assert!(graph.add_dep(crate3, CrateName::new("crate1").unwrap(), crate1).is_err());
    }

    #[test]
    fn detect_cyclic_dependency_direct() {
        let mut graph = CrateGraph::default();
        let crate1 = graph.add_crate_root(
            FileId(1u32),
            Edition2018,
            None,
            CfgOptions::default(),
            Env::default(),
            Default::default(),
        );
        let crate2 = graph.add_crate_root(
            FileId(2u32),
            Edition2018,
            None,
            CfgOptions::default(),
            Env::default(),
            Default::default(),
        );
        assert!(graph.add_dep(crate1, CrateName::new("crate2").unwrap(), crate2).is_ok());
        assert!(graph.add_dep(crate2, CrateName::new("crate2").unwrap(), crate2).is_err());
    }

    #[test]
    fn it_works() {
        let mut graph = CrateGraph::default();
        let crate1 = graph.add_crate_root(
            FileId(1u32),
            Edition2018,
            None,
            CfgOptions::default(),
            Env::default(),
            Default::default(),
        );
        let crate2 = graph.add_crate_root(
            FileId(2u32),
            Edition2018,
            None,
            CfgOptions::default(),
            Env::default(),
            Default::default(),
        );
        let crate3 = graph.add_crate_root(
            FileId(3u32),
            Edition2018,
            None,
            CfgOptions::default(),
            Env::default(),
            Default::default(),
        );
        assert!(graph.add_dep(crate1, CrateName::new("crate2").unwrap(), crate2).is_ok());
        assert!(graph.add_dep(crate2, CrateName::new("crate3").unwrap(), crate3).is_ok());
    }

    #[test]
    fn dashes_are_normalized() {
        let mut graph = CrateGraph::default();
        let crate1 = graph.add_crate_root(
            FileId(1u32),
            Edition2018,
            None,
            CfgOptions::default(),
            Env::default(),
            Default::default(),
        );
        let crate2 = graph.add_crate_root(
            FileId(2u32),
            Edition2018,
            None,
            CfgOptions::default(),
            Env::default(),
            Default::default(),
        );
        assert!(graph
            .add_dep(crate1, CrateName::normalize_dashes("crate-name-with-dashes"), crate2)
            .is_ok());
        assert_eq!(
            graph[crate1].dependencies,
            vec![Dependency {
                crate_id: crate2,
                name: CrateName::new("crate_name_with_dashes").unwrap()
            }]
        );
    }
}
