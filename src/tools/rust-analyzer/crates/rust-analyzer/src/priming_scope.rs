//! Scope computation for cache priming.

use ide_db::{RootDatabase, base_db::Crate};
use rustc_hash::FxHashSet;
use triomphe::Arc;

/// Close `seeds` under transitive dependencies.
pub(crate) fn compute(db: &RootDatabase, seeds: impl IntoIterator<Item = Crate>) -> Arc<[Crate]> {
    let mut closure: FxHashSet<Crate> = FxHashSet::default();
    let mut worklist: Vec<Crate> = seeds.into_iter().collect();
    while let Some(krate) = worklist.pop() {
        if !closure.insert(krate) {
            continue;
        }
        worklist.extend(krate.data(db).dependencies.iter().map(|dep| dep.crate_id));
    }
    closure.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use ide_db::{
        RootDatabase,
        base_db::{
            CrateGraphBuilder, CrateName, CrateOrigin, CrateWorkspaceData, CratesIdMap,
            DependencyBuilder, Env, LangCrateOrigin,
        },
        span::{Edition, FileId},
    };
    use rustc_hash::FxHashMap;
    use triomphe::Arc as TriompheArc;
    use vfs::AbsPathBuf;

    use super::*;

    fn empty_ws_data() -> TriompheArc<CrateWorkspaceData> {
        TriompheArc::new(CrateWorkspaceData { target: Err("".into()), toolchain: None })
    }

    /// Builds a synthetic crate graph in a fresh `RootDatabase` and returns
    /// the resolved `Crate` IDs keyed by the symbolic names used in `crates`.
    ///
    /// `crates` is a list of `(name, origin, deps)`. The crate root file id
    /// is derived from the crate's index. Dependencies must refer to crates
    /// declared earlier in the list.
    fn build(crates: &[(&str, CrateOrigin, &[&str])]) -> (RootDatabase, FxHashMap<String, Crate>) {
        let mut db = RootDatabase::default();
        let mut graph = CrateGraphBuilder::default();
        let proc_macro_cwd =
            TriompheArc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap()));

        let mut ids = Vec::with_capacity(crates.len());
        for (i, (name, origin, _)) in crates.iter().enumerate() {
            let id = graph.add_crate_root(
                FileId::from_raw((i + 1) as u32),
                Edition::Edition2021,
                None,
                None,
                Default::default(),
                Default::default(),
                Env::default(),
                origin.clone(),
                Vec::new(),
                false,
                proc_macro_cwd.clone(),
                empty_ws_data(),
            );
            ids.push((name.to_string(), id));
        }

        for (i, (_, _, deps)) in crates.iter().enumerate() {
            for dep_name in *deps {
                let from = ids[i].1;
                let to = ids
                    .iter()
                    .find(|(n, _)| n == dep_name)
                    .unwrap_or_else(|| panic!("unknown dep `{dep_name}`"))
                    .1;
                graph
                    .add_dep(from, DependencyBuilder::new(CrateName::new(dep_name).unwrap(), to))
                    .unwrap();
            }
        }

        let resolved: CratesIdMap = graph.set_in_db(&mut db);
        let by_name = ids.into_iter().map(|(n, id)| (n, resolved[&id])).collect();
        (db, by_name)
    }

    fn lang(name: &str) -> CrateOrigin {
        CrateOrigin::Lang(LangCrateOrigin::from(name))
    }

    fn local() -> CrateOrigin {
        CrateOrigin::Local { repo: None, name: None }
    }

    fn library(name: &str) -> CrateOrigin {
        CrateOrigin::Library { repo: None, name: CrateName::new(name).unwrap().symbol().clone() }
    }

    fn primed(scope: &Arc<[Crate]>, by_name: &FxHashMap<String, Crate>) -> Vec<String> {
        let lookup: FxHashMap<Crate, &str> =
            by_name.iter().map(|(n, c)| (*c, n.as_str())).collect();
        let mut names: Vec<String> = scope.iter().map(|c| lookup[c].to_owned()).collect();
        names.sort();
        names
    }

    #[test]
    fn closes_seeds_under_deps() {
        // core ← lib_a ← lib_b ← bin_c
        //                    ↖ test_d
        let (db, by) = build(&[
            ("core", lang("core"), &[]),
            ("lib_a", local(), &["core"]),
            ("lib_b", local(), &["lib_a"]),
            ("bin_c", local(), &["lib_b"]),
            ("test_d", local(), &["lib_b"]),
        ]);

        let scope = compute(&db, [by["lib_a"], by["lib_b"]]);
        assert_eq!(primed(&scope, &by), vec!["core", "lib_a", "lib_b"]);
    }

    #[test]
    fn active_seed_pulls_transitive_deps() {
        let (db, by) = build(&[
            ("core", lang("core"), &[]),
            ("lib_a", local(), &["core"]),
            ("lib_b", local(), &["lib_a"]),
            ("bin_c", local(), &["lib_b"]),
        ]);

        let scope = compute(&db, [by["bin_c"]]);
        assert_eq!(primed(&scope, &by), vec!["bin_c", "core", "lib_a", "lib_b"]);
    }

    #[test]
    fn does_not_pull_reverse_deps() {
        // Seeding only the leaf must not warm crates that depend on it.
        let (db, by) = build(&[
            ("core", lang("core"), &[]),
            ("lib_leaf", local(), &["core"]),
            ("dependent_a", local(), &["lib_leaf"]),
            ("dependent_b", local(), &["lib_leaf"]),
        ]);

        let scope = compute(&db, [by["lib_leaf"]]);
        assert_eq!(primed(&scope, &by), vec!["core", "lib_leaf"]);
    }

    #[test]
    fn pulls_library_dep_via_seeded_crate() {
        // A `Library` (non-workspace-member) origin behaves like any other —
        // dep walk is origin-agnostic, only the seed selection cares.
        let (db, by) = build(&[
            ("core", lang("core"), &[]),
            ("ext", library("serde"), &["core"]),
            ("lib_a", local(), &["ext"]),
        ]);

        let scope = compute(&db, [by["lib_a"]]);
        assert_eq!(primed(&scope, &by), vec!["core", "ext", "lib_a"]);
    }
}
