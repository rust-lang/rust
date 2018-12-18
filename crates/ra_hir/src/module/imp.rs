use std::sync::Arc;

use ra_syntax::{
    ast::{self, NameOwner},
    SmolStr,
};
use rustc_hash::{FxHashMap, FxHashSet};
use arrayvec::ArrayVec;
use ra_db::{SourceRoot, SourceRootId, Cancelable, FileId};

use crate::{
    HirDatabase,
};

use super::{
    LinkData, LinkId, ModuleData, ModuleId, ModuleSource,
    ModuleTree, Problem,
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Submodule {
    Declaration(SmolStr),
    Definition(SmolStr, ModuleSource),
}

impl Submodule {
    fn name(&self) -> &SmolStr {
        match self {
            Submodule::Declaration(name) => name,
            Submodule::Definition(name, _) => name,
        }
    }
}

pub(crate) fn modules<'a>(
    root: impl ast::ModuleItemOwner<'a>,
) -> impl Iterator<Item = (SmolStr, ast::Module<'a>)> {
    root.items()
        .filter_map(|item| match item {
            ast::ModuleItem::Module(m) => Some(m),
            _ => None,
        })
        .filter_map(|module| {
            let name = module.name()?.text();
            Some((name, module))
        })
}

pub(crate) fn module_tree(
    db: &impl HirDatabase,
    source_root: SourceRootId,
) -> Cancelable<Arc<ModuleTree>> {
    db.check_canceled()?;
    let res = create_module_tree(db, source_root)?;
    Ok(Arc::new(res))
}

fn create_module_tree<'a>(
    db: &impl HirDatabase,
    source_root: SourceRootId,
) -> Cancelable<ModuleTree> {
    let mut tree = ModuleTree::default();

    let mut roots = FxHashMap::default();
    let mut visited = FxHashSet::default();

    let source_root = db.source_root(source_root);
    for &file_id in source_root.files.iter() {
        let source = ModuleSource::new_file(file_id);
        if visited.contains(&source) {
            continue; // TODO: use explicit crate_roots here
        }
        assert!(!roots.contains_key(&file_id));
        let module_id = build_subtree(
            db,
            &source_root,
            &mut tree,
            &mut visited,
            &mut roots,
            None,
            source,
        )?;
        roots.insert(file_id, module_id);
    }
    Ok(tree)
}

fn build_subtree(
    db: &impl HirDatabase,
    source_root: &SourceRoot,
    tree: &mut ModuleTree,
    visited: &mut FxHashSet<ModuleSource>,
    roots: &mut FxHashMap<FileId, ModuleId>,
    parent: Option<LinkId>,
    source: ModuleSource,
) -> Cancelable<ModuleId> {
    visited.insert(source);
    let id = tree.push_mod(ModuleData {
        source,
        parent,
        children: Vec::new(),
    });
    for sub in db.submodules(source)?.iter() {
        let link = tree.push_link(LinkData {
            name: sub.name().clone(),
            owner: id,
            points_to: Vec::new(),
            problem: None,
        });

        let (points_to, problem) = match sub {
            Submodule::Declaration(name) => {
                let (points_to, problem) = resolve_submodule(db, source, &name);
                let points_to = points_to
                    .into_iter()
                    .map(|file_id| match roots.remove(&file_id) {
                        Some(module_id) => {
                            tree.mods[module_id].parent = Some(link);
                            Ok(module_id)
                        }
                        None => build_subtree(
                            db,
                            source_root,
                            tree,
                            visited,
                            roots,
                            Some(link),
                            ModuleSource::new_file(file_id),
                        ),
                    })
                    .collect::<Cancelable<Vec<_>>>()?;
                (points_to, problem)
            }
            Submodule::Definition(_name, submodule_source) => {
                let points_to = build_subtree(
                    db,
                    source_root,
                    tree,
                    visited,
                    roots,
                    Some(link),
                    *submodule_source,
                )?;
                (vec![points_to], None)
            }
        };

        tree.links[link].points_to = points_to;
        tree.links[link].problem = problem;
    }
    Ok(id)
}

fn resolve_submodule(
    db: &impl HirDatabase,
    source: ModuleSource,
    name: &SmolStr,
) -> (Vec<FileId>, Option<Problem>) {
    // FIXME: handle submodules of inline modules properly
    let file_id = source.file_id();
    let source_root_id = db.file_source_root(file_id);
    let path = db.file_relative_path(file_id);
    let dir_path = path.parent().unwrap();
    let mod_name = path.file_stem().unwrap_or("unknown");
    let is_dir_owner = mod_name == "mod" || mod_name == "lib" || mod_name == "main";

    let file_mod = dir_path.join(format!("{}.rs", name));
    let dir_mod = dir_path.join(format!("{}/mod.rs", name));
    let file_dir_mod = dir_path.join(format!("{}/{}.rs", mod_name, name));
    let mut candidates = ArrayVec::<[_; 2]>::new();
    if is_dir_owner {
        candidates.push(file_mod.clone());
        candidates.push(dir_mod);
    } else {
        candidates.push(file_dir_mod.clone());
    };

    let points_to = candidates
        .into_iter()
        .filter_map(|path| db.source_root_file_by_path(source_root_id, path))
        .collect::<Vec<_>>();
    let problem = if points_to.is_empty() {
        Some(Problem::UnresolvedModule {
            candidate: if is_dir_owner { file_mod } else { file_dir_mod },
        })
    } else {
        None
    };
    (points_to, problem)
}
