use std::sync::Arc;

use relative_path::RelativePathBuf;
use rustc_hash::{FxHashMap, FxHashSet};
use ra_syntax::{
    SmolStr,
    ast::{self, NameOwner},
};

use crate::{
    FileId, Cancelable, FileResolverImp, db,
    input::{SourceRoot, SourceRootId},
};

use super::{
    ModuleData, ModuleTree, ModuleId, LinkId, LinkData, Problem, ModulesDatabase
};


pub(super) fn submodules(db: &impl ModulesDatabase, file_id: FileId) -> Cancelable<Arc<Vec<SmolStr>>> {
    db::check_canceled(db)?;
    let file = db.file_syntax(file_id);
    let root = file.ast();
    let submodules = modules(root).map(|(name, _)| name).collect();
    Ok(Arc::new(submodules))
}

pub(super) fn modules(root: ast::Root<'_>) -> impl Iterator<Item = (SmolStr, ast::Module<'_>)> {
    root.modules().filter_map(|module| {
        let name = module.name()?.text();
        if !module.has_semi() {
            return None;
        }
        Some((name, module))
    })
}

pub(super) fn module_tree(
    db: &impl ModulesDatabase,
    source_root: SourceRootId,
) -> Cancelable<Arc<ModuleTree>> {
    db::check_canceled(db)?;
    let res = create_module_tree(db, source_root)?;
    Ok(Arc::new(res))
}


#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Submodule {
    pub name: SmolStr,
}


fn create_module_tree<'a>(
    db: &impl ModulesDatabase,
    source_root: SourceRootId,
) -> Cancelable<ModuleTree> {
    let mut tree = ModuleTree {
        mods: Vec::new(),
        links: Vec::new(),
    };

    let mut roots = FxHashMap::default();
    let mut visited = FxHashSet::default();

    let source_root = db.source_root(source_root);
    for &file_id in source_root.files.iter() {
        if visited.contains(&file_id) {
            continue; // TODO: use explicit crate_roots here
        }
        assert!(!roots.contains_key(&file_id));
        let module_id = build_subtree(db, &source_root, &mut tree, &mut visited, &mut roots, None, file_id)?;
        roots.insert(file_id, module_id);
    }
    Ok(tree)
}

fn build_subtree(
    db: &impl ModulesDatabase,
    source_root: &SourceRoot,
    tree: &mut ModuleTree,
    visited: &mut FxHashSet<FileId>,
    roots: &mut FxHashMap<FileId, ModuleId>,
    parent: Option<LinkId>,
    file_id: FileId,
) -> Cancelable<ModuleId> {
    visited.insert(file_id);
    let id = tree.push_mod(ModuleData {
        file_id,
        parent,
        children: Vec::new(),
    });
    for name in db.submodules(file_id)?.iter() {
        let (points_to, problem) = resolve_submodule(file_id, name, &source_root.file_resolver);
        let link = tree.push_link(LinkData {
            name: name.clone(),
            owner: id,
            points_to: Vec::new(),
            problem: None,
        });

        let points_to = points_to
            .into_iter()
            .map(|file_id| match roots.remove(&file_id) {
                Some(module_id) => {
                    tree.module_mut(module_id).parent = Some(link);
                    Ok(module_id)
                }
                None => build_subtree(db, source_root, tree, visited, roots, Some(link), file_id),
            })
            .collect::<Cancelable<Vec<_>>>()?;
        tree.link_mut(link).points_to = points_to;
        tree.link_mut(link).problem = problem;
    }
    Ok(id)
}

fn resolve_submodule(
    file_id: FileId,
    name: &SmolStr,
    file_resolver: &FileResolverImp,
) -> (Vec<FileId>, Option<Problem>) {
    let mod_name = file_resolver.file_stem(file_id);
    let is_dir_owner = mod_name == "mod" || mod_name == "lib" || mod_name == "main";

    let file_mod = RelativePathBuf::from(format!("../{}.rs", name));
    let dir_mod = RelativePathBuf::from(format!("../{}/mod.rs", name));
    let points_to: Vec<FileId>;
    let problem: Option<Problem>;
    if is_dir_owner {
        points_to = [&file_mod, &dir_mod]
            .iter()
            .filter_map(|path| file_resolver.resolve(file_id, path))
            .collect();
        problem = if points_to.is_empty() {
            Some(Problem::UnresolvedModule {
                candidate: file_mod,
            })
        } else {
            None
        }
    } else {
        points_to = Vec::new();
        problem = Some(Problem::NotDirOwner {
            move_to: RelativePathBuf::from(format!("../{}/mod.rs", mod_name)),
            candidate: file_mod,
        });
    }
    (points_to, problem)
}
