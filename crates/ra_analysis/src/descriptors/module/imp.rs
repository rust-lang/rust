use std::sync::Arc;

use ra_syntax::{
    ast::{self, ModuleItemOwner, NameOwner},
    SmolStr,
};
use relative_path::RelativePathBuf;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    db,
    descriptors::DescriptorDatabase,
    input::{SourceRoot, SourceRootId},
    Cancelable, FileId, FileResolverImp,
};

use super::{
    LinkData, LinkId, ModuleData, ModuleId, ModuleScope, ModuleSource, ModuleSourceNode,
    ModuleTree, Problem,
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub(crate) struct Submodule {
    name: SmolStr,
}

pub(crate) fn submodules(
    db: &impl DescriptorDatabase,
    source: ModuleSource,
) -> Cancelable<Arc<Vec<Submodule>>> {
    db::check_canceled(db)?;
    let submodules = match source.resolve(db) {
        ModuleSourceNode::Root(it) => collect_submodules(it.ast()),
        ModuleSourceNode::Inline(it) => it
            .ast()
            .item_list()
            .map(collect_submodules)
            .unwrap_or_else(Vec::new),
    };
    return Ok(Arc::new(submodules));

    fn collect_submodules<'a>(root: impl ast::ModuleItemOwner<'a>) -> Vec<Submodule> {
        modules(root)
            .filter(|(_, m)| m.has_semi())
            .map(|(name, _)| Submodule { name })
            .collect()
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

pub(crate) fn module_scope(
    db: &impl DescriptorDatabase,
    source_root_id: SourceRootId,
    module_id: ModuleId,
) -> Cancelable<Arc<ModuleScope>> {
    let tree = db.module_tree(source_root_id)?;
    let source = module_id.source(&tree).resolve(db);
    let res = match source {
        ModuleSourceNode::Root(root) => ModuleScope::new(root.ast().items()),
        ModuleSourceNode::Inline(inline) => match inline.ast().item_list() {
            Some(items) => ModuleScope::new(items.items()),
            None => ModuleScope::new(std::iter::empty()),
        },
    };
    Ok(Arc::new(res))
}

pub(crate) fn module_tree(
    db: &impl DescriptorDatabase,
    source_root: SourceRootId,
) -> Cancelable<Arc<ModuleTree>> {
    db::check_canceled(db)?;
    let res = create_module_tree(db, source_root)?;
    Ok(Arc::new(res))
}

fn create_module_tree<'a>(
    db: &impl DescriptorDatabase,
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
        let module_id = build_subtree(
            db,
            &source_root,
            &mut tree,
            &mut visited,
            &mut roots,
            None,
            file_id,
        )?;
        roots.insert(file_id, module_id);
    }
    Ok(tree)
}

fn build_subtree(
    db: &impl DescriptorDatabase,
    source_root: &SourceRoot,
    tree: &mut ModuleTree,
    visited: &mut FxHashSet<FileId>,
    roots: &mut FxHashMap<FileId, ModuleId>,
    parent: Option<LinkId>,
    file_id: FileId,
) -> Cancelable<ModuleId> {
    visited.insert(file_id);
    let id = tree.push_mod(ModuleData {
        source: ModuleSource::File(file_id),
        parent,
        children: Vec::new(),
    });
    for sub in db.submodules(ModuleSource::File(file_id))?.iter() {
        let name = sub.name.clone();
        let (points_to, problem) = resolve_submodule(file_id, &name, &source_root.file_resolver);
        let link = tree.push_link(LinkData {
            name,
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
