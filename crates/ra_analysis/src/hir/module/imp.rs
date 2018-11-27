use std::sync::Arc;

use ra_syntax::{
    ast::{self, NameOwner},
    SmolStr,
};
use relative_path::RelativePathBuf;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    db,
    hir::DescriptorDatabase,
    input::{SourceRoot, SourceRootId},
    Cancelable, FileId, FileResolverImp,
};

use super::{
    LinkData, LinkId, ModuleData, ModuleId, ModuleSource, ModuleSourceNode,
    ModuleTree, Problem,
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub(crate) enum Submodule {
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

pub(crate) fn submodules(
    db: &impl DescriptorDatabase,
    source: ModuleSource,
) -> Cancelable<Arc<Vec<Submodule>>> {
    db::check_canceled(db)?;
    let file_id = source.file_id();
    let submodules = match source.resolve(db) {
        ModuleSourceNode::SourceFile(it) => collect_submodules(file_id, it.borrowed()),
        ModuleSourceNode::Module(it) => it
            .borrowed()
            .item_list()
            .map(|it| collect_submodules(file_id, it))
            .unwrap_or_else(Vec::new),
    };
    return Ok(Arc::new(submodules));

    fn collect_submodules<'a>(
        file_id: FileId,
        root: impl ast::ModuleItemOwner<'a>,
    ) -> Vec<Submodule> {
        modules(root)
            .map(|(name, m)| {
                if m.has_semi() {
                    Submodule::Declaration(name)
                } else {
                    let src = ModuleSource::new_inline(file_id, m);
                    Submodule::Definition(name, src)
                }
            })
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
    let mut tree = ModuleTree::default();

    let mut roots = FxHashMap::default();
    let mut visited = FxHashSet::default();

    let source_root = db.source_root(source_root);
    for &file_id in source_root.files.iter() {
        let source = ModuleSource::SourceFile(file_id);
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
    db: &impl DescriptorDatabase,
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
    for sub in db._submodules(source)?.iter() {
        let link = tree.push_link(LinkData {
            name: sub.name().clone(),
            owner: id,
            points_to: Vec::new(),
            problem: None,
        });

        let (points_to, problem) = match sub {
            Submodule::Declaration(name) => {
                let (points_to, problem) =
                    resolve_submodule(source, &name, &source_root.file_resolver);
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
                            ModuleSource::SourceFile(file_id),
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
    source: ModuleSource,
    name: &SmolStr,
    file_resolver: &FileResolverImp,
) -> (Vec<FileId>, Option<Problem>) {
    let file_id = match source {
        ModuleSource::SourceFile(it) => it,
        ModuleSource::Module(..) => {
            // TODO
            return (Vec::new(), None);
        }
    };
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
