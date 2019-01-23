use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};
use arrayvec::ArrayVec;
use relative_path::RelativePathBuf;
use ra_db::{FileId, SourceRootId, SourceRoot};
use ra_syntax::{
    SyntaxNode, TreeArc,
    algo::generate,
    ast::{self, AstNode, NameOwner},
};
use ra_arena::{Arena, RawId, impl_arena_id};
use test_utils::tested_by;

use crate::{Name, AsName, HirDatabase, SourceItemId, HirFileId, Problem, SourceFileItems, ModuleSource};

impl ModuleSource {
    pub(crate) fn from_source_item_id(
        db: &impl HirDatabase,
        source_item_id: SourceItemId,
    ) -> ModuleSource {
        let module_syntax = db.file_item(source_item_id);
        if let Some(source_file) = ast::SourceFile::cast(&module_syntax) {
            ModuleSource::SourceFile(source_file.to_owned())
        } else if let Some(module) = ast::Module::cast(&module_syntax) {
            assert!(module.item_list().is_some(), "expected inline module");
            ModuleSource::Module(module.to_owned())
        } else {
            panic!("expected file or inline module")
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Submodule {
    name: Name,
    is_declaration: bool,
    source: SourceItemId,
}

impl Submodule {
    pub(crate) fn submodules_query(
        db: &impl HirDatabase,
        source: SourceItemId,
    ) -> Arc<Vec<Submodule>> {
        db.check_canceled();
        let file_id = source.file_id;
        let file_items = db.file_items(file_id);
        let module_source = ModuleSource::from_source_item_id(db, source);
        let submodules = match module_source {
            ModuleSource::SourceFile(source_file) => {
                collect_submodules(file_id, &file_items, &*source_file)
            }
            ModuleSource::Module(module) => {
                collect_submodules(file_id, &file_items, module.item_list().unwrap())
            }
        };
        return Arc::new(submodules);

        fn collect_submodules(
            file_id: HirFileId,
            file_items: &SourceFileItems,
            root: &impl ast::ModuleItemOwner,
        ) -> Vec<Submodule> {
            root.items()
                .filter_map(|item| match item.kind() {
                    ast::ModuleItemKind::Module(m) => Some(m),
                    _ => None,
                })
                .filter_map(|module| {
                    let name = module.name()?.as_name();
                    if !module.has_semi() && module.item_list().is_none() {
                        tested_by!(name_res_works_for_broken_modules);
                        return None;
                    }
                    let sub = Submodule {
                        name,
                        is_declaration: module.has_semi(),
                        source: SourceItemId {
                            file_id,
                            item_id: Some(file_items.id_of(file_id, module.syntax())),
                        },
                    };
                    Some(sub)
                })
                .collect()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModuleId(RawId);
impl_arena_id!(ModuleId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LinkId(RawId);
impl_arena_id!(LinkId);

/// Physically, rust source is organized as a set of files, but logically it is
/// organized as a tree of modules. Usually, a single file corresponds to a
/// single module, but it is not neccessarily always the case.
///
/// `ModuleTree` encapsulates the logic of transitioning from the fuzzy world of files
/// (which can have multiple parents) to the precise world of modules (which
/// always have one parent).
#[derive(Default, Debug, PartialEq, Eq)]
pub struct ModuleTree {
    mods: Arena<ModuleId, ModuleData>,
    links: Arena<LinkId, LinkData>,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ModuleData {
    source: SourceItemId,
    parent: Option<LinkId>,
    children: Vec<LinkId>,
}

#[derive(Hash, Debug, PartialEq, Eq)]
struct LinkData {
    source: SourceItemId,
    owner: ModuleId,
    name: Name,
    points_to: Vec<ModuleId>,
    problem: Option<Problem>,
}

impl ModuleTree {
    pub(crate) fn module_tree_query(
        db: &impl HirDatabase,
        source_root: SourceRootId,
    ) -> Arc<ModuleTree> {
        db.check_canceled();
        let res = create_module_tree(db, source_root);
        Arc::new(res)
    }

    pub(crate) fn modules<'a>(&'a self) -> impl Iterator<Item = ModuleId> + 'a {
        self.mods.iter().map(|(id, _)| id)
    }

    pub(crate) fn find_module_by_source(&self, source: SourceItemId) -> Option<ModuleId> {
        let (res, _) = self.mods.iter().find(|(_, m)| m.source == source)?;
        Some(res)
    }

    fn alloc_mod(&mut self, data: ModuleData) -> ModuleId {
        self.mods.alloc(data)
    }

    fn alloc_link(&mut self, data: LinkData) -> LinkId {
        let owner = data.owner;
        let id = self.links.alloc(data);
        self.mods[owner].children.push(id);
        id
    }
}

impl ModuleId {
    pub(crate) fn source(self, tree: &ModuleTree) -> SourceItemId {
        tree.mods[self].source
    }
    pub(crate) fn parent_link(self, tree: &ModuleTree) -> Option<LinkId> {
        tree.mods[self].parent
    }
    pub(crate) fn parent(self, tree: &ModuleTree) -> Option<ModuleId> {
        let link = self.parent_link(tree)?;
        Some(tree.links[link].owner)
    }
    pub(crate) fn crate_root(self, tree: &ModuleTree) -> ModuleId {
        generate(Some(self), move |it| it.parent(tree))
            .last()
            .unwrap()
    }
    pub(crate) fn child(self, tree: &ModuleTree, name: &Name) -> Option<ModuleId> {
        let link = tree.mods[self]
            .children
            .iter()
            .map(|&it| &tree.links[it])
            .find(|it| it.name == *name)?;
        Some(*link.points_to.first()?)
    }
    pub(crate) fn children<'a>(
        self,
        tree: &'a ModuleTree,
    ) -> impl Iterator<Item = (Name, ModuleId)> + 'a {
        tree.mods[self].children.iter().filter_map(move |&it| {
            let link = &tree.links[it];
            let module = *link.points_to.first()?;
            Some((link.name.clone(), module))
        })
    }
    pub(crate) fn problems(
        self,
        tree: &ModuleTree,
        db: &impl HirDatabase,
    ) -> Vec<(TreeArc<SyntaxNode>, Problem)> {
        tree.mods[self]
            .children
            .iter()
            .filter_map(|&link| {
                let p = tree.links[link].problem.clone()?;
                let s = link.source(tree, db);
                let s = s.name().unwrap().syntax().to_owned();
                Some((s, p))
            })
            .collect()
    }
}

impl LinkId {
    pub(crate) fn owner(self, tree: &ModuleTree) -> ModuleId {
        tree.links[self].owner
    }
    pub(crate) fn name(self, tree: &ModuleTree) -> &Name {
        &tree.links[self].name
    }
    pub(crate) fn source(self, tree: &ModuleTree, db: &impl HirDatabase) -> TreeArc<ast::Module> {
        let syntax_node = db.file_item(tree.links[self].source);
        ast::Module::cast(&syntax_node).unwrap().to_owned()
    }
}

fn create_module_tree<'a>(db: &impl HirDatabase, source_root: SourceRootId) -> ModuleTree {
    let mut tree = ModuleTree::default();

    let mut roots = FxHashMap::default();
    let mut visited = FxHashSet::default();

    let source_root = db.source_root(source_root);
    for &file_id in source_root.files.values() {
        let source = SourceItemId {
            file_id: file_id.into(),
            item_id: None,
        };
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
        );
        roots.insert(file_id, module_id);
    }
    tree
}

fn build_subtree(
    db: &impl HirDatabase,
    source_root: &SourceRoot,
    tree: &mut ModuleTree,
    visited: &mut FxHashSet<SourceItemId>,
    roots: &mut FxHashMap<FileId, ModuleId>,
    parent: Option<LinkId>,
    source: SourceItemId,
) -> ModuleId {
    visited.insert(source);
    let id = tree.alloc_mod(ModuleData {
        source,
        parent,
        children: Vec::new(),
    });
    for sub in db.submodules(source).iter() {
        let link = tree.alloc_link(LinkData {
            source: sub.source,
            name: sub.name.clone(),
            owner: id,
            points_to: Vec::new(),
            problem: None,
        });

        let (points_to, problem) = if sub.is_declaration {
            let (points_to, problem) = resolve_submodule(db, source.file_id, &sub.name);
            let points_to = points_to
                .into_iter()
                .map(|file_id| match roots.remove(&file_id) {
                    Some(module_id) => {
                        tree.mods[module_id].parent = Some(link);
                        module_id
                    }
                    None => build_subtree(
                        db,
                        source_root,
                        tree,
                        visited,
                        roots,
                        Some(link),
                        SourceItemId {
                            file_id: file_id.into(),
                            item_id: None,
                        },
                    ),
                })
                .collect::<Vec<_>>();
            (points_to, problem)
        } else {
            let points_to = build_subtree(
                db,
                source_root,
                tree,
                visited,
                roots,
                Some(link),
                sub.source,
            );
            (vec![points_to], None)
        };

        tree.links[link].points_to = points_to;
        tree.links[link].problem = problem;
    }
    id
}

fn resolve_submodule(
    db: &impl HirDatabase,
    file_id: HirFileId,
    name: &Name,
) -> (Vec<FileId>, Option<Problem>) {
    // FIXME: handle submodules of inline modules properly
    let file_id = file_id.original_file(db);
    let source_root_id = db.file_source_root(file_id);
    let path = db.file_relative_path(file_id);
    let root = RelativePathBuf::default();
    let dir_path = path.parent().unwrap_or(&root);
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
    let sr = db.source_root(source_root_id);
    let points_to = candidates
        .into_iter()
        .filter_map(|path| sr.files.get(&path))
        .map(|&it| it)
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
