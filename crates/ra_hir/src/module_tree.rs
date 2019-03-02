use std::sync::Arc;

use arrayvec::ArrayVec;
use relative_path::RelativePathBuf;
use ra_db::{FileId, SourceRoot};
use ra_syntax::{
    SyntaxNode, TreeArc,
    algo::generate,
    ast::{self, AstNode, NameOwner},
};
use ra_arena::{Arena, RawId, impl_arena_id};
use test_utils::tested_by;

use crate::{
    Name, AsName, HirDatabase, SourceItemId, HirFileId, Problem, SourceFileItems, ModuleSource,
    PersistentHirDatabase,
    Crate,
    ids::SourceFileItemId,
};

impl ModuleSource {
    pub(crate) fn new(
        db: &impl PersistentHirDatabase,
        file_id: HirFileId,
        decl_id: Option<SourceFileItemId>,
    ) -> ModuleSource {
        match decl_id {
            Some(item_id) => {
                let module = db.file_item(SourceItemId { file_id, item_id });
                let module = ast::Module::cast(&*module).unwrap();
                assert!(module.item_list().is_some(), "expected inline module");
                ModuleSource::Module(module.to_owned())
            }
            None => {
                let source_file = db.hir_parse(file_id);
                ModuleSource::SourceFile(source_file)
            }
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Submodule {
    name: Name,
    is_declaration: bool,
    decl_id: SourceFileItemId,
}

impl Submodule {
    pub(crate) fn submodules_query(
        db: &impl PersistentHirDatabase,
        file_id: HirFileId,
        decl_id: Option<SourceFileItemId>,
    ) -> Arc<Vec<Submodule>> {
        db.check_canceled();
        let file_items = db.file_items(file_id);
        let module_source = ModuleSource::new(db, file_id, decl_id);
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
                        decl_id: file_items.id_of(file_id, module.syntax()),
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

#[derive(Debug, PartialEq, Eq)]
pub struct ModuleData {
    file_id: HirFileId,
    /// Points to `ast::Module`, `None` for the whole file.
    decl_id: Option<SourceFileItemId>,
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
        db: &impl PersistentHirDatabase,
        krate: Crate,
    ) -> Arc<ModuleTree> {
        db.check_canceled();
        let mut res = ModuleTree::default();
        res.init_crate(db, krate);
        Arc::new(res)
    }

    pub(crate) fn modules<'a>(&'a self) -> impl Iterator<Item = ModuleId> + 'a {
        self.mods.iter().map(|(id, _)| id)
    }

    pub(crate) fn find_module_by_source(
        &self,
        file_id: HirFileId,
        decl_id: Option<SourceFileItemId>,
    ) -> Option<ModuleId> {
        let (res, _) =
            self.mods.iter().find(|(_, m)| (m.file_id, m.decl_id) == (file_id, decl_id))?;
        Some(res)
    }

    fn init_crate(&mut self, db: &impl PersistentHirDatabase, krate: Crate) {
        let crate_graph = db.crate_graph();
        let file_id = crate_graph.crate_root(krate.crate_id);
        let source_root_id = db.file_source_root(file_id);

        let source_root = db.source_root(source_root_id);
        self.init_subtree(db, &source_root, None, file_id.into(), None);
    }

    fn init_subtree(
        &mut self,
        db: &impl PersistentHirDatabase,
        source_root: &SourceRoot,
        parent: Option<LinkId>,
        file_id: HirFileId,
        decl_id: Option<SourceFileItemId>,
    ) -> ModuleId {
        let is_root = parent.is_none();
        let id = self.alloc_mod(ModuleData { file_id, decl_id, parent, children: Vec::new() });
        for sub in db.submodules(file_id, decl_id).iter() {
            let link = self.alloc_link(LinkData {
                source: SourceItemId { file_id, item_id: sub.decl_id },
                name: sub.name.clone(),
                owner: id,
                points_to: Vec::new(),
                problem: None,
            });

            let (points_to, problem) = if sub.is_declaration {
                let (points_to, problem) = resolve_submodule(db, file_id, &sub.name, is_root);
                let points_to = points_to
                    .into_iter()
                    .map(|file_id| {
                        self.init_subtree(db, source_root, Some(link), file_id.into(), None)
                    })
                    .collect::<Vec<_>>();
                (points_to, problem)
            } else {
                let points_to =
                    self.init_subtree(db, source_root, Some(link), file_id, Some(sub.decl_id));
                (vec![points_to], None)
            };

            self.links[link].points_to = points_to;
            self.links[link].problem = problem;
        }
        id
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
    pub(crate) fn file_id(self, tree: &ModuleTree) -> HirFileId {
        tree.mods[self].file_id
    }
    pub(crate) fn decl_id(self, tree: &ModuleTree) -> Option<SourceFileItemId> {
        tree.mods[self].decl_id
    }
    pub(crate) fn parent_link(self, tree: &ModuleTree) -> Option<LinkId> {
        tree.mods[self].parent
    }
    pub(crate) fn parent(self, tree: &ModuleTree) -> Option<ModuleId> {
        let link = self.parent_link(tree)?;
        Some(tree.links[link].owner)
    }
    pub(crate) fn crate_root(self, tree: &ModuleTree) -> ModuleId {
        generate(Some(self), move |it| it.parent(tree)).last().unwrap()
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
    pub(crate) fn source(
        self,
        tree: &ModuleTree,
        db: &impl PersistentHirDatabase,
    ) -> TreeArc<ast::Module> {
        let syntax_node = db.file_item(tree.links[self].source);
        ast::Module::cast(&syntax_node).unwrap().to_owned()
    }
}

pub(crate) fn resolve_module_declaration(
    db: &impl PersistentHirDatabase,
    file_id: HirFileId,
    name: &Name,
    is_root: bool,
) -> Option<FileId> {
    resolve_submodule(db, file_id, name, is_root).0.first().map(|it| *it)
}

fn resolve_submodule(
    db: &impl PersistentHirDatabase,
    file_id: HirFileId,
    name: &Name,
    is_root: bool,
) -> (Vec<FileId>, Option<Problem>) {
    // FIXME: handle submodules of inline modules properly
    let file_id = file_id.original_file(db);
    let source_root_id = db.file_source_root(file_id);
    let path = db.file_relative_path(file_id);
    let root = RelativePathBuf::default();
    let dir_path = path.parent().unwrap_or(&root);
    let mod_name = path.file_stem().unwrap_or("unknown");
    let is_dir_owner = is_root || mod_name == "mod";

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
