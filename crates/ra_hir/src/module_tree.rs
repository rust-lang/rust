use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};
use arrayvec::ArrayVec;
use relative_path::RelativePathBuf;
use ra_db::{FileId, SourceRootId, Cancelable, SourceRoot};
use ra_syntax::{
    algo::generate,
    ast::{self, AstNode, NameOwner},
    SyntaxNode,
};
use ra_arena::{Arena, RawId, impl_arena_id};

use crate::{Name, AsName, HirDatabase, SourceItemId, SourceFileItemId, HirFileId, Problem};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Submodule {
    Declaration(Name),
    Definition(Name, ModuleSource),
}

impl Submodule {
    pub(crate) fn submodules_query(
        db: &impl HirDatabase,
        source: ModuleSource,
    ) -> Cancelable<Arc<Vec<Submodule>>> {
        db.check_canceled()?;
        let file_id = source.file_id();
        let submodules = match source.resolve(db) {
            ModuleSourceNode::SourceFile(it) => collect_submodules(db, file_id, it.borrowed()),
            ModuleSourceNode::Module(it) => it
                .borrowed()
                .item_list()
                .map(|it| collect_submodules(db, file_id, it))
                .unwrap_or_else(Vec::new),
        };
        return Ok(Arc::new(submodules));

        fn collect_submodules<'a>(
            db: &impl HirDatabase,
            file_id: HirFileId,
            root: impl ast::ModuleItemOwner<'a>,
        ) -> Vec<Submodule> {
            modules(root)
                .map(|(name, m)| {
                    if m.has_semi() {
                        Submodule::Declaration(name)
                    } else {
                        let src = ModuleSource::new_inline(db, file_id, m);
                        Submodule::Definition(name, src)
                    }
                })
                .collect()
        }
    }

    fn name(&self) -> &Name {
        match self {
            Submodule::Declaration(name) => name,
            Submodule::Definition(name, _) => name,
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
/// single module, but it is not nessary the case.
///
/// Module encapsulate the logic of transitioning from the fuzzy world of files
/// (which can have multiple parents) to the precise world of modules (which
/// always have one parent).
#[derive(Default, Debug, PartialEq, Eq)]
pub struct ModuleTree {
    mods: Arena<ModuleId, ModuleData>,
    links: Arena<LinkId, LinkData>,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ModuleData {
    source: ModuleSource,
    parent: Option<LinkId>,
    children: Vec<LinkId>,
}

#[derive(Hash, Debug, PartialEq, Eq)]
struct LinkData {
    owner: ModuleId,
    name: Name,
    points_to: Vec<ModuleId>,
    problem: Option<Problem>,
}

impl ModuleTree {
    pub(crate) fn module_tree_query(
        db: &impl HirDatabase,
        source_root: SourceRootId,
    ) -> Cancelable<Arc<ModuleTree>> {
        db.check_canceled()?;
        let res = create_module_tree(db, source_root)?;
        Ok(Arc::new(res))
    }

    pub(crate) fn modules<'a>(&'a self) -> impl Iterator<Item = ModuleId> + 'a {
        self.mods.iter().map(|(id, _)| id)
    }

    pub(crate) fn modules_with_sources<'a>(
        &'a self,
    ) -> impl Iterator<Item = (ModuleId, ModuleSource)> + 'a {
        self.mods.iter().map(|(id, m)| (id, m.source))
    }
}

/// `ModuleSource` is the syntax tree element that produced this module:
/// either a file, or an inlinde module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModuleSource(pub(crate) SourceItemId);

/// An owned syntax node for a module. Unlike `ModuleSource`,
/// this holds onto the AST for the whole file.
pub(crate) enum ModuleSourceNode {
    SourceFile(ast::SourceFileNode),
    Module(ast::ModuleNode),
}

impl ModuleId {
    pub(crate) fn source(self, tree: &ModuleTree) -> ModuleSource {
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
    ) -> Vec<(SyntaxNode, Problem)> {
        tree.mods[self]
            .children
            .iter()
            .filter_map(|&it| {
                let p = tree.links[it].problem.clone()?;
                let s = it.bind_source(tree, db);
                let s = s.borrowed().name().unwrap().syntax().owned();
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
    pub(crate) fn bind_source<'a>(
        self,
        tree: &ModuleTree,
        db: &impl HirDatabase,
    ) -> ast::ModuleNode {
        let owner = self.owner(tree);
        match owner.source(tree).resolve(db) {
            ModuleSourceNode::SourceFile(root) => {
                let ast = modules(root.borrowed())
                    .find(|(name, _)| name == &tree.links[self].name)
                    .unwrap()
                    .1;
                ast.owned()
            }
            ModuleSourceNode::Module(it) => it,
        }
    }
}

impl ModuleSource {
    // precondition: item_id **must** point to module
    fn new(file_id: HirFileId, item_id: Option<SourceFileItemId>) -> ModuleSource {
        let source_item_id = SourceItemId { file_id, item_id };
        ModuleSource(source_item_id)
    }

    pub(crate) fn new_file(file_id: HirFileId) -> ModuleSource {
        ModuleSource::new(file_id, None)
    }

    pub(crate) fn new_inline(
        db: &impl HirDatabase,
        file_id: HirFileId,
        m: ast::Module,
    ) -> ModuleSource {
        assert!(!m.has_semi());
        let file_items = db.file_items(file_id);
        let item_id = file_items.id_of(file_id, m.syntax());
        ModuleSource::new(file_id, Some(item_id))
    }

    pub(crate) fn file_id(self) -> HirFileId {
        self.0.file_id
    }

    pub(crate) fn resolve(self, db: &impl HirDatabase) -> ModuleSourceNode {
        let syntax_node = db.file_item(self.0);
        let syntax_node = syntax_node.borrowed();
        if let Some(file) = ast::SourceFile::cast(syntax_node) {
            return ModuleSourceNode::SourceFile(file.owned());
        }
        let module = ast::Module::cast(syntax_node).unwrap();
        ModuleSourceNode::Module(module.owned())
    }
}

impl ModuleTree {
    fn push_mod(&mut self, data: ModuleData) -> ModuleId {
        self.mods.alloc(data)
    }
    fn push_link(&mut self, data: LinkData) -> LinkId {
        let owner = data.owner;
        let id = self.links.alloc(data);
        self.mods[owner].children.push(id);
        id
    }
}

fn modules<'a>(
    root: impl ast::ModuleItemOwner<'a>,
) -> impl Iterator<Item = (Name, ast::Module<'a>)> {
    root.items()
        .filter_map(|item| match item {
            ast::ModuleItem::Module(m) => Some(m),
            _ => None,
        })
        .filter_map(|module| {
            let name = module.name()?.as_name();
            Some((name, module))
        })
}

fn create_module_tree<'a>(
    db: &impl HirDatabase,
    source_root: SourceRootId,
) -> Cancelable<ModuleTree> {
    let mut tree = ModuleTree::default();

    let mut roots = FxHashMap::default();
    let mut visited = FxHashSet::default();

    let source_root = db.source_root(source_root);
    for &file_id in source_root.files.values() {
        let source = ModuleSource::new_file(file_id.into());
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
                            ModuleSource::new_file(file_id.into()),
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
    name: &Name,
) -> (Vec<FileId>, Option<Problem>) {
    // FIXME: handle submodules of inline modules properly
    let file_id = source.file_id().original_file(db);
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
