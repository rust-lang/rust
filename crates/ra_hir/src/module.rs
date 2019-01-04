pub(super) mod imp;
pub(super) mod nameres;

use std::sync::Arc;
use log;

use ra_syntax::{
    algo::generate,
    ast::{self, AstNode, NameOwner},
    SyntaxNode,
};
use ra_arena::{Arena, RawId, impl_arena_id};
use ra_db::{SourceRootId, FileId, Cancelable};
use relative_path::RelativePathBuf;

use crate::{
    Def, DefKind, DefLoc, DefId,
    Name, Path, PathKind, HirDatabase, SourceItemId, SourceFileItemId, Crate,
    HirFileId,
};

pub use self::nameres::{ModuleScope, Resolution, Namespace, PerNs};

/// `Module` is API entry point to get all the information
/// about a particular module.
#[derive(Debug, Clone)]
pub struct Module {
    tree: Arc<ModuleTree>,
    pub(crate) source_root_id: SourceRootId,
    pub(crate) module_id: ModuleId,
}

impl Module {
    pub(super) fn new(
        db: &impl HirDatabase,
        source_root_id: SourceRootId,
        module_id: ModuleId,
    ) -> Cancelable<Module> {
        let module_tree = db.module_tree(source_root_id)?;
        let res = Module {
            tree: module_tree,
            source_root_id,
            module_id,
        };
        Ok(res)
    }

    /// Returns `mod foo;` or `mod foo {}` node whihc declared this module.
    /// Returns `None` for the root module
    pub fn parent_link_source(&self, db: &impl HirDatabase) -> Option<(FileId, ast::ModuleNode)> {
        let link = self.module_id.parent_link(&self.tree)?;
        let file_id = link
            .owner(&self.tree)
            .source(&self.tree)
            .file_id()
            .as_original_file();
        let src = link.bind_source(&self.tree, db);
        Some((file_id, src))
    }

    pub fn file_id(&self) -> FileId {
        self.source().file_id().as_original_file()
    }

    /// Parent module. Returns `None` if this is a root module.
    pub fn parent(&self) -> Option<Module> {
        let parent_id = self.module_id.parent(&self.tree)?;
        Some(Module {
            module_id: parent_id,
            ..self.clone()
        })
    }

    /// Returns an iterator of all children of this module.
    pub fn children<'a>(&'a self) -> impl Iterator<Item = (Name, Module)> + 'a {
        self.module_id
            .children(&self.tree)
            .map(move |(name, module_id)| {
                (
                    name,
                    Module {
                        module_id,
                        ..self.clone()
                    },
                )
            })
    }

    /// Returns the crate this module is part of.
    pub fn krate(&self, db: &impl HirDatabase) -> Option<Crate> {
        let root_id = self.module_id.crate_root(&self.tree);
        let file_id = root_id.source(&self.tree).file_id().as_original_file();
        let crate_graph = db.crate_graph();
        let crate_id = crate_graph.crate_id_for_crate_root(file_id)?;
        Some(Crate::new(crate_id))
    }

    /// Returns the all modules on the way to the root.
    pub fn path_to_root(&self) -> Vec<Module> {
        generate(Some(self.clone()), move |it| it.parent()).collect::<Vec<Module>>()
    }

    /// The root of the tree this module is part of
    pub fn crate_root(&self) -> Module {
        let root_id = self.module_id.crate_root(&self.tree);
        Module {
            module_id: root_id,
            ..self.clone()
        }
    }

    /// `name` is `None` for the crate's root module
    pub fn name(&self) -> Option<&Name> {
        let link = self.module_id.parent_link(&self.tree)?;
        Some(link.name(&self.tree))
    }

    pub fn def_id(&self, db: &impl HirDatabase) -> DefId {
        let def_loc = DefLoc {
            kind: DefKind::Module,
            source_root_id: self.source_root_id,
            module_id: self.module_id,
            source_item_id: self.module_id.source(&self.tree).0,
        };
        def_loc.id(db)
    }

    /// Finds a child module with the specified name.
    pub fn child(&self, name: &Name) -> Option<Module> {
        let child_id = self.module_id.child(&self.tree, name)?;
        Some(Module {
            module_id: child_id,
            ..self.clone()
        })
    }

    /// Returns a `ModuleScope`: a set of items, visible in this module.
    pub fn scope(&self, db: &impl HirDatabase) -> Cancelable<ModuleScope> {
        let item_map = db.item_map(self.source_root_id)?;
        let res = item_map.per_module[&self.module_id].clone();
        Ok(res)
    }

    pub fn resolve_path(&self, db: &impl HirDatabase, path: &Path) -> Cancelable<PerNs<DefId>> {
        let mut curr_per_ns = PerNs::types(
            match path.kind {
                PathKind::Crate => self.crate_root(),
                PathKind::Self_ | PathKind::Plain => self.clone(),
                PathKind::Super => {
                    if let Some(p) = self.parent() {
                        p
                    } else {
                        return Ok(PerNs::none());
                    }
                }
            }
            .def_id(db),
        );

        let segments = &path.segments;
        for name in segments.iter() {
            let curr = if let Some(r) = curr_per_ns.as_ref().take(Namespace::Types) {
                r
            } else {
                return Ok(PerNs::none());
            };
            let module = match curr.resolve(db)? {
                Def::Module(it) => it,
                // TODO here would be the place to handle enum variants...
                _ => return Ok(PerNs::none()),
            };
            let scope = module.scope(db)?;
            curr_per_ns = if let Some(r) = scope.get(&name) {
                r.def_id
            } else {
                return Ok(PerNs::none());
            };
        }
        Ok(curr_per_ns)
    }

    pub fn problems(&self, db: &impl HirDatabase) -> Vec<(SyntaxNode, Problem)> {
        self.module_id.problems(&self.tree, db)
    }

    pub(crate) fn source(&self) -> ModuleSource {
        self.module_id.source(&self.tree)
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

impl ModuleTree {
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

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Problem {
    UnresolvedModule {
        candidate: RelativePathBuf,
    },
    NotDirOwner {
        move_to: RelativePathBuf,
        candidate: RelativePathBuf,
    },
}

impl ModuleId {
    pub(crate) fn source(self, tree: &ModuleTree) -> ModuleSource {
        tree.mods[self].source
    }
    fn parent_link(self, tree: &ModuleTree) -> Option<LinkId> {
        tree.mods[self].parent
    }
    fn parent(self, tree: &ModuleTree) -> Option<ModuleId> {
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
    fn children<'a>(self, tree: &'a ModuleTree) -> impl Iterator<Item = (Name, ModuleId)> + 'a {
        tree.mods[self].children.iter().filter_map(move |&it| {
            let link = &tree.links[it];
            let module = *link.points_to.first()?;
            Some((link.name.clone(), module))
        })
    }
    fn problems(self, tree: &ModuleTree, db: &impl HirDatabase) -> Vec<(SyntaxNode, Problem)> {
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
    fn owner(self, tree: &ModuleTree) -> ModuleId {
        tree.links[self].owner
    }
    fn name(self, tree: &ModuleTree) -> &Name {
        &tree.links[self].name
    }
    fn bind_source<'a>(self, tree: &ModuleTree, db: &impl HirDatabase) -> ast::ModuleNode {
        let owner = self.owner(tree);
        match owner.source(tree).resolve(db) {
            ModuleSourceNode::SourceFile(root) => {
                let ast = imp::modules(root.borrowed())
                    .find(|(name, _)| name == &tree.links[self].name)
                    .unwrap()
                    .1;
                ast.owned()
            }
            ModuleSourceNode::Module(it) => it,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ModuleData {
    source: ModuleSource,
    parent: Option<LinkId>,
    children: Vec<LinkId>,
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

#[derive(Hash, Debug, PartialEq, Eq)]
struct LinkData {
    owner: ModuleId,
    name: Name,
    points_to: Vec<ModuleId>,
    problem: Option<Problem>,
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
