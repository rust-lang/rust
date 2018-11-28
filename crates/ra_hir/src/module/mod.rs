pub(super) mod imp;
pub(super) mod nameres;

use std::sync::Arc;

use ra_editor::find_node_at_offset;

use ra_syntax::{
    algo::generate,
    ast::{self, AstNode, NameOwner},
    SmolStr, SyntaxNode,
};
use ra_db::{SourceRootId, FileId, FilePosition, Cancelable};
use relative_path::RelativePathBuf;

use crate::{
    DefLoc, DefId, Path, PathKind, HirDatabase, SourceItemId,
    arena::{Arena, Id},
};

pub use self::nameres::ModuleScope;

/// `Module` is API entry point to get all the information
/// about a particular module.
#[derive(Debug, Clone)]
pub struct Module {
    tree: Arc<ModuleTree>,
    source_root_id: SourceRootId,
    module_id: ModuleId,
}

impl Module {
    /// Lookup `Module` by `FileId`. Note that this is inherently
    /// lossy transformation: in general, a single source might correspond to
    /// several modules.
    pub fn guess_from_file_id(
        db: &impl HirDatabase,
        file_id: FileId,
    ) -> Cancelable<Option<Module>> {
        Module::guess_from_source(db, file_id, ModuleSource::SourceFile(file_id))
    }

    /// Lookup `Module` by position in the source code. Note that this
    /// is inherently lossy transformation: in general, a single source might
    /// correspond to several modules.
    pub fn guess_from_position(
        db: &impl HirDatabase,
        position: FilePosition,
    ) -> Cancelable<Option<Module>> {
        let file = db.source_file(position.file_id);
        let module_source = match find_node_at_offset::<ast::Module>(file.syntax(), position.offset)
        {
            Some(m) if !m.has_semi() => ModuleSource::new_inline(db, position.file_id, m),
            _ => ModuleSource::SourceFile(position.file_id),
        };
        Module::guess_from_source(db, position.file_id, module_source)
    }

    fn guess_from_source(
        db: &impl HirDatabase,
        file_id: FileId,
        module_source: ModuleSource,
    ) -> Cancelable<Option<Module>> {
        let source_root_id = db.file_source_root(file_id);
        let module_tree = db.module_tree(source_root_id)?;

        let res = match module_tree.any_module_for_source(module_source) {
            None => None,
            Some(module_id) => Some(Module {
                tree: module_tree,
                source_root_id,
                module_id,
            }),
        };
        Ok(res)
    }

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
        let file_id = link.owner(&self.tree).source(&self.tree).file_id();
        let src = link.bind_source(&self.tree, db);
        Some((file_id, src))
    }

    pub fn source(&self) -> ModuleSource {
        self.module_id.source(&self.tree)
    }

    /// Parent module. Returns `None` if this is a root module.
    pub fn parent(&self) -> Option<Module> {
        let parent_id = self.module_id.parent(&self.tree)?;
        Some(Module {
            module_id: parent_id,
            ..self.clone()
        })
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
    pub fn name(&self) -> Option<SmolStr> {
        let link = self.module_id.parent_link(&self.tree)?;
        Some(link.name(&self.tree))
    }

    pub fn def_id(&self, db: &impl HirDatabase) -> DefId {
        let def_loc = DefLoc::Module {
            id: self.module_id,
            source_root: self.source_root_id,
        };
        def_loc.id(db)
    }

    /// Finds a child module with the specified name.
    pub fn child(&self, name: &str) -> Option<Module> {
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

    pub fn resolve_path(&self, db: &impl HirDatabase, path: Path) -> Cancelable<Option<DefId>> {
        let mut curr = match path.kind {
            PathKind::Crate => self.crate_root(),
            PathKind::Self_ | PathKind::Plain => self.clone(),
            PathKind::Super => ctry!(self.parent()),
        }
        .def_id(db);

        let segments = path.segments;
        for name in segments.iter() {
            let module = match curr.loc(db) {
                DefLoc::Module { id, source_root } => Module::new(db, source_root, id)?,
                _ => return Ok(None),
            };
            let scope = module.scope(db)?;
            curr = ctry!(ctry!(scope.get(&name)).def_id);
        }
        Ok(Some(curr))
    }

    pub fn problems(&self, db: &impl HirDatabase) -> Vec<(SyntaxNode, Problem)> {
        self.module_id.problems(&self.tree, db)
    }
}

/// Phisically, rust source is organized as a set of files, but logically it is
/// organized as a tree of modules. Usually, a single file corresponds to a
/// single module, but it is not nessary the case.
///
/// Module encapsulate the logic of transitioning from the fuzzy world of files
/// (which can have multiple parents) to the precise world of modules (which
/// always have one parent).
#[derive(Default, Debug, PartialEq, Eq)]
pub struct ModuleTree {
    mods: Arena<ModuleData>,
    links: Arena<LinkData>,
}

impl ModuleTree {
    pub(crate) fn modules<'a>(&'a self) -> impl Iterator<Item = ModuleId> + 'a {
        self.mods.iter().map(|(id, _)| id)
    }

    fn modules_for_source(&self, source: ModuleSource) -> Vec<ModuleId> {
        self.mods
            .iter()
            .filter(|(_idx, it)| it.source == source)
            .map(|(idx, _)| idx)
            .collect()
    }

    fn any_module_for_source(&self, source: ModuleSource) -> Option<ModuleId> {
        self.modules_for_source(source).pop()
    }
}

/// `ModuleSource` is the syntax tree element that produced this module:
/// either a file, or an inlinde module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModuleSource {
    SourceFile(FileId),
    Module(SourceItemId),
}

/// An owned syntax node for a module. Unlike `ModuleSource`,
/// this holds onto the AST for the whole file.
pub(crate) enum ModuleSourceNode {
    SourceFile(ast::SourceFileNode),
    Module(ast::ModuleNode),
}

pub type ModuleId = Id<ModuleData>;
type LinkId = Id<LinkData>;

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
    fn crate_root(self, tree: &ModuleTree) -> ModuleId {
        generate(Some(self), move |it| it.parent(tree))
            .last()
            .unwrap()
    }
    fn child(self, tree: &ModuleTree, name: &str) -> Option<ModuleId> {
        let link = tree.mods[self]
            .children
            .iter()
            .map(|&it| &tree.links[it])
            .find(|it| it.name == name)?;
        Some(*link.points_to.first()?)
    }
    fn children<'a>(self, tree: &'a ModuleTree) -> impl Iterator<Item = (SmolStr, ModuleId)> + 'a {
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
    fn name(self, tree: &ModuleTree) -> SmolStr {
        tree.links[self].name.clone()
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
    pub(crate) fn new_inline(
        db: &impl HirDatabase,
        file_id: FileId,
        module: ast::Module,
    ) -> ModuleSource {
        assert!(!module.has_semi());
        let items = db.file_items(file_id);
        let item_id = items.id_of(module.syntax());
        let id = SourceItemId { file_id, item_id };
        ModuleSource::Module(id)
    }

    pub fn as_file(self) -> Option<FileId> {
        match self {
            ModuleSource::SourceFile(f) => Some(f),
            ModuleSource::Module(..) => None,
        }
    }

    pub fn file_id(self) -> FileId {
        match self {
            ModuleSource::SourceFile(f) => f,
            ModuleSource::Module(source_item_id) => source_item_id.file_id,
        }
    }

    pub(crate) fn resolve(self, db: &impl HirDatabase) -> ModuleSourceNode {
        match self {
            ModuleSource::SourceFile(file_id) => {
                let syntax = db.source_file(file_id);
                ModuleSourceNode::SourceFile(syntax.ast().owned())
            }
            ModuleSource::Module(item_id) => {
                let syntax = db.file_item(item_id);
                let syntax = syntax.borrowed();
                let module = ast::Module::cast(syntax).unwrap();
                ModuleSourceNode::Module(module.owned())
            }
        }
    }
}

#[derive(Hash, Debug, PartialEq, Eq)]
struct LinkData {
    owner: ModuleId,
    name: SmolStr,
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
