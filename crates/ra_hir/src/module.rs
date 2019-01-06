pub(super) mod imp;
pub(super) mod nameres;

use ra_syntax::{
    algo::generate,
    ast::{self, AstNode, NameOwner},
    SyntaxNode,
};
use ra_arena::{Arena, RawId, impl_arena_id};
use relative_path::RelativePathBuf;

use crate::{Name, HirDatabase, SourceItemId, SourceFileItemId, HirFileId};

pub use self::nameres::{ModuleScope, Resolution, Namespace, PerNs};

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
    fn children<'a>(self, tree: &'a ModuleTree) -> impl Iterator<Item = (Name, ModuleId)> + 'a {
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
