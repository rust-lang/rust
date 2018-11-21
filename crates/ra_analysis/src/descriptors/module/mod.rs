pub(super) mod imp;
pub(super) mod nameres;

use std::sync::Arc;

use ra_editor::find_node_at_offset;

use ra_syntax::{
    algo::generate,
    ast::{self, AstNode, NameOwner},
    SmolStr, SyntaxNode,
};
use relative_path::RelativePathBuf;

use crate::{
    db::SyntaxDatabase, syntax_ptr::SyntaxPtr, FileId, FilePosition, Cancelable,
    descriptors::{Path, PathKind, DescriptorDatabase},
    input::SourceRootId
};

pub(crate) use self::nameres::ModuleScope;

/// `ModuleDescriptor` is API entry point to get all the information
/// about a particular module.
#[derive(Debug, Clone)]
pub(crate) struct ModuleDescriptor {
    tree: Arc<ModuleTree>,
    source_root_id: SourceRootId,
    module_id: ModuleId,
}

impl ModuleDescriptor {
    /// Lookup `ModuleDescriptor` by `FileId`. Note that this is inherently
    /// lossy transformation: in general, a single source might correspond to
    /// several modules.
    pub fn guess_from_file_id(
        db: &impl DescriptorDatabase,
        file_id: FileId,
    ) -> Cancelable<Option<ModuleDescriptor>> {
        ModuleDescriptor::guess_from_source(db, file_id, ModuleSource::SourceFile(file_id))
    }

    /// Lookup `ModuleDescriptor` by position in the source code. Note that this
    /// is inherently lossy transformation: in general, a single source might
    /// correspond to several modules.
    pub fn guess_from_position(
        db: &impl DescriptorDatabase,
        position: FilePosition,
    ) -> Cancelable<Option<ModuleDescriptor>> {
        let file = db.file_syntax(position.file_id);
        let module_source = match find_node_at_offset::<ast::Module>(file.syntax(), position.offset)
        {
            Some(m) if !m.has_semi() => ModuleSource::new_inline(position.file_id, m),
            _ => ModuleSource::SourceFile(position.file_id),
        };
        ModuleDescriptor::guess_from_source(db, position.file_id, module_source)
    }

    fn guess_from_source(
        db: &impl DescriptorDatabase,
        file_id: FileId,
        module_source: ModuleSource,
    ) -> Cancelable<Option<ModuleDescriptor>> {
        let source_root_id = db.file_source_root(file_id);
        let module_tree = db._module_tree(source_root_id)?;

        let res = match module_tree.any_module_for_source(module_source) {
            None => None,
            Some(module_id) => Some(ModuleDescriptor {
                tree: module_tree,
                source_root_id,
                module_id,
            }),
        };
        Ok(res)
    }

    /// Returns `mod foo;` or `mod foo {}` node whihc declared this module.
    /// Returns `None` for the root module
    pub fn parent_link_source(
        &self,
        db: &impl DescriptorDatabase,
    ) -> Option<(FileId, ast::ModuleNode)> {
        let link = self.module_id.parent_link(&self.tree)?;
        let file_id = link.owner(&self.tree).source(&self.tree).file_id();
        let src = link.bind_source(&self.tree, db);
        Some((file_id, src))
    }

    pub fn source(&self) -> ModuleSource {
        self.module_id.source(&self.tree)
    }

    /// Parent module. Returns `None` if this is a root module.
    pub fn parent(&self) -> Option<ModuleDescriptor> {
        let parent_id = self.module_id.parent(&self.tree)?;
        Some(ModuleDescriptor {
            module_id: parent_id,
            ..self.clone()
        })
    }

    /// The root of the tree this module is part of
    pub fn crate_root(&self) -> ModuleDescriptor {
        let root_id = self.module_id.crate_root(&self.tree);
        ModuleDescriptor {
            module_id: root_id,
            ..self.clone()
        }
    }

    /// `name` is `None` for the crate's root module
    #[allow(unused)]
    pub fn name(&self) -> Option<SmolStr> {
        let link = self.module_id.parent_link(&self.tree)?;
        Some(link.name(&self.tree))
    }

    /// Finds a child module with the specified name.
    pub fn child(&self, name: &str) -> Option<ModuleDescriptor> {
        let child_id = self.module_id.child(&self.tree, name)?;
        Some(ModuleDescriptor {
            module_id: child_id,
            ..self.clone()
        })
    }

    /// Returns a `ModuleScope`: a set of items, visible in this module.
    pub(crate) fn scope(&self, db: &impl DescriptorDatabase) -> Cancelable<ModuleScope> {
        let item_map = db._item_map(self.source_root_id)?;
        let res = item_map.per_module[&self.module_id].clone();
        Ok(res)
    }

    pub(crate) fn resolve_path(&self, path: Path) -> Option<ModuleDescriptor> {
        let mut curr = match path.kind {
            PathKind::Crate => self.crate_root(),
            PathKind::Self_ | PathKind::Plain => self.clone(),
            PathKind::Super => self.parent()?,
        };
        let segments = path.segments;
        for name in segments {
            curr = curr.child(&name)?;
        }
        Some(curr)
    }

    pub fn problems(&self, db: &impl DescriptorDatabase) -> Vec<(SyntaxNode, Problem)> {
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
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct ModuleTree {
    mods: Vec<ModuleData>,
    links: Vec<LinkData>,
}

impl ModuleTree {
    fn modules<'a>(&'a self) -> impl Iterator<Item = ModuleId> + 'a {
        self.mods
            .iter()
            .enumerate()
            .map(|(idx, _)| ModuleId(idx as u32))
    }

    fn modules_for_source(&self, source: ModuleSource) -> Vec<ModuleId> {
        self.mods
            .iter()
            .enumerate()
            .filter(|(_idx, it)| it.source == source)
            .map(|(idx, _)| ModuleId(idx as u32))
            .collect()
    }

    fn any_module_for_source(&self, source: ModuleSource) -> Option<ModuleId> {
        self.modules_for_source(source).pop()
    }
}

/// `ModuleSource` is the syntax tree element that produced this module:
/// either a file, or an inlinde module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ModuleSource {
    SourceFile(FileId),
    #[allow(dead_code)]
    Module(SyntaxPtr),
}

/// An owned syntax node for a module. Unlike `ModuleSource`,
/// this holds onto the AST for the whole file.
enum ModuleSourceNode {
    SourceFile(ast::SourceFileNode),
    Module(ast::ModuleNode),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub(crate) struct ModuleId(u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct LinkId(u32);

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
    fn source(self, tree: &ModuleTree) -> ModuleSource {
        tree.module(self).source
    }
    fn parent_link(self, tree: &ModuleTree) -> Option<LinkId> {
        tree.module(self).parent
    }
    fn parent(self, tree: &ModuleTree) -> Option<ModuleId> {
        let link = self.parent_link(tree)?;
        Some(tree.link(link).owner)
    }
    fn crate_root(self, tree: &ModuleTree) -> ModuleId {
        generate(Some(self), move |it| it.parent(tree))
            .last()
            .unwrap()
    }
    fn child(self, tree: &ModuleTree, name: &str) -> Option<ModuleId> {
        let link = tree
            .module(self)
            .children
            .iter()
            .map(|&it| tree.link(it))
            .find(|it| it.name == name)?;
        Some(*link.points_to.first()?)
    }
    fn children<'a>(self, tree: &'a ModuleTree) -> impl Iterator<Item = (SmolStr, ModuleId)> + 'a {
        tree.module(self).children.iter().filter_map(move |&it| {
            let link = tree.link(it);
            let module = *link.points_to.first()?;
            Some((link.name.clone(), module))
        })
    }
    fn problems(self, tree: &ModuleTree, db: &impl SyntaxDatabase) -> Vec<(SyntaxNode, Problem)> {
        tree.module(self)
            .children
            .iter()
            .filter_map(|&it| {
                let p = tree.link(it).problem.clone()?;
                let s = it.bind_source(tree, db);
                let s = s.borrowed().name().unwrap().syntax().owned();
                Some((s, p))
            })
            .collect()
    }
}

impl LinkId {
    fn owner(self, tree: &ModuleTree) -> ModuleId {
        tree.link(self).owner
    }
    fn name(self, tree: &ModuleTree) -> SmolStr {
        tree.link(self).name.clone()
    }
    fn bind_source<'a>(self, tree: &ModuleTree, db: &impl SyntaxDatabase) -> ast::ModuleNode {
        let owner = self.owner(tree);
        match owner.source(tree).resolve(db) {
            ModuleSourceNode::SourceFile(root) => {
                let ast = imp::modules(root.borrowed())
                    .find(|(name, _)| name == &tree.link(self).name)
                    .unwrap()
                    .1;
                ast.owned()
            }
            ModuleSourceNode::Module(it) => it,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct ModuleData {
    source: ModuleSource,
    parent: Option<LinkId>,
    children: Vec<LinkId>,
}

impl ModuleSource {
    fn new_inline(file_id: FileId, module: ast::Module) -> ModuleSource {
        assert!(!module.has_semi());
        let ptr = SyntaxPtr::new(file_id, module.syntax());
        ModuleSource::Module(ptr)
    }

    pub(crate) fn as_file(self) -> Option<FileId> {
        match self {
            ModuleSource::SourceFile(f) => Some(f),
            ModuleSource::Module(..) => None,
        }
    }

    pub(crate) fn file_id(self) -> FileId {
        match self {
            ModuleSource::SourceFile(f) => f,
            ModuleSource::Module(ptr) => ptr.file_id(),
        }
    }

    fn resolve(self, db: &impl SyntaxDatabase) -> ModuleSourceNode {
        match self {
            ModuleSource::SourceFile(file_id) => {
                let syntax = db.file_syntax(file_id);
                ModuleSourceNode::SourceFile(syntax.ast().owned())
            }
            ModuleSource::Module(ptr) => {
                let syntax = db.resolve_syntax_ptr(ptr);
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
    fn module(&self, id: ModuleId) -> &ModuleData {
        &self.mods[id.0 as usize]
    }
    fn module_mut(&mut self, id: ModuleId) -> &mut ModuleData {
        &mut self.mods[id.0 as usize]
    }
    fn link(&self, id: LinkId) -> &LinkData {
        &self.links[id.0 as usize]
    }
    fn link_mut(&mut self, id: LinkId) -> &mut LinkData {
        &mut self.links[id.0 as usize]
    }

    fn push_mod(&mut self, data: ModuleData) -> ModuleId {
        let id = ModuleId(self.mods.len() as u32);
        self.mods.push(data);
        id
    }
    fn push_link(&mut self, data: LinkData) -> LinkId {
        let id = LinkId(self.links.len() as u32);
        self.mods[data.owner.0 as usize].children.push(id);
        self.links.push(data);
        id
    }
}
