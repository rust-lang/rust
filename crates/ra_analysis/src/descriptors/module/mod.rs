pub(super) mod imp;
pub(crate) mod scope;

use ra_syntax::{
    ast::{self, AstNode, NameOwner},
    SmolStr, SyntaxNode,
};
use relative_path::RelativePathBuf;

use crate::{db::SyntaxDatabase, syntax_ptr::SyntaxPtr, FileId};

pub(crate) use self::scope::ModuleScope;

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
    pub(crate) fn modules_for_source(&self, source: ModuleSource) -> Vec<ModuleId> {
        self.mods
            .iter()
            .enumerate()
            .filter(|(_idx, it)| it.source == source)
            .map(|(idx, _)| ModuleId(idx as u32))
            .collect()
    }

    pub(crate) fn any_module_for_source(&self, source: ModuleSource) -> Option<ModuleId> {
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
pub(crate) struct LinkId(u32);

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
        tree.module(self).source
    }
    pub(crate) fn parent_link(self, tree: &ModuleTree) -> Option<LinkId> {
        tree.module(self).parent
    }
    pub(crate) fn parent(self, tree: &ModuleTree) -> Option<ModuleId> {
        let link = self.parent_link(tree)?;
        Some(tree.link(link).owner)
    }
    pub(crate) fn root(self, tree: &ModuleTree) -> ModuleId {
        let mut curr = self;
        let mut i = 0;
        while let Some(next) = curr.parent(tree) {
            curr = next;
            i += 1;
            // simplistic cycle detection
            if i > 100 {
                return self;
            }
        }
        curr
    }
    pub(crate) fn child(self, tree: &ModuleTree, name: &str) -> Option<ModuleId> {
        let link = tree
            .module(self)
            .children
            .iter()
            .map(|&it| tree.link(it))
            .find(|it| it.name == name)?;
        Some(*link.points_to.first()?)
    }
    pub(crate) fn problems(
        self,
        tree: &ModuleTree,
        db: &impl SyntaxDatabase,
    ) -> Vec<(SyntaxNode, Problem)> {
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
    pub(crate) fn owner(self, tree: &ModuleTree) -> ModuleId {
        tree.link(self).owner
    }
    pub(crate) fn bind_source<'a>(
        self,
        tree: &ModuleTree,
        db: &impl SyntaxDatabase,
    ) -> ast::ModuleNode {
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
    pub(crate) fn new_inline(file_id: FileId, module: ast::Module) -> ModuleSource {
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
