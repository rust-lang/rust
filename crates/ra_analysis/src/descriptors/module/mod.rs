mod imp;

use std::sync::Arc;

use relative_path::RelativePathBuf;
use ra_syntax::{ast::{self, NameOwner, AstNode}, SmolStr, SyntaxNode};

use crate::{
    FileId, Cancelable,
    db::SyntaxDatabase,
    input::SourceRootId,
};

salsa::query_group! {
    pub(crate) trait ModulesDatabase: SyntaxDatabase {
        fn module_tree(source_root_id: SourceRootId) -> Cancelable<Arc<ModuleTree>> {
            type ModuleTreeQuery;
            use fn imp::module_tree;
        }
        fn submodules(file_id: FileId) -> Cancelable<Arc<Vec<SmolStr>>> {
            type SubmodulesQuery;
            use fn imp::submodules;
        }
    }
}


#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct ModuleTree {
    mods: Vec<ModuleData>,
    links: Vec<LinkData>,
}

impl ModuleTree {
    pub(crate) fn modules_for_file(&self, file_id: FileId) -> Vec<ModuleId> {
        self.mods.iter()
            .enumerate()
            .filter(|(_idx, it)| it.file_id == file_id).map(|(idx, _)| ModuleId(idx as u32))
            .collect()
    }

    pub(crate) fn any_module_for_file(&self, file_id: FileId) -> Option<ModuleId> {
        self.modules_for_file(file_id).pop()
    }
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
    pub(crate) fn file_id(self, tree: &ModuleTree) -> FileId {
        tree.module(self).file_id
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
            if i > 100 {
                return self;
            }
        }
        curr
    }
    pub(crate) fn child(self, tree: &ModuleTree, name: &str) -> Option<ModuleId> {
        let link = tree.module(self)
            .children
            .iter()
            .map(|&it| tree.link(it))
            .find(|it| it.name == name)?;
        Some(*link.points_to.first()?)
    }
    pub(crate) fn problems(
        self,
        tree: &ModuleTree,
        root: ast::Root,
    ) -> Vec<(SyntaxNode, Problem)> {
        tree.module(self)
            .children
            .iter()
            .filter_map(|&it| {
                let p = tree.link(it).problem.clone()?;
                let s = it.bind_source(tree, root);
                let s = s.name().unwrap().syntax().owned();
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
        root: ast::Root<'a>,
    ) -> ast::Module<'a> {
        imp::modules(root)
            .find(|(name, _)| name == &tree.link(self).name)
            .unwrap()
            .1
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct ModuleData {
    file_id: FileId,
    parent: Option<LinkId>,
    children: Vec<LinkId>,
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

