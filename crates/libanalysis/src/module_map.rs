use std::{
    path::{PathBuf},
};

use libsyntax2::{
    ast::{self, AstNode, NameOwner},
    SyntaxNode, ParsedFile, SmolStr,
};
use {FileId, FileResolver};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModuleId(FileId);

#[derive(Clone, Debug, Default)]
pub struct ModuleMap {
    links: Vec<Link>,
}

#[derive(Clone, Debug)]
struct Link {
    owner: ModuleId,
    syntax: SyntaxNode,
    points_to: Vec<ModuleId>,
}

impl ModuleMap {
    pub fn update_file(
        &mut self,
        file_id: FileId,
        syntax: Option<&ParsedFile>,
        file_resolver: &FileResolver,
    ) {
        let mod_id = ModuleId(file_id);
        self.links.retain(|link| link.owner != mod_id);
        match syntax {
            None => {
                for link in self.links.iter_mut() {
                    link.points_to.retain(|&x| x != mod_id);
                }
            }
            Some(syntax) => {
                self.links.extend(
                    syntax.ast().modules().filter_map(|it| {
                        Link::new(mod_id, it)
                    })
                )
            }
        }
        self.links.iter_mut().for_each(|link| {
            link.resolve(file_resolver)
        })
    }

    pub fn module2file(&self, m: ModuleId) -> FileId {
        m.0
    }

    pub fn file2module(&self, file_id: FileId) -> ModuleId {
        ModuleId(file_id)
    }

    pub fn child_module_by_name(&self, parent_mod: ModuleId, child_mod: &str) -> Vec<ModuleId> {
        self.links
            .iter()
            .filter(|link| link.owner == parent_mod)
            .filter(|link| link.name() == child_mod)
            .filter_map(|it| it.points_to.first())
            .map(|&it| it)
            .collect()
    }

    pub fn parent_modules<'a>(&'a self, m: ModuleId) -> impl Iterator<Item=(ModuleId, ast::Module<'a>)> + 'a {
        self.links
            .iter()
            .filter(move |link| link.points_to.iter().any(|&it| it == m))
            .map(|link| {
                (link.owner, link.ast())
            })
    }
}

impl Link {
    fn new(owner: ModuleId, module: ast::Module) -> Option<Link> {
        if module.name().is_none() {
            return None;
        }
        let link = Link {
            owner,
            syntax: module.syntax().owned(),
            points_to: Vec::new(),
        };
        Some(link)
    }

    fn name(&self) -> SmolStr {
        self.ast().name()
            .unwrap()
            .text()
    }

    fn ast(&self) -> ast::Module {
        ast::Module::cast(self.syntax.borrowed())
            .unwrap()
    }

    fn resolve(&mut self, file_resolver: &FileResolver) {
        let name = self.name();
        let paths = &[
            PathBuf::from(format!("../{}.rs", name)),
            PathBuf::from(format!("../{}/mod.rs", name)),
        ];
        self.points_to = paths.iter()
            .filter_map(|path| file_resolver(self.owner.0, path))
            .map(ModuleId)
            .collect();
    }
}
