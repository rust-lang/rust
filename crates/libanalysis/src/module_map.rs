use std::{
    path::{PathBuf},
};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use libsyntax2::{
    ParsedFile,
    ast::{self, AstNode, NameOwner},
    SyntaxNode, SmolStr,
};
use {FileId, FileResolver};

type SyntaxProvider<'a> = dyn Fn(FileId) -> ParsedFile + 'a;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModuleId(FileId);

#[derive(Debug, Default)]
pub struct ModuleMap {
    state: RwLock<State>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    Delete, Insert, Update
}

impl Clone for ModuleMap {
    fn clone(&self) -> ModuleMap {
        let state = self.state.read().clone();
        ModuleMap { state: RwLock::new(state) }
    }
}

#[derive(Clone, Debug, Default)]
struct State {
    changes: Vec<(FileId, ChangeKind)>,
    links: Vec<Link>,
}

#[derive(Clone, Debug)]
struct Link {
    owner: ModuleId,
    syntax: SyntaxNode,
    points_to: Vec<ModuleId>,
}

impl ModuleMap {
    pub fn update_file(&mut self, file: FileId, change_kind: ChangeKind) {
        self.state.get_mut().changes.push((file, change_kind));
    }

    pub fn module2file(&self, m: ModuleId) -> FileId {
        m.0
    }

    pub fn file2module(&self, file_id: FileId) -> ModuleId {
        ModuleId(file_id)
    }

    pub fn child_module_by_name<'a>(
        &self,
        parent_mod: ModuleId,
        child_mod: &str,
        file_resolver: &FileResolver,
        syntax_provider: &SyntaxProvider,
    ) -> Vec<ModuleId> {
        self.links(file_resolver, syntax_provider)
            .links
            .iter()
            .filter(|link| link.owner == parent_mod)
            .filter(|link| link.name() == child_mod)
            .filter_map(|it| it.points_to.first())
            .map(|&it| it)
            .collect()
    }

    pub fn parent_modules(
        &self,
        m: ModuleId,
        file_resolver: &FileResolver,
        syntax_provider: &SyntaxProvider,
    ) -> Vec<(ModuleId, SmolStr, SyntaxNode)> {
        let links = self.links(file_resolver, syntax_provider);
        let res = links
            .links
            .iter()
            .filter(move |link| link.points_to.iter().any(|&it| it == m))
            .map(|link| {
                (link.owner, link.name().clone(), link.syntax.clone())
            })
            .collect();
        res
    }

    fn links(
        &self,
        file_resolver: &FileResolver,
        syntax_provider: &SyntaxProvider,
    ) -> RwLockReadGuard<State> {
        {
            let guard = self.state.read();
            if guard.changes.is_empty() {
                return guard;
            }
        }
        let mut guard = self.state.write();
        if !guard.changes.is_empty() {
            guard.apply_changes(file_resolver, syntax_provider);
        }
        assert!(guard.changes.is_empty());
        RwLockWriteGuard::downgrade(guard)
    }
}

impl State {
    pub fn apply_changes(
        &mut self,
        file_resolver: &FileResolver,
        syntax_provider: &SyntaxProvider,
    ) {
        let mut reresolve = false;
        for (file_id, kind) in self.changes.drain(..) {
            let mod_id = ModuleId(file_id);
            self.links.retain(|link| link.owner != mod_id);
            match kind {
                ChangeKind::Delete => {
                    for link in self.links.iter_mut() {
                        link.points_to.retain(|&x| x != mod_id);
                    }
                }
                ChangeKind::Insert => {
                    let file = syntax_provider(file_id);
                    self.links.extend(
                        file
                            .ast()
                            .modules()
                            .filter_map(|it| Link::new(mod_id, it))
                    );
                    reresolve = true;
                }
                ChangeKind::Update => {
                    let file = syntax_provider(file_id);
                    self.links.extend(
                        file
                            .ast()
                            .modules()
                            .filter_map(|it| Link::new(mod_id, it))
                            .map(|mut link| {
                                link.resolve(file_resolver);
                                link
                            })
                    );
                }
            }
        }
        if reresolve {
            for link in self.links.iter_mut() {
                link.resolve(file_resolver)
            }
        }
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
