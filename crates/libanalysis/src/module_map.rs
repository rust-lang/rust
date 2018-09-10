use relative_path::RelativePathBuf;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use libsyntax2::{
    File,
    ast::{self, AstNode, NameOwner},
    SyntaxNode, SmolStr,
};
use {FileId, imp::FileResolverImp};

type SyntaxProvider<'a> = dyn Fn(FileId) -> &'a File + 'a;

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
    file_resolver: FileResolverImp,
    changes: Vec<(FileId, ChangeKind)>,
    links: Vec<Link>,
}

#[derive(Clone, Debug)]
struct Link {
    owner: ModuleId,
    syntax: SyntaxNode,
    points_to: Vec<ModuleId>,
    problem: Option<Problem>,
}

#[derive(Clone, Debug)]
pub enum Problem {
    UnresolvedModule {
        candidate: RelativePathBuf,
    },
    NotDirOwner {
        move_to: RelativePathBuf,
        candidate: RelativePathBuf,
    }
}

impl ModuleMap {
    pub fn new() -> ModuleMap {
        Default::default()
    }
    pub fn update_file(&mut self, file_id: FileId, change_kind: ChangeKind) {
        self.state.get_mut().changes.push((file_id, change_kind));
    }
    pub(crate) fn set_file_resolver(&mut self, file_resolver: FileResolverImp) {
        self.state.get_mut().file_resolver = file_resolver;
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
        syntax_provider: &SyntaxProvider,
    ) -> Vec<ModuleId> {
        self.links(syntax_provider)
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
        syntax_provider: &SyntaxProvider,
    ) -> Vec<(ModuleId, SmolStr, SyntaxNode)> {
        let mut res = Vec::new();
        self.for_each_parent_link(m, syntax_provider, |link| {
            res.push(
                (link.owner, link.name().clone(), link.syntax.clone())
            )
        });
        res
    }

    pub fn parent_module_ids(
        &self,
        m: ModuleId,
        syntax_provider: &SyntaxProvider,
    ) -> Vec<ModuleId> {
        let mut res = Vec::new();
        self.for_each_parent_link(m, syntax_provider, |link| res.push(link.owner));
        res
    }

    fn for_each_parent_link(
        &self,
        m: ModuleId,
        syntax_provider: &SyntaxProvider,
        f: impl FnMut(&Link)
    ) {
        self.links(syntax_provider)
            .links
            .iter()
            .filter(move |link| link.points_to.iter().any(|&it| it == m))
            .for_each(f)
    }

    pub fn problems(
        &self,
        file: FileId,
        syntax_provider: &SyntaxProvider,
        mut cb: impl FnMut(ast::Name, &Problem),
    ) {
        let module = self.file2module(file);
        let links = self.links(syntax_provider);
        links
            .links
            .iter()
            .filter(|link| link.owner == module)
            .filter_map(|link| {
                let problem = link.problem.as_ref()?;
                Some((link, problem))
            })
            .for_each(|(link, problem)| cb(link.name_node(), problem))
    }

    fn links(
        &self,
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
            guard.apply_changes(syntax_provider);
        }
        assert!(guard.changes.is_empty());
        RwLockWriteGuard::downgrade(guard)
    }
}

impl State {
    pub fn apply_changes(
        &mut self,
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
                    let resolver = &self.file_resolver;
                    self.links.extend(
                        file
                            .ast()
                            .modules()
                            .filter_map(|it| Link::new(mod_id, it))
                            .map(|mut link| {
                                link.resolve(resolver);
                                link
                            })
                    );
                }
            }
        }
        if reresolve {
            for link in self.links.iter_mut() {
                link.resolve(&self.file_resolver)
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
            problem: None,
        };
        Some(link)
    }

    fn name(&self) -> SmolStr {
        self.name_node().text()
    }

    fn name_node(&self) -> ast::Name {
        self.ast().name().unwrap()
    }

    fn ast(&self) -> ast::Module {
        ast::Module::cast(self.syntax.borrowed())
            .unwrap()
    }

    fn resolve(&mut self, file_resolver: &FileResolverImp) {
        if !self.ast().has_semi() {
            self.problem = None;
            self.points_to = Vec::new();
            return;
        }
        let (points_to, problem) = resolve_submodule(self.owner.0, &self.name(), file_resolver);
        self.problem = problem;
        self.points_to = points_to.into_iter().map(ModuleId).collect();
    }
}

pub(crate) fn resolve_submodule(file_id: FileId, name: &SmolStr, file_resolver: &FileResolverImp) -> (Vec<FileId>, Option<Problem>) {
    let mod_name = file_resolver.file_stem(file_id);
    let is_dir_owner =
        mod_name == "mod" || mod_name == "lib" || mod_name == "main";

    let file_mod = RelativePathBuf::from(format!("../{}.rs", name));
    let dir_mod = RelativePathBuf::from(format!("../{}/mod.rs", name));
    let points_to: Vec<FileId>;
    let problem: Option<Problem>;
    if is_dir_owner {
        points_to = [&file_mod, &dir_mod].iter()
            .filter_map(|path| file_resolver.resolve(file_id, path))
            .collect();
        problem = if points_to.is_empty() {
            Some(Problem::UnresolvedModule {
                candidate: file_mod,
            })
        } else {
            None
        }
    } else {
        points_to = Vec::new();
        problem = Some(Problem::NotDirOwner {
            move_to: RelativePathBuf::from(format!("../{}/mod.rs", mod_name)),
            candidate: file_mod,
        });
    }
    (points_to, problem)
}
