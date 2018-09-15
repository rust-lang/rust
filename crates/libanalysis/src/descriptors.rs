use std::{
    collections::BTreeMap,
};
use relative_path::RelativePathBuf;
use libsyntax2::{
    SmolStr,
    ast::{self, NameOwner},
};
use {
    FileId,
    imp::FileResolverImp,
};

#[derive(Debug, Hash)]
pub struct ModuleDescriptor {
    pub submodules: Vec<Submodule>
}

impl ModuleDescriptor {
    pub fn new(root: ast::Root) -> ModuleDescriptor {
        let submodules = modules(root)
            .map(|(name, _)| Submodule { name })
            .collect();

        ModuleDescriptor { submodules } }
}

fn modules<'a>(root: ast::Root<'a>) -> impl Iterator<Item=(SmolStr, ast::Module<'a>)> {
    root
        .modules()
        .filter_map(|module| {
            let name = module.name()?.text();
            if !module.has_semi() {
                return None;
            }
            Some((name, module))
        })
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Submodule {
    pub name: SmolStr,
}

#[derive(Hash)]
pub(crate) struct ModuleTreeDescriptor {
    nodes: Vec<NodeData>,
    links: Vec<LinkData>,
    file_id2node: BTreeMap<FileId, Node>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Node(usize);
#[derive(Hash)]
struct NodeData {
    file_id: FileId,
    links: Vec<Link>,
    parents: Vec<Link>
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Link(usize);
#[derive(Hash)]
struct LinkData {
    owner: Node,
    name: SmolStr,
    points_to: Vec<Node>,
    problem: Option<Problem>,
}


#[derive(Clone, Debug, Hash)]
pub enum Problem {
    UnresolvedModule {
        candidate: RelativePathBuf,
    },
    NotDirOwner {
        move_to: RelativePathBuf,
        candidate: RelativePathBuf,
    }
}

impl ModuleTreeDescriptor {
    pub(crate) fn new<'a>(
        files: impl Iterator<Item=(FileId, &'a ModuleDescriptor)> + Clone,
        file_resolver: &FileResolverImp,
    ) -> ModuleTreeDescriptor {
        let mut file_id2node = BTreeMap::new();
        let mut nodes: Vec<NodeData> = files.clone().enumerate()
            .map(|(idx, (file_id, _))| {
                file_id2node.insert(file_id, Node(idx));
                NodeData {
                    file_id,
                    links: Vec::new(),
                    parents: Vec::new(),
                }
            })
            .collect();
        let mut links = Vec::new();

        for (idx, (file_id, descr)) in files.enumerate() {
            let owner = Node(idx);
            for sub in descr.submodules.iter() {
                let link = Link(links.len());
                nodes[owner.0].links.push(link);
                let (points_to, problem) = resolve_submodule(file_id, &sub.name, file_resolver);
                let points_to = points_to
                    .into_iter()
                    .map(|file_id| {
                        let node = file_id2node[&file_id];
                        nodes[node.0].parents.push(link);
                        node
                    })
                    .collect();

                links.push(LinkData {
                    owner,
                    name: sub.name.clone(),
                    points_to,
                    problem,
                })

            }
        }

        ModuleTreeDescriptor {
            nodes, links, file_id2node
        }
    }

    pub(crate) fn parent_modules(&self, file_id: FileId) -> Vec<Link> {
        let node = self.file_id2node[&file_id];
        self.node(node)
            .parents
            .clone()
    }
    pub(crate) fn child_module_by_name(&self, file_id: FileId, name: &str) -> Vec<FileId> {
        let node = self.file_id2node[&file_id];
        self.node(node)
            .links
            .iter()
            .filter(|it| it.name(self) == name)
            .map(|link| link.owner(self))
            .collect()
    }
    pub(crate) fn problems<'a, 'b>(&'b self, file_id: FileId, root: ast::Root<'a>) -> Vec<(ast::Name<'a>, &'b Problem)> {
        let node = self.file_id2node[&file_id];
        self.node(node)
            .links
            .iter()
            .filter_map(|&link| {
                let problem = self.link(link).problem.as_ref()?;
                let name = link.bind_source(self, root).name()?;
                Some((name, problem))
            })
            .collect()
    }

    fn node(&self, node: Node) -> &NodeData {
        &self.nodes[node.0]
    }
    fn link(&self, link: Link) -> &LinkData {
        &self.links[link.0]
    }
}

impl Link {
    pub(crate) fn name(self, tree: &ModuleTreeDescriptor) -> SmolStr {
        tree.link(self).name.clone()
    }
    pub(crate) fn owner(self, tree: &ModuleTreeDescriptor) -> FileId {
        let owner = tree.link(self).owner;
        tree.node(owner).file_id
    }
    pub(crate) fn bind_source<'a>(self, tree: &ModuleTreeDescriptor, root: ast::Root<'a>) -> ast::Module<'a> {
        modules(root)
            .filter(|(name, _)| name == &tree.link(self).name)
            .next()
            .unwrap()
            .1
    }
}


fn resolve_submodule(
    file_id: FileId,
    name: &SmolStr,
    file_resolver: &FileResolverImp
) -> (Vec<FileId>, Option<Problem>) {
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
