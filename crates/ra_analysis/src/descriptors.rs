use crate::{imp::FileResolverImp, FileId};
use ra_syntax::{
    ast::{self, AstNode, NameOwner},
    text_utils::is_subrange,
    SmolStr,
};
use relative_path::RelativePathBuf;

use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ModuleDescriptor {
    pub submodules: Vec<Submodule>,
}

impl ModuleDescriptor {
    pub fn new(root: ast::Root) -> ModuleDescriptor {
        let submodules = modules(root).map(|(name, _)| Submodule { name }).collect();

        ModuleDescriptor { submodules }
    }
}

fn modules(root: ast::Root<'_>) -> impl Iterator<Item = (SmolStr, ast::Module<'_>)> {
    root.modules().filter_map(|module| {
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

#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct ModuleTreeDescriptor {
    nodes: Vec<NodeData>,
    links: Vec<LinkData>,
    file_id2node: BTreeMap<FileId, Node>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Node(usize);
#[derive(Hash, Debug, PartialEq, Eq)]
struct NodeData {
    file_id: FileId,
    links: Vec<Link>,
    parents: Vec<Link>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Link(usize);
#[derive(Hash, Debug, PartialEq, Eq)]
struct LinkData {
    owner: Node,
    name: SmolStr,
    points_to: Vec<Node>,
    problem: Option<Problem>,
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

impl ModuleTreeDescriptor {
    pub(crate) fn new<'a>(
        files: impl Iterator<Item = (FileId, &'a ModuleDescriptor)> + Clone,
        file_resolver: &FileResolverImp,
    ) -> ModuleTreeDescriptor {
        let mut file_id2node = BTreeMap::new();
        let mut nodes: Vec<NodeData> = files
            .clone()
            .enumerate()
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
            nodes,
            links,
            file_id2node,
        }
    }

    pub(crate) fn parent_modules(&self, file_id: FileId) -> Vec<Link> {
        let node = self.file_id2node[&file_id];
        self.node(node).parents.clone()
    }
    pub(crate) fn child_module_by_name(&self, file_id: FileId, name: &str) -> Vec<FileId> {
        let node = self.file_id2node[&file_id];
        self.node(node)
            .links
            .iter()
            .filter(|it| it.name(self) == name)
            .flat_map(|link| {
                link.points_to(self)
                    .iter()
                    .map(|&node| self.node(node).file_id)
            })
            .collect()
    }
    pub(crate) fn problems<'a, 'b>(
        &'b self,
        file_id: FileId,
        root: ast::Root<'a>,
    ) -> Vec<(ast::Name<'a>, &'b Problem)> {
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
    fn points_to(self, tree: &ModuleTreeDescriptor) -> &[Node] {
        &tree.link(self).points_to
    }
    pub(crate) fn bind_source<'a>(
        self,
        tree: &ModuleTreeDescriptor,
        root: ast::Root<'a>,
    ) -> ast::Module<'a> {
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
    file_resolver: &FileResolverImp,
) -> (Vec<FileId>, Option<Problem>) {
    let mod_name = file_resolver.file_stem(file_id);
    let is_dir_owner = mod_name == "mod" || mod_name == "lib" || mod_name == "main";

    let file_mod = RelativePathBuf::from(format!("../{}.rs", name));
    let dir_mod = RelativePathBuf::from(format!("../{}/mod.rs", name));
    let points_to: Vec<FileId>;
    let problem: Option<Problem>;
    if is_dir_owner {
        points_to = [&file_mod, &dir_mod]
            .iter()
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

#[derive(Debug, Clone)]
pub struct FnDescriptor {
    pub name: String,
    pub label: String,
    pub ret_type: Option<String>,
    pub params: Vec<String>,
}

impl FnDescriptor {
    pub fn new_opt(node: ast::FnDef) -> Option<Self> {
        let name = node.name()?.text().to_string();

        // Strip the body out for the label.
        let label: String = if let Some(body) = node.body() {
            let body_range = body.syntax().range();
            let label: String = node
                .syntax()
                .children()
                .filter(|child| !is_subrange(body_range, child.range()))
                .map(|node| node.text().to_string())
                .collect();
            label
        } else {
            node.syntax().text().to_string()
        };

        let params = FnDescriptor::param_list(node);
        let ret_type = node.ret_type().map(|r| r.syntax().text().to_string());

        Some(FnDescriptor {
            name,
            ret_type,
            params,
            label,
        })
    }

    fn param_list(node: ast::FnDef) -> Vec<String> {
        let mut res = vec![];
        if let Some(param_list) = node.param_list() {
            if let Some(self_param) = param_list.self_param() {
                res.push(self_param.syntax().text().to_string())
            }

            // Maybe use param.pat here? See if we can just extract the name?
            //res.extend(param_list.params().map(|p| p.syntax().text().to_string()));
            res.extend(
                param_list
                    .params()
                    .filter_map(|p| p.pat())
                    .map(|pat| pat.syntax().text().to_string()),
            );
        }
        res
    }
}
