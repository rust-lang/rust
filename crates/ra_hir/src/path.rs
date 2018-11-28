use ra_syntax::{SmolStr, ast, AstNode, TextRange};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Path {
    pub kind: PathKind,
    pub segments: Vec<SmolStr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathKind {
    Plain,
    Self_,
    Super,
    Crate,
}

impl Path {
    /// Calls `cb` with all paths, represented by this use item.
    pub fn expand_use_item(item: ast::UseItem, mut cb: impl FnMut(Path, Option<TextRange>)) {
        if let Some(tree) = item.use_tree() {
            expand_use_tree(None, tree, &mut cb);
        }
    }

    /// Converts an `ast::Path` to `Path`. Works with use trees.
    pub fn from_ast(mut path: ast::Path) -> Option<Path> {
        let mut kind = PathKind::Plain;
        let mut segments = Vec::new();
        loop {
            let segment = path.segment()?;
            match segment.kind()? {
                ast::PathSegmentKind::Name(name) => segments.push(name.text()),
                ast::PathSegmentKind::CrateKw => {
                    kind = PathKind::Crate;
                    break;
                }
                ast::PathSegmentKind::SelfKw => {
                    kind = PathKind::Self_;
                    break;
                }
                ast::PathSegmentKind::SuperKw => {
                    kind = PathKind::Super;
                    break;
                }
            }
            path = match qualifier(path) {
                Some(it) => it,
                None => break,
            };
        }
        segments.reverse();
        return Some(Path { kind, segments });

        fn qualifier(path: ast::Path) -> Option<ast::Path> {
            if let Some(q) = path.qualifier() {
                return Some(q);
            }
            // TODO: this bottom up traversal is not too precise.
            // Should we handle do a top-down analysiss, recording results?
            let use_tree_list = path.syntax().ancestors().find_map(ast::UseTreeList::cast)?;
            let use_tree = use_tree_list.parent_use_tree();
            use_tree.path()
        }
    }

    /// `true` is this path is a single identifier, like `foo`
    pub fn is_ident(&self) -> bool {
        self.kind == PathKind::Plain && self.segments.len() == 1
    }
}

fn expand_use_tree(
    prefix: Option<Path>,
    tree: ast::UseTree,
    cb: &mut impl FnMut(Path, Option<TextRange>),
) {
    if let Some(use_tree_list) = tree.use_tree_list() {
        let prefix = match tree.path() {
            None => prefix,
            Some(path) => match convert_path(prefix, path) {
                Some(it) => Some(it),
                None => return, // TODO: report errors somewhere
            },
        };
        for tree in use_tree_list.use_trees() {
            expand_use_tree(prefix.clone(), tree, cb);
        }
    } else {
        if let Some(ast_path) = tree.path() {
            if let Some(path) = convert_path(prefix, ast_path) {
                let range = if tree.has_star() {
                    None
                } else {
                    let range = ast_path.segment().unwrap().syntax().range();
                    Some(range)
                };
                cb(path, range)
            }
        }
    }
}

fn convert_path(prefix: Option<Path>, path: ast::Path) -> Option<Path> {
    let prefix = if let Some(qual) = path.qualifier() {
        Some(convert_path(prefix, qual)?)
    } else {
        None
    };
    let segment = path.segment()?;
    let res = match segment.kind()? {
        ast::PathSegmentKind::Name(name) => {
            let mut res = prefix.unwrap_or_else(|| Path {
                kind: PathKind::Plain,
                segments: Vec::with_capacity(1),
            });
            res.segments.push(name.text());
            res
        }
        ast::PathSegmentKind::CrateKw => {
            if prefix.is_some() {
                return None;
            }
            Path {
                kind: PathKind::Crate,
                segments: Vec::new(),
            }
        }
        ast::PathSegmentKind::SelfKw => {
            if prefix.is_some() {
                return None;
            }
            Path {
                kind: PathKind::Self_,
                segments: Vec::new(),
            }
        }
        ast::PathSegmentKind::SuperKw => {
            if prefix.is_some() {
                return None;
            }
            Path {
                kind: PathKind::Super,
                segments: Vec::new(),
            }
        }
    };
    Some(res)
}
