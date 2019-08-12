use std::sync::Arc;

use ra_syntax::{
    ast::{self, NameOwner},
    AstNode,
};

use crate::{type_ref::TypeRef, AsName, Name};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Path {
    pub kind: PathKind,
    pub segments: Vec<PathSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PathSegment {
    pub name: Name,
    pub args_and_bindings: Option<Arc<GenericArgs>>,
}

/// Generic arguments to a path segment (e.g. the `i32` in `Option<i32>`). This
/// can (in the future) also include bindings of associated types, like in
/// `Iterator<Item = Foo>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenericArgs {
    pub args: Vec<GenericArg>,
    /// This specifies whether the args contain a Self type as the first
    /// element. This is the case for path segments like `<T as Trait>`, where
    /// `T` is actually a type parameter for the path `Trait` specifying the
    /// Self type. Otherwise, when we have a path `Trait<X, Y>`, the Self type
    /// is left out.
    pub has_self_type: bool,
    // someday also bindings
}

/// A single generic argument.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericArg {
    Type(TypeRef),
    // or lifetime...
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathKind {
    Plain,
    Self_,
    Super,
    Crate,
    // Absolute path
    Abs,
}

impl Path {
    /// Calls `cb` with all paths, represented by this use item.
    pub fn expand_use_item(
        item: &ast::UseItem,
        mut cb: impl FnMut(Path, &ast::UseTree, bool, Option<Name>),
    ) {
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

            if segment.has_colon_colon() {
                kind = PathKind::Abs;
            }

            match segment.kind()? {
                ast::PathSegmentKind::Name(name) => {
                    let args =
                        segment.type_arg_list().and_then(GenericArgs::from_ast).map(Arc::new);
                    let segment = PathSegment { name: name.as_name(), args_and_bindings: args };
                    segments.push(segment);
                }
                ast::PathSegmentKind::Type { type_ref, trait_ref } => {
                    assert!(path.qualifier().is_none()); // this can only occur at the first segment

                    // FIXME: handle <T> syntax (type segments without trait)

                    // <T as Trait<A>>::Foo desugars to Trait<Self=T, A>::Foo
                    let path = Path::from_ast(trait_ref?.path()?)?;
                    kind = path.kind;
                    let mut prefix_segments = path.segments;
                    prefix_segments.reverse();
                    segments.extend(prefix_segments);
                    // Insert the type reference (T in the above example) as Self parameter for the trait
                    let self_type = TypeRef::from_ast(type_ref?);
                    let mut last_segment = segments.last_mut()?;
                    if last_segment.args_and_bindings.is_none() {
                        last_segment.args_and_bindings = Some(Arc::new(GenericArgs::empty()));
                    };
                    let args = last_segment.args_and_bindings.as_mut().unwrap();
                    let mut args_inner = Arc::make_mut(args);
                    args_inner.has_self_type = true;
                    args_inner.args.insert(0, GenericArg::Type(self_type));
                }
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
            path = match qualifier(&path) {
                Some(it) => it,
                None => break,
            };
        }
        segments.reverse();
        return Some(Path { kind, segments });

        fn qualifier(path: &ast::Path) -> Option<ast::Path> {
            if let Some(q) = path.qualifier() {
                return Some(q);
            }
            // FIXME: this bottom up traversal is not too precise.
            // Should we handle do a top-down analysis, recording results?
            let use_tree_list = path.syntax().ancestors().find_map(ast::UseTreeList::cast)?;
            let use_tree = use_tree_list.parent_use_tree();
            use_tree.path()
        }
    }

    /// Converts an `ast::NameRef` into a single-identifier `Path`.
    pub fn from_name_ref(name_ref: &ast::NameRef) -> Path {
        name_ref.as_name().into()
    }

    /// `true` is this path is a single identifier, like `foo`
    pub fn is_ident(&self) -> bool {
        self.kind == PathKind::Plain && self.segments.len() == 1
    }

    /// `true` if this path is just a standalone `self`
    pub fn is_self(&self) -> bool {
        self.kind == PathKind::Self_ && self.segments.is_empty()
    }

    /// If this path is a single identifier, like `foo`, return its name.
    pub fn as_ident(&self) -> Option<&Name> {
        if self.kind != PathKind::Plain || self.segments.len() > 1 {
            return None;
        }
        self.segments.first().map(|s| &s.name)
    }

    pub fn expand_macro_expr(&self) -> Option<Name> {
        self.as_ident().and_then(|name| Some(name.clone()))
    }
}

impl GenericArgs {
    pub(crate) fn from_ast(node: ast::TypeArgList) -> Option<GenericArgs> {
        let mut args = Vec::new();
        for type_arg in node.type_args() {
            let type_ref = TypeRef::from_ast_opt(type_arg.type_ref());
            args.push(GenericArg::Type(type_ref));
        }
        // lifetimes and assoc type args ignored for now
        if !args.is_empty() {
            Some(GenericArgs { args, has_self_type: false })
        } else {
            None
        }
    }

    pub(crate) fn empty() -> GenericArgs {
        GenericArgs { args: Vec::new(), has_self_type: false }
    }
}

impl From<Name> for Path {
    fn from(name: Name) -> Path {
        Path {
            kind: PathKind::Plain,
            segments: vec![PathSegment { name, args_and_bindings: None }],
        }
    }
}

fn expand_use_tree(
    prefix: Option<Path>,
    tree: ast::UseTree,
    cb: &mut impl FnMut(Path, &ast::UseTree, bool, Option<Name>),
) {
    if let Some(use_tree_list) = tree.use_tree_list() {
        let prefix = match tree.path() {
            // E.g. use something::{{{inner}}};
            None => prefix,
            // E.g. `use something::{inner}` (prefix is `None`, path is `something`)
            // or `use something::{path::{inner::{innerer}}}` (prefix is `something::path`, path is `inner`)
            Some(path) => match convert_path(prefix, path) {
                Some(it) => Some(it),
                None => return, // FIXME: report errors somewhere
            },
        };
        for child_tree in use_tree_list.use_trees() {
            expand_use_tree(prefix.clone(), child_tree, cb);
        }
    } else {
        let alias = tree.alias().and_then(|a| a.name()).map(|a| a.as_name());
        if let Some(ast_path) = tree.path() {
            // Handle self in a path.
            // E.g. `use something::{self, <...>}`
            if ast_path.qualifier().is_none() {
                if let Some(segment) = ast_path.segment() {
                    if segment.kind() == Some(ast::PathSegmentKind::SelfKw) {
                        if let Some(prefix) = prefix {
                            cb(prefix, &tree, false, alias);
                            return;
                        }
                    }
                }
            }
            if let Some(path) = convert_path(prefix, ast_path) {
                let is_glob = tree.has_star();
                cb(path, &tree, is_glob, alias)
            }
            // FIXME: report errors somewhere
            // We get here if we do
        }
    }
}

fn convert_path(prefix: Option<Path>, path: ast::Path) -> Option<Path> {
    let prefix =
        if let Some(qual) = path.qualifier() { Some(convert_path(prefix, qual)?) } else { prefix };
    let segment = path.segment()?;
    let res = match segment.kind()? {
        ast::PathSegmentKind::Name(name) => {
            let mut res = prefix
                .unwrap_or_else(|| Path { kind: PathKind::Plain, segments: Vec::with_capacity(1) });
            res.segments.push(PathSegment {
                name: name.as_name(),
                args_and_bindings: None, // no type args in use
            });
            res
        }
        ast::PathSegmentKind::CrateKw => {
            if prefix.is_some() {
                return None;
            }
            Path { kind: PathKind::Crate, segments: Vec::new() }
        }
        ast::PathSegmentKind::SelfKw => {
            if prefix.is_some() {
                return None;
            }
            Path { kind: PathKind::Self_, segments: Vec::new() }
        }
        ast::PathSegmentKind::SuperKw => {
            if prefix.is_some() {
                return None;
            }
            Path { kind: PathKind::Super, segments: Vec::new() }
        }
        ast::PathSegmentKind::Type { .. } => {
            // not allowed in imports
            return None;
        }
    };
    Some(res)
}
