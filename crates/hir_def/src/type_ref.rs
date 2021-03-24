//! HIR for references to types. Paths in these are not yet resolved. They can
//! be directly created from an ast::TypeRef, without further queries.
use hir_expand::name::Name;
use syntax::ast;

use crate::{body::LowerCtx, path::Path};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Mutability {
    Shared,
    Mut,
}

impl Mutability {
    pub fn from_mutable(mutable: bool) -> Mutability {
        if mutable {
            Mutability::Mut
        } else {
            Mutability::Shared
        }
    }

    pub fn as_keyword_for_ref(self) -> &'static str {
        match self {
            Mutability::Shared => "",
            Mutability::Mut => "mut ",
        }
    }

    pub fn as_keyword_for_ptr(self) -> &'static str {
        match self {
            Mutability::Shared => "const ",
            Mutability::Mut => "mut ",
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Rawness {
    RawPtr,
    Ref,
}

impl Rawness {
    pub fn from_raw(is_raw: bool) -> Rawness {
        if is_raw {
            Rawness::RawPtr
        } else {
            Rawness::Ref
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TraitRef {
    Path(Path),
    Error,
}

impl TraitRef {
    /// Converts an `ast::PathType` to a `hir::TraitRef`.
    pub(crate) fn from_ast(ctx: &LowerCtx, node: ast::Type) -> Self {
        // FIXME: Use `Path::from_src`
        match node {
            ast::Type::PathType(path) => path
                .path()
                .and_then(|it| ctx.lower_path(it))
                .map_or(TraitRef::Error, TraitRef::Path),
            _ => TraitRef::Error,
        }
    }
}
/// Compare ty::Ty
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TypeRef {
    Never,
    Placeholder,
    Tuple(Vec<TypeRef>),
    Path(Path),
    RawPtr(Box<TypeRef>, Mutability),
    Reference(Box<TypeRef>, Option<LifetimeRef>, Mutability),
    Array(Box<TypeRef> /*, Expr*/),
    Slice(Box<TypeRef>),
    /// A fn pointer. Last element of the vector is the return type.
    Fn(Vec<TypeRef>, bool /*varargs*/),
    // For
    ImplTrait(Vec<TypeBound>),
    DynTrait(Vec<TypeBound>),
    Error,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct LifetimeRef {
    pub name: Name,
}

impl LifetimeRef {
    pub(crate) fn new_name(name: Name) -> Self {
        LifetimeRef { name }
    }

    pub(crate) fn new(lifetime: &ast::Lifetime) -> Self {
        LifetimeRef { name: Name::new_lifetime(lifetime) }
    }

    pub fn missing() -> LifetimeRef {
        LifetimeRef { name: Name::missing() }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TypeBound {
    Path(Path),
    // ForLifetime(Vec<LifetimeRef>, Path), FIXME ForLifetime
    Lifetime(LifetimeRef),
    Error,
}

impl TypeRef {
    /// Converts an `ast::TypeRef` to a `hir::TypeRef`.
    pub(crate) fn from_ast(ctx: &LowerCtx, node: ast::Type) -> Self {
        match node {
            ast::Type::ParenType(inner) => TypeRef::from_ast_opt(&ctx, inner.ty()),
            ast::Type::TupleType(inner) => {
                TypeRef::Tuple(inner.fields().map(|it| TypeRef::from_ast(ctx, it)).collect())
            }
            ast::Type::NeverType(..) => TypeRef::Never,
            ast::Type::PathType(inner) => {
                // FIXME: Use `Path::from_src`
                inner
                    .path()
                    .and_then(|it| ctx.lower_path(it))
                    .map(TypeRef::Path)
                    .unwrap_or(TypeRef::Error)
            }
            ast::Type::PtrType(inner) => {
                let inner_ty = TypeRef::from_ast_opt(&ctx, inner.ty());
                let mutability = Mutability::from_mutable(inner.mut_token().is_some());
                TypeRef::RawPtr(Box::new(inner_ty), mutability)
            }
            ast::Type::ArrayType(inner) => {
                TypeRef::Array(Box::new(TypeRef::from_ast_opt(&ctx, inner.ty())))
            }
            ast::Type::SliceType(inner) => {
                TypeRef::Slice(Box::new(TypeRef::from_ast_opt(&ctx, inner.ty())))
            }
            ast::Type::RefType(inner) => {
                let inner_ty = TypeRef::from_ast_opt(&ctx, inner.ty());
                let lifetime = inner.lifetime().map(|lt| LifetimeRef::new(&lt));
                let mutability = Mutability::from_mutable(inner.mut_token().is_some());
                TypeRef::Reference(Box::new(inner_ty), lifetime, mutability)
            }
            ast::Type::InferType(_inner) => TypeRef::Placeholder,
            ast::Type::FnPtrType(inner) => {
                let ret_ty = inner
                    .ret_type()
                    .and_then(|rt| rt.ty())
                    .map(|it| TypeRef::from_ast(ctx, it))
                    .unwrap_or_else(|| TypeRef::Tuple(Vec::new()));
                let mut is_varargs = false;
                let mut params = if let Some(pl) = inner.param_list() {
                    if let Some(param) = pl.params().last() {
                        is_varargs = param.dotdotdot_token().is_some();
                    }

                    pl.params().map(|p| p.ty()).map(|it| TypeRef::from_ast_opt(&ctx, it)).collect()
                } else {
                    Vec::new()
                };
                params.push(ret_ty);
                TypeRef::Fn(params, is_varargs)
            }
            // for types are close enough for our purposes to the inner type for now...
            ast::Type::ForType(inner) => TypeRef::from_ast_opt(&ctx, inner.ty()),
            ast::Type::ImplTraitType(inner) => {
                TypeRef::ImplTrait(type_bounds_from_ast(ctx, inner.type_bound_list()))
            }
            ast::Type::DynTraitType(inner) => {
                TypeRef::DynTrait(type_bounds_from_ast(ctx, inner.type_bound_list()))
            }
            // FIXME: Macros in type position are not yet supported.
            ast::Type::MacroType(_) => TypeRef::Error,
        }
    }

    pub(crate) fn from_ast_opt(ctx: &LowerCtx, node: Option<ast::Type>) -> Self {
        if let Some(node) = node {
            TypeRef::from_ast(ctx, node)
        } else {
            TypeRef::Error
        }
    }

    pub(crate) fn unit() -> TypeRef {
        TypeRef::Tuple(Vec::new())
    }

    pub fn walk(&self, f: &mut impl FnMut(&TypeRef)) {
        go(self, f);

        fn go(type_ref: &TypeRef, f: &mut impl FnMut(&TypeRef)) {
            f(type_ref);
            match type_ref {
                TypeRef::Fn(types, _) | TypeRef::Tuple(types) => {
                    types.iter().for_each(|t| go(t, f))
                }
                TypeRef::RawPtr(type_ref, _)
                | TypeRef::Reference(type_ref, ..)
                | TypeRef::Array(type_ref)
                | TypeRef::Slice(type_ref) => go(&type_ref, f),
                TypeRef::ImplTrait(bounds) | TypeRef::DynTrait(bounds) => {
                    for bound in bounds {
                        match bound {
                            TypeBound::Path(path) => go_path(path, f),
                            TypeBound::Lifetime(_) | TypeBound::Error => (),
                        }
                    }
                }
                TypeRef::Path(path) => go_path(path, f),
                TypeRef::Never | TypeRef::Placeholder | TypeRef::Error => {}
            };
        }

        fn go_path(path: &Path, f: &mut impl FnMut(&TypeRef)) {
            if let Some(type_ref) = path.type_anchor() {
                go(type_ref, f);
            }
            for segment in path.segments().iter() {
                if let Some(args_and_bindings) = segment.args_and_bindings {
                    for arg in &args_and_bindings.args {
                        match arg {
                            crate::path::GenericArg::Type(type_ref) => {
                                go(type_ref, f);
                            }
                            crate::path::GenericArg::Lifetime(_) => {}
                        }
                    }
                    for binding in &args_and_bindings.bindings {
                        if let Some(type_ref) = &binding.type_ref {
                            go(type_ref, f);
                        }
                        for bound in &binding.bounds {
                            match bound {
                                TypeBound::Path(path) => go_path(path, f),
                                TypeBound::Lifetime(_) | TypeBound::Error => (),
                            }
                        }
                    }
                }
            }
        }
    }
}

pub(crate) fn type_bounds_from_ast(
    lower_ctx: &LowerCtx,
    type_bounds_opt: Option<ast::TypeBoundList>,
) -> Vec<TypeBound> {
    if let Some(type_bounds) = type_bounds_opt {
        type_bounds.bounds().map(|it| TypeBound::from_ast(lower_ctx, it)).collect()
    } else {
        vec![]
    }
}

impl TypeBound {
    pub(crate) fn from_ast(ctx: &LowerCtx, node: ast::TypeBound) -> Self {
        match node.kind() {
            ast::TypeBoundKind::PathType(path_type) => {
                let path = match path_type.path() {
                    Some(p) => p,
                    None => return TypeBound::Error,
                };

                let path = match ctx.lower_path(path) {
                    Some(p) => p,
                    None => return TypeBound::Error,
                };
                TypeBound::Path(path)
            }
            ast::TypeBoundKind::ForType(_) => TypeBound::Error, // FIXME ForType
            ast::TypeBoundKind::Lifetime(lifetime) => {
                TypeBound::Lifetime(LifetimeRef::new(&lifetime))
            }
        }
    }

    pub fn as_path(&self) -> Option<&Path> {
        match self {
            TypeBound::Path(p) => Some(p),
            _ => None,
        }
    }
}
