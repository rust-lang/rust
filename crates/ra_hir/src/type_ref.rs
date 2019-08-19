//! HIR for references to types. Paths in these are not yet resolved. They can
//! be directly created from an ast::TypeRef, without further queries.

use ra_syntax::ast::{self, TypeAscriptionOwner};

use crate::Path;

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

/// Compare ty::Ty
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TypeRef {
    Never,
    Placeholder,
    Tuple(Vec<TypeRef>),
    Path(Path),
    RawPtr(Box<TypeRef>, Mutability),
    Reference(Box<TypeRef>, Mutability),
    Array(Box<TypeRef> /*, Expr*/),
    Slice(Box<TypeRef>),
    /// A fn pointer. Last element of the vector is the return type.
    Fn(Vec<TypeRef>),
    // For
    // ImplTrait,
    // DynTrait,
    Error,
}

impl TypeRef {
    /// Converts an `ast::TypeRef` to a `hir::TypeRef`.
    pub(crate) fn from_ast(node: ast::TypeRef) -> Self {
        match node {
            ast::TypeRef::ParenType(inner) => TypeRef::from_ast_opt(inner.type_ref()),
            ast::TypeRef::TupleType(inner) => {
                TypeRef::Tuple(inner.fields().map(TypeRef::from_ast).collect())
            }
            ast::TypeRef::NeverType(..) => TypeRef::Never,
            ast::TypeRef::PathType(inner) => {
                inner.path().and_then(Path::from_ast).map(TypeRef::Path).unwrap_or(TypeRef::Error)
            }
            ast::TypeRef::PointerType(inner) => {
                let inner_ty = TypeRef::from_ast_opt(inner.type_ref());
                let mutability = Mutability::from_mutable(inner.is_mut());
                TypeRef::RawPtr(Box::new(inner_ty), mutability)
            }
            ast::TypeRef::ArrayType(inner) => {
                TypeRef::Array(Box::new(TypeRef::from_ast_opt(inner.type_ref())))
            }
            ast::TypeRef::SliceType(inner) => {
                TypeRef::Slice(Box::new(TypeRef::from_ast_opt(inner.type_ref())))
            }
            ast::TypeRef::ReferenceType(inner) => {
                let inner_ty = TypeRef::from_ast_opt(inner.type_ref());
                let mutability = Mutability::from_mutable(inner.is_mut());
                TypeRef::Reference(Box::new(inner_ty), mutability)
            }
            ast::TypeRef::PlaceholderType(_inner) => TypeRef::Placeholder,
            ast::TypeRef::FnPointerType(inner) => {
                let ret_ty = TypeRef::from_ast_opt(inner.ret_type().and_then(|rt| rt.type_ref()));
                let mut params = if let Some(pl) = inner.param_list() {
                    pl.params().map(|p| p.ascribed_type()).map(TypeRef::from_ast_opt).collect()
                } else {
                    Vec::new()
                };
                params.push(ret_ty);
                TypeRef::Fn(params)
            }
            // for types are close enough for our purposes to the inner type for now...
            ast::TypeRef::ForType(inner) => TypeRef::from_ast_opt(inner.type_ref()),
            ast::TypeRef::ImplTraitType(_inner) => TypeRef::Error,
            ast::TypeRef::DynTraitType(_inner) => TypeRef::Error,
        }
    }

    pub(crate) fn from_ast_opt(node: Option<ast::TypeRef>) -> Self {
        if let Some(node) = node {
            TypeRef::from_ast(node)
        } else {
            TypeRef::Error
        }
    }

    pub fn unit() -> TypeRef {
        TypeRef::Tuple(Vec::new())
    }
}
