//! HIR for references to types. Paths in these are not yet resolved. They can
//! be directly created from an ast::TypeRef, without further queries.

use std::fmt::Write;

use hir_expand::{
    name::{AsName, Name},
    AstId,
};
use syntax::ast::{self, HasName};

use crate::{
    body::LowerCtx,
    builtin_type::{BuiltinInt, BuiltinType, BuiltinUint},
    expr::Literal,
    intern::Interned,
    path::Path,
};

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

    /// Returns `true` if the mutability is [`Mut`].
    ///
    /// [`Mut`]: Mutability::Mut
    #[must_use]
    pub fn is_mut(&self) -> bool {
        matches!(self, Self::Mut)
    }

    /// Returns `true` if the mutability is [`Shared`].
    ///
    /// [`Shared`]: Mutability::Shared
    #[must_use]
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Shared)
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

    pub fn is_raw(&self) -> bool {
        matches!(self, Self::RawPtr)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TraitRef {
    pub path: Path,
}

impl TraitRef {
    /// Converts an `ast::PathType` to a `hir::TraitRef`.
    pub(crate) fn from_ast(ctx: &LowerCtx<'_>, node: ast::Type) -> Option<Self> {
        // FIXME: Use `Path::from_src`
        match node {
            ast::Type::PathType(path) => {
                path.path().and_then(|it| ctx.lower_path(it)).map(|path| TraitRef { path })
            }
            _ => None,
        }
    }
}

/// Compare ty::Ty
///
/// Note: Most users of `TypeRef` that end up in the salsa database intern it using
/// `Interned<TypeRef>` to save space. But notably, nested `TypeRef`s are not interned, since that
/// does not seem to save any noticeable amount of memory.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TypeRef {
    Never,
    Placeholder,
    Tuple(Vec<TypeRef>),
    Path(Path),
    RawPtr(Box<TypeRef>, Mutability),
    Reference(Box<TypeRef>, Option<LifetimeRef>, Mutability),
    // FIXME: for full const generics, the latter element (length) here is going to have to be an
    // expression that is further lowered later in hir_ty.
    Array(Box<TypeRef>, ConstScalarOrPath),
    Slice(Box<TypeRef>),
    /// A fn pointer. Last element of the vector is the return type.
    Fn(Vec<(Option<Name>, TypeRef)>, bool /*varargs*/),
    ImplTrait(Vec<Interned<TypeBound>>),
    DynTrait(Vec<Interned<TypeBound>>),
    Macro(AstId<ast::MacroCall>),
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
    Path(Path, TraitBoundModifier),
    ForLifetime(Box<[Name]>, Path),
    Lifetime(LifetimeRef),
    Error,
}

/// A modifier on a bound, currently this is only used for `?Sized`, where the
/// modifier is `Maybe`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TraitBoundModifier {
    None,
    Maybe,
}

impl TypeRef {
    /// Converts an `ast::TypeRef` to a `hir::TypeRef`.
    pub fn from_ast(ctx: &LowerCtx<'_>, node: ast::Type) -> Self {
        match node {
            ast::Type::ParenType(inner) => TypeRef::from_ast_opt(ctx, inner.ty()),
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
                let inner_ty = TypeRef::from_ast_opt(ctx, inner.ty());
                let mutability = Mutability::from_mutable(inner.mut_token().is_some());
                TypeRef::RawPtr(Box::new(inner_ty), mutability)
            }
            ast::Type::ArrayType(inner) => {
                // FIXME: This is a hack. We should probably reuse the machinery of
                // `hir_def::body::lower` to lower this into an `Expr` and then evaluate it at the
                // `hir_ty` level, which would allow knowing the type of:
                // let v: [u8; 2 + 2] = [0u8; 4];
                let len = ConstScalarOrPath::from_expr_opt(inner.expr());
                TypeRef::Array(Box::new(TypeRef::from_ast_opt(ctx, inner.ty())), len)
            }
            ast::Type::SliceType(inner) => {
                TypeRef::Slice(Box::new(TypeRef::from_ast_opt(ctx, inner.ty())))
            }
            ast::Type::RefType(inner) => {
                let inner_ty = TypeRef::from_ast_opt(ctx, inner.ty());
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

                    pl.params()
                        .map(|it| {
                            let type_ref = TypeRef::from_ast_opt(ctx, it.ty());
                            let name = match it.pat() {
                                Some(ast::Pat::IdentPat(it)) => Some(
                                    it.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing),
                                ),
                                _ => None,
                            };
                            (name, type_ref)
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                params.push((None, ret_ty));
                TypeRef::Fn(params, is_varargs)
            }
            // for types are close enough for our purposes to the inner type for now...
            ast::Type::ForType(inner) => TypeRef::from_ast_opt(ctx, inner.ty()),
            ast::Type::ImplTraitType(inner) => {
                TypeRef::ImplTrait(type_bounds_from_ast(ctx, inner.type_bound_list()))
            }
            ast::Type::DynTraitType(inner) => {
                TypeRef::DynTrait(type_bounds_from_ast(ctx, inner.type_bound_list()))
            }
            ast::Type::MacroType(mt) => match mt.macro_call() {
                Some(mc) => ctx.ast_id(ctx.db, &mc).map(TypeRef::Macro).unwrap_or(TypeRef::Error),
                None => TypeRef::Error,
            },
        }
    }

    pub(crate) fn from_ast_opt(ctx: &LowerCtx<'_>, node: Option<ast::Type>) -> Self {
        match node {
            Some(node) => TypeRef::from_ast(ctx, node),
            None => TypeRef::Error,
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
                TypeRef::Fn(params, _) => {
                    params.iter().for_each(|(_, param_type)| go(param_type, f))
                }
                TypeRef::Tuple(types) => types.iter().for_each(|t| go(t, f)),
                TypeRef::RawPtr(type_ref, _)
                | TypeRef::Reference(type_ref, ..)
                | TypeRef::Array(type_ref, _)
                | TypeRef::Slice(type_ref) => go(type_ref, f),
                TypeRef::ImplTrait(bounds) | TypeRef::DynTrait(bounds) => {
                    for bound in bounds {
                        match bound.as_ref() {
                            TypeBound::Path(path, _) | TypeBound::ForLifetime(_, path) => {
                                go_path(path, f)
                            }
                            TypeBound::Lifetime(_) | TypeBound::Error => (),
                        }
                    }
                }
                TypeRef::Path(path) => go_path(path, f),
                TypeRef::Never | TypeRef::Placeholder | TypeRef::Macro(_) | TypeRef::Error => {}
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
                            crate::path::GenericArg::Const(_)
                            | crate::path::GenericArg::Lifetime(_) => {}
                        }
                    }
                    for binding in &args_and_bindings.bindings {
                        if let Some(type_ref) = &binding.type_ref {
                            go(type_ref, f);
                        }
                        for bound in &binding.bounds {
                            match bound.as_ref() {
                                TypeBound::Path(path, _) | TypeBound::ForLifetime(_, path) => {
                                    go_path(path, f)
                                }
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
    lower_ctx: &LowerCtx<'_>,
    type_bounds_opt: Option<ast::TypeBoundList>,
) -> Vec<Interned<TypeBound>> {
    if let Some(type_bounds) = type_bounds_opt {
        type_bounds.bounds().map(|it| Interned::new(TypeBound::from_ast(lower_ctx, it))).collect()
    } else {
        vec![]
    }
}

impl TypeBound {
    pub(crate) fn from_ast(ctx: &LowerCtx<'_>, node: ast::TypeBound) -> Self {
        let lower_path_type = |path_type: ast::PathType| ctx.lower_path(path_type.path()?);

        match node.kind() {
            ast::TypeBoundKind::PathType(path_type) => {
                let m = match node.question_mark_token() {
                    Some(_) => TraitBoundModifier::Maybe,
                    None => TraitBoundModifier::None,
                };
                lower_path_type(path_type)
                    .map(|p| TypeBound::Path(p, m))
                    .unwrap_or(TypeBound::Error)
            }
            ast::TypeBoundKind::ForType(for_type) => {
                let lt_refs = match for_type.generic_param_list() {
                    Some(gpl) => gpl
                        .lifetime_params()
                        .flat_map(|lp| lp.lifetime().map(|lt| Name::new_lifetime(&lt)))
                        .collect(),
                    None => Box::default(),
                };
                let path = for_type.ty().and_then(|ty| match ty {
                    ast::Type::PathType(path_type) => lower_path_type(path_type),
                    _ => None,
                });
                match path {
                    Some(p) => TypeBound::ForLifetime(lt_refs, p),
                    None => TypeBound::Error,
                }
            }
            ast::TypeBoundKind::Lifetime(lifetime) => {
                TypeBound::Lifetime(LifetimeRef::new(&lifetime))
            }
        }
    }

    pub fn as_path(&self) -> Option<(&Path, &TraitBoundModifier)> {
        match self {
            TypeBound::Path(p, m) => Some((p, m)),
            TypeBound::ForLifetime(_, p) => Some((p, &TraitBoundModifier::None)),
            TypeBound::Lifetime(_) | TypeBound::Error => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstScalarOrPath {
    Scalar(ConstScalar),
    Path(Name),
}

impl std::fmt::Display for ConstScalarOrPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstScalarOrPath::Scalar(s) => s.fmt(f),
            ConstScalarOrPath::Path(n) => n.fmt(f),
        }
    }
}

impl ConstScalarOrPath {
    pub(crate) fn from_expr_opt(expr: Option<ast::Expr>) -> Self {
        match expr {
            Some(x) => Self::from_expr(x),
            None => Self::Scalar(ConstScalar::Unknown),
        }
    }

    // FIXME: as per the comments on `TypeRef::Array`, this evaluation should not happen at this
    // parse stage.
    fn from_expr(expr: ast::Expr) -> Self {
        match expr {
            ast::Expr::PathExpr(p) => {
                match p.path().and_then(|x| x.segment()).and_then(|x| x.name_ref()) {
                    Some(x) => Self::Path(x.as_name()),
                    None => Self::Scalar(ConstScalar::Unknown),
                }
            }
            ast::Expr::PrefixExpr(prefix_expr) => match prefix_expr.op_kind() {
                Some(ast::UnaryOp::Neg) => {
                    let unsigned = Self::from_expr_opt(prefix_expr.expr());
                    // Add sign
                    match unsigned {
                        Self::Scalar(ConstScalar::UInt(num)) => {
                            Self::Scalar(ConstScalar::Int(-(num as i128)))
                        }
                        other => other,
                    }
                }
                _ => Self::from_expr_opt(prefix_expr.expr()),
            },
            ast::Expr::Literal(literal) => Self::Scalar(match literal.kind() {
                ast::LiteralKind::IntNumber(num) => {
                    num.value().map(ConstScalar::UInt).unwrap_or(ConstScalar::Unknown)
                }
                ast::LiteralKind::Char(c) => {
                    c.value().map(ConstScalar::Char).unwrap_or(ConstScalar::Unknown)
                }
                ast::LiteralKind::Bool(f) => ConstScalar::Bool(f),
                _ => ConstScalar::Unknown,
            }),
            _ => Self::Scalar(ConstScalar::Unknown),
        }
    }
}

/// A concrete constant value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstScalar {
    Int(i128),
    UInt(u128),
    Bool(bool),
    Char(char),

    /// Case of an unknown value that rustc might know but we don't
    // FIXME: this is a hack to get around chalk not being able to represent unevaluatable
    // constants
    // https://github.com/rust-lang/rust-analyzer/pull/8813#issuecomment-840679177
    // https://rust-lang.zulipchat.com/#narrow/stream/144729-wg-traits/topic/Handling.20non.20evaluatable.20constants'.20equality/near/238386348
    Unknown,
}

impl ConstScalar {
    pub fn builtin_type(&self) -> BuiltinType {
        match self {
            ConstScalar::UInt(_) | ConstScalar::Unknown => BuiltinType::Uint(BuiltinUint::U128),
            ConstScalar::Int(_) => BuiltinType::Int(BuiltinInt::I128),
            ConstScalar::Char(_) => BuiltinType::Char,
            ConstScalar::Bool(_) => BuiltinType::Bool,
        }
    }
}

impl From<Literal> for ConstScalar {
    fn from(literal: Literal) -> Self {
        match literal {
            Literal::Char(c) => Self::Char(c),
            Literal::Bool(flag) => Self::Bool(flag),
            Literal::Int(num, _) => Self::Int(num),
            Literal::Uint(num, _) => Self::UInt(num),
            _ => Self::Unknown,
        }
    }
}

impl std::fmt::Display for ConstScalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            ConstScalar::Int(num) => num.fmt(f),
            ConstScalar::UInt(num) => num.fmt(f),
            ConstScalar::Bool(flag) => flag.fmt(f),
            ConstScalar::Char(c) => write!(f, "'{c}'"),
            ConstScalar::Unknown => f.write_char('_'),
        }
    }
}
