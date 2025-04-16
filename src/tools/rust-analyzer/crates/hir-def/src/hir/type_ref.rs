//! HIR for references to types. Paths in these are not yet resolved. They can
//! be directly created from an ast::TypeRef, without further queries.

use core::fmt;
use std::{fmt::Write, ops::Index};

use hir_expand::{
    db::ExpandDatabase,
    name::{AsName, Name},
    AstId, InFile,
};
use intern::{sym, Symbol};
use la_arena::{Arena, ArenaMap, Idx};
use span::Edition;
use stdx::thin_vec::{thin_vec_with_header_struct, EmptyOptimizedThinVec, ThinVec};
use syntax::{
    ast::{self, HasGenericArgs, HasName, IsString},
    AstPtr,
};

use crate::{
    builtin_type::{BuiltinInt, BuiltinType, BuiltinUint},
    hir::Literal,
    lower::LowerCtx,
    path::{GenericArg, Path},
    SyntheticSyntax,
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
/// A `TypeRefId` that is guaranteed to always be `TypeRef::Path`. We use this for things like
/// impl's trait, that are always paths but need to be traced back to source code.
pub struct PathId(TypeRefId);

impl PathId {
    #[inline]
    pub fn from_type_ref_unchecked(type_ref: TypeRefId) -> Self {
        Self(type_ref)
    }

    #[inline]
    pub fn type_ref(self) -> TypeRefId {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TraitRef {
    pub path: PathId,
}

impl TraitRef {
    /// Converts an `ast::PathType` to a `hir::TraitRef`.
    pub(crate) fn from_ast(ctx: &mut LowerCtx<'_>, node: ast::Type) -> Option<Self> {
        // FIXME: Use `Path::from_src`
        match &node {
            ast::Type::PathType(path) => path
                .path()
                .and_then(|it| ctx.lower_path(it))
                .map(|path| TraitRef { path: ctx.alloc_path(path, AstPtr::new(&node)) }),
            _ => None,
        }
    }
}

thin_vec_with_header_struct! {
    pub new(pub(crate)) struct FnType, FnTypeHeader {
        pub params: [(Option<Name>, TypeRefId)],
        pub is_varargs: bool,
        pub is_unsafe: bool,
        pub abi: Option<Symbol>; ref,
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ArrayType {
    pub ty: TypeRefId,
    // FIXME: This should be Ast<ConstArg>
    pub len: ConstRef,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RefType {
    pub ty: TypeRefId,
    pub lifetime: Option<LifetimeRef>,
    pub mutability: Mutability,
}

/// Compare ty::Ty
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TypeRef {
    Never,
    Placeholder,
    Tuple(EmptyOptimizedThinVec<TypeRefId>),
    Path(Path),
    RawPtr(TypeRefId, Mutability),
    Reference(Box<RefType>),
    Array(Box<ArrayType>),
    Slice(TypeRefId),
    /// A fn pointer. Last element of the vector is the return type.
    Fn(FnType),
    ImplTrait(ThinVec<TypeBound>),
    DynTrait(ThinVec<TypeBound>),
    Macro(AstId<ast::MacroCall>),
    Error,
}

#[cfg(target_arch = "x86_64")]
const _: () = assert!(size_of::<TypeRef>() == 16);

pub type TypeRefId = Idx<TypeRef>;

#[derive(Default, Clone, PartialEq, Eq, Debug, Hash)]
pub struct TypesMap {
    pub(crate) types: Arena<TypeRef>,
}

impl TypesMap {
    pub const EMPTY: &TypesMap = &TypesMap { types: Arena::new() };

    pub(crate) fn shrink_to_fit(&mut self) {
        let TypesMap { types } = self;
        types.shrink_to_fit();
    }
}

impl Index<TypeRefId> for TypesMap {
    type Output = TypeRef;

    #[inline]
    fn index(&self, index: TypeRefId) -> &Self::Output {
        &self.types[index]
    }
}

impl Index<PathId> for TypesMap {
    type Output = Path;

    #[inline]
    fn index(&self, index: PathId) -> &Self::Output {
        let TypeRef::Path(path) = &self[index.type_ref()] else {
            unreachable!("`PathId` always points to `TypeRef::Path`");
        };
        path
    }
}

pub type TypePtr = AstPtr<ast::Type>;
pub type TypeSource = InFile<TypePtr>;

#[derive(Default, Clone, PartialEq, Eq, Debug, Hash)]
pub struct TypesSourceMap {
    pub(crate) types_map_back: ArenaMap<TypeRefId, TypeSource>,
}

impl TypesSourceMap {
    pub const EMPTY: Self = Self { types_map_back: ArenaMap::new() };

    pub fn type_syntax(&self, id: TypeRefId) -> Result<TypeSource, SyntheticSyntax> {
        self.types_map_back.get(id).cloned().ok_or(SyntheticSyntax)
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        let TypesSourceMap { types_map_back } = self;
        types_map_back.shrink_to_fit();
    }
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
    Path(PathId, TraitBoundModifier),
    ForLifetime(Box<[Name]>, PathId),
    Lifetime(LifetimeRef),
    Use(Box<[UseArgRef]>),
    Error,
}

#[cfg(target_pointer_width = "64")]
const _: [(); 24] = [(); size_of::<TypeBound>()];

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum UseArgRef {
    Name(Name),
    Lifetime(LifetimeRef),
}

/// A modifier on a bound, currently this is only used for `?Sized`, where the
/// modifier is `Maybe`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TraitBoundModifier {
    None,
    Maybe,
}

impl TypeRef {
    /// Converts an `ast::TypeRef` to a `hir::TypeRef`.
    pub fn from_ast(ctx: &mut LowerCtx<'_>, node: ast::Type) -> TypeRefId {
        let ty = match &node {
            ast::Type::ParenType(inner) => return TypeRef::from_ast_opt(ctx, inner.ty()),
            ast::Type::TupleType(inner) => TypeRef::Tuple(EmptyOptimizedThinVec::from_iter(
                Vec::from_iter(inner.fields().map(|it| TypeRef::from_ast(ctx, it))),
            )),
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
                TypeRef::RawPtr(inner_ty, mutability)
            }
            ast::Type::ArrayType(inner) => {
                let len = ConstRef::from_const_arg(ctx, inner.const_arg());
                TypeRef::Array(Box::new(ArrayType {
                    ty: TypeRef::from_ast_opt(ctx, inner.ty()),
                    len,
                }))
            }
            ast::Type::SliceType(inner) => TypeRef::Slice(TypeRef::from_ast_opt(ctx, inner.ty())),
            ast::Type::RefType(inner) => {
                let inner_ty = TypeRef::from_ast_opt(ctx, inner.ty());
                let lifetime = inner.lifetime().map(|lt| LifetimeRef::new(&lt));
                let mutability = Mutability::from_mutable(inner.mut_token().is_some());
                TypeRef::Reference(Box::new(RefType { ty: inner_ty, lifetime, mutability }))
            }
            ast::Type::InferType(_inner) => TypeRef::Placeholder,
            ast::Type::FnPtrType(inner) => {
                let ret_ty = inner
                    .ret_type()
                    .and_then(|rt| rt.ty())
                    .map(|it| TypeRef::from_ast(ctx, it))
                    .unwrap_or_else(|| ctx.alloc_type_ref_desugared(TypeRef::unit()));
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
                    Vec::with_capacity(1)
                };
                fn lower_abi(abi: ast::Abi) -> Symbol {
                    match abi.abi_string() {
                        Some(tok) => Symbol::intern(tok.text_without_quotes()),
                        // `extern` default to be `extern "C"`.
                        _ => sym::C.clone(),
                    }
                }

                let abi = inner.abi().map(lower_abi);
                params.push((None, ret_ty));
                TypeRef::Fn(FnType::new(is_varargs, inner.unsafe_token().is_some(), abi, params))
            }
            // for types are close enough for our purposes to the inner type for now...
            ast::Type::ForType(inner) => return TypeRef::from_ast_opt(ctx, inner.ty()),
            ast::Type::ImplTraitType(inner) => {
                if ctx.outer_impl_trait() {
                    // Disallow nested impl traits
                    TypeRef::Error
                } else {
                    ctx.with_outer_impl_trait_scope(true, |ctx| {
                        TypeRef::ImplTrait(type_bounds_from_ast(ctx, inner.type_bound_list()))
                    })
                }
            }
            ast::Type::DynTraitType(inner) => {
                TypeRef::DynTrait(type_bounds_from_ast(ctx, inner.type_bound_list()))
            }
            ast::Type::MacroType(mt) => match mt.macro_call() {
                Some(mc) => TypeRef::Macro(ctx.ast_id(&mc)),
                None => TypeRef::Error,
            },
        };
        ctx.alloc_type_ref(ty, AstPtr::new(&node))
    }

    pub(crate) fn from_ast_opt(ctx: &mut LowerCtx<'_>, node: Option<ast::Type>) -> TypeRefId {
        match node {
            Some(node) => TypeRef::from_ast(ctx, node),
            None => ctx.alloc_error_type(),
        }
    }

    pub(crate) fn unit() -> TypeRef {
        TypeRef::Tuple(EmptyOptimizedThinVec::empty())
    }

    pub fn walk(this: TypeRefId, map: &TypesMap, f: &mut impl FnMut(&TypeRef)) {
        go(this, f, map);

        fn go(type_ref: TypeRefId, f: &mut impl FnMut(&TypeRef), map: &TypesMap) {
            let type_ref = &map[type_ref];
            f(type_ref);
            match type_ref {
                TypeRef::Fn(fn_) => {
                    fn_.params().iter().for_each(|&(_, param_type)| go(param_type, f, map))
                }
                TypeRef::Tuple(types) => types.iter().for_each(|&t| go(t, f, map)),
                TypeRef::RawPtr(type_ref, _) | TypeRef::Slice(type_ref) => go(*type_ref, f, map),
                TypeRef::Reference(it) => go(it.ty, f, map),
                TypeRef::Array(it) => go(it.ty, f, map),
                TypeRef::ImplTrait(bounds) | TypeRef::DynTrait(bounds) => {
                    for bound in bounds {
                        match bound {
                            &TypeBound::Path(path, _) | &TypeBound::ForLifetime(_, path) => {
                                go_path(&map[path], f, map)
                            }
                            TypeBound::Lifetime(_) | TypeBound::Error | TypeBound::Use(_) => (),
                        }
                    }
                }
                TypeRef::Path(path) => go_path(path, f, map),
                TypeRef::Never | TypeRef::Placeholder | TypeRef::Macro(_) | TypeRef::Error => {}
            };
        }

        fn go_path(path: &Path, f: &mut impl FnMut(&TypeRef), map: &TypesMap) {
            if let Some(type_ref) = path.type_anchor() {
                go(type_ref, f, map);
            }
            for segment in path.segments().iter() {
                if let Some(args_and_bindings) = segment.args_and_bindings {
                    for arg in args_and_bindings.args.iter() {
                        match arg {
                            GenericArg::Type(type_ref) => {
                                go(*type_ref, f, map);
                            }
                            GenericArg::Const(_) | GenericArg::Lifetime(_) => {}
                        }
                    }
                    for binding in args_and_bindings.bindings.iter() {
                        if let Some(type_ref) = binding.type_ref {
                            go(type_ref, f, map);
                        }
                        for bound in binding.bounds.iter() {
                            match bound {
                                &TypeBound::Path(path, _) | &TypeBound::ForLifetime(_, path) => {
                                    go_path(&map[path], f, map)
                                }
                                TypeBound::Lifetime(_) | TypeBound::Error | TypeBound::Use(_) => (),
                            }
                        }
                    }
                }
            }
        }
    }
}

pub(crate) fn type_bounds_from_ast(
    lower_ctx: &mut LowerCtx<'_>,
    type_bounds_opt: Option<ast::TypeBoundList>,
) -> ThinVec<TypeBound> {
    if let Some(type_bounds) = type_bounds_opt {
        ThinVec::from_iter(Vec::from_iter(
            type_bounds.bounds().map(|it| TypeBound::from_ast(lower_ctx, it)),
        ))
    } else {
        ThinVec::from_iter([])
    }
}

impl TypeBound {
    pub(crate) fn from_ast(ctx: &mut LowerCtx<'_>, node: ast::TypeBound) -> Self {
        let mut lower_path_type = |path_type: &ast::PathType| ctx.lower_path(path_type.path()?);

        match node.kind() {
            ast::TypeBoundKind::PathType(path_type) => {
                let m = match node.question_mark_token() {
                    Some(_) => TraitBoundModifier::Maybe,
                    None => TraitBoundModifier::None,
                };
                lower_path_type(&path_type)
                    .map(|p| {
                        TypeBound::Path(ctx.alloc_path(p, AstPtr::new(&path_type).upcast()), m)
                    })
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
                let path = for_type.ty().and_then(|ty| match &ty {
                    ast::Type::PathType(path_type) => lower_path_type(path_type).map(|p| (p, ty)),
                    _ => None,
                });
                match path {
                    Some((p, ty)) => {
                        TypeBound::ForLifetime(lt_refs, ctx.alloc_path(p, AstPtr::new(&ty)))
                    }
                    None => TypeBound::Error,
                }
            }
            ast::TypeBoundKind::Use(gal) => TypeBound::Use(
                gal.use_bound_generic_args()
                    .map(|p| match p {
                        ast::UseBoundGenericArg::Lifetime(l) => {
                            UseArgRef::Lifetime(LifetimeRef::new(&l))
                        }
                        ast::UseBoundGenericArg::NameRef(n) => UseArgRef::Name(n.as_name()),
                    })
                    .collect(),
            ),
            ast::TypeBoundKind::Lifetime(lifetime) => {
                TypeBound::Lifetime(LifetimeRef::new(&lifetime))
            }
        }
    }

    pub fn as_path<'a>(&self, map: &'a TypesMap) -> Option<(&'a Path, TraitBoundModifier)> {
        match self {
            &TypeBound::Path(p, m) => Some((&map[p], m)),
            &TypeBound::ForLifetime(_, p) => Some((&map[p], TraitBoundModifier::None)),
            TypeBound::Lifetime(_) | TypeBound::Error | TypeBound::Use(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstRef {
    Scalar(Box<LiteralConstRef>),
    Path(Name),
    Complex(AstId<ast::ConstArg>),
}

impl ConstRef {
    pub(crate) fn from_const_arg(lower_ctx: &LowerCtx<'_>, arg: Option<ast::ConstArg>) -> Self {
        if let Some(arg) = arg {
            if let Some(expr) = arg.expr() {
                return Self::from_expr(expr, Some(lower_ctx.ast_id(&arg)));
            }
        }
        Self::Scalar(Box::new(LiteralConstRef::Unknown))
    }

    pub(crate) fn from_const_param(
        lower_ctx: &LowerCtx<'_>,
        param: &ast::ConstParam,
    ) -> Option<Self> {
        param.default_val().map(|default| Self::from_const_arg(lower_ctx, Some(default)))
    }

    pub fn display<'a>(
        &'a self,
        db: &'a dyn ExpandDatabase,
        edition: Edition,
    ) -> impl fmt::Display + 'a {
        struct Display<'a>(&'a dyn ExpandDatabase, &'a ConstRef, Edition);
        impl fmt::Display for Display<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.1 {
                    ConstRef::Scalar(s) => s.fmt(f),
                    ConstRef::Path(n) => n.display(self.0, self.2).fmt(f),
                    ConstRef::Complex(_) => f.write_str("{const}"),
                }
            }
        }
        Display(db, self, edition)
    }

    // We special case literals and single identifiers, to speed up things.
    fn from_expr(expr: ast::Expr, ast_id: Option<AstId<ast::ConstArg>>) -> Self {
        fn is_path_ident(p: &ast::PathExpr) -> bool {
            let Some(path) = p.path() else {
                return false;
            };
            if path.coloncolon_token().is_some() {
                return false;
            }
            if let Some(s) = path.segment() {
                if s.coloncolon_token().is_some() || s.generic_arg_list().is_some() {
                    return false;
                }
            }
            true
        }
        match expr {
            ast::Expr::PathExpr(p) if is_path_ident(&p) => {
                match p.path().and_then(|it| it.segment()).and_then(|it| it.name_ref()) {
                    Some(it) => Self::Path(it.as_name()),
                    None => Self::Scalar(Box::new(LiteralConstRef::Unknown)),
                }
            }
            ast::Expr::Literal(literal) => Self::Scalar(Box::new(match literal.kind() {
                ast::LiteralKind::IntNumber(num) => {
                    num.value().map(LiteralConstRef::UInt).unwrap_or(LiteralConstRef::Unknown)
                }
                ast::LiteralKind::Char(c) => {
                    c.value().map(LiteralConstRef::Char).unwrap_or(LiteralConstRef::Unknown)
                }
                ast::LiteralKind::Bool(f) => LiteralConstRef::Bool(f),
                _ => LiteralConstRef::Unknown,
            })),
            _ => {
                if let Some(ast_id) = ast_id {
                    Self::Complex(ast_id)
                } else {
                    Self::Scalar(Box::new(LiteralConstRef::Unknown))
                }
            }
        }
    }
}

/// A literal constant value
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LiteralConstRef {
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

impl LiteralConstRef {
    pub fn builtin_type(&self) -> BuiltinType {
        match self {
            LiteralConstRef::UInt(_) | LiteralConstRef::Unknown => {
                BuiltinType::Uint(BuiltinUint::U128)
            }
            LiteralConstRef::Int(_) => BuiltinType::Int(BuiltinInt::I128),
            LiteralConstRef::Char(_) => BuiltinType::Char,
            LiteralConstRef::Bool(_) => BuiltinType::Bool,
        }
    }
}

impl From<Literal> for LiteralConstRef {
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

impl std::fmt::Display for LiteralConstRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            LiteralConstRef::Int(num) => num.fmt(f),
            LiteralConstRef::UInt(num) => num.fmt(f),
            LiteralConstRef::Bool(flag) => flag.fmt(f),
            LiteralConstRef::Char(c) => write!(f, "'{c}'"),
            LiteralConstRef::Unknown => f.write_char('_'),
        }
    }
}
