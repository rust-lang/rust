//! The type system. We currently use this to infer types for completion.
//!
//! For type inference, compare the implementations in rustc (the various
//! check_* methods in librustc_typeck/check/mod.rs are a good entry point) and
//! IntelliJ-Rust (org.rust.lang.core.types.infer). Our entry point for
//! inference here is the `infer` function, which infers the types of all
//! expressions in a given function.
//!
//! The central struct here is `Ty`, which represents a type. During inference,
//! it can contain type 'variables' which represent currently unknown types; as
//! we walk through the expressions, we might determine that certain variables
//! need to be equal to each other, or to certain types. To record this, we use
//! the union-find implementation from the `ena` crate, which is extracted from
//! rustc.

mod primitive;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::{fmt, mem};

use log;
use rustc_hash::FxHashMap;
use ena::unify::{InPlaceUnificationTable, UnifyKey, UnifyValue, NoError};

use ra_db::{LocalSyntaxPtr, Cancelable};
use ra_syntax::{
    ast::{self, AstNode, LoopBodyOwner, ArgListOwner, PrefixOp},
    SyntaxNodeRef
};

use crate::{
    Def, DefId, FnScopes, Module, Function, Struct, Enum, Path, Name, AsName, ImplBlock,
    db::HirDatabase,
    type_ref::{TypeRef, Mutability},
    name::KnownName,
};

/// The ID of a type variable.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TypeVarId(u32);

impl UnifyKey for TypeVarId {
    type Value = TypeVarValue;

    fn index(&self) -> u32 {
        self.0
    }

    fn from_index(i: u32) -> Self {
        TypeVarId(i)
    }

    fn tag() -> &'static str {
        "TypeVarId"
    }
}

/// The value of a type variable: either we already know the type, or we don't
/// know it yet.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TypeVarValue {
    Known(Ty),
    Unknown,
}

impl TypeVarValue {
    fn known(&self) -> Option<&Ty> {
        match self {
            TypeVarValue::Known(ty) => Some(ty),
            TypeVarValue::Unknown => None,
        }
    }
}

impl UnifyValue for TypeVarValue {
    type Error = NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, NoError> {
        match (value1, value2) {
            // We should never equate two type variables, both of which have
            // known types. Instead, we recursively equate those types.
            (TypeVarValue::Known(..), TypeVarValue::Known(..)) => {
                panic!("equating two type variables, both of which have known types")
            }

            // If one side is known, prefer that one.
            (TypeVarValue::Known(..), TypeVarValue::Unknown) => Ok(value1.clone()),
            (TypeVarValue::Unknown, TypeVarValue::Known(..)) => Ok(value2.clone()),

            (TypeVarValue::Unknown, TypeVarValue::Unknown) => Ok(TypeVarValue::Unknown),
        }
    }
}

/// The kinds of placeholders we need during type inference. Currently, we only
/// have type variables; in the future, we will probably also need int and float
/// variables, for inference of literal values (e.g. `100` could be one of
/// several integer types).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum InferTy {
    TypeVar(TypeVarId),
}

/// When inferring an expression, we propagate downward whatever type hint we
/// are able in the form of an `Expectation`.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Expectation {
    ty: Ty,
    // TODO: In some cases, we need to be aware whether the expectation is that
    // the type match exactly what we passed, or whether it just needs to be
    // coercible to the expected type. See Expectation::rvalue_hint in rustc.
}

impl Expectation {
    /// The expectation that the type of the expression needs to equal the given
    /// type.
    fn has_type(ty: Ty) -> Self {
        Expectation { ty }
    }

    /// This expresses no expectation on the type.
    fn none() -> Self {
        Expectation { ty: Ty::Unknown }
    }
}

/// A type. This is based on the `TyKind` enum in rustc (librustc/ty/sty.rs).
///
/// This should be cheap to clone.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Ty {
    /// The primitive boolean type. Written as `bool`.
    Bool,

    /// The primitive character type; holds a Unicode scalar value
    /// (a non-surrogate code point).  Written as `char`.
    Char,

    /// A primitive signed integer type. For example, `i32`.
    Int(primitive::IntTy),

    /// A primitive unsigned integer type. For example, `u32`.
    Uint(primitive::UintTy),

    /// A primitive floating-point type. For example, `f64`.
    Float(primitive::FloatTy),

    /// Structures, enumerations and unions.
    Adt {
        /// The DefId of the struct/enum.
        def_id: DefId,
        /// The name, for displaying.
        name: Name,
        // later we'll need generic substitutions here
    },

    /// The pointee of a string slice. Written as `str`.
    Str,

    // An array with the given length. Written as `[T; n]`.
    // Array(Ty, ty::Const),
    /// The pointee of an array slice.  Written as `[T]`.
    Slice(Arc<Ty>),

    /// A raw pointer. Written as `*mut T` or `*const T`
    RawPtr(Arc<Ty>, Mutability),

    /// A reference; a pointer with an associated lifetime. Written as
    /// `&'a mut T` or `&'a T`.
    Ref(Arc<Ty>, Mutability),

    /// A pointer to a function.  Written as `fn() -> i32`.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar: fn() -> i32 = foo;
    /// ```
    FnPtr(Arc<FnSig>),

    // rustc has a separate type for each function, which just coerces to the
    // above function pointer type. Once we implement generics, we will probably
    // need this as well.

    // A trait, defined with `dyn trait`.
    // Dynamic(),
    // The anonymous type of a closure. Used to represent the type of
    // `|a| a`.
    // Closure(DefId, ClosureSubsts<'tcx>),

    // The anonymous type of a generator. Used to represent the type of
    // `|a| yield a`.
    // Generator(DefId, GeneratorSubsts<'tcx>, hir::GeneratorMovability),

    // A type representin the types stored inside a generator.
    // This should only appear in GeneratorInteriors.
    // GeneratorWitness(Binder<&'tcx List<Ty<'tcx>>>),
    /// The never type `!`.
    Never,

    /// A tuple type.  For example, `(i32, bool)`.
    Tuple(Arc<[Ty]>),

    // The projection of an associated type.  For example,
    // `<T as Trait<..>>::N`.pub
    // Projection(ProjectionTy),

    // Opaque (`impl Trait`) type found in a return type.
    // Opaque(DefId, Substs),

    // A type parameter; for example, `T` in `fn f<T>(x: T) {}
    // Param(ParamTy),
    /// A type variable used during type checking. Not to be confused with a
    /// type parameter.
    Infer(InferTy),

    /// A placeholder for a type which could not be computed; this is propagated
    /// to avoid useless error messages. Doubles as a placeholder where type
    /// variables are inserted before type checking, since we want to try to
    /// infer a better type here anyway -- for the IDE use case, we want to try
    /// to infer as much as possible even in the presence of type errors.
    Unknown,
}

/// A function signature.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FnSig {
    input: Vec<Ty>,
    output: Ty,
}

impl Ty {
    pub(crate) fn from_hir(
        db: &impl HirDatabase,
        module: &Module,
        impl_block: Option<&ImplBlock>,
        type_ref: &TypeRef,
    ) -> Cancelable<Self> {
        Ok(match type_ref {
            TypeRef::Never => Ty::Never,
            TypeRef::Tuple(inner) => {
                let inner_tys = inner
                    .iter()
                    .map(|tr| Ty::from_hir(db, module, impl_block, tr))
                    .collect::<Cancelable<Vec<_>>>()?;
                Ty::Tuple(inner_tys.into())
            }
            TypeRef::Path(path) => Ty::from_hir_path(db, module, impl_block, path)?,
            TypeRef::RawPtr(inner, mutability) => {
                let inner_ty = Ty::from_hir(db, module, impl_block, inner)?;
                Ty::RawPtr(Arc::new(inner_ty), *mutability)
            }
            TypeRef::Array(_inner) => Ty::Unknown, // TODO
            TypeRef::Slice(inner) => {
                let inner_ty = Ty::from_hir(db, module, impl_block, inner)?;
                Ty::Slice(Arc::new(inner_ty))
            }
            TypeRef::Reference(inner, mutability) => {
                let inner_ty = Ty::from_hir(db, module, impl_block, inner)?;
                Ty::Ref(Arc::new(inner_ty), *mutability)
            }
            TypeRef::Placeholder => Ty::Unknown,
            TypeRef::Fn(params) => {
                let mut inner_tys = params
                    .iter()
                    .map(|tr| Ty::from_hir(db, module, impl_block, tr))
                    .collect::<Cancelable<Vec<_>>>()?;
                let return_ty = inner_tys
                    .pop()
                    .expect("TypeRef::Fn should always have at least return type");
                let sig = FnSig {
                    input: inner_tys,
                    output: return_ty,
                };
                Ty::FnPtr(Arc::new(sig))
            }
            TypeRef::Error => Ty::Unknown,
        })
    }

    pub(crate) fn from_hir_opt(
        db: &impl HirDatabase,
        module: &Module,
        impl_block: Option<&ImplBlock>,
        type_ref: Option<&TypeRef>,
    ) -> Cancelable<Self> {
        type_ref
            .map(|t| Ty::from_hir(db, module, impl_block, t))
            .unwrap_or(Ok(Ty::Unknown))
    }

    pub(crate) fn from_hir_path(
        db: &impl HirDatabase,
        module: &Module,
        impl_block: Option<&ImplBlock>,
        path: &Path,
    ) -> Cancelable<Self> {
        if let Some(name) = path.as_ident() {
            if let Some(int_ty) = primitive::IntTy::from_name(name) {
                return Ok(Ty::Int(int_ty));
            } else if let Some(uint_ty) = primitive::UintTy::from_name(name) {
                return Ok(Ty::Uint(uint_ty));
            } else if let Some(float_ty) = primitive::FloatTy::from_name(name) {
                return Ok(Ty::Float(float_ty));
            } else if name.as_known_name() == Some(KnownName::Self_) {
                return Ty::from_hir_opt(db, module, None, impl_block.map(|i| i.target()));
            }
        }

        // Resolve in module (in type namespace)
        let resolved = if let Some(r) = module.resolve_path(db, path)?.take_types() {
            r
        } else {
            return Ok(Ty::Unknown);
        };
        let ty = db.type_for_def(resolved)?;
        Ok(ty)
    }

    // TODO: These should not be necessary long-term, since everything will work on HIR
    pub(crate) fn from_ast_opt(
        db: &impl HirDatabase,
        module: &Module,
        impl_block: Option<&ImplBlock>,
        node: Option<ast::TypeRef>,
    ) -> Cancelable<Self> {
        node.map(|n| Ty::from_ast(db, module, impl_block, n))
            .unwrap_or(Ok(Ty::Unknown))
    }

    pub(crate) fn from_ast(
        db: &impl HirDatabase,
        module: &Module,
        impl_block: Option<&ImplBlock>,
        node: ast::TypeRef,
    ) -> Cancelable<Self> {
        Ty::from_hir(db, module, impl_block, &TypeRef::from_ast(node))
    }

    pub fn unit() -> Self {
        Ty::Tuple(Arc::new([]))
    }

    fn walk_mut(&mut self, f: &mut impl FnMut(&mut Ty)) {
        f(self);
        match self {
            Ty::Slice(t) => Arc::make_mut(t).walk_mut(f),
            Ty::RawPtr(t, _) => Arc::make_mut(t).walk_mut(f),
            Ty::Ref(t, _) => Arc::make_mut(t).walk_mut(f),
            Ty::Tuple(ts) => {
                // Without an Arc::make_mut_slice, we can't avoid the clone here:
                let mut v: Vec<_> = ts.iter().cloned().collect();
                for t in &mut v {
                    t.walk_mut(f);
                }
                *ts = v.into();
            }
            Ty::FnPtr(sig) => {
                let sig_mut = Arc::make_mut(sig);
                for input in &mut sig_mut.input {
                    input.walk_mut(f);
                }
                sig_mut.output.walk_mut(f);
            }
            Ty::Adt { .. } => {} // need to walk type parameters later
            _ => {}
        }
    }

    fn fold(mut self, f: &mut impl FnMut(Ty) -> Ty) -> Ty {
        self.walk_mut(&mut |ty_mut| {
            let ty = mem::replace(ty_mut, Ty::Unknown);
            *ty_mut = f(ty);
        });
        self
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Ty::Bool => write!(f, "bool"),
            Ty::Char => write!(f, "char"),
            Ty::Int(t) => write!(f, "{}", t.ty_to_string()),
            Ty::Uint(t) => write!(f, "{}", t.ty_to_string()),
            Ty::Float(t) => write!(f, "{}", t.ty_to_string()),
            Ty::Str => write!(f, "str"),
            Ty::Slice(t) => write!(f, "[{}]", t),
            Ty::RawPtr(t, m) => write!(f, "*{}{}", m.as_keyword_for_ptr(), t),
            Ty::Ref(t, m) => write!(f, "&{}{}", m.as_keyword_for_ref(), t),
            Ty::Never => write!(f, "!"),
            Ty::Tuple(ts) => {
                write!(f, "(")?;
                for t in ts.iter() {
                    write!(f, "{},", t)?;
                }
                write!(f, ")")
            }
            Ty::FnPtr(sig) => {
                write!(f, "fn(")?;
                for t in &sig.input {
                    write!(f, "{},", t)?;
                }
                write!(f, ") -> {}", sig.output)
            }
            Ty::Adt { name, .. } => write!(f, "{}", name),
            Ty::Unknown => write!(f, "[unknown]"),
            Ty::Infer(..) => write!(f, "_"),
        }
    }
}

// Functions returning declared types for items

/// Compute the declared type of a function. This should not need to look at the
/// function body (but currently uses the function AST, so does anyway - TODO).
fn type_for_fn(db: &impl HirDatabase, f: Function) -> Cancelable<Ty> {
    let syntax = f.syntax(db);
    let module = f.module(db)?;
    let impl_block = f.impl_block(db)?;
    let node = syntax.borrowed();
    // TODO we ignore type parameters for now
    let input = node
        .param_list()
        .map(|pl| {
            pl.params()
                .map(|p| Ty::from_ast_opt(db, &module, impl_block.as_ref(), p.type_ref()))
                .collect()
        })
        .unwrap_or_else(|| Ok(Vec::new()))?;
    let output = if let Some(type_ref) = node.ret_type().and_then(|rt| rt.type_ref()) {
        Ty::from_ast(db, &module, impl_block.as_ref(), type_ref)?
    } else {
        Ty::unit()
    };
    let sig = FnSig { input, output };
    Ok(Ty::FnPtr(Arc::new(sig)))
}

fn type_for_struct(db: &impl HirDatabase, s: Struct) -> Cancelable<Ty> {
    Ok(Ty::Adt {
        def_id: s.def_id(),
        name: s.name(db)?.unwrap_or_else(Name::missing),
    })
}

pub fn type_for_enum(db: &impl HirDatabase, s: Enum) -> Cancelable<Ty> {
    Ok(Ty::Adt {
        def_id: s.def_id(),
        name: s.name(db)?.unwrap_or_else(Name::missing),
    })
}

pub(super) fn type_for_def(db: &impl HirDatabase, def_id: DefId) -> Cancelable<Ty> {
    let def = def_id.resolve(db)?;
    match def {
        Def::Module(..) => {
            log::debug!("trying to get type for module {:?}", def_id);
            Ok(Ty::Unknown)
        }
        Def::Function(f) => type_for_fn(db, f),
        Def::Struct(s) => type_for_struct(db, s),
        Def::Enum(e) => type_for_enum(db, e),
        Def::Item => {
            log::debug!("trying to get type for item of unknown type {:?}", def_id);
            Ok(Ty::Unknown)
        }
    }
}

pub(super) fn type_for_field(db: &impl HirDatabase, def_id: DefId, field: Name) -> Cancelable<Ty> {
    let def = def_id.resolve(db)?;
    let variant_data = match def {
        Def::Struct(s) => {
            let variant_data = s.variant_data(db)?;
            variant_data
        }
        // TODO: unions
        // TODO: enum variants
        _ => panic!(
            "trying to get type for field in non-struct/variant {:?}",
            def_id
        ),
    };
    let module = def_id.module(db)?;
    let impl_block = def_id.impl_block(db)?;
    let type_ref = if let Some(tr) = variant_data.get_field_type_ref(&field) {
        tr
    } else {
        return Ok(Ty::Unknown);
    };
    Ty::from_hir(db, &module, impl_block.as_ref(), &type_ref)
}

/// The result of type inference: A mapping from expressions and patterns to types.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InferenceResult {
    type_of: FxHashMap<LocalSyntaxPtr, Ty>,
}

impl InferenceResult {
    /// Returns the type of the given syntax node, if it was inferred. Will
    /// return `None` for syntax nodes not in the inferred function or not
    /// pointing to an expression/pattern, `Some(Ty::Unknown)` for
    /// expressions/patterns that could not be inferred.
    pub fn type_of_node(&self, node: SyntaxNodeRef) -> Option<Ty> {
        self.type_of.get(&LocalSyntaxPtr::new(node)).cloned()
    }
}

/// The inference context contains all information needed during type inference.
#[derive(Clone, Debug)]
struct InferenceContext<'a, D: HirDatabase> {
    db: &'a D,
    scopes: Arc<FnScopes>,
    /// The self param for the current method, if it exists.
    self_param: Option<LocalSyntaxPtr>,
    module: Module,
    impl_block: Option<ImplBlock>,
    var_unification_table: InPlaceUnificationTable<TypeVarId>,
    type_of: FxHashMap<LocalSyntaxPtr, Ty>,
}

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    fn new(
        db: &'a D,
        scopes: Arc<FnScopes>,
        module: Module,
        impl_block: Option<ImplBlock>,
    ) -> Self {
        InferenceContext {
            type_of: FxHashMap::default(),
            var_unification_table: InPlaceUnificationTable::new(),
            self_param: None, // set during parameter typing
            db,
            scopes,
            module,
            impl_block,
        }
    }

    fn resolve_all(mut self) -> InferenceResult {
        let mut types = mem::replace(&mut self.type_of, FxHashMap::default());
        for ty in types.values_mut() {
            let resolved = self.resolve_ty_completely(mem::replace(ty, Ty::Unknown));
            *ty = resolved;
        }
        InferenceResult { type_of: types }
    }

    fn write_ty(&mut self, node: SyntaxNodeRef, ty: Ty) {
        self.type_of.insert(LocalSyntaxPtr::new(node), ty);
    }

    fn unify(&mut self, ty1: &Ty, ty2: &Ty) -> bool {
        match (ty1, ty2) {
            (Ty::Unknown, ..) => true,
            (.., Ty::Unknown) => true,
            (Ty::Bool, _)
            | (Ty::Str, _)
            | (Ty::Never, _)
            | (Ty::Char, _)
            | (Ty::Int(..), Ty::Int(..))
            | (Ty::Uint(..), Ty::Uint(..))
            | (Ty::Float(..), Ty::Float(..)) => ty1 == ty2,
            (
                Ty::Adt {
                    def_id: def_id1, ..
                },
                Ty::Adt {
                    def_id: def_id2, ..
                },
            ) if def_id1 == def_id2 => true,
            (Ty::Slice(t1), Ty::Slice(t2)) => self.unify(t1, t2),
            (Ty::RawPtr(t1, m1), Ty::RawPtr(t2, m2)) if m1 == m2 => self.unify(t1, t2),
            (Ty::Ref(t1, m1), Ty::Ref(t2, m2)) if m1 == m2 => self.unify(t1, t2),
            (Ty::FnPtr(sig1), Ty::FnPtr(sig2)) if sig1 == sig2 => true,
            (Ty::Tuple(ts1), Ty::Tuple(ts2)) if ts1.len() == ts2.len() => ts1
                .iter()
                .zip(ts2.iter())
                .all(|(t1, t2)| self.unify(t1, t2)),
            (Ty::Infer(InferTy::TypeVar(tv1)), Ty::Infer(InferTy::TypeVar(tv2))) => {
                self.var_unification_table.union(*tv1, *tv2);
                true
            }
            (Ty::Infer(InferTy::TypeVar(tv)), other) | (other, Ty::Infer(InferTy::TypeVar(tv))) => {
                self.var_unification_table
                    .union_value(*tv, TypeVarValue::Known(other.clone()));
                true
            }
            _ => false,
        }
    }

    fn new_type_var(&mut self) -> Ty {
        Ty::Infer(InferTy::TypeVar(
            self.var_unification_table.new_key(TypeVarValue::Unknown),
        ))
    }

    /// Replaces Ty::Unknown by a new type var, so we can maybe still infer it.
    fn insert_type_vars_shallow(&mut self, ty: Ty) -> Ty {
        match ty {
            Ty::Unknown => self.new_type_var(),
            _ => ty,
        }
    }

    fn insert_type_vars(&mut self, ty: Ty) -> Ty {
        ty.fold(&mut |ty| self.insert_type_vars_shallow(ty))
    }

    /// Resolves the type as far as currently possible, replacing type variables
    /// by their known types. All types returned by the infer_* functions should
    /// be resolved as far as possible, i.e. contain no type variables with
    /// known type.
    fn resolve_ty_as_possible(&mut self, ty: Ty) -> Ty {
        ty.fold(&mut |ty| match ty {
            Ty::Infer(InferTy::TypeVar(tv)) => {
                if let Some(known_ty) = self.var_unification_table.probe_value(tv).known() {
                    // known_ty may contain other variables that are known by now
                    self.resolve_ty_as_possible(known_ty.clone())
                } else {
                    Ty::Infer(InferTy::TypeVar(tv))
                }
            }
            _ => ty,
        })
    }

    /// Resolves the type completely; type variables without known type are
    /// replaced by Ty::Unknown.
    fn resolve_ty_completely(&mut self, ty: Ty) -> Ty {
        ty.fold(&mut |ty| match ty {
            Ty::Infer(InferTy::TypeVar(tv)) => {
                if let Some(known_ty) = self.var_unification_table.probe_value(tv).known() {
                    // known_ty may contain other variables that are known by now
                    self.resolve_ty_completely(known_ty.clone())
                } else {
                    Ty::Unknown
                }
            }
            _ => ty,
        })
    }

    fn infer_path_expr(&mut self, expr: ast::PathExpr) -> Cancelable<Option<Ty>> {
        let ast_path = ctry!(expr.path());
        let path = ctry!(Path::from_ast(ast_path));
        if path.is_ident() {
            // resolve locally
            let name = ctry!(ast_path.segment().and_then(|s| s.name_ref()));
            if let Some(scope_entry) = self.scopes.resolve_local_name(name) {
                let ty = ctry!(self.type_of.get(&scope_entry.ptr()));
                let ty = self.resolve_ty_as_possible(ty.clone());
                return Ok(Some(ty));
            };
        } else if path.is_self() {
            // resolve `self` param
            let self_param = ctry!(self.self_param);
            let ty = ctry!(self.type_of.get(&self_param));
            let ty = self.resolve_ty_as_possible(ty.clone());
            return Ok(Some(ty));
        };

        // resolve in module
        let resolved = ctry!(self.module.resolve_path(self.db, &path)?.take_values());
        let ty = self.db.type_for_def(resolved)?;
        let ty = self.insert_type_vars(ty);
        Ok(Some(ty))
    }

    fn resolve_variant(&self, path: Option<ast::Path>) -> Cancelable<(Ty, Option<DefId>)> {
        let path = if let Some(path) = path.and_then(Path::from_ast) {
            path
        } else {
            return Ok((Ty::Unknown, None));
        };
        let def_id = if let Some(def_id) = self.module.resolve_path(self.db, &path)?.take_types() {
            def_id
        } else {
            return Ok((Ty::Unknown, None));
        };
        Ok(match def_id.resolve(self.db)? {
            Def::Struct(s) => {
                let ty = type_for_struct(self.db, s)?;
                (ty, Some(def_id))
            }
            _ => (Ty::Unknown, None),
        })
    }

    fn infer_expr_opt(
        &mut self,
        expr: Option<ast::Expr>,
        expected: &Expectation,
    ) -> Cancelable<Ty> {
        if let Some(e) = expr {
            self.infer_expr(e, expected)
        } else {
            Ok(Ty::Unknown)
        }
    }

    fn infer_expr(&mut self, expr: ast::Expr, expected: &Expectation) -> Cancelable<Ty> {
        let ty = match expr {
            ast::Expr::IfExpr(e) => {
                if let Some(condition) = e.condition() {
                    let expected = if condition.pat().is_none() {
                        Expectation::has_type(Ty::Bool)
                    } else {
                        Expectation::none()
                    };
                    self.infer_expr_opt(condition.expr(), &expected)?;
                    // TODO write type for pat
                };
                let if_ty = self.infer_block_opt(e.then_branch(), expected)?;
                if let Some(else_branch) = e.else_branch() {
                    self.infer_block(else_branch, expected)?;
                } else {
                    // no else branch -> unit
                    self.unify(&expected.ty, &Ty::unit()); // actually coerce
                }
                if_ty
            }
            ast::Expr::BlockExpr(e) => self.infer_block_opt(e.block(), expected)?,
            ast::Expr::LoopExpr(e) => {
                self.infer_block_opt(e.loop_body(), &Expectation::has_type(Ty::unit()))?;
                // TODO never, or the type of the break param
                Ty::Unknown
            }
            ast::Expr::WhileExpr(e) => {
                if let Some(condition) = e.condition() {
                    let expected = if condition.pat().is_none() {
                        Expectation::has_type(Ty::Bool)
                    } else {
                        Expectation::none()
                    };
                    self.infer_expr_opt(condition.expr(), &expected)?;
                    // TODO write type for pat
                };
                self.infer_block_opt(e.loop_body(), &Expectation::has_type(Ty::unit()))?;
                // TODO always unit?
                Ty::unit()
            }
            ast::Expr::ForExpr(e) => {
                let _iterable_ty = self.infer_expr_opt(e.iterable(), &Expectation::none());
                if let Some(_pat) = e.pat() {
                    // TODO write type for pat
                }
                self.infer_block_opt(e.loop_body(), &Expectation::has_type(Ty::unit()))?;
                // TODO always unit?
                Ty::unit()
            }
            ast::Expr::LambdaExpr(e) => {
                let _body_ty = self.infer_expr_opt(e.body(), &Expectation::none())?;
                Ty::Unknown
            }
            ast::Expr::CallExpr(e) => {
                let callee_ty = self.infer_expr_opt(e.expr(), &Expectation::none())?;
                let (arg_tys, ret_ty) = match &callee_ty {
                    Ty::FnPtr(sig) => (&sig.input[..], sig.output.clone()),
                    _ => {
                        // not callable
                        // TODO report an error?
                        (&[][..], Ty::Unknown)
                    }
                };
                if let Some(arg_list) = e.arg_list() {
                    for (i, arg) in arg_list.args().enumerate() {
                        self.infer_expr(
                            arg,
                            &Expectation::has_type(arg_tys.get(i).cloned().unwrap_or(Ty::Unknown)),
                        )?;
                    }
                }
                ret_ty
            }
            ast::Expr::MethodCallExpr(e) => {
                let _receiver_ty = self.infer_expr_opt(e.expr(), &Expectation::none())?;
                if let Some(arg_list) = e.arg_list() {
                    for arg in arg_list.args() {
                        // TODO unify / expect argument type
                        self.infer_expr(arg, &Expectation::none())?;
                    }
                }
                Ty::Unknown
            }
            ast::Expr::MatchExpr(e) => {
                let _ty = self.infer_expr_opt(e.expr(), &Expectation::none())?;
                if let Some(match_arm_list) = e.match_arm_list() {
                    for arm in match_arm_list.arms() {
                        // TODO type the bindings in pat
                        // TODO type the guard
                        let _ty = self.infer_expr_opt(arm.expr(), &Expectation::none())?;
                    }
                    // TODO unify all the match arm types
                    Ty::Unknown
                } else {
                    Ty::Unknown
                }
            }
            ast::Expr::TupleExpr(_e) => Ty::Unknown,
            ast::Expr::ArrayExpr(_e) => Ty::Unknown,
            ast::Expr::PathExpr(e) => self.infer_path_expr(e)?.unwrap_or(Ty::Unknown),
            ast::Expr::ContinueExpr(_e) => Ty::Never,
            ast::Expr::BreakExpr(_e) => Ty::Never,
            ast::Expr::ParenExpr(e) => self.infer_expr_opt(e.expr(), expected)?,
            ast::Expr::Label(_e) => Ty::Unknown,
            ast::Expr::ReturnExpr(e) => {
                // TODO expect return type of function
                self.infer_expr_opt(e.expr(), &Expectation::none())?;
                Ty::Never
            }
            ast::Expr::MatchArmList(_) | ast::Expr::MatchArm(_) | ast::Expr::MatchGuard(_) => {
                // Can this even occur outside of a match expression?
                Ty::Unknown
            }
            ast::Expr::StructLit(e) => {
                let (ty, def_id) = self.resolve_variant(e.path())?;
                if let Some(nfl) = e.named_field_list() {
                    for field in nfl.fields() {
                        let field_ty = if let (Some(def_id), Some(nr)) = (def_id, field.name_ref())
                        {
                            self.db.type_for_field(def_id, nr.as_name())?
                        } else {
                            Ty::Unknown
                        };
                        self.infer_expr_opt(field.expr(), &Expectation::has_type(field_ty))?;
                    }
                }
                ty
            }
            ast::Expr::NamedFieldList(_) | ast::Expr::NamedField(_) => {
                // Can this even occur outside of a struct literal?
                Ty::Unknown
            }
            ast::Expr::IndexExpr(_e) => Ty::Unknown,
            ast::Expr::FieldExpr(e) => {
                let receiver_ty = self.infer_expr_opt(e.expr(), &Expectation::none())?;
                if let Some(nr) = e.name_ref() {
                    let ty = match receiver_ty {
                        Ty::Tuple(fields) => {
                            let i = nr.text().parse::<usize>().ok();
                            i.and_then(|i| fields.get(i).cloned())
                                .unwrap_or(Ty::Unknown)
                        }
                        Ty::Adt { def_id, .. } => self.db.type_for_field(def_id, nr.as_name())?,
                        _ => Ty::Unknown,
                    };
                    self.insert_type_vars(ty)
                } else {
                    Ty::Unknown
                }
            }
            ast::Expr::TryExpr(e) => {
                let _inner_ty = self.infer_expr_opt(e.expr(), &Expectation::none())?;
                Ty::Unknown
            }
            ast::Expr::CastExpr(e) => {
                let _inner_ty = self.infer_expr_opt(e.expr(), &Expectation::none())?;
                let cast_ty = Ty::from_ast_opt(
                    self.db,
                    &self.module,
                    self.impl_block.as_ref(),
                    e.type_ref(),
                )?;
                let cast_ty = self.insert_type_vars(cast_ty);
                // TODO do the coercion...
                cast_ty
            }
            ast::Expr::RefExpr(e) => {
                // TODO pass the expectation down
                let inner_ty = self.infer_expr_opt(e.expr(), &Expectation::none())?;
                let m = Mutability::from_mutable(e.is_mut());
                // TODO reference coercions etc.
                Ty::Ref(Arc::new(inner_ty), m)
            }
            ast::Expr::PrefixExpr(e) => {
                let inner_ty = self.infer_expr_opt(e.expr(), &Expectation::none())?;
                match e.op() {
                    Some(PrefixOp::Deref) => {
                        match inner_ty {
                            // builtin deref:
                            Ty::Ref(ref_inner, _) => (*ref_inner).clone(),
                            Ty::RawPtr(ptr_inner, _) => (*ptr_inner).clone(),
                            // TODO Deref::deref
                            _ => Ty::Unknown,
                        }
                    }
                    _ => Ty::Unknown,
                }
            }
            ast::Expr::RangeExpr(_e) => Ty::Unknown,
            ast::Expr::BinExpr(_e) => Ty::Unknown,
            ast::Expr::Literal(_e) => Ty::Unknown,
        };
        // use a new type variable if we got Ty::Unknown here
        let ty = self.insert_type_vars_shallow(ty);
        self.unify(&ty, &expected.ty);
        self.write_ty(expr.syntax(), ty.clone());
        Ok(ty)
    }

    fn infer_block_opt(
        &mut self,
        node: Option<ast::Block>,
        expected: &Expectation,
    ) -> Cancelable<Ty> {
        if let Some(b) = node {
            self.infer_block(b, expected)
        } else {
            Ok(Ty::Unknown)
        }
    }

    fn infer_block(&mut self, node: ast::Block, expected: &Expectation) -> Cancelable<Ty> {
        for stmt in node.statements() {
            match stmt {
                ast::Stmt::LetStmt(stmt) => {
                    let decl_ty = Ty::from_ast_opt(
                        self.db,
                        &self.module,
                        self.impl_block.as_ref(),
                        stmt.type_ref(),
                    )?;
                    let decl_ty = self.insert_type_vars(decl_ty);
                    let ty = if let Some(expr) = stmt.initializer() {
                        let expr_ty = self.infer_expr(expr, &Expectation::has_type(decl_ty))?;
                        expr_ty
                    } else {
                        decl_ty
                    };

                    if let Some(pat) = stmt.pat() {
                        self.write_ty(pat.syntax(), ty);
                    };
                }
                ast::Stmt::ExprStmt(expr_stmt) => {
                    self.infer_expr_opt(expr_stmt.expr(), &Expectation::none())?;
                }
            }
        }
        let ty = if let Some(expr) = node.expr() {
            self.infer_expr(expr, expected)?
        } else {
            Ty::unit()
        };
        self.write_ty(node.syntax(), ty.clone());
        Ok(ty)
    }
}

pub fn infer(db: &impl HirDatabase, def_id: DefId) -> Cancelable<Arc<InferenceResult>> {
    let function = Function::new(def_id); // TODO: consts also need inference
    let scopes = function.scopes(db);
    let module = function.module(db)?;
    let impl_block = function.impl_block(db)?;
    let mut ctx = InferenceContext::new(db, scopes, module, impl_block);

    let syntax = function.syntax(db);
    let node = syntax.borrowed();

    if let Some(param_list) = node.param_list() {
        if let Some(self_param) = param_list.self_param() {
            let self_type = if let Some(impl_block) = &ctx.impl_block {
                if let Some(type_ref) = self_param.type_ref() {
                    let ty = Ty::from_ast(db, &ctx.module, ctx.impl_block.as_ref(), type_ref)?;
                    ctx.insert_type_vars(ty)
                } else {
                    // TODO this should be handled by desugaring during HIR conversion
                    let ty = Ty::from_hir(
                        db,
                        &ctx.module,
                        ctx.impl_block.as_ref(),
                        impl_block.target(),
                    )?;
                    let ty = match self_param.flavor() {
                        ast::SelfParamFlavor::Owned => ty,
                        ast::SelfParamFlavor::Ref => Ty::Ref(Arc::new(ty), Mutability::Shared),
                        ast::SelfParamFlavor::MutRef => Ty::Ref(Arc::new(ty), Mutability::Mut),
                    };
                    ctx.insert_type_vars(ty)
                }
            } else {
                log::debug!(
                    "No impl block found, but self param for function {:?}",
                    def_id
                );
                ctx.new_type_var()
            };
            if let Some(self_kw) = self_param.self_kw() {
                let self_param = LocalSyntaxPtr::new(self_kw.syntax());
                ctx.self_param = Some(self_param);
                ctx.type_of.insert(self_param, self_type);
            }
        }
        for param in param_list.params() {
            let pat = if let Some(pat) = param.pat() {
                pat
            } else {
                continue;
            };
            let ty = if let Some(type_ref) = param.type_ref() {
                let ty = Ty::from_ast(db, &ctx.module, ctx.impl_block.as_ref(), type_ref)?;
                ctx.insert_type_vars(ty)
            } else {
                // missing type annotation
                ctx.new_type_var()
            };
            ctx.type_of.insert(LocalSyntaxPtr::new(pat.syntax()), ty);
        }
    }

    let ret_ty = if let Some(type_ref) = node.ret_type().and_then(|n| n.type_ref()) {
        let ty = Ty::from_ast(db, &ctx.module, ctx.impl_block.as_ref(), type_ref)?;
        ctx.insert_type_vars(ty)
    } else {
        Ty::unit()
    };

    if let Some(block) = node.body() {
        ctx.infer_block(block, &Expectation::has_type(ret_ty))?;
    }

    Ok(Arc::new(ctx.resolve_all()))
}
