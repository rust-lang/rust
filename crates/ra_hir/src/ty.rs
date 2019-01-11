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

mod autoderef;
mod primitive;
#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::ops::Index;
use std::sync::Arc;
use std::{fmt, mem};

use log;
use ena::unify::{InPlaceUnificationTable, UnifyKey, UnifyValue, NoError};
use ra_arena::map::ArenaMap;
use join_to_string::join;

use ra_db::Cancelable;

use crate::{
    Def, DefId, Module, Function, Struct, Enum, EnumVariant, Path, Name, ImplBlock,
    FnSignature, FnScopes,
    db::HirDatabase,
    type_ref::{TypeRef, Mutability},
    name::KnownName,
    expr::{Body, Expr, ExprId, PatId, UnaryOp, BinaryOp, Statement},
};

fn transpose<T>(x: Cancelable<Option<T>>) -> Option<Cancelable<T>> {
    match x {
        Ok(Some(t)) => Some(Ok(t)),
        Ok(None) => None,
        Err(e) => Some(Err(e)),
    }
}

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
            (TypeVarValue::Known(t1), TypeVarValue::Known(t2)) => panic!(
                "equating two type variables, both of which have known types: {:?} and {:?}",
                t1, t2
            ),

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
    /// (a non-surrogate code point). Written as `char`.
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

    // A type representing the types stored inside a generator.
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
            if let Some(KnownName::Bool) = name.as_known_name() {
                return Ok(Ty::Bool);
            } else if let Some(KnownName::Char) = name.as_known_name() {
                return Ok(Ty::Char);
            } else if let Some(KnownName::Str) = name.as_known_name() {
                return Ok(Ty::Str);
            } else if let Some(int_ty) = primitive::IntTy::from_name(name) {
                return Ok(Ty::Int(int_ty));
            } else if let Some(uint_ty) = primitive::UintTy::from_name(name) {
                return Ok(Ty::Uint(uint_ty));
            } else if let Some(float_ty) = primitive::FloatTy::from_name(name) {
                return Ok(Ty::Float(float_ty));
            } else if name.as_known_name() == Some(KnownName::SelfType) {
                return Ty::from_hir_opt(db, module, None, impl_block.map(|i| i.target_type()));
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

    fn builtin_deref(&self) -> Option<Ty> {
        match self {
            Ty::Ref(t, _) => Some(Ty::clone(t)),
            Ty::RawPtr(t, _) => Some(Ty::clone(t)),
            _ => None,
        }
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
                if ts.len() == 1 {
                    write!(f, "({},)", ts[0])
                } else {
                    join(ts.iter())
                        .surround_with("(", ")")
                        .separator(", ")
                        .to_fmt(f)
                }
            }
            Ty::FnPtr(sig) => {
                join(sig.input.iter())
                    .surround_with("fn(", ")")
                    .separator(", ")
                    .to_fmt(f)?;
                write!(f, " -> {}", sig.output)
            }
            Ty::Adt { name, .. } => write!(f, "{}", name),
            Ty::Unknown => write!(f, "[unknown]"),
            Ty::Infer(..) => write!(f, "_"),
        }
    }
}

// Functions returning declared types for items

/// Compute the declared type of a function. This should not need to look at the
/// function body.
fn type_for_fn(db: &impl HirDatabase, f: Function) -> Cancelable<Ty> {
    let signature = f.signature(db);
    let module = f.module(db)?;
    let impl_block = f.impl_block(db)?;
    // TODO we ignore type parameters for now
    let input = signature
        .args()
        .iter()
        .map(|tr| Ty::from_hir(db, &module, impl_block.as_ref(), tr))
        .collect::<Cancelable<Vec<_>>>()?;
    let output = Ty::from_hir(db, &module, impl_block.as_ref(), signature.ret_type())?;
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

pub fn type_for_enum_variant(db: &impl HirDatabase, ev: EnumVariant) -> Cancelable<Ty> {
    let enum_parent = ev.parent_enum(db)?;

    type_for_enum(db, enum_parent)
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
        Def::EnumVariant(ev) => type_for_enum_variant(db, ev),
        _ => {
            log::debug!(
                "trying to get type for item of unknown type {:?} {:?}",
                def_id,
                def
            );
            Ok(Ty::Unknown)
        }
    }
}

pub(super) fn type_for_field(
    db: &impl HirDatabase,
    def_id: DefId,
    field: Name,
) -> Cancelable<Option<Ty>> {
    let def = def_id.resolve(db)?;
    let variant_data = match def {
        Def::Struct(s) => s.variant_data(db)?,
        Def::EnumVariant(ev) => ev.variant_data(db)?,
        // TODO: unions
        _ => panic!(
            "trying to get type for field in non-struct/variant {:?}",
            def_id
        ),
    };
    let module = def_id.module(db)?;
    let impl_block = def_id.impl_block(db)?;
    let type_ref = ctry!(variant_data.get_field_type_ref(&field));
    Ok(Some(Ty::from_hir(
        db,
        &module,
        impl_block.as_ref(),
        &type_ref,
    )?))
}

/// The result of type inference: A mapping from expressions and patterns to types.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InferenceResult {
    type_of_expr: ArenaMap<ExprId, Ty>,
    type_of_pat: ArenaMap<PatId, Ty>,
}

impl Index<ExprId> for InferenceResult {
    type Output = Ty;

    fn index(&self, expr: ExprId) -> &Ty {
        self.type_of_expr.get(expr).unwrap_or(&Ty::Unknown)
    }
}

impl Index<PatId> for InferenceResult {
    type Output = Ty;

    fn index(&self, pat: PatId) -> &Ty {
        self.type_of_pat.get(pat).unwrap_or(&Ty::Unknown)
    }
}

/// The inference context contains all information needed during type inference.
#[derive(Clone, Debug)]
struct InferenceContext<'a, D: HirDatabase> {
    db: &'a D,
    body: Arc<Body>,
    scopes: Arc<FnScopes>,
    module: Module,
    impl_block: Option<ImplBlock>,
    var_unification_table: InPlaceUnificationTable<TypeVarId>,
    type_of_expr: ArenaMap<ExprId, Ty>,
    type_of_pat: ArenaMap<PatId, Ty>,
    /// The return type of the function being inferred.
    return_ty: Ty,
}

fn binary_op_return_ty(op: BinaryOp, rhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::BooleanOr
        | BinaryOp::BooleanAnd
        | BinaryOp::EqualityTest
        | BinaryOp::LesserEqualTest
        | BinaryOp::GreaterEqualTest
        | BinaryOp::LesserTest
        | BinaryOp::GreaterTest => Ty::Bool,
        BinaryOp::Assignment
        | BinaryOp::AddAssign
        | BinaryOp::SubAssign
        | BinaryOp::DivAssign
        | BinaryOp::MulAssign
        | BinaryOp::RemAssign
        | BinaryOp::ShrAssign
        | BinaryOp::ShlAssign
        | BinaryOp::BitAndAssign
        | BinaryOp::BitOrAssign
        | BinaryOp::BitXorAssign => Ty::unit(),
        BinaryOp::Addition
        | BinaryOp::Subtraction
        | BinaryOp::Multiplication
        | BinaryOp::Division
        | BinaryOp::Remainder
        | BinaryOp::LeftShift
        | BinaryOp::RightShift
        | BinaryOp::BitwiseAnd
        | BinaryOp::BitwiseOr
        | BinaryOp::BitwiseXor => match rhs_ty {
            Ty::Uint(..) | Ty::Int(..) | Ty::Float(..) => rhs_ty,
            _ => Ty::Unknown,
        },
        BinaryOp::RangeRightOpen | BinaryOp::RangeRightClosed => Ty::Unknown,
    }
}

fn binary_op_rhs_expectation(op: BinaryOp, lhs_ty: Ty) -> Ty {
    match op {
        BinaryOp::BooleanAnd | BinaryOp::BooleanOr => Ty::Bool,
        BinaryOp::Assignment | BinaryOp::EqualityTest => match lhs_ty {
            Ty::Uint(..) | Ty::Int(..) | Ty::Float(..) | Ty::Str | Ty::Char | Ty::Bool => lhs_ty,
            _ => Ty::Unknown,
        },
        BinaryOp::LesserEqualTest
        | BinaryOp::GreaterEqualTest
        | BinaryOp::LesserTest
        | BinaryOp::GreaterTest
        | BinaryOp::AddAssign
        | BinaryOp::SubAssign
        | BinaryOp::DivAssign
        | BinaryOp::MulAssign
        | BinaryOp::RemAssign
        | BinaryOp::ShrAssign
        | BinaryOp::ShlAssign
        | BinaryOp::BitAndAssign
        | BinaryOp::BitOrAssign
        | BinaryOp::BitXorAssign
        | BinaryOp::Addition
        | BinaryOp::Subtraction
        | BinaryOp::Multiplication
        | BinaryOp::Division
        | BinaryOp::Remainder
        | BinaryOp::LeftShift
        | BinaryOp::RightShift
        | BinaryOp::BitwiseAnd
        | BinaryOp::BitwiseOr
        | BinaryOp::BitwiseXor => match lhs_ty {
            Ty::Uint(..) | Ty::Int(..) | Ty::Float(..) => lhs_ty,
            _ => Ty::Unknown,
        },
        _ => Ty::Unknown,
    }
}

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    fn new(
        db: &'a D,
        body: Arc<Body>,
        scopes: Arc<FnScopes>,
        module: Module,
        impl_block: Option<ImplBlock>,
    ) -> Self {
        InferenceContext {
            type_of_expr: ArenaMap::default(),
            type_of_pat: ArenaMap::default(),
            var_unification_table: InPlaceUnificationTable::new(),
            return_ty: Ty::Unknown, // set in collect_fn_signature
            db,
            body,
            scopes,
            module,
            impl_block,
        }
    }

    fn resolve_all(mut self) -> InferenceResult {
        let mut expr_types = mem::replace(&mut self.type_of_expr, ArenaMap::default());
        for ty in expr_types.values_mut() {
            let resolved = self.resolve_ty_completely(mem::replace(ty, Ty::Unknown));
            *ty = resolved;
        }
        let mut pat_types = mem::replace(&mut self.type_of_pat, ArenaMap::default());
        for ty in pat_types.values_mut() {
            let resolved = self.resolve_ty_completely(mem::replace(ty, Ty::Unknown));
            *ty = resolved;
        }
        InferenceResult {
            type_of_expr: expr_types,
            type_of_pat: pat_types,
        }
    }

    fn write_expr_ty(&mut self, expr: ExprId, ty: Ty) {
        self.type_of_expr.insert(expr, ty);
    }

    fn write_pat_ty(&mut self, pat: PatId, ty: Ty) {
        self.type_of_pat.insert(pat, ty);
    }

    fn make_ty(&self, type_ref: &TypeRef) -> Cancelable<Ty> {
        Ty::from_hir(self.db, &self.module, self.impl_block.as_ref(), type_ref)
    }

    fn unify(&mut self, ty1: &Ty, ty2: &Ty) -> bool {
        // try to resolve type vars first
        let ty1 = self.resolve_ty_shallow(ty1);
        let ty2 = self.resolve_ty_shallow(ty2);
        match (&*ty1, &*ty2) {
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
                // both type vars are unknown since we tried to resolve them
                self.var_unification_table.union(*tv1, *tv2);
                true
            }
            (Ty::Infer(InferTy::TypeVar(tv)), other) | (other, Ty::Infer(InferTy::TypeVar(tv))) => {
                // the type var is unknown since we tried to resolve it
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

    /// If `ty` is a type variable with known type, returns that type;
    /// otherwise, return ty.
    fn resolve_ty_shallow<'b>(&mut self, ty: &'b Ty) -> Cow<'b, Ty> {
        match ty {
            Ty::Infer(InferTy::TypeVar(tv)) => {
                match self.var_unification_table.probe_value(*tv).known() {
                    Some(known_ty) => {
                        // The known_ty can't be a type var itself
                        Cow::Owned(known_ty.clone())
                    }
                    _ => Cow::Borrowed(ty),
                }
            }
            _ => Cow::Borrowed(ty),
        }
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

    fn infer_path_expr(&mut self, expr: ExprId, path: &Path) -> Cancelable<Option<Ty>> {
        if path.is_ident() || path.is_self() {
            // resolve locally
            let name = path.as_ident().cloned().unwrap_or_else(Name::self_param);
            if let Some(scope_entry) = self.scopes.resolve_local_name(expr, name) {
                let ty = ctry!(self.type_of_pat.get(scope_entry.pat()));
                let ty = self.resolve_ty_as_possible(ty.clone());
                return Ok(Some(ty));
            };
        };

        // resolve in module
        let resolved = ctry!(self.module.resolve_path(self.db, &path)?.take_values());
        let ty = self.db.type_for_def(resolved)?;
        let ty = self.insert_type_vars(ty);
        Ok(Some(ty))
    }

    fn resolve_variant(&self, path: Option<&Path>) -> Cancelable<(Ty, Option<DefId>)> {
        let path = if let Some(path) = path {
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
            Def::EnumVariant(ev) => {
                let ty = type_for_enum_variant(self.db, ev)?;
                (ty, Some(def_id))
            }
            _ => (Ty::Unknown, None),
        })
    }

    fn infer_expr(&mut self, expr: ExprId, expected: &Expectation) -> Cancelable<Ty> {
        let body = Arc::clone(&self.body); // avoid borrow checker problem
        let ty = match &body[expr] {
            Expr::Missing => Ty::Unknown,
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                // if let is desugared to match, so this is always simple if
                self.infer_expr(*condition, &Expectation::has_type(Ty::Bool))?;
                let then_ty = self.infer_expr(*then_branch, expected)?;
                match else_branch {
                    Some(else_branch) => {
                        self.infer_expr(*else_branch, expected)?;
                    }
                    None => {
                        // no else branch -> unit
                        self.unify(&then_ty, &Ty::unit()); // actually coerce
                    }
                };
                then_ty
            }
            Expr::Block { statements, tail } => self.infer_block(statements, *tail, expected)?,
            Expr::Loop { body } => {
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()))?;
                // TODO handle break with value
                Ty::Never
            }
            Expr::While { condition, body } => {
                // while let is desugared to a match loop, so this is always simple while
                self.infer_expr(*condition, &Expectation::has_type(Ty::Bool))?;
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()))?;
                Ty::unit()
            }
            Expr::For { iterable, body, .. } => {
                let _iterable_ty = self.infer_expr(*iterable, &Expectation::none());
                // TODO write type for pat
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()))?;
                Ty::unit()
            }
            Expr::Lambda { body, .. } => {
                // TODO write types for args, infer lambda type etc.
                let _body_ty = self.infer_expr(*body, &Expectation::none())?;
                Ty::Unknown
            }
            Expr::Call { callee, args } => {
                let callee_ty = self.infer_expr(*callee, &Expectation::none())?;
                let (arg_tys, ret_ty) = match &callee_ty {
                    Ty::FnPtr(sig) => (&sig.input[..], sig.output.clone()),
                    _ => {
                        // not callable
                        // TODO report an error?
                        (&[][..], Ty::Unknown)
                    }
                };
                for (i, arg) in args.iter().enumerate() {
                    self.infer_expr(
                        *arg,
                        &Expectation::has_type(arg_tys.get(i).cloned().unwrap_or(Ty::Unknown)),
                    )?;
                }
                ret_ty
            }
            Expr::MethodCall { receiver, args, .. } => {
                let _receiver_ty = self.infer_expr(*receiver, &Expectation::none())?;
                // TODO resolve method...
                for (_i, arg) in args.iter().enumerate() {
                    // TODO unify / expect argument type
                    self.infer_expr(*arg, &Expectation::none())?;
                }
                Ty::Unknown
            }
            Expr::Match { expr, arms } => {
                let _ty = self.infer_expr(*expr, &Expectation::none())?;
                for arm in arms {
                    // TODO type the bindings in pats
                    // TODO type the guard
                    let _ty = self.infer_expr(arm.expr, &Expectation::none())?;
                }
                // TODO unify all the match arm types
                Ty::Unknown
            }
            Expr::Path(p) => self.infer_path_expr(expr, p)?.unwrap_or(Ty::Unknown),
            Expr::Continue => Ty::Never,
            Expr::Break { expr } => {
                if let Some(expr) = expr {
                    // TODO handle break with value
                    self.infer_expr(*expr, &Expectation::none())?;
                }
                Ty::Never
            }
            Expr::Return { expr } => {
                if let Some(expr) = expr {
                    self.infer_expr(*expr, &Expectation::has_type(self.return_ty.clone()))?;
                }
                Ty::Never
            }
            Expr::StructLit {
                path,
                fields,
                spread,
            } => {
                let (ty, def_id) = self.resolve_variant(path.as_ref())?;
                for field in fields {
                    let field_ty = if let Some(def_id) = def_id {
                        self.db
                            .type_for_field(def_id, field.name.clone())?
                            .unwrap_or(Ty::Unknown)
                    } else {
                        Ty::Unknown
                    };
                    self.infer_expr(field.expr, &Expectation::has_type(field_ty))?;
                }
                if let Some(expr) = spread {
                    self.infer_expr(*expr, &Expectation::has_type(ty.clone()))?;
                }
                ty
            }
            Expr::Field { expr, name } => {
                let receiver_ty = self.infer_expr(*expr, &Expectation::none())?;
                let ty = receiver_ty
                    .autoderef(self.db)
                    .find_map(|derefed_ty| match derefed_ty {
                        // this is more complicated than necessary because type_for_field is cancelable
                        Ty::Tuple(fields) => {
                            let i = name.to_string().parse::<usize>().ok();
                            i.and_then(|i| fields.get(i).cloned()).map(Ok)
                        }
                        Ty::Adt { def_id, .. } => {
                            transpose(self.db.type_for_field(def_id, name.clone()))
                        }
                        _ => None,
                    })
                    .unwrap_or(Ok(Ty::Unknown))?;
                self.insert_type_vars(ty)
            }
            Expr::Try { expr } => {
                let _inner_ty = self.infer_expr(*expr, &Expectation::none())?;
                Ty::Unknown
            }
            Expr::Cast { expr, type_ref } => {
                let _inner_ty = self.infer_expr(*expr, &Expectation::none())?;
                let cast_ty =
                    Ty::from_hir(self.db, &self.module, self.impl_block.as_ref(), type_ref)?;
                let cast_ty = self.insert_type_vars(cast_ty);
                // TODO check the cast...
                cast_ty
            }
            Expr::Ref { expr, mutability } => {
                // TODO pass the expectation down
                let inner_ty = self.infer_expr(*expr, &Expectation::none())?;
                // TODO reference coercions etc.
                Ty::Ref(Arc::new(inner_ty), *mutability)
            }
            Expr::UnaryOp { expr, op } => {
                let inner_ty = self.infer_expr(*expr, &Expectation::none())?;
                match op {
                    Some(UnaryOp::Deref) => {
                        if let Some(derefed_ty) = inner_ty.builtin_deref() {
                            derefed_ty
                        } else {
                            // TODO Deref::deref
                            Ty::Unknown
                        }
                    }
                    _ => Ty::Unknown,
                }
            }
            Expr::BinaryOp { lhs, rhs, op } => match op {
                Some(op) => {
                    let lhs_expectation = match op {
                        BinaryOp::BooleanAnd | BinaryOp::BooleanOr => {
                            Expectation::has_type(Ty::Bool)
                        }
                        _ => Expectation::none(),
                    };
                    let lhs_ty = self.infer_expr(*lhs, &lhs_expectation)?;
                    // TODO: find implementation of trait corresponding to operation
                    // symbol and resolve associated `Output` type
                    let rhs_expectation = binary_op_rhs_expectation(*op, lhs_ty);
                    let rhs_ty = self.infer_expr(*rhs, &Expectation::has_type(rhs_expectation))?;

                    // TODO: similar as above, return ty is often associated trait type
                    binary_op_return_ty(*op, rhs_ty)
                }
                _ => Ty::Unknown,
            },
        };
        // use a new type variable if we got Ty::Unknown here
        let ty = self.insert_type_vars_shallow(ty);
        self.unify(&ty, &expected.ty);
        let ty = self.resolve_ty_as_possible(ty);
        self.write_expr_ty(expr, ty.clone());
        Ok(ty)
    }

    fn infer_block(
        &mut self,
        statements: &[Statement],
        tail: Option<ExprId>,
        expected: &Expectation,
    ) -> Cancelable<Ty> {
        for stmt in statements {
            match stmt {
                Statement::Let {
                    pat,
                    type_ref,
                    initializer,
                } => {
                    let decl_ty = Ty::from_hir_opt(
                        self.db,
                        &self.module,
                        self.impl_block.as_ref(),
                        type_ref.as_ref(),
                    )?;
                    let decl_ty = self.insert_type_vars(decl_ty);
                    let ty = if let Some(expr) = initializer {
                        let expr_ty = self.infer_expr(*expr, &Expectation::has_type(decl_ty))?;
                        expr_ty
                    } else {
                        decl_ty
                    };

                    self.write_pat_ty(*pat, ty);
                }
                Statement::Expr(expr) => {
                    self.infer_expr(*expr, &Expectation::none())?;
                }
            }
        }
        let ty = if let Some(expr) = tail {
            self.infer_expr(expr, expected)?
        } else {
            Ty::unit()
        };
        Ok(ty)
    }

    fn collect_fn_signature(&mut self, signature: &FnSignature) -> Cancelable<()> {
        let body = Arc::clone(&self.body); // avoid borrow checker problem
        for (type_ref, pat) in signature.args().iter().zip(body.args()) {
            let ty = self.make_ty(type_ref)?;
            let ty = self.insert_type_vars(ty);
            self.write_pat_ty(*pat, ty);
        }
        self.return_ty = {
            let ty = self.make_ty(signature.ret_type())?;
            let ty = self.insert_type_vars(ty);
            ty
        };
        Ok(())
    }

    fn infer_body(&mut self) -> Cancelable<()> {
        self.infer_expr(
            self.body.body_expr(),
            &Expectation::has_type(self.return_ty.clone()),
        )?;
        Ok(())
    }
}

pub fn infer(db: &impl HirDatabase, def_id: DefId) -> Cancelable<Arc<InferenceResult>> {
    db.check_canceled()?;
    let function = Function::new(def_id); // TODO: consts also need inference
    let body = function.body(db)?;
    let scopes = db.fn_scopes(def_id)?;
    let module = function.module(db)?;
    let impl_block = function.impl_block(db)?;
    let mut ctx = InferenceContext::new(db, body, scopes, module, impl_block);

    let signature = function.signature(db);
    ctx.collect_fn_signature(&signature)?;

    ctx.infer_body()?;

    Ok(Arc::new(ctx.resolve_all()))
}
