mod primitive;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::fmt;

use rustc_hash::{FxHashMap};

use ra_db::LocalSyntaxPtr;
use ra_syntax::{
    SmolStr,
    ast::{self, AstNode, LoopBodyOwner, ArgListOwner},
    SyntaxNodeRef
};

use crate::{
    FnScopes,
    db::HirDatabase,
};

// pub(crate) type TypeId = Id<Ty>;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
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
    ///
    /// Substs here, possibly against intuition, *may* contain `Param`s.
    /// That is, even after substitution it is possible that there are type
    /// variables. This happens when the `Adt` corresponds to an ADT
    /// definition and not a concrete use of it.
    // Adt(&'tcx AdtDef, &'tcx Substs<'tcx>),

    // Foreign(DefId),

    /// The pointee of a string slice. Written as `str`.
    Str,

    /// An array with the given length. Written as `[T; n]`.
    // Array(Ty<'tcx>, &'tcx ty::Const<'tcx>),

    /// The pointee of an array slice.  Written as `[T]`.
    Slice(TyRef),

    /// A raw pointer. Written as `*mut T` or `*const T`
    // RawPtr(TypeAndMut<'tcx>),

    /// A reference; a pointer with an associated lifetime. Written as
    /// `&'a mut T` or `&'a T`.
    // Ref(Region<'tcx>, Ty<'tcx>, hir::Mutability),

    /// The anonymous type of a function declaration/definition. Each
    /// function has a unique type, which is output (for a function
    /// named `foo` returning an `i32`) as `fn() -> i32 {foo}`.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar = foo; // bar: fn() -> i32 {foo}
    /// ```
    // FnDef(DefId, &'tcx Substs<'tcx>),

    /// A pointer to a function.  Written as `fn() -> i32`.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar: fn() -> i32 = foo;
    /// ```
    // FnPtr(PolyFnSig<'tcx>),

    /// A trait, defined with `trait`.
    // Dynamic(Binder<&'tcx List<ExistentialPredicate<'tcx>>>, ty::Region<'tcx>),

    /// The anonymous type of a closure. Used to represent the type of
    /// `|a| a`.
    // Closure(DefId, ClosureSubsts<'tcx>),

    /// The anonymous type of a generator. Used to represent the type of
    /// `|a| yield a`.
    // Generator(DefId, GeneratorSubsts<'tcx>, hir::GeneratorMovability),

    /// A type representin the types stored inside a generator.
    /// This should only appear in GeneratorInteriors.
    // GeneratorWitness(Binder<&'tcx List<Ty<'tcx>>>),

    /// The never type `!`
    Never,

    /// A tuple type.  For example, `(i32, bool)`.
    Tuple(Vec<Ty>),

    /// The projection of an associated type.  For example,
    /// `<T as Trait<..>>::N`.
    // Projection(ProjectionTy<'tcx>),

    /// Opaque (`impl Trait`) type found in a return type.
    /// The `DefId` comes either from
    /// * the `impl Trait` ast::Ty node,
    /// * or the `existential type` declaration
    /// The substitutions are for the generics of the function in question.
    /// After typeck, the concrete type can be found in the `types` map.
    // Opaque(DefId, &'tcx Substs<'tcx>),

    /// A type parameter; for example, `T` in `fn f<T>(x: T) {}
    // Param(ParamTy),

    /// Bound type variable, used only when preparing a trait query.
    // Bound(ty::DebruijnIndex, BoundTy),

    /// A placeholder type - universally quantified higher-ranked type.
    // Placeholder(ty::PlaceholderType),

    /// A type variable used during type checking.
    // Infer(InferTy),

    /// A placeholder for a type which could not be computed; this is
    /// propagated to avoid useless error messages.
    Unknown,
}

type TyRef = Arc<Ty>;

impl Ty {
    pub fn new(node: ast::TypeRef) -> Self {
        use ra_syntax::ast::TypeRef::*;
        match node {
            ParenType(_inner) => Ty::Unknown, // TODO
            TupleType(_inner) => Ty::Unknown, // TODO
            NeverType(..) => Ty::Never,
            PathType(inner) => {
                let path = if let Some(p) = inner.path() {
                    p
                } else {
                    return Ty::Unknown;
                };
                if path.qualifier().is_none() {
                    let name = path
                        .segment()
                        .and_then(|s| s.name_ref())
                        .map(|n| n.text())
                        .unwrap_or(SmolStr::new(""));
                    if let Some(int_ty) = primitive::IntTy::from_string(&name) {
                        Ty::Int(int_ty)
                    } else if let Some(uint_ty) = primitive::UintTy::from_string(&name) {
                        Ty::Uint(uint_ty)
                    } else if let Some(float_ty) = primitive::FloatTy::from_string(&name) {
                        Ty::Float(float_ty)
                    } else {
                        // TODO
                        Ty::Unknown
                    }
                } else {
                    // TODO
                    Ty::Unknown
                }
            }
            PointerType(_inner) => Ty::Unknown,     // TODO
            ArrayType(_inner) => Ty::Unknown,       // TODO
            SliceType(_inner) => Ty::Unknown,       // TODO
            ReferenceType(_inner) => Ty::Unknown,   // TODO
            PlaceholderType(_inner) => Ty::Unknown, // TODO
            FnPointerType(_inner) => Ty::Unknown,   // TODO
            ForType(_inner) => Ty::Unknown,         // TODO
            ImplTraitType(_inner) => Ty::Unknown,   // TODO
            DynTraitType(_inner) => Ty::Unknown,    // TODO
        }
    }

    pub fn unit() -> Self {
        Ty::Tuple(Vec::new())
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
            Ty::Never => write!(f, "!"),
            Ty::Tuple(ts) => {
                write!(f, "(")?;
                for t in ts {
                    write!(f, "{},", t)?;
                }
                write!(f, ")")
            }
            Ty::Unknown => write!(f, "[unknown]"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InferenceResult {
    type_for: FxHashMap<LocalSyntaxPtr, Ty>,
}

impl InferenceResult {
    pub fn type_of_node(&self, node: SyntaxNodeRef) -> Option<Ty> {
        self.type_for.get(&LocalSyntaxPtr::new(node)).cloned()
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InferenceContext {
    scopes: Arc<FnScopes>,
    // TODO unification tables...
    type_for: FxHashMap<LocalSyntaxPtr, Ty>,
}

impl InferenceContext {
    fn new(scopes: Arc<FnScopes>) -> Self {
        InferenceContext {
            type_for: FxHashMap::default(),
            scopes,
        }
    }

    fn write_ty(&mut self, node: SyntaxNodeRef, ty: Ty) {
        self.type_for.insert(LocalSyntaxPtr::new(node), ty);
    }

    fn unify(&mut self, _ty1: &Ty, _ty2: &Ty) -> bool {
        unimplemented!()
    }

    fn infer_path_expr(&mut self, expr: ast::PathExpr) -> Option<Ty> {
        let p = expr.path()?;
        if p.qualifier().is_none() {
            let name = p.segment().and_then(|s| s.name_ref())?;
            let scope_entry = self.scopes.resolve_local_name(name)?;
            let ty = self.type_for.get(&scope_entry.ptr())?;
            Some(ty.clone())
        } else {
            // TODO resolve path
            Some(Ty::Unknown)
        }
    }

    fn infer_expr(&mut self, expr: ast::Expr) -> Ty {
        let ty = match expr {
            ast::Expr::IfExpr(e) => {
                if let Some(condition) = e.condition() {
                    if let Some(e) = condition.expr() {
                        // TODO if no pat, this should be bool
                        self.infer_expr(e);
                    }
                    // TODO write type for pat
                };
                let if_ty = if let Some(block) = e.then_branch() {
                    self.infer_block(block)
                } else {
                    Ty::Unknown
                };
                let else_ty = if let Some(block) = e.else_branch() {
                    self.infer_block(block)
                } else {
                    Ty::Unknown
                };
                if self.unify(&if_ty, &else_ty) {
                    // TODO actually, need to take the 'more specific' type (not unknown, never, ...)
                    if_ty
                } else {
                    // TODO report diagnostic
                    Ty::Unknown
                }
            }
            ast::Expr::BlockExpr(e) => {
                if let Some(block) = e.block() {
                    self.infer_block(block)
                } else {
                    Ty::Unknown
                }
            }
            ast::Expr::LoopExpr(e) => {
                if let Some(block) = e.loop_body() {
                    self.infer_block(block);
                };
                // TODO never, or the type of the break param
                Ty::Unknown
            }
            ast::Expr::WhileExpr(e) => {
                if let Some(condition) = e.condition() {
                    if let Some(e) = condition.expr() {
                        // TODO if no pat, this should be bool
                        self.infer_expr(e);
                    }
                    // TODO write type for pat
                };
                if let Some(block) = e.loop_body() {
                    // TODO
                    self.infer_block(block);
                };
                // TODO always unit?
                Ty::Unknown
            }
            ast::Expr::ForExpr(e) => {
                if let Some(expr) = e.iterable() {
                    self.infer_expr(expr);
                }
                if let Some(_pat) = e.pat() {
                    // TODO write type for pat
                }
                if let Some(block) = e.loop_body() {
                    self.infer_block(block);
                }
                // TODO always unit?
                Ty::Unknown
            }
            ast::Expr::LambdaExpr(e) => {
                let _body_ty = if let Some(body) = e.body() {
                    self.infer_expr(body)
                } else {
                    Ty::Unknown
                };
                Ty::Unknown
            }
            ast::Expr::CallExpr(e) => {
                if let Some(arg_list) = e.arg_list() {
                    for arg in arg_list.args() {
                        // TODO unify / expect argument type
                        self.infer_expr(arg);
                    }
                }
                Ty::Unknown
            }
            ast::Expr::MethodCallExpr(e) => {
                if let Some(arg_list) = e.arg_list() {
                    for arg in arg_list.args() {
                        // TODO unify / expect argument type
                        self.infer_expr(arg);
                    }
                }
                Ty::Unknown
            }
            ast::Expr::MatchExpr(e) => {
                let _ty = if let Some(match_expr) = e.expr() {
                    self.infer_expr(match_expr)
                } else {
                    Ty::Unknown
                };
                if let Some(match_arm_list) = e.match_arm_list() {
                    for arm in match_arm_list.arms() {
                        // TODO type the bindings in pat
                        // TODO type the guard
                        let _ty = if let Some(e) = arm.expr() {
                            self.infer_expr(e)
                        } else {
                            Ty::Unknown
                        };
                    }
                    // TODO unify all the match arm types
                    Ty::Unknown
                } else {
                    Ty::Unknown
                }
            }
            ast::Expr::TupleExpr(_e) => Ty::Unknown,
            ast::Expr::ArrayExpr(_e) => Ty::Unknown,
            ast::Expr::PathExpr(e) => self.infer_path_expr(e).unwrap_or(Ty::Unknown),
            ast::Expr::ContinueExpr(_e) => Ty::Never,
            ast::Expr::BreakExpr(_e) => Ty::Never,
            ast::Expr::ParenExpr(e) => {
                if let Some(e) = e.expr() {
                    self.infer_expr(e)
                } else {
                    Ty::Unknown
                }
            }
            ast::Expr::Label(_e) => Ty::Unknown,
            ast::Expr::ReturnExpr(e) => {
                if let Some(e) = e.expr() {
                    // TODO unify with return type
                    self.infer_expr(e);
                };
                Ty::Never
            }
            ast::Expr::MatchArmList(_) | ast::Expr::MatchArm(_) | ast::Expr::MatchGuard(_) => {
                // Can this even occur outside of a match expression?
                Ty::Unknown
            }
            ast::Expr::StructLit(_e) => Ty::Unknown,
            ast::Expr::NamedFieldList(_) | ast::Expr::NamedField(_) => {
                // Can this even occur outside of a struct literal?
                Ty::Unknown
            }
            ast::Expr::IndexExpr(_e) => Ty::Unknown,
            ast::Expr::FieldExpr(_e) => Ty::Unknown,
            ast::Expr::TryExpr(e) => {
                let _inner_ty = if let Some(e) = e.expr() {
                    self.infer_expr(e)
                } else {
                    Ty::Unknown
                };
                Ty::Unknown
            }
            ast::Expr::CastExpr(e) => {
                let _inner_ty = if let Some(e) = e.expr() {
                    self.infer_expr(e)
                } else {
                    Ty::Unknown
                };
                let cast_ty = e.type_ref().map(Ty::new).unwrap_or(Ty::Unknown);
                // TODO do the coercion...
                cast_ty
            }
            ast::Expr::RefExpr(e) => {
                let _inner_ty = if let Some(e) = e.expr() {
                    self.infer_expr(e)
                } else {
                    Ty::Unknown
                };
                Ty::Unknown
            }
            ast::Expr::PrefixExpr(e) => {
                let _inner_ty = if let Some(e) = e.expr() {
                    self.infer_expr(e)
                } else {
                    Ty::Unknown
                };
                Ty::Unknown
            }
            ast::Expr::RangeExpr(_e) => Ty::Unknown,
            ast::Expr::BinExpr(_e) => Ty::Unknown,
            ast::Expr::Literal(_e) => Ty::Unknown,
        };
        self.write_ty(expr.syntax(), ty.clone());
        ty
    }

    fn infer_block(&mut self, node: ast::Block) -> Ty {
        for stmt in node.statements() {
            match stmt {
                ast::Stmt::LetStmt(stmt) => {
                    if let Some(expr) = stmt.initializer() {
                        self.infer_expr(expr);
                    }
                }
                ast::Stmt::ExprStmt(expr_stmt) => {
                    if let Some(expr) = expr_stmt.expr() {
                        self.infer_expr(expr);
                    }
                }
            }
        }
        let ty = if let Some(expr) = node.expr() {
            self.infer_expr(expr)
        } else {
            Ty::unit()
        };
        self.write_ty(node.syntax(), ty.clone());
        ty
    }
}

pub fn infer(_db: &impl HirDatabase, node: ast::FnDef, scopes: Arc<FnScopes>) -> InferenceResult {
    let mut ctx = InferenceContext::new(scopes);

    for param in node.param_list().unwrap().params() {
        let pat = param.pat().unwrap();
        let type_ref = param.type_ref().unwrap();
        let ty = Ty::new(type_ref);
        ctx.type_for.insert(LocalSyntaxPtr::new(pat.syntax()), ty);
    }

    // TODO get Ty for node.ret_type() and pass that to infer_block as expectation
    // (see Expectation in rustc_typeck)

    ctx.infer_block(node.body().unwrap());

    // TODO 'resolve' the types: replace inference variables by their inferred results

    InferenceResult {
        type_for: ctx.type_for,
    }
}
