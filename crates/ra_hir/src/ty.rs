mod primitive;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::fmt;

use log;
use rustc_hash::{FxHashMap};

use ra_db::{LocalSyntaxPtr, Cancelable};
use ra_syntax::{
    SmolStr,
    ast::{self, AstNode, LoopBodyOwner, ArgListOwner},
    SyntaxNodeRef
};

use crate::{Def, DefId, FnScopes, Module, Function, Path, db::HirDatabase};

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

    /// A pointer to a function.  Written as `fn() -> i32`.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar: fn() -> i32 = foo;
    /// ```
    FnPtr(Arc<FnSig>),

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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct FnSig {
    input: Vec<Ty>,
    output: Ty,
}

impl Ty {
    pub fn new(_db: &impl HirDatabase, node: ast::TypeRef) -> Cancelable<Self> {
        use ra_syntax::ast::TypeRef::*;
        Ok(match node {
            ParenType(_inner) => Ty::Unknown, // TODO
            TupleType(_inner) => Ty::Unknown, // TODO
            NeverType(..) => Ty::Never,
            PathType(inner) => {
                let path = if let Some(p) = inner.path() {
                    p
                } else {
                    return Ok(Ty::Unknown);
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
        })
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
            Ty::FnPtr(sig) => {
                write!(f, "fn(")?;
                for t in &sig.input {
                    write!(f, "{},", t)?;
                }
                write!(f, ") -> {}", sig.output)
            }
            Ty::Unknown => write!(f, "[unknown]"),
        }
    }
}

pub fn type_for_fn(db: &impl HirDatabase, f: Function) -> Cancelable<Ty> {
    let syntax = f.syntax(db);
    let node = syntax.borrowed();
    // TODO we ignore type parameters for now
    let input = node
        .param_list()
        .map(|pl| {
            pl.params()
                .map(|p| {
                    p.type_ref()
                        .map(|t| Ty::new(db, t))
                        .unwrap_or(Ok(Ty::Unknown))
                })
                .collect()
        })
        .unwrap_or_else(|| Ok(Vec::new()))?;
    let output = node
        .ret_type()
        .and_then(|rt| rt.type_ref())
        .map(|t| Ty::new(db, t))
        .unwrap_or(Ok(Ty::Unknown))?;
    let sig = FnSig { input, output };
    Ok(Ty::FnPtr(Arc::new(sig)))
}

// TODO this should probably be per namespace (i.e. types vs. values), since for
// a tuple struct `struct Foo(Bar)`, Foo has function type as a value, but
// defines the struct type Foo when used in the type namespace. rustc has a
// separate DefId for the constructor, but with the current DefId approach, that
// seems complicated.
pub fn type_for_def(db: &impl HirDatabase, def_id: DefId) -> Cancelable<Ty> {
    let def = def_id.resolve(db)?;
    match def {
        Def::Module(..) => {
            log::debug!("trying to get type for module {:?}", def_id);
            Ok(Ty::Unknown)
        }
        Def::Function(f) => type_for_fn(db, f),
        Def::Item => {
            log::debug!("trying to get type for item of unknown type {:?}", def_id);
            Ok(Ty::Unknown)
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

#[derive(Clone, Debug)]
pub struct InferenceContext<'a, D: HirDatabase> {
    db: &'a D,
    scopes: Arc<FnScopes>,
    module: Module,
    // TODO unification tables...
    type_for: FxHashMap<LocalSyntaxPtr, Ty>,
}

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    fn new(db: &'a D, scopes: Arc<FnScopes>, module: Module) -> Self {
        InferenceContext {
            type_for: FxHashMap::default(),
            db,
            scopes,
            module,
        }
    }

    fn write_ty(&mut self, node: SyntaxNodeRef, ty: Ty) {
        self.type_for.insert(LocalSyntaxPtr::new(node), ty);
    }

    fn unify(&mut self, ty1: &Ty, ty2: &Ty) -> Option<Ty> {
        if *ty1 == Ty::Unknown {
            return Some(ty2.clone());
        }
        if *ty2 == Ty::Unknown {
            return Some(ty1.clone());
        }
        if ty1 == ty2 {
            return Some(ty1.clone());
        }
        // TODO implement actual unification
        return None;
    }

    fn unify_with_coercion(&mut self, ty1: &Ty, ty2: &Ty) -> Option<Ty> {
        // TODO implement coercion
        self.unify(ty1, ty2)
    }

    fn infer_path_expr(&mut self, expr: ast::PathExpr) -> Cancelable<Option<Ty>> {
        let ast_path = ctry!(expr.path());
        let path = ctry!(Path::from_ast(ast_path));
        if path.is_ident() {
            // resolve locally
            let name = ctry!(ast_path.segment().and_then(|s| s.name_ref()));
            if let Some(scope_entry) = self.scopes.resolve_local_name(name) {
                let ty = ctry!(self.type_for.get(&scope_entry.ptr()));
                return Ok(Some(ty.clone()));
            };
        };

        // resolve in module
        let resolved = ctry!(self.module.resolve_path(self.db, path)?);
        let ty = self.db.type_for_def(resolved)?;
        // TODO we will need to add type variables for type parameters etc. here
        Ok(Some(ty))
    }

    fn infer_expr(&mut self, expr: ast::Expr) -> Cancelable<Ty> {
        let ty = match expr {
            ast::Expr::IfExpr(e) => {
                if let Some(condition) = e.condition() {
                    if let Some(e) = condition.expr() {
                        // TODO if no pat, this should be bool
                        self.infer_expr(e)?;
                    }
                    // TODO write type for pat
                };
                let if_ty = if let Some(block) = e.then_branch() {
                    self.infer_block(block)?
                } else {
                    Ty::Unknown
                };
                let else_ty = if let Some(block) = e.else_branch() {
                    self.infer_block(block)?
                } else {
                    Ty::Unknown
                };
                if let Some(ty) = self.unify(&if_ty, &else_ty) {
                    ty
                } else {
                    // TODO report diagnostic
                    Ty::Unknown
                }
            }
            ast::Expr::BlockExpr(e) => {
                if let Some(block) = e.block() {
                    self.infer_block(block)?
                } else {
                    Ty::Unknown
                }
            }
            ast::Expr::LoopExpr(e) => {
                if let Some(block) = e.loop_body() {
                    self.infer_block(block)?;
                };
                // TODO never, or the type of the break param
                Ty::Unknown
            }
            ast::Expr::WhileExpr(e) => {
                if let Some(condition) = e.condition() {
                    if let Some(e) = condition.expr() {
                        // TODO if no pat, this should be bool
                        self.infer_expr(e)?;
                    }
                    // TODO write type for pat
                };
                if let Some(block) = e.loop_body() {
                    // TODO
                    self.infer_block(block)?;
                };
                // TODO always unit?
                Ty::Unknown
            }
            ast::Expr::ForExpr(e) => {
                if let Some(expr) = e.iterable() {
                    self.infer_expr(expr)?;
                }
                if let Some(_pat) = e.pat() {
                    // TODO write type for pat
                }
                if let Some(block) = e.loop_body() {
                    self.infer_block(block)?;
                }
                // TODO always unit?
                Ty::Unknown
            }
            ast::Expr::LambdaExpr(e) => {
                let _body_ty = if let Some(body) = e.body() {
                    self.infer_expr(body)?
                } else {
                    Ty::Unknown
                };
                Ty::Unknown
            }
            ast::Expr::CallExpr(e) => {
                let callee_ty = if let Some(e) = e.expr() {
                    self.infer_expr(e)?
                } else {
                    Ty::Unknown
                };
                if let Some(arg_list) = e.arg_list() {
                    for arg in arg_list.args() {
                        // TODO unify / expect argument type
                        self.infer_expr(arg)?;
                    }
                }
                match callee_ty {
                    Ty::FnPtr(sig) => sig.output.clone(),
                    _ => {
                        // not callable
                        // TODO report an error?
                        Ty::Unknown
                    }
                }
            }
            ast::Expr::MethodCallExpr(e) => {
                let _receiver_ty = if let Some(e) = e.expr() {
                    self.infer_expr(e)?
                } else {
                    Ty::Unknown
                };
                if let Some(arg_list) = e.arg_list() {
                    for arg in arg_list.args() {
                        // TODO unify / expect argument type
                        self.infer_expr(arg)?;
                    }
                }
                Ty::Unknown
            }
            ast::Expr::MatchExpr(e) => {
                let _ty = if let Some(match_expr) = e.expr() {
                    self.infer_expr(match_expr)?
                } else {
                    Ty::Unknown
                };
                if let Some(match_arm_list) = e.match_arm_list() {
                    for arm in match_arm_list.arms() {
                        // TODO type the bindings in pat
                        // TODO type the guard
                        let _ty = if let Some(e) = arm.expr() {
                            self.infer_expr(e)?
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
            ast::Expr::PathExpr(e) => self.infer_path_expr(e)?.unwrap_or(Ty::Unknown),
            ast::Expr::ContinueExpr(_e) => Ty::Never,
            ast::Expr::BreakExpr(_e) => Ty::Never,
            ast::Expr::ParenExpr(e) => {
                if let Some(e) = e.expr() {
                    self.infer_expr(e)?
                } else {
                    Ty::Unknown
                }
            }
            ast::Expr::Label(_e) => Ty::Unknown,
            ast::Expr::ReturnExpr(e) => {
                if let Some(e) = e.expr() {
                    // TODO unify with return type
                    self.infer_expr(e)?;
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
                    self.infer_expr(e)?
                } else {
                    Ty::Unknown
                };
                Ty::Unknown
            }
            ast::Expr::CastExpr(e) => {
                let _inner_ty = if let Some(e) = e.expr() {
                    self.infer_expr(e)?
                } else {
                    Ty::Unknown
                };
                let cast_ty = e
                    .type_ref()
                    .map(|t| Ty::new(self.db, t))
                    .unwrap_or(Ok(Ty::Unknown))?;
                // TODO do the coercion...
                cast_ty
            }
            ast::Expr::RefExpr(e) => {
                let _inner_ty = if let Some(e) = e.expr() {
                    self.infer_expr(e)?
                } else {
                    Ty::Unknown
                };
                Ty::Unknown
            }
            ast::Expr::PrefixExpr(e) => {
                let _inner_ty = if let Some(e) = e.expr() {
                    self.infer_expr(e)?
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
        Ok(ty)
    }

    fn infer_block(&mut self, node: ast::Block) -> Cancelable<Ty> {
        for stmt in node.statements() {
            match stmt {
                ast::Stmt::LetStmt(stmt) => {
                    let decl_ty = if let Some(type_ref) = stmt.type_ref() {
                        Ty::new(self.db, type_ref)?
                    } else {
                        Ty::Unknown
                    };
                    let ty = if let Some(expr) = stmt.initializer() {
                        // TODO pass expectation
                        let expr_ty = self.infer_expr(expr)?;
                        self.unify_with_coercion(&expr_ty, &decl_ty)
                            .unwrap_or(decl_ty)
                    } else {
                        decl_ty
                    };

                    if let Some(pat) = stmt.pat() {
                        self.write_ty(pat.syntax(), ty);
                    };
                }
                ast::Stmt::ExprStmt(expr_stmt) => {
                    if let Some(expr) = expr_stmt.expr() {
                        self.infer_expr(expr)?;
                    }
                }
            }
        }
        let ty = if let Some(expr) = node.expr() {
            self.infer_expr(expr)?
        } else {
            Ty::unit()
        };
        self.write_ty(node.syntax(), ty.clone());
        Ok(ty)
    }
}

pub fn infer(db: &impl HirDatabase, function: Function) -> Cancelable<InferenceResult> {
    let scopes = function.scopes(db);
    let module = function.module(db)?;
    let mut ctx = InferenceContext::new(db, scopes, module);

    let syntax = function.syntax(db);
    let node = syntax.borrowed();

    if let Some(param_list) = node.param_list() {
        for param in param_list.params() {
            let pat = if let Some(pat) = param.pat() {
                pat
            } else {
                continue;
            };
            if let Some(type_ref) = param.type_ref() {
                let ty = Ty::new(db, type_ref)?;
                ctx.type_for.insert(LocalSyntaxPtr::new(pat.syntax()), ty);
            } else {
                // TODO self param
                ctx.type_for
                    .insert(LocalSyntaxPtr::new(pat.syntax()), Ty::Unknown);
            };
        }
    }

    // TODO get Ty for node.ret_type() and pass that to infer_block as expectation
    // (see Expectation in rustc_typeck)

    if let Some(block) = node.body() {
        ctx.infer_block(block)?;
    }

    // TODO 'resolve' the types: replace inference variables by their inferred results

    Ok(InferenceResult {
        type_for: ctx.type_for,
    })
}
