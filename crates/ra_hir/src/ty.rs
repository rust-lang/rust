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
    ast::{self, AstNode, LoopBodyOwner, ArgListOwner, PrefixOp},
    SyntaxNodeRef
};

use crate::{
    Def, DefId, FnScopes, Module, Function, Struct, Enum, Path,
    db::HirDatabase,
    adt::VariantData,
    type_ref::{TypeRef, Mutability},
};

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
        name: SmolStr,
        // later we'll need generic substitutions here
    },

    /// The pointee of a string slice. Written as `str`.
    Str,

    // An array with the given length. Written as `[T; n]`.
    // Array(Ty, ty::Const),
    /// The pointee of an array slice.  Written as `[T]`.
    Slice(TyRef),

    /// A raw pointer. Written as `*mut T` or `*const T`
    RawPtr(TyRef, Mutability),

    /// A reference; a pointer with an associated lifetime. Written as
    /// `&'a mut T` or `&'a T`.
    Ref(TyRef, Mutability),

    /// A pointer to a function.  Written as `fn() -> i32`.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar: fn() -> i32 = foo;
    /// ```
    FnPtr(Arc<FnSig>),

    // A trait, defined with `dyn trait`.
    // Dynamic(),
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

    // The projection of an associated type.  For example,
    // `<T as Trait<..>>::N`.
    // Projection(ProjectionTy),

    // Opaque (`impl Trait`) type found in a return type.
    // The `DefId` comes either from
    // * the `impl Trait` ast::Ty node,
    // * or the `existential type` declaration
    // The substitutions are for the generics of the function in question.
    // Opaque(DefId, Substs),

    // A type parameter; for example, `T` in `fn f<T>(x: T) {}
    // Param(ParamTy),

    // A placeholder type - universally quantified higher-ranked type.
    // Placeholder(ty::PlaceholderType),

    // A type variable used during type checking.
    // Infer(InferTy),
    /// A placeholder for a type which could not be computed; this is
    /// propagated to avoid useless error messages.
    Unknown,
}

type TyRef = Arc<Ty>;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FnSig {
    input: Vec<Ty>,
    output: Ty,
}

impl Ty {
    pub(crate) fn from_hir(
        db: &impl HirDatabase,
        module: &Module,
        type_ref: &TypeRef,
    ) -> Cancelable<Self> {
        Ok(match type_ref {
            TypeRef::Never => Ty::Never,
            TypeRef::Tuple(inner) => {
                let inner_tys = inner
                    .iter()
                    .map(|tr| Ty::from_hir(db, module, tr))
                    .collect::<Cancelable<_>>()?;
                Ty::Tuple(inner_tys)
            }
            TypeRef::Path(path) => Ty::from_hir_path(db, module, path)?,
            TypeRef::RawPtr(inner, mutability) => {
                let inner_ty = Ty::from_hir(db, module, inner)?;
                Ty::RawPtr(Arc::new(inner_ty), *mutability)
            }
            TypeRef::Array(_inner) => Ty::Unknown, // TODO
            TypeRef::Slice(inner) => {
                let inner_ty = Ty::from_hir(db, module, inner)?;
                Ty::Slice(Arc::new(inner_ty))
            }
            TypeRef::Reference(inner, mutability) => {
                let inner_ty = Ty::from_hir(db, module, inner)?;
                Ty::Ref(Arc::new(inner_ty), *mutability)
            }
            TypeRef::Placeholder => Ty::Unknown, // TODO
            TypeRef::Fn(params) => {
                let mut inner_tys = params
                    .iter()
                    .map(|tr| Ty::from_hir(db, module, tr))
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

    pub(crate) fn from_hir_path(
        db: &impl HirDatabase,
        module: &Module,
        path: &Path,
    ) -> Cancelable<Self> {
        if let Some(name) = path.as_ident() {
            let name = name.as_str(); // :-(
            if let Some(int_ty) = primitive::IntTy::from_string(name) {
                return Ok(Ty::Int(int_ty));
            } else if let Some(uint_ty) = primitive::UintTy::from_string(name) {
                return Ok(Ty::Uint(uint_ty));
            } else if let Some(float_ty) = primitive::FloatTy::from_string(name) {
                return Ok(Ty::Float(float_ty));
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
        node: Option<ast::TypeRef>,
    ) -> Cancelable<Self> {
        node.map(|n| Ty::from_ast(db, module, n))
            .unwrap_or(Ok(Ty::Unknown))
    }

    pub(crate) fn from_ast(
        db: &impl HirDatabase,
        module: &Module,
        node: ast::TypeRef,
    ) -> Cancelable<Self> {
        Ty::from_hir(db, module, &TypeRef::from_ast(node))
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
            Ty::RawPtr(t, m) => write!(f, "*{}{}", m.as_keyword_for_ptr(), t),
            Ty::Ref(t, m) => write!(f, "&{}{}", m.as_keyword_for_ref(), t),
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
            Ty::Adt { name, .. } => write!(f, "{}", name),
            Ty::Unknown => write!(f, "[unknown]"),
        }
    }
}

pub fn type_for_fn(db: &impl HirDatabase, f: Function) -> Cancelable<Ty> {
    let syntax = f.syntax(db);
    let module = f.module(db)?;
    let node = syntax.borrowed();
    // TODO we ignore type parameters for now
    let input = node
        .param_list()
        .map(|pl| {
            pl.params()
                .map(|p| Ty::from_ast_opt(db, &module, p.type_ref()))
                .collect()
        })
        .unwrap_or_else(|| Ok(Vec::new()))?;
    let output = Ty::from_ast_opt(db, &module, node.ret_type().and_then(|rt| rt.type_ref()))?;
    let sig = FnSig { input, output };
    Ok(Ty::FnPtr(Arc::new(sig)))
}

pub fn type_for_struct(db: &impl HirDatabase, s: Struct) -> Cancelable<Ty> {
    Ok(Ty::Adt {
        def_id: s.def_id(),
        name: s
            .name(db)?
            .unwrap_or_else(|| SmolStr::new("[unnamed struct]")),
    })
}

pub fn type_for_enum(db: &impl HirDatabase, s: Enum) -> Cancelable<Ty> {
    Ok(Ty::Adt {
        def_id: s.def_id(),
        name: s
            .name(db)?
            .unwrap_or_else(|| SmolStr::new("[unnamed enum]")),
    })
}

pub fn type_for_def(db: &impl HirDatabase, def_id: DefId) -> Cancelable<Ty> {
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

pub(super) fn type_for_field(
    db: &impl HirDatabase,
    def_id: DefId,
    field: SmolStr,
) -> Cancelable<Ty> {
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
    let type_ref = if let Some(tr) = variant_data.get_field_type_ref(&field) {
        tr
    } else {
        return Ok(Ty::Unknown);
    };
    Ty::from_hir(db, &module, &type_ref)
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InferenceResult {
    type_of: FxHashMap<LocalSyntaxPtr, Ty>,
}

impl InferenceResult {
    pub fn type_of_node(&self, node: SyntaxNodeRef) -> Option<Ty> {
        self.type_of.get(&LocalSyntaxPtr::new(node)).cloned()
    }
}

#[derive(Clone, Debug)]
pub struct InferenceContext<'a, D: HirDatabase> {
    db: &'a D,
    scopes: Arc<FnScopes>,
    module: Module,
    // TODO unification tables...
    type_of: FxHashMap<LocalSyntaxPtr, Ty>,
}

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    fn new(db: &'a D, scopes: Arc<FnScopes>, module: Module) -> Self {
        InferenceContext {
            type_of: FxHashMap::default(),
            db,
            scopes,
            module,
        }
    }

    fn write_ty(&mut self, node: SyntaxNodeRef, ty: Ty) {
        self.type_of.insert(LocalSyntaxPtr::new(node), ty);
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
                let ty = ctry!(self.type_of.get(&scope_entry.ptr()));
                return Ok(Some(ty.clone()));
            };
        };

        // resolve in module
        let resolved = ctry!(self.module.resolve_path(self.db, &path)?.take_values());
        let ty = self.db.type_for_def(resolved)?;
        // TODO we will need to add type variables for type parameters etc. here
        Ok(Some(ty))
    }

    fn resolve_variant(
        &self,
        path: Option<ast::Path>,
    ) -> Cancelable<(Ty, Option<Arc<VariantData>>)> {
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
                let struct_data = self.db.struct_data(def_id)?;
                let ty = type_for_struct(self.db, s)?;
                (ty, Some(struct_data.variant_data().clone()))
            }
            _ => (Ty::Unknown, None),
        })
    }

    fn infer_expr_opt(&mut self, expr: Option<ast::Expr>) -> Cancelable<Ty> {
        if let Some(e) = expr {
            self.infer_expr(e)
        } else {
            Ok(Ty::Unknown)
        }
    }

    fn infer_expr(&mut self, expr: ast::Expr) -> Cancelable<Ty> {
        let ty = match expr {
            ast::Expr::IfExpr(e) => {
                if let Some(condition) = e.condition() {
                    // TODO if no pat, this should be bool
                    self.infer_expr_opt(condition.expr())?;
                    // TODO write type for pat
                };
                let if_ty = self.infer_block_opt(e.then_branch())?;
                let else_ty = self.infer_block_opt(e.else_branch())?;
                if let Some(ty) = self.unify(&if_ty, &else_ty) {
                    ty
                } else {
                    // TODO report diagnostic
                    Ty::Unknown
                }
            }
            ast::Expr::BlockExpr(e) => self.infer_block_opt(e.block())?,
            ast::Expr::LoopExpr(e) => {
                self.infer_block_opt(e.loop_body())?;
                // TODO never, or the type of the break param
                Ty::Unknown
            }
            ast::Expr::WhileExpr(e) => {
                if let Some(condition) = e.condition() {
                    // TODO if no pat, this should be bool
                    self.infer_expr_opt(condition.expr())?;
                    // TODO write type for pat
                };
                self.infer_block_opt(e.loop_body())?;
                // TODO always unit?
                Ty::Unknown
            }
            ast::Expr::ForExpr(e) => {
                let _iterable_ty = self.infer_expr_opt(e.iterable());
                if let Some(_pat) = e.pat() {
                    // TODO write type for pat
                }
                self.infer_block_opt(e.loop_body())?;
                // TODO always unit?
                Ty::Unknown
            }
            ast::Expr::LambdaExpr(e) => {
                let _body_ty = self.infer_expr_opt(e.body())?;
                Ty::Unknown
            }
            ast::Expr::CallExpr(e) => {
                let callee_ty = self.infer_expr_opt(e.expr())?;
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
                let _receiver_ty = self.infer_expr_opt(e.expr())?;
                if let Some(arg_list) = e.arg_list() {
                    for arg in arg_list.args() {
                        // TODO unify / expect argument type
                        self.infer_expr(arg)?;
                    }
                }
                Ty::Unknown
            }
            ast::Expr::MatchExpr(e) => {
                let _ty = self.infer_expr_opt(e.expr())?;
                if let Some(match_arm_list) = e.match_arm_list() {
                    for arm in match_arm_list.arms() {
                        // TODO type the bindings in pat
                        // TODO type the guard
                        let _ty = self.infer_expr_opt(arm.expr())?;
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
            ast::Expr::ParenExpr(e) => self.infer_expr_opt(e.expr())?,
            ast::Expr::Label(_e) => Ty::Unknown,
            ast::Expr::ReturnExpr(e) => {
                self.infer_expr_opt(e.expr())?;
                Ty::Never
            }
            ast::Expr::MatchArmList(_) | ast::Expr::MatchArm(_) | ast::Expr::MatchGuard(_) => {
                // Can this even occur outside of a match expression?
                Ty::Unknown
            }
            ast::Expr::StructLit(e) => {
                let (ty, _variant_data) = self.resolve_variant(e.path())?;
                if let Some(nfl) = e.named_field_list() {
                    for field in nfl.fields() {
                        // TODO unify with / expect field type
                        self.infer_expr_opt(field.expr())?;
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
                let receiver_ty = self.infer_expr_opt(e.expr())?;
                if let Some(nr) = e.name_ref() {
                    let text = nr.text();
                    match receiver_ty {
                        Ty::Tuple(fields) => {
                            let i = text.parse::<usize>().ok();
                            i.and_then(|i| fields.get(i).cloned())
                                .unwrap_or(Ty::Unknown)
                        }
                        Ty::Adt { def_id, .. } => self.db.type_for_field(def_id, text)?,
                        _ => Ty::Unknown,
                    }
                } else {
                    Ty::Unknown
                }
            }
            ast::Expr::TryExpr(e) => {
                let _inner_ty = self.infer_expr_opt(e.expr())?;
                Ty::Unknown
            }
            ast::Expr::CastExpr(e) => {
                let _inner_ty = self.infer_expr_opt(e.expr())?;
                let cast_ty = Ty::from_ast_opt(self.db, &self.module, e.type_ref())?;
                // TODO do the coercion...
                cast_ty
            }
            ast::Expr::RefExpr(e) => {
                let inner_ty = self.infer_expr_opt(e.expr())?;
                let m = Mutability::from_mutable(e.is_mut());
                // TODO reference coercions etc.
                Ty::Ref(Arc::new(inner_ty), m)
            }
            ast::Expr::PrefixExpr(e) => {
                let inner_ty = self.infer_expr_opt(e.expr())?;
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
        self.write_ty(expr.syntax(), ty.clone());
        Ok(ty)
    }

    fn infer_block_opt(&mut self, node: Option<ast::Block>) -> Cancelable<Ty> {
        if let Some(b) = node {
            self.infer_block(b)
        } else {
            Ok(Ty::Unknown)
        }
    }

    fn infer_block(&mut self, node: ast::Block) -> Cancelable<Ty> {
        for stmt in node.statements() {
            match stmt {
                ast::Stmt::LetStmt(stmt) => {
                    let decl_ty = Ty::from_ast_opt(self.db, &self.module, stmt.type_ref())?;
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
                    self.infer_expr_opt(expr_stmt.expr())?;
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
                let ty = Ty::from_ast(db, &ctx.module, type_ref)?;
                ctx.type_of.insert(LocalSyntaxPtr::new(pat.syntax()), ty);
            } else {
                // TODO self param
                ctx.type_of
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
        type_of: ctx.type_of,
    })
}
