//! Type inference, i.e. the process of walking through the code and determining
//! the type of each expression and pattern.
//!
//! For type inference, compare the implementations in rustc (the various
//! check_* methods in librustc_typeck/check/mod.rs are a good entry point) and
//! IntelliJ-Rust (org.rust.lang.core.types.infer). Our entry point for
//! inference here is the `infer` function, which infers the types of all
//! expressions in a given function.
//!
//! During inference, types (i.e. the `Ty` struct) can contain type 'variables'
//! which represent currently unknown types; as we walk through the expressions,
//! we might determine that certain variables need to be equal to each other, or
//! to certain types. To record this, we use the union-find implementation from
//! the `ena` crate, which is extracted from rustc.

use std::borrow::Cow;
use std::iter::repeat;
use std::ops::Index;
use std::sync::Arc;
use std::mem;

use ena::unify::{InPlaceUnificationTable, UnifyKey, UnifyValue, NoError};
use ra_arena::map::ArenaMap;
use rustc_hash::FxHashMap;

use test_utils::tested_by;

use crate::{
    Function, StructField, Path, Name,
    FnSignature, AdtDef,
    HirDatabase,
    type_ref::{TypeRef, Mutability},
    expr::{Body, Expr, BindingAnnotation, Literal, ExprId, Pat, PatId, UnaryOp, BinaryOp, Statement, FieldPat, self},
    generics::GenericParams,
    path::{GenericArgs, GenericArg},
    adt::VariantDef,
    resolve::{Resolver, Resolution},
    nameres::Namespace
};
use super::{Ty, TypableDef, Substs, primitive, op};

/// The entry point of type inference.
pub fn infer(db: &impl HirDatabase, func: Function) -> Arc<InferenceResult> {
    db.check_canceled();
    let body = func.body(db);
    let resolver = func.resolver(db);
    let mut ctx = InferenceContext::new(db, body, resolver);

    let signature = func.signature(db);
    ctx.collect_fn_signature(&signature);

    ctx.infer_body();

    Arc::new(ctx.resolve_all())
}

/// The result of type inference: A mapping from expressions and patterns to types.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InferenceResult {
    /// For each method call expr, records the function it resolves to.
    method_resolutions: FxHashMap<ExprId, Function>,
    /// For each field access expr, records the field it resolves to.
    field_resolutions: FxHashMap<ExprId, StructField>,
    pub(super) type_of_expr: ArenaMap<ExprId, Ty>,
    pub(super) type_of_pat: ArenaMap<PatId, Ty>,
}

impl InferenceResult {
    pub fn method_resolution(&self, expr: ExprId) -> Option<Function> {
        self.method_resolutions.get(&expr).map(|it| *it)
    }
    pub fn field_resolution(&self, expr: ExprId) -> Option<StructField> {
        self.field_resolutions.get(&expr).map(|it| *it)
    }
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
    resolver: Resolver,
    var_unification_table: InPlaceUnificationTable<TypeVarId>,
    method_resolutions: FxHashMap<ExprId, Function>,
    field_resolutions: FxHashMap<ExprId, StructField>,
    type_of_expr: ArenaMap<ExprId, Ty>,
    type_of_pat: ArenaMap<PatId, Ty>,
    /// The return type of the function being inferred.
    return_ty: Ty,
}

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    fn new(db: &'a D, body: Arc<Body>, resolver: Resolver) -> Self {
        InferenceContext {
            method_resolutions: FxHashMap::default(),
            field_resolutions: FxHashMap::default(),
            type_of_expr: ArenaMap::default(),
            type_of_pat: ArenaMap::default(),
            var_unification_table: InPlaceUnificationTable::new(),
            return_ty: Ty::Unknown, // set in collect_fn_signature
            db,
            body,
            resolver,
        }
    }

    fn resolve_all(mut self) -> InferenceResult {
        let mut tv_stack = Vec::new();
        let mut expr_types = mem::replace(&mut self.type_of_expr, ArenaMap::default());
        for ty in expr_types.values_mut() {
            let resolved = self.resolve_ty_completely(&mut tv_stack, mem::replace(ty, Ty::Unknown));
            *ty = resolved;
        }
        let mut pat_types = mem::replace(&mut self.type_of_pat, ArenaMap::default());
        for ty in pat_types.values_mut() {
            let resolved = self.resolve_ty_completely(&mut tv_stack, mem::replace(ty, Ty::Unknown));
            *ty = resolved;
        }
        InferenceResult {
            method_resolutions: self.method_resolutions,
            field_resolutions: self.field_resolutions,
            type_of_expr: expr_types,
            type_of_pat: pat_types,
        }
    }

    fn write_expr_ty(&mut self, expr: ExprId, ty: Ty) {
        self.type_of_expr.insert(expr, ty);
    }

    fn write_method_resolution(&mut self, expr: ExprId, func: Function) {
        self.method_resolutions.insert(expr, func);
    }

    fn write_field_resolution(&mut self, expr: ExprId, field: StructField) {
        self.field_resolutions.insert(expr, field);
    }

    fn write_pat_ty(&mut self, pat: PatId, ty: Ty) {
        self.type_of_pat.insert(pat, ty);
    }

    fn make_ty(&mut self, type_ref: &TypeRef) -> Ty {
        let ty = Ty::from_hir(
            self.db,
            // TODO use right resolver for block
            &self.resolver,
            type_ref,
        );
        let ty = self.insert_type_vars(ty);
        ty
    }

    fn unify_substs(&mut self, substs1: &Substs, substs2: &Substs, depth: usize) -> bool {
        substs1.0.iter().zip(substs2.0.iter()).all(|(t1, t2)| self.unify_inner(t1, t2, depth))
    }

    fn unify(&mut self, ty1: &Ty, ty2: &Ty) -> bool {
        self.unify_inner(ty1, ty2, 0)
    }

    fn unify_inner(&mut self, ty1: &Ty, ty2: &Ty, depth: usize) -> bool {
        if depth > 1000 {
            // prevent stackoverflows
            panic!("infinite recursion in unification");
        }
        if ty1 == ty2 {
            return true;
        }
        // try to resolve type vars first
        let ty1 = self.resolve_ty_shallow(ty1);
        let ty2 = self.resolve_ty_shallow(ty2);
        match (&*ty1, &*ty2) {
            (Ty::Unknown, ..) => true,
            (.., Ty::Unknown) => true,
            (Ty::Int(t1), Ty::Int(t2)) => match (t1, t2) {
                (primitive::UncertainIntTy::Unknown, _)
                | (_, primitive::UncertainIntTy::Unknown) => true,
                _ => t1 == t2,
            },
            (Ty::Float(t1), Ty::Float(t2)) => match (t1, t2) {
                (primitive::UncertainFloatTy::Unknown, _)
                | (_, primitive::UncertainFloatTy::Unknown) => true,
                _ => t1 == t2,
            },
            (Ty::Bool, _) | (Ty::Str, _) | (Ty::Never, _) | (Ty::Char, _) => ty1 == ty2,
            (
                Ty::Adt { def_id: def_id1, substs: substs1, .. },
                Ty::Adt { def_id: def_id2, substs: substs2, .. },
            ) if def_id1 == def_id2 => self.unify_substs(substs1, substs2, depth + 1),
            (Ty::Slice(t1), Ty::Slice(t2)) => self.unify_inner(t1, t2, depth + 1),
            (Ty::RawPtr(t1, m1), Ty::RawPtr(t2, m2)) if m1 == m2 => {
                self.unify_inner(t1, t2, depth + 1)
            }
            (Ty::Ref(t1, m1), Ty::Ref(t2, m2)) if m1 == m2 => self.unify_inner(t1, t2, depth + 1),
            (Ty::FnPtr(sig1), Ty::FnPtr(sig2)) if sig1 == sig2 => true,
            (Ty::Tuple(ts1), Ty::Tuple(ts2)) if ts1.len() == ts2.len() => {
                ts1.iter().zip(ts2.iter()).all(|(t1, t2)| self.unify_inner(t1, t2, depth + 1))
            }
            (Ty::Infer(InferTy::TypeVar(tv1)), Ty::Infer(InferTy::TypeVar(tv2)))
            | (Ty::Infer(InferTy::IntVar(tv1)), Ty::Infer(InferTy::IntVar(tv2)))
            | (Ty::Infer(InferTy::FloatVar(tv1)), Ty::Infer(InferTy::FloatVar(tv2))) => {
                // both type vars are unknown since we tried to resolve them
                self.var_unification_table.union(*tv1, *tv2);
                true
            }
            (Ty::Infer(InferTy::TypeVar(tv)), other)
            | (other, Ty::Infer(InferTy::TypeVar(tv)))
            | (Ty::Infer(InferTy::IntVar(tv)), other)
            | (other, Ty::Infer(InferTy::IntVar(tv)))
            | (Ty::Infer(InferTy::FloatVar(tv)), other)
            | (other, Ty::Infer(InferTy::FloatVar(tv))) => {
                // the type var is unknown since we tried to resolve it
                self.var_unification_table.union_value(*tv, TypeVarValue::Known(other.clone()));
                true
            }
            _ => false,
        }
    }

    fn new_type_var(&mut self) -> Ty {
        Ty::Infer(InferTy::TypeVar(self.var_unification_table.new_key(TypeVarValue::Unknown)))
    }

    fn new_integer_var(&mut self) -> Ty {
        Ty::Infer(InferTy::IntVar(self.var_unification_table.new_key(TypeVarValue::Unknown)))
    }

    fn new_float_var(&mut self) -> Ty {
        Ty::Infer(InferTy::FloatVar(self.var_unification_table.new_key(TypeVarValue::Unknown)))
    }

    /// Replaces Ty::Unknown by a new type var, so we can maybe still infer it.
    fn insert_type_vars_shallow(&mut self, ty: Ty) -> Ty {
        match ty {
            Ty::Unknown => self.new_type_var(),
            Ty::Int(primitive::UncertainIntTy::Unknown) => self.new_integer_var(),
            Ty::Float(primitive::UncertainFloatTy::Unknown) => self.new_float_var(),
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
    fn resolve_ty_as_possible(&mut self, tv_stack: &mut Vec<TypeVarId>, ty: Ty) -> Ty {
        ty.fold(&mut |ty| match ty {
            Ty::Infer(tv) => {
                let inner = tv.to_inner();
                if tv_stack.contains(&inner) {
                    tested_by!(type_var_cycles_resolve_as_possible);
                    // recursive type
                    return tv.fallback_value();
                }
                if let Some(known_ty) = self.var_unification_table.probe_value(inner).known() {
                    // known_ty may contain other variables that are known by now
                    tv_stack.push(inner);
                    let result = self.resolve_ty_as_possible(tv_stack, known_ty.clone());
                    tv_stack.pop();
                    result
                } else {
                    ty
                }
            }
            _ => ty,
        })
    }

    /// If `ty` is a type variable with known type, returns that type;
    /// otherwise, return ty.
    fn resolve_ty_shallow<'b>(&mut self, ty: &'b Ty) -> Cow<'b, Ty> {
        let mut ty = Cow::Borrowed(ty);
        // The type variable could resolve to a int/float variable. Hence try
        // resolving up to three times; each type of variable shouldn't occur
        // more than once
        for i in 0..3 {
            if i > 0 {
                tested_by!(type_var_resolves_to_int_var);
            }
            match &*ty {
                Ty::Infer(tv) => {
                    let inner = tv.to_inner();
                    match self.var_unification_table.probe_value(inner).known() {
                        Some(known_ty) => {
                            // The known_ty can't be a type var itself
                            ty = Cow::Owned(known_ty.clone());
                        }
                        _ => return ty,
                    }
                }
                _ => return ty,
            }
        }
        log::error!("Inference variable still not resolved: {:?}", ty);
        ty
    }

    /// Resolves the type completely; type variables without known type are
    /// replaced by Ty::Unknown.
    fn resolve_ty_completely(&mut self, tv_stack: &mut Vec<TypeVarId>, ty: Ty) -> Ty {
        ty.fold(&mut |ty| match ty {
            Ty::Infer(tv) => {
                let inner = tv.to_inner();
                if tv_stack.contains(&inner) {
                    tested_by!(type_var_cycles_resolve_completely);
                    // recursive type
                    return tv.fallback_value();
                }
                if let Some(known_ty) = self.var_unification_table.probe_value(inner).known() {
                    // known_ty may contain other variables that are known by now
                    tv_stack.push(inner);
                    let result = self.resolve_ty_completely(tv_stack, known_ty.clone());
                    tv_stack.pop();
                    result
                } else {
                    tv.fallback_value()
                }
            }
            _ => ty,
        })
    }

    fn infer_path_expr(&mut self, resolver: &Resolver, path: &Path) -> Option<Ty> {
        let resolved = resolver.resolve_path_segments(self.db, &path);

        let (def, remaining_index) = resolved.into_inner();

        log::debug!(
            "path {:?} resolved to {:?} with remaining index {:?}",
            path,
            def,
            remaining_index
        );

        // if the remaining_index is None, we expect the path
        // to be fully resolved, in this case we continue with
        // the default by attempting to `take_values´ from the resolution.
        // Otherwise the path was partially resolved, which means
        // we might have resolved into a type for which
        // we may find some associated item starting at the
        // path.segment pointed to by `remaining_index´
        let mut resolved =
            if remaining_index.is_none() { def.take_values()? } else { def.take_types()? };

        let remaining_index = remaining_index.unwrap_or(path.segments.len());

        // resolve intermediate segments
        for segment in &path.segments[remaining_index..] {
            let ty = match resolved {
                Resolution::Def(def) => {
                    let typable: Option<TypableDef> = def.into();
                    let typable = typable?;

                    let substs =
                        Ty::substs_from_path_segment(self.db, &self.resolver, segment, typable);
                    self.db.type_for_def(typable, Namespace::Types).apply_substs(substs)
                }
                Resolution::LocalBinding(_) => {
                    // can't have a local binding in an associated item path
                    return None;
                }
                Resolution::GenericParam(..) => {
                    // TODO associated item of generic param
                    return None;
                }
                Resolution::SelfType(_) => {
                    // TODO associated item of self type
                    return None;
                }
            };

            // Attempt to find an impl_item for the type which has a name matching
            // the current segment
            log::debug!("looking for path segment: {:?}", segment);
            let item = ty.iterate_impl_items(self.db, |item| match item {
                crate::ImplItem::Method(func) => {
                    let sig = func.signature(self.db);
                    if segment.name == *sig.name() {
                        return Some(func);
                    }
                    None
                }

                // TODO: Resolve associated const
                crate::ImplItem::Const(_) => None,

                // TODO: Resolve associated types
                crate::ImplItem::TypeAlias(_) => None,
            })?;
            resolved = Resolution::Def(item.into());
        }

        match resolved {
            Resolution::Def(def) => {
                let typable: Option<TypableDef> = def.into();
                let typable = typable?;

                let substs = Ty::substs_from_path(self.db, &self.resolver, path, typable);
                let ty = self.db.type_for_def(typable, Namespace::Values).apply_substs(substs);
                let ty = self.insert_type_vars(ty);
                Some(ty)
            }
            Resolution::LocalBinding(pat) => {
                let ty = self.type_of_pat.get(pat)?;
                let ty = self.resolve_ty_as_possible(&mut vec![], ty.clone());
                Some(ty)
            }
            Resolution::GenericParam(..) => {
                // generic params can't refer to values... yet
                None
            }
            Resolution::SelfType(_) => {
                log::error!("path expr {:?} resolved to Self type in values ns", path);
                None
            }
        }
    }

    fn resolve_variant(&mut self, path: Option<&Path>) -> (Ty, Option<VariantDef>) {
        let path = match path {
            Some(path) => path,
            None => return (Ty::Unknown, None),
        };
        let resolver = &self.resolver;
        let typable: Option<TypableDef> = match resolver.resolve_path(self.db, &path).take_types() {
            Some(Resolution::Def(def)) => def.into(),
            Some(Resolution::LocalBinding(..)) => {
                // this cannot happen
                log::error!("path resolved to local binding in type ns");
                return (Ty::Unknown, None);
            }
            Some(Resolution::GenericParam(..)) => {
                // generic params can't be used in struct literals
                return (Ty::Unknown, None);
            }
            Some(Resolution::SelfType(..)) => {
                // TODO this is allowed in an impl for a struct, handle this
                return (Ty::Unknown, None);
            }
            None => return (Ty::Unknown, None),
        };
        let def = match typable {
            None => return (Ty::Unknown, None),
            Some(it) => it,
        };
        // TODO remove the duplication between here and `Ty::from_path`?
        let substs = Ty::substs_from_path(self.db, resolver, path, def);
        match def {
            TypableDef::Struct(s) => {
                let ty = s.ty(self.db);
                let ty = self.insert_type_vars(ty.apply_substs(substs));
                (ty, Some(s.into()))
            }
            TypableDef::EnumVariant(var) => {
                let ty = var.parent_enum(self.db).ty(self.db);
                let ty = self.insert_type_vars(ty.apply_substs(substs));
                (ty, Some(var.into()))
            }
            TypableDef::TypeAlias(_) | TypableDef::Function(_) | TypableDef::Enum(_) => {
                (Ty::Unknown, None)
            }
        }
    }

    fn infer_tuple_struct_pat(
        &mut self,
        path: Option<&Path>,
        subpats: &[PatId],
        expected: &Ty,
    ) -> Ty {
        let (ty, def) = self.resolve_variant(path);

        self.unify(&ty, expected);

        let substs = ty.substs().unwrap_or_else(Substs::empty);

        for (i, &subpat) in subpats.iter().enumerate() {
            let expected_ty = def
                .and_then(|d| d.field(self.db, &Name::tuple_field_name(i)))
                .map_or(Ty::Unknown, |field| field.ty(self.db))
                .subst(&substs);
            self.infer_pat(subpat, &expected_ty);
        }

        ty
    }

    fn infer_struct_pat(&mut self, path: Option<&Path>, subpats: &[FieldPat], expected: &Ty) -> Ty {
        let (ty, def) = self.resolve_variant(path);

        self.unify(&ty, expected);

        let substs = ty.substs().unwrap_or_else(Substs::empty);

        for subpat in subpats {
            let matching_field = def.and_then(|it| it.field(self.db, &subpat.name));
            let expected_ty =
                matching_field.map_or(Ty::Unknown, |field| field.ty(self.db)).subst(&substs);
            self.infer_pat(subpat.pat, &expected_ty);
        }

        ty
    }

    fn infer_pat(&mut self, pat: PatId, expected: &Ty) -> Ty {
        let body = Arc::clone(&self.body); // avoid borrow checker problem

        let ty = match &body[pat] {
            Pat::Tuple(ref args) => {
                let expectations = match *expected {
                    Ty::Tuple(ref tuple_args) => &**tuple_args,
                    _ => &[],
                };
                let expectations_iter = expectations.iter().chain(repeat(&Ty::Unknown));

                let inner_tys = args
                    .iter()
                    .zip(expectations_iter)
                    .map(|(&pat, ty)| self.infer_pat(pat, ty))
                    .collect::<Vec<_>>()
                    .into();

                Ty::Tuple(inner_tys)
            }
            Pat::Ref { pat, mutability } => {
                let expectation = match *expected {
                    Ty::Ref(ref sub_ty, exp_mut) => {
                        if *mutability != exp_mut {
                            // TODO: emit type error?
                        }
                        &**sub_ty
                    }
                    _ => &Ty::Unknown,
                };
                let subty = self.infer_pat(*pat, expectation);
                Ty::Ref(subty.into(), *mutability)
            }
            Pat::TupleStruct { path: ref p, args: ref subpats } => {
                self.infer_tuple_struct_pat(p.as_ref(), subpats, expected)
            }
            Pat::Struct { path: ref p, args: ref fields } => {
                self.infer_struct_pat(p.as_ref(), fields, expected)
            }
            Pat::Path(path) => {
                // TODO use correct resolver for the surrounding expression
                let resolver = self.resolver.clone();
                self.infer_path_expr(&resolver, &path).unwrap_or(Ty::Unknown)
            }
            Pat::Bind { mode, name: _name, subpat } => {
                let inner_ty = if let Some(subpat) = subpat {
                    self.infer_pat(*subpat, expected)
                } else {
                    expected.clone()
                };
                let inner_ty = self.insert_type_vars_shallow(inner_ty);

                let bound_ty = match mode {
                    BindingAnnotation::Ref => Ty::Ref(inner_ty.clone().into(), Mutability::Shared),
                    BindingAnnotation::RefMut => Ty::Ref(inner_ty.clone().into(), Mutability::Mut),
                    BindingAnnotation::Mutable | BindingAnnotation::Unannotated => inner_ty.clone(),
                };
                let bound_ty = self.resolve_ty_as_possible(&mut vec![], bound_ty);
                self.write_pat_ty(pat, bound_ty);
                return inner_ty;
            }
            _ => Ty::Unknown,
        };
        // use a new type variable if we got Ty::Unknown here
        let ty = self.insert_type_vars_shallow(ty);
        self.unify(&ty, expected);
        let ty = self.resolve_ty_as_possible(&mut vec![], ty);
        self.write_pat_ty(pat, ty.clone());
        ty
    }

    fn substs_for_method_call(
        &mut self,
        def_generics: Option<Arc<GenericParams>>,
        generic_args: &Option<GenericArgs>,
    ) -> Substs {
        let (parent_param_count, param_count) =
            def_generics.map_or((0, 0), |g| (g.count_parent_params(), g.params.len()));
        let mut substs = Vec::with_capacity(parent_param_count + param_count);
        for _ in 0..parent_param_count {
            substs.push(Ty::Unknown);
        }
        // handle provided type arguments
        if let Some(generic_args) = generic_args {
            // if args are provided, it should be all of them, but we can't rely on that
            for arg in generic_args.args.iter().take(param_count) {
                match arg {
                    GenericArg::Type(type_ref) => {
                        let ty = self.make_ty(type_ref);
                        substs.push(ty);
                    }
                }
            }
        };
        let supplied_params = substs.len();
        for _ in supplied_params..parent_param_count + param_count {
            substs.push(Ty::Unknown);
        }
        assert_eq!(substs.len(), parent_param_count + param_count);
        Substs(substs.into())
    }

    fn infer_expr(&mut self, tgt_expr: ExprId, expected: &Expectation) -> Ty {
        let body = Arc::clone(&self.body); // avoid borrow checker problem
        let ty = match &body[tgt_expr] {
            Expr::Missing => Ty::Unknown,
            Expr::If { condition, then_branch, else_branch } => {
                // if let is desugared to match, so this is always simple if
                self.infer_expr(*condition, &Expectation::has_type(Ty::Bool));
                let then_ty = self.infer_expr(*then_branch, expected);
                match else_branch {
                    Some(else_branch) => {
                        self.infer_expr(*else_branch, expected);
                    }
                    None => {
                        // no else branch -> unit
                        self.unify(&then_ty, &Ty::unit()); // actually coerce
                    }
                };
                then_ty
            }
            Expr::Block { statements, tail } => self.infer_block(statements, *tail, expected),
            Expr::Loop { body } => {
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));
                // TODO handle break with value
                Ty::Never
            }
            Expr::While { condition, body } => {
                // while let is desugared to a match loop, so this is always simple while
                self.infer_expr(*condition, &Expectation::has_type(Ty::Bool));
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));
                Ty::unit()
            }
            Expr::For { iterable, body, pat } => {
                let _iterable_ty = self.infer_expr(*iterable, &Expectation::none());
                self.infer_pat(*pat, &Ty::Unknown);
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));
                Ty::unit()
            }
            Expr::Lambda { body, args, arg_types } => {
                assert_eq!(args.len(), arg_types.len());

                for (arg_pat, arg_type) in args.iter().zip(arg_types.iter()) {
                    let expected = if let Some(type_ref) = arg_type {
                        let ty = self.make_ty(type_ref);
                        ty
                    } else {
                        Ty::Unknown
                    };
                    self.infer_pat(*arg_pat, &expected);
                }

                // TODO: infer lambda type etc.
                let _body_ty = self.infer_expr(*body, &Expectation::none());
                Ty::Unknown
            }
            Expr::Call { callee, args } => {
                let callee_ty = self.infer_expr(*callee, &Expectation::none());
                let (param_tys, ret_ty) = match &callee_ty {
                    Ty::FnPtr(sig) => (sig.input.clone(), sig.output.clone()),
                    Ty::FnDef { substs, sig, .. } => {
                        let ret_ty = sig.output.clone().subst(&substs);
                        let param_tys =
                            sig.input.iter().map(|ty| ty.clone().subst(&substs)).collect();
                        (param_tys, ret_ty)
                    }
                    _ => {
                        // not callable
                        // TODO report an error?
                        (Vec::new(), Ty::Unknown)
                    }
                };
                let param_iter = param_tys.into_iter().chain(repeat(Ty::Unknown));
                for (arg, param) in args.iter().zip(param_iter) {
                    self.infer_expr(*arg, &Expectation::has_type(param));
                }
                ret_ty
            }
            Expr::MethodCall { receiver, args, method_name, generic_args } => {
                let receiver_ty = self.infer_expr(*receiver, &Expectation::none());
                let resolved = receiver_ty.clone().lookup_method(self.db, method_name);
                let (derefed_receiver_ty, method_ty, def_generics) = match resolved {
                    Some((ty, func)) => {
                        self.write_method_resolution(tgt_expr, func);
                        (
                            ty,
                            self.db.type_for_def(func.into(), Namespace::Values),
                            Some(func.generic_params(self.db)),
                        )
                    }
                    None => (Ty::Unknown, receiver_ty, None),
                };
                let substs = self.substs_for_method_call(def_generics, generic_args);
                let method_ty = method_ty.apply_substs(substs);
                let method_ty = self.insert_type_vars(method_ty);
                let (expected_receiver_ty, param_tys, ret_ty) = match &method_ty {
                    Ty::FnPtr(sig) => {
                        if !sig.input.is_empty() {
                            (sig.input[0].clone(), sig.input[1..].to_vec(), sig.output.clone())
                        } else {
                            (Ty::Unknown, Vec::new(), sig.output.clone())
                        }
                    }
                    Ty::FnDef { substs, sig, .. } => {
                        let ret_ty = sig.output.clone().subst(&substs);

                        if !sig.input.is_empty() {
                            let mut arg_iter = sig.input.iter().map(|ty| ty.clone().subst(&substs));
                            let receiver_ty = arg_iter.next().unwrap();
                            (receiver_ty, arg_iter.collect(), ret_ty)
                        } else {
                            (Ty::Unknown, Vec::new(), ret_ty)
                        }
                    }
                    _ => (Ty::Unknown, Vec::new(), Ty::Unknown),
                };
                // Apply autoref so the below unification works correctly
                let actual_receiver_ty = match expected_receiver_ty {
                    Ty::Ref(_, mutability) => Ty::Ref(Arc::new(derefed_receiver_ty), mutability),
                    _ => derefed_receiver_ty,
                };
                self.unify(&expected_receiver_ty, &actual_receiver_ty);

                let param_iter = param_tys.into_iter().chain(repeat(Ty::Unknown));
                for (arg, param) in args.iter().zip(param_iter) {
                    self.infer_expr(*arg, &Expectation::has_type(param));
                }
                ret_ty
            }
            Expr::Match { expr, arms } => {
                let expected = if expected.ty == Ty::Unknown {
                    Expectation::has_type(self.new_type_var())
                } else {
                    expected.clone()
                };
                let input_ty = self.infer_expr(*expr, &Expectation::none());

                for arm in arms {
                    for &pat in &arm.pats {
                        let _pat_ty = self.infer_pat(pat, &input_ty);
                    }
                    if let Some(guard_expr) = arm.guard {
                        self.infer_expr(guard_expr, &Expectation::has_type(Ty::Bool));
                    }
                    self.infer_expr(arm.expr, &expected);
                }

                expected.ty
            }
            Expr::Path(p) => {
                // TODO this could be more efficient...
                let resolver = expr::resolver_for_expr(self.body.clone(), self.db, tgt_expr);
                self.infer_path_expr(&resolver, p).unwrap_or(Ty::Unknown)
            }
            Expr::Continue => Ty::Never,
            Expr::Break { expr } => {
                if let Some(expr) = expr {
                    // TODO handle break with value
                    self.infer_expr(*expr, &Expectation::none());
                }
                Ty::Never
            }
            Expr::Return { expr } => {
                if let Some(expr) = expr {
                    self.infer_expr(*expr, &Expectation::has_type(self.return_ty.clone()));
                }
                Ty::Never
            }
            Expr::StructLit { path, fields, spread } => {
                let (ty, def_id) = self.resolve_variant(path.as_ref());
                let substs = ty.substs().unwrap_or_else(Substs::empty);
                for field in fields {
                    let field_ty = def_id
                        .and_then(|it| it.field(self.db, &field.name))
                        .map_or(Ty::Unknown, |field| field.ty(self.db))
                        .subst(&substs);
                    self.infer_expr(field.expr, &Expectation::has_type(field_ty));
                }
                if let Some(expr) = spread {
                    self.infer_expr(*expr, &Expectation::has_type(ty.clone()));
                }
                ty
            }
            Expr::Field { expr, name } => {
                let receiver_ty = self.infer_expr(*expr, &Expectation::none());
                let ty = receiver_ty
                    .autoderef(self.db)
                    .find_map(|derefed_ty| match derefed_ty {
                        Ty::Tuple(fields) => {
                            let i = name.to_string().parse::<usize>().ok();
                            i.and_then(|i| fields.get(i).cloned())
                        }
                        Ty::Adt { def_id: AdtDef::Struct(s), ref substs, .. } => {
                            s.field(self.db, name).map(|field| {
                                self.write_field_resolution(tgt_expr, field);
                                field.ty(self.db).subst(substs)
                            })
                        }
                        _ => None,
                    })
                    .unwrap_or(Ty::Unknown);
                self.insert_type_vars(ty)
            }
            Expr::Try { expr } => {
                let _inner_ty = self.infer_expr(*expr, &Expectation::none());
                Ty::Unknown
            }
            Expr::Cast { expr, type_ref } => {
                let _inner_ty = self.infer_expr(*expr, &Expectation::none());
                let cast_ty = self.make_ty(type_ref);
                // TODO check the cast...
                cast_ty
            }
            Expr::Ref { expr, mutability } => {
                let expectation = if let Ty::Ref(ref subty, expected_mutability) = expected.ty {
                    if expected_mutability == Mutability::Mut && *mutability == Mutability::Shared {
                        // TODO: throw type error - expected mut reference but found shared ref,
                        // which cannot be coerced
                    }
                    Expectation::has_type((**subty).clone())
                } else {
                    Expectation::none()
                };
                // TODO reference coercions etc.
                let inner_ty = self.infer_expr(*expr, &expectation);
                Ty::Ref(Arc::new(inner_ty), *mutability)
            }
            Expr::UnaryOp { expr, op } => {
                let inner_ty = self.infer_expr(*expr, &Expectation::none());
                match op {
                    UnaryOp::Deref => {
                        if let Some(derefed_ty) = inner_ty.builtin_deref() {
                            derefed_ty
                        } else {
                            // TODO Deref::deref
                            Ty::Unknown
                        }
                    }
                    UnaryOp::Neg => {
                        match inner_ty {
                            Ty::Int(primitive::UncertainIntTy::Unknown)
                            | Ty::Int(primitive::UncertainIntTy::Signed(..))
                            | Ty::Infer(InferTy::IntVar(..))
                            | Ty::Infer(InferTy::FloatVar(..))
                            | Ty::Float(..) => inner_ty,
                            // TODO: resolve ops::Neg trait
                            _ => Ty::Unknown,
                        }
                    }
                    UnaryOp::Not => {
                        match inner_ty {
                            Ty::Bool | Ty::Int(_) | Ty::Infer(InferTy::IntVar(..)) => inner_ty,
                            // TODO: resolve ops::Not trait for inner_ty
                            _ => Ty::Unknown,
                        }
                    }
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
                    let lhs_ty = self.infer_expr(*lhs, &lhs_expectation);
                    // TODO: find implementation of trait corresponding to operation
                    // symbol and resolve associated `Output` type
                    let rhs_expectation = op::binary_op_rhs_expectation(*op, lhs_ty);
                    let rhs_ty = self.infer_expr(*rhs, &Expectation::has_type(rhs_expectation));

                    // TODO: similar as above, return ty is often associated trait type
                    op::binary_op_return_ty(*op, rhs_ty)
                }
                _ => Ty::Unknown,
            },
            Expr::Tuple { exprs } => {
                let mut ty_vec = Vec::with_capacity(exprs.len());
                for arg in exprs.iter() {
                    ty_vec.push(self.infer_expr(*arg, &Expectation::none()));
                }

                Ty::Tuple(Arc::from(ty_vec))
            }
            Expr::Array { exprs } => {
                let elem_ty = match &expected.ty {
                    Ty::Slice(inner) | Ty::Array(inner) => Ty::clone(&inner),
                    _ => self.new_type_var(),
                };

                for expr in exprs.iter() {
                    self.infer_expr(*expr, &Expectation::has_type(elem_ty.clone()));
                }

                Ty::Array(Arc::new(elem_ty))
            }
            Expr::Literal(lit) => match lit {
                Literal::Bool(..) => Ty::Bool,
                Literal::String(..) => Ty::Ref(Arc::new(Ty::Str), Mutability::Shared),
                Literal::ByteString(..) => {
                    let byte_type = Arc::new(Ty::Int(primitive::UncertainIntTy::Unsigned(
                        primitive::UintTy::U8,
                    )));
                    let slice_type = Arc::new(Ty::Slice(byte_type));
                    Ty::Ref(slice_type, Mutability::Shared)
                }
                Literal::Char(..) => Ty::Char,
                Literal::Int(_v, ty) => Ty::Int(*ty),
                Literal::Float(_v, ty) => Ty::Float(*ty),
            },
        };
        // use a new type variable if we got Ty::Unknown here
        let ty = self.insert_type_vars_shallow(ty);
        self.unify(&ty, &expected.ty);
        let ty = self.resolve_ty_as_possible(&mut vec![], ty);
        self.write_expr_ty(tgt_expr, ty.clone());
        ty
    }

    fn infer_block(
        &mut self,
        statements: &[Statement],
        tail: Option<ExprId>,
        expected: &Expectation,
    ) -> Ty {
        for stmt in statements {
            match stmt {
                Statement::Let { pat, type_ref, initializer } => {
                    let decl_ty =
                        type_ref.as_ref().map(|tr| self.make_ty(tr)).unwrap_or(Ty::Unknown);
                    let decl_ty = self.insert_type_vars(decl_ty);
                    let ty = if let Some(expr) = initializer {
                        let expr_ty = self.infer_expr(*expr, &Expectation::has_type(decl_ty));
                        expr_ty
                    } else {
                        decl_ty
                    };

                    self.infer_pat(*pat, &ty);
                }
                Statement::Expr(expr) => {
                    self.infer_expr(*expr, &Expectation::none());
                }
            }
        }
        let ty = if let Some(expr) = tail { self.infer_expr(expr, expected) } else { Ty::unit() };
        ty
    }

    fn collect_fn_signature(&mut self, signature: &FnSignature) {
        let body = Arc::clone(&self.body); // avoid borrow checker problem
        for (type_ref, pat) in signature.params().iter().zip(body.params()) {
            let ty = self.make_ty(type_ref);

            self.infer_pat(*pat, &ty);
        }
        self.return_ty = self.make_ty(signature.ret_type());
    }

    fn infer_body(&mut self) {
        self.infer_expr(self.body.body_expr(), &Expectation::has_type(self.return_ty.clone()));
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

/// The kinds of placeholders we need during type inference. There's separate
/// values for general types, and for integer and float variables. The latter
/// two are used for inference of literal values (e.g. `100` could be one of
/// several integer types).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InferTy {
    TypeVar(TypeVarId),
    IntVar(TypeVarId),
    FloatVar(TypeVarId),
}

impl InferTy {
    fn to_inner(self) -> TypeVarId {
        match self {
            InferTy::TypeVar(ty) | InferTy::IntVar(ty) | InferTy::FloatVar(ty) => ty,
        }
    }

    fn fallback_value(self) -> Ty {
        match self {
            InferTy::TypeVar(..) => Ty::Unknown,
            InferTy::IntVar(..) => {
                Ty::Int(primitive::UncertainIntTy::Signed(primitive::IntTy::I32))
            }
            InferTy::FloatVar(..) => {
                Ty::Float(primitive::UncertainFloatTy::Known(primitive::FloatTy::F64))
            }
        }
    }
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
