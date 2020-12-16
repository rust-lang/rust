use crate::traits::{ObligationCause, ObligationCauseCode};
use crate::ty::diagnostics::suggest_constraining_type_param;
use crate::ty::{self, BoundRegion, Region, Ty, TyCtxt};
use rustc_ast as ast;
use rustc_errors::Applicability::{MachineApplicable, MaybeIncorrect};
use rustc_errors::{pluralize, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{BytePos, MultiSpan, Span};
use rustc_target::spec::abi;

use std::borrow::Cow;
use std::fmt;
use std::ops::Deref;

#[derive(Clone, Copy, Debug, PartialEq, Eq, TypeFoldable)]
pub struct ExpectedFound<T> {
    pub expected: T,
    pub found: T,
}

impl<T> ExpectedFound<T> {
    pub fn new(a_is_expected: bool, a: T, b: T) -> Self {
        if a_is_expected {
            ExpectedFound { expected: a, found: b }
        } else {
            ExpectedFound { expected: b, found: a }
        }
    }
}

// Data structures used in type unification
#[derive(Clone, Debug, TypeFoldable)]
pub enum TypeError<'tcx> {
    Mismatch,
    UnsafetyMismatch(ExpectedFound<hir::Unsafety>),
    AbiMismatch(ExpectedFound<abi::Abi>),
    Mutability,
    TupleSize(ExpectedFound<usize>),
    FixedArraySize(ExpectedFound<u64>),
    ArgCount,

    RegionsDoesNotOutlive(Region<'tcx>, Region<'tcx>),
    RegionsInsufficientlyPolymorphic(BoundRegion, Region<'tcx>),
    RegionsOverlyPolymorphic(BoundRegion, Region<'tcx>),
    RegionsPlaceholderMismatch,

    Sorts(ExpectedFound<Ty<'tcx>>),
    IntMismatch(ExpectedFound<ty::IntVarValue>),
    FloatMismatch(ExpectedFound<ast::FloatTy>),
    Traits(ExpectedFound<DefId>),
    VariadicMismatch(ExpectedFound<bool>),

    /// Instantiating a type variable with the given type would have
    /// created a cycle (because it appears somewhere within that
    /// type).
    CyclicTy(Ty<'tcx>),
    CyclicConst(&'tcx ty::Const<'tcx>),
    ProjectionMismatched(ExpectedFound<DefId>),
    ExistentialMismatch(ExpectedFound<&'tcx ty::List<ty::Binder<ty::ExistentialPredicate<'tcx>>>>),
    ObjectUnsafeCoercion(DefId),
    ConstMismatch(ExpectedFound<&'tcx ty::Const<'tcx>>),

    IntrinsicCast,
    /// Safe `#[target_feature]` functions are not assignable to safe function pointers.
    TargetFeatureCast(DefId),
}

pub enum UnconstrainedNumeric {
    UnconstrainedFloat,
    UnconstrainedInt,
    Neither,
}

/// Explains the source of a type err in a short, human readable way. This is meant to be placed
/// in parentheses after some larger message. You should also invoke `note_and_explain_type_err()`
/// afterwards to present additional details, particularly when it comes to lifetime-related
/// errors.
impl<'tcx> fmt::Display for TypeError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use self::TypeError::*;
        fn report_maybe_different(
            f: &mut fmt::Formatter<'_>,
            expected: &str,
            found: &str,
        ) -> fmt::Result {
            // A naive approach to making sure that we're not reporting silly errors such as:
            // (expected closure, found closure).
            if expected == found {
                write!(f, "expected {}, found a different {}", expected, found)
            } else {
                write!(f, "expected {}, found {}", expected, found)
            }
        }

        let br_string = |br: ty::BoundRegion| match br {
            ty::BrNamed(_, name) => format!(" {}", name),
            _ => String::new(),
        };

        match *self {
            CyclicTy(_) => write!(f, "cyclic type of infinite size"),
            CyclicConst(_) => write!(f, "encountered a self-referencing constant"),
            Mismatch => write!(f, "types differ"),
            UnsafetyMismatch(values) => {
                write!(f, "expected {} fn, found {} fn", values.expected, values.found)
            }
            AbiMismatch(values) => {
                write!(f, "expected {} fn, found {} fn", values.expected, values.found)
            }
            Mutability => write!(f, "types differ in mutability"),
            TupleSize(values) => write!(
                f,
                "expected a tuple with {} element{}, \
                           found one with {} element{}",
                values.expected,
                pluralize!(values.expected),
                values.found,
                pluralize!(values.found)
            ),
            FixedArraySize(values) => write!(
                f,
                "expected an array with a fixed size of {} element{}, \
                           found one with {} element{}",
                values.expected,
                pluralize!(values.expected),
                values.found,
                pluralize!(values.found)
            ),
            ArgCount => write!(f, "incorrect number of function parameters"),
            RegionsDoesNotOutlive(..) => write!(f, "lifetime mismatch"),
            RegionsInsufficientlyPolymorphic(br, _) => write!(
                f,
                "expected bound lifetime parameter{}, found concrete lifetime",
                br_string(br)
            ),
            RegionsOverlyPolymorphic(br, _) => write!(
                f,
                "expected concrete lifetime, found bound lifetime parameter{}",
                br_string(br)
            ),
            RegionsPlaceholderMismatch => write!(f, "one type is more general than the other"),
            Sorts(values) => ty::tls::with(|tcx| {
                report_maybe_different(
                    f,
                    &values.expected.sort_string(tcx),
                    &values.found.sort_string(tcx),
                )
            }),
            Traits(values) => ty::tls::with(|tcx| {
                report_maybe_different(
                    f,
                    &format!("trait `{}`", tcx.def_path_str(values.expected)),
                    &format!("trait `{}`", tcx.def_path_str(values.found)),
                )
            }),
            IntMismatch(ref values) => {
                write!(f, "expected `{:?}`, found `{:?}`", values.expected, values.found)
            }
            FloatMismatch(ref values) => {
                write!(f, "expected `{:?}`, found `{:?}`", values.expected, values.found)
            }
            VariadicMismatch(ref values) => write!(
                f,
                "expected {} fn, found {} function",
                if values.expected { "variadic" } else { "non-variadic" },
                if values.found { "variadic" } else { "non-variadic" }
            ),
            ProjectionMismatched(ref values) => ty::tls::with(|tcx| {
                write!(
                    f,
                    "expected {}, found {}",
                    tcx.def_path_str(values.expected),
                    tcx.def_path_str(values.found)
                )
            }),
            ExistentialMismatch(ref values) => report_maybe_different(
                f,
                &format!("trait `{}`", values.expected),
                &format!("trait `{}`", values.found),
            ),
            ConstMismatch(ref values) => {
                write!(f, "expected `{}`, found `{}`", values.expected, values.found)
            }
            IntrinsicCast => write!(f, "cannot coerce intrinsics to function pointers"),
            TargetFeatureCast(_) => write!(
                f,
                "cannot coerce functions with `#[target_feature]` to safe function pointers"
            ),
            ObjectUnsafeCoercion(_) => write!(f, "coercion to object-unsafe trait object"),
        }
    }
}

impl<'tcx> TypeError<'tcx> {
    pub fn must_include_note(&self) -> bool {
        use self::TypeError::*;
        match self {
            CyclicTy(_) | CyclicConst(_) | UnsafetyMismatch(_) | Mismatch | AbiMismatch(_)
            | FixedArraySize(_) | Sorts(_) | IntMismatch(_) | FloatMismatch(_)
            | VariadicMismatch(_) | TargetFeatureCast(_) => false,

            Mutability
            | TupleSize(_)
            | ArgCount
            | RegionsDoesNotOutlive(..)
            | RegionsInsufficientlyPolymorphic(..)
            | RegionsOverlyPolymorphic(..)
            | RegionsPlaceholderMismatch
            | Traits(_)
            | ProjectionMismatched(_)
            | ExistentialMismatch(_)
            | ConstMismatch(_)
            | IntrinsicCast
            | ObjectUnsafeCoercion(_) => true,
        }
    }
}

impl<'tcx> ty::TyS<'tcx> {
    pub fn sort_string(&self, tcx: TyCtxt<'_>) -> Cow<'static, str> {
        match *self.kind() {
            ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Str | ty::Never => {
                format!("`{}`", self).into()
            }
            ty::Tuple(ref tys) if tys.is_empty() => format!("`{}`", self).into(),

            ty::Adt(def, _) => format!("{} `{}`", def.descr(), tcx.def_path_str(def.did)).into(),
            ty::Foreign(def_id) => format!("extern type `{}`", tcx.def_path_str(def_id)).into(),
            ty::Array(t, n) => {
                let n = tcx.lift(n).unwrap();
                match n.try_eval_usize(tcx, ty::ParamEnv::empty()) {
                    _ if t.is_simple_ty() => format!("array `{}`", self).into(),
                    Some(n) => format!("array of {} element{}", n, pluralize!(n)).into(),
                    None => "array".into(),
                }
            }
            ty::Slice(ty) if ty.is_simple_ty() => format!("slice `{}`", self).into(),
            ty::Slice(_) => "slice".into(),
            ty::RawPtr(_) => "*-ptr".into(),
            ty::Ref(_, ty, mutbl) => {
                let tymut = ty::TypeAndMut { ty, mutbl };
                let tymut_string = tymut.to_string();
                if tymut_string != "_"
                    && (ty.is_simple_text() || tymut_string.len() < "mutable reference".len())
                {
                    format!("`&{}`", tymut_string).into()
                } else {
                    // Unknown type name, it's long or has type arguments
                    match mutbl {
                        hir::Mutability::Mut => "mutable reference",
                        _ => "reference",
                    }
                    .into()
                }
            }
            ty::FnDef(..) => "fn item".into(),
            ty::FnPtr(_) => "fn pointer".into(),
            ty::Dynamic(ref inner, ..) => {
                if let Some(principal) = inner.principal() {
                    format!("trait object `dyn {}`", tcx.def_path_str(principal.def_id())).into()
                } else {
                    "trait object".into()
                }
            }
            ty::Closure(..) => "closure".into(),
            ty::Generator(..) => "generator".into(),
            ty::GeneratorWitness(..) => "generator witness".into(),
            ty::Tuple(..) => "tuple".into(),
            ty::Infer(ty::TyVar(_)) => "inferred type".into(),
            ty::Infer(ty::IntVar(_)) => "integer".into(),
            ty::Infer(ty::FloatVar(_)) => "floating-point number".into(),
            ty::Placeholder(..) => "placeholder type".into(),
            ty::Bound(..) => "bound type".into(),
            ty::Infer(ty::FreshTy(_)) => "fresh type".into(),
            ty::Infer(ty::FreshIntTy(_)) => "fresh integral type".into(),
            ty::Infer(ty::FreshFloatTy(_)) => "fresh floating-point type".into(),
            ty::Projection(_) => "associated type".into(),
            ty::Param(p) => format!("type parameter `{}`", p).into(),
            ty::Opaque(..) => "opaque type".into(),
            ty::Error(_) => "type error".into(),
        }
    }

    pub fn prefix_string(&self) -> Cow<'static, str> {
        match *self.kind() {
            ty::Infer(_)
            | ty::Error(_)
            | ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Never => "type".into(),
            ty::Tuple(ref tys) if tys.is_empty() => "unit type".into(),
            ty::Adt(def, _) => def.descr().into(),
            ty::Foreign(_) => "extern type".into(),
            ty::Array(..) => "array".into(),
            ty::Slice(_) => "slice".into(),
            ty::RawPtr(_) => "raw pointer".into(),
            ty::Ref(.., mutbl) => match mutbl {
                hir::Mutability::Mut => "mutable reference",
                _ => "reference",
            }
            .into(),
            ty::FnDef(..) => "fn item".into(),
            ty::FnPtr(_) => "fn pointer".into(),
            ty::Dynamic(..) => "trait object".into(),
            ty::Closure(..) => "closure".into(),
            ty::Generator(..) => "generator".into(),
            ty::GeneratorWitness(..) => "generator witness".into(),
            ty::Tuple(..) => "tuple".into(),
            ty::Placeholder(..) => "higher-ranked type".into(),
            ty::Bound(..) => "bound type variable".into(),
            ty::Projection(_) => "associated type".into(),
            ty::Param(_) => "type parameter".into(),
            ty::Opaque(..) => "opaque type".into(),
        }
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn note_and_explain_type_err(
        self,
        db: &mut DiagnosticBuilder<'_>,
        err: &TypeError<'tcx>,
        cause: &ObligationCause<'tcx>,
        sp: Span,
        body_owner_def_id: DefId,
    ) {
        use self::TypeError::*;
        debug!("note_and_explain_type_err err={:?} cause={:?}", err, cause);
        match err {
            Sorts(values) => {
                match (values.expected.kind(), values.found.kind()) {
                    (ty::Closure(..), ty::Closure(..)) => {
                        db.note("no two closures, even if identical, have the same type");
                        db.help("consider boxing your closure and/or using it as a trait object");
                    }
                    (ty::Opaque(..), ty::Opaque(..)) => {
                        // Issue #63167
                        db.note("distinct uses of `impl Trait` result in different opaque types");
                    }
                    (ty::Float(_), ty::Infer(ty::IntVar(_))) => {
                        if let Ok(
                            // Issue #53280
                            snippet,
                        ) = self.sess.source_map().span_to_snippet(sp)
                        {
                            if snippet.chars().all(|c| c.is_digit(10) || c == '-' || c == '_') {
                                db.span_suggestion(
                                    sp,
                                    "use a float literal",
                                    format!("{}.0", snippet),
                                    MachineApplicable,
                                );
                            }
                        }
                    }
                    (ty::Param(expected), ty::Param(found)) => {
                        let generics = self.generics_of(body_owner_def_id);
                        let e_span = self.def_span(generics.type_param(expected, self).def_id);
                        if !sp.contains(e_span) {
                            db.span_label(e_span, "expected type parameter");
                        }
                        let f_span = self.def_span(generics.type_param(found, self).def_id);
                        if !sp.contains(f_span) {
                            db.span_label(f_span, "found type parameter");
                        }
                        db.note(
                            "a type parameter was expected, but a different one was found; \
                             you might be missing a type parameter or trait bound",
                        );
                        db.note(
                            "for more information, visit \
                             https://doc.rust-lang.org/book/ch10-02-traits.html\
                             #traits-as-parameters",
                        );
                    }
                    (ty::Projection(_), ty::Projection(_)) => {
                        db.note("an associated type was expected, but a different one was found");
                    }
                    (ty::Param(p), ty::Projection(proj)) | (ty::Projection(proj), ty::Param(p)) => {
                        let generics = self.generics_of(body_owner_def_id);
                        let p_span = self.def_span(generics.type_param(p, self).def_id);
                        if !sp.contains(p_span) {
                            db.span_label(p_span, "this type parameter");
                        }
                        let hir = self.hir();
                        let mut note = true;
                        if let Some(generics) = generics
                            .type_param(p, self)
                            .def_id
                            .as_local()
                            .map(|id| hir.local_def_id_to_hir_id(id))
                            .and_then(|id| self.hir().find(self.hir().get_parent_node(id)))
                            .as_ref()
                            .and_then(|node| node.generics())
                        {
                            // Synthesize the associated type restriction `Add<Output = Expected>`.
                            // FIXME: extract this logic for use in other diagnostics.
                            let trait_ref = proj.trait_ref(self);
                            let path =
                                self.def_path_str_with_substs(trait_ref.def_id, trait_ref.substs);
                            let item_name = self.item_name(proj.item_def_id);
                            let path = if path.ends_with('>') {
                                format!("{}, {} = {}>", &path[..path.len() - 1], item_name, p)
                            } else {
                                format!("{}<{} = {}>", path, item_name, p)
                            };
                            note = !suggest_constraining_type_param(
                                self,
                                generics,
                                db,
                                &format!("{}", proj.self_ty()),
                                &path,
                                None,
                            );
                        }
                        if note {
                            db.note("you might be missing a type parameter or trait bound");
                        }
                    }
                    (ty::Param(p), ty::Dynamic(..) | ty::Opaque(..))
                    | (ty::Dynamic(..) | ty::Opaque(..), ty::Param(p)) => {
                        let generics = self.generics_of(body_owner_def_id);
                        let p_span = self.def_span(generics.type_param(p, self).def_id);
                        if !sp.contains(p_span) {
                            db.span_label(p_span, "this type parameter");
                        }
                        db.help("type parameters must be constrained to match other types");
                        if self.sess.teach(&db.get_code().unwrap()) {
                            db.help(
                                "given a type parameter `T` and a method `foo`:
```
trait Trait<T> { fn foo(&self) -> T; }
```
the only ways to implement method `foo` are:
- constrain `T` with an explicit type:
```
impl Trait<String> for X {
    fn foo(&self) -> String { String::new() }
}
```
- add a trait bound to `T` and call a method on that trait that returns `Self`:
```
impl<T: std::default::Default> Trait<T> for X {
    fn foo(&self) -> T { <T as std::default::Default>::default() }
}
```
- change `foo` to return an argument of type `T`:
```
impl<T> Trait<T> for X {
    fn foo(&self, x: T) -> T { x }
}
```",
                            );
                        }
                        db.note(
                            "for more information, visit \
                             https://doc.rust-lang.org/book/ch10-02-traits.html\
                             #traits-as-parameters",
                        );
                    }
                    (ty::Param(p), ty::Closure(..) | ty::Generator(..)) => {
                        let generics = self.generics_of(body_owner_def_id);
                        let p_span = self.def_span(generics.type_param(p, self).def_id);
                        if !sp.contains(p_span) {
                            db.span_label(p_span, "this type parameter");
                        }
                        db.help(&format!(
                            "every closure has a distinct type and so could not always match the \
                             caller-chosen type of parameter `{}`",
                            p
                        ));
                    }
                    (ty::Param(p), _) | (_, ty::Param(p)) => {
                        let generics = self.generics_of(body_owner_def_id);
                        let p_span = self.def_span(generics.type_param(p, self).def_id);
                        if !sp.contains(p_span) {
                            db.span_label(p_span, "this type parameter");
                        }
                    }
                    (ty::Projection(proj_ty), _) => {
                        self.expected_projection(
                            db,
                            proj_ty,
                            values,
                            body_owner_def_id,
                            &cause.code,
                        );
                    }
                    (_, ty::Projection(proj_ty)) => {
                        let msg = format!(
                            "consider constraining the associated type `{}` to `{}`",
                            values.found, values.expected,
                        );
                        if !self.suggest_constraint(
                            db,
                            &msg,
                            body_owner_def_id,
                            proj_ty,
                            values.expected,
                        ) {
                            db.help(&msg);
                            db.note(
                                "for more information, visit \
                                https://doc.rust-lang.org/book/ch19-03-advanced-traits.html",
                            );
                        }
                    }
                    _ => {}
                }
                debug!(
                    "note_and_explain_type_err expected={:?} ({:?}) found={:?} ({:?})",
                    values.expected,
                    values.expected.kind(),
                    values.found,
                    values.found.kind(),
                );
            }
            CyclicTy(ty) => {
                // Watch out for various cases of cyclic types and try to explain.
                if ty.is_closure() || ty.is_generator() {
                    db.note(
                        "closures cannot capture themselves or take themselves as argument;\n\
                         this error may be the result of a recent compiler bug-fix,\n\
                         see issue #46062 <https://github.com/rust-lang/rust/issues/46062>\n\
                         for more information",
                    );
                }
            }
            TargetFeatureCast(def_id) => {
                let attrs = self.get_attrs(*def_id);
                let target_spans = attrs
                    .deref()
                    .iter()
                    .filter(|attr| attr.has_name(sym::target_feature))
                    .map(|attr| attr.span);
                db.note(
                    "functions with `#[target_feature]` can only be coerced to `unsafe` function pointers"
                );
                db.span_labels(target_spans, "`#[target_feature]` added here");
            }
            _ => {}
        }
    }

    fn suggest_constraint(
        self,
        db: &mut DiagnosticBuilder<'_>,
        msg: &str,
        body_owner_def_id: DefId,
        proj_ty: &ty::ProjectionTy<'tcx>,
        ty: Ty<'tcx>,
    ) -> bool {
        let assoc = self.associated_item(proj_ty.item_def_id);
        let trait_ref = proj_ty.trait_ref(self);
        if let Some(item) = self.hir().get_if_local(body_owner_def_id) {
            if let Some(hir_generics) = item.generics() {
                // Get the `DefId` for the type parameter corresponding to `A` in `<A as T>::Foo`.
                // This will also work for `impl Trait`.
                let def_id = if let ty::Param(param_ty) = proj_ty.self_ty().kind() {
                    let generics = self.generics_of(body_owner_def_id);
                    generics.type_param(param_ty, self).def_id
                } else {
                    return false;
                };

                // First look in the `where` clause, as this might be
                // `fn foo<T>(x: T) where T: Trait`.
                for predicate in hir_generics.where_clause.predicates {
                    if let hir::WherePredicate::BoundPredicate(pred) = predicate {
                        if let hir::TyKind::Path(hir::QPath::Resolved(None, path)) =
                            pred.bounded_ty.kind
                        {
                            if path.res.opt_def_id() == Some(def_id) {
                                // This predicate is binding type param `A` in `<A as T>::Foo` to
                                // something, potentially `T`.
                            } else {
                                continue;
                            }
                        } else {
                            continue;
                        }

                        if self.constrain_generic_bound_associated_type_structured_suggestion(
                            db,
                            &trait_ref,
                            pred.bounds,
                            &assoc,
                            ty,
                            msg,
                        ) {
                            return true;
                        }
                    }
                }
                for param in hir_generics.params {
                    if self.hir().opt_local_def_id(param.hir_id).map(|id| id.to_def_id())
                        == Some(def_id)
                    {
                        // This is type param `A` in `<A as T>::Foo`.
                        return self.constrain_generic_bound_associated_type_structured_suggestion(
                            db,
                            &trait_ref,
                            param.bounds,
                            &assoc,
                            ty,
                            msg,
                        );
                    }
                }
            }
        }
        false
    }

    /// An associated type was expected and a different type was found.
    ///
    /// We perform a few different checks to see what we can suggest:
    ///
    ///  - In the current item, look for associated functions that return the expected type and
    ///    suggest calling them. (Not a structured suggestion.)
    ///  - If any of the item's generic bounds can be constrained, we suggest constraining the
    ///    associated type to the found type.
    ///  - If the associated type has a default type and was expected inside of a `trait`, we
    ///    mention that this is disallowed.
    ///  - If all other things fail, and the error is not because of a mismatch between the `trait`
    ///    and the `impl`, we provide a generic `help` to constrain the assoc type or call an assoc
    ///    fn that returns the type.
    fn expected_projection(
        self,
        db: &mut DiagnosticBuilder<'_>,
        proj_ty: &ty::ProjectionTy<'tcx>,
        values: &ExpectedFound<Ty<'tcx>>,
        body_owner_def_id: DefId,
        cause_code: &ObligationCauseCode<'_>,
    ) {
        let msg = format!(
            "consider constraining the associated type `{}` to `{}`",
            values.expected, values.found
        );
        let body_owner = self.hir().get_if_local(body_owner_def_id);
        let current_method_ident = body_owner.and_then(|n| n.ident()).map(|i| i.name);

        // We don't want to suggest calling an assoc fn in a scope where that isn't feasible.
        let callable_scope = match body_owner {
            Some(
                hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(..), .. })
                | hir::Node::TraitItem(hir::TraitItem { kind: hir::TraitItemKind::Fn(..), .. })
                | hir::Node::ImplItem(hir::ImplItem { kind: hir::ImplItemKind::Fn(..), .. }),
            ) => true,
            _ => false,
        };
        let impl_comparison = matches!(
            cause_code,
            ObligationCauseCode::CompareImplMethodObligation { .. }
                | ObligationCauseCode::CompareImplTypeObligation { .. }
                | ObligationCauseCode::CompareImplConstObligation
        );
        let assoc = self.associated_item(proj_ty.item_def_id);
        if !callable_scope || impl_comparison {
            // We do not want to suggest calling functions when the reason of the
            // type error is a comparison of an `impl` with its `trait` or when the
            // scope is outside of a `Body`.
        } else {
            // If we find a suitable associated function that returns the expected type, we don't
            // want the more general suggestion later in this method about "consider constraining
            // the associated type or calling a method that returns the associated type".
            let point_at_assoc_fn = self.point_at_methods_that_satisfy_associated_type(
                db,
                assoc.container.id(),
                current_method_ident,
                proj_ty.item_def_id,
                values.expected,
            );
            // Possibly suggest constraining the associated type to conform to the
            // found type.
            if self.suggest_constraint(db, &msg, body_owner_def_id, proj_ty, values.found)
                || point_at_assoc_fn
            {
                return;
            }
        }

        if let ty::Opaque(def_id, _) = *proj_ty.self_ty().kind() {
            // When the expected `impl Trait` is not defined in the current item, it will come from
            // a return type. This can occur when dealing with `TryStream` (#71035).
            if self.constrain_associated_type_structured_suggestion(
                db,
                self.def_span(def_id),
                &assoc,
                values.found,
                &msg,
            ) {
                return;
            }
        }

        if self.point_at_associated_type(db, body_owner_def_id, values.found) {
            return;
        }

        if !impl_comparison {
            // Generic suggestion when we can't be more specific.
            if callable_scope {
                db.help(&format!("{} or calling a method that returns `{}`", msg, values.expected));
            } else {
                db.help(&msg);
            }
            db.note(
                "for more information, visit \
                 https://doc.rust-lang.org/book/ch19-03-advanced-traits.html",
            );
        }
        if self.sess.teach(&db.get_code().unwrap()) {
            db.help(
                "given an associated type `T` and a method `foo`:
```
trait Trait {
type T;
fn foo(&self) -> Self::T;
}
```
the only way of implementing method `foo` is to constrain `T` with an explicit associated type:
```
impl Trait for X {
type T = String;
fn foo(&self) -> Self::T { String::new() }
}
```",
            );
        }
    }

    fn point_at_methods_that_satisfy_associated_type(
        self,
        db: &mut DiagnosticBuilder<'_>,
        assoc_container_id: DefId,
        current_method_ident: Option<Symbol>,
        proj_ty_item_def_id: DefId,
        expected: Ty<'tcx>,
    ) -> bool {
        let items = self.associated_items(assoc_container_id);
        // Find all the methods in the trait that could be called to construct the
        // expected associated type.
        // FIXME: consider suggesting the use of associated `const`s.
        let methods: Vec<(Span, String)> = items
            .items
            .iter()
            .filter(|(name, item)| {
                ty::AssocKind::Fn == item.kind && Some(**name) != current_method_ident
            })
            .filter_map(|(_, item)| {
                let method = self.fn_sig(item.def_id);
                match *method.output().skip_binder().kind() {
                    ty::Projection(ty::ProjectionTy { item_def_id, .. })
                        if item_def_id == proj_ty_item_def_id =>
                    {
                        Some((
                            self.sess.source_map().guess_head_span(self.def_span(item.def_id)),
                            format!("consider calling `{}`", self.def_path_str(item.def_id)),
                        ))
                    }
                    _ => None,
                }
            })
            .collect();
        if !methods.is_empty() {
            // Use a single `help:` to show all the methods in the trait that can
            // be used to construct the expected associated type.
            let mut span: MultiSpan =
                methods.iter().map(|(sp, _)| *sp).collect::<Vec<Span>>().into();
            let msg = format!(
                "{some} method{s} {are} available that return{r} `{ty}`",
                some = if methods.len() == 1 { "a" } else { "some" },
                s = pluralize!(methods.len()),
                are = if methods.len() == 1 { "is" } else { "are" },
                r = if methods.len() == 1 { "s" } else { "" },
                ty = expected
            );
            for (sp, label) in methods.into_iter() {
                span.push_span_label(sp, label);
            }
            db.span_help(span, &msg);
            return true;
        }
        false
    }

    fn point_at_associated_type(
        self,
        db: &mut DiagnosticBuilder<'_>,
        body_owner_def_id: DefId,
        found: Ty<'tcx>,
    ) -> bool {
        let hir_id =
            match body_owner_def_id.as_local().map(|id| self.hir().local_def_id_to_hir_id(id)) {
                Some(hir_id) => hir_id,
                None => return false,
            };
        // When `body_owner` is an `impl` or `trait` item, look in its associated types for
        // `expected` and point at it.
        let parent_id = self.hir().get_parent_item(hir_id);
        let item = self.hir().find(parent_id);
        debug!("expected_projection parent item {:?}", item);
        match item {
            Some(hir::Node::Item(hir::Item { kind: hir::ItemKind::Trait(.., items), .. })) => {
                // FIXME: account for `#![feature(specialization)]`
                for item in &items[..] {
                    match item.kind {
                        hir::AssocItemKind::Type => {
                            // FIXME: account for returning some type in a trait fn impl that has
                            // an assoc type as a return type (#72076).
                            if let hir::Defaultness::Default { has_value: true } = item.defaultness
                            {
                                if self.type_of(self.hir().local_def_id(item.id.hir_id)) == found {
                                    db.span_label(
                                        item.span,
                                        "associated type defaults can't be assumed inside the \
                                            trait defining them",
                                    );
                                    return true;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some(hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Impl { items, .. }, ..
            })) => {
                for item in &items[..] {
                    if let hir::AssocItemKind::Type = item.kind {
                        if self.type_of(self.hir().local_def_id(item.id.hir_id)) == found {
                            db.span_label(item.span, "expected this associated type");
                            return true;
                        }
                    }
                }
            }
            _ => {}
        }
        false
    }

    /// Given a slice of `hir::GenericBound`s, if any of them corresponds to the `trait_ref`
    /// requirement, provide a strucuted suggestion to constrain it to a given type `ty`.
    fn constrain_generic_bound_associated_type_structured_suggestion(
        self,
        db: &mut DiagnosticBuilder<'_>,
        trait_ref: &ty::TraitRef<'tcx>,
        bounds: hir::GenericBounds<'_>,
        assoc: &ty::AssocItem,
        ty: Ty<'tcx>,
        msg: &str,
    ) -> bool {
        // FIXME: we would want to call `resolve_vars_if_possible` on `ty` before suggesting.
        bounds.iter().any(|bound| match bound {
            hir::GenericBound::Trait(ptr, hir::TraitBoundModifier::None) => {
                // Relate the type param against `T` in `<A as T>::Foo`.
                ptr.trait_ref.trait_def_id() == Some(trait_ref.def_id)
                    && self.constrain_associated_type_structured_suggestion(
                        db, ptr.span, assoc, ty, msg,
                    )
            }
            _ => false,
        })
    }

    /// Given a span corresponding to a bound, provide a structured suggestion to set an
    /// associated type to a given type `ty`.
    fn constrain_associated_type_structured_suggestion(
        self,
        db: &mut DiagnosticBuilder<'_>,
        span: Span,
        assoc: &ty::AssocItem,
        ty: Ty<'tcx>,
        msg: &str,
    ) -> bool {
        if let Ok(has_params) =
            self.sess.source_map().span_to_snippet(span).map(|snippet| snippet.ends_with('>'))
        {
            let (span, sugg) = if has_params {
                let pos = span.hi() - BytePos(1);
                let span = Span::new(pos, pos, span.ctxt());
                (span, format!(", {} = {}", assoc.ident, ty))
            } else {
                (span.shrink_to_hi(), format!("<{} = {}>", assoc.ident, ty))
            };
            db.span_suggestion_verbose(span, msg, sugg, MaybeIncorrect);
            return true;
        }
        false
    }
}
