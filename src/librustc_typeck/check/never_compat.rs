use crate::check::{FnCtxt, InferredPath};
use crate::rustc::ty::TypeFoldable;
use rustc::infer::type_variable::TypeVariableOriginKind;
use rustc::infer::InferCtxt;
use rustc::ty;
use rustc::ty::fold::TypeFolder;
use rustc::ty::{Ty, TyCtxt, TyVid};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::HirId;
use std::borrow::Cow;

/// Code to detect cases where using `!` (never-type) fallback instead of `()` fallback
/// may result in the introduction of undefined behavior;
///
/// TL;DR: We look for method calls whose arguments are all inhabited, but
/// that have a generic parameter (e.g. `T` in `fn foo<T>() { ... } `)
/// that's uninhabited due to fallback. We emit an error, forcing
/// users to add explicit type annotations indicating whether they
/// actually want the `!` type, or something else.
///
/// Background: When we first tried to enable never-type fallback, it was found to
/// cause code in the 'obj' crate to segfault. The root cause was a piece
/// of code that looked like this:
///
/// ```rust
///fn unconstrained_return<T>() -> Result<T, String> {
///     let ffi: fn() -> T = transmute(some_pointer);
///     Ok(ffi())
/// }
///fn foo() {
///     match unconstrained_return::<_>() {
///         Ok(x) => x,  // `x` has type `_`, which is unconstrained
///         Err(s) => panic!(s),  // … except for unifying with the type of `panic!()`
///         // so that both `match` arms have the same type.
///         // Therefore `_` resolves to `!` and we "return" an `Ok(!)` value.
///     };
///}
/// ```
///
/// Previously, the return type of `panic!()` would end up as `()` due to fallback
/// (even though `panic()` is declared as returning `!`). This meant that
/// the function pointer would get transmuted to `fn() -> ()`.
///
/// With never-type fallback enabled, the function pointer was transmuted
/// to `fn() -> !`, despite the fact that it actually returned. This lead
/// to undefined behavior, since we actually "produced" an instance of `!`
/// at runtime.
///
/// We want to prevent this code from compiling, unless the user provides
/// explicit type annotations indicating that they want the `!` type to be used.
///
/// However, we want code like this to continue working:
///
/// ```rust
/// fn foo(e: !) -> Box<dyn std::error::Error> {
///    Box::<_>::new(e)
/// }```
///
/// From the perspective of fallback, these two snippets of code are very
/// similar. We have some expression that starts out with type `!`, which
/// we replace with a divergent inference variable. When fallback runs,
/// we unify the inference variable with `!`, which leads to other
/// expressions getting a type of `!` as well (`Box<!>` in this case),
/// becoming uninhabited as well.
///
/// Our task is to distinguish things that are "legitimately" uninhabited,
/// (such as `Box<!>` in the second example), from things that are
/// "suspiciously" uninhabited and may lead to undefined behavior
/// (the `Result<!, !>` in the first example).
///
/// The key difference between these two snippets of code turns out
/// to be the *arguments* used in a function call. In the first example,
/// we call `unconstrained_return` with no arguments, and an inferred
/// type parameter of `!` (a.k.a `unconstrained_return::<!>()`).
///
/// In the second example, we call `Box::new(!)`, again inferring
/// the type parameter to `!`. Since at least one argument to `Box::new`,
/// is uninhabited, we know that this call is *statically unreachable* -
/// if it were ever executed, we would have instantaited an uninhabited
/// type for a parameter, which is impossible. This means that while
/// the fallback to `!` has affected the type-checking of this call,
/// it won't actually affect the runtime behavior of the call, since
/// the call can never be executed.
///
/// In the first example, the call to `unconstrained_return::<!>()` is
/// still (potentially) live, since it trivially has no uninhabited arguments
/// (it has no arguments at all). This means that our fallback to `!` may
/// have changed the runtime behavior of `unconstrained_return` (in this case,
/// it did) - we may actually execute the call at runtime, but with an uninhabited
/// type parameter that the user did not intend to provide.
///
/// This distinction forms the basis for our lint. We look for
/// any function/method calls that meet the following requirements:
///
/// * Unresolved inference variables are present in the generic substs prior to fallback
/// * All arguments are inhabited
/// * After fallback, at least one generic subsst is uninhabited.
///
/// Let's break down each of these requirements:
///
/// 1. We only look at method calls, not other expressions.
/// This is because the issue boils down to a generic subst being
/// uninhabited due to fallback. Generic parameters can only be used
/// in two places - as arguments to types, and arguments to methods.
/// We don't care about types - the only relevant expression would be
/// a constructor (e.g. `MyStruct { a: true, b: 25 }`). In and of itself,
/// constructing a type this way (after all field values are evaluated)
/// can never cause undefined behavior.
///
/// Methods calls, on the other hand, might cause undefined behavior
/// due to how they use their generic paramters. Note that this
/// still true if the undefineed behavior occurs in the same function
/// where fallback occurs - there still must be a call `transmute`
/// or some other intrinsic that allows 'instantating' an `!` instance
/// somehow.
///
/// 2. Only function calls with inference variables in their generic
/// substs can possibly be affected by fallback - if all types are resolved,
/// then fallback cannot possibly have any effect.
///
/// 3. If any arguments are uninhabited, then the function call cannot ever
/// cause undefined behavior, since it can never be executed. This check
/// ensures that we don't lint on `Box::<_>::new(!)` - even though
/// we're inferring a generic subst (`_`) to `!`, it has no runtime
/// effect.
///
/// 4. If a generic substs is uninhabited after fallback, then we have
/// the potential for undefined behavior. Note that this check can
/// have false positives - if a generic argument was already uninhabited
/// prior to fallback, we might end up triggering the lint even if fallback
/// didn't cause any generic substs to become uninhabited. However, this isn't
/// a big deal for a few reasons:
///
/// * This lint should be incredibly rare (as of the time of writing, only
/// `obj` has run into the relevant issue).
/// * Even if fallback didn't cause anything to become uninhabited
/// (e.g. a `*const _` being inferred to `*const !`), it's still influencing
/// the generic substs of a potentially live (all arguments inhabited) function
/// call. This is arguably something worth linting against, since the user
/// might not be aware that a live call is being affected by a seemingly
/// unrelated expression.
///
/// * It should always be possible to resolve this lint while keeping the
/// existing behavior, simply by adding explicit type annotations. This
/// ensures that fallback never runs (since the type variable will
/// actually be constrained to `!` or something else). At worst,
/// we will force users to write some arguably unecessary type annotation,
/// but we will not permanently break any code.

/// The main interface to this module.
/// It exposes two hooks: `pre_fallback` and `post_fallback`,
/// which are invoked by `typeck_tables_of_with_fallback` respectively before
/// and after fallback is run.
pub struct NeverCompatHandler<'tcx> {
    /// A map from method call `HirId`'s to `InferredPaths`.
    /// This is computed from `FnCtxt.inferred_paths`, which
    /// is in turn populated during typecheck.
    unresolved_paths: FxHashMap<HirId, InferredPath<'tcx>>,
    /// All divering type variables that were unconstrained
    /// prior to fallback. We use this to help generate
    /// better diagnostics by determining which
    /// inference variables were equated with a diverging
    /// fallback variable.
    unconstrained_diverging: Vec<Ty<'tcx>>,
}

/// A simpler helper to collect all `InferTy::TyVar`s found
/// in a `TypeFoldable`.
struct TyVarFinder<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    vars: Vec<Ty<'tcx>>,
}
impl<'a, 'tcx> TypeFolder<'tcx> for TyVarFinder<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Infer(ty::InferTy::TyVar(_)) = t.kind {
            self.vars.push(t);
        }
        t.super_fold_with(self)
    }
}

/// The core logic of this check. Given
/// an `InferredPath`, we determine
/// if this method call is "questionable" using the criteria
/// described at the top of the module.
///
/// If the method call is "questionable", and should
/// be linted, we return Ok(tys), where `tys`
/// is a list of all inference variables involved
/// with the generic paramter affected by fallback.
///
/// Otherwise, we return None
fn find_questionable_call<'a, 'tcx>(
    path: &'a InferredPath<'tcx>,
    fcx: &FnCtxt<'a, 'tcx>,
) -> Option<&'a [Ty<'tcx>]> {
    let tcx = fcx.tcx;
    let ty = fcx.infcx.resolve_vars_if_possible(&path.ty);
    debug!("find_questionable_call: Fully resolved ty: {:?}", ty);

    let ty = ty.unwrap_or_else(|| bug!("Missing ty in path: {:?}", path));

    // Check that this InferredPath actually corresponds to a method
    // invocation. Currently, type-checking resolves generic paths
    // (e.g. `Box::<_>::new` separately from determining that a method call
    // is occurring). We use this check to skip over `InferredPaths` for
    // non-method-calls (e.g. struct constructors).
    if let ty::FnDef(did, substs) = ty.kind {
        debug!("find_questionable_call: Got substs: {:?}", substs);

        if tcx.is_constructor(did) {
            debug!("find_questionable_call: found constructor {:?}, bailing out", did);
            return None;
        }

        // See if we can find any arguments that are definitely uninhabited.
        // If we can, we're done - the method call is dead, so fallback
        // cannot affect its runtime behavior.
        for arg in &**path.args.as_ref().unwrap() {
            let resolved_arg = fcx.infcx.resolve_vars_if_possible(arg);

            // We use `conservative_is_privately_uninhabited` so that we only
            // bail out if we're certain that a type is uninhabited.
            if resolved_arg.conservative_is_privately_uninhabited(tcx) {
                debug!(
                    "find_questionable_call: bailing out due to uninhabited arg: {:?}",
                    resolved_arg
                );
                return None;
            } else {
                debug!("find_questionable_call: Arg is inhabited: {:?}", resolved_arg);
            }
        }

        // Now, check the generic substs for the method (e.g. `T` and `V` in `foo::<T, V>()`.
        // If we find an uninhabited subst,
        for (subst_ty, vars) in substs.types().zip(path.unresolved_vars.iter()) {
            let resolved_subst = fcx.infcx.resolve_vars_if_possible(&subst_ty);
            // We use `is_ty_uninhabited_from_any_module` instead of `conservative_is_privately_uninhabited`.
            // Using a broader check for uninhabitedness ensures that we lint
            // whenever the subst is uninhabted, regardless of whether or not
            // the user code could know this.
            if tcx.is_ty_uninhabited_from_any_module(resolved_subst) {
                debug!("find_questionable_call: Subst is uninhabited: {:?}", resolved_subst);
                if !vars.is_empty() {
                    debug!("find_questionable_call: Found fallback vars: {:?}", vars);
                    debug!(
                        "find_questionable_call: All arguments are inhabited, at least one subst is not inhabited!"
                    );
                    // These variables were recorded prior to fallback runntime.
                    // When generating a diagnostic message, we will use these
                    // to determine which spans will we show to the user.
                    return Some(vars);
                } else {
                    debug!("find_questionable_call: No fallback vars")
                }
            } else {
                debug!("find_questionable_call: Subst is inhabited: {:?}", resolved_subst);
            }
        }
    }
    return None;
}

/// Holds the "best" type variables that we want
/// to use to generate a message for the user.
///
/// Note that these two variables are unified with each other,
/// and therefore represent the same type. However, they have different
/// origin information, which we can use to generate a nice error message.
struct VarData {
    /// A type variable that was actually used in an uninhabited
    /// generic subst (e.g. `foo::<_#0t>()` where `_#0t = !`)
    /// This is used to point at the span where the U.B potentially
    /// occurs
    best_var: TyVid,
    /// The diverging type variable that is ultimately responsible
    /// for the U.B. occuring. In the `obj` example, this would
    /// be the type variable corresponding the `panic!()` expression.
    best_diverging_var: TyVid,
}

impl<'tcx> NeverCompatHandler<'tcx> {
    /// The pre-fallback hook invoked by typecheck. We collect
    /// all unconstrained inference variables used in method call substs,
    /// so that we can check if any changes occur due to fallback.
    pub fn pre_fallback(fcx: &FnCtxt<'a, 'tcx>) -> NeverCompatHandler<'tcx> {
        let unresolved_paths: FxHashMap<HirId, InferredPath<'tcx>> = fcx
            .inferred_paths
            .borrow()
            .iter()
            .map(|(id, path)| (*id, path.clone()))
            .filter_map(|(hir_id, mut path)| {
                debug!("pre_fallback: inspecting path ({:?}, {:?})", hir_id, path);

                let ty_resolved = fcx.infcx.resolve_vars_if_possible(&path.ty);

                // We only care about method calls, not other uses of generic paths.
                let fn_substs = match ty_resolved {
                    Some(ty::TyS { kind: ty::FnDef(_, substs), .. }) => substs,
                    _ => {
                        debug!("pre_fallback: non-fn ty {:?}, skipping", ty_resolved);
                        return None;
                    }
                };

                let args_infer = match path.args.as_ref().unwrap() {
                    Cow::Borrowed(b) => b.iter().any(|ty| {
                        fcx.infcx
                            .unresolved_type_vars(&fcx.infcx.resolve_vars_if_possible(ty))
                            .is_some()
                    }),
                    Cow::Owned(o) => fcx
                        .infcx
                        .unresolved_type_vars(&fcx.infcx.resolve_vars_if_possible(o))
                        .is_some(),
                };

                if args_infer {
                    debug!(
                        "pre_fallback: skipping due to inference vars in fn {:?} args {:?}",
                        ty_resolved,
                        path.args.unwrap()
                    );
                    return None;
                }

                // Any method call with inference variables in its substs
                // could potentially be affected by fallback.
                if fcx.infcx.unresolved_type_vars(fn_substs).is_some() {
                    for subst in fn_substs.types() {
                        let mut finder = TyVarFinder { infcx: &fcx.infcx, vars: vec![] };
                        subst.fold_with(&mut finder);
                        path.unresolved_vars.push(finder.vars);
                    }

                    debug!(
                        "pre_fallback: unresolved vars in ty {:?} : {:?}",
                        ty_resolved, path.unresolved_vars
                    );

                    Some((hir_id, path))
                } else {
                    debug!("pre_fallback: all vars resolved in ty: {:?}", ty_resolved);
                    None
                }
            })
            .collect();

        // Collect all divering inference variables, so that we
        // can later compare them against other inference variables.
        let unconstrained_diverging: Vec<_> = fcx
            .unsolved_variables()
            .iter()
            .cloned()
            .filter(|ty| fcx.infcx.type_var_diverges(ty))
            .collect();

        NeverCompatHandler { unresolved_paths, unconstrained_diverging }
    }

    /// Finds the 'best' type variables to use for display to the user,
    /// given a list of the type variables originally used in the
    /// generic substs for a method.
    ///
    /// We look for a type variable that ended up unified with a diverging
    /// inference variable. This represents a place where fallback to a never
    /// type ended up affecting a live method call. We then return the
    /// inference variable, along with the corresponding divering inference
    /// variable that was unified with it
    ///
    /// Note that their be multiple such pairs of inference variables - howver,
    /// we only display one to the user, to avoid overwhelming them with information.
    fn find_best_vars(&self, fcx: &FnCtxt<'a, 'tcx>, vars: &[Ty<'tcx>]) -> VarData {
        for var in vars {
            for diverging_var in &self.unconstrained_diverging {
                match (&var.kind, &diverging_var.kind) {
                    (ty::Infer(ty::InferTy::TyVar(vid1)), ty::Infer(ty::InferTy::TyVar(vid2))) => {
                        if fcx.infcx.type_variables.borrow_mut().sub_unified(*vid1, *vid2) {
                            debug!(
                                "Type variable {:?} is equal to diverging var {:?}",
                                var, diverging_var
                            );

                            debug!(
                                "Var origin: {:?}",
                                fcx.infcx.type_variables.borrow().var_origin(*vid1)
                            );
                            return VarData { best_var: *vid1, best_diverging_var: *vid2 };
                        }
                    }
                    _ => bug!("Unexpected types: var={:?} diverging_var={:?}", var, diverging_var),
                }
            }
        }
        bug!("No vars were equated to diverging vars: {:?}", vars)
    }

    /// The hook used by typecheck after fallback has been run.
    /// We look for any "questionable" calls, generating diagnostic
    /// message for them.
    pub fn post_fallback(self, fcx: &FnCtxt<'a, 'tcx>) {
        let tcx = fcx.tcx;
        for (call_id, path) in &self.unresolved_paths {
            debug!(
                "post_fallback: resolved ty: {:?} at span {:?} : expr={:?} parent={:?} path={:?}",
                path.span,
                path.ty,
                tcx.hir().get(*call_id),
                tcx.hir().get(tcx.hir().get_parent_node(*call_id)),
                path
            );

            let span = path.span;
            if let Some(vars) = find_questionable_call(path, fcx) {
                let VarData { best_var, best_diverging_var } = self.find_best_vars(fcx, &vars);

                let var_origin = *fcx.infcx.type_variables.borrow().var_origin(best_var);
                let diverging_var_span =
                    fcx.infcx.type_variables.borrow().var_origin(best_diverging_var).span;

                let mut err = tcx
                    .sess
                    .struct_span_err(span, "Fallback to `!` may introduce undefined behavior");

                // There are two possible cases here:
                match var_origin.kind {
                    // 1. This inference variable was automatically generated due to the user
                    // not using the turbofish syntax. For example, `foo()` where foo is deinfed
                    // as `fn foo<T>() { ... }`.
                    // In this case, we point at the definition of the type variable (e.g. the `T`
                    // in `foo<T>`), to better explain to the user what's going on.
                    TypeVariableOriginKind::TypeParameterDefinition(name, did) => {
                        err.span_note(
                            var_origin.span,
                            &format!("the type parameter {} here was inferred to `!`", name),
                        );
                        if let Some(did) = did {
                            err.span_note(fcx.tcx.def_span(did), "(type parameter defined here)");
                        }
                    }
                    // The user wrote an explicit inference variable: e.g. `foo::<_>()`. We point
                    // directly at its span, since this is sufficient to explain to the user
                    // what's going on.
                    _ => {
                        err.span_note(var_origin.span, "the type here was inferred to `!`");
                    }
                }

                err.span_note(diverging_var_span, "... due to this expression evaluating to `!`")
                    .note("If you want the `!` type to be used here, add explicit type annotations")
                    .emit();
            }
        }
    }
}
