//! The outlives relation `T: 'a` or `'a: 'b`. This code frequently
//! refers to rules defined in RFC 1214 (`OutlivesFooBar`), so see that
//! RFC for reference.

use smallvec::{smallvec, SmallVec};
use tracing::debug;

use crate::data_structures::SsoHashSet;
use crate::inherent::*;
use crate::visit::TypeVisitableExt as _;
use crate::{self as ty, Interner};

#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
pub enum Component<I: Interner> {
    Region(I::Region),
    Param(I::ParamTy),
    Placeholder(I::PlaceholderTy),
    UnresolvedInferenceVariable(ty::InferTy),

    // Projections like `T::Foo` are tricky because a constraint like
    // `T::Foo: 'a` can be satisfied in so many ways. There may be a
    // where-clause that says `T::Foo: 'a`, or the defining trait may
    // include a bound like `type Foo: 'static`, or -- in the most
    // conservative way -- we can prove that `T: 'a` (more generally,
    // that all components in the projection outlive `'a`). This code
    // is not in a position to judge which is the best technique, so
    // we just product the projection as a component and leave it to
    // the consumer to decide (but see `EscapingProjection` below).
    Alias(ty::AliasTy<I>),

    // In the case where a projection has escaping regions -- meaning
    // regions bound within the type itself -- we always use
    // the most conservative rule, which requires that all components
    // outlive the bound. So for example if we had a type like this:
    //
    //     for<'a> Trait1<  <T as Trait2<'a,'b>>::Foo  >
    //                      ~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    // then the inner projection (underlined) has an escaping region
    // `'a`. We consider that outer trait `'c` to meet a bound if `'b`
    // outlives `'b: 'c`, and we don't consider whether the trait
    // declares that `Foo: 'static` etc. Therefore, we just return the
    // free components of such a projection (in this case, `'b`).
    //
    // However, in the future, we may want to get smarter, and
    // actually return a "higher-ranked projection" here. Therefore,
    // we mark that these components are part of an escaping
    // projection, so that implied bounds code can avoid relying on
    // them. This gives us room to improve the regionck reasoning in
    // the future without breaking backwards compat.
    EscapingAlias(Vec<Component<I>>),
}

/// Push onto `out` all the things that must outlive `'a` for the condition
/// `ty0: 'a` to hold. Note that `ty0` must be a **fully resolved type**.
pub fn push_outlives_components<I: Interner>(
    tcx: I,
    ty0: I::Ty,
    out: &mut SmallVec<[Component<I>; 4]>,
) {
    let mut visited = SsoHashSet::new();
    compute_components_for_ty(tcx, ty0, out, &mut visited);
    debug!("components({:?}) = {:?}", ty0, out);
}

fn compute_components_for_arg<I: Interner>(
    tcx: I,
    arg: I::GenericArg,
    out: &mut SmallVec<[Component<I>; 4]>,
    visited: &mut SsoHashSet<I::GenericArg>,
) {
    match arg.kind() {
        ty::GenericArgKind::Type(ty) => {
            compute_components_for_ty(tcx, ty, out, visited);
        }
        ty::GenericArgKind::Lifetime(lt) => {
            compute_components_for_lt(lt, out);
        }
        ty::GenericArgKind::Const(ct) => {
            compute_components_for_const(tcx, ct, out, visited);
        }
    }
}

fn compute_components_for_ty<I: Interner>(
    tcx: I,
    ty: I::Ty,
    out: &mut SmallVec<[Component<I>; 4]>,
    visited: &mut SsoHashSet<I::GenericArg>,
) {
    if !visited.insert(ty.into()) {
        return;
    }
    // Descend through the types, looking for the various "base"
    // components and collecting them into `out`. This is not written
    // with `collect()` because of the need to sometimes skip subtrees
    // in the `subtys` iterator (e.g., when encountering a
    // projection).
    match ty.kind() {
        ty::FnDef(_, args) => {
            // HACK(eddyb) ignore lifetimes found shallowly in `args`.
            // This is inconsistent with `ty::Adt` (including all args)
            // and with `ty::Closure` (ignoring all args other than
            // upvars, of which a `ty::FnDef` doesn't have any), but
            // consistent with previous (accidental) behavior.
            // See https://github.com/rust-lang/rust/issues/70917
            // for further background and discussion.
            for child in args.iter() {
                match child.kind() {
                    ty::GenericArgKind::Type(ty) => {
                        compute_components_for_ty(tcx, ty, out, visited);
                    }
                    ty::GenericArgKind::Lifetime(_) => {}
                    ty::GenericArgKind::Const(ct) => {
                        compute_components_for_const(tcx, ct, out, visited);
                    }
                }
            }
        }

        ty::Pat(element, _) | ty::Array(element, _) => {
            compute_components_for_ty(tcx, element, out, visited);
        }

        ty::Closure(_, args) => {
            let tupled_ty = args.as_closure().tupled_upvars_ty();
            compute_components_for_ty(tcx, tupled_ty, out, visited);
        }

        ty::CoroutineClosure(_, args) => {
            let tupled_ty = args.as_coroutine_closure().tupled_upvars_ty();
            compute_components_for_ty(tcx, tupled_ty, out, visited);
        }

        ty::Coroutine(_, args) => {
            // Same as the closure case
            let tupled_ty = args.as_coroutine().tupled_upvars_ty();
            compute_components_for_ty(tcx, tupled_ty, out, visited);

            // We ignore regions in the coroutine interior as we don't
            // want these to affect region inference
        }

        // All regions are bound inside a witness, and we don't emit
        // higher-ranked outlives components currently.
        ty::CoroutineWitness(..) => {},

        // OutlivesTypeParameterEnv -- the actual checking that `X:'a`
        // is implied by the environment is done in regionck.
        ty::Param(p) => {
            out.push(Component::Param(p));
        }

        ty::Placeholder(p) => {
            out.push(Component::Placeholder(p));
        }

        // For projections, we prefer to generate an obligation like
        // `<P0 as Trait<P1...Pn>>::Foo: 'a`, because this gives the
        // regionck more ways to prove that it holds. However,
        // regionck is not (at least currently) prepared to deal with
        // higher-ranked regions that may appear in the
        // trait-ref. Therefore, if we see any higher-ranked regions,
        // we simply fallback to the most restrictive rule, which
        // requires that `Pi: 'a` for all `i`.
        ty::Alias(_, alias_ty) => {
            if !alias_ty.has_escaping_bound_vars() {
                // best case: no escaping regions, so push the
                // projection and skip the subtree (thus generating no
                // constraints for Pi). This defers the choice between
                // the rules OutlivesProjectionEnv,
                // OutlivesProjectionTraitDef, and
                // OutlivesProjectionComponents to regionck.
                out.push(Component::Alias(alias_ty));
            } else {
                // fallback case: hard code
                // OutlivesProjectionComponents. Continue walking
                // through and constrain Pi.
                let mut subcomponents = smallvec![];
                let mut subvisited = SsoHashSet::new();
                compute_alias_components_recursive(tcx, ty, &mut subcomponents, &mut subvisited);
                out.push(Component::EscapingAlias(subcomponents.into_iter().collect()));
            }
        }

        // We assume that inference variables are fully resolved.
        // So, if we encounter an inference variable, just record
        // the unresolved variable as a component.
        ty::Infer(infer_ty) => {
            out.push(Component::UnresolvedInferenceVariable(infer_ty));
        }

        // Most types do not introduce any region binders, nor
        // involve any other subtle cases, and so the WF relation
        // simply constraints any regions referenced directly by
        // the type and then visits the types that are lexically
        // contained within. (The comments refer to relevant rules
        // from RFC1214.)

        ty::Bool |            // OutlivesScalar
        ty::Char |            // OutlivesScalar
        ty::Int(..) |         // OutlivesScalar
        ty::Uint(..) |        // OutlivesScalar
        ty::Float(..) |       // OutlivesScalar
        ty::Never |           // OutlivesScalar
        ty::Foreign(..) |     // OutlivesNominalType
        ty::Str |             // OutlivesScalar (ish)
        ty::Bound(..) |
        ty::Error(_) => {
            // Trivial.
        }

        // OutlivesNominalType
        ty::Adt(_, args) => {
            for arg in args.iter() {
                compute_components_for_arg(tcx, arg, out, visited);
            }
        }

        // OutlivesNominalType
        ty::Slice(ty) |
        ty::RawPtr(ty, _) => {
            compute_components_for_ty(tcx, ty, out, visited);
        }
        ty::Tuple(tys) => {
            for ty in tys.iter() {
                compute_components_for_ty(tcx, ty, out, visited);
            }
        }

        // OutlivesReference
        ty::Ref(lt, ty, _) => {
            compute_components_for_lt(lt, out);
            compute_components_for_ty(tcx, ty, out, visited);
        }

        ty::Dynamic(preds, lt, _) => {
            compute_components_for_lt(lt, out);
            for pred in preds.iter() {
                match pred.skip_binder() {
                    ty::ExistentialPredicate::Trait(tr) => {
                        for arg in tr.args.iter() {
                            compute_components_for_arg(tcx, arg, out, visited);
                        }
                    }
                    ty::ExistentialPredicate::Projection(proj) => {
                        for arg in proj.args.iter() {
                            compute_components_for_arg(tcx, arg, out, visited);
                        }
                        match proj.term.kind() {
                            ty::TermKind::Ty(ty) => {
                                compute_components_for_ty(tcx, ty, out, visited)
                            }
                            ty::TermKind::Const(ct) => {
                                compute_components_for_const(tcx, ct, out, visited)
                            }
                        }
                    }
                    ty::ExistentialPredicate::AutoTrait(..) => {}
                }
            }
        }

        ty::FnPtr(sig) => {
            for ty in sig.skip_binder().inputs_and_output.iter() {
                compute_components_for_ty(tcx, ty, out, visited);
            }
        }
    }
}

/// Collect [Component]s for *all* the args of `parent`.
///
/// This should not be used to get the components of `parent` itself.
/// Use [push_outlives_components] instead.
pub fn compute_alias_components_recursive<I: Interner>(
    tcx: I,
    alias_ty: I::Ty,
    out: &mut SmallVec<[Component<I>; 4]>,
    visited: &mut SsoHashSet<I::GenericArg>,
) {
    let ty::Alias(kind, alias_ty) = alias_ty.kind() else {
        unreachable!("can only call `compute_alias_components_recursive` on an alias type")
    };

    let opt_variances =
        if kind == ty::Opaque { Some(tcx.variances_of(alias_ty.def_id)) } else { None };

    for (index, child) in alias_ty.args.iter().enumerate() {
        if opt_variances.and_then(|variances| variances.get(index)) == Some(ty::Bivariant) {
            continue;
        }
        compute_components_for_arg(tcx, child, out, visited);
    }
}

fn compute_components_for_lt<I: Interner>(lt: I::Region, out: &mut SmallVec<[Component<I>; 4]>) {
    if !lt.is_bound() {
        out.push(Component::Region(lt));
    }
}

fn compute_components_for_const<I: Interner>(
    tcx: I,
    ct: I::Const,
    out: &mut SmallVec<[Component<I>; 4]>,
    visited: &mut SsoHashSet<I::GenericArg>,
) {
    if !visited.insert(ct.into()) {
        return;
    }
    match ct.kind() {
        ty::ConstKind::Param(_)
        | ty::ConstKind::Bound(_, _)
        | ty::ConstKind::Infer(_)
        | ty::ConstKind::Placeholder(_)
        | ty::ConstKind::Error(_) => {
            // Trivial
        }
        ty::ConstKind::Expr(e) => {
            for arg in e.args().iter() {
                compute_components_for_arg(tcx, arg, out, visited);
            }
        }
        ty::ConstKind::Value(ty, _) => {
            compute_components_for_ty(tcx, ty, out, visited);
        }
        ty::ConstKind::Unevaluated(uv) => {
            for arg in uv.args.iter() {
                compute_components_for_arg(tcx, arg, out, visited);
            }
        }
    }
}
