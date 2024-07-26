//! The outlives relation `T: 'a` or `'a: 'b`. This code frequently
//! refers to rules defined in RFC 1214 (`OutlivesFooBar`), so see that
//! RFC for reference.

use derive_where::derive_where;
use smallvec::{smallvec, SmallVec};

use crate::data_structures::SsoHashSet;
use crate::inherent::*;
use crate::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitableExt as _, TypeVisitor};
use crate::{self as ty, Interner};

#[derive_where(Debug; I: Interner)]
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
    cx: I,
    ty: I::Ty,
    out: &mut SmallVec<[Component<I>; 4]>,
) {
    ty.visit_with(&mut OutlivesCollector { cx, out, visited: Default::default() });
}

struct OutlivesCollector<'a, I: Interner> {
    cx: I,
    out: &'a mut SmallVec<[Component<I>; 4]>,
    visited: SsoHashSet<I::Ty>,
}

impl<I: Interner> TypeVisitor<I> for OutlivesCollector<'_, I> {
    fn visit_ty(&mut self, ty: I::Ty) -> Self::Result {
        if !self.visited.insert(ty) {
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
                        ty::GenericArgKind::Lifetime(_) => {}
                        ty::GenericArgKind::Type(_) | ty::GenericArgKind::Const(_) => {
                            child.visit_with(self);
                        }
                    }
                }
            }

            ty::Closure(_, args) => {
                args.as_closure().tupled_upvars_ty().visit_with(self);
            }

            ty::CoroutineClosure(_, args) => {
                args.as_coroutine_closure().tupled_upvars_ty().visit_with(self);
            }

            ty::Coroutine(_, args) => {
                args.as_coroutine().tupled_upvars_ty().visit_with(self);

                // We ignore regions in the coroutine interior as we don't
                // want these to affect region inference
            }

            // All regions are bound inside a witness, and we don't emit
            // higher-ranked outlives components currently.
            ty::CoroutineWitness(..) => {}

            // OutlivesTypeParameterEnv -- the actual checking that `X:'a`
            // is implied by the environment is done in regionck.
            ty::Param(p) => {
                self.out.push(Component::Param(p));
            }

            ty::Placeholder(p) => {
                self.out.push(Component::Placeholder(p));
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
                    self.out.push(Component::Alias(alias_ty));
                } else {
                    // fallback case: hard code
                    // OutlivesProjectionComponents. Continue walking
                    // through and constrain Pi.
                    let mut subcomponents = smallvec![];
                    compute_alias_components_recursive(self.cx, ty, &mut subcomponents);
                    self.out.push(Component::EscapingAlias(subcomponents.into_iter().collect()));
                }
            }

            // We assume that inference variables are fully resolved.
            // So, if we encounter an inference variable, just record
            // the unresolved variable as a component.
            ty::Infer(infer_ty) => {
                self.out.push(Component::UnresolvedInferenceVariable(infer_ty));
            }

            // Most types do not introduce any region binders, nor
            // involve any other subtle cases, and so the WF relation
            // simply constraints any regions referenced directly by
            // the type and then visits the types that are lexically
            // contained within.
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Never
            | ty::Error(_) => {
                // Trivial
            }

            ty::Bound(_, _) => {
                // FIXME: Bound vars matter here!
            }

            ty::Adt(_, _)
            | ty::Foreign(_)
            | ty::Array(_, _)
            | ty::Pat(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::Ref(_, _, _)
            | ty::FnPtr(_)
            | ty::Dynamic(_, _, _)
            | ty::Tuple(_) => {
                ty.super_visit_with(self);
            }
        }
    }

    fn visit_region(&mut self, lt: I::Region) -> Self::Result {
        if !lt.is_bound() {
            self.out.push(Component::Region(lt));
        }
    }
}

/// Collect [Component]s for *all* the args of `parent`.
///
/// This should not be used to get the components of `parent` itself.
/// Use [push_outlives_components] instead.
pub fn compute_alias_components_recursive<I: Interner>(
    cx: I,
    alias_ty: I::Ty,
    out: &mut SmallVec<[Component<I>; 4]>,
) {
    let ty::Alias(kind, alias_ty) = alias_ty.kind() else {
        unreachable!("can only call `compute_alias_components_recursive` on an alias type")
    };

    let opt_variances =
        if kind == ty::Opaque { Some(cx.variances_of(alias_ty.def_id)) } else { None };

    let mut visitor = OutlivesCollector { cx, out, visited: Default::default() };

    for (index, child) in alias_ty.args.iter().enumerate() {
        if opt_variances.and_then(|variances| variances.get(index)) == Some(ty::Bivariant) {
            continue;
        }
        child.visit_with(&mut visitor);
    }
}
