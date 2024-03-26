//! Orphan checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use crate::errors;

use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::ErrorGuaranteed;
use rustc_infer::infer::type_variable::TypeVariableOriginKind;
use rustc_infer::infer::InferCtxt;
use rustc_lint_defs::builtin::UNCOVERED_PARAM_IN_PROJECTION;
use rustc_middle::ty::{self, AliasKind, Ty, TyCtxt};
use rustc_middle::ty::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_middle::ty::{TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, Symbol};
use rustc_trait_selection::traits::{self, IsFirstInputType};

#[instrument(skip(tcx), level = "debug")]
pub(crate) fn orphan_check_impl(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap().instantiate_identity();
    trait_ref.error_reported()?;

    match traits::orphan_check(tcx, impl_def_id.to_def_id()) {
        traits::OrphanCheckResult::Ok => {}
        traits::OrphanCheckResult::Issue99554(data, infcx) => {
            lint_uncovered_ty_params(&infcx, data.ty, data.local_ty, impl_def_id);
        }
        traits::OrphanCheckResult::Err(err, infcx) => {
            let item = tcx.hir().expect_item(impl_def_id);
            let impl_ = item.expect_impl();
            let hir_trait_ref = impl_.of_trait.as_ref().unwrap();

            emit_orphan_check_error(
                &infcx,
                tcx.def_span(impl_def_id),
                item.span,
                hir_trait_ref.path.span,
                trait_ref,
                impl_.self_ty.span,
                tcx.generics_of(impl_def_id),
                err,
            )?
        }
    }

    let trait_def_id = trait_ref.def_id;

    // In addition to the above rules, we restrict impls of auto traits
    // so that they can only be implemented on nominal types, such as structs,
    // enums or foreign types. To see why this restriction exists, consider the
    // following example (#22978). Imagine that crate A defines an auto trait
    // `Foo` and a fn that operates on pairs of types:
    //
    // ```
    // // Crate A
    // auto trait Foo { }
    // fn two_foos<A:Foo,B:Foo>(..) {
    //     one_foo::<(A,B)>(..)
    // }
    // fn one_foo<T:Foo>(..) { .. }
    // ```
    //
    // This type-checks fine; in particular the fn
    // `two_foos` is able to conclude that `(A,B):Foo`
    // because `A:Foo` and `B:Foo`.
    //
    // Now imagine that crate B comes along and does the following:
    //
    // ```
    // struct A { }
    // struct B { }
    // impl Foo for A { }
    // impl Foo for B { }
    // impl !Foo for (A, B) { }
    // ```
    //
    // This final impl is legal according to the orphan
    // rules, but it invalidates the reasoning from
    // `two_foos` above.
    debug!(
        "trait_ref={:?} trait_def_id={:?} trait_is_auto={}",
        trait_ref,
        trait_def_id,
        tcx.trait_is_auto(trait_def_id)
    );

    if tcx.trait_is_auto(trait_def_id) {
        let self_ty = trait_ref.self_ty();

        // If the impl is in the same crate as the auto-trait, almost anything
        // goes.
        //
        //     impl MyAuto for Rc<Something> {}  // okay
        //     impl<T> !MyAuto for *const T {}   // okay
        //     impl<T> MyAuto for T {}           // okay
        //
        // But there is one important exception: implementing for a trait object
        // is not allowed.
        //
        //     impl MyAuto for dyn Trait {}      // NOT OKAY
        //     impl<T: ?Sized> MyAuto for T {}   // NOT OKAY
        //
        // With this restriction, it's guaranteed that an auto-trait is
        // implemented for a trait object if and only if the auto-trait is one
        // of the trait object's trait bounds (or a supertrait of a bound). In
        // other words `dyn Trait + AutoTrait` always implements AutoTrait,
        // while `dyn Trait` never implements AutoTrait.
        //
        // This is necessary in order for autotrait bounds on methods of trait
        // objects to be sound.
        //
        //     auto trait AutoTrait {}
        //
        //     trait ObjectSafeTrait {
        //         fn f(&self) where Self: AutoTrait;
        //     }
        //
        // We can allow f to be called on `dyn ObjectSafeTrait + AutoTrait`.
        //
        // If we didn't deny `impl AutoTrait for dyn Trait`, it would be unsound
        // for the ObjectSafeTrait shown above to be object safe because someone
        // could take some type implementing ObjectSafeTrait but not AutoTrait,
        // unsize it to `dyn ObjectSafeTrait`, and call .f() which has no
        // concrete implementation (issue #50781).
        enum LocalImpl {
            Allow,
            Disallow { problematic_kind: &'static str },
        }

        // If the auto-trait is from a dependency, it must only be getting
        // implemented for a nominal type, and specifically one local to the
        // current crate.
        //
        //     impl<T> Sync for MyStruct<T> {}   // okay
        //
        //     impl Sync for Rc<MyStruct> {}     // NOT OKAY
        enum NonlocalImpl {
            Allow,
            DisallowBecauseNonlocal,
            DisallowOther,
        }

        // Exhaustive match considering that this logic is essential for
        // soundness.
        let (local_impl, nonlocal_impl) = match self_ty.kind() {
            // struct Struct<T>;
            // impl AutoTrait for Struct<Foo> {}
            ty::Adt(self_def, _) => (
                LocalImpl::Allow,
                if self_def.did().is_local() {
                    NonlocalImpl::Allow
                } else {
                    NonlocalImpl::DisallowBecauseNonlocal
                },
            ),

            // extern { type OpaqueType; }
            // impl AutoTrait for OpaqueType {}
            ty::Foreign(did) => (
                LocalImpl::Allow,
                if did.is_local() {
                    NonlocalImpl::Allow
                } else {
                    NonlocalImpl::DisallowBecauseNonlocal
                },
            ),

            // impl AutoTrait for dyn Trait {}
            ty::Dynamic(..) => (
                LocalImpl::Disallow { problematic_kind: "trait object" },
                NonlocalImpl::DisallowOther,
            ),

            // impl<T> AutoTrait for T {}
            // impl<T: ?Sized> AutoTrait for T {}
            ty::Param(..) => (
                if self_ty.is_sized(tcx, tcx.param_env(impl_def_id)) {
                    LocalImpl::Allow
                } else {
                    LocalImpl::Disallow { problematic_kind: "generic type" }
                },
                NonlocalImpl::DisallowOther,
            ),

            ty::Alias(kind, _) => {
                let problematic_kind = match kind {
                    // trait Id { type This: ?Sized; }
                    // impl<T: ?Sized> Id for T {
                    //     type This = T;
                    // }
                    // impl<T: ?Sized> AutoTrait for <T as Id>::This {}
                    AliasKind::Projection => "associated type",
                    // type Foo = (impl Sized, bool)
                    // impl AutoTrait for Foo {}
                    AliasKind::Weak => "type alias",
                    // type Opaque = impl Trait;
                    // impl AutoTrait for Opaque {}
                    AliasKind::Opaque => "opaque type",
                    // ```
                    // struct S<T>(T);
                    // impl<T: ?Sized> S<T> {
                    //     type This = T;
                    // }
                    // impl<T: ?Sized> AutoTrait for S<T>::This {}
                    // ```
                    // FIXME(inherent_associated_types): The example code above currently leads to a cycle
                    AliasKind::Inherent => "associated type",
                };
                (LocalImpl::Disallow { problematic_kind }, NonlocalImpl::DisallowOther)
            }

            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Str
            | ty::Array(..)
            | ty::Slice(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Never
            | ty::Tuple(..) => (LocalImpl::Allow, NonlocalImpl::DisallowOther),

            ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Infer(..) => {
                let sp = tcx.def_span(impl_def_id);
                span_bug!(sp, "weird self type for autotrait impl")
            }

            ty::Error(..) => (LocalImpl::Allow, NonlocalImpl::Allow),
        };

        if trait_def_id.is_local() {
            match local_impl {
                LocalImpl::Allow => {}
                LocalImpl::Disallow { problematic_kind } => {
                    return Err(tcx.dcx().emit_err(errors::TraitsWithDefaultImpl {
                        span: tcx.def_span(impl_def_id),
                        traits: tcx.def_path_str(trait_def_id),
                        problematic_kind,
                        self_ty,
                    }));
                }
            }
        } else {
            match nonlocal_impl {
                NonlocalImpl::Allow => {}
                NonlocalImpl::DisallowBecauseNonlocal => {
                    return Err(tcx.dcx().emit_err(errors::CrossCrateTraitsDefined {
                        span: tcx.def_span(impl_def_id),
                        traits: tcx.def_path_str(trait_def_id),
                    }));
                }
                NonlocalImpl::DisallowOther => {
                    return Err(tcx.dcx().emit_err(errors::CrossCrateTraits {
                        span: tcx.def_span(impl_def_id),
                        traits: tcx.def_path_str(trait_def_id),
                        self_ty,
                    }));
                }
            }
        }
    }

    Ok(())
}

fn emit_orphan_check_error<'tcx>(
    infcx: &InferCtxt<'tcx>,
    sp: Span,
    full_impl_span: Span,
    trait_span: Span,
    trait_ref: ty::TraitRef<'tcx>,
    self_ty_span: Span,
    generics: &'tcx ty::Generics,
    err: traits::OrphanCheckErr<'tcx>,
) -> Result<!, ErrorGuaranteed> {
    let tcx = infcx.tcx;

    Err(match err {
        traits::OrphanCheckErr::NonLocalInputType(tys) => {
            let self_ty = trait_ref.self_ty();

            let (mut opaque, mut foreign, mut name, mut pointer, mut ty_diag) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut sugg = None;
            for &(mut ty, is_target_ty) in &tys {
                let span = if matches!(is_target_ty, IsFirstInputType::Yes) {
                    // Point at `D<A>` in `impl<A, B> for C<B> in D<A>`
                    self_ty_span
                } else {
                    // Point at `C<B>` in `impl<A, B> for C<B> in D<A>`
                    trait_span
                };

                ty = ty.fold_with(&mut TyVarReplacer { infcx, generics });
                ty = tcx.erase_regions(ty);

                ty = match ty.kind() {
                    // Remove the type arguments from the output, as they are not relevant.
                    // You can think of this as the reverse of `resolve_vars_if_possible`.
                    // That way if we had `Vec<MyType>`, we will properly attribute the
                    // problem to `Vec<T>` and avoid confusing the user if they were to see
                    // `MyType` in the error.
                    ty::Adt(def, _) => Ty::new_adt(tcx, *def, ty::List::empty()),
                    _ => ty,
                };

                fn push_to_foreign_or_name<'tcx>(
                    is_foreign: bool,
                    foreign: &mut Vec<errors::OnlyCurrentTraitsForeign>,
                    name: &mut Vec<errors::OnlyCurrentTraitsName<'tcx>>,
                    span: Span,
                    sname: &'tcx str,
                ) {
                    if is_foreign {
                        foreign.push(errors::OnlyCurrentTraitsForeign { span })
                    } else {
                        name.push(errors::OnlyCurrentTraitsName { span, name: sname });
                    }
                }

                let is_foreign =
                    !trait_ref.def_id.is_local() && matches!(is_target_ty, IsFirstInputType::No);

                match *ty.kind() {
                    ty::Slice(_) => {
                        push_to_foreign_or_name(
                            is_foreign,
                            &mut foreign,
                            &mut name,
                            span,
                            "slices",
                        );
                    }
                    ty::Array(..) => {
                        push_to_foreign_or_name(
                            is_foreign,
                            &mut foreign,
                            &mut name,
                            span,
                            "arrays",
                        );
                    }
                    ty::Tuple(..) => {
                        push_to_foreign_or_name(
                            is_foreign,
                            &mut foreign,
                            &mut name,
                            span,
                            "tuples",
                        );
                    }
                    ty::Alias(ty::Opaque, ..) => {
                        opaque.push(errors::OnlyCurrentTraitsOpaque { span })
                    }
                    ty::RawPtr(ptr_ty, mutbl) => {
                        if !self_ty.has_param() {
                            let mut_key = mutbl.prefix_str();
                            sugg = Some(errors::OnlyCurrentTraitsPointerSugg {
                                wrapper_span: self_ty_span,
                                struct_span: full_impl_span.shrink_to_lo(),
                                mut_key,
                                ptr_ty,
                            });
                        }
                        pointer.push(errors::OnlyCurrentTraitsPointer { span, pointer: ty });
                    }
                    _ => ty_diag.push(errors::OnlyCurrentTraitsTy { span, ty }),
                }
            }

            let err_struct = match self_ty.kind() {
                ty::Adt(..) => errors::OnlyCurrentTraits::Outside {
                    span: sp,
                    note: (),
                    opaque,
                    foreign,
                    name,
                    pointer,
                    ty: ty_diag,
                    sugg,
                },
                _ if self_ty.is_primitive() => errors::OnlyCurrentTraits::Primitive {
                    span: sp,
                    note: (),
                    opaque,
                    foreign,
                    name,
                    pointer,
                    ty: ty_diag,
                    sugg,
                },
                _ => errors::OnlyCurrentTraits::Arbitrary {
                    span: sp,
                    note: (),
                    opaque,
                    foreign,
                    name,
                    pointer,
                    ty: ty_diag,
                    sugg,
                },
            };
            tcx.dcx().emit_err(err_struct)
        }
        traits::OrphanCheckErr::UncoveredTyParams(err) => {
            let mut collector =
                UncoveredTyParamCollector { infcx, uncovered_params: Default::default() };
            err.ty.visit_with(&mut collector);

            let mut reported = None;
            for (param, span) in collector.uncovered_params {
                reported.get_or_insert(match err.local_ty {
                    Some(local_type) => tcx.dcx().emit_err(errors::TyParamFirstLocal {
                        span,
                        note: (),
                        param,
                        local_type,
                    }),
                    None => tcx.dcx().emit_err(errors::TyParamSome { span, note: (), param }),
                });
            }
            // FIXME: possibly reachable
            reported.unwrap_or_else(|| bug!("failed to find ty param in `{}`", err.ty))
        }
    })
}

fn lint_uncovered_ty_params<'tcx>(
    infcx: &InferCtxt<'tcx>,
    uncovered_ty_params: Ty<'tcx>,
    local_ty: Option<Ty<'tcx>>,
    impl_def_id: LocalDefId,
) {
    let tcx = infcx.tcx;
    let hir_id = tcx.local_def_id_to_hir_id(impl_def_id);

    let mut collector = UncoveredTyParamCollector { infcx, uncovered_params: Default::default() };
    uncovered_ty_params.visit_with(&mut collector);
    debug_assert!(!collector.uncovered_params.is_empty()); // FIXME: possibly reachable

    for (param, span) in collector.uncovered_params {
        match local_ty {
            Some(local_type) => tcx.emit_node_span_lint(
                UNCOVERED_PARAM_IN_PROJECTION,
                hir_id,
                span,
                errors::TyParamFirstLocalLint { span, note: (), param, local_type },
            ),
            None => tcx.emit_node_span_lint(
                UNCOVERED_PARAM_IN_PROJECTION,
                hir_id,
                span,
                errors::TyParamSomeLint { span, note: (), param },
            ),
        };
    }
}

struct TyVarReplacer<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'tcx>,
    generics: &'tcx ty::Generics,
}

impl<'cx, 'tcx> TypeFolder<TyCtxt<'tcx>> for TyVarReplacer<'cx, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !ty.has_type_flags(ty::TypeFlags::HAS_TY_INFER) {
            return ty;
        }
        let Some(origin) = self.infcx.type_var_origin(ty) else {
            return ty.super_fold_with(self);
        };
        if let TypeVariableOriginKind::TypeParameterDefinition(name, def_id) = origin.kind
            && let Some(index) = self.generics.param_def_id_to_index(self.infcx.tcx, def_id)
        {
            Ty::new_param(self.infcx.tcx, index, name)
        } else {
            ty
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if !ct.has_type_flags(ty::TypeFlags::HAS_TY_INFER) {
            return ct;
        }
        ct.super_fold_with(self)
    }
}

struct UncoveredTyParamCollector<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'tcx>,
    uncovered_params: FxIndexMap<Symbol, Span>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for UncoveredTyParamCollector<'_, 'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        if !ty.has_type_flags(ty::TypeFlags::HAS_TY_INFER) {
            return;
        }
        let Some(origin) = self.infcx.type_var_origin(ty) else {
            return ty.super_visit_with(self);
        };
        if let TypeVariableOriginKind::TypeParameterDefinition(name, def_id) = origin.kind {
            let span = self.infcx.tcx.def_ident_span(def_id).unwrap();
            self.uncovered_params.insert(name, span);
        }
    }

    fn visit_const(&mut self, ct: ty::Const<'tcx>) -> Self::Result {
        if ct.has_type_flags(ty::TypeFlags::HAS_TY_INFER) {
            ct.super_visit_with(self)
        }
    }
}
