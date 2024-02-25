//! Orphan checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use crate::errors;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_middle::ty::{self, AliasKind, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::def_id::LocalDefId;
use rustc_span::Span;
use rustc_trait_selection::traits::{self, IsFirstInputType};

#[instrument(skip(tcx), level = "debug")]
pub(crate) fn orphan_check_impl(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap().instantiate_identity();
    trait_ref.error_reported()?;

    let trait_def_id = trait_ref.def_id;

    match traits::orphan_check(tcx, impl_def_id.to_def_id()) {
        Ok(()) => {}
        Err(err) => {
            let item = tcx.hir().expect_item(impl_def_id);
            let hir::ItemKind::Impl(impl_) = item.kind else {
                bug!("{:?} is not an impl: {:?}", impl_def_id, item);
            };
            let tr = impl_.of_trait.as_ref().unwrap();
            let sp = tcx.def_span(impl_def_id);

            emit_orphan_check_error(
                tcx,
                sp,
                item.span,
                tr.path.span,
                trait_ref,
                impl_.self_ty.span,
                impl_.generics,
                err,
            )?
        }
    }

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
    tcx: TyCtxt<'tcx>,
    sp: Span,
    full_impl_span: Span,
    trait_span: Span,
    trait_ref: ty::TraitRef<'tcx>,
    self_ty_span: Span,
    generics: &hir::Generics<'tcx>,
    err: traits::OrphanCheckErr<'tcx>,
) -> Result<!, ErrorGuaranteed> {
    let self_ty = trait_ref.self_ty();
    Err(match err {
        traits::OrphanCheckErr::NonLocalInputType(tys) => {
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

                match &ty.kind() {
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
                    ty::RawPtr(ptr_ty) => {
                        if !self_ty.has_param() {
                            let mut_key = ptr_ty.mutbl.prefix_str();
                            sugg = Some(errors::OnlyCurrentTraitsPointerSugg {
                                wrapper_span: self_ty_span,
                                struct_span: full_impl_span.shrink_to_lo(),
                                mut_key,
                                ptr_ty: ptr_ty.ty,
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
        traits::OrphanCheckErr::UncoveredTy(param_ty, local_type) => {
            let mut sp = sp;
            for param in generics.params {
                if param.name.ident().to_string() == param_ty.to_string() {
                    sp = param.span;
                }
            }

            match local_type {
                Some(local_type) => tcx.dcx().emit_err(errors::TyParamFirstLocal {
                    span: sp,
                    note: (),
                    param_ty,
                    local_type,
                }),
                None => tcx.dcx().emit_err(errors::TyParamSome { span: sp, note: (), param_ty }),
            }
        }
    })
}
