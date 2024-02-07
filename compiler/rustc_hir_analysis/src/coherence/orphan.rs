//! Orphan checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{DelayDm, ErrorGuaranteed};
use rustc_hir as hir;
use rustc_middle::ty::util::CheckRegions;
use rustc_middle::ty::GenericArgs;
use rustc_middle::ty::{
    self, AliasKind, ImplPolarity, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
    TypeVisitor,
};
use rustc_session::lint;
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::Span;
use rustc_trait_selection::traits;
use std::ops::ControlFlow;

use crate::errors;

#[instrument(skip(tcx), level = "debug")]
pub(crate) fn orphan_check_impl(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    let trait_ref = tcx.impl_trait_ref(impl_def_id).unwrap().instantiate_identity();
    trait_ref.error_reported()?;

    let ret = do_orphan_check_impl(tcx, trait_ref, impl_def_id);
    if tcx.trait_is_auto(trait_ref.def_id) {
        lint_auto_trait_impl(tcx, trait_ref, impl_def_id);
    }

    ret
}

fn do_orphan_check_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    let trait_def_id = trait_ref.def_id;

    match traits::orphan_check(tcx, def_id.to_def_id()) {
        Ok(()) => {}
        Err(err) => {
            let item = tcx.hir().expect_item(def_id);
            let hir::ItemKind::Impl(impl_) = item.kind else {
                bug!("{:?} is not an impl: {:?}", def_id, item);
            };
            let tr = impl_.of_trait.as_ref().unwrap();
            let sp = tcx.def_span(def_id);

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
                if self_ty.is_sized(tcx, tcx.param_env(def_id)) {
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
                let sp = tcx.def_span(def_id);
                span_bug!(sp, "weird self type for autotrait impl")
            }

            ty::Error(..) => (LocalImpl::Allow, NonlocalImpl::Allow),
        };

        if trait_def_id.is_local() {
            match local_impl {
                LocalImpl::Allow => {}
                LocalImpl::Disallow { problematic_kind } => {
                    return Err(tcx.dcx().emit_err(errors::TraitsWithDefaultImpl {
                        span: tcx.def_span(def_id),
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
                        span: tcx.def_span(def_id),
                        traits: tcx.def_path_str(trait_def_id),
                    }));
                }
                NonlocalImpl::DisallowOther => {
                    return Err(tcx.dcx().emit_err(errors::CrossCrateTraits {
                        span: tcx.def_span(def_id),
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
                let span = if is_target_ty {
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

                let is_foreign = !trait_ref.def_id.is_local() && !is_target_ty;

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

/// Lint impls of auto traits if they are likely to have
/// unsound or surprising effects on auto impls.
fn lint_auto_trait_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    impl_def_id: LocalDefId,
) {
    if trait_ref.args.len() != 1 {
        tcx.dcx().span_delayed_bug(
            tcx.def_span(impl_def_id),
            "auto traits cannot have generic parameters",
        );
        return;
    }
    let self_ty = trait_ref.self_ty();
    let (self_type_did, args) = match self_ty.kind() {
        ty::Adt(def, args) => (def.did(), args),
        _ => {
            // FIXME: should also lint for stuff like `&i32` but
            // considering that auto traits are unstable, that
            // isn't too important for now as this only affects
            // crates using `nightly`, and std.
            return;
        }
    };

    // Impls which completely cover a given root type are fine as they
    // disable auto impls entirely. So only lint if the args
    // are not a permutation of the identity args.
    let Err(arg) = tcx.uses_unique_generic_params(args, CheckRegions::No) else {
        // ok
        return;
    };

    // Ideally:
    //
    // - compute the requirements for the auto impl candidate
    // - check whether these are implied by the non covering impls
    // - if not, emit the lint
    //
    // What we do here is a bit simpler:
    //
    // - badly check if an auto impl candidate definitely does not apply
    //   for the given simplified type
    // - if so, do not lint
    if fast_reject_auto_impl(tcx, trait_ref.def_id, self_ty) {
        // ok
        return;
    }

    tcx.node_span_lint(
        lint::builtin::SUSPICIOUS_AUTO_TRAIT_IMPLS,
        tcx.local_def_id_to_hir_id(impl_def_id),
        tcx.def_span(impl_def_id),
        DelayDm(|| {
            format!(
                "cross-crate traits with a default impl, like `{}`, \
                         should not be specialized",
                tcx.def_path_str(trait_ref.def_id),
            )
        }),
        |lint| {
            let item_span = tcx.def_span(self_type_did);
            let self_descr = tcx.def_descr(self_type_did);
            match arg {
                ty::util::NotUniqueParam::DuplicateParam(arg) => {
                    lint.note(format!("`{arg}` is mentioned multiple times"));
                }
                ty::util::NotUniqueParam::NotParam(arg) => {
                    lint.note(format!("`{arg}` is not a generic parameter"));
                }
            }
            lint.span_note(
                item_span,
                format!(
                    "try using the same sequence of generic parameters as the {self_descr} definition",
                ),
            );
        },
    );
}

fn fast_reject_auto_impl<'tcx>(tcx: TyCtxt<'tcx>, trait_def_id: DefId, self_ty: Ty<'tcx>) -> bool {
    struct DisableAutoTraitVisitor<'tcx> {
        tcx: TyCtxt<'tcx>,
        trait_def_id: DefId,
        self_ty_root: Ty<'tcx>,
        seen: FxHashSet<DefId>,
    }

    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for DisableAutoTraitVisitor<'tcx> {
        type BreakTy = ();
        fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
            let tcx = self.tcx;
            if ty != self.self_ty_root {
                for impl_def_id in tcx.non_blanket_impls_for_ty(self.trait_def_id, ty) {
                    match tcx.impl_polarity(impl_def_id) {
                        ImplPolarity::Negative => return ControlFlow::Break(()),
                        ImplPolarity::Reservation => {}
                        // FIXME(@lcnr): That's probably not good enough, idk
                        //
                        // We might just want to take the rustdoc code and somehow avoid
                        // explicit impls for `Self`.
                        ImplPolarity::Positive => return ControlFlow::Continue(()),
                    }
                }
            }

            match ty.kind() {
                ty::Adt(def, args) if def.is_phantom_data() => args.visit_with(self),
                ty::Adt(def, args) => {
                    // @lcnr: This is the only place where cycles can happen. We avoid this
                    // by only visiting each `DefId` once.
                    //
                    // This will be is incorrect in subtle cases, but I don't care :)
                    if self.seen.insert(def.did()) {
                        for ty in def.all_fields().map(|field| field.ty(tcx, args)) {
                            ty.visit_with(self)?;
                        }
                    }

                    ControlFlow::Continue(())
                }
                _ => ty.super_visit_with(self),
            }
        }
    }

    let self_ty_root = match self_ty.kind() {
        ty::Adt(def, _) => Ty::new_adt(tcx, *def, GenericArgs::identity_for_item(tcx, def.did())),
        _ => unimplemented!("unexpected self ty {:?}", self_ty),
    };

    self_ty_root
        .visit_with(&mut DisableAutoTraitVisitor {
            tcx,
            self_ty_root,
            trait_def_id,
            seen: FxHashSet::default(),
        })
        .is_break()
}
