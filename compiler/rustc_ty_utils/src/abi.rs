use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_middle::query::Providers;
use rustc_middle::ty::layout::{
    fn_can_unwind, FnAbiError, HasParamEnv, HasTyCtxt, LayoutCx, LayoutOf, TyAndLayout,
};
use rustc_middle::ty::{self, InstanceDef, Ty, TyCtxt};
use rustc_session::config::OptLevel;
use rustc_span::def_id::DefId;
use rustc_target::abi::call::{
    ArgAbi, ArgAttribute, ArgAttributes, ArgExtension, Conv, FnAbi, PassMode, Reg, RegKind,
    RiscvInterruptKind,
};
use rustc_target::abi::*;
use rustc_target::spec::abi::Abi as SpecAbi;

use std::iter;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { fn_abi_of_fn_ptr, fn_abi_of_instance, ..*providers };
}

// NOTE(eddyb) this is private to avoid using it from outside of
// `fn_abi_of_instance` - any other uses are either too high-level
// for `Instance` (e.g. typeck would use `Ty::fn_sig` instead),
// or should go through `FnAbi` instead, to avoid losing any
// adjustments `fn_abi_of_instance` might be performing.
#[tracing::instrument(level = "debug", skip(tcx, param_env))]
fn fn_sig_for_fn_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> ty::PolyFnSig<'tcx> {
    if let InstanceDef::ThreadLocalShim(..) = instance.def {
        return ty::Binder::dummy(tcx.mk_fn_sig(
            [],
            tcx.thread_local_ptr_ty(instance.def_id()),
            false,
            hir::Unsafety::Normal,
            rustc_target::spec::abi::Abi::Unadjusted,
        ));
    }

    let ty = instance.ty(tcx, param_env);
    match *ty.kind() {
        ty::FnDef(..) => {
            // HACK(davidtwco,eddyb): This is a workaround for polymorphization considering
            // parameters unused if they show up in the signature, but not in the `mir::Body`
            // (i.e. due to being inside a projection that got normalized, see
            // `tests/ui/polymorphization/normalized_sig_types.rs`), and codegen not keeping
            // track of a polymorphization `ParamEnv` to allow normalizing later.
            //
            // We normalize the `fn_sig` again after instantiating at a later point.
            let mut sig = match *ty.kind() {
                ty::FnDef(def_id, args) => tcx
                    .fn_sig(def_id)
                    .map_bound(|fn_sig| {
                        tcx.normalize_erasing_regions(tcx.param_env(def_id), fn_sig)
                    })
                    .instantiate(tcx, args),
                _ => unreachable!(),
            };

            if let ty::InstanceDef::VTableShim(..) = instance.def {
                // Modify `fn(self, ...)` to `fn(self: *mut Self, ...)`.
                sig = sig.map_bound(|mut sig| {
                    let mut inputs_and_output = sig.inputs_and_output.to_vec();
                    inputs_and_output[0] = Ty::new_mut_ptr(tcx, inputs_and_output[0]);
                    sig.inputs_and_output = tcx.mk_type_list(&inputs_and_output);
                    sig
                });
            }
            sig
        }
        ty::Closure(def_id, args) => {
            let sig = args.as_closure().sig();

            let bound_vars = tcx.mk_bound_variable_kinds_from_iter(
                sig.bound_vars().iter().chain(iter::once(ty::BoundVariableKind::Region(ty::BrEnv))),
            );
            let br = ty::BoundRegion {
                var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                kind: ty::BoundRegionKind::BrEnv,
            };
            let env_region = ty::Region::new_bound(tcx, ty::INNERMOST, br);
            let env_ty = tcx.closure_env_ty(
                Ty::new_closure(tcx, def_id, args),
                args.as_closure().kind(),
                env_region,
            );

            let sig = sig.skip_binder();
            ty::Binder::bind_with_vars(
                tcx.mk_fn_sig(
                    iter::once(env_ty).chain(sig.inputs().iter().cloned()),
                    sig.output(),
                    sig.c_variadic,
                    sig.unsafety,
                    sig.abi,
                ),
                bound_vars,
            )
        }
        ty::CoroutineClosure(def_id, args) => {
            let sig = args.as_coroutine_closure().coroutine_closure_sig();
            let bound_vars = tcx.mk_bound_variable_kinds_from_iter(
                sig.bound_vars().iter().chain(iter::once(ty::BoundVariableKind::Region(ty::BrEnv))),
            );
            let br = ty::BoundRegion {
                var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                kind: ty::BoundRegionKind::BrEnv,
            };
            let env_region = ty::Region::new_bound(tcx, ty::INNERMOST, br);

            // When this `CoroutineClosure` comes from a `ConstructCoroutineInClosureShim`,
            // make sure we respect the `target_kind` in that shim.
            // FIXME(async_closures): This shouldn't be needed, and we should be populating
            // a separate def-id for these bodies.
            let mut kind = args.as_coroutine_closure().kind();
            if let InstanceDef::ConstructCoroutineInClosureShim { target_kind, .. } = instance.def {
                kind = target_kind;
            }

            let env_ty =
                tcx.closure_env_ty(Ty::new_coroutine_closure(tcx, def_id, args), kind, env_region);

            let sig = sig.skip_binder();
            ty::Binder::bind_with_vars(
                tcx.mk_fn_sig(
                    iter::once(env_ty).chain([sig.tupled_inputs_ty]),
                    sig.to_coroutine_given_kind_and_upvars(
                        tcx,
                        args.as_coroutine_closure().parent_args(),
                        tcx.coroutine_for_closure(def_id),
                        kind,
                        env_region,
                        args.as_coroutine_closure().tupled_upvars_ty(),
                        args.as_coroutine_closure().coroutine_captures_by_ref_ty(),
                    ),
                    sig.c_variadic,
                    sig.unsafety,
                    sig.abi,
                ),
                bound_vars,
            )
        }
        ty::Coroutine(did, args) => {
            let coroutine_kind = tcx.coroutine_kind(did).unwrap();
            let sig = args.as_coroutine().sig();

            let bound_vars = tcx.mk_bound_variable_kinds_from_iter(iter::once(
                ty::BoundVariableKind::Region(ty::BrEnv),
            ));
            let br = ty::BoundRegion {
                var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                kind: ty::BoundRegionKind::BrEnv,
            };

            let mut ty = ty;
            // When this `Closure` comes from a `CoroutineKindShim`,
            // make sure we respect the `target_kind` in that shim.
            // FIXME(async_closures): This shouldn't be needed, and we should be populating
            // a separate def-id for these bodies.
            if let InstanceDef::CoroutineKindShim { target_kind, .. } = instance.def {
                // Grab the parent coroutine-closure. It has the same args for the purposes
                // of instantiation, so this will be okay to do.
                let ty::CoroutineClosure(_, coroutine_closure_args) = *tcx
                    .instantiate_and_normalize_erasing_regions(
                        args,
                        param_env,
                        tcx.type_of(tcx.parent(did)),
                    )
                    .kind()
                else {
                    bug!("CoroutineKindShim comes from calling a coroutine-closure");
                };
                let coroutine_closure_args = coroutine_closure_args.as_coroutine_closure();
                ty = tcx.instantiate_bound_regions_with_erased(
                    coroutine_closure_args.coroutine_closure_sig().map_bound(|sig| {
                        sig.to_coroutine_given_kind_and_upvars(
                            tcx,
                            coroutine_closure_args.parent_args(),
                            did,
                            target_kind,
                            tcx.lifetimes.re_erased,
                            coroutine_closure_args.tupled_upvars_ty(),
                            coroutine_closure_args.coroutine_captures_by_ref_ty(),
                        )
                    }),
                );
            }
            let env_ty = Ty::new_mut_ref(tcx, ty::Region::new_bound(tcx, ty::INNERMOST, br), ty);

            let pin_did = tcx.require_lang_item(LangItem::Pin, None);
            let pin_adt_ref = tcx.adt_def(pin_did);
            let pin_args = tcx.mk_args(&[env_ty.into()]);
            let env_ty = match coroutine_kind {
                hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen, _) => {
                    // Iterator::next doesn't accept a pinned argument,
                    // unlike for all other coroutine kinds.
                    env_ty
                }
                hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _)
                | hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen, _)
                | hir::CoroutineKind::Coroutine(_) => Ty::new_adt(tcx, pin_adt_ref, pin_args),
            };

            // The `FnSig` and the `ret_ty` here is for a coroutines main
            // `Coroutine::resume(...) -> CoroutineState` function in case we
            // have an ordinary coroutine, the `Future::poll(...) -> Poll`
            // function in case this is a special coroutine backing an async construct
            // or the `Iterator::next(...) -> Option` function in case this is a
            // special coroutine backing a gen construct.
            let (resume_ty, ret_ty) = match coroutine_kind {
                hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _) => {
                    // The signature should be `Future::poll(_, &mut Context<'_>) -> Poll<Output>`
                    assert_eq!(sig.yield_ty, tcx.types.unit);

                    let poll_did = tcx.require_lang_item(LangItem::Poll, None);
                    let poll_adt_ref = tcx.adt_def(poll_did);
                    let poll_args = tcx.mk_args(&[sig.return_ty.into()]);
                    let ret_ty = Ty::new_adt(tcx, poll_adt_ref, poll_args);

                    // We have to replace the `ResumeTy` that is used for type and borrow checking
                    // with `&mut Context<'_>` which is used in codegen.
                    #[cfg(debug_assertions)]
                    {
                        if let ty::Adt(resume_ty_adt, _) = sig.resume_ty.kind() {
                            let expected_adt =
                                tcx.adt_def(tcx.require_lang_item(LangItem::ResumeTy, None));
                            assert_eq!(*resume_ty_adt, expected_adt);
                        } else {
                            panic!("expected `ResumeTy`, found `{:?}`", sig.resume_ty);
                        };
                    }
                    let context_mut_ref = Ty::new_task_context(tcx);

                    (Some(context_mut_ref), ret_ty)
                }
                hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen, _) => {
                    // The signature should be `Iterator::next(_) -> Option<Yield>`
                    let option_did = tcx.require_lang_item(LangItem::Option, None);
                    let option_adt_ref = tcx.adt_def(option_did);
                    let option_args = tcx.mk_args(&[sig.yield_ty.into()]);
                    let ret_ty = Ty::new_adt(tcx, option_adt_ref, option_args);

                    assert_eq!(sig.return_ty, tcx.types.unit);
                    assert_eq!(sig.resume_ty, tcx.types.unit);

                    (None, ret_ty)
                }
                hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen, _) => {
                    // The signature should be
                    // `AsyncIterator::poll_next(_, &mut Context<'_>) -> Poll<Option<Output>>`
                    assert_eq!(sig.return_ty, tcx.types.unit);

                    // Yield type is already `Poll<Option<yield_ty>>`
                    let ret_ty = sig.yield_ty;

                    // We have to replace the `ResumeTy` that is used for type and borrow checking
                    // with `&mut Context<'_>` which is used in codegen.
                    #[cfg(debug_assertions)]
                    {
                        if let ty::Adt(resume_ty_adt, _) = sig.resume_ty.kind() {
                            let expected_adt =
                                tcx.adt_def(tcx.require_lang_item(LangItem::ResumeTy, None));
                            assert_eq!(*resume_ty_adt, expected_adt);
                        } else {
                            panic!("expected `ResumeTy`, found `{:?}`", sig.resume_ty);
                        };
                    }
                    let context_mut_ref = Ty::new_task_context(tcx);

                    (Some(context_mut_ref), ret_ty)
                }
                hir::CoroutineKind::Coroutine(_) => {
                    // The signature should be `Coroutine::resume(_, Resume) -> CoroutineState<Yield, Return>`
                    let state_did = tcx.require_lang_item(LangItem::CoroutineState, None);
                    let state_adt_ref = tcx.adt_def(state_did);
                    let state_args = tcx.mk_args(&[sig.yield_ty.into(), sig.return_ty.into()]);
                    let ret_ty = Ty::new_adt(tcx, state_adt_ref, state_args);

                    (Some(sig.resume_ty), ret_ty)
                }
            };

            let fn_sig = if let Some(resume_ty) = resume_ty {
                tcx.mk_fn_sig(
                    [env_ty, resume_ty],
                    ret_ty,
                    false,
                    hir::Unsafety::Normal,
                    rustc_target::spec::abi::Abi::Rust,
                )
            } else {
                // `Iterator::next` doesn't have a `resume` argument.
                tcx.mk_fn_sig(
                    [env_ty],
                    ret_ty,
                    false,
                    hir::Unsafety::Normal,
                    rustc_target::spec::abi::Abi::Rust,
                )
            };
            ty::Binder::bind_with_vars(fn_sig, bound_vars)
        }
        _ => bug!("unexpected type {:?} in Instance::fn_sig", ty),
    }
}

#[inline]
fn conv_from_spec_abi(tcx: TyCtxt<'_>, abi: SpecAbi, c_variadic: bool) -> Conv {
    use rustc_target::spec::abi::Abi::*;
    match tcx.sess.target.adjust_abi(abi, c_variadic) {
        RustIntrinsic | PlatformIntrinsic | Rust | RustCall => Conv::Rust,

        // This is intentionally not using `Conv::Cold`, as that has to preserve
        // even SIMD registers, which is generally not a good trade-off.
        RustCold => Conv::PreserveMost,

        // It's the ABI's job to select this, not ours.
        System { .. } => bug!("system abi should be selected elsewhere"),
        EfiApi => bug!("eficall abi should be selected elsewhere"),

        Stdcall { .. } => Conv::X86Stdcall,
        Fastcall { .. } => Conv::X86Fastcall,
        Vectorcall { .. } => Conv::X86VectorCall,
        Thiscall { .. } => Conv::X86ThisCall,
        C { .. } => Conv::C,
        Unadjusted => Conv::C,
        Win64 { .. } => Conv::X86_64Win64,
        SysV64 { .. } => Conv::X86_64SysV,
        Aapcs { .. } => Conv::ArmAapcs,
        CCmseNonSecureCall => Conv::CCmseNonSecureCall,
        PtxKernel => Conv::PtxKernel,
        Msp430Interrupt => Conv::Msp430Intr,
        X86Interrupt => Conv::X86Intr,
        AvrInterrupt => Conv::AvrInterrupt,
        AvrNonBlockingInterrupt => Conv::AvrNonBlockingInterrupt,
        RiscvInterruptM => Conv::RiscvInterrupt { kind: RiscvInterruptKind::Machine },
        RiscvInterruptS => Conv::RiscvInterrupt { kind: RiscvInterruptKind::Supervisor },
        Wasm => Conv::C,

        // These API constants ought to be more specific...
        Cdecl { .. } => Conv::C,
    }
}

fn fn_abi_of_fn_ptr<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, (ty::PolyFnSig<'tcx>, &'tcx ty::List<Ty<'tcx>>)>,
) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, &'tcx FnAbiError<'tcx>> {
    let (param_env, (sig, extra_args)) = query.into_parts();

    let cx = LayoutCx { tcx, param_env };
    fn_abi_new_uncached(&cx, sig, extra_args, None, None, false)
}

fn fn_abi_of_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, (ty::Instance<'tcx>, &'tcx ty::List<Ty<'tcx>>)>,
) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, &'tcx FnAbiError<'tcx>> {
    let (param_env, (instance, extra_args)) = query.into_parts();

    let sig = fn_sig_for_fn_abi(tcx, instance, param_env);

    let caller_location =
        instance.def.requires_caller_location(tcx).then(|| tcx.caller_location_ty());

    fn_abi_new_uncached(
        &LayoutCx { tcx, param_env },
        sig,
        extra_args,
        caller_location,
        Some(instance.def_id()),
        matches!(instance.def, ty::InstanceDef::Virtual(..)),
    )
}

// Handle safe Rust thin and fat pointers.
fn adjust_for_rust_scalar<'tcx>(
    cx: LayoutCx<'tcx, TyCtxt<'tcx>>,
    attrs: &mut ArgAttributes,
    scalar: Scalar,
    layout: TyAndLayout<'tcx>,
    offset: Size,
    is_return: bool,
    drop_target_pointee: Option<Ty<'tcx>>,
) {
    // Booleans are always a noundef i1 that needs to be zero-extended.
    if scalar.is_bool() {
        attrs.ext(ArgExtension::Zext);
        attrs.set(ArgAttribute::NoUndef);
        return;
    }

    if !scalar.is_uninit_valid() {
        attrs.set(ArgAttribute::NoUndef);
    }

    // Only pointer types handled below.
    let Scalar::Initialized { value: Pointer(_), valid_range } = scalar else { return };

    // Set `nonnull` if the validity range excludes zero, or for the argument to `drop_in_place`,
    // which must be nonnull per its documented safety requirements.
    if !valid_range.contains(0) || drop_target_pointee.is_some() {
        attrs.set(ArgAttribute::NonNull);
    }

    if let Some(pointee) = layout.pointee_info_at(&cx, offset) {
        let kind = if let Some(kind) = pointee.safe {
            Some(kind)
        } else if let Some(pointee) = drop_target_pointee {
            // The argument to `drop_in_place` is semantically equivalent to a mutable reference.
            Some(PointerKind::MutableRef { unpin: pointee.is_unpin(cx.tcx, cx.param_env()) })
        } else {
            None
        };
        if let Some(kind) = kind {
            attrs.pointee_align = Some(pointee.align);

            // `Box` are not necessarily dereferenceable for the entire duration of the function as
            // they can be deallocated at any time. Same for non-frozen shared references (see
            // <https://github.com/rust-lang/rust/pull/98017>), and for mutable references to
            // potentially self-referential types (see
            // <https://github.com/rust-lang/unsafe-code-guidelines/issues/381>). If LLVM had a way
            // to say "dereferenceable on entry" we could use it here.
            attrs.pointee_size = match kind {
                PointerKind::Box { .. }
                | PointerKind::SharedRef { frozen: false }
                | PointerKind::MutableRef { unpin: false } => Size::ZERO,
                PointerKind::SharedRef { frozen: true }
                | PointerKind::MutableRef { unpin: true } => pointee.size,
            };

            // The aliasing rules for `Box<T>` are still not decided, but currently we emit
            // `noalias` for it. This can be turned off using an unstable flag.
            // See https://github.com/rust-lang/unsafe-code-guidelines/issues/326
            let noalias_for_box = cx.tcx.sess.opts.unstable_opts.box_noalias;

            // LLVM prior to version 12 had known miscompiles in the presence of noalias attributes
            // (see #54878), so it was conditionally disabled, but we don't support earlier
            // versions at all anymore. We still support turning it off using -Zmutable-noalias.
            let noalias_mut_ref = cx.tcx.sess.opts.unstable_opts.mutable_noalias;

            // `&T` where `T` contains no `UnsafeCell<U>` is immutable, and can be marked as both
            // `readonly` and `noalias`, as LLVM's definition of `noalias` is based solely on memory
            // dependencies rather than pointer equality. However this only applies to arguments,
            // not return values.
            //
            // `&mut T` and `Box<T>` where `T: Unpin` are unique and hence `noalias`.
            let no_alias = match kind {
                PointerKind::SharedRef { frozen } => frozen,
                PointerKind::MutableRef { unpin } => unpin && noalias_mut_ref,
                PointerKind::Box { unpin } => unpin && noalias_for_box,
            };
            // We can never add `noalias` in return position; that LLVM attribute has some very surprising semantics
            // (see <https://github.com/rust-lang/unsafe-code-guidelines/issues/385#issuecomment-1368055745>).
            if no_alias && !is_return {
                attrs.set(ArgAttribute::NoAlias);
            }

            if matches!(kind, PointerKind::SharedRef { frozen: true }) && !is_return {
                attrs.set(ArgAttribute::ReadOnly);
            }
        }
    }
}

/// Ensure that the ABI makes basic sense.
fn fn_abi_sanity_check<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
    spec_abi: SpecAbi,
) {
    fn fn_arg_sanity_check<'tcx>(
        cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        spec_abi: SpecAbi,
        arg: &ArgAbi<'tcx, Ty<'tcx>>,
    ) {
        match &arg.mode {
            PassMode::Ignore => {}
            PassMode::Direct(_) => {
                // Here the Rust type is used to determine the actual ABI, so we have to be very
                // careful. Scalar/ScalarPair is fine, since backends will generally use
                // `layout.abi` and ignore everything else. We should just reject `Aggregate`
                // entirely here, but some targets need to be fixed first.
                if matches!(arg.layout.abi, Abi::Aggregate { .. }) {
                    // For an unsized type we'd only pass the sized prefix, so there is no universe
                    // in which we ever want to allow this.
                    assert!(
                        arg.layout.is_sized(),
                        "`PassMode::Direct` for unsized type in ABI: {:#?}",
                        fn_abi
                    );
                    // This really shouldn't happen even for sized aggregates, since
                    // `immediate_llvm_type` will use `layout.fields` to turn this Rust type into an
                    // LLVM type. This means all sorts of Rust type details leak into the ABI.
                    // However wasm sadly *does* currently use this mode so we have to allow it --
                    // but we absolutely shouldn't let any more targets do that.
                    // (Also see <https://github.com/rust-lang/rust/issues/115666>.)
                    //
                    // The unstable abi `PtxKernel` also uses Direct for now.
                    // It needs to switch to something else before stabilization can happen.
                    // (See issue: https://github.com/rust-lang/rust/issues/117271)
                    assert!(
                        matches!(&*cx.tcx.sess.target.arch, "wasm32" | "wasm64")
                            || matches!(spec_abi, SpecAbi::PtxKernel | SpecAbi::Unadjusted),
                        r#"`PassMode::Direct` for aggregates only allowed for "unadjusted" and "ptx-kernel" functions and on wasm\nProblematic type: {:#?}"#,
                        arg.layout,
                    );
                }
            }
            PassMode::Pair(_, _) => {
                // Similar to `Direct`, we need to make sure that backends use `layout.abi` and
                // ignore the rest of the layout.
                assert!(
                    matches!(arg.layout.abi, Abi::ScalarPair(..)),
                    "PassMode::Pair for type {}",
                    arg.layout.ty
                );
            }
            PassMode::Cast { .. } => {
                // `Cast` means "transmute to `CastType`"; that only makes sense for sized types.
                assert!(arg.layout.is_sized());
            }
            PassMode::Indirect { meta_attrs: None, .. } => {
                // No metadata, must be sized.
                // Conceptually, unsized arguments must be copied around, which requires dynamically
                // determining their size, which we cannot do without metadata. Consult
                // t-opsem before removing this check.
                assert!(arg.layout.is_sized());
            }
            PassMode::Indirect { meta_attrs: Some(_), on_stack, .. } => {
                // With metadata. Must be unsized and not on the stack.
                assert!(arg.layout.is_unsized() && !on_stack);
                // Also, must not be `extern` type.
                let tail = cx.tcx.struct_tail_with_normalize(arg.layout.ty, |ty| ty, || {});
                if matches!(tail.kind(), ty::Foreign(..)) {
                    // These types do not have metadata, so having `meta_attrs` is bogus.
                    // Conceptually, unsized arguments must be copied around, which requires dynamically
                    // determining their size. Therefore, we cannot allow `extern` types here. Consult
                    // t-opsem before removing this check.
                    panic!("unsized arguments must not be `extern` types");
                }
            }
        }
    }

    for arg in fn_abi.args.iter() {
        fn_arg_sanity_check(cx, fn_abi, spec_abi, arg);
    }
    fn_arg_sanity_check(cx, fn_abi, spec_abi, &fn_abi.ret);
}

// FIXME(eddyb) perhaps group the signature/type-containing (or all of them?)
// arguments of this method, into a separate `struct`.
#[tracing::instrument(level = "debug", skip(cx, caller_location, fn_def_id, force_thin_self_ptr))]
fn fn_abi_new_uncached<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    sig: ty::PolyFnSig<'tcx>,
    extra_args: &[Ty<'tcx>],
    caller_location: Option<Ty<'tcx>>,
    fn_def_id: Option<DefId>,
    // FIXME(eddyb) replace this with something typed, like an `enum`.
    force_thin_self_ptr: bool,
) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, &'tcx FnAbiError<'tcx>> {
    let sig = cx.tcx.normalize_erasing_late_bound_regions(cx.param_env, sig);

    let conv = conv_from_spec_abi(cx.tcx(), sig.abi, sig.c_variadic);

    let mut inputs = sig.inputs();
    let extra_args = if sig.abi == RustCall {
        assert!(!sig.c_variadic && extra_args.is_empty());

        if let Some(input) = sig.inputs().last() {
            if let ty::Tuple(tupled_arguments) = input.kind() {
                inputs = &sig.inputs()[0..sig.inputs().len() - 1];
                tupled_arguments
            } else {
                bug!(
                    "argument to function with \"rust-call\" ABI \
                        is not a tuple"
                );
            }
        } else {
            bug!(
                "argument to function with \"rust-call\" ABI \
                    is not a tuple"
            );
        }
    } else {
        assert!(sig.c_variadic || extra_args.is_empty());
        extra_args
    };

    let target = &cx.tcx.sess.target;
    let target_env_gnu_like = matches!(&target.env[..], "gnu" | "musl" | "uclibc");
    let win_x64_gnu = target.os == "windows" && target.arch == "x86_64" && target.env == "gnu";
    let linux_s390x_gnu_like =
        target.os == "linux" && target.arch == "s390x" && target_env_gnu_like;
    let linux_sparc64_gnu_like =
        target.os == "linux" && target.arch == "sparc64" && target_env_gnu_like;
    let linux_powerpc_gnu_like =
        target.os == "linux" && target.arch == "powerpc" && target_env_gnu_like;
    use SpecAbi::*;
    let rust_abi = matches!(sig.abi, RustIntrinsic | PlatformIntrinsic | Rust | RustCall);

    let is_drop_in_place =
        fn_def_id.is_some() && fn_def_id == cx.tcx.lang_items().drop_in_place_fn();

    let arg_of = |ty: Ty<'tcx>, arg_idx: Option<usize>| -> Result<_, &'tcx FnAbiError<'tcx>> {
        let span = tracing::debug_span!("arg_of");
        let _entered = span.enter();
        let is_return = arg_idx.is_none();
        let is_drop_target = is_drop_in_place && arg_idx == Some(0);
        let drop_target_pointee = is_drop_target.then(|| match ty.kind() {
            ty::RawPtr(ty::TypeAndMut { ty, .. }) => *ty,
            _ => bug!("argument to drop_in_place is not a raw ptr: {:?}", ty),
        });

        let layout =
            cx.layout_of(ty).map_err(|err| &*cx.tcx.arena.alloc(FnAbiError::Layout(*err)))?;
        let layout = if force_thin_self_ptr && arg_idx == Some(0) {
            // Don't pass the vtable, it's not an argument of the virtual fn.
            // Instead, pass just the data pointer, but give it the type `*const/mut dyn Trait`
            // or `&/&mut dyn Trait` because this is special-cased elsewhere in codegen
            make_thin_self_ptr(cx, layout)
        } else {
            layout
        };

        let mut arg = ArgAbi::new(cx, layout, |layout, scalar, offset| {
            let mut attrs = ArgAttributes::new();
            adjust_for_rust_scalar(
                *cx,
                &mut attrs,
                scalar,
                *layout,
                offset,
                is_return,
                drop_target_pointee,
            );
            attrs
        });

        if arg.layout.is_zst() {
            // For some forsaken reason, x86_64-pc-windows-gnu
            // doesn't ignore zero-sized struct arguments.
            // The same is true for {s390x,sparc64,powerpc}-unknown-linux-{gnu,musl,uclibc}.
            if is_return
                || rust_abi
                || (!win_x64_gnu
                    && !linux_s390x_gnu_like
                    && !linux_sparc64_gnu_like
                    && !linux_powerpc_gnu_like)
            {
                arg.mode = PassMode::Ignore;
            }
        }

        Ok(arg)
    };

    let mut fn_abi = FnAbi {
        ret: arg_of(sig.output(), None)?,
        args: inputs
            .iter()
            .copied()
            .chain(extra_args.iter().copied())
            .chain(caller_location)
            .enumerate()
            .map(|(i, ty)| arg_of(ty, Some(i)))
            .collect::<Result<_, _>>()?,
        c_variadic: sig.c_variadic,
        fixed_count: inputs.len() as u32,
        conv,
        can_unwind: fn_can_unwind(cx.tcx(), fn_def_id, sig.abi),
    };
    fn_abi_adjust_for_abi(cx, &mut fn_abi, sig.abi, fn_def_id)?;
    debug!("fn_abi_new_uncached = {:?}", fn_abi);
    fn_abi_sanity_check(cx, &fn_abi, sig.abi);
    Ok(cx.tcx.arena.alloc(fn_abi))
}

#[tracing::instrument(level = "trace", skip(cx))]
fn fn_abi_adjust_for_abi<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    fn_abi: &mut FnAbi<'tcx, Ty<'tcx>>,
    abi: SpecAbi,
    fn_def_id: Option<DefId>,
) -> Result<(), &'tcx FnAbiError<'tcx>> {
    if abi == SpecAbi::Unadjusted {
        // The "unadjusted" ABI passes aggregates in "direct" mode. That's fragile but needed for
        // some LLVM intrinsics.
        fn unadjust<'tcx>(arg: &mut ArgAbi<'tcx, Ty<'tcx>>) {
            // This still uses `PassMode::Pair` for ScalarPair types. That's unlikely to be intended,
            // but who knows what breaks if we change this now.
            if matches!(arg.layout.abi, Abi::Aggregate { .. }) {
                assert!(
                    arg.layout.abi.is_sized(),
                    "'unadjusted' ABI does not support unsized arguments"
                );
            }
            arg.make_direct_deprecated();
        }

        unadjust(&mut fn_abi.ret);
        for arg in fn_abi.args.iter_mut() {
            unadjust(arg);
        }
        return Ok(());
    }

    if abi == SpecAbi::Rust
        || abi == SpecAbi::RustCall
        || abi == SpecAbi::RustIntrinsic
        || abi == SpecAbi::PlatformIntrinsic
    {
        // Look up the deduced parameter attributes for this function, if we have its def ID and
        // we're optimizing in non-incremental mode. We'll tag its parameters with those attributes
        // as appropriate.
        let deduced_param_attrs = if cx.tcx.sess.opts.optimize != OptLevel::No
            && cx.tcx.sess.opts.incremental.is_none()
        {
            fn_def_id.map(|fn_def_id| cx.tcx.deduced_param_attrs(fn_def_id)).unwrap_or_default()
        } else {
            &[]
        };

        let fixup = |arg: &mut ArgAbi<'tcx, Ty<'tcx>>, arg_idx: Option<usize>| {
            if arg.is_ignore() {
                return;
            }

            match arg.layout.abi {
                Abi::Aggregate { .. } => {}

                // This is a fun case! The gist of what this is doing is
                // that we want callers and callees to always agree on the
                // ABI of how they pass SIMD arguments. If we were to *not*
                // make these arguments indirect then they'd be immediates
                // in LLVM, which means that they'd used whatever the
                // appropriate ABI is for the callee and the caller. That
                // means, for example, if the caller doesn't have AVX
                // enabled but the callee does, then passing an AVX argument
                // across this boundary would cause corrupt data to show up.
                //
                // This problem is fixed by unconditionally passing SIMD
                // arguments through memory between callers and callees
                // which should get them all to agree on ABI regardless of
                // target feature sets. Some more information about this
                // issue can be found in #44367.
                //
                // Note that the platform intrinsic ABI is exempt here as
                // that's how we connect up to LLVM and it's unstable
                // anyway, we control all calls to it in libstd.
                Abi::Vector { .. }
                    if abi != SpecAbi::PlatformIntrinsic
                        && cx.tcx.sess.target.simd_types_indirect =>
                {
                    arg.make_indirect();
                    return;
                }

                _ => return,
            }
            // Compute `Aggregate` ABI.

            let is_indirect_not_on_stack =
                matches!(arg.mode, PassMode::Indirect { on_stack: false, .. });
            assert!(is_indirect_not_on_stack, "{:?}", arg);

            let size = arg.layout.size;
            if !arg.layout.is_unsized() && size <= Pointer(AddressSpace::DATA).size(cx) {
                // We want to pass small aggregates as immediates, but using
                // an LLVM aggregate type for this leads to bad optimizations,
                // so we pick an appropriately sized integer type instead.
                arg.cast_to(Reg { kind: RegKind::Integer, size });
            }

            // If we deduced that this parameter was read-only, add that to the attribute list now.
            //
            // The `readonly` parameter only applies to pointers, so we can only do this if the
            // argument was passed indirectly. (If the argument is passed directly, it's an SSA
            // value, so it's implicitly immutable.)
            if let (Some(arg_idx), &mut PassMode::Indirect { ref mut attrs, .. }) =
                (arg_idx, &mut arg.mode)
            {
                // The `deduced_param_attrs` list could be empty if this is a type of function
                // we can't deduce any parameters for, so make sure the argument index is in
                // bounds.
                if let Some(deduced_param_attrs) = deduced_param_attrs.get(arg_idx) {
                    if deduced_param_attrs.read_only {
                        attrs.regular.insert(ArgAttribute::ReadOnly);
                        debug!("added deduced read-only attribute");
                    }
                }
            }
        };

        fixup(&mut fn_abi.ret, None);
        for (arg_idx, arg) in fn_abi.args.iter_mut().enumerate() {
            fixup(arg, Some(arg_idx));
        }
    } else {
        fn_abi
            .adjust_for_foreign_abi(cx, abi)
            .map_err(|err| &*cx.tcx.arena.alloc(FnAbiError::AdjustForForeignAbi(err)))?;
    }

    Ok(())
}

#[tracing::instrument(level = "debug", skip(cx))]
fn make_thin_self_ptr<'tcx>(
    cx: &(impl HasTyCtxt<'tcx> + HasParamEnv<'tcx>),
    layout: TyAndLayout<'tcx>,
) -> TyAndLayout<'tcx> {
    let tcx = cx.tcx();
    let fat_pointer_ty = if layout.is_unsized() {
        // unsized `self` is passed as a pointer to `self`
        // FIXME (mikeyhew) change this to use &own if it is ever added to the language
        Ty::new_mut_ptr(tcx, layout.ty)
    } else {
        match layout.abi {
            Abi::ScalarPair(..) | Abi::Scalar(..) => (),
            _ => bug!("receiver type has unsupported layout: {:?}", layout),
        }

        // In the case of Rc<Self>, we need to explicitly pass a *mut RcBox<Self>
        // with a Scalar (not ScalarPair) ABI. This is a hack that is understood
        // elsewhere in the compiler as a method on a `dyn Trait`.
        // To get the type `*mut RcBox<Self>`, we just keep unwrapping newtypes until we
        // get a built-in pointer type
        let mut fat_pointer_layout = layout;
        while !fat_pointer_layout.ty.is_unsafe_ptr() && !fat_pointer_layout.ty.is_ref() {
            fat_pointer_layout = fat_pointer_layout
                .non_1zst_field(cx)
                .expect("not exactly one non-1-ZST field in a `DispatchFromDyn` type")
                .1
        }

        fat_pointer_layout.ty
    };

    // we now have a type like `*mut RcBox<dyn Trait>`
    // change its layout to that of `*mut ()`, a thin pointer, but keep the same type
    // this is understood as a special case elsewhere in the compiler
    let unit_ptr_ty = Ty::new_mut_ptr(tcx, Ty::new_unit(tcx));

    TyAndLayout {
        ty: fat_pointer_ty,

        // NOTE(eddyb) using an empty `ParamEnv`, and `unwrap`-ing the `Result`
        // should always work because the type is always `*mut ()`.
        ..tcx.layout_of(ty::ParamEnv::reveal_all().and(unit_ptr_ty)).unwrap()
    }
}
