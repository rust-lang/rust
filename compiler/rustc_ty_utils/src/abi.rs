use std::{assert_matches, iter};

use rustc_abi::Primitive::Pointer;
use rustc_abi::{Align, BackendRepr, ExternAbi, PointerKind, Scalar, Size};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{self as hir, find_attr};
use rustc_middle::bug;
use rustc_middle::middle::deduced_param_attrs::DeducedParamAttrs;
use rustc_middle::query::Providers;
use rustc_middle::ty::layout::{
    FnAbiError, HasTyCtxt, HasTypingEnv, LayoutCx, LayoutOf, TyAndLayout, fn_can_unwind,
};
use rustc_middle::ty::{self, InstanceKind, Ty, TyCtxt};
use rustc_span::DUMMY_SP;
use rustc_span::def_id::DefId;
use rustc_target::callconv::{
    AbiMap, ArgAbi, ArgAttribute, ArgAttributes, ArgExtension, FnAbi, PassMode,
};
use tracing::debug;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        fn_abi_of_fn_ptr,
        fn_abi_of_instance_no_deduced_attrs,
        fn_abi_of_instance_raw,
        ..*providers
    };
}

// NOTE(eddyb) this is private to avoid using it from outside of
// `fn_abi_of_instance` - any other uses are either too high-level
// for `Instance` (e.g. typeck would use `Ty::fn_sig` instead),
// or should go through `FnAbi` instead, to avoid losing any
// adjustments `fn_abi_of_instance` might be performing.
#[tracing::instrument(level = "debug", skip(tcx, typing_env))]
fn fn_sig_for_fn_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
) -> ty::FnSig<'tcx> {
    if let InstanceKind::ThreadLocalShim(..) = instance.def {
        return tcx.mk_fn_sig(
            [],
            tcx.thread_local_ptr_ty(instance.def_id()),
            false,
            hir::Safety::Safe,
            rustc_abi::ExternAbi::Rust,
        );
    }

    let ty = instance.ty(tcx, typing_env);
    match *ty.kind() {
        ty::FnDef(def_id, args) => {
            let mut sig = tcx
                .instantiate_bound_regions_with_erased(tcx.fn_sig(def_id).instantiate(tcx, args));

            // Modify `fn(self, ...)` to `fn(self: *mut Self, ...)`.
            if let ty::InstanceKind::VTableShim(..) = instance.def {
                let mut inputs_and_output = sig.inputs_and_output.to_vec();
                inputs_and_output[0] = Ty::new_mut_ptr(tcx, inputs_and_output[0]);
                sig.inputs_and_output = tcx.mk_type_list(&inputs_and_output);
            }

            sig
        }
        ty::Closure(def_id, args) => {
            let sig = tcx.instantiate_bound_regions_with_erased(args.as_closure().sig());
            let env_ty = tcx.closure_env_ty(
                Ty::new_closure(tcx, def_id, args),
                args.as_closure().kind(),
                tcx.lifetimes.re_erased,
            );

            tcx.mk_fn_sig(
                iter::once(env_ty).chain(sig.inputs().iter().cloned()),
                sig.output(),
                sig.c_variadic,
                sig.safety,
                sig.abi,
            )
        }
        ty::CoroutineClosure(def_id, args) => {
            let coroutine_ty = Ty::new_coroutine_closure(tcx, def_id, args);
            let sig = args.as_coroutine_closure().coroutine_closure_sig();

            // When this `CoroutineClosure` comes from a `ConstructCoroutineInClosureShim`,
            // make sure we respect the `target_kind` in that shim.
            // FIXME(async_closures): This shouldn't be needed, and we should be populating
            // a separate def-id for these bodies.
            let mut coroutine_kind = args.as_coroutine_closure().kind();

            let env_ty =
                if let InstanceKind::ConstructCoroutineInClosureShim { receiver_by_ref, .. } =
                    instance.def
                {
                    coroutine_kind = ty::ClosureKind::FnOnce;

                    // Implementations of `FnMut` and `Fn` for coroutine-closures
                    // still take their receiver by ref.
                    if receiver_by_ref {
                        Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, coroutine_ty)
                    } else {
                        coroutine_ty
                    }
                } else {
                    tcx.closure_env_ty(coroutine_ty, coroutine_kind, tcx.lifetimes.re_erased)
                };

            let sig = tcx.instantiate_bound_regions_with_erased(sig);

            tcx.mk_fn_sig(
                iter::once(env_ty).chain([sig.tupled_inputs_ty]),
                sig.to_coroutine_given_kind_and_upvars(
                    tcx,
                    args.as_coroutine_closure().parent_args(),
                    tcx.coroutine_for_closure(def_id),
                    coroutine_kind,
                    tcx.lifetimes.re_erased,
                    args.as_coroutine_closure().tupled_upvars_ty(),
                    args.as_coroutine_closure().coroutine_captures_by_ref_ty(),
                ),
                sig.c_variadic,
                sig.safety,
                sig.abi,
            )
        }
        ty::Coroutine(did, args) => {
            let coroutine_kind = tcx.coroutine_kind(did).unwrap();
            let sig = args.as_coroutine().sig();

            let env_ty = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, ty);

            let pin_did = tcx.require_lang_item(LangItem::Pin, DUMMY_SP);
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

                    let poll_did = tcx.require_lang_item(LangItem::Poll, DUMMY_SP);
                    let poll_adt_ref = tcx.adt_def(poll_did);
                    let poll_args = tcx.mk_args(&[sig.return_ty.into()]);
                    let ret_ty = Ty::new_adt(tcx, poll_adt_ref, poll_args);

                    // We have to replace the `ResumeTy` that is used for type and borrow checking
                    // with `&mut Context<'_>` which is used in codegen.
                    #[cfg(debug_assertions)]
                    {
                        if let ty::Adt(resume_ty_adt, _) = sig.resume_ty.kind() {
                            let expected_adt =
                                tcx.adt_def(tcx.require_lang_item(LangItem::ResumeTy, DUMMY_SP));
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
                    let option_did = tcx.require_lang_item(LangItem::Option, DUMMY_SP);
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
                                tcx.adt_def(tcx.require_lang_item(LangItem::ResumeTy, DUMMY_SP));
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
                    let state_did = tcx.require_lang_item(LangItem::CoroutineState, DUMMY_SP);
                    let state_adt_ref = tcx.adt_def(state_did);
                    let state_args = tcx.mk_args(&[sig.yield_ty.into(), sig.return_ty.into()]);
                    let ret_ty = Ty::new_adt(tcx, state_adt_ref, state_args);

                    (Some(sig.resume_ty), ret_ty)
                }
            };

            if let Some(resume_ty) = resume_ty {
                tcx.mk_fn_sig(
                    [env_ty, resume_ty],
                    ret_ty,
                    false,
                    hir::Safety::Safe,
                    rustc_abi::ExternAbi::Rust,
                )
            } else {
                // `Iterator::next` doesn't have a `resume` argument.
                tcx.mk_fn_sig(
                    [env_ty],
                    ret_ty,
                    false,
                    hir::Safety::Safe,
                    rustc_abi::ExternAbi::Rust,
                )
            }
        }
        _ => bug!("unexpected type {:?} in Instance::fn_sig", ty),
    }
}

/// Describes a function for determination of its ABI.
struct FnAbiDesc<'tcx> {
    layout_cx: LayoutCx<'tcx>,
    sig: ty::FnSig<'tcx>,

    /// The function's definition, if its body can be used to deduce parameter attributes.
    determined_fn_def_id: Option<DefId>,
    caller_location: Option<Ty<'tcx>>,
    is_virtual_call: bool,
    extra_args: &'tcx [Ty<'tcx>],
}

impl<'tcx> FnAbiDesc<'tcx> {
    fn for_fn_ptr(
        tcx: TyCtxt<'tcx>,
        query: ty::PseudoCanonicalInput<'tcx, (ty::PolyFnSig<'tcx>, &'tcx ty::List<Ty<'tcx>>)>,
    ) -> Self {
        let ty::PseudoCanonicalInput { typing_env, value: (sig, extra_args) } = query;
        Self {
            layout_cx: LayoutCx::new(tcx, typing_env),
            sig: tcx.normalize_erasing_regions(
                typing_env,
                tcx.instantiate_bound_regions_with_erased(sig),
            ),
            // Parameter attributes can never be deduced for indirect calls, as there is no
            // function body available to use.
            determined_fn_def_id: None,
            caller_location: None,
            is_virtual_call: false,
            extra_args,
        }
    }

    fn for_instance(
        tcx: TyCtxt<'tcx>,
        query: ty::PseudoCanonicalInput<'tcx, (ty::Instance<'tcx>, &'tcx ty::List<Ty<'tcx>>)>,
    ) -> Self {
        let ty::PseudoCanonicalInput { typing_env, value: (instance, extra_args) } = query;
        let is_virtual_call = matches!(instance.def, ty::InstanceKind::Virtual(..));
        let is_tls_shim_call = matches!(instance.def, ty::InstanceKind::ThreadLocalShim(_));
        Self {
            layout_cx: LayoutCx::new(tcx, typing_env),
            sig: tcx.normalize_erasing_regions(
                typing_env,
                fn_sig_for_fn_abi(tcx, instance, typing_env),
            ),
            // Parameter attributes can be deduced from the bodies of neither:
            // - virtual calls, as they might call other functions from the vtable; nor
            // - TLS shims, as they would refer to the underlying static.
            determined_fn_def_id: (!is_virtual_call && !is_tls_shim_call)
                .then(|| instance.def_id()),
            caller_location: instance
                .def
                .requires_caller_location(tcx)
                .then(|| tcx.caller_location_ty()),
            is_virtual_call,
            extra_args,
        }
    }
}

fn fn_abi_of_fn_ptr<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::PseudoCanonicalInput<'tcx, (ty::PolyFnSig<'tcx>, &'tcx ty::List<Ty<'tcx>>)>,
) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, &'tcx FnAbiError<'tcx>> {
    let desc = FnAbiDesc::for_fn_ptr(tcx, query);
    fn_abi_new_uncached(desc)
}

fn fn_abi_of_instance_no_deduced_attrs<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::PseudoCanonicalInput<'tcx, (ty::Instance<'tcx>, &'tcx ty::List<Ty<'tcx>>)>,
) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, &'tcx FnAbiError<'tcx>> {
    let desc = FnAbiDesc::for_instance(tcx, query);
    fn_abi_new_uncached(desc)
}

fn fn_abi_of_instance_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::PseudoCanonicalInput<'tcx, (ty::Instance<'tcx>, &'tcx ty::List<Ty<'tcx>>)>,
) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, &'tcx FnAbiError<'tcx>> {
    // The `fn_abi_of_instance_no_deduced_attrs` query may have been called during CTFE, so we
    // delegate to it here in order to reuse (and, if necessary, augment) its result.
    tcx.fn_abi_of_instance_no_deduced_attrs(query).map(|fn_abi| {
        let params = FnAbiDesc::for_instance(tcx, query);
        // If the function's body can be used to deduce parameter attributes, then adjust such
        // "no deduced attrs" ABI; otherwise, return that ABI unadjusted.
        params.determined_fn_def_id.map_or(fn_abi, |fn_def_id| {
            fn_abi_adjust_for_deduced_attrs(&params.layout_cx, fn_abi, params.sig.abi, fn_def_id)
        })
    })
}

/// Returns argument attributes for a scalar argument.
///
/// `drop_target_pointee`, if set, causes pointer-typed scalars to be treated like mutable
/// references to the given type. This is used to special-case the argument of `ptr::drop_in_place`,
/// interpreting it as `&mut T` instead of `*mut T`, for the purposes of attributes (which is valid
/// as per its safety contract). If `drop_target_pointee` is set, `offset` must be 0 and `layout.ty`
/// must be a pointer to the given type. Note that for wide pointers this function is called twice
/// -- once for the data pointer and once for the vtable pointer. `drop_target_pointee` must only
/// be set for the data pointer.
fn arg_attrs_for_rust_scalar<'tcx>(
    cx: LayoutCx<'tcx>,
    scalar: Scalar,
    layout: TyAndLayout<'tcx>,
    offset: Size,
    is_return: bool,
    drop_target_pointee: Option<Ty<'tcx>>,
    def_id: Option<DefId>,
) -> ArgAttributes {
    let mut attrs = ArgAttributes::new();

    // Booleans are always a noundef i1 that needs to be zero-extended.
    if scalar.is_bool() {
        attrs.ext(ArgExtension::Zext);
        attrs.set(ArgAttribute::NoUndef);
        return attrs;
    }

    if !scalar.is_uninit_valid() {
        attrs.set(ArgAttribute::NoUndef);
    }

    // Only pointer types handled below.
    let Scalar::Initialized { value: Pointer(_), valid_range } = scalar else { return attrs };

    // Set `nonnull` if the validity range excludes zero, or for the argument to `drop_in_place`,
    // which must be nonnull per its documented safety requirements.
    if !valid_range.contains(0) || drop_target_pointee.is_some() {
        attrs.set(ArgAttribute::NonNull);
    }

    let tcx = cx.tcx();

    let drop_target_pointee_info = drop_target_pointee.and_then(|pointee| {
        assert_eq!(pointee, layout.ty.builtin_deref(true).unwrap());
        assert_eq!(offset, Size::ZERO);
        // The argument to `drop_in_place` is semantically equivalent to a mutable reference.
        let mutref = Ty::new_mut_ref(tcx, tcx.lifetimes.re_erased, pointee);
        let layout = cx.layout_of(mutref).unwrap();
        layout.pointee_info_at(&cx, offset)
    });

    if let Some(pointee) = drop_target_pointee_info.or_else(|| layout.pointee_info_at(&cx, offset))
    {
        if pointee.align > Align::ONE {
            attrs.pointee_align =
                Some(pointee.align.min(cx.tcx().sess.target.max_reliable_alignment()));
        }

        // LLVM dereferenceable attribute has unclear semantics on the return type,
        // they seem to be "dereferenceable until the end of the program", which is
        // generally, not valid for references. See
        // <https://rust-lang.zulipchat.com/#narrow/channel/136281-t-opsem/topic/LLVM.20dereferenceable.20on.20return.20type/with/563001493>
        if !is_return {
            attrs.pointee_size = pointee.size;
        };

        if let Some(kind) = pointee.safe {
            // The aliasing rules for `Box<T>` are still not decided, but currently we emit
            // `noalias` for it. This can be turned off using an unstable flag.
            // See https://github.com/rust-lang/unsafe-code-guidelines/issues/326
            let noalias_for_box = tcx.sess.opts.unstable_opts.box_noalias;

            // LLVM prior to version 12 had known miscompiles in the presence of noalias attributes
            // (see #54878), so it was conditionally disabled, but we don't support earlier
            // versions at all anymore. We still support turning it off using -Zmutable-noalias.
            let noalias_mut_ref = tcx.sess.opts.unstable_opts.mutable_noalias;

            // `&T` where `T` contains no `UnsafeCell<U>` is immutable, and can be marked as both
            // `readonly` and `noalias`, as LLVM's definition of `noalias` is based solely on memory
            // dependencies rather than pointer equality. However this only applies to arguments,
            // not return values.
            //
            // `&mut T` and `Box<T>` where `T: Unpin` are unique and hence `noalias`.
            let no_alias = match kind {
                PointerKind::SharedRef { frozen } => frozen,
                PointerKind::MutableRef { unpin } => unpin && noalias_mut_ref,
                PointerKind::Box { unpin, global } => unpin && global && noalias_for_box,
            };
            // We can never add `noalias` in return position; that LLVM attribute has some very surprising semantics
            // (see <https://github.com/rust-lang/unsafe-code-guidelines/issues/385#issuecomment-1368055745>).
            if no_alias && !is_return {
                attrs.set(ArgAttribute::NoAlias);

                // set writable if no_alias is set, it's a mutable reference and the feature is not disabled
                let no_writable = match def_id {
                    Some(def_id) => find_attr!(tcx, def_id, RustcNoWritable),
                    None => false, // If no def_id exists, there can't exist an attribute for that def_id so rustc_no_writable can't be set
                } || tcx.sess.opts.unstable_opts.no_writable;
                if matches!(kind, PointerKind::MutableRef { .. }) && !no_writable {
                    attrs.set(ArgAttribute::Writable);
                }
            }

            if matches!(kind, PointerKind::SharedRef { frozen: true }) && !is_return {
                attrs.set(ArgAttribute::ReadOnly);
                attrs.set(ArgAttribute::CapturesReadOnly);
            }
        }
    }

    attrs
}

/// Ensure that the ABI makes basic sense.
fn fn_abi_sanity_check<'tcx>(
    cx: &LayoutCx<'tcx>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
    spec_abi: ExternAbi,
) {
    fn fn_arg_sanity_check<'tcx>(
        cx: &LayoutCx<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        spec_abi: ExternAbi,
        arg: &ArgAbi<'tcx, Ty<'tcx>>,
    ) {
        let tcx = cx.tcx();

        if spec_abi.is_rustic_abi() {
            if arg.layout.is_zst() {
                // Casting closures to function pointers depends on ZST closure types being
                // omitted entirely in the calling convention.
                assert!(arg.is_ignore());
            }
            if let PassMode::Indirect { on_stack, .. } = arg.mode {
                assert!(!on_stack, "rust abi shouldn't use on_stack");
            }
        } else if arg.layout.pass_indirectly_in_non_rustic_abis(cx) {
            assert_matches!(
                arg.mode,
                PassMode::Indirect { on_stack: false, .. },
                "the {spec_abi} ABI does not implement `#[rustc_pass_indirectly_in_non_rustic_abis]`"
            );
        }

        match &arg.mode {
            PassMode::Ignore => {
                assert!(arg.layout.is_zst());
            }
            PassMode::Direct(_) => {
                // Here the Rust type is used to determine the actual ABI, so we have to be very
                // careful. Scalar/Vector is fine, since backends will generally use
                // `layout.backend_repr` and ignore everything else. We should just reject
                //`Aggregate` entirely here, but some targets need to be fixed first.
                match arg.layout.backend_repr {
                    BackendRepr::Scalar(_)
                    | BackendRepr::SimdVector { .. }
                    | BackendRepr::SimdScalableVector { .. } => {}
                    BackendRepr::ScalarPair(..) => {
                        panic!("`PassMode::Direct` used for ScalarPair type {}", arg.layout.ty)
                    }
                    BackendRepr::Memory { sized } => {
                        // For an unsized type we'd only pass the sized prefix, so there is no universe
                        // in which we ever want to allow this.
                        assert!(sized, "`PassMode::Direct` for unsized type in ABI: {:#?}", fn_abi);

                        // This really shouldn't happen even for sized aggregates, since
                        // `immediate_llvm_type` will use `layout.fields` to turn this Rust type into an
                        // LLVM type. This means all sorts of Rust type details leak into the ABI.
                        // The unadjusted ABI however uses Direct for all args. It is ill-specified,
                        // but unfortunately we need it for calling certain LLVM intrinsics.
                        assert!(
                            matches!(spec_abi, ExternAbi::Unadjusted),
                            "`PassMode::Direct` for aggregates only allowed for \"unadjusted\"\n\
                             Problematic type: {:#?}",
                            arg.layout,
                        );
                    }
                }
            }
            PassMode::Pair(_, _) => {
                // Similar to `Direct`, we need to make sure that backends use `layout.backend_repr`
                // and ignore the rest of the layout.
                assert!(
                    matches!(arg.layout.backend_repr, BackendRepr::ScalarPair(..)),
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
                let tail = tcx.struct_tail_for_codegen(arg.layout.ty, cx.typing_env);
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

#[tracing::instrument(
    level = "debug",
    skip(cx, caller_location, determined_fn_def_id, is_virtual_call)
)]
fn fn_abi_new_uncached<'tcx>(
    FnAbiDesc {
        layout_cx: ref cx,
        sig,
        determined_fn_def_id,
        caller_location,
        is_virtual_call,
        extra_args,
    }: FnAbiDesc<'tcx>,
) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, &'tcx FnAbiError<'tcx>> {
    let tcx = cx.tcx();

    let abi_map = AbiMap::from_target(&tcx.sess.target);
    let conv = abi_map.canonize_abi(sig.abi, sig.c_variadic).unwrap();

    let mut inputs = sig.inputs();
    let extra_args = if sig.abi == ExternAbi::RustCall {
        assert!(!sig.c_variadic && extra_args.is_empty());

        if let Some(input) = sig.inputs().last()
            && let ty::Tuple(tupled_arguments) = input.kind()
        {
            inputs = &sig.inputs()[0..sig.inputs().len() - 1];
            tupled_arguments
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

    let is_drop_in_place = determined_fn_def_id.is_some_and(|def_id| {
        tcx.is_lang_item(def_id, LangItem::DropInPlace)
            || tcx.is_lang_item(def_id, LangItem::AsyncDropInPlace)
    });

    let arg_of = |ty: Ty<'tcx>, arg_idx: Option<usize>| -> Result<_, &'tcx FnAbiError<'tcx>> {
        let span = tracing::debug_span!("arg_of");
        let _entered = span.enter();
        let is_return = arg_idx.is_none();
        let is_drop_target = is_drop_in_place && arg_idx == Some(0);
        let drop_target_pointee = is_drop_target.then(|| match ty.kind() {
            ty::RawPtr(ty, _) => *ty,
            _ => bug!("argument to drop_in_place is not a raw ptr: {:?}", ty),
        });

        let layout = cx.layout_of(ty).map_err(|err| &*tcx.arena.alloc(FnAbiError::Layout(*err)))?;
        let layout = if is_virtual_call && arg_idx == Some(0) {
            // Don't pass the vtable, it's not an argument of the virtual fn.
            // Instead, pass just the data pointer, but give it the type `*const/mut dyn Trait`
            // or `&/&mut dyn Trait` because this is special-cased elsewhere in codegen
            make_thin_self_ptr(cx, layout)
        } else {
            layout
        };

        Ok(ArgAbi::new(cx, layout, |scalar, offset| {
            arg_attrs_for_rust_scalar(
                *cx,
                scalar,
                layout,
                offset,
                is_return,
                // Only set `drop_target_pointee` for the data part of a wide pointer.
                // See `arg_attrs_for_rust_scalar` docs for more information.
                drop_target_pointee.filter(|_| offset == Size::ZERO),
                determined_fn_def_id,
            )
        }))
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
        // FIXME return false for tls shim
        can_unwind: fn_can_unwind(
            tcx,
            // Since `#[rustc_nounwind]` can change unwinding, we cannot infer unwinding by `fn_def_id` for a virtual call.
            determined_fn_def_id,
            sig.abi,
        ),
    };
    fn_abi_adjust_for_abi(cx, &mut fn_abi, sig.abi);
    debug!("fn_abi_new_uncached = {:?}", fn_abi);
    fn_abi_sanity_check(cx, &fn_abi, sig.abi);
    Ok(tcx.arena.alloc(fn_abi))
}

#[tracing::instrument(level = "trace", skip(cx))]
fn fn_abi_adjust_for_abi<'tcx>(
    cx: &LayoutCx<'tcx>,
    fn_abi: &mut FnAbi<'tcx, Ty<'tcx>>,
    abi: ExternAbi,
) {
    if abi == ExternAbi::Unadjusted {
        // The "unadjusted" ABI passes aggregates in "direct" mode. That's fragile but needed for
        // some LLVM intrinsics.
        fn unadjust<'tcx>(arg: &mut ArgAbi<'tcx, Ty<'tcx>>) {
            // This still uses `PassMode::Pair` for ScalarPair types. That's unlikely to be intended,
            // but who knows what breaks if we change this now.
            if matches!(arg.layout.backend_repr, BackendRepr::Memory { .. }) {
                assert!(
                    arg.layout.backend_repr.is_sized(),
                    "'unadjusted' ABI does not support unsized arguments"
                );
            }
            arg.make_direct_deprecated();
        }

        unadjust(&mut fn_abi.ret);
        for arg in fn_abi.args.iter_mut() {
            unadjust(arg);
        }
    } else if abi.is_rustic_abi() {
        fn_abi.adjust_for_rust_abi(cx);
    } else {
        fn_abi.adjust_for_foreign_abi(cx, abi);
    }
}

#[tracing::instrument(level = "trace", skip(cx))]
fn fn_abi_adjust_for_deduced_attrs<'tcx>(
    cx: &LayoutCx<'tcx>,
    fn_abi: &'tcx FnAbi<'tcx, Ty<'tcx>>,
    abi: ExternAbi,
    fn_def_id: DefId,
) -> &'tcx FnAbi<'tcx, Ty<'tcx>> {
    let tcx = cx.tcx();
    // Look up the deduced parameter attributes for this function, if we have its def ID.
    // We'll tag its parameters with those attributes as appropriate.
    let deduced = if abi.is_rustic_abi() { tcx.deduced_param_attrs(fn_def_id) } else { &[] };
    if deduced.is_empty() {
        fn_abi
    } else {
        let mut fn_abi = fn_abi.clone();
        apply_deduced_attributes(cx, deduced, 0, &mut fn_abi.ret);
        for (arg_idx, arg) in fn_abi.args.iter_mut().enumerate() {
            apply_deduced_attributes(cx, deduced, arg_idx + 1, arg);
        }
        debug!("fn_abi_adjust_for_deduced_attrs = {:?}", fn_abi);
        fn_abi_sanity_check(cx, &fn_abi, abi);
        tcx.arena.alloc(fn_abi)
    }
}

/// Apply deduced optimization attributes to a parameter using an indirect pass mode.
///
/// `deduced` is a possibly truncated list of deduced attributes for a return place and arguments.
/// `idx` the index of the parameter on the list (0 for a return place, and 1.. for arguments).
fn apply_deduced_attributes<'tcx>(
    cx: &LayoutCx<'tcx>,
    deduced: &[DeducedParamAttrs],
    idx: usize,
    arg: &mut ArgAbi<'tcx, Ty<'tcx>>,
) {
    // Deduction is performed under the assumption of the indirection pass mode.
    let PassMode::Indirect { ref mut attrs, .. } = arg.mode else {
        return;
    };
    // The default values at the tail of the list are not encoded.
    let Some(deduced) = deduced.get(idx) else {
        return;
    };
    if deduced.read_only(cx.tcx(), cx.typing_env, arg.layout.ty) {
        debug!("added deduced ReadOnly attribute");
        attrs.regular.insert(ArgAttribute::ReadOnly);
    }
    if deduced.captures_none(cx.tcx(), cx.typing_env, arg.layout.ty) {
        debug!("added deduced CapturesNone attribute");
        attrs.regular.insert(ArgAttribute::CapturesNone);
    }
}

#[tracing::instrument(level = "debug", skip(cx))]
fn make_thin_self_ptr<'tcx>(
    cx: &(impl HasTyCtxt<'tcx> + HasTypingEnv<'tcx>),
    layout: TyAndLayout<'tcx>,
) -> TyAndLayout<'tcx> {
    let tcx = cx.tcx();
    let wide_pointer_ty = if layout.is_unsized() {
        // unsized `self` is passed as a pointer to `self`
        // FIXME (mikeyhew) change this to use &own if it is ever added to the language
        Ty::new_mut_ptr(tcx, layout.ty)
    } else {
        match layout.backend_repr {
            BackendRepr::ScalarPair(..) | BackendRepr::Scalar(..) => (),
            _ => bug!("receiver type has unsupported layout: {:?}", layout),
        }

        // In the case of Rc<Self>, we need to explicitly pass a *mut RcInner<Self>
        // with a Scalar (not ScalarPair) ABI. This is a hack that is understood
        // elsewhere in the compiler as a method on a `dyn Trait`.
        // To get the type `*mut RcInner<Self>`, we just keep unwrapping newtypes until we
        // get a built-in pointer type
        let mut wide_pointer_layout = layout;
        while !wide_pointer_layout.ty.is_raw_ptr() && !wide_pointer_layout.ty.is_ref() {
            wide_pointer_layout = wide_pointer_layout
                .non_1zst_field(cx)
                .expect("not exactly one non-1-ZST field in a `DispatchFromDyn` type")
                .1
        }

        wide_pointer_layout.ty
    };

    // we now have a type like `*mut RcInner<dyn Trait>`
    // change its layout to that of `*mut ()`, a thin pointer, but keep the same type
    // this is understood as a special case elsewhere in the compiler
    let unit_ptr_ty = Ty::new_mut_ptr(tcx, tcx.types.unit);

    TyAndLayout {
        ty: wide_pointer_ty,

        // NOTE(eddyb) using an empty `ParamEnv`, and `unwrap`-ing the `Result`
        // should always work because the type is always `*mut ()`.
        ..tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(unit_ptr_ty)).unwrap()
    }
}
