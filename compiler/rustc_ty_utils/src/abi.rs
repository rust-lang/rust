use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_middle::ty::layout::{
    fn_can_unwind, FnAbiError, HasParamEnv, HasTyCtxt, LayoutCx, LayoutOf, TyAndLayout,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::config::OptLevel;
use rustc_span::def_id::DefId;
use rustc_target::abi::call::{
    ArgAbi, ArgAttribute, ArgAttributes, ArgExtension, Conv, FnAbi, PassMode, Reg, RegKind,
};
use rustc_target::abi::*;
use rustc_target::spec::abi::Abi as SpecAbi;

use std::iter;

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers { fn_abi_of_fn_ptr, fn_abi_of_instance, ..*providers };
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
    let ty = instance.ty(tcx, param_env);
    match *ty.kind() {
        ty::FnDef(..) => {
            // HACK(davidtwco,eddyb): This is a workaround for polymorphization considering
            // parameters unused if they show up in the signature, but not in the `mir::Body`
            // (i.e. due to being inside a projection that got normalized, see
            // `tests/ui/polymorphization/normalized_sig_types.rs`), and codegen not keeping
            // track of a polymorphization `ParamEnv` to allow normalizing later.
            //
            // We normalize the `fn_sig` again after substituting at a later point.
            let mut sig = match *ty.kind() {
                ty::FnDef(def_id, substs) => tcx
                    .fn_sig(def_id)
                    .map_bound(|fn_sig| {
                        tcx.normalize_erasing_regions(tcx.param_env(def_id), fn_sig)
                    })
                    .subst(tcx, substs),
                _ => unreachable!(),
            };

            if let ty::InstanceDef::VTableShim(..) = instance.def {
                // Modify `fn(self, ...)` to `fn(self: *mut Self, ...)`.
                sig = sig.map_bound(|mut sig| {
                    let mut inputs_and_output = sig.inputs_and_output.to_vec();
                    inputs_and_output[0] = tcx.mk_mut_ptr(inputs_and_output[0]);
                    sig.inputs_and_output = tcx.mk_type_list(&inputs_and_output);
                    sig
                });
            }
            sig
        }
        ty::Closure(def_id, substs) => {
            let sig = substs.as_closure().sig();

            let bound_vars = tcx.mk_bound_variable_kinds_from_iter(
                sig.bound_vars().iter().chain(iter::once(ty::BoundVariableKind::Region(ty::BrEnv))),
            );
            let br = ty::BoundRegion {
                var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                kind: ty::BoundRegionKind::BrEnv,
            };
            let env_region = tcx.mk_re_late_bound(ty::INNERMOST, br);
            let env_ty = tcx.closure_env_ty(def_id, substs, env_region).unwrap();

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
        ty::Generator(did, substs, _) => {
            let sig = substs.as_generator().poly_sig();

            let bound_vars = tcx.mk_bound_variable_kinds_from_iter(
                sig.bound_vars().iter().chain(iter::once(ty::BoundVariableKind::Region(ty::BrEnv))),
            );
            let br = ty::BoundRegion {
                var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                kind: ty::BoundRegionKind::BrEnv,
            };
            let env_ty = tcx.mk_mut_ref(tcx.mk_re_late_bound(ty::INNERMOST, br), ty);

            let pin_did = tcx.require_lang_item(LangItem::Pin, None);
            let pin_adt_ref = tcx.adt_def(pin_did);
            let pin_substs = tcx.mk_substs(&[env_ty.into()]);
            let env_ty = tcx.mk_adt(pin_adt_ref, pin_substs);

            let sig = sig.skip_binder();
            // The `FnSig` and the `ret_ty` here is for a generators main
            // `Generator::resume(...) -> GeneratorState` function in case we
            // have an ordinary generator, or the `Future::poll(...) -> Poll`
            // function in case this is a special generator backing an async construct.
            let (resume_ty, ret_ty) = if tcx.generator_is_async(did) {
                // The signature should be `Future::poll(_, &mut Context<'_>) -> Poll<Output>`
                let poll_did = tcx.require_lang_item(LangItem::Poll, None);
                let poll_adt_ref = tcx.adt_def(poll_did);
                let poll_substs = tcx.mk_substs(&[sig.return_ty.into()]);
                let ret_ty = tcx.mk_adt(poll_adt_ref, poll_substs);

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
                let context_mut_ref = tcx.mk_task_context();

                (context_mut_ref, ret_ty)
            } else {
                // The signature should be `Generator::resume(_, Resume) -> GeneratorState<Yield, Return>`
                let state_did = tcx.require_lang_item(LangItem::GeneratorState, None);
                let state_adt_ref = tcx.adt_def(state_did);
                let state_substs = tcx.mk_substs(&[sig.yield_ty.into(), sig.return_ty.into()]);
                let ret_ty = tcx.mk_adt(state_adt_ref, state_substs);

                (sig.resume_ty, ret_ty)
            };

            ty::Binder::bind_with_vars(
                tcx.mk_fn_sig(
                    [env_ty, resume_ty],
                    ret_ty,
                    false,
                    hir::Unsafety::Normal,
                    rustc_target::spec::abi::Abi::Rust,
                ),
                bound_vars,
            )
        }
        _ => bug!("unexpected type {:?} in Instance::fn_sig", ty),
    }
}

#[inline]
fn conv_from_spec_abi(tcx: TyCtxt<'_>, abi: SpecAbi) -> Conv {
    use rustc_target::spec::abi::Abi::*;
    match tcx.sess.target.adjust_abi(abi) {
        RustIntrinsic | PlatformIntrinsic | Rust | RustCall => Conv::Rust,
        RustCold => Conv::RustCold,

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
        AmdGpuKernel => Conv::AmdGpuKernel,
        AvrInterrupt => Conv::AvrInterrupt,
        AvrNonBlockingInterrupt => Conv::AvrNonBlockingInterrupt,
        Wasm => Conv::C,

        // These API constants ought to be more specific...
        Cdecl { .. } => Conv::C,
    }
}

fn fn_abi_of_fn_ptr<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, (ty::PolyFnSig<'tcx>, &'tcx ty::List<Ty<'tcx>>)>,
) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, FnAbiError<'tcx>> {
    let (param_env, (sig, extra_args)) = query.into_parts();

    let cx = LayoutCx { tcx, param_env };
    fn_abi_new_uncached(&cx, sig, extra_args, None, None, false)
}

fn fn_abi_of_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, (ty::Instance<'tcx>, &'tcx ty::List<Ty<'tcx>>)>,
) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, FnAbiError<'tcx>> {
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
    let Scalar::Initialized { value: Pointer(_), valid_range} = scalar else { return };

    if !valid_range.contains(0) {
        attrs.set(ArgAttribute::NonNull);
    }

    if let Some(pointee) = layout.pointee_info_at(&cx, offset) {
        if let Some(kind) = pointee.safe {
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
) -> Result<&'tcx FnAbi<'tcx, Ty<'tcx>>, FnAbiError<'tcx>> {
    let sig = cx.tcx.normalize_erasing_late_bound_regions(cx.param_env, sig);

    let conv = conv_from_spec_abi(cx.tcx(), sig.abi);

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

    let arg_of = |ty: Ty<'tcx>, arg_idx: Option<usize>| -> Result<_, FnAbiError<'tcx>> {
        let span = tracing::debug_span!("arg_of");
        let _entered = span.enter();
        let is_return = arg_idx.is_none();

        let layout = cx.layout_of(ty)?;
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
            adjust_for_rust_scalar(*cx, &mut attrs, scalar, *layout, offset, is_return);
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
    Ok(cx.tcx.arena.alloc(fn_abi))
}

#[tracing::instrument(level = "trace", skip(cx))]
fn fn_abi_adjust_for_abi<'tcx>(
    cx: &LayoutCx<'tcx, TyCtxt<'tcx>>,
    fn_abi: &mut FnAbi<'tcx, Ty<'tcx>>,
    abi: SpecAbi,
    fn_def_id: Option<DefId>,
) -> Result<(), FnAbiError<'tcx>> {
    if abi == SpecAbi::Unadjusted {
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

            let size = arg.layout.size;
            if arg.layout.is_unsized() || size > Pointer(AddressSpace::DATA).size(cx) {
                arg.make_indirect();
            } else {
                // We want to pass small aggregates as immediates, but using
                // a LLVM aggregate type for this leads to bad optimizations,
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
        fn_abi.adjust_for_foreign_abi(cx, abi)?;
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
        tcx.mk_mut_ptr(layout.ty)
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
        'descend_newtypes: while !fat_pointer_layout.ty.is_unsafe_ptr()
            && !fat_pointer_layout.ty.is_region_ptr()
        {
            for i in 0..fat_pointer_layout.fields.count() {
                let field_layout = fat_pointer_layout.field(cx, i);

                if !field_layout.is_zst() {
                    fat_pointer_layout = field_layout;
                    continue 'descend_newtypes;
                }
            }

            bug!("receiver has no non-zero-sized fields {:?}", fat_pointer_layout);
        }

        fat_pointer_layout.ty
    };

    // we now have a type like `*mut RcBox<dyn Trait>`
    // change its layout to that of `*mut ()`, a thin pointer, but keep the same type
    // this is understood as a special case elsewhere in the compiler
    let unit_ptr_ty = tcx.mk_mut_ptr(tcx.mk_unit());

    TyAndLayout {
        ty: fat_pointer_ty,

        // NOTE(eddyb) using an empty `ParamEnv`, and `unwrap`-ing the `Result`
        // should always work because the type is always `*mut ()`.
        ..tcx.layout_of(ty::ParamEnv::reveal_all().and(unit_ptr_ty)).unwrap()
    }
}
