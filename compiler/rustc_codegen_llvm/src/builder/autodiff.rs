use std::ptr;

use rustc_ast::expand::autodiff_attrs::{AutoDiffAttrs, DiffActivity, DiffMode};
use rustc_ast::expand::typetree::FncTree;
use rustc_codegen_ssa::common::TypeKind;
use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods, BuilderMethods};
use rustc_middle::ty::{Instance, PseudoCanonicalInput, TyCtxt, TypingEnv};
use rustc_middle::{bug, ty};
use rustc_target::callconv::PassMode;
use tracing::debug;

use crate::builder::{Builder, PlaceRef, UNNAMED};
use crate::context::SimpleCx;
use crate::declare::declare_simple_fn;
use crate::llvm;
use crate::llvm::{Metadata, TRUE, Type};
use crate::value::Value;

pub(crate) fn adjust_activity_to_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    typing_env: TypingEnv<'tcx>,
    da: &mut Vec<DiffActivity>,
) {
    let fn_ty = instance.ty(tcx, typing_env);

    if !matches!(fn_ty.kind(), ty::FnDef(..)) {
        bug!("expected fn def for autodiff, got {:?}", fn_ty);
    }

    // We don't actually pass the types back into the type system.
    // All we do is decide how to handle the arguments.
    let sig = fn_ty.fn_sig(tcx).skip_binder();

    // FIXME(Sa4dUs): pass proper varargs once we have support for differentiating variadic functions
    let Ok(fn_abi) =
        tcx.fn_abi_of_instance(typing_env.as_query_input((instance, ty::List::empty())))
    else {
        bug!("failed to get fn_abi of instance with empty varargs");
    };

    let mut new_activities = vec![];
    let mut new_positions = vec![];
    let mut del_activities = 0;
    for (i, ty) in sig.inputs().iter().enumerate() {
        if let Some(inner_ty) = ty.builtin_deref(true) {
            if inner_ty.is_slice() {
                // Now we need to figure out the size of each slice element in memory to allow
                // safety checks and usability improvements in the backend.
                let sty = match inner_ty.builtin_index() {
                    Some(sty) => sty,
                    None => {
                        panic!("slice element type unknown");
                    }
                };
                let pci = PseudoCanonicalInput {
                    typing_env: TypingEnv::fully_monomorphized(),
                    value: sty,
                };

                let layout = tcx.layout_of(pci);
                let elem_size = match layout {
                    Ok(layout) => layout.size,
                    Err(_) => {
                        bug!("autodiff failed to compute slice element size");
                    }
                };
                let elem_size: u32 = elem_size.bytes() as u32;

                // We know that the length will be passed as extra arg.
                if !da.is_empty() {
                    // We are looking at a slice. The length of that slice will become an
                    // extra integer on llvm level. Integers are always const.
                    // However, if the slice get's duplicated, we want to know to later check the
                    // size. So we mark the new size argument as FakeActivitySize.
                    // There is one FakeActivitySize per slice, so for convenience we store the
                    // slice element size in bytes in it. We will use the size in the backend.
                    let activity = match da[i] {
                        DiffActivity::DualOnly
                        | DiffActivity::Dual
                        | DiffActivity::Dualv
                        | DiffActivity::DuplicatedOnly
                        | DiffActivity::Duplicated => {
                            DiffActivity::FakeActivitySize(Some(elem_size))
                        }
                        DiffActivity::Const => DiffActivity::Const,
                        _ => bug!("unexpected activity for ptr/ref"),
                    };
                    new_activities.push(activity);
                    new_positions.push(i + 1);
                }

                continue;
            }
        }

        let pci = PseudoCanonicalInput { typing_env: TypingEnv::fully_monomorphized(), value: *ty };

        let layout = match tcx.layout_of(pci) {
            Ok(layout) => layout.layout,
            Err(_) => {
                bug!("failed to compute layout for type {:?}", ty);
            }
        };

        let pass_mode = &fn_abi.args[i].mode;

        // For ZST, just ignore and don't add its activity, as this arg won't be present
        // in the LLVM passed to Enzyme.
        // Some targets pass ZST indirectly in the C ABI, in that case, handle it as a normal arg
        // FIXME(Sa4dUs): Enforce ZST corresponding diff activity be `Const`
        if *pass_mode == PassMode::Ignore {
            del_activities += 1;
            da.remove(i);
        }

        // If the argument is lowered as a `ScalarPair`, we need to duplicate its activity.
        // Otherwise, the number of activities won't match the number of LLVM arguments and
        // this will lead to errors when verifying the Enzyme call.
        if let rustc_abi::BackendRepr::ScalarPair(_, _) = layout.backend_repr() {
            new_activities.push(da[i].clone());
            new_positions.push(i + 1 - del_activities);
        }
    }
    // now add the extra activities coming from slices
    // Reverse order to not invalidate the indices
    for _ in 0..new_activities.len() {
        let pos = new_positions.pop().unwrap();
        let activity = new_activities.pop().unwrap();
        da.insert(pos, activity);
    }
}

// When we call the `__enzyme_autodiff` or `__enzyme_fwddiff` function, we need to pass all the
// original inputs, as well as metadata and the additional shadow arguments.
// This function matches the arguments from the outer function to the inner enzyme call.
//
// This function also considers that Rust level arguments not always match the llvm-ir level
// arguments. A slice, `&[f32]`, for example, is represented as a pointer and a length on
// llvm-ir level. The number of activities matches the number of Rust level arguments, so we
// need to match those.
// FIXME(ZuseZ4): This logic is a bit more complicated than it should be, can we simplify it
// using iterators and peek()?
fn match_args_from_caller_to_enzyme<'ll, 'tcx>(
    cx: &SimpleCx<'ll>,
    builder: &mut Builder<'_, 'll, 'tcx>,
    width: u32,
    args: &mut Vec<&'ll llvm::Value>,
    inputs: &[DiffActivity],
    outer_args: &[&'ll llvm::Value],
) {
    debug!("matching autodiff arguments");
    // We now handle the issue that Rust level arguments not always match the llvm-ir level
    // arguments. A slice, `&[f32]`, for example, is represented as a pointer and a length on
    // llvm-ir level. The number of activities matches the number of Rust level arguments, so we
    // need to match those.
    // FIXME(ZuseZ4): This logic is a bit more complicated than it should be, can we simplify it
    // using iterators and peek()?
    let mut outer_pos: usize = 0;
    let mut activity_pos = 0;

    let enzyme_const = cx.create_metadata(b"enzyme_const");
    let enzyme_out = cx.create_metadata(b"enzyme_out");
    let enzyme_dup = cx.create_metadata(b"enzyme_dup");
    let enzyme_dupv = cx.create_metadata(b"enzyme_dupv");
    let enzyme_dupnoneed = cx.create_metadata(b"enzyme_dupnoneed");
    let enzyme_dupnoneedv = cx.create_metadata(b"enzyme_dupnoneedv");

    while activity_pos < inputs.len() {
        let diff_activity = inputs[activity_pos as usize];
        // Duplicated arguments received a shadow argument, into which enzyme will write the
        // gradient.
        let (activity, duplicated): (&Metadata, bool) = match diff_activity {
            DiffActivity::None => panic!("not a valid input activity"),
            DiffActivity::Const => (enzyme_const, false),
            DiffActivity::Active => (enzyme_out, false),
            DiffActivity::ActiveOnly => (enzyme_out, false),
            DiffActivity::Dual => (enzyme_dup, true),
            DiffActivity::Dualv => (enzyme_dupv, true),
            DiffActivity::DualOnly => (enzyme_dupnoneed, true),
            DiffActivity::DualvOnly => (enzyme_dupnoneedv, true),
            DiffActivity::Duplicated => (enzyme_dup, true),
            DiffActivity::DuplicatedOnly => (enzyme_dupnoneed, true),
            DiffActivity::FakeActivitySize(_) => (enzyme_const, false),
        };
        let outer_arg = outer_args[outer_pos];
        args.push(cx.get_metadata_value(activity));
        if matches!(diff_activity, DiffActivity::Dualv) {
            let next_outer_arg = outer_args[outer_pos + 1];
            let elem_bytes_size: u64 = match inputs[activity_pos + 1] {
                DiffActivity::FakeActivitySize(Some(s)) => s.into(),
                _ => bug!("incorrect Dualv handling recognized."),
            };
            // stride: sizeof(T) * n_elems.
            // n_elems is the next integer.
            // Now we multiply `4 * next_outer_arg` to get the stride.
            let mul = unsafe {
                llvm::LLVMBuildMul(
                    builder.llbuilder,
                    cx.get_const_int(cx.type_i64(), elem_bytes_size),
                    next_outer_arg,
                    UNNAMED,
                )
            };
            args.push(mul);
        }
        args.push(outer_arg);
        if duplicated {
            // We know that duplicated args by construction have a following argument,
            // so this can not be out of bounds.
            let next_outer_arg = outer_args[outer_pos + 1];
            let next_outer_ty = cx.val_ty(next_outer_arg);
            // FIXME(ZuseZ4): We should add support for Vec here too, but it's less urgent since
            // vectors behind references (&Vec<T>) are already supported. Users can not pass a
            // Vec by value for reverse mode, so this would only help forward mode autodiff.
            let slice = {
                if activity_pos + 1 >= inputs.len() {
                    // If there is no arg following our ptr, it also can't be a slice,
                    // since that would lead to a ptr, int pair.
                    false
                } else {
                    let next_activity = inputs[activity_pos + 1];
                    // We analyze the MIR types and add this dummy activity if we visit a slice.
                    matches!(next_activity, DiffActivity::FakeActivitySize(_))
                }
            };
            if slice {
                // A duplicated slice will have the following two outer_fn arguments:
                // (..., ptr1, int1, ptr2, int2, ...). We add the following llvm-ir to our __enzyme call:
                // (..., metadata! enzyme_dup, ptr, ptr, int1, ...).
                // FIXME(ZuseZ4): We will upstream a safety check later which asserts that
                // int2 >= int1, which means the shadow vector is large enough to store the gradient.
                assert_eq!(cx.type_kind(next_outer_ty), TypeKind::Integer);

                let iterations =
                    if matches!(diff_activity, DiffActivity::Dualv) { 1 } else { width as usize };

                for i in 0..iterations {
                    let next_outer_arg2 = outer_args[outer_pos + 2 * (i + 1)];
                    let next_outer_ty2 = cx.val_ty(next_outer_arg2);
                    assert_eq!(cx.type_kind(next_outer_ty2), TypeKind::Pointer);
                    let next_outer_arg3 = outer_args[outer_pos + 2 * (i + 1) + 1];
                    let next_outer_ty3 = cx.val_ty(next_outer_arg3);
                    assert_eq!(cx.type_kind(next_outer_ty3), TypeKind::Integer);
                    args.push(next_outer_arg2);
                }
                args.push(cx.get_metadata_value(enzyme_const));
                args.push(next_outer_arg);
                outer_pos += 2 + 2 * iterations;
                activity_pos += 2;
            } else {
                // A duplicated pointer will have the following two outer_fn arguments:
                // (..., ptr, ptr, ...). We add the following llvm-ir to our __enzyme call:
                // (..., metadata! enzyme_dup, ptr, ptr, ...).
                if matches!(diff_activity, DiffActivity::Duplicated | DiffActivity::DuplicatedOnly)
                {
                    assert_eq!(cx.type_kind(next_outer_ty), TypeKind::Pointer);
                }
                // In the case of Dual we don't have assumptions, e.g. f32 would be valid.
                args.push(next_outer_arg);
                outer_pos += 2;
                activity_pos += 1;

                // Now, if width > 1, we need to account for that
                for _ in 1..width {
                    let next_outer_arg = outer_args[outer_pos];
                    args.push(next_outer_arg);
                    outer_pos += 1;
                }
            }
        } else {
            // We do not differentiate with resprect to this argument.
            // We already added the metadata and argument above, so just increase the counters.
            outer_pos += 1;
            activity_pos += 1;
        }
    }
}

/// When differentiating `fn_to_diff`, take a `outer_fn` and generate another
/// function with expected naming and calling conventions[^1] which will be
/// discovered by the enzyme LLVM pass and its body populated with the differentiated
/// `fn_to_diff`. `outer_fn` is then modified to have a call to the generated
/// function and handle the differences between the Rust calling convention and
/// Enzyme.
/// [^1]: <https://enzyme.mit.edu/getting_started/CallingConvention/>
// FIXME(ZuseZ4): `outer_fn` should include upstream safety checks to
// cover some assumptions of enzyme/autodiff, which could lead to UB otherwise.
pub(crate) fn generate_enzyme_call<'ll, 'tcx>(
    builder: &mut Builder<'_, 'll, 'tcx>,
    cx: &SimpleCx<'ll>,
    fn_to_diff: &'ll Value,
    outer_name: &str,
    ret_ty: &'ll Type,
    fn_args: &[&'ll Value],
    attrs: AutoDiffAttrs,
    dest: PlaceRef<'tcx, &'ll Value>,
    fnc_tree: FncTree,
) {
    // We have to pick the name depending on whether we want forward or reverse mode autodiff.
    let mut ad_name: String = match attrs.mode {
        DiffMode::Forward => "__enzyme_fwddiff",
        DiffMode::Reverse => "__enzyme_autodiff",
        _ => panic!("logic bug in autodiff, unrecognized mode"),
    }
    .to_string();

    // add outer_name to ad_name to make it unique, in case users apply autodiff to multiple
    // functions. Unwrap will only panic, if LLVM gave us an invalid string.
    ad_name.push_str(outer_name);

    // Let us assume the user wrote the following function square:
    //
    // ```llvm
    // define double @square(double %x) {
    // entry:
    //  %0 = fmul double %x, %x
    //  ret double %0
    // }
    //
    // define double @dsquare(double %x) {
    //  return 0.0;
    // }
    // ```
    //
    // so our `outer_fn` will be `dsquare`. The unsafe code section below now removes the placeholder
    // code and inserts an autodiff call. We also add a declaration for the __enzyme_autodiff call.
    // Again, the arguments to all functions are slightly simplified.
    // ```llvm
    // declare double @__enzyme_autodiff_square(...)
    //
    // define double @dsquare(double %x) {
    // entry:
    //   %0 = tail call double (...) @__enzyme_autodiff_square(double (double)* nonnull @square, double %x)
    //   ret double %0
    // }
    // ```
    let enzyme_ty = unsafe { llvm::LLVMFunctionType(ret_ty, ptr::null(), 0, TRUE) };

    // FIXME(ZuseZ4): the CC/Addr/Vis values are best effort guesses, we should look at tests and
    // think a bit more about what should go here.
    let cc = unsafe { llvm::LLVMGetFunctionCallConv(fn_to_diff) };
    let ad_fn = declare_simple_fn(
        cx,
        &ad_name,
        llvm::CallConv::try_from(cc).expect("invalid callconv"),
        llvm::UnnamedAddr::No,
        llvm::Visibility::Default,
        enzyme_ty,
    );

    let num_args = llvm::LLVMCountParams(&fn_to_diff);
    let mut args = Vec::with_capacity(num_args as usize + 1);
    args.push(fn_to_diff);

    let enzyme_primal_ret = cx.create_metadata(b"enzyme_primal_return");
    if matches!(attrs.ret_activity, DiffActivity::Dual | DiffActivity::Active) {
        args.push(cx.get_metadata_value(enzyme_primal_ret));
    }
    if attrs.width > 1 {
        let enzyme_width = cx.create_metadata(b"enzyme_width");
        args.push(cx.get_metadata_value(enzyme_width));
        args.push(cx.get_const_int(cx.type_i64(), attrs.width as u64));
    }

    match_args_from_caller_to_enzyme(
        &cx,
        builder,
        attrs.width,
        &mut args,
        &attrs.input_activity,
        fn_args,
    );

    if !fnc_tree.args.is_empty() || !fnc_tree.ret.0.is_empty() {
        crate::typetree::add_tt(cx.llmod, cx.llcx, fn_to_diff, fnc_tree);
    }

    let call = builder.call(enzyme_ty, None, None, ad_fn, &args, None, None);

    let fn_ret_ty = builder.cx.val_ty(call);
    if fn_ret_ty != builder.cx.type_void() && fn_ret_ty != builder.cx.type_struct(&[], false) {
        // If we return void or an empty struct, then our caller (due to how we generated it)
        // does not expect a return value. As such, we have no pointer (or place) into which
        // we could store our value, and would store into an undef, which would cause UB.
        // As such, we just ignore the return value in those cases.
        builder.store_to_place(call, dest.val);
    }
}
