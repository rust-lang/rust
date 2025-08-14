use std::ptr;

use rustc_ast::expand::autodiff_attrs::{AutoDiffAttrs, AutoDiffItem, DiffActivity, DiffMode};
use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_ssa::common::TypeKind;
use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods, BuilderMethods};
use rustc_errors::FatalError;
use rustc_middle::bug;
use tracing::{debug, trace};

use crate::back::write::llvm_err;
use crate::builder::{Builder, PlaceRef, UNNAMED};
use crate::context::SimpleCx;
use crate::declare::declare_simple_fn;
use crate::errors::{AutoDiffWithoutEnable, LlvmError};
use crate::llvm::AttributePlace::Function;
use crate::llvm::{Metadata, True, Type};
use crate::value::Value;
use crate::{CodegenContext, LlvmCodegenBackend, ModuleLlvm, attributes, llvm};

fn _get_params(fnc: &Value) -> Vec<&Value> {
    let param_num = llvm::LLVMCountParams(fnc) as usize;
    let mut fnc_args: Vec<&Value> = vec![];
    fnc_args.reserve(param_num);
    unsafe {
        llvm::LLVMGetParams(fnc, fnc_args.as_mut_ptr());
        fnc_args.set_len(param_num);
    }
    fnc_args
}

fn _has_sret(fnc: &Value) -> bool {
    let num_args = llvm::LLVMCountParams(fnc) as usize;
    if num_args == 0 {
        false
    } else {
        unsafe { llvm::LLVMRustHasAttributeAtIndex(fnc, 0, llvm::AttributeKind::StructRet) }
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

    let enzyme_const = cx.create_metadata("enzyme_const".to_string()).unwrap();
    let enzyme_out = cx.create_metadata("enzyme_out".to_string()).unwrap();
    let enzyme_dup = cx.create_metadata("enzyme_dup".to_string()).unwrap();
    let enzyme_dupv = cx.create_metadata("enzyme_dupv".to_string()).unwrap();
    let enzyme_dupnoneed = cx.create_metadata("enzyme_dupnoneed".to_string()).unwrap();
    let enzyme_dupnoneedv = cx.create_metadata("enzyme_dupnoneedv".to_string()).unwrap();

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
    // ```
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
    let enzyme_ty = unsafe { llvm::LLVMFunctionType(ret_ty, ptr::null(), 0, True) };

    // FIXME(ZuseZ4): the CC/Addr/Vis values are best effort guesses, we should look at tests and
    // think a bit more about what should go here.
    // FIXME(Sa4dUs): have to find a way to get the cc, using `FastCallConv` for now
    let cc = 8;
    let ad_fn = declare_simple_fn(
        cx,
        &ad_name,
        llvm::CallConv::try_from(cc).expect("invalid callconv"),
        llvm::UnnamedAddr::No,
        llvm::Visibility::Default,
        enzyme_ty,
    );

    // Otherwise LLVM might inline our temporary code before the enzyme pass has a chance to
    // do it's work.
    let attr = llvm::AttributeKind::NoInline.create_attr(cx.llcx);
    attributes::apply_to_llfn(ad_fn, Function, &[attr]);

    let num_args = llvm::LLVMCountParams(&fn_to_diff);
    let mut args = Vec::with_capacity(num_args as usize + 1);
    args.push(fn_to_diff);

    let enzyme_primal_ret = cx.create_metadata("enzyme_primal_return".to_string()).unwrap();
    if matches!(attrs.ret_activity, DiffActivity::Dual | DiffActivity::Active) {
        args.push(cx.get_metadata_value(enzyme_primal_ret));
    }
    if attrs.width > 1 {
        let enzyme_width = cx.create_metadata("enzyme_width".to_string()).unwrap();
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

    let call = builder.call(enzyme_ty, None, None, ad_fn, &args, None, None);

    builder.store_to_place(call, dest.val);
}

pub(crate) fn differentiate<'ll>(
    module: &'ll ModuleCodegen<ModuleLlvm>,
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    diff_items: Vec<AutoDiffItem>,
) -> Result<(), FatalError> {
    // TODO(Sa4dUs): delete all this logic
    for item in &diff_items {
        trace!("{}", item);
    }

    let diag_handler = cgcx.create_dcx();

    let cx = SimpleCx::new(module.module_llvm.llmod(), module.module_llvm.llcx, cgcx.pointer_size);

    // First of all, did the user try to use autodiff without using the -Zautodiff=Enable flag?
    if !diff_items.is_empty()
        && !cgcx.opts.unstable_opts.autodiff.contains(&rustc_session::config::AutoDiff::Enable)
    {
        return Err(diag_handler.handle().emit_almost_fatal(AutoDiffWithoutEnable));
    }

    // Here we replace the placeholder code with the actual autodiff code, which calls Enzyme.
    for item in diff_items.iter() {
        let name = item.source.clone();
        let fn_def: Option<&llvm::Value> = cx.get_function(&name);
        let Some(_fn_def) = fn_def else {
            return Err(llvm_err(
                diag_handler.handle(),
                LlvmError::PrepareAutoDiff {
                    src: item.source.clone(),
                    target: item.target.clone(),
                    error: "could not find source function".to_owned(),
                },
            ));
        };
        debug!(?item.target);
        let fn_target: Option<&llvm::Value> = cx.get_function(&item.target);
        let Some(_fn_target) = fn_target else {
            return Err(llvm_err(
                diag_handler.handle(),
                LlvmError::PrepareAutoDiff {
                    src: item.source.clone(),
                    target: item.target.clone(),
                    error: "could not find target function".to_owned(),
                },
            ));
        };

        // generate_enzyme_call(&cx, fn_def, fn_target, item.attrs.clone());
    }

    // FIXME(ZuseZ4): support SanitizeHWAddress and prevent illegal/unsupported opts

    trace!("done with differentiate()");

    Ok(())
}
