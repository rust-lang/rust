use std::ptr;

use rustc_ast::expand::autodiff_attrs::{AutoDiffAttrs, AutoDiffItem, DiffActivity, DiffMode};
use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_ssa::back::write::ModuleConfig;
use rustc_errors::{DiagCtxt, FatalError};
use tracing::{debug, trace};

use crate::back::write::llvm_err;
use crate::builder::SBuilder;
use crate::context::SimpleCx;
use crate::declare::declare_simple_fn;
use crate::errors::{AutoDiffUnusedArgs, AutoDiffWithoutEnable, LlvmError};
use crate::llvm::AttributePlace::Function;
use crate::llvm::{Metadata, True};
use crate::value::Value;
use crate::{CodegenContext, LlvmCodegenBackend, ModuleLlvm, attributes, llvm};

fn get_params(fnc: &Value) -> Vec<&Value> {
    unsafe {
        let param_num = llvm::LLVMCountParams(fnc) as usize;
        let mut fnc_args: Vec<&Value> = vec![];
        fnc_args.reserve(param_num);
        llvm::LLVMGetParams(fnc, fnc_args.as_mut_ptr());
        fnc_args.set_len(param_num);
        fnc_args
    }
}

// A helper object to make sure, that we pass all of the input and output arguments of the outer
// wrapper into the inner enzyme call.
struct FunctionArgs {
    input_args: Vec<bool>,
    return_arg: bool,
    has_sret: bool,
}


impl FunctionArgs {
    fn fully_used(&self) -> bool {
        self.input_args.iter().all(|x| *x) && self.return_arg
    }

    fn use_input(&mut self, idx: usize) {
        assert!(!self.input_args[idx]);
        self.input_args[idx] = true;
    }

    fn use_output(&mut self) {
        assert!(!self.return_arg);
        self.return_arg = true;
    }

    fn has_sret(&self) -> bool {
        self.has_sret
    }
}

impl<'ll> From<(&'ll llvm::Context, &'ll Value)> for FunctionArgs {
    fn from(wrapper: (&'ll llvm::Context, &'ll Value)) -> Self {
        let (llcx, fnc) = wrapper;
        let num_args = unsafe { llvm::LLVMCountParams(fnc) as usize };
        let input_args = vec![false; num_args];

        let fn_ty = unsafe { llvm::LLVMGlobalGetValueType(fnc) };
        let ret_ty = unsafe { llvm::LLVMGetReturnType(fn_ty) };
        let void_ty = unsafe { llvm::LLVMVoidTypeInContext(llcx) };
        let return_arg = ret_ty != void_ty;


        let has_sret = if num_args == 0 {
            false
        } else {
            unsafe { llvm::LLVMRustHasAttributeAtIndex(fnc, 0, llvm::AttributeKind::StructRet) }
        };
        if has_sret {
            dbg!("has sret");
        } else {
            dbg!("no sret");
        }

        FunctionArgs { input_args, return_arg, has_sret }
    }
}

// The drop implementation makes sure that when we're done, all input and output args are matched.
//impl Drop for FunctionArgs {
//    fn drop(&mut self) {
//        assert!(self.fully_used());
//    }
//}

/// When differentiating `fn_to_diff`, take a `outer_fn` and generate another
/// function with expected naming and calling conventions[^1] which will be
/// discovered by the enzyme LLVM pass and its body populated with the differentiated
/// `fn_to_diff`. `outer_fn` is then modified to have a call to the generated
/// function and handle the differences between the Rust calling convention and
/// Enzyme.
/// [^1]: <https://enzyme.mit.edu/getting_started/CallingConvention/>
// FIXME(ZuseZ4): `outer_fn` should include upstream safety checks to
// cover some assumptions of enzyme/autodiff, which could lead to UB otherwise.
fn generate_enzyme_call<'ll>(
    _dcx: &DiagCtxt,
    cx: &SimpleCx<'ll>,
    fn_to_diff: &'ll Value,
    outer_fn: &'ll Value,
    attrs: AutoDiffAttrs,
) -> Result<(), FatalError>
{
    let inputs = attrs.input_activity;
    let output = attrs.ret_activity;

    let fa = FunctionArgs::from((cx.llcx, outer_fn));

    // We have to pick the name depending on whether we want forward or reverse mode autodiff.
    let mut ad_name: String = match attrs.mode {
        DiffMode::Forward => "__enzyme_fwddiff",
        DiffMode::Reverse => "__enzyme_autodiff",
        _ => panic!("logic bug in autodiff, unrecognized mode"),
    }
    .to_string();

    // add outer_fn name to ad_name to make it unique, in case users apply autodiff to multiple
    // functions. Unwrap will only panic, if LLVM gave us an invalid string.
    let name = llvm::get_value_name(outer_fn);
    let outer_fn_name = std::str::from_utf8(name).unwrap();
    ad_name.push_str(outer_fn_name);

    // Let us assume the user wrote the following function square:
    //
    // ```llvm
    // define double @square(double %x) {
    // entry:
    //  %0 = fmul double %x, %x
    //  ret double %0
    // }
    // ```
    //
    // The user now applies autodiff to the function square, in which case fn_to_diff will be `square`.
    // Our macro generates the following placeholder code (slightly simplified):
    //
    // ```llvm
    // define double @dsquare(double %x) {
    //  ; placeholder code
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
    unsafe {
        // On LLVM-IR, we can luckily declare __enzyme_ functions without specifying the input
        // arguments. We do however need to declare them with their correct return type.
        // We already figured the correct return type out in our frontend, when generating the outer_fn,
        // so we can now just go ahead and use that. FIXME(ZuseZ4): This doesn't handle sret yet.
        let fn_ty = llvm::LLVMGlobalGetValueType(outer_fn);
        let ret_ty = llvm::LLVMGetReturnType(fn_ty);

        dbg!(&outer_fn);
        dbg!(&fn_ty);

        // LLVM can figure out the input types on it's own, so we take a shortcut here.
        let enzyme_ty = llvm::LLVMFunctionType(ret_ty, ptr::null(), 0, True);

        //FIXME(ZuseZ4): the CC/Addr/Vis values are best effort guesses, we should look at tests and
        // think a bit more about what should go here.
        let cc = llvm::LLVMGetFunctionCallConv(outer_fn);
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

        // first, remove all calls from fnc
        let entry = llvm::LLVMGetFirstBasicBlock(outer_fn);
        let br = llvm::LLVMRustGetTerminator(entry);
        llvm::LLVMRustEraseInstFromParent(br);

        let last_inst = llvm::LLVMRustGetLastInstruction(entry).unwrap();
        let mut builder = SBuilder::build(cx, entry);

        let num_args = llvm::LLVMCountParams(&fn_to_diff);
        let mut args = Vec::with_capacity(num_args as usize + 1);
        args.push(fn_to_diff);

        let enzyme_const = cx.create_metadata("enzyme_const".to_string()).unwrap();
        let enzyme_out = cx.create_metadata("enzyme_out".to_string()).unwrap();
        let enzyme_dup = cx.create_metadata("enzyme_dup".to_string()).unwrap();
        let enzyme_dupnoneed = cx.create_metadata("enzyme_dupnoneed".to_string()).unwrap();
        let enzyme_primal_ret = cx.create_metadata("enzyme_primal_return".to_string()).unwrap();
        let enzyme_width = cx.create_metadata("enzyme_width".to_string()).unwrap();

        // FIXME(ZuseZ4): Find out, how enzyme_primal_ret and enzyme_width are combinable.
        if attrs.width > 1 {
            args.push(cx.get_metadata_value(enzyme_width));
            args.push(cx.get_const_i64(attrs.width as u64));
        }

        match output {
            DiffActivity::Dual => {
                args.push(cx.get_metadata_value(enzyme_primal_ret));
            }
            DiffActivity::Active => {
                args.push(cx.get_metadata_value(enzyme_primal_ret));
            }
            _ => {}
        }

        debug!("matching autodiff arguments");
        // We now handle the issue that Rust level arguments not always match the llvm-ir level
        // arguments. A slice, `&[f32]`, for example, is represented as a pointer and a length on
        // llvm-ir level. The number of activities matches the number of Rust level arguments, so we
        // need to match those.
        // FIXME(ZuseZ4): This logic is a bit more complicated than it should be, can we simplify it
        // using iterators and peek()?
        let mut outer_pos: usize = 0;
        let mut activity_pos = 0;

        if fa.has_sret() {
            // Then the first outer arg is the sret pointer. Enzyme doesn't know about sret, so the
            // inner function will still return something. We increase our outer_pos by one,
            // and once we're done with all other args we will take the return of the inner call and
            // update the sret pointer with it
            outer_pos = 1;
        }

        let outer_args: Vec<&llvm::Value> = get_params(outer_fn);
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
                DiffActivity::DualOnly => (enzyme_dupnoneed, true),
                DiffActivity::Duplicated => (enzyme_dup, true),
                DiffActivity::DuplicatedOnly => (enzyme_dupnoneed, true),
                DiffActivity::FakeActivitySize => (enzyme_const, false),
            };
            let outer_arg = outer_args[outer_pos];
            args.push(cx.get_metadata_value(activity));
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
                        next_activity == DiffActivity::FakeActivitySize
                    }
                };
                if slice {
                    // A duplicated slice will have the following two outer_fn arguments:
                    // (..., ptr1, int1, ptr2, int2, ...). We add the following llvm-ir to our __enzyme call:
                    // (..., metadata! enzyme_dup, ptr, ptr, int1, ...).
                    // FIXME(ZuseZ4): We will upstream a safety check later which asserts that
                    // int2 >= int1, which means the shadow vector is large enough to store the gradient.
                    assert!(llvm::LLVMRustGetTypeKind(next_outer_ty) == llvm::TypeKind::Integer);

                    for _ in 0..attrs.width {
                        let next_outer_arg2 = outer_args[outer_pos + 2];
                        let next_outer_ty2 = cx.val_ty(next_outer_arg2);
                        assert!(llvm::LLVMRustGetTypeKind(next_outer_ty2) == llvm::TypeKind::Pointer);
                        let next_outer_arg3 = outer_args[outer_pos + 3];
                        let next_outer_ty3 = cx.val_ty(next_outer_arg3);
                        assert!(llvm::LLVMRustGetTypeKind(next_outer_ty3) == llvm::TypeKind::Integer);
                        args.push(next_outer_arg2);
                    }


                    args.push(cx.get_metadata_value(enzyme_const));
                    args.push(next_outer_arg);
                    outer_pos += 2 + 2 * attrs.width as usize;
                    activity_pos += 2;
                } else {
                    // A duplicated pointer will have the following two outer_fn arguments:
                    // (..., ptr, ptr, ...). We add the following llvm-ir to our __enzyme call:
                    // (..., metadata! enzyme_dup, ptr, ptr, ...).
                    if matches!(
                        diff_activity,
                        DiffActivity::Duplicated | DiffActivity::DuplicatedOnly
                    ) {
                        assert!(
                            llvm::LLVMRustGetTypeKind(next_outer_ty) == llvm::TypeKind::Pointer
                        );
                    }
                    // In the case of Dual we don't have assumptions, e.g. f32 would be valid.
                    args.push(next_outer_arg);
                    outer_pos += 2;
                    activity_pos += 1;

                    // Now, if width > 1, we need to account for that
                    for _ in 1..attrs.width {
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

        let call = builder.call(enzyme_ty, ad_fn, &args, None);

        // This part is a bit iffy. LLVM requires that a call to an inlineable function has some
        // metadata attachted to it, but we just created this code oota. Given that the
        // differentiated function already has partly confusing metadata, and given that this
        // affects nothing but the auttodiff IR, we take a shortcut and just steal metadata from the
        // dummy code which we inserted at a higher level.
        // FIXME(ZuseZ4): Work with Enzyme core devs to clarify what debug metadata issues we have,
        // and how to best improve it for enzyme core and rust-enzyme.
        let md_ty = cx.get_md_kind_id("dbg");
        if llvm::LLVMRustHasMetadata(last_inst, md_ty) {
            let md = llvm::LLVMRustDIGetInstMetadata(last_inst)
                .expect("failed to get instruction metadata");
            let md_todiff = cx.get_metadata_value(md);
            llvm::LLVMSetMetadata(call, md_ty, md_todiff);
        } else {
            // We don't panic, since depending on whether we are in debug or release mode, we might
            // have no debug info to copy, which would then be ok.
            trace!("no dbg info");
        }

        // Now that we copied the metadata, get rid of dummy code.
        llvm::LLVMRustEraseInstUntilInclusive(entry, last_inst);

        if cx.val_ty(call) == cx.type_void() {
            builder.ret_void();
        } else {
            builder.ret(call);
        }

        // Let's crash in case that we messed something up above and generated invalid IR.
        llvm::LLVMRustVerifyFunction(
            outer_fn,
            llvm::LLVMRustVerifierFailureAction::LLVMAbortProcessAction,
        );
    }

    //if !fa.fully_used() {
    //    return Err(dcx.handle().emit_almost_fatal(AutoDiffUnusedArgs));
    //}
    Ok(())
}

pub(crate) fn differentiate<'ll>(
    module: &'ll ModuleCodegen<ModuleLlvm>,
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    diff_items: Vec<AutoDiffItem>,
    _config: &ModuleConfig,
) -> Result<(), FatalError> {
    for item in &diff_items {
        trace!("{}", item);
    }

    let diag_handler = cgcx.create_dcx();
    let cx = SimpleCx { llmod: module.module_llvm.llmod(), llcx: module.module_llvm.llcx };

    // First of all, did the user try to use autodiff without using the -Zautodiff=Enable flag?
    if !diff_items.is_empty()
        && !cgcx.opts.unstable_opts.autodiff.contains(&rustc_session::config::AutoDiff::Enable)
    {
        return Err(diag_handler.handle().emit_almost_fatal(AutoDiffWithoutEnable));
    }

    // Before dumping the module, we want all the TypeTrees to become part of the module.
    for item in diff_items.iter() {
        let name = item.source.clone();
        let fn_def: Option<&llvm::Value> = cx.get_function(&name);
        let Some(fn_def) = fn_def else {
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
        let Some(fn_target) = fn_target else {
            return Err(llvm_err(
                diag_handler.handle(),
                LlvmError::PrepareAutoDiff {
                    src: item.source.clone(),
                    target: item.target.clone(),
                    error: "could not find target function".to_owned(),
                },
            ));
        };

        generate_enzyme_call(&diag_handler, &cx, fn_def, fn_target, item.attrs.clone())?;
    }

    // FIXME(ZuseZ4): support SanitizeHWAddress and prevent illegal/unsupported opts

    trace!("done with differentiate()");

    Ok(())
}
