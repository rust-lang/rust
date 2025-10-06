use libc::c_uint;
use rustc_ast::expand::allocator::{
    ALLOCATOR_METHODS, AllocatorKind, AllocatorTy, NO_ALLOC_SHIM_IS_UNSTABLE,
    alloc_error_handler_name, default_fn_name, global_fn_name,
};
use rustc_codegen_ssa::traits::BaseTypeCodegenMethods as _;
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{DebugInfo, OomStrategy};
use rustc_span::sym;
use rustc_symbol_mangling::mangle_internal_symbol;

use crate::attributes::llfn_attrs_from_instance;
use crate::builder::SBuilder;
use crate::declare::declare_simple_fn;
use crate::llvm::{self, FALSE, FromGeneric, TRUE, Type, Value};
use crate::{SimpleCx, attributes, debuginfo};

pub(crate) unsafe fn codegen(
    tcx: TyCtxt<'_>,
    cx: SimpleCx<'_>,
    module_name: &str,
    kind: AllocatorKind,
    alloc_error_handler_kind: AllocatorKind,
) {
    let usize = match tcx.sess.target.pointer_width {
        16 => cx.type_i16(),
        32 => cx.type_i32(),
        64 => cx.type_i64(),
        tws => bug!("Unsupported target word size for int: {}", tws),
    };
    let i8 = cx.type_i8();
    let i8p = cx.type_ptr();

    if kind == AllocatorKind::Default {
        for method in ALLOCATOR_METHODS {
            let mut args = Vec::with_capacity(method.inputs.len());
            for input in method.inputs.iter() {
                match input.ty {
                    AllocatorTy::Layout => {
                        args.push(usize); // size
                        args.push(usize); // align
                    }
                    AllocatorTy::Ptr => args.push(i8p),
                    AllocatorTy::Usize => args.push(usize),

                    AllocatorTy::ResultPtr | AllocatorTy::Unit => panic!("invalid allocator arg"),
                }
            }
            let output = match method.output {
                AllocatorTy::ResultPtr => Some(i8p),
                AllocatorTy::Unit => None,

                AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => {
                    panic!("invalid allocator output")
                }
            };

            let from_name = mangle_internal_symbol(tcx, &global_fn_name(method.name));
            let to_name = mangle_internal_symbol(tcx, &default_fn_name(method.name));

            let alloc_attr_flag = match method.name {
                sym::alloc => CodegenFnAttrFlags::ALLOCATOR,
                sym::dealloc => CodegenFnAttrFlags::DEALLOCATOR,
                sym::realloc => CodegenFnAttrFlags::REALLOCATOR,
                sym::alloc_zeroed => CodegenFnAttrFlags::ALLOCATOR_ZEROED,
                _ => unreachable!("Unknown allocator method!"),
            };

            let mut attrs = CodegenFnAttrs::new();
            attrs.flags |= alloc_attr_flag;
            create_wrapper_function(
                tcx,
                &cx,
                &from_name,
                Some(&to_name),
                &args,
                output,
                false,
                &attrs,
            );
        }
    }

    // rust alloc error handler
    create_wrapper_function(
        tcx,
        &cx,
        &mangle_internal_symbol(tcx, "__rust_alloc_error_handler"),
        Some(&mangle_internal_symbol(tcx, alloc_error_handler_name(alloc_error_handler_kind))),
        &[usize, usize], // size, align
        None,
        true,
        &CodegenFnAttrs::new(),
    );

    unsafe {
        // __rust_alloc_error_handler_should_panic_v2
        create_const_value_function(
            tcx,
            &cx,
            &mangle_internal_symbol(tcx, OomStrategy::SYMBOL),
            &i8,
            &llvm::LLVMConstInt(i8, tcx.sess.opts.unstable_opts.oom.should_panic() as u64, FALSE),
        );

        // __rust_no_alloc_shim_is_unstable_v2
        create_wrapper_function(
            tcx,
            &cx,
            &mangle_internal_symbol(tcx, NO_ALLOC_SHIM_IS_UNSTABLE),
            None,
            &[],
            None,
            false,
            &CodegenFnAttrs::new(),
        );
    }

    if tcx.sess.opts.debuginfo != DebugInfo::None {
        let dbg_cx = debuginfo::CodegenUnitDebugContext::new(cx.llmod);
        debuginfo::metadata::build_compile_unit_di_node(tcx, module_name, &dbg_cx);
        dbg_cx.finalize(tcx.sess);
    }
}

fn create_const_value_function(
    tcx: TyCtxt<'_>,
    cx: &SimpleCx<'_>,
    name: &str,
    output: &Type,
    value: &Value,
) {
    let ty = cx.type_func(&[], output);
    let llfn = declare_simple_fn(
        &cx,
        name,
        llvm::CallConv::CCallConv,
        llvm::UnnamedAddr::Global,
        llvm::Visibility::from_generic(tcx.sess.default_visibility()),
        ty,
    );

    attributes::apply_to_llfn(
        llfn,
        llvm::AttributePlace::Function,
        &[llvm::AttributeKind::AlwaysInline.create_attr(cx.llcx)],
    );

    let llbb = unsafe { llvm::LLVMAppendBasicBlockInContext(cx.llcx, llfn, c"entry".as_ptr()) };
    let mut bx = SBuilder::build(&cx, llbb);
    bx.ret(value);
}

fn create_wrapper_function(
    tcx: TyCtxt<'_>,
    cx: &SimpleCx<'_>,
    from_name: &str,
    to_name: Option<&str>,
    args: &[&Type],
    output: Option<&Type>,
    no_return: bool,
    attrs: &CodegenFnAttrs,
) {
    let ty = cx.type_func(args, output.unwrap_or_else(|| cx.type_void()));
    let llfn = declare_simple_fn(
        &cx,
        from_name,
        llvm::CallConv::CCallConv,
        llvm::UnnamedAddr::Global,
        llvm::Visibility::from_generic(tcx.sess.default_visibility()),
        ty,
    );

    llfn_attrs_from_instance(cx, tcx, llfn, attrs, None);

    let no_return = if no_return {
        // -> ! DIFlagNoReturn
        let no_return = llvm::AttributeKind::NoReturn.create_attr(cx.llcx);
        attributes::apply_to_llfn(llfn, llvm::AttributePlace::Function, &[no_return]);
        Some(no_return)
    } else {
        None
    };

    let llbb = unsafe { llvm::LLVMAppendBasicBlockInContext(cx.llcx, llfn, c"entry".as_ptr()) };
    let mut bx = SBuilder::build(&cx, llbb);

    if let Some(to_name) = to_name {
        let callee = declare_simple_fn(
            &cx,
            to_name,
            llvm::CallConv::CCallConv,
            llvm::UnnamedAddr::Global,
            llvm::Visibility::Hidden,
            ty,
        );
        if let Some(no_return) = no_return {
            // -> ! DIFlagNoReturn
            attributes::apply_to_llfn(callee, llvm::AttributePlace::Function, &[no_return]);
        }
        llvm::set_visibility(callee, llvm::Visibility::Hidden);

        let args = args
            .iter()
            .enumerate()
            .map(|(i, _)| llvm::get_param(llfn, i as c_uint))
            .collect::<Vec<_>>();
        let ret = bx.call(ty, callee, &args, None);
        llvm::LLVMSetTailCall(ret, TRUE);
        if output.is_some() {
            bx.ret(ret);
        } else {
            bx.ret_void()
        }
    } else {
        assert!(output.is_none());
        bx.ret_void()
    }
}
