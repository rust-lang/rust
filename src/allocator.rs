#[cfg(feature="master")]
use gccjit::FnAttribute;
use gccjit::{Context, FunctionType, GlobalKind, ToRValue, Type};
use rustc_ast::expand::allocator::{
    alloc_error_handler_name, default_fn_name, global_fn_name, AllocatorKind, AllocatorTy,
    ALLOCATOR_METHODS, NO_ALLOC_SHIM_IS_UNSTABLE,
};
use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::OomStrategy;

use crate::GccContext;

pub(crate) unsafe fn codegen(tcx: TyCtxt<'_>, mods: &mut GccContext, _module_name: &str, kind: AllocatorKind, alloc_error_handler_kind: AllocatorKind) {
    let context = &mods.context;
    let usize =
        match tcx.sess.target.pointer_width {
            16 => context.new_type::<u16>(),
            32 => context.new_type::<u32>(),
            64 => context.new_type::<u64>(),
            tws => bug!("Unsupported target word size for int: {}", tws),
        };
    let i8 = context.new_type::<i8>();
    let i8p = i8.make_pointer();

    if kind == AllocatorKind::Default {
        for method in ALLOCATOR_METHODS {
            let mut types = Vec::with_capacity(method.inputs.len());
            for input in method.inputs.iter() {
                match input.ty {
                    AllocatorTy::Layout => {
                        types.push(usize);
                        types.push(usize);
                    }
                    AllocatorTy::Ptr => types.push(i8p),
                    AllocatorTy::Usize => types.push(usize),

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
            let from_name = global_fn_name(method.name);
            let to_name = default_fn_name(method.name);

            create_wrapper_function(tcx, context, &from_name, &to_name, &types, output);
        }
    }

    // FIXME(bjorn3): Add noreturn attribute
    create_wrapper_function(
        tcx,
        context,
        "__rust_alloc_error_handler",
        &alloc_error_handler_name(alloc_error_handler_kind),
        &[usize, usize],
        None,
    );

    let name = OomStrategy::SYMBOL.to_string();
    let global = context.new_global(None, GlobalKind::Exported, i8, name);
    let value = tcx.sess.opts.unstable_opts.oom.should_panic();
    let value = context.new_rvalue_from_int(i8, value as i32);
    global.global_set_initializer_rvalue(value);

    let name = NO_ALLOC_SHIM_IS_UNSTABLE.to_string();
    let global = context.new_global(None, GlobalKind::Exported, i8, name);
    let value = context.new_rvalue_from_int(i8, 0);
    global.global_set_initializer_rvalue(value);
}

fn create_wrapper_function(
    tcx: TyCtxt<'_>,
    context: &Context<'_>,
    from_name: &str,
    to_name: &str,
    types: &[Type<'_>],
    output: Option<Type<'_>>,
) {
    let void = context.new_type::<()>();

    let args: Vec<_> = types.iter().enumerate()
        .map(|(index, typ)| context.new_parameter(None, *typ, &format!("param{}", index)))
        .collect();
    let func = context.new_function(None, FunctionType::Exported, output.unwrap_or(void), &args, from_name, false);

    if tcx.sess.target.options.default_hidden_visibility {
        #[cfg(feature="master")]
        func.add_attribute(FnAttribute::Visibility(gccjit::Visibility::Hidden));
    }
    if tcx.sess.must_emit_unwind_tables() {
        // TODO(antoyo): emit unwind tables.
    }

    let args: Vec<_> = types.iter().enumerate()
        .map(|(index, typ)| context.new_parameter(None, *typ, &format!("param{}", index)))
        .collect();
    let callee = context.new_function(None, FunctionType::Extern, output.unwrap_or(void), &args, to_name, false);
    #[cfg(feature="master")]
    callee.add_attribute(FnAttribute::Visibility(gccjit::Visibility::Hidden));

    let block = func.new_block("entry");

    let args = args
        .iter()
        .enumerate()
        .map(|(i, _)| func.get_param(i as i32).to_rvalue())
        .collect::<Vec<_>>();
    let ret = context.new_call(None, callee, &args);
    //llvm::LLVMSetTailCall(ret, True);
    if output.is_some() {
        block.end_with_return(None, ret);
    }
    else {
        block.end_with_void_return(None);
    }

    // TODO(@Commeownist): Check if we need to emit some extra debugging info in certain circumstances
    // as described in https://github.com/rust-lang/rust/commit/77a96ed5646f7c3ee8897693decc4626fe380643
}
