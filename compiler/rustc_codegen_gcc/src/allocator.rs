use gccjit::{Context, FunctionType, GlobalKind, ToRValue, Type};
#[cfg(feature = "master")]
use gccjit::{FnAttribute, VarAttribute};
use rustc_ast::expand::allocator::NO_ALLOC_SHIM_IS_UNSTABLE;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::OomStrategy;
use rustc_symbol_mangling::mangle_internal_symbol;

use crate::GccContext;
#[cfg(feature = "master")]
use crate::base::symbol_visibility_to_gcc;

pub(crate) unsafe fn codegen(tcx: TyCtxt<'_>, mods: &mut GccContext, _module_name: &str) {
    let context = &mods.context;
    let i8 = context.new_type::<i8>();

    let name = mangle_internal_symbol(tcx, OomStrategy::SYMBOL);
    let global = context.new_global(None, GlobalKind::Exported, i8, name);
    #[cfg(feature = "master")]
    global.add_attribute(VarAttribute::Visibility(symbol_visibility_to_gcc(
        tcx.sess.default_visibility(),
    )));
    let value = tcx.sess.opts.unstable_opts.oom.should_panic();
    let value = context.new_rvalue_from_int(i8, value as i32);
    global.global_set_initializer_rvalue(value);

    create_wrapper_function(
        tcx,
        context,
        &mangle_internal_symbol(tcx, NO_ALLOC_SHIM_IS_UNSTABLE),
        None,
        &[],
        None,
    );
}

fn create_wrapper_function(
    tcx: TyCtxt<'_>,
    context: &Context<'_>,
    from_name: &str,
    to_name: Option<&str>,
    types: &[Type<'_>],
    output: Option<Type<'_>>,
) {
    let void = context.new_type::<()>();

    let args: Vec<_> = types
        .iter()
        .enumerate()
        .map(|(index, typ)| context.new_parameter(None, *typ, format!("param{}", index)))
        .collect();
    let func = context.new_function(
        None,
        FunctionType::Exported,
        output.unwrap_or(void),
        &args,
        from_name,
        false,
    );

    #[cfg(feature = "master")]
    func.add_attribute(FnAttribute::Visibility(symbol_visibility_to_gcc(
        tcx.sess.default_visibility(),
    )));

    if tcx.sess.must_emit_unwind_tables() {
        // TODO(antoyo): emit unwind tables.
    }

    let block = func.new_block("entry");

    if let Some(to_name) = to_name {
        let args: Vec<_> = types
            .iter()
            .enumerate()
            .map(|(index, typ)| context.new_parameter(None, *typ, format!("param{}", index)))
            .collect();
        let callee = context.new_function(
            None,
            FunctionType::Extern,
            output.unwrap_or(void),
            &args,
            to_name,
            false,
        );
        #[cfg(feature = "master")]
        callee.add_attribute(FnAttribute::Visibility(gccjit::Visibility::Hidden));

        let args = args
            .iter()
            .enumerate()
            .map(|(i, _)| func.get_param(i as i32).to_rvalue())
            .collect::<Vec<_>>();
        let ret = context.new_call(None, callee, &args);
        //llvm::LLVMSetTailCall(ret, True);
        if output.is_some() {
            block.end_with_return(None, ret);
        } else {
            block.add_eval(None, ret);
            block.end_with_void_return(None);
        }
    } else {
        assert!(output.is_none());
        block.end_with_void_return(None);
    }

    // TODO(@Commeownist): Check if we need to emit some extra debugging info in certain circumstances
    // as described in https://github.com/rust-lang/rust/commit/77a96ed5646f7c3ee8897693decc4626fe380643
}
