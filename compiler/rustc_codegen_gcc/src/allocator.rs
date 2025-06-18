use gccjit::GlobalKind;
#[cfg(feature = "master")]
use gccjit::VarAttribute;
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

    let name = mangle_internal_symbol(tcx, NO_ALLOC_SHIM_IS_UNSTABLE);
    let global = context.new_global(None, GlobalKind::Exported, i8, name);
    #[cfg(feature = "master")]
    global.add_attribute(VarAttribute::Visibility(symbol_visibility_to_gcc(
        tcx.sess.default_visibility(),
    )));
    let value = context.new_rvalue_from_int(i8, 0);
    global.global_set_initializer_rvalue(value);
}
