//! Atomic intrinsics are implemented using a global lock for now, as Cranelift doesn't support
//! atomic operations yet.

// FIXME implement atomic instructions in Cranelift.

use crate::prelude::*;

#[no_mangle]
pub static mut __cg_clif_global_atomic_mutex: libc::pthread_mutex_t = libc::PTHREAD_MUTEX_INITIALIZER;

pub fn init_global_lock(module: &mut Module<impl Backend>, bcx: &mut FunctionBuilder<'_>) {
    if std::env::var("CG_CLIF_JIT").is_ok () {
        // When using JIT, dylibs won't find the __cg_clif_global_atomic_mutex data object defined here,
        // so instead define it in the cg_clif dylib.

        return;
    }

    let mut data_ctx = DataContext::new();
    data_ctx.define_zeroinit(1024); // 1024 bytes should be big enough on all platforms.
    let atomic_mutex = module.declare_data(
        "__cg_clif_global_atomic_mutex",
        Linkage::Export,
        true,
        false,
        Some(16),
    ).unwrap();
    module.define_data(atomic_mutex, &data_ctx).unwrap();

    let pthread_mutex_init = module.declare_function("pthread_mutex_init", Linkage::Import, &cranelift_codegen::ir::Signature {
        call_conv: module.target_config().default_call_conv,
        params: vec![
            AbiParam::new(module.target_config().pointer_type() /* *mut pthread_mutex_t */),
            AbiParam::new(module.target_config().pointer_type() /* *const pthread_mutex_attr_t */),
        ],
        returns: vec![AbiParam::new(types::I32 /* c_int */)],
    }).unwrap();

    let pthread_mutex_init = module.declare_func_in_func(pthread_mutex_init, bcx.func);

    let atomic_mutex = module.declare_data_in_func(atomic_mutex, bcx.func);
    let atomic_mutex = bcx.ins().global_value(module.target_config().pointer_type(), atomic_mutex);

    let nullptr = bcx.ins().iconst(module.target_config().pointer_type(), 0);

    bcx.ins().call(pthread_mutex_init, &[atomic_mutex, nullptr]);
}

pub fn lock_global_lock(fx: &mut FunctionCx<'_, '_, impl Backend>) {
    let atomic_mutex = fx.module.declare_data(
        "__cg_clif_global_atomic_mutex",
        Linkage::Import,
        true,
        false,
        None,
    ).unwrap();

    let pthread_mutex_lock = fx.module.declare_function("pthread_mutex_lock", Linkage::Import, &cranelift_codegen::ir::Signature {
        call_conv: fx.module.target_config().default_call_conv,
        params: vec![
            AbiParam::new(fx.module.target_config().pointer_type() /* *mut pthread_mutex_t */),
        ],
        returns: vec![AbiParam::new(types::I32 /* c_int */)],
    }).unwrap();

    let pthread_mutex_lock = fx.module.declare_func_in_func(pthread_mutex_lock, fx.bcx.func);

    let atomic_mutex = fx.module.declare_data_in_func(atomic_mutex, fx.bcx.func);
    let atomic_mutex = fx.bcx.ins().global_value(fx.module.target_config().pointer_type(), atomic_mutex);

    fx.bcx.ins().call(pthread_mutex_lock, &[atomic_mutex]);
}

pub fn unlock_global_lock(fx: &mut FunctionCx<'_, '_, impl Backend>) {
    let atomic_mutex = fx.module.declare_data(
        "__cg_clif_global_atomic_mutex",
        Linkage::Import,
        true,
        false,
        None,
    ).unwrap();

    let pthread_mutex_unlock = fx.module.declare_function("pthread_mutex_unlock", Linkage::Import, &cranelift_codegen::ir::Signature {
        call_conv: fx.module.target_config().default_call_conv,
        params: vec![
            AbiParam::new(fx.module.target_config().pointer_type() /* *mut pthread_mutex_t */),
        ],
        returns: vec![AbiParam::new(types::I32 /* c_int */)],
    }).unwrap();

    let pthread_mutex_unlock = fx.module.declare_func_in_func(pthread_mutex_unlock, fx.bcx.func);

    let atomic_mutex = fx.module.declare_data_in_func(atomic_mutex, fx.bcx.func);
    let atomic_mutex = fx.bcx.ins().global_value(fx.module.target_config().pointer_type(), atomic_mutex);

    fx.bcx.ins().call(pthread_mutex_unlock, &[atomic_mutex]);
}
