use crate::prelude::*;

/// Create the `main` function which will initialize the rust runtime and call
/// users main function.
pub fn maybe_create_entry_wrapper<'a, 'tcx: 'a>(
    tcx: TyCtxt<'tcx, 'tcx>,
    module: &mut Module<impl Backend + 'static>,
) {
    use rustc::middle::lang_items::StartFnLangItem;
    use rustc::session::config::EntryFnType;

    let (main_def_id, use_start_lang_item) = match tcx.entry_fn(LOCAL_CRATE) {
        Some((def_id, entry_ty)) => (
            def_id,
            match entry_ty {
                EntryFnType::Main => true,
                EntryFnType::Start => false,
            },
        ),
        None => return,
    };

    create_entry_fn(tcx, module, main_def_id, use_start_lang_item);;

    fn create_entry_fn<'a, 'tcx: 'a>(
        tcx: TyCtxt<'tcx, 'tcx>,
        m: &mut Module<impl Backend + 'static>,
        rust_main_def_id: DefId,
        use_start_lang_item: bool,
    ) {
        let main_ret_ty = tcx.fn_sig(rust_main_def_id).output();
        // Given that `main()` has no arguments,
        // then its return type cannot have
        // late-bound regions, since late-bound
        // regions must appear in the argument
        // listing.
        let main_ret_ty = tcx.erase_regions(&main_ret_ty.no_bound_vars().unwrap());

        let cmain_sig = Signature {
            params: vec![
                AbiParam::new(m.target_config().pointer_type()),
                AbiParam::new(m.target_config().pointer_type()),
            ],
            returns: vec![AbiParam::new(
                m.target_config().pointer_type(), /*isize*/
            )],
            call_conv: CallConv::SystemV,
        };

        let cmain_func_id = m
            .declare_function("main", Linkage::Export, &cmain_sig)
            .unwrap();

        let instance = Instance::mono(tcx, rust_main_def_id);

        let (main_name, main_sig) = get_function_name_and_sig(tcx, instance, false);
        let main_func_id = m
            .declare_function(&main_name, Linkage::Import, &main_sig)
            .unwrap();

        let mut ctx = Context::new();
        ctx.func = Function::with_name_signature(ExternalName::user(0, 0), cmain_sig.clone());
        {
            let mut func_ctx = FunctionBuilderContext::new();
            let mut bcx: FunctionBuilder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

            let ebb = bcx.create_ebb();
            bcx.switch_to_block(ebb);
            let arg_argc = bcx.append_ebb_param(ebb, m.target_config().pointer_type());
            let arg_argv = bcx.append_ebb_param(ebb, m.target_config().pointer_type());

            let main_func_ref = m.declare_func_in_func(main_func_id, &mut bcx.func);

            let call_inst = if use_start_lang_item {
                let start_def_id = tcx.require_lang_item(StartFnLangItem);
                let start_instance = Instance::resolve(
                    tcx,
                    ParamEnv::reveal_all(),
                    start_def_id,
                    tcx.intern_substs(&[main_ret_ty.into()]),
                )
                .unwrap();
                let start_func_id = import_function(tcx, m, start_instance);

                let main_val = bcx
                    .ins()
                    .func_addr(m.target_config().pointer_type(), main_func_ref);

                let func_ref = m.declare_func_in_func(start_func_id, &mut bcx.func);
                bcx.ins().call(func_ref, &[main_val, arg_argc, arg_argv])
            } else {
                // using user-defined start fn
                bcx.ins().call(main_func_ref, &[arg_argc, arg_argv])
            };

            let result = bcx.inst_results(call_inst)[0];
            bcx.ins().return_(&[result]);
            bcx.seal_all_blocks();
            bcx.finalize();
        }
        m.define_function(cmain_func_id, &mut ctx).unwrap();
    }
}
