use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use rustc_hir::LangItem;
use rustc_middle::ty::{AssocTag, GenericArg};
use rustc_session::config::EntryFnType;
use rustc_span::{DUMMY_SP, Ident};

use crate::prelude::*;

/// Create the `main` function which will initialize the rust runtime and call
/// users main function.
pub(crate) fn maybe_create_entry_wrapper(
    tcx: TyCtxt<'_>,
    module: &mut dyn Module,
    is_jit: bool,
    is_primary_cgu: bool,
) {
    let (main_def_id, sigpipe) = match tcx.entry_fn(()) {
        Some((def_id, entry_ty)) => (
            def_id,
            match entry_ty {
                EntryFnType::Main { sigpipe } => sigpipe,
            },
        ),
        None => return,
    };

    if main_def_id.is_local() {
        let instance = Instance::mono(tcx, main_def_id);
        if module.get_name(tcx.symbol_name(instance).name).is_none() {
            return;
        }
    } else if !is_primary_cgu {
        return;
    }

    create_entry_fn(tcx, module, main_def_id, is_jit, sigpipe);

    fn create_entry_fn(
        tcx: TyCtxt<'_>,
        m: &mut dyn Module,
        rust_main_def_id: DefId,
        ignore_lang_start_wrapper: bool,
        sigpipe: u8,
    ) {
        let main_ret_ty = tcx.fn_sig(rust_main_def_id).no_bound_vars().unwrap().output();
        // Given that `main()` has no arguments,
        // then its return type cannot have
        // late-bound regions, since late-bound
        // regions must appear in the argument
        // listing.
        let main_ret_ty = tcx.normalize_erasing_regions(
            ty::TypingEnv::fully_monomorphized(),
            main_ret_ty.no_bound_vars().unwrap(),
        );

        let cmain_sig = Signature {
            params: vec![
                AbiParam::new(m.target_config().pointer_type()),
                AbiParam::new(m.target_config().pointer_type()),
            ],
            returns: vec![AbiParam::new(m.target_config().pointer_type() /*isize*/)],
            call_conv: crate::conv_to_call_conv(
                tcx.sess,
                tcx.sess.target.options.entry_abi,
                m.target_config().default_call_conv,
            ),
        };

        let entry_name = tcx.sess.target.options.entry_name.as_ref();
        let cmain_func_id = match m.declare_function(entry_name, Linkage::Export, &cmain_sig) {
            Ok(func_id) => func_id,
            Err(err) => {
                tcx.dcx()
                    .fatal(format!("entry symbol `{entry_name}` declared multiple times: {err}"));
            }
        };

        let instance = Instance::mono(tcx, rust_main_def_id);

        let main_name = tcx.symbol_name(instance).name;
        let main_sig = get_function_sig(tcx, m.target_config().default_call_conv, instance);
        let main_func_id = m.declare_function(main_name, Linkage::Import, &main_sig).unwrap();

        let mut ctx = Context::new();
        ctx.func.signature = cmain_sig;
        {
            let mut func_ctx = FunctionBuilderContext::new();
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

            let block = bcx.create_block();
            bcx.switch_to_block(block);
            let arg_argc = bcx.append_block_param(block, m.target_config().pointer_type());
            let arg_argv = bcx.append_block_param(block, m.target_config().pointer_type());
            let arg_sigpipe = bcx.ins().iconst(types::I8, sigpipe as i64);

            let main_func_ref = m.declare_func_in_func(main_func_id, &mut bcx.func);

            let result = if ignore_lang_start_wrapper {
                // ignoring #[lang = "start"] as we are running in the jit
                // FIXME set program arguments somehow
                let call_inst = bcx.ins().call(main_func_ref, &[]);
                let call_results = bcx.func.dfg.inst_results(call_inst).to_owned();

                let termination_trait = tcx.require_lang_item(LangItem::Termination, None);
                let report = tcx
                    .associated_items(termination_trait)
                    .find_by_ident_and_kind(
                        tcx,
                        Ident::from_str("report"),
                        AssocTag::Fn,
                        termination_trait,
                    )
                    .unwrap();
                let report = Instance::expect_resolve(
                    tcx,
                    ty::TypingEnv::fully_monomorphized(),
                    report.def_id,
                    tcx.mk_args(&[GenericArg::from(main_ret_ty)]),
                    DUMMY_SP,
                );

                let report_name = tcx.symbol_name(report).name;
                let report_sig = get_function_sig(tcx, m.target_config().default_call_conv, report);
                let report_func_id =
                    m.declare_function(report_name, Linkage::Import, &report_sig).unwrap();
                let report_func_ref = m.declare_func_in_func(report_func_id, &mut bcx.func);

                // FIXME do proper abi handling instead of expecting the pass mode to be identical
                // for returns and arguments.
                let report_call_inst = bcx.ins().call(report_func_ref, &call_results);
                let res = bcx.func.dfg.inst_results(report_call_inst)[0];
                match m.target_config().pointer_type() {
                    types::I32 => res,
                    types::I64 => bcx.ins().sextend(types::I64, res),
                    _ => unimplemented!("16bit systems are not yet supported"),
                }
            } else {
                // Regular main fn invoked via start lang item.
                let start_def_id = tcx.require_lang_item(LangItem::Start, None);
                let start_instance = Instance::expect_resolve(
                    tcx,
                    ty::TypingEnv::fully_monomorphized(),
                    start_def_id,
                    tcx.mk_args(&[main_ret_ty.into()]),
                    DUMMY_SP,
                );
                let start_func_id = import_function(tcx, m, start_instance);

                let main_val = bcx.ins().func_addr(m.target_config().pointer_type(), main_func_ref);

                let func_ref = m.declare_func_in_func(start_func_id, &mut bcx.func);
                let call_inst =
                    bcx.ins().call(func_ref, &[main_val, arg_argc, arg_argv, arg_sigpipe]);
                bcx.inst_results(call_inst)[0]
            };

            bcx.ins().return_(&[result]);
            bcx.seal_all_blocks();
            bcx.finalize();
        }

        if let Err(err) = m.define_function(cmain_func_id, &mut ctx) {
            tcx.dcx().fatal(format!("entry symbol `{entry_name}` defined multiple times: {err}"));
        }
    }
}
