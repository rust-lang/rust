//@ run-pass
//@ run-flags: --sysroot {{sysroot-base}} --edition=2021
//@ ignore-stage1 (requires matching sysroot built with in-tree compiler)
//@ ignore-cross-compile
//@ ignore-remote
#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_mir_transform;
extern crate rustc_span;

use std::process::ExitCode;

use rustc_driver::Compilation;
use rustc_hir::LangItem;
use rustc_hir::def::DefKind;
use rustc_interface::interface::Compiler;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::DUMMY_SP;

fn main() -> ExitCode {
    rustc_driver::catch_with_exit_code(move || {
        std::fs::write(
            "drop_glue_polymorphic_const_generic_input.rs",
            r#"
                struct Unit;

                struct ContainsArray<const K: usize> {
                    array: [Unit; K],
                    make_non_trivial_drop: Box<u32>,
                }

                fn foo(_: ContainsArray<42>) {}
            "#,
        )
        .unwrap();

        let mut args: Vec<_> = std::env::args().collect();
        args.push("--crate-type=lib".to_owned());
        args.push("drop_glue_polymorphic_const_generic_input.rs".to_owned());

        rustc_driver::run_compiler(&args, &mut CompilerCalls);
    })
}

struct CompilerCalls;

impl rustc_driver::Callbacks for CompilerCalls {
    fn after_analysis<'tcx>(&mut self, _compiler: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        tcx.sess.dcx().abort_if_errors();

        let drop_glue = tcx.require_lang_item(LangItem::DropGlue, DUMMY_SP);
        let contains_array = tcx
            .hir_crate_items(())
            .free_items()
            .map(|id| id.owner_id.to_def_id())
            .find(|&def_id| {
                tcx.def_kind(def_id) == DefKind::Struct
                    && tcx.def_path_str(def_id).ends_with("ContainsArray")
            })
            .unwrap();

        // Regression test for ICE due to the drop glue code not correctly handling generic
        // contexts.
        let ty = tcx.type_of(contains_array).instantiate_identity().skip_norm_wip();
        let typing_env = ty::TypingEnv::post_analysis(tcx, contains_array);
        rustc_mir_transform::build_drop_shim(tcx, drop_glue, Some(ty), typing_env);

        Compilation::Stop
    }
}
