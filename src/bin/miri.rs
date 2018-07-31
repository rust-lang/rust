#![feature(rustc_private)]

extern crate getopts;
extern crate miri;
extern crate rustc;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_codegen_utils;
extern crate env_logger;
extern crate log_settings;
extern crate syntax;
extern crate log;

use rustc::session::Session;
use rustc::middle::cstore::CrateStore;
use rustc_driver::{Compilation, CompilerCalls, RustcDefaultCalls};
use rustc_driver::driver::{CompileState, CompileController};
use rustc::session::config::{self, Input, ErrorOutputType};
use rustc::hir::{self, itemlikevisit};
use rustc::ty::TyCtxt;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use syntax::ast;
use std::path::PathBuf;

struct MiriCompilerCalls {
    default: Box<RustcDefaultCalls>,
    /// Whether to begin interpretation at the start_fn lang item or not
    ///
    /// If false, the interpretation begins at the `main` function
    start_fn: bool,
}

impl<'a> CompilerCalls<'a> for MiriCompilerCalls {
    fn early_callback(
        &mut self,
        matches: &getopts::Matches,
        sopts: &config::Options,
        cfg: &ast::CrateConfig,
        descriptions: &rustc_errors::registry::Registry,
        output: ErrorOutputType,
    ) -> Compilation {
        self.default.early_callback(
            matches,
            sopts,
            cfg,
            descriptions,
            output,
        )
    }
    fn no_input(
        &mut self,
        matches: &getopts::Matches,
        sopts: &config::Options,
        cfg: &ast::CrateConfig,
        odir: &Option<PathBuf>,
        ofile: &Option<PathBuf>,
        descriptions: &rustc_errors::registry::Registry,
    ) -> Option<(Input, Option<PathBuf>)> {
        self.default.no_input(
            matches,
            sopts,
            cfg,
            odir,
            ofile,
            descriptions,
        )
    }
    fn late_callback(
        &mut self,
        codegen_backend: &CodegenBackend,
        matches: &getopts::Matches,
        sess: &Session,
        cstore: &CrateStore,
        input: &Input,
        odir: &Option<PathBuf>,
        ofile: &Option<PathBuf>,
    ) -> Compilation {
        self.default.late_callback(codegen_backend, matches, sess, cstore, input, odir, ofile)
    }
    fn build_controller(
        self: Box<Self>,
        sess: &Session,
        matches: &getopts::Matches,
    ) -> CompileController<'a> {
        let this = *self;
        let mut control = this.default.build_controller(sess, matches);
        control.after_hir_lowering.callback = Box::new(after_hir_lowering);
        let start_fn = this.start_fn;
        control.after_analysis.callback = Box::new(move |state| after_analysis(state, start_fn));
        control.after_analysis.stop = Compilation::Stop;
        control
    }
}

fn after_hir_lowering(state: &mut CompileState) {
    let attr = (
        String::from("miri"),
        syntax::feature_gate::AttributeType::Whitelisted,
    );
    state.session.plugin_attributes.borrow_mut().push(attr);
}

fn after_analysis<'a, 'tcx>(state: &mut CompileState<'a, 'tcx>, use_start_fn: bool) {
    state.session.abort_if_errors();

    let tcx = state.tcx.unwrap();

    if std::env::args().any(|arg| arg == "--test") {
        struct Visitor<'a, 'tcx: 'a>(
            TyCtxt<'a, 'tcx, 'tcx>,
            &'a CompileState<'a, 'tcx>
        );
        impl<'a, 'tcx: 'a, 'hir> itemlikevisit::ItemLikeVisitor<'hir> for Visitor<'a, 'tcx> {
            fn visit_item(&mut self, i: &'hir hir::Item) {
                if let hir::ItemKind::Fn(.., body_id) = i.node {
                    if i.attrs.iter().any(|attr| {
                        attr.name() == "test"
                    })
                    {
                        let did = self.0.hir.body_owner_def_id(body_id);
                        println!(
                            "running test: {}",
                            self.0.def_path_debug_str(did),
                        );
                        miri::eval_main(self.0, did, None);
                        self.1.session.abort_if_errors();
                    }
                }
            }
            fn visit_trait_item(&mut self, _trait_item: &'hir hir::TraitItem) {}
            fn visit_impl_item(&mut self, _impl_item: &'hir hir::ImplItem) {}
        }
        state.hir_crate.unwrap().visit_all_item_likes(
            &mut Visitor(tcx, state),
        );
    } else if let Some((entry_node_id, _, _)) = *state.session.entry_fn.borrow() {
        let entry_def_id = tcx.hir.local_def_id(entry_node_id);
        // Use start_fn lang item if we have -Zmiri-start-fn set
        let start_wrapper = if use_start_fn {
            Some(tcx.lang_items().start_fn().unwrap())
        } else {
            None
        };
        miri::eval_main(tcx, entry_def_id, start_wrapper);

        state.session.abort_if_errors();
    } else {
        println!("no main function found, assuming auxiliary build");
    }
}

fn init_logger() {
    let format = |formatter: &mut env_logger::fmt::Formatter, record: &log::Record| {
        use std::io::Write;
        if record.level() == log::Level::Trace {
            // prepend frame number
            let indentation = log_settings::settings().indentation;
            writeln!(
                formatter,
                "{indentation}:{lvl}:{module}: {text}",
                lvl = record.level(),
                module = record.module_path().unwrap_or("<unknown module>"),
                indentation = indentation,
                text = record.args(),
            )
        } else {
            writeln!(
                formatter,
                "{lvl}:{module}: {text}",
                lvl = record.level(),
                module = record.module_path().unwrap_or("<unknown_module>"),
                text = record.args(),
            )
        }
    };

    let mut builder = env_logger::Builder::new();
    builder.format(format).filter(
        None,
        log::LevelFilter::Info,
    );

    if std::env::var("MIRI_LOG").is_ok() {
        builder.parse(&std::env::var("MIRI_LOG").unwrap());
    }

    builder.init();
}

fn find_sysroot() -> String {
    if let Ok(sysroot) = std::env::var("MIRI_SYSROOT") {
        return sysroot;
    }

    // Taken from https://github.com/Manishearth/rust-clippy/pull/911.
    let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
    let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
    match (home, toolchain) {
        (Some(home), Some(toolchain)) => format!("{}/toolchains/{}", home, toolchain),
        _ => {
            option_env!("RUST_SYSROOT")
                .expect(
                    "Could not find sysroot. Either set MIRI_SYSROOT at run-time, or at \
                     build-time specify RUST_SYSROOT env var or use rustup or multirust",
                )
                .to_owned()
        }
    }
}

fn main() {
    rustc_driver::init_rustc_env_logger();
    init_logger();
    let mut args: Vec<String> = std::env::args().collect();

    let sysroot_flag = String::from("--sysroot");
    if !args.contains(&sysroot_flag) {
        args.push(sysroot_flag);
        args.push(find_sysroot());
    }

    let mut start_fn = false;
    args.retain(|arg| {
        if arg == "-Zmiri-start-fn" {
            start_fn = true;
            false
        } else {
            true
        }
    });


    let result = rustc_driver::run(move || {
        rustc_driver::run_compiler(&args, Box::new(MiriCompilerCalls {
            default: Box::new(RustcDefaultCalls),
            start_fn,
        }), None, None)
    });
    std::process::exit(result as i32);
}
