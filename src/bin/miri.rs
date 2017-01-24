#![feature(rustc_private, i128_type)]

extern crate getopts;
extern crate miri;
extern crate rustc;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate env_logger;
extern crate log_settings;
extern crate syntax;
#[macro_use] extern crate log;

use rustc::session::Session;
use rustc_driver::{Compilation, CompilerCalls, RustcDefaultCalls};
use rustc_driver::driver::{CompileState, CompileController};
use rustc::session::config::{self, Input, ErrorOutputType};
use rustc::hir::{self, itemlikevisit};
use rustc::ty::TyCtxt;
use syntax::ast::{MetaItemKind, NestedMetaItemKind, self};
use std::path::PathBuf;

struct MiriCompilerCalls(RustcDefaultCalls);

impl<'a> CompilerCalls<'a> for MiriCompilerCalls {
    fn early_callback(
        &mut self,
        matches: &getopts::Matches,
        sopts: &config::Options,
        cfg: &ast::CrateConfig,
        descriptions: &rustc_errors::registry::Registry,
        output: ErrorOutputType
    ) -> Compilation {
        self.0.early_callback(matches, sopts, cfg, descriptions, output)
    }
    fn no_input(
        &mut self,
        matches: &getopts::Matches,
        sopts: &config::Options,
        cfg: &ast::CrateConfig,
        odir: &Option<PathBuf>,
        ofile: &Option<PathBuf>,
        descriptions: &rustc_errors::registry::Registry
    ) -> Option<(Input, Option<PathBuf>)> {
        self.0.no_input(matches, sopts, cfg, odir, ofile, descriptions)
    }
    fn late_callback(
        &mut self,
        matches: &getopts::Matches,
        sess: &Session,
        input: &Input,
        odir: &Option<PathBuf>,
        ofile: &Option<PathBuf>
    ) -> Compilation {
        self.0.late_callback(matches, sess, input, odir, ofile)
    }
    fn build_controller(&mut self, sess: &Session, matches: &getopts::Matches) -> CompileController<'a> {
        let mut control = self.0.build_controller(sess, matches);
        control.after_hir_lowering.callback = Box::new(after_hir_lowering);
        control.after_analysis.callback = Box::new(after_analysis);
        if std::env::var("MIRI_HOST_TARGET") != Ok("yes".to_owned()) {
            // only fully compile targets on the host
            control.after_analysis.stop = Compilation::Stop;
        }
        control
    }
}

fn after_hir_lowering(state: &mut CompileState) {
    let attr = (String::from("miri"), syntax::feature_gate::AttributeType::Whitelisted);
    state.session.plugin_attributes.borrow_mut().push(attr);
}

fn after_analysis<'a, 'tcx>(state: &mut CompileState<'a, 'tcx>) {
    state.session.abort_if_errors();

    let tcx = state.tcx.unwrap();
    miri::run_mir_passes(tcx);
    let limits = resource_limits_from_attributes(state);

    if std::env::args().any(|arg| arg == "--test") {
        struct Visitor<'a, 'tcx: 'a>(miri::ResourceLimits, TyCtxt<'a, 'tcx, 'tcx>, &'a CompileState<'a, 'tcx>);
        impl<'a, 'tcx: 'a, 'hir> itemlikevisit::ItemLikeVisitor<'hir> for Visitor<'a, 'tcx> {
            fn visit_item(&mut self, i: &'hir hir::Item) {
                if let hir::Item_::ItemFn(_, _, _, _, _, body_id) = i.node {
                    if i.attrs.iter().any(|attr| attr.value.name == "test") {
                        let did = self.1.map.body_owner_def_id(body_id);
                        println!("running test: {}", self.1.map.def_path(did).to_string(self.1));
                        miri::eval_main(self.1, did, self.0);
                        self.2.session.abort_if_errors();
                    }
                }
            }
            fn visit_trait_item(&mut self, _trait_item: &'hir hir::TraitItem) {}
            fn visit_impl_item(&mut self, _impl_item: &'hir hir::ImplItem) {}
        }
        state.hir_crate.unwrap().visit_all_item_likes(&mut Visitor(limits, tcx, state));
    } else {
        if let Some((entry_node_id, _)) = *state.session.entry_fn.borrow() {
            let entry_def_id = tcx.map.local_def_id(entry_node_id);
            miri::eval_main(tcx, entry_def_id, limits);

            state.session.abort_if_errors();
        } else {
            println!("no main function found, assuming auxiliary build");
        }
    }
}

fn resource_limits_from_attributes(state: &CompileState) -> miri::ResourceLimits {
    let mut limits = miri::ResourceLimits::default();
    let krate = state.hir_crate.as_ref().unwrap();
    let err_msg = "miri attributes need to be in the form `miri(key = value)`";
    let extract_int = |lit: &syntax::ast::Lit| -> u128 {
        match lit.node {
            syntax::ast::LitKind::Int(i, _) => i,
            _ => state.session.span_fatal(lit.span, "expected an integer literal"),
        }
    };

    for attr in krate.attrs.iter().filter(|a| a.name() == "miri") {
        if let MetaItemKind::List(ref items) = attr.value.node {
            for item in items {
                if let NestedMetaItemKind::MetaItem(ref inner) = item.node {
                    if let MetaItemKind::NameValue(ref value) = inner.node {
                        match &inner.name().as_str()[..] {
                            "memory_size" => limits.memory_size = extract_int(value) as u64,
                            "step_limit" => limits.step_limit = extract_int(value) as u64,
                            "stack_limit" => limits.stack_limit = extract_int(value) as usize,
                            _ => state.session.span_err(item.span, "unknown miri attribute"),
                        }
                    } else {
                        state.session.span_err(inner.span, err_msg);
                    }
                } else {
                    state.session.span_err(item.span, err_msg);
                }
            }
        } else {
            state.session.span_err(attr.span, err_msg);
        }
    }
    limits
}

fn init_logger() {
    const MAX_INDENT: usize = 40;

    let format = |record: &log::LogRecord| {
        if record.level() == log::LogLevel::Trace {
            // prepend spaces to indent the final string
            let indentation = log_settings::settings().indentation;
            format!("{lvl}:{module}{depth:2}{indent:<indentation$} {text}",
                lvl = record.level(),
                module = record.location().module_path(),
                depth = indentation / MAX_INDENT,
                indentation = indentation % MAX_INDENT,
                indent = "",
                text = record.args())
        } else {
            format!("{lvl}:{module}: {text}",
                lvl = record.level(),
                module = record.location().module_path(),
                text = record.args())
        }
    };

    let mut builder = env_logger::LogBuilder::new();
    builder.format(format).filter(None, log::LogLevelFilter::Info);

    if std::env::var("MIRI_LOG").is_ok() {
        builder.parse(&std::env::var("MIRI_LOG").unwrap());
    }

    builder.init().unwrap();
}

fn find_sysroot() -> String {
    // Taken from https://github.com/Manishearth/rust-clippy/pull/911.
    let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
    let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
    match (home, toolchain) {
        (Some(home), Some(toolchain)) => format!("{}/toolchains/{}", home, toolchain),
        _ => option_env!("RUST_SYSROOT")
            .expect("need to specify RUST_SYSROOT env var or use rustup or multirust")
            .to_owned(),
    }
}

fn main() {
    init_logger();
    let mut args: Vec<String> = std::env::args().collect();

    let sysroot_flag = String::from("--sysroot");
    if !args.contains(&sysroot_flag) {
        args.push(sysroot_flag);
        args.push(find_sysroot());
    }
    // we run the optimization passes inside miri
    // if we ran them twice we'd get funny failures due to borrowck ElaborateDrops only working on
    // unoptimized MIR
    // FIXME: add an after-mir-passes hook to rustc driver
    args.push("-Zmir-opt-level=0".to_owned());
    // for auxilary builds in unit tests
    args.push("-Zalways-encode-mir".to_owned());

    rustc_driver::run_compiler(&args, &mut MiriCompilerCalls(RustcDefaultCalls), None, None);
}
