#![feature(rustc_private, i128_type)]

extern crate getopts;
extern crate miri;
extern crate rustc;
extern crate rustc_driver;
extern crate env_logger;
extern crate log_settings;
extern crate syntax;
#[macro_use] extern crate log;

use rustc::session::Session;
use rustc_driver::{Compilation, CompilerCalls};
use rustc_driver::driver::{CompileState, CompileController};
use syntax::ast::{MetaItemKind, NestedMetaItemKind};

struct MiriCompilerCalls;

impl<'a> CompilerCalls<'a> for MiriCompilerCalls {
    fn build_controller(&mut self, _: &Session, _: &getopts::Matches) -> CompileController<'a> {
        let mut control = CompileController::basic();
        control.after_hir_lowering.callback = Box::new(after_hir_lowering);
        control.after_analysis.callback = Box::new(after_analysis);
        control.after_analysis.stop = Compilation::Stop;
        control
    }
}

fn after_hir_lowering(state: &mut CompileState) {
    let attr = (String::from("miri"), syntax::feature_gate::AttributeType::Whitelisted);
    state.session.plugin_attributes.borrow_mut().push(attr);
}

fn after_analysis(state: &mut CompileState) {
    state.session.abort_if_errors();

    let tcx = state.tcx.unwrap();
    if let Some((entry_node_id, _)) = *state.session.entry_fn.borrow() {
        let entry_def_id = tcx.map.local_def_id(entry_node_id);
        let limits = resource_limits_from_attributes(state);
        miri::run_mir_passes(tcx);
        miri::eval_main(tcx, entry_def_id, limits);

        state.session.abort_if_errors();
    } else {
        println!("no main function found, assuming auxiliary build");
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
    args.push("-Zalways-encode-mir".to_owned());

    rustc_driver::run_compiler(&args, &mut MiriCompilerCalls, None, None);
}
