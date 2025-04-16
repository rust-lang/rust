//@ edition: 2021
//@ run-pass
//@ run-flags: {{sysroot-base}} {{target-linker}}
//@ ignore-stage1 (requires matching sysroot built with in-tree compiler)

// Regression test for <https://github.com/rust-lang/rust/issues/19371>.
//
// This test ensures that `compile_input` can be called twice in one task
// without causing a panic.

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_session;
extern crate rustc_span;

use std::path::{Path, PathBuf};

use rustc_interface::{Linker, interface};
use rustc_session::config::{Input, Options, OutFileName, OutputType, OutputTypes};
use rustc_span::FileName;

fn main() {
    let src = r#"
    fn main() {}
    "#;

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        panic!("expected sysroot (and optional linker)");
    }

    let sysroot = PathBuf::from(&args[1]);
    let linker = args.get(2).map(PathBuf::from);

    // compiletest sets the current dir to `output_base_dir` when running.
    let tmpdir = std::env::current_dir().unwrap().join("tmp");
    std::fs::create_dir_all(&tmpdir).unwrap();

    compile(src.to_string(), tmpdir.join("out"), sysroot.clone(), linker.as_deref());
    compile(src.to_string(), tmpdir.join("out"), sysroot.clone(), linker.as_deref());
}

fn compile(code: String, output: PathBuf, sysroot: PathBuf, linker: Option<&Path>) {
    let mut opts = Options::default();
    opts.output_types = OutputTypes::new(&[(OutputType::Exe, None)]);
    opts.sysroot = sysroot;

    if let Some(linker) = linker {
        opts.cg.linker = Some(linker.to_owned());
    }

    let name = FileName::anon_source_code(&code);
    let input = Input::Str { name, input: code };

    let config = interface::Config {
        opts,
        crate_cfg: Default::default(),
        crate_check_cfg: Default::default(),
        input,
        output_file: Some(OutFileName::Real(output)),
        output_dir: None,
        ice_file: None,
        file_loader: None,
        locale_resources: Vec::new(),
        lint_caps: Default::default(),
        psess_created: None,
        hash_untracked_state: None,
        register_lints: None,
        override_queries: None,
        extra_symbols: Vec::new(),
        make_codegen_backend: None,
        registry: rustc_driver::diagnostics_registry(),
        using_internal_features: &rustc_driver::USING_INTERNAL_FEATURES,
        expanded_args: Default::default(),
    };

    interface::run_compiler(config, |compiler| {
        let krate = rustc_interface::passes::parse(&compiler.sess);
        let linker = rustc_interface::create_and_enter_global_ctxt(&compiler, krate, |tcx| {
            let _ = tcx.analysis(());
            Linker::codegen_and_build_linker(tcx, &*compiler.codegen_backend)
        });
        linker.link(&compiler.sess, &*compiler.codegen_backend);
    });
}
