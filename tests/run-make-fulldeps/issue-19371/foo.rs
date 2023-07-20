#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_session;
extern crate rustc_span;

use rustc_interface::interface;
use rustc_session::config::{Input, Options, OutFileName, OutputType, OutputTypes};
use rustc_span::source_map::FileName;

use std::path::PathBuf;

fn main() {
    let src = r#"
    fn main() {}
    "#;

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 {
        panic!("expected rustc path");
    }

    let tmpdir = PathBuf::from(&args[1]);

    let mut sysroot = PathBuf::from(&args[3]);
    sysroot.pop();
    sysroot.pop();

    compile(src.to_string(), tmpdir.join("out"), sysroot.clone());

    compile(src.to_string(), tmpdir.join("out"), sysroot.clone());
}

fn compile(code: String, output: PathBuf, sysroot: PathBuf) {
    let mut opts = Options::default();
    opts.output_types = OutputTypes::new(&[(OutputType::Exe, None)]);
    opts.maybe_sysroot = Some(sysroot);

    if let Ok(linker) = std::env::var("RUSTC_LINKER") {
        opts.cg.linker = Some(linker.into());
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
        locale_resources: &[],
        lint_caps: Default::default(),
        parse_sess_created: None,
        register_lints: None,
        override_queries: None,
        make_codegen_backend: None,
        registry: rustc_driver::diagnostics_registry(),
    };

    interface::run_compiler(config, |compiler| {
        let linker = compiler.enter(|queries| {
            queries.global_ctxt()?.enter(|tcx| tcx.analysis(()))?;
            let ongoing_codegen = queries.ongoing_codegen()?;
            queries.linker(ongoing_codegen)
        });
        linker.unwrap().link().unwrap();
    });
}
