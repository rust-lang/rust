#![feature(rustc_private)]

extern crate rustc;
extern crate rustc_driver;
extern crate rustc_lint;
extern crate rustc_metadata;
extern crate rustc_errors;
extern crate rustc_codegen_utils;
extern crate rustc_interface;
extern crate syntax;

use rustc::session::{build_session, Session};
use rustc::session::config::{Input, Options,
                             OutputType, OutputTypes};
use rustc_driver::driver::{self, compile_input, CompileController};
use rustc_metadata::cstore::CStore;
use rustc_errors::registry::Registry;
use rustc_interface::util;
use syntax::source_map::FileName;
use rustc_codegen_utils::codegen_backend::CodegenBackend;

use std::path::PathBuf;
use std::rc::Rc;

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

fn basic_sess(opts: Options) -> (Session, Rc<CStore>, Box<CodegenBackend>) {
    let descriptions = Registry::new(&rustc::DIAGNOSTICS);
    let sess = build_session(opts, None, descriptions);
    let codegen_backend = util::get_codegen_backend(&sess);
    let cstore = Rc::new(CStore::new(codegen_backend.metadata_loader()));
    rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));
    (sess, cstore, codegen_backend)
}

fn compile(code: String, output: PathBuf, sysroot: PathBuf) {
    syntax::with_globals(|| {
        let mut opts = Options::default();
        opts.output_types = OutputTypes::new(&[(OutputType::Exe, None)]);
        opts.maybe_sysroot = Some(sysroot);
        if let Ok(linker) = std::env::var("RUSTC_LINKER") {
            opts.cg.linker = Some(linker.into());
        }
        driver::spawn_thread_pool(opts, |opts| {
            let (sess, cstore, codegen_backend) = basic_sess(opts);
            let control = CompileController::basic();
            let name = FileName::anon_source_code(&code);
            let input = Input::Str { name, input: code };
            let _ = compile_input(
                codegen_backend,
                &sess,
                &cstore,
                &None,
                &input,
                &None,
                &Some(output),
                None,
                &control
            );
        });
    });
}
