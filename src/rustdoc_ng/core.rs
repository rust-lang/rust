use rustc;
use rustc::{driver, middle};

use syntax;
use syntax::parse;
use syntax::ast;

use std::os;
use std::local_data;

use visit_ast::RustdocVisitor;
use clean;
use clean::Clean;

pub struct DocContext {
    crate: @ast::Crate,
    tycx: middle::ty::ctxt,
    sess: driver::session::Session
}

/// Parses, resolves, and typechecks the given crate
fn get_ast_and_resolve(cpath: &Path, libs: ~[Path]) -> DocContext {
    use syntax::codemap::dummy_spanned;
    use rustc::driver::driver::*;

    let parsesess = parse::new_parse_sess(None);
    let input = file_input(cpath.clone());

    let sessopts = @driver::session::options {
        binary: @"rustdoc",
        maybe_sysroot: Some(@os::self_exe_path().unwrap().pop()),
        addl_lib_search_paths: @mut libs,
        .. (*rustc::driver::session::basic_options()).clone()
    };


    let diagnostic_handler = syntax::diagnostic::mk_handler(None);
    let span_diagnostic_handler =
        syntax::diagnostic::mk_span_handler(diagnostic_handler, parsesess.cm);

    let sess = driver::driver::build_session_(sessopts, parsesess.cm,
                                                  syntax::diagnostic::emit,
                                                  span_diagnostic_handler);

    let mut cfg = build_configuration(sess);
    cfg.push(@dummy_spanned(ast::MetaWord(@"stage2")));

    let mut crate = phase_1_parse_input(sess, cfg.clone(), &input);
    crate = phase_2_configure_and_expand(sess, cfg, crate);
    let analysis = phase_3_run_analysis_passes(sess, crate);

    debug!("crate: %?", crate);
    DocContext { crate: crate, tycx: analysis.ty_cx, sess: sess }
}

pub fn run_core (libs: ~[Path], path: &Path) -> clean::Crate {
    let ctxt = @get_ast_and_resolve(path, libs);
    debug!("defmap:");
    for (k, v) in ctxt.tycx.def_map.iter() {
        debug!("%?: %?", k, v);
    }
    local_data::set(super::ctxtkey, ctxt);

    let v = @mut RustdocVisitor::new();
    v.visit(ctxt.crate);

    v.clean()
}
