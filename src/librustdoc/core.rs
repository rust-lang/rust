// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc;
use rustc::{driver, middle};
use rustc::middle::privacy;

use syntax::ast;
use syntax::diagnostic;
use syntax::parse;
use syntax;

use std::os;
use std::local_data;
use std::hashmap::{HashSet};

use visit_ast::RustdocVisitor;
use clean;
use clean::Clean;

pub struct DocContext {
    crate: ast::Crate,
    tycx: middle::ty::ctxt,
    sess: driver::session::Session
}

pub struct CrateAnalysis {
    exported_items: privacy::ExportedItems,
}

/// Parses, resolves, and typechecks the given crate
fn get_ast_and_resolve(cpath: &Path,
                       libs: HashSet<Path>) -> (DocContext, CrateAnalysis) {
    use syntax::codemap::dummy_spanned;
    use rustc::driver::driver::{file_input, build_configuration,
                                phase_1_parse_input,
                                phase_2_configure_and_expand,
                                phase_3_run_analysis_passes};

    let parsesess = parse::new_parse_sess(None);
    let input = file_input(cpath.clone());

    let sessopts = @driver::session::options {
        binary: @"rustdoc",
        maybe_sysroot: Some(@os::self_exe_path().unwrap().dir_path()),
        addl_lib_search_paths: @mut libs,
        .. (*rustc::driver::session::basic_options()).clone()
    };


    let diagnostic_handler = syntax::diagnostic::mk_handler(None);
    let span_diagnostic_handler =
        syntax::diagnostic::mk_span_handler(diagnostic_handler, parsesess.cm);

    let sess = driver::driver::build_session_(sessopts,
                                              parsesess.cm,
                                              @diagnostic::DefaultEmitter as
                                                @diagnostic::Emitter,
                                              span_diagnostic_handler);

    let mut cfg = build_configuration(sess);
    cfg.push(@dummy_spanned(ast::MetaWord(@"stage2")));

    let mut crate = phase_1_parse_input(sess, cfg.clone(), &input);
    crate = phase_2_configure_and_expand(sess, cfg, crate);
    let driver::driver::CrateAnalysis {
        exported_items, ty_cx, _
    } = phase_3_run_analysis_passes(sess, &crate);

    debug!("crate: {:?}", crate);
    return (DocContext { crate: crate, tycx: ty_cx, sess: sess },
            CrateAnalysis { exported_items: exported_items });
}

pub fn run_core (libs: HashSet<Path>, path: &Path) -> (clean::Crate, CrateAnalysis) {
    let (ctxt, analysis) = get_ast_and_resolve(path, libs);
    let ctxt = @ctxt;
    debug!("defmap:");
    for (k, v) in ctxt.tycx.def_map.iter() {
        debug!("{:?}: {:?}", k, v);
    }
    local_data::set(super::ctxtkey, ctxt);

    let v = @mut RustdocVisitor::new();
    v.visit(&ctxt.crate);

    (v.clean(), analysis)
}
