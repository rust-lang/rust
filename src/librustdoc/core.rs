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
use rustc::metadata::creader::Loader;
use rustc::middle::privacy;

use syntax::ast;
use syntax::diagnostic;
use syntax::parse::token;
use syntax::parse;
use syntax;

use std::cell::RefCell;
use std::os;
use std::local_data;
use std::hashmap::{HashSet};

use visit_ast::RustdocVisitor;
use clean;
use clean::Clean;

pub struct DocContext {
    crate: ast::Crate,
    tycx: Option<middle::ty::ctxt>,
    sess: driver::session::Session
}

pub struct CrateAnalysis {
    exported_items: privacy::ExportedItems,
    public_items: privacy::PublicItems,
}

/// Parses, resolves, and typechecks the given crate
fn get_ast_and_resolve(cpath: &Path,
                       libs: HashSet<Path>, cfgs: ~[~str]) -> (DocContext, CrateAnalysis) {
    use syntax::codemap::dummy_spanned;
    use rustc::driver::driver::{FileInput, build_configuration,
                                phase_1_parse_input,
                                phase_2_configure_and_expand,
                                phase_3_run_analysis_passes};

    let parsesess = parse::new_parse_sess(None);
    let input = FileInput(cpath.clone());

    let sessopts = @driver::session::Options {
        binary: ~"rustdoc",
        maybe_sysroot: Some(@os::self_exe_path().unwrap().dir_path()),
        addl_lib_search_paths: @RefCell::new(libs),
        outputs: ~[driver::session::OutputDylib],
        .. (*rustc::driver::session::basic_options()).clone()
    };


    let diagnostic_handler = syntax::diagnostic::mk_handler(None);
    let span_diagnostic_handler =
        syntax::diagnostic::mk_span_handler(diagnostic_handler, parsesess.cm);

    let sess = driver::driver::build_session_(sessopts,
                                              Some(cpath.clone()),
                                              parsesess.cm,
                                              @diagnostic::DefaultEmitter,
                                              span_diagnostic_handler);

    let mut cfg = build_configuration(sess);
    for cfg_ in cfgs.move_iter() {
        let cfg_ = token::intern_and_get_ident(cfg_);
        cfg.push(@dummy_spanned(ast::MetaWord(cfg_)));
    }

    let crate = phase_1_parse_input(sess, cfg.clone(), &input);
    let loader = &mut Loader::new(sess);
    let (crate, ast_map) = phase_2_configure_and_expand(sess, cfg, loader, crate);
    let driver::driver::CrateAnalysis {
        exported_items, public_items, ty_cx, ..
    } = phase_3_run_analysis_passes(sess, &crate, ast_map);

    debug!("crate: {:?}", crate);
    return (DocContext { crate: crate, tycx: Some(ty_cx), sess: sess },
            CrateAnalysis {
                exported_items: exported_items,
                public_items: public_items,
            });
}

pub fn run_core (libs: HashSet<Path>, cfgs: ~[~str], path: &Path) -> (clean::Crate, CrateAnalysis) {
    let (ctxt, analysis) = get_ast_and_resolve(path, libs, cfgs);
    let ctxt = @ctxt;
    local_data::set(super::ctxtkey, ctxt);

    let crate = {
        let mut v = RustdocVisitor::new(ctxt, Some(&analysis));
        v.visit(&ctxt.crate);
        v.clean()
    };

    (crate, analysis)
}
