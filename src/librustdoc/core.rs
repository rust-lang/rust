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
use rustc::middle::lint;

use syntax::ast;
use syntax::parse::token;
use syntax;

use std::cell::RefCell;
use std::os;
use collections::{HashSet, HashMap};

use visit_ast::RustdocVisitor;
use clean;
use clean::Clean;

pub enum MaybeTyped {
    Typed(middle::ty::ctxt),
    NotTyped(driver::session::Session)
}

pub type ExternalPaths = RefCell<Option<HashMap<ast::DefId,
                                                (Vec<~str>, clean::TypeKind)>>>;

pub struct DocContext {
    pub krate: ast::Crate,
    pub maybe_typed: MaybeTyped,
    pub src: Path,
    pub external_paths: ExternalPaths,
}

impl DocContext {
    pub fn sess<'a>(&'a self) -> &'a driver::session::Session {
        match self.maybe_typed {
            Typed(ref tcx) => &tcx.sess,
            NotTyped(ref sess) => sess
        }
    }
}

pub struct CrateAnalysis {
    pub exported_items: privacy::ExportedItems,
    pub public_items: privacy::PublicItems,
    pub external_paths: ExternalPaths,
}

/// Parses, resolves, and typechecks the given crate
fn get_ast_and_resolve(cpath: &Path, libs: HashSet<Path>, cfgs: Vec<~str>)
                       -> (DocContext, CrateAnalysis) {
    use syntax::codemap::dummy_spanned;
    use rustc::driver::driver::{FileInput, build_configuration,
                                phase_1_parse_input,
                                phase_2_configure_and_expand,
                                phase_3_run_analysis_passes};

    let input = FileInput(cpath.clone());

    let sessopts = driver::session::Options {
        maybe_sysroot: Some(os::self_exe_path().unwrap().dir_path()),
        addl_lib_search_paths: RefCell::new(libs),
        crate_types: vec!(driver::session::CrateTypeDylib),
        lint_opts: vec!((lint::Warnings, lint::allow)),
        ..rustc::driver::session::basic_options().clone()
    };


    let codemap = syntax::codemap::CodeMap::new();
    let diagnostic_handler = syntax::diagnostic::default_handler();
    let span_diagnostic_handler =
        syntax::diagnostic::mk_span_handler(diagnostic_handler, codemap);

    let sess = driver::driver::build_session_(sessopts,
                                              Some(cpath.clone()),
                                              span_diagnostic_handler);

    let mut cfg = build_configuration(&sess);
    for cfg_ in cfgs.move_iter() {
        let cfg_ = token::intern_and_get_ident(cfg_);
        cfg.push(@dummy_spanned(ast::MetaWord(cfg_)));
    }

    let krate = phase_1_parse_input(&sess, cfg, &input);
    let (krate, ast_map) = phase_2_configure_and_expand(&sess, &mut Loader::new(&sess),
                                                        krate, &from_str("rustdoc").unwrap());
    let driver::driver::CrateAnalysis {
        exported_items, public_items, ty_cx, ..
    } = phase_3_run_analysis_passes(sess, &krate, ast_map);

    debug!("crate: {:?}", krate);
    (DocContext {
        krate: krate,
        maybe_typed: Typed(ty_cx),
        src: cpath.clone(),
        external_paths: RefCell::new(Some(HashMap::new())),
    }, CrateAnalysis {
        exported_items: exported_items,
        public_items: public_items,
        external_paths: RefCell::new(None),
    })
}

pub fn run_core(libs: HashSet<Path>, cfgs: Vec<~str>, path: &Path)
                -> (clean::Crate, CrateAnalysis) {
    let (ctxt, analysis) = get_ast_and_resolve(path, libs, cfgs);
    let ctxt = @ctxt;
    super::ctxtkey.replace(Some(ctxt));

    let krate = {
        let mut v = RustdocVisitor::new(ctxt, Some(&analysis));
        v.visit(&ctxt.krate);
        v.clean()
    };

    let external_paths = ctxt.external_paths.borrow_mut().take();
    *analysis.external_paths.borrow_mut() = external_paths;
    (krate, analysis)
}
