// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
pub use self::MaybeTyped::*;

use rustc_driver::driver;
use rustc::session::{mod, config};
use rustc::metadata::decoder;
use rustc::middle::{privacy, ty};
use rustc::lint;
use rustc_trans::back::link;

use syntax::{ast, ast_map, ast_util, codemap, diagnostic};
use syntax::parse::token;
use syntax::ptr::P;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use arena::TypedArena;

use visit_ast::RustdocVisitor;
use clean;
use clean::Clean;

/// Are we generating documentation (`Typed`) or tests (`NotTyped`)?
pub enum MaybeTyped<'tcx> {
    Typed(ty::ctxt<'tcx>),
    NotTyped(session::Session)
}

pub type ExternalPaths = RefCell<Option<HashMap<ast::DefId,
                                                (Vec<String>, clean::TypeKind)>>>;

pub struct DocContext<'tcx> {
    pub krate: &'tcx ast::Crate,
    pub maybe_typed: MaybeTyped<'tcx>,
    pub src: Path,
    pub external_paths: ExternalPaths,
    pub external_traits: RefCell<Option<HashMap<ast::DefId, clean::Trait>>>,
    pub external_typarams: RefCell<Option<HashMap<ast::DefId, String>>>,
    pub inlined: RefCell<Option<HashSet<ast::DefId>>>,
    pub populated_crate_impls: RefCell<HashSet<ast::CrateNum>>,
}

impl<'tcx> DocContext<'tcx> {
    pub fn sess<'a>(&'a self) -> &'a session::Session {
        match self.maybe_typed {
            Typed(ref tcx) => &tcx.sess,
            NotTyped(ref sess) => sess
        }
    }

    pub fn tcx_opt<'a>(&'a self) -> Option<&'a ty::ctxt<'tcx>> {
        match self.maybe_typed {
            Typed(ref tcx) => Some(tcx),
            NotTyped(_) => None
        }
    }

    pub fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        let tcx_opt = self.tcx_opt();
        tcx_opt.expect("tcx not present")
    }
}

pub struct CrateAnalysis {
    pub exported_items: privacy::ExportedItems,
    pub public_items: privacy::PublicItems,
    pub external_paths: ExternalPaths,
    pub external_traits: RefCell<Option<HashMap<ast::DefId, clean::Trait>>>,
    pub external_typarams: RefCell<Option<HashMap<ast::DefId, String>>>,
    pub inlined: RefCell<Option<HashSet<ast::DefId>>>,
}

pub type Externs = HashMap<String, Vec<String>>;

pub fn run_core(libs: Vec<Path>, cfgs: Vec<String>, externs: Externs,
                cpath: &Path, triple: Option<String>)
                -> (clean::Crate, CrateAnalysis) {

    // Parse, resolve, and typecheck the given crate.

    let input = config::Input::File(cpath.clone());

    let warning_lint = lint::builtin::WARNINGS.name_lower();

    let sessopts = config::Options {
        maybe_sysroot: None,
        addl_lib_search_paths: RefCell::new(libs),
        crate_types: vec!(config::CrateTypeRlib),
        lint_opts: vec!((warning_lint, lint::Allow)),
        externs: externs,
        target_triple: triple.unwrap_or(config::host_triple().to_string()),
        cfg: config::parse_cfgspecs(cfgs),
        ..config::basic_options().clone()
    };


    let codemap = codemap::CodeMap::new();
    let diagnostic_handler = diagnostic::default_handler(diagnostic::Auto, None);
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, codemap);

    let sess = session::build_session_(sessopts,
                                       Some(cpath.clone()),
                                       span_diagnostic_handler);

    let cfg = config::build_configuration(&sess);

    let krate = driver::phase_1_parse_input(&sess, cfg, &input);

    let name = link::find_crate_name(Some(&sess), krate.attrs.as_slice(),
                                     &input);

    let krate = driver::phase_2_configure_and_expand(&sess, krate, name.as_slice(), None)
                    .expect("phase_2_configure_and_expand aborted in rustdoc!");

    let mut forest = ast_map::Forest::new(krate);
    let ast_map = driver::assign_node_ids_and_map(&sess, &mut forest);

    let type_arena = TypedArena::new();
    let ty::CrateAnalysis {
        exported_items, public_items, ty_cx, ..
    } = driver::phase_3_run_analysis_passes(sess, ast_map, &type_arena, name);

    let ctxt = DocContext {
        krate: ty_cx.map.krate(),
        maybe_typed: Typed(ty_cx),
        src: cpath.clone(),
        external_traits: RefCell::new(Some(HashMap::new())),
        external_typarams: RefCell::new(Some(HashMap::new())),
        external_paths: RefCell::new(Some(HashMap::new())),
        inlined: RefCell::new(Some(HashSet::new())),
        populated_crate_impls: RefCell::new(HashSet::new()),
    };
    debug!("crate: {}", ctxt.krate);

    let analysis = CrateAnalysis {
        exported_items: exported_items,
        public_items: public_items,
        external_paths: RefCell::new(None),
        external_traits: RefCell::new(None),
        external_typarams: RefCell::new(None),
        inlined: RefCell::new(None),
    };

    let krate = {
        let mut v = RustdocVisitor::new(&ctxt, Some(&analysis));
        v.visit(ctxt.krate);
        v.clean(&ctxt)
    };

    let external_paths = ctxt.external_paths.borrow_mut().take();
    *analysis.external_paths.borrow_mut() = external_paths;
    let map = ctxt.external_traits.borrow_mut().take();
    *analysis.external_traits.borrow_mut() = map;
    let map = ctxt.external_typarams.borrow_mut().take();
    *analysis.external_typarams.borrow_mut() = map;
    let map = ctxt.inlined.borrow_mut().take();
    *analysis.inlined.borrow_mut() = map;
    (krate, analysis)
}

pub fn run_core_with_lib(libpath: &Path, srcpath: &Path) -> (clean::Crate, CrateAnalysis) {
    let codemap = codemap::CodeMap::new();
    let diagnostic_handler = diagnostic::default_handler(diagnostic::Auto, None);
    let span_diagnostic_handler = diagnostic::mk_span_handler(diagnostic_handler, codemap);

    let dummy_crate_name = "__crate";
    let dummy_view_name = "__view";

    let mut opts = config::basic_options();
    opts.externs.insert(dummy_crate_name.into_string(),
                        vec![libpath.as_str().unwrap().into_string()]);

    let sess = session::build_session_(opts, None, span_diagnostic_handler);

    // dummy AST to faciliate the crate loading
    let dummy_crate_ident = ast::Ident::new(token::gensym(dummy_crate_name));
    let dummy_view_ident = ast::Ident::new(token::gensym(dummy_view_name));
    let krate = ast::Crate {
        module: ast::Mod {
            inner: codemap::DUMMY_SP,
            view_items: vec![
                ast::ViewItem {
                    node: ast::ViewItemExternCrate(
                        dummy_view_ident,
                        Some((token::get_ident(dummy_crate_ident), ast::CookedStr)),
                        ast::DUMMY_NODE_ID),
                    attrs: Vec::new(),
                    vis: ast::Inherited,
                    span: codemap::DUMMY_SP,
                },
                ast::ViewItem {
                    node: ast::ViewItemUse(
                        P(codemap::dummy_spanned(ast::ViewPathSimple(
                            dummy_crate_ident,
                            ast::Path {
                                span: codemap::DUMMY_SP,
                                global: false,
                                segments: vec![
                                    ast::PathSegment {
                                        identifier: dummy_view_ident, 
                                        parameters: ast::PathParameters::none(),
                                    },
                                ],
                            },
                            ast::DUMMY_NODE_ID)
                        ))
                    ),
                    attrs: Vec::new(),
                    vis: ast::Public,
                    span: codemap::DUMMY_SP,
                },
            ],
            items: Vec::new(),
        },
        attrs: Vec::new(),
        config: Vec::new(),
        span: codemap::DUMMY_SP,
        exported_macros: Vec::new(),
    };

    let mut forest = ast_map::Forest::new(krate);
    let ast_map = driver::assign_node_ids_and_map(&sess, &mut forest);

    let type_arena = TypedArena::new();
    let ty::CrateAnalysis {
        exported_items, public_items, ty_cx, ..
    } = driver::phase_3_run_analysis_passes(sess, ast_map, &type_arena,
                                            dummy_crate_name.into_string());

    let ctxt = DocContext {
        krate: ty_cx.map.krate(),
        maybe_typed: Typed(ty_cx),
        src: srcpath.clone(),
        external_traits: RefCell::new(Some(HashMap::new())),
        external_typarams: RefCell::new(Some(HashMap::new())),
        external_paths: RefCell::new(Some(HashMap::new())),
        inlined: RefCell::new(Some(HashSet::new())),
        populated_crate_impls: RefCell::new(HashSet::new()),
    };

    // there should be only one Def available, namely reexport
    let mut view_node_id = None;
    for (id, _) in ctxt.tcx().def_map.borrow().iter() {
        assert!(view_node_id.is_none(), "multiple Defs available");
        view_node_id = Some(*id);
    }
    let view_node_id = view_node_id.expect("no Def available");

    // we now have all necessary environments, try to inline.
    let inlined = clean::inline::try_inline(&ctxt, view_node_id, None);
    let inlined = inlined.expect("cannot inline crate");
    if inlined.len() != 1 {
        panic!("cannot inline crate");
    }
    let inlined = inlined.into_iter().next().unwrap();

    // we still have to fill some gaps, so get the crate data in our hands
    let crate_num = 1; // we don't have std injection so this should be the first
    let crate_data = ctxt.sess().cstore.get_crate_data(crate_num);
    let crate_name = crate_data.name();

    // fix external_paths for the given crate_num
    {
        let mut external_paths = ctxt.external_paths.borrow_mut();
        for (def_id, &(ref mut fqn, _)) in external_paths.as_mut().unwrap().iter_mut() {
            if def_id.krate == crate_num {
                assert_eq!(fqn.head().map(|s| s.as_slice()), Some(dummy_crate_name));
                if fqn.len() == 1 {
                    fqn[0] = "".into_string();
                } else {
                    fqn.remove(0);
                }
            }
        }
    }

    let postclean = RefCell::new(Some((crate_name, srcpath.clone(), inlined)));
    let mut krate = postclean.clean(&ctxt);

    // why do we have both crate attributes and item attributes?!
    let crate_attrs = decoder::get_crate_attributes(crate_data.data());
    {
        let mut attrs = &mut krate.module.as_mut().unwrap().attrs;
        attrs.extend(crate_attrs.clean(&ctxt).into_iter());
    }

    // the reconstructed crate doesn't have exported macros (yet)
    let macros = decoder::get_exported_macros(crate_data.data());
    {
        let mut module = match krate.module {
            Some(clean::Item { inner: clean::ModuleItem(ref mut module), .. }) => module,
            _ => panic!("unexpectedly cleaned crate")
        };
        for macro in macros.into_iter() {
            // XXX okay, this is bad. the metadata doesn't have a direct macro name.
            // for now we try to recognize `macro_rules!\s*([^/({\[]+)`.
            // hope someone doesn't come up with `macro_rules! /*screw doc*/ foo()`...
            let macname = {
                let macro = macro.trim_left();
                if macro.starts_with("macro_rules!") {
                    let macro = macro.slice_from(12).trim_left();
                    let sep = macro.find(['/', '(', '{', '['].as_slice());
                    if let Some(sep) = sep {
                        Some(format!("{}!", macro.slice_to(sep).trim_right()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            module.items.push(clean::Item {
                name: macname,
                attrs: Vec::new(),
                source: clean::Span::empty(),
                visibility: ast::Public.clean(&ctxt),
                stability: None,
                def_id: ast_util::local_def(ast::DUMMY_NODE_ID),
                inner: clean::MacroItem(clean::Macro {
                    source: macro,
                }),
            });
        }
    }

    // we need the analysis for later uses
    let DocContext {
        external_paths, external_traits, external_typarams, inlined, ..
    } = ctxt;
    let analysis = CrateAnalysis {
        exported_items: exported_items,
        public_items: public_items,
        external_paths: external_paths,
        external_traits: external_traits,
        external_typarams: external_typarams,
        inlined: inlined,
    };

    (krate, analysis)
}
