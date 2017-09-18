// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Standalone Tests for the Inference Module

use std::path::PathBuf;
use std::sync::mpsc;

use driver;
use rustc_lint;
use rustc_resolve::MakeGlobMap;
use rustc_trans;
use rustc::middle::free_region::FreeRegionMap;
use rustc::middle::region;
use rustc::middle::resolve_lifetime;
use rustc::ty::subst::{Kind, Subst};
use rustc::traits::{ObligationCause, Reveal};
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc::infer::{self, InferOk, InferResult};
use rustc::infer::type_variable::TypeVariableOrigin;
use rustc_metadata::cstore::CStore;
use rustc::hir::map as hir_map;
use rustc::mir::transform::Passes;
use rustc::session::{self, config};
use rustc::session::config::{OutputFilenames, OutputTypes};
use std::rc::Rc;
use syntax::ast;
use syntax::abi::Abi;
use syntax::codemap::{CodeMap, FilePathMapping};
use errors;
use errors::emitter::Emitter;
use errors::{Level, DiagnosticBuilder};
use syntax::feature_gate::UnstableFeatures;
use syntax::symbol::Symbol;
use syntax_pos::DUMMY_SP;
use arena::DroplessArena;

use rustc::hir;

struct Env<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    infcx: &'a infer::InferCtxt<'a, 'gcx, 'tcx>,
    region_scope_tree: &'a mut region::ScopeTree,
    param_env: ty::ParamEnv<'tcx>,
}

struct RH<'a> {
    id: hir::ItemLocalId,
    sub: &'a [RH<'a>],
}

const EMPTY_SOURCE_STR: &'static str = "#![feature(no_core)] #![no_core]";

struct ExpectErrorEmitter {
    messages: Vec<String>,
}

fn remove_message(e: &mut ExpectErrorEmitter, msg: &str, lvl: Level) {
    match lvl {
        Level::Bug | Level::Fatal | Level::Error => {}
        _ => {
            return;
        }
    }

    debug!("Error: {}", msg);
    match e.messages.iter().position(|m| msg.contains(m)) {
        Some(i) => {
            e.messages.remove(i);
        }
        None => {
            debug!("Unexpected error: {} Expected: {:?}", msg, e.messages);
            panic!("Unexpected error: {} Expected: {:?}", msg, e.messages);
        }
    }
}

impl Emitter for ExpectErrorEmitter {
    fn emit(&mut self, db: &DiagnosticBuilder) {
        remove_message(self, &db.message(), db.level);
        for child in &db.children {
            remove_message(self, &child.message(), child.level);
        }
    }
}

fn errors(msgs: &[&str]) -> (Box<Emitter + Send>, usize) {
    let v = msgs.iter().map(|m| m.to_string()).collect();
    (box ExpectErrorEmitter { messages: v } as Box<Emitter + Send>, msgs.len())
}

fn test_env<F>(source_string: &str,
               (emitter, expected_err_count): (Box<Emitter + Send>, usize),
               body: F)
    where F: FnOnce(Env)
{
    let mut options = config::basic_options();
    options.debugging_opts.verbose = true;
    options.unstable_features = UnstableFeatures::Allow;
    let diagnostic_handler = errors::Handler::with_emitter(true, false, emitter);

    let cstore = Rc::new(CStore::new(box ::MetadataLoader));
    let sess = session::build_session_(options,
                                       None,
                                       diagnostic_handler,
                                       Rc::new(CodeMap::new(FilePathMapping::empty())));
    rustc_trans::init(&sess);
    rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));
    let input = config::Input::Str {
        name: driver::anon_src(),
        input: source_string.to_string(),
    };
    let krate = driver::phase_1_parse_input(&driver::CompileController::basic(),
                                            &sess,
                                            &input).unwrap();
    let driver::ExpansionResult { defs, resolutions, mut hir_forest, .. } = {
        driver::phase_2_configure_and_expand(&sess,
                                             &cstore,
                                             krate,
                                             None,
                                             "test",
                                             None,
                                             MakeGlobMap::No,
                                             |_| Ok(()))
            .expect("phase 2 aborted")
    };

    let arena = DroplessArena::new();
    let arenas = ty::GlobalArenas::new();
    let hir_map = hir_map::map_crate(&mut hir_forest, &defs);

    // run just enough stuff to build a tcx:
    let named_region_map = resolve_lifetime::krate(&sess, &*cstore, &hir_map);
    let (tx, _rx) = mpsc::channel();
    let outputs = OutputFilenames {
        out_directory: PathBuf::new(),
        out_filestem: String::new(),
        single_output_file: None,
        extra: String::new(),
        outputs: OutputTypes::new(&[]),
    };
    TyCtxt::create_and_enter(&sess,
                             &*cstore,
                             ty::maps::Providers::default(),
                             ty::maps::Providers::default(),
                             Rc::new(Passes::new()),
                             &arenas,
                             &arena,
                             resolutions,
                             named_region_map.unwrap(),
                             hir_map,
                             "test_crate",
                             tx,
                             &outputs,
                             |tcx| {
        tcx.infer_ctxt().enter(|infcx| {
            let mut region_scope_tree = region::ScopeTree::default();
            body(Env {
                infcx: &infcx,
                region_scope_tree: &mut region_scope_tree,
                param_env: ty::ParamEnv::empty(Reveal::UserFacing),
            });
            let free_regions = FreeRegionMap::new();
            let def_id = tcx.hir.local_def_id(ast::CRATE_NODE_ID);
            infcx.resolve_regions_and_report_errors(def_id, &region_scope_tree, &free_regions);
            assert_eq!(tcx.sess.err_count(), expected_err_count);
        });
    });
}

impl<'a, 'gcx, 'tcx> Env<'a, 'gcx, 'tcx> {
    pub fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    pub fn create_region_hierarchy(&mut self, rh: &RH, parent: region::Scope) {
        let me = region::Scope::Node(rh.id);
        self.region_scope_tree.record_scope_parent(me, Some(parent));
        for child_rh in rh.sub {
            self.create_region_hierarchy(child_rh, me);
        }
    }

    pub fn create_simple_region_hierarchy(&mut self) {
        // creates a region hierarchy where 1 is root, 10 and 11 are
        // children of 1, etc

        let dscope = region::Scope::Destruction(hir::ItemLocalId(1));
        self.region_scope_tree.record_scope_parent(dscope, None);
        self.create_region_hierarchy(&RH {
            id: hir::ItemLocalId(1),
            sub: &[RH {
                id: hir::ItemLocalId(10),
                sub: &[],
            },
            RH {
                id: hir::ItemLocalId(11),
                sub: &[],
            }],
        }, dscope);
    }

    #[allow(dead_code)] // this seems like it could be useful, even if we don't use it now
    pub fn lookup_item(&self, names: &[String]) -> ast::NodeId {
        return match search_mod(self, &self.infcx.tcx.hir.krate().module, 0, names) {
            Some(id) => id,
            None => {
                panic!("no item found: `{}`", names.join("::"));
            }
        };

        fn search_mod(this: &Env,
                      m: &hir::Mod,
                      idx: usize,
                      names: &[String])
                      -> Option<ast::NodeId> {
            assert!(idx < names.len());
            for item in &m.item_ids {
                let item = this.infcx.tcx.hir.expect_item(item.id);
                if item.name.to_string() == names[idx] {
                    return search(this, item, idx + 1, names);
                }
            }
            return None;
        }

        fn search(this: &Env, it: &hir::Item, idx: usize, names: &[String]) -> Option<ast::NodeId> {
            if idx == names.len() {
                return Some(it.id);
            }

            return match it.node {
                hir::ItemUse(..) |
                hir::ItemExternCrate(..) |
                hir::ItemConst(..) |
                hir::ItemStatic(..) |
                hir::ItemFn(..) |
                hir::ItemForeignMod(..) |
                hir::ItemGlobalAsm(..) |
                hir::ItemTy(..) => None,

                hir::ItemEnum(..) |
                hir::ItemStruct(..) |
                hir::ItemUnion(..) |
                hir::ItemTrait(..) |
                hir::ItemImpl(..) |
                hir::ItemDefaultImpl(..) => None,

                hir::ItemMod(ref m) => search_mod(this, m, idx, names),
            };
        }
    }

    pub fn make_subtype(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
        match self.infcx.at(&ObligationCause::dummy(), self.param_env).sub(a, b) {
            Ok(_) => true,
            Err(ref e) => panic!("Encountered error: {}", e),
        }
    }

    pub fn is_subtype(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
        self.infcx.can_sub(self.param_env, a, b).is_ok()
    }

    pub fn assert_subtype(&self, a: Ty<'tcx>, b: Ty<'tcx>) {
        if !self.is_subtype(a, b) {
            panic!("{} is not a subtype of {}, but it should be", a, b);
        }
    }

    pub fn assert_eq(&self, a: Ty<'tcx>, b: Ty<'tcx>) {
        self.assert_subtype(a, b);
        self.assert_subtype(b, a);
    }

    pub fn t_fn(&self, input_tys: &[Ty<'tcx>], output_ty: Ty<'tcx>) -> Ty<'tcx> {
        self.infcx.tcx.mk_fn_ptr(ty::Binder(self.infcx.tcx.mk_fn_sig(
            input_tys.iter().cloned(),
            output_ty,
            false,
            hir::Unsafety::Normal,
            Abi::Rust
        )))
    }

    pub fn t_nil(&self) -> Ty<'tcx> {
        self.infcx.tcx.mk_nil()
    }

    pub fn t_pair(&self, ty1: Ty<'tcx>, ty2: Ty<'tcx>) -> Ty<'tcx> {
        self.infcx.tcx.intern_tup(&[ty1, ty2], false)
    }

    pub fn t_param(&self, index: u32) -> Ty<'tcx> {
        let name = format!("T{}", index);
        self.infcx.tcx.mk_param(index, Symbol::intern(&name))
    }

    pub fn re_early_bound(&self, index: u32, name: &'static str) -> ty::Region<'tcx> {
        let name = Symbol::intern(name);
        self.infcx.tcx.mk_region(ty::ReEarlyBound(ty::EarlyBoundRegion {
            def_id: self.infcx.tcx.hir.local_def_id(ast::CRATE_NODE_ID),
            index,
            name,
        }))
    }

    pub fn re_late_bound_with_debruijn(&self,
                                       id: u32,
                                       debruijn: ty::DebruijnIndex)
                                       -> ty::Region<'tcx> {
        self.infcx.tcx.mk_region(ty::ReLateBound(debruijn, ty::BrAnon(id)))
    }

    pub fn t_rptr(&self, r: ty::Region<'tcx>) -> Ty<'tcx> {
        self.infcx.tcx.mk_imm_ref(r, self.tcx().types.isize)
    }

    pub fn t_rptr_late_bound(&self, id: u32) -> Ty<'tcx> {
        let r = self.re_late_bound_with_debruijn(id, ty::DebruijnIndex::new(1));
        self.infcx.tcx.mk_imm_ref(r, self.tcx().types.isize)
    }

    pub fn t_rptr_late_bound_with_debruijn(&self,
                                           id: u32,
                                           debruijn: ty::DebruijnIndex)
                                           -> Ty<'tcx> {
        let r = self.re_late_bound_with_debruijn(id, debruijn);
        self.infcx.tcx.mk_imm_ref(r, self.tcx().types.isize)
    }

    pub fn t_rptr_scope(&self, id: u32) -> Ty<'tcx> {
        let r = ty::ReScope(region::Scope::Node(hir::ItemLocalId(id)));
        self.infcx.tcx.mk_imm_ref(self.infcx.tcx.mk_region(r), self.tcx().types.isize)
    }

    pub fn re_free(&self, id: u32) -> ty::Region<'tcx> {
        self.infcx.tcx.mk_region(ty::ReFree(ty::FreeRegion {
            scope: self.infcx.tcx.hir.local_def_id(ast::CRATE_NODE_ID),
            bound_region: ty::BrAnon(id),
        }))
    }

    pub fn t_rptr_free(&self, id: u32) -> Ty<'tcx> {
        let r = self.re_free(id);
        self.infcx.tcx.mk_imm_ref(r, self.tcx().types.isize)
    }

    pub fn t_rptr_static(&self) -> Ty<'tcx> {
        self.infcx.tcx.mk_imm_ref(self.infcx.tcx.types.re_static,
                                  self.tcx().types.isize)
    }

    pub fn t_rptr_empty(&self) -> Ty<'tcx> {
        self.infcx.tcx.mk_imm_ref(self.infcx.tcx.types.re_empty,
                                  self.tcx().types.isize)
    }

    pub fn sub(&self, t1: Ty<'tcx>, t2: Ty<'tcx>) -> InferResult<'tcx, ()> {
        self.infcx.at(&ObligationCause::dummy(), self.param_env).sub(t1, t2)
    }

    pub fn lub(&self, t1: Ty<'tcx>, t2: Ty<'tcx>) -> InferResult<'tcx, Ty<'tcx>> {
        self.infcx.at(&ObligationCause::dummy(), self.param_env).lub(t1, t2)
    }

    pub fn glb(&self, t1: Ty<'tcx>, t2: Ty<'tcx>) -> InferResult<'tcx, Ty<'tcx>> {
        self.infcx.at(&ObligationCause::dummy(), self.param_env).glb(t1, t2)
    }

    /// Checks that `t1 <: t2` is true (this may register additional
    /// region checks).
    pub fn check_sub(&self, t1: Ty<'tcx>, t2: Ty<'tcx>) {
        match self.sub(t1, t2) {
            Ok(InferOk { obligations, value: () }) => {
                // None of these tests should require nested obligations:
                assert!(obligations.is_empty());
            }
            Err(ref e) => {
                panic!("unexpected error computing sub({:?},{:?}): {}", t1, t2, e);
            }
        }
    }

    /// Checks that `t1 <: t2` is false (this may register additional
    /// region checks).
    pub fn check_not_sub(&self, t1: Ty<'tcx>, t2: Ty<'tcx>) {
        match self.sub(t1, t2) {
            Err(_) => {}
            Ok(_) => {
                panic!("unexpected success computing sub({:?},{:?})", t1, t2);
            }
        }
    }

    /// Checks that `LUB(t1,t2) == t_lub`
    pub fn check_lub(&self, t1: Ty<'tcx>, t2: Ty<'tcx>, t_lub: Ty<'tcx>) {
        match self.lub(t1, t2) {
            Ok(InferOk { obligations, value: t }) => {
                // None of these tests should require nested obligations:
                assert!(obligations.is_empty());

                self.assert_eq(t, t_lub);
            }
            Err(ref e) => panic!("unexpected error in LUB: {}", e),
        }
    }

    /// Checks that `GLB(t1,t2) == t_glb`
    pub fn check_glb(&self, t1: Ty<'tcx>, t2: Ty<'tcx>, t_glb: Ty<'tcx>) {
        debug!("check_glb(t1={}, t2={}, t_glb={})", t1, t2, t_glb);
        match self.glb(t1, t2) {
            Err(e) => panic!("unexpected error computing LUB: {:?}", e),
            Ok(InferOk { obligations, value: t }) => {
                // None of these tests should require nested obligations:
                assert!(obligations.is_empty());

                self.assert_eq(t, t_glb);

                // sanity check for good measure:
                self.assert_subtype(t, t1);
                self.assert_subtype(t, t2);
            }
        }
    }
}

#[test]
fn contravariant_region_ptr_ok() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |mut env| {
        env.create_simple_region_hierarchy();
        let t_rptr1 = env.t_rptr_scope(1);
        let t_rptr10 = env.t_rptr_scope(10);
        env.assert_eq(t_rptr1, t_rptr1);
        env.assert_eq(t_rptr10, t_rptr10);
        env.make_subtype(t_rptr1, t_rptr10);
    })
}

#[test]
fn contravariant_region_ptr_err() {
    test_env(EMPTY_SOURCE_STR, errors(&["mismatched types"]), |mut env| {
        env.create_simple_region_hierarchy();
        let t_rptr1 = env.t_rptr_scope(1);
        let t_rptr10 = env.t_rptr_scope(10);
        env.assert_eq(t_rptr1, t_rptr1);
        env.assert_eq(t_rptr10, t_rptr10);

        // will cause an error when regions are resolved
        env.make_subtype(t_rptr10, t_rptr1);
    })
}

#[test]
fn sub_free_bound_false() {
    //! Test that:
    //!
    //!     fn(&'a isize) <: for<'b> fn(&'b isize)
    //!
    //! does NOT hold.

    test_env(EMPTY_SOURCE_STR, errors(&[]), |mut env| {
        env.create_simple_region_hierarchy();
        let t_rptr_free1 = env.t_rptr_free(1);
        let t_rptr_bound1 = env.t_rptr_late_bound(1);
        env.check_not_sub(env.t_fn(&[t_rptr_free1], env.tcx().types.isize),
                          env.t_fn(&[t_rptr_bound1], env.tcx().types.isize));
    })
}

#[test]
fn sub_bound_free_true() {
    //! Test that:
    //!
    //!     for<'a> fn(&'a isize) <: fn(&'b isize)
    //!
    //! DOES hold.

    test_env(EMPTY_SOURCE_STR, errors(&[]), |mut env| {
        env.create_simple_region_hierarchy();
        let t_rptr_bound1 = env.t_rptr_late_bound(1);
        let t_rptr_free1 = env.t_rptr_free(1);
        env.check_sub(env.t_fn(&[t_rptr_bound1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_free1], env.tcx().types.isize));
    })
}

#[test]
fn sub_free_bound_false_infer() {
    //! Test that:
    //!
    //!     fn(_#1) <: for<'b> fn(&'b isize)
    //!
    //! does NOT hold for any instantiation of `_#1`.

    test_env(EMPTY_SOURCE_STR, errors(&[]), |env| {
        let t_infer1 = env.infcx.next_ty_var(TypeVariableOrigin::MiscVariable(DUMMY_SP));
        let t_rptr_bound1 = env.t_rptr_late_bound(1);
        env.check_not_sub(env.t_fn(&[t_infer1], env.tcx().types.isize),
                          env.t_fn(&[t_rptr_bound1], env.tcx().types.isize));
    })
}

#[test]
fn lub_free_bound_infer() {
    //! Test result of:
    //!
    //!     LUB(fn(_#1), for<'b> fn(&'b isize))
    //!
    //! This should yield `fn(&'_ isize)`. We check
    //! that it yields `fn(&'x isize)` for some free `'x`,
    //! anyhow.

    test_env(EMPTY_SOURCE_STR, errors(&[]), |mut env| {
        env.create_simple_region_hierarchy();
        let t_infer1 = env.infcx.next_ty_var(TypeVariableOrigin::MiscVariable(DUMMY_SP));
        let t_rptr_bound1 = env.t_rptr_late_bound(1);
        let t_rptr_free1 = env.t_rptr_free(1);
        env.check_lub(env.t_fn(&[t_infer1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_bound1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_free1], env.tcx().types.isize));
    });
}

#[test]
fn lub_bound_bound() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(1);
        let t_rptr_bound2 = env.t_rptr_late_bound(2);
        env.check_lub(env.t_fn(&[t_rptr_bound1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_bound2], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_bound1], env.tcx().types.isize));
    })
}

#[test]
fn lub_bound_free() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |mut env| {
        env.create_simple_region_hierarchy();
        let t_rptr_bound1 = env.t_rptr_late_bound(1);
        let t_rptr_free1 = env.t_rptr_free(1);
        env.check_lub(env.t_fn(&[t_rptr_bound1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_free1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_free1], env.tcx().types.isize));
    })
}

#[test]
fn lub_bound_static() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(1);
        let t_rptr_static = env.t_rptr_static();
        env.check_lub(env.t_fn(&[t_rptr_bound1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_static], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_static], env.tcx().types.isize));
    })
}

#[test]
fn lub_bound_bound_inverse_order() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(1);
        let t_rptr_bound2 = env.t_rptr_late_bound(2);
        env.check_lub(env.t_fn(&[t_rptr_bound1, t_rptr_bound2], t_rptr_bound1),
                      env.t_fn(&[t_rptr_bound2, t_rptr_bound1], t_rptr_bound1),
                      env.t_fn(&[t_rptr_bound1, t_rptr_bound1], t_rptr_bound1));
    })
}

#[test]
fn lub_free_free() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |mut env| {
        env.create_simple_region_hierarchy();
        let t_rptr_free1 = env.t_rptr_free(1);
        let t_rptr_free2 = env.t_rptr_free(2);
        let t_rptr_static = env.t_rptr_static();
        env.check_lub(env.t_fn(&[t_rptr_free1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_free2], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_static], env.tcx().types.isize));
    })
}

#[test]
fn lub_returning_scope() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |mut env| {
        env.create_simple_region_hierarchy();
        let t_rptr_scope10 = env.t_rptr_scope(10);
        let t_rptr_scope11 = env.t_rptr_scope(11);
        let t_rptr_empty = env.t_rptr_empty();
        env.check_lub(env.t_fn(&[t_rptr_scope10], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_scope11], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_empty], env.tcx().types.isize));
    });
}

#[test]
fn glb_free_free_with_common_scope() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |mut env| {
        env.create_simple_region_hierarchy();
        let t_rptr_free1 = env.t_rptr_free(1);
        let t_rptr_free2 = env.t_rptr_free(2);
        let t_rptr_scope = env.t_rptr_scope(1);
        env.check_glb(env.t_fn(&[t_rptr_free1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_free2], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_scope], env.tcx().types.isize));
    })
}

#[test]
fn glb_bound_bound() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(1);
        let t_rptr_bound2 = env.t_rptr_late_bound(2);
        env.check_glb(env.t_fn(&[t_rptr_bound1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_bound2], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_bound1], env.tcx().types.isize));
    })
}

#[test]
fn glb_bound_free() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |mut env| {
        env.create_simple_region_hierarchy();
        let t_rptr_bound1 = env.t_rptr_late_bound(1);
        let t_rptr_free1 = env.t_rptr_free(1);
        env.check_glb(env.t_fn(&[t_rptr_bound1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_free1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_bound1], env.tcx().types.isize));
    })
}

#[test]
fn glb_bound_free_infer() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(1);
        let t_infer1 = env.infcx.next_ty_var(TypeVariableOrigin::MiscVariable(DUMMY_SP));

        // compute GLB(fn(_) -> isize, for<'b> fn(&'b isize) -> isize),
        // which should yield for<'b> fn(&'b isize) -> isize
        env.check_glb(env.t_fn(&[t_rptr_bound1], env.tcx().types.isize),
                      env.t_fn(&[t_infer1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_bound1], env.tcx().types.isize));

        // as a side-effect, computing GLB should unify `_` with
        // `&'_ isize`
        let t_resolve1 = env.infcx.shallow_resolve(t_infer1);
        match t_resolve1.sty {
            ty::TyRef(..) => {}
            _ => {
                panic!("t_resolve1={:?}", t_resolve1);
            }
        }
    })
}

#[test]
fn glb_bound_static() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(1);
        let t_rptr_static = env.t_rptr_static();
        env.check_glb(env.t_fn(&[t_rptr_bound1], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_static], env.tcx().types.isize),
                      env.t_fn(&[t_rptr_bound1], env.tcx().types.isize));
    })
}

/// Test substituting a bound region into a function, which introduces another level of binding.
/// This requires adjusting the Debruijn index.
#[test]
fn subst_ty_renumber_bound() {

    test_env(EMPTY_SOURCE_STR, errors(&[]), |env| {
        // Situation:
        // Theta = [A -> &'a foo]

        let t_rptr_bound1 = env.t_rptr_late_bound(1);

        // t_source = fn(A)
        let t_source = {
            let t_param = env.t_param(0);
            env.t_fn(&[t_param], env.t_nil())
        };

        let substs = env.infcx.tcx.intern_substs(&[Kind::from(t_rptr_bound1)]);
        let t_substituted = t_source.subst(env.infcx.tcx, substs);

        // t_expected = fn(&'a isize)
        let t_expected = {
            let t_ptr_bound2 = env.t_rptr_late_bound_with_debruijn(1, ty::DebruijnIndex::new(2));
            env.t_fn(&[t_ptr_bound2], env.t_nil())
        };

        debug!("subst_bound: t_source={:?} substs={:?} t_substituted={:?} t_expected={:?}",
               t_source,
               substs,
               t_substituted,
               t_expected);

        assert_eq!(t_substituted, t_expected);
    })
}

/// Test substituting a bound region into a function, which introduces another level of binding.
/// This requires adjusting the Debruijn index.
#[test]
fn subst_ty_renumber_some_bounds() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |env| {
        // Situation:
        // Theta = [A -> &'a foo]

        let t_rptr_bound1 = env.t_rptr_late_bound(1);

        // t_source = (A, fn(A))
        let t_source = {
            let t_param = env.t_param(0);
            env.t_pair(t_param, env.t_fn(&[t_param], env.t_nil()))
        };

        let substs = env.infcx.tcx.intern_substs(&[Kind::from(t_rptr_bound1)]);
        let t_substituted = t_source.subst(env.infcx.tcx, substs);

        // t_expected = (&'a isize, fn(&'a isize))
        //
        // but not that the Debruijn index is different in the different cases.
        let t_expected = {
            let t_rptr_bound2 = env.t_rptr_late_bound_with_debruijn(1, ty::DebruijnIndex::new(2));
            env.t_pair(t_rptr_bound1, env.t_fn(&[t_rptr_bound2], env.t_nil()))
        };

        debug!("subst_bound: t_source={:?} substs={:?} t_substituted={:?} t_expected={:?}",
               t_source,
               substs,
               t_substituted,
               t_expected);

        assert_eq!(t_substituted, t_expected);
    })
}

/// Test that we correctly compute whether a type has escaping regions or not.
#[test]
fn escaping() {

    test_env(EMPTY_SOURCE_STR, errors(&[]), |mut env| {
        // Situation:
        // Theta = [A -> &'a foo]
        env.create_simple_region_hierarchy();

        assert!(!env.t_nil().has_escaping_regions());

        let t_rptr_free1 = env.t_rptr_free(1);
        assert!(!t_rptr_free1.has_escaping_regions());

        let t_rptr_bound1 = env.t_rptr_late_bound_with_debruijn(1, ty::DebruijnIndex::new(1));
        assert!(t_rptr_bound1.has_escaping_regions());

        let t_rptr_bound2 = env.t_rptr_late_bound_with_debruijn(1, ty::DebruijnIndex::new(2));
        assert!(t_rptr_bound2.has_escaping_regions());

        // t_fn = fn(A)
        let t_param = env.t_param(0);
        assert!(!t_param.has_escaping_regions());
        let t_fn = env.t_fn(&[t_param], env.t_nil());
        assert!(!t_fn.has_escaping_regions());
    })
}

/// Test applying a substitution where the value being substituted for an early-bound region is a
/// late-bound region.
#[test]
fn subst_region_renumber_region() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |env| {
        let re_bound1 = env.re_late_bound_with_debruijn(1, ty::DebruijnIndex::new(1));

        // type t_source<'a> = fn(&'a isize)
        let t_source = {
            let re_early = env.re_early_bound(0, "'a");
            env.t_fn(&[env.t_rptr(re_early)], env.t_nil())
        };

        let substs = env.infcx.tcx.intern_substs(&[Kind::from(re_bound1)]);
        let t_substituted = t_source.subst(env.infcx.tcx, substs);

        // t_expected = fn(&'a isize)
        //
        // but not that the Debruijn index is different in the different cases.
        let t_expected = {
            let t_rptr_bound2 = env.t_rptr_late_bound_with_debruijn(1, ty::DebruijnIndex::new(2));
            env.t_fn(&[t_rptr_bound2], env.t_nil())
        };

        debug!("subst_bound: t_source={:?} substs={:?} t_substituted={:?} t_expected={:?}",
               t_source,
               substs,
               t_substituted,
               t_expected);

        assert_eq!(t_substituted, t_expected);
    })
}

#[test]
fn walk_ty() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |env| {
        let tcx = env.infcx.tcx;
        let int_ty = tcx.types.isize;
        let usize_ty = tcx.types.usize;
        let tup1_ty = tcx.intern_tup(&[int_ty, usize_ty, int_ty, usize_ty], false);
        let tup2_ty = tcx.intern_tup(&[tup1_ty, tup1_ty, usize_ty], false);
        let walked: Vec<_> = tup2_ty.walk().collect();
        assert_eq!(walked,
                   [tup2_ty, tup1_ty, int_ty, usize_ty, int_ty, usize_ty, tup1_ty, int_ty,
                    usize_ty, int_ty, usize_ty, usize_ty]);
    })
}

#[test]
fn walk_ty_skip_subtree() {
    test_env(EMPTY_SOURCE_STR, errors(&[]), |env| {
        let tcx = env.infcx.tcx;
        let int_ty = tcx.types.isize;
        let usize_ty = tcx.types.usize;
        let tup1_ty = tcx.intern_tup(&[int_ty, usize_ty, int_ty, usize_ty], false);
        let tup2_ty = tcx.intern_tup(&[tup1_ty, tup1_ty, usize_ty], false);

        // types we expect to see (in order), plus a boolean saying
        // whether to skip the subtree.
        let mut expected = vec![(tup2_ty, false),
                                (tup1_ty, false),
                                (int_ty, false),
                                (usize_ty, false),
                                (int_ty, false),
                                (usize_ty, false),
                                (tup1_ty, true), // skip the isize/usize/isize/usize
                                (usize_ty, false)];
        expected.reverse();

        let mut walker = tup2_ty.walk();
        while let Some(t) = walker.next() {
            debug!("walked to {:?}", t);
            let (expected_ty, skip) = expected.pop().unwrap();
            assert_eq!(t, expected_ty);
            if skip {
                walker.skip_current_subtree();
            }
        }

        assert!(expected.is_empty());
    })
}
