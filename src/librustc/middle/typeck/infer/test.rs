// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# Standalone Tests for the Inference Module

*/

// This is only used by tests, hence allow dead code.
#![allow(dead_code)]

use driver::config;
use driver::diagnostic;
use driver::diagnostic::Emitter;
use driver::driver;
use driver::session;
use middle::freevars;
use middle::lang_items;
use middle::region;
use middle::resolve;
use middle::resolve_lifetime;
use middle::stability;
use middle::ty;
use middle::typeck::infer::combine::Combine;
use middle::typeck::infer;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::glb::Glb;
use syntax::codemap;
use syntax::codemap::{Span, CodeMap, DUMMY_SP};
use syntax::diagnostic::{Level, RenderSpan, Bug, Fatal, Error, Warning, Note};
use syntax::ast;
use util::ppaux::{ty_to_string, UserString};

struct Env<'a> {
    krate: ast::Crate,
    tcx: &'a ty::ctxt,
    infcx: &'a infer::InferCtxt<'a>,
}

struct RH<'a> {
    id: ast::NodeId,
    sub: &'a [RH<'a>]
}

static EMPTY_SOURCE_STR: &'static str = "#![no_std]";

struct ExpectErrorEmitter {
    messages: Vec<String>
}

fn remove_message(e: &mut ExpectErrorEmitter, msg: &str, lvl: Level) {
    match lvl {
        Bug | Fatal | Error => { }
        Warning | Note => { return; }
    }

    debug!("Error: {}", msg);
    match e.messages.iter().position(|m| msg.contains(m.as_slice())) {
        Some(i) => {
            e.messages.remove(i);
        }
        None => {
            fail!("Unexpected error: {} Expected: {}",
                  msg, e.messages);
        }
    }
}

impl Emitter for ExpectErrorEmitter {
    fn emit(&mut self,
            _cmsp: Option<(&codemap::CodeMap, Span)>,
            msg: &str,
            _: Option<&str>,
            lvl: Level)
    {
        remove_message(self, msg, lvl);
    }

    fn custom_emit(&mut self,
                   _cm: &codemap::CodeMap,
                   _sp: RenderSpan,
                   msg: &str,
                   lvl: Level)
    {
        remove_message(self, msg, lvl);
    }
}

fn errors(msgs: &[&str]) -> (Box<Emitter+Send>, uint) {
    let v = Vec::from_fn(msgs.len(), |i| msgs[i].to_owned());
    (box ExpectErrorEmitter { messages: v } as Box<Emitter+Send>, msgs.len())
}

fn test_env(_test_name: &str,
            source_string: &str,
            (emitter, expected_err_count): (Box<Emitter+Send>, uint),
            body: |Env|) {
    let options =
        config::basic_options();
    let codemap =
        CodeMap::new();
    let diagnostic_handler =
        diagnostic::mk_handler(emitter);
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, codemap);

    let sess = session::build_session_(options, None, span_diagnostic_handler);
    let krate_config = Vec::new();
    let input = driver::StrInput(source_string.to_owned());
    let krate = driver::phase_1_parse_input(&sess, krate_config, &input);
    let (krate, ast_map) =
        driver::phase_2_configure_and_expand(&sess, krate, "test")
            .expect("phase 2 aborted");

    // run just enough stuff to build a tcx:
    let lang_items = lang_items::collect_language_items(&krate, &sess);
    let resolve::CrateMap { def_map: def_map, .. } =
        resolve::resolve_crate(&sess, &lang_items, &krate);
    let freevars_map = freevars::annotate_freevars(&def_map, &krate);
    let named_region_map = resolve_lifetime::krate(&sess, &krate);
    let region_map = region::resolve_crate(&sess, &krate);
    let stability_index = stability::Index::build(&krate);
    let tcx = ty::mk_ctxt(sess, def_map, named_region_map, ast_map,
                          freevars_map, region_map, lang_items, stability_index);
    let infcx = infer::new_infer_ctxt(&tcx);
    let env = Env {krate: krate,
                   tcx: &tcx,
                   infcx: &infcx};
    body(env);
    infcx.resolve_regions_and_report_errors();
    assert_eq!(tcx.sess.err_count(), expected_err_count);
}

impl<'a> Env<'a> {
    pub fn create_region_hierarchy(&self, rh: &RH) {
        for child_rh in rh.sub.iter() {
            self.create_region_hierarchy(child_rh);
            self.tcx.region_maps.record_encl_scope(child_rh.id, rh.id);
        }
    }

    pub fn create_simple_region_hierarchy(&self) {
        // creates a region hierarchy where 1 is root, 10 and 11 are
        // children of 1, etc
        self.create_region_hierarchy(
            &RH {id: 1,
                 sub: &[RH {id: 10,
                            sub: &[]},
                        RH {id: 11,
                            sub: &[]}]});
    }

    pub fn lookup_item(&self, names: &[String]) -> ast::NodeId {
        return match search_mod(self, &self.krate.module, 0, names) {
            Some(id) => id,
            None => {
                fail!("no item found: `{}`", names.connect("::"));
            }
        };

        fn search_mod(this: &Env,
                      m: &ast::Mod,
                      idx: uint,
                      names: &[String])
                      -> Option<ast::NodeId> {
            assert!(idx < names.len());
            for item in m.items.iter() {
                if item.ident.user_string(this.tcx) == names[idx] {
                    return search(this, &**item, idx+1, names);
                }
            }
            return None;
        }

        fn search(this: &Env,
                  it: &ast::Item,
                  idx: uint,
                  names: &[String])
                  -> Option<ast::NodeId> {
            if idx == names.len() {
                return Some(it.id);
            }

            return match it.node {
                ast::ItemStatic(..) | ast::ItemFn(..) |
                ast::ItemForeignMod(..) | ast::ItemTy(..) => {
                    None
                }

                ast::ItemEnum(..) | ast::ItemStruct(..) |
                ast::ItemTrait(..) | ast::ItemImpl(..) |
                ast::ItemMac(..) => {
                    None
                }

                ast::ItemMod(ref m) => {
                    search_mod(this, m, idx, names)
                }
            };
        }
    }

    pub fn make_subtype(&self, a: ty::t, b: ty::t) -> bool {
        match infer::mk_subty(self.infcx, true, infer::Misc(DUMMY_SP), a, b) {
            Ok(_) => true,
            Err(ref e) => fail!("Encountered error: {}",
                                ty::type_err_to_str(self.tcx, e))
        }
    }

    pub fn is_subtype(&self, a: ty::t, b: ty::t) -> bool {
        match infer::can_mk_subty(self.infcx, a, b) {
            Ok(_) => true,
            Err(_) => false
        }
    }

    pub fn assert_subtype(&self, a: ty::t, b: ty::t) {
        if !self.is_subtype(a, b) {
            fail!("{} is not a subtype of {}, but it should be",
                  self.ty_to_string(a),
                  self.ty_to_string(b));
        }
    }

    pub fn assert_not_subtype(&self, a: ty::t, b: ty::t) {
        if self.is_subtype(a, b) {
            fail!("{} is a subtype of {}, but it shouldn't be",
                  self.ty_to_string(a),
                  self.ty_to_string(b));
        }
    }

    pub fn assert_eq(&self, a: ty::t, b: ty::t) {
        self.assert_subtype(a, b);
        self.assert_subtype(b, a);
    }

    pub fn ty_to_string(&self, a: ty::t) -> String {
        ty_to_string(self.tcx, a)
    }

    pub fn t_fn(&self,
                binder_id: ast::NodeId,
                input_tys: &[ty::t],
                output_ty: ty::t)
                -> ty::t
    {
        ty::mk_ctor_fn(self.tcx, binder_id, input_tys, output_ty)
    }

    pub fn t_int(&self) -> ty::t {
        ty::mk_int()
    }

    pub fn t_rptr_late_bound(&self, binder_id: ast::NodeId, id: uint) -> ty::t {
        ty::mk_imm_rptr(self.tcx, ty::ReLateBound(binder_id, ty::BrAnon(id)),
                        self.t_int())
    }

    pub fn t_rptr_scope(&self, id: ast::NodeId) -> ty::t {
        ty::mk_imm_rptr(self.tcx, ty::ReScope(id), self.t_int())
    }

    pub fn t_rptr_free(&self, nid: ast::NodeId, id: uint) -> ty::t {
        ty::mk_imm_rptr(self.tcx,
                        ty::ReFree(ty::FreeRegion {scope_id: nid,
                                                    bound_region: ty::BrAnon(id)}),
                        self.t_int())
    }

    pub fn t_rptr_static(&self) -> ty::t {
        ty::mk_imm_rptr(self.tcx, ty::ReStatic, self.t_int())
    }

    pub fn dummy_type_trace(&self) -> infer::TypeTrace {
        infer::TypeTrace {
            origin: infer::Misc(DUMMY_SP),
            values: infer::Types(ty::expected_found {
                expected: ty::mk_err(),
                found: ty::mk_err(),
            })
        }
    }

    pub fn lub(&self) -> Lub<'a> {
        let trace = self.dummy_type_trace();
        Lub(self.infcx.combine_fields(true, trace))
    }

    pub fn glb(&self) -> Glb<'a> {
        let trace = self.dummy_type_trace();
        Glb(self.infcx.combine_fields(true, trace))
    }

    pub fn resolve_regions(&self) {
        self.infcx.resolve_regions_and_report_errors();
    }

    pub fn make_lub_ty(&self, t1: ty::t, t2: ty::t) -> ty::t {
        match self.lub().tys(t1, t2) {
            Ok(t) => t,
            Err(ref e) => fail!("unexpected error computing LUB: {:?}",
                                ty::type_err_to_str(self.tcx, e))
        }
    }

    /// Checks that `LUB(t1,t2) == t_lub`
    pub fn check_lub(&self, t1: ty::t, t2: ty::t, t_lub: ty::t) {
        match self.lub().tys(t1, t2) {
            Ok(t) => {
                self.assert_eq(t, t_lub);
            }
            Err(ref e) => {
                fail!("unexpected error in LUB: {}",
                      ty::type_err_to_str(self.tcx, e))
            }
        }
    }

    /// Checks that `GLB(t1,t2) == t_glb`
    pub fn check_glb(&self, t1: ty::t, t2: ty::t, t_glb: ty::t) {
        debug!("check_glb(t1={}, t2={}, t_glb={})",
               self.ty_to_string(t1),
               self.ty_to_string(t2),
               self.ty_to_string(t_glb));
        match self.glb().tys(t1, t2) {
            Err(e) => {
                fail!("unexpected error computing LUB: {:?}", e)
            }
            Ok(t) => {
                self.assert_eq(t, t_glb);

                // sanity check for good measure:
                self.assert_subtype(t, t1);
                self.assert_subtype(t, t2);
            }
        }
    }

    /// Checks that `LUB(t1,t2)` is undefined
    pub fn check_no_lub(&self, t1: ty::t, t2: ty::t) {
        match self.lub().tys(t1, t2) {
            Err(_) => {}
            Ok(t) => {
                fail!("unexpected success computing LUB: {}", self.ty_to_string(t))
            }
        }
    }

    /// Checks that `GLB(t1,t2)` is undefined
    pub fn check_no_glb(&self, t1: ty::t, t2: ty::t) {
        match self.glb().tys(t1, t2) {
            Err(_) => {}
            Ok(t) => {
                fail!("unexpected success computing GLB: {}", self.ty_to_string(t))
            }
        }
    }
}

#[test]
fn contravariant_region_ptr_ok() {
    test_env("contravariant_region_ptr", EMPTY_SOURCE_STR, errors([]), |env| {
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
    test_env("contravariant_region_ptr",
             EMPTY_SOURCE_STR,
             errors(["lifetime mismatch"]),
             |env| {
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
fn lub_bound_bound() {
    test_env("contravariant_region_ptr", EMPTY_SOURCE_STR, errors([]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(22, 1);
        let t_rptr_bound2 = env.t_rptr_late_bound(22, 2);
        env.check_lub(env.t_fn(22, [t_rptr_bound1], env.t_int()),
                      env.t_fn(22, [t_rptr_bound2], env.t_int()),
                      env.t_fn(22, [t_rptr_bound1], env.t_int()));
    })
}

#[test]
fn lub_bound_free() {
    test_env("contravariant_region_ptr", EMPTY_SOURCE_STR, errors([]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(22, 1);
        let t_rptr_free1 = env.t_rptr_free(0, 1);
        env.check_lub(env.t_fn(22, [t_rptr_bound1], env.t_int()),
                      env.t_fn(22, [t_rptr_free1], env.t_int()),
                      env.t_fn(22, [t_rptr_free1], env.t_int()));
    })
}

#[test]
fn lub_bound_static() {
    test_env("contravariant_region_ptr", EMPTY_SOURCE_STR, errors([]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(22, 1);
        let t_rptr_static = env.t_rptr_static();
        env.check_lub(env.t_fn(22, [t_rptr_bound1], env.t_int()),
                      env.t_fn(22, [t_rptr_static], env.t_int()),
                      env.t_fn(22, [t_rptr_static], env.t_int()));
    })
}

#[test]
fn lub_bound_bound_inverse_order() {
    test_env("contravariant_region_ptr", EMPTY_SOURCE_STR, errors([]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(22, 1);
        let t_rptr_bound2 = env.t_rptr_late_bound(22, 2);
        env.check_lub(env.t_fn(22, [t_rptr_bound1, t_rptr_bound2], t_rptr_bound1),
                      env.t_fn(22, [t_rptr_bound2, t_rptr_bound1], t_rptr_bound1),
                      env.t_fn(22, [t_rptr_bound1, t_rptr_bound1], t_rptr_bound1));
    })
}

#[test]
fn lub_free_free() {
    test_env("contravariant_region_ptr", EMPTY_SOURCE_STR, errors([]), |env| {
        let t_rptr_free1 = env.t_rptr_free(0, 1);
        let t_rptr_free2 = env.t_rptr_free(0, 2);
        let t_rptr_static = env.t_rptr_static();
        env.check_lub(env.t_fn(22, [t_rptr_free1], env.t_int()),
                      env.t_fn(22, [t_rptr_free2], env.t_int()),
                      env.t_fn(22, [t_rptr_static], env.t_int()));
    })
}

#[test]
fn lub_returning_scope() {
    test_env("contravariant_region_ptr", EMPTY_SOURCE_STR,
             errors(["cannot infer an appropriate lifetime"]), |env| {
                 let t_rptr_scope10 = env.t_rptr_scope(10);
                 let t_rptr_scope11 = env.t_rptr_scope(11);

                 // this should generate an error when regions are resolved
                 env.make_lub_ty(env.t_fn(22, [], t_rptr_scope10),
                                 env.t_fn(22, [], t_rptr_scope11));
             })
}

#[test]
fn glb_free_free_with_common_scope() {
    test_env("contravariant_region_ptr", EMPTY_SOURCE_STR, errors([]), |env| {
        let t_rptr_free1 = env.t_rptr_free(0, 1);
        let t_rptr_free2 = env.t_rptr_free(0, 2);
        let t_rptr_scope = env.t_rptr_scope(0);
        env.check_glb(env.t_fn(22, [t_rptr_free1], env.t_int()),
                      env.t_fn(22, [t_rptr_free2], env.t_int()),
                      env.t_fn(22, [t_rptr_scope], env.t_int()));
    })
}

#[test]
fn glb_bound_bound() {
    test_env("contravariant_region_ptr", EMPTY_SOURCE_STR, errors([]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(22, 1);
        let t_rptr_bound2 = env.t_rptr_late_bound(22, 2);
        env.check_glb(env.t_fn(22, [t_rptr_bound1], env.t_int()),
                      env.t_fn(22, [t_rptr_bound2], env.t_int()),
                      env.t_fn(22, [t_rptr_bound1], env.t_int()));
    })
}

#[test]
fn glb_bound_free() {
    test_env("contravariant_region_ptr", EMPTY_SOURCE_STR, errors([]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(22, 1);
        let t_rptr_free1 = env.t_rptr_free(0, 1);
        env.check_glb(env.t_fn(22, [t_rptr_bound1], env.t_int()),
                      env.t_fn(22, [t_rptr_free1], env.t_int()),
                      env.t_fn(22, [t_rptr_bound1], env.t_int()));
    })
}

#[test]
fn glb_bound_static() {
    test_env("contravariant_region_ptr", EMPTY_SOURCE_STR, errors([]), |env| {
        let t_rptr_bound1 = env.t_rptr_late_bound(22, 1);
        let t_rptr_static = env.t_rptr_static();
        env.check_glb(env.t_fn(22, [t_rptr_bound1], env.t_int()),
                      env.t_fn(22, [t_rptr_static], env.t_int()),
                      env.t_fn(22, [t_rptr_bound1], env.t_int()));
    })
}
