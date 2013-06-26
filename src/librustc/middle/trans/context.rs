// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use back::{upcall};
use driver::session;
use lib::llvm::{ContextRef, ModuleRef, ValueRef};
use lib::llvm::{llvm, TargetData, TypeNames};
use lib::llvm::{mk_target_data};
use metadata::common::LinkMeta;
use middle::astencode;
use middle::resolve;
use middle::trans::adt;
use middle::trans::base;
use middle::trans::debuginfo;
use middle::trans::reachable;
use middle::trans::shape;
use middle::trans::type_use;
use middle::ty;

use middle::trans::type_::Type;

use core::hash;
use core::hashmap::{HashMap, HashSet};
use core::str;
use core::local_data;
use extra::time;
use syntax::ast;

use middle::trans::common::{ExternMap,tydesc_info,BuilderRef_res,Stats,namegen};
use middle::trans::common::{mono_id,new_namegen};

use middle::trans::base::{decl_crate_map};

use middle::trans::shape::{mk_ctxt};

pub struct CrateContext {
     sess: session::Session,
     llmod: ModuleRef,
     llcx: ContextRef,
     td: TargetData,
     tn: TypeNames,
     externs: ExternMap,
     intrinsics: HashMap<&'static str, ValueRef>,
     item_vals: HashMap<ast::node_id, ValueRef>,
     exp_map2: resolve::ExportMap2,
     reachable: reachable::map,
     item_symbols: HashMap<ast::node_id, ~str>,
     link_meta: LinkMeta,
     enum_sizes: HashMap<ty::t, uint>,
     discrims: HashMap<ast::def_id, ValueRef>,
     discrim_symbols: HashMap<ast::node_id, @str>,
     tydescs: HashMap<ty::t, @mut tydesc_info>,
     // Set when running emit_tydescs to enforce that no more tydescs are
     // created.
     finished_tydescs: bool,
     // Track mapping of external ids to local items imported for inlining
     external: HashMap<ast::def_id, Option<ast::node_id>>,
     // Cache instances of monomorphized functions
     monomorphized: HashMap<mono_id, ValueRef>,
     monomorphizing: HashMap<ast::def_id, uint>,
     // Cache computed type parameter uses (see type_use.rs)
     type_use_cache: HashMap<ast::def_id, @~[type_use::type_uses]>,
     // Cache generated vtables
     vtables: HashMap<mono_id, ValueRef>,
     // Cache of constant strings,
     const_cstr_cache: HashMap<@str, ValueRef>,

     // Reverse-direction for const ptrs cast from globals.
     // Key is an int, cast from a ValueRef holding a *T,
     // Val is a ValueRef holding a *[T].
     //
     // Needed because LLVM loses pointer->pointee association
     // when we ptrcast, and we have to ptrcast during translation
     // of a [T] const because we form a slice, a [*T,int] pair, not
     // a pointer to an LLVM array type.
     const_globals: HashMap<int, ValueRef>,

     // Cache of emitted const values
     const_values: HashMap<ast::node_id, ValueRef>,

     // Cache of external const values
     extern_const_values: HashMap<ast::def_id, ValueRef>,

     impl_method_cache: HashMap<(ast::def_id, ast::ident), ast::def_id>,

     module_data: HashMap<~str, ValueRef>,
     lltypes: HashMap<ty::t, Type>,
     llsizingtypes: HashMap<ty::t, Type>,
     adt_reprs: HashMap<ty::t, @adt::Repr>,
     names: namegen,
     symbol_hasher: hash::State,
     type_hashcodes: HashMap<ty::t, @str>,
     type_short_names: HashMap<ty::t, ~str>,
     all_llvm_symbols: HashSet<@str>,
     tcx: ty::ctxt,
     maps: astencode::Maps,
     stats: Stats,
     upcalls: @upcall::Upcalls,
     tydesc_type: Type,
     int_type: Type,
     float_type: Type,
     opaque_vec_type: Type,
     builder: BuilderRef_res,
     shape_cx: shape::Ctxt,
     crate_map: ValueRef,
     // Set when at least one function uses GC. Needed so that
     // decl_gc_metadata knows whether to link to the module metadata, which
     // is not emitted by LLVM's GC pass when no functions use GC.
     uses_gc: bool,
     dbg_cx: Option<debuginfo::DebugContext>,
     do_not_commit_warning_issued: bool
}

impl CrateContext {
    pub fn new(sess: session::Session, name: &str, tcx: ty::ctxt,
               emap2: resolve::ExportMap2, maps: astencode::Maps,
               symbol_hasher: hash::State, link_meta: LinkMeta,
               reachable: reachable::map) -> CrateContext {
        unsafe {
            let llcx = llvm::LLVMContextCreate();
            set_task_llcx(llcx);
            let llmod = str::as_c_str(name, |buf| {
                llvm::LLVMModuleCreateWithNameInContext(buf, llcx)
            });
            let data_layout: &str = sess.targ_cfg.target_strs.data_layout;
            let targ_triple: &str = sess.targ_cfg.target_strs.target_triple;
            str::as_c_str(data_layout, |buf| llvm::LLVMSetDataLayout(llmod, buf));
            str::as_c_str(targ_triple, |buf| llvm::LLVMSetTarget(llmod, buf));
            let targ_cfg = sess.targ_cfg;

            let td = mk_target_data(sess.targ_cfg.target_strs.data_layout);
            let mut tn = TypeNames::new();

            let mut intrinsics = base::declare_intrinsics(llmod);
            if sess.opts.extra_debuginfo {
                base::declare_dbg_intrinsics(llmod, &mut intrinsics);
            }
            let int_type = Type::int(targ_cfg.arch);
            let float_type = Type::float(targ_cfg.arch);
            let tydesc_type = Type::tydesc(targ_cfg.arch);
            let opaque_vec_type = Type::opaque_vec(targ_cfg.arch);

            let mut str_slice_ty = Type::named_struct("str_slice");
            str_slice_ty.set_struct_body([Type::i8p(), int_type], false);

            tn.associate_type("tydesc", &tydesc_type);
            tn.associate_type("str_slice", &str_slice_ty);

            let crate_map = decl_crate_map(sess, link_meta, llmod);
            let dbg_cx = if sess.opts.debuginfo {
                Some(debuginfo::DebugContext::new(llmod, name.to_owned()))
            } else {
                None
            };

            if sess.count_llvm_insns() {
                base::init_insn_ctxt()
            }

            CrateContext {
                  sess: sess,
                  llmod: llmod,
                  llcx: llcx,
                  td: td,
                  tn: tn,
                  externs: HashMap::new(),
                  intrinsics: intrinsics,
                  item_vals: HashMap::new(),
                  exp_map2: emap2,
                  reachable: reachable,
                  item_symbols: HashMap::new(),
                  link_meta: link_meta,
                  enum_sizes: HashMap::new(),
                  discrims: HashMap::new(),
                  discrim_symbols: HashMap::new(),
                  tydescs: HashMap::new(),
                  finished_tydescs: false,
                  external: HashMap::new(),
                  monomorphized: HashMap::new(),
                  monomorphizing: HashMap::new(),
                  type_use_cache: HashMap::new(),
                  vtables: HashMap::new(),
                  const_cstr_cache: HashMap::new(),
                  const_globals: HashMap::new(),
                  const_values: HashMap::new(),
                  extern_const_values: HashMap::new(),
                  impl_method_cache: HashMap::new(),
                  module_data: HashMap::new(),
                  lltypes: HashMap::new(),
                  llsizingtypes: HashMap::new(),
                  adt_reprs: HashMap::new(),
                  names: new_namegen(),
                  symbol_hasher: symbol_hasher,
                  type_hashcodes: HashMap::new(),
                  type_short_names: HashMap::new(),
                  all_llvm_symbols: HashSet::new(),
                  tcx: tcx,
                  maps: maps,
                  stats: Stats {
                    n_static_tydescs: 0u,
                    n_glues_created: 0u,
                    n_null_glues: 0u,
                    n_real_glues: 0u,
                    n_fns: 0u,
                    n_monos: 0u,
                    n_inlines: 0u,
                    n_closures: 0u,
                    llvm_insns: HashMap::new(),
                    fn_times: ~[]
                  },
                  upcalls: upcall::declare_upcalls(targ_cfg, llmod),
                  tydesc_type: tydesc_type,
                  int_type: int_type,
                  float_type: float_type,
                  opaque_vec_type: opaque_vec_type,
                  builder: BuilderRef_res(llvm::LLVMCreateBuilderInContext(llcx)),
                  shape_cx: mk_ctxt(llmod),
                  crate_map: crate_map,
                  uses_gc: false,
                  dbg_cx: dbg_cx,
                  do_not_commit_warning_issued: false
            }
        }
    }

    pub fn log_fn_time(&mut self, name: ~str, start: time::Timespec, end: time::Timespec) {
        let elapsed = 1000 * ((end.sec - start.sec) as int) +
            ((end.nsec as int) - (start.nsec as int)) / 1000000;
        self.stats.fn_times.push((name, elapsed));
    }
}

#[unsafe_destructor]
impl Drop for CrateContext {
    fn drop(&self) {
        unsafe {
            unset_task_llcx();
        }
    }
}

fn task_local_llcx_key(_v: @ContextRef) {}
pub fn task_llcx() -> ContextRef {
    let opt = unsafe { local_data::local_data_get(task_local_llcx_key) };
    *opt.expect("task-local LLVMContextRef wasn't ever set!")
}

unsafe fn set_task_llcx(c: ContextRef) {
    local_data::local_data_set(task_local_llcx_key, @c);
}

unsafe fn unset_task_llcx() {
    local_data::local_data_pop(task_local_llcx_key);
}
