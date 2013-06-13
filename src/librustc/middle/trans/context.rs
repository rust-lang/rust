use core::prelude::*;

use back::{upcall};
use driver::session;
use lib::llvm::{ContextRef, ModuleRef, ValueRef, TypeRef};
use lib::llvm::{llvm, TargetData, TypeNames};
use lib::llvm::{mk_target_data, mk_type_names};
use lib;
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

use core::hash;
use core::hashmap::{HashMap, HashSet};
use core::str;
use core::local_data;
use syntax::ast;

use middle::trans::common::{ExternMap,tydesc_info,BuilderRef_res,Stats,namegen,addrspace_gen};
use middle::trans::common::{mono_id,T_int,T_float,T_tydesc,T_opaque_vec};
use middle::trans::common::{new_namegen,new_addrspace_gen};

use middle::trans::base::{decl_crate_map};

use middle::trans::shape::{mk_ctxt};

pub struct CrateContext {
     sess: session::Session,
     llmod: ModuleRef,
     llcx: ContextRef,
     td: TargetData,
     tn: @TypeNames,
     externs: ExternMap,
     intrinsics: HashMap<&'static str, ValueRef>,
     item_vals: @mut HashMap<ast::node_id, ValueRef>,
     exp_map2: resolve::ExportMap2,
     reachable: reachable::map,
     item_symbols: @mut HashMap<ast::node_id, ~str>,
     link_meta: LinkMeta,
     enum_sizes: @mut HashMap<ty::t, uint>,
     discrims: @mut HashMap<ast::def_id, ValueRef>,
     discrim_symbols: @mut HashMap<ast::node_id, @str>,
     tydescs: @mut HashMap<ty::t, @mut tydesc_info>,
     // Set when running emit_tydescs to enforce that no more tydescs are
     // created.
     finished_tydescs: @mut bool,
     // Track mapping of external ids to local items imported for inlining
     external: @mut HashMap<ast::def_id, Option<ast::node_id>>,
     // Cache instances of monomorphized functions
     monomorphized: @mut HashMap<mono_id, ValueRef>,
     monomorphizing: @mut HashMap<ast::def_id, uint>,
     // Cache computed type parameter uses (see type_use.rs)
     type_use_cache: @mut HashMap<ast::def_id, @~[type_use::type_uses]>,
     // Cache generated vtables
     vtables: @mut HashMap<mono_id, ValueRef>,
     // Cache of constant strings,
     const_cstr_cache: @mut HashMap<@str, ValueRef>,

     // Reverse-direction for const ptrs cast from globals.
     // Key is an int, cast from a ValueRef holding a *T,
     // Val is a ValueRef holding a *[T].
     //
     // Needed because LLVM loses pointer->pointee association
     // when we ptrcast, and we have to ptrcast during translation
     // of a [T] const because we form a slice, a [*T,int] pair, not
     // a pointer to an LLVM array type.
     const_globals: @mut HashMap<int, ValueRef>,

     // Cache of emitted const values
     const_values: @mut HashMap<ast::node_id, ValueRef>,

     // Cache of external const values
     extern_const_values: @mut HashMap<ast::def_id, ValueRef>,

     module_data: @mut HashMap<~str, ValueRef>,
     lltypes: @mut HashMap<ty::t, TypeRef>,
     llsizingtypes: @mut HashMap<ty::t, TypeRef>,
     adt_reprs: @mut HashMap<ty::t, @adt::Repr>,
     names: namegen,
     next_addrspace: addrspace_gen,
     symbol_hasher: @mut hash::State,
     type_hashcodes: @mut HashMap<ty::t, @str>,
     type_short_names: @mut HashMap<ty::t, ~str>,
     all_llvm_symbols: @mut HashSet<@str>,
     tcx: ty::ctxt,
     maps: astencode::Maps,
     stats: @mut Stats,
     upcalls: @upcall::Upcalls,
     tydesc_type: TypeRef,
     int_type: TypeRef,
     float_type: TypeRef,
     opaque_vec_type: TypeRef,
     builder: BuilderRef_res,
     shape_cx: shape::Ctxt,
     crate_map: ValueRef,
     // Set when at least one function uses GC. Needed so that
     // decl_gc_metadata knows whether to link to the module metadata, which
     // is not emitted by LLVM's GC pass when no functions use GC.
     uses_gc: @mut bool,
     dbg_cx: Option<debuginfo::DebugContext>,
     do_not_commit_warning_issued: @mut bool
}

impl CrateContext {
    pub fn new(sess: session::Session, name: &str, tcx: ty::ctxt,
               emap2: resolve::ExportMap2, maps: astencode::Maps,
               symbol_hasher: @mut hash::State, link_meta: LinkMeta,
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
            let tn = mk_type_names();
            let mut intrinsics = base::declare_intrinsics(llmod);
            if sess.opts.extra_debuginfo {
                base::declare_dbg_intrinsics(llmod, &mut intrinsics);
            }
            let int_type = T_int(targ_cfg);
            let float_type = T_float(targ_cfg);
            let tydesc_type = T_tydesc(targ_cfg);
            lib::llvm::associate_type(tn, @"tydesc", tydesc_type);
            let crate_map = decl_crate_map(sess, link_meta, llmod);
            let dbg_cx = if sess.opts.debuginfo {
                Some(debuginfo::mk_ctxt(name.to_owned()))
            } else {
                None
            };

            CrateContext {
                  sess: sess,
                  llmod: llmod,
                  llcx: llcx,
                  td: td,
                  tn: tn,
                  externs: @mut HashMap::new(),
                  intrinsics: intrinsics,
                  item_vals: @mut HashMap::new(),
                  exp_map2: emap2,
                  reachable: reachable,
                  item_symbols: @mut HashMap::new(),
                  link_meta: link_meta,
                  enum_sizes: @mut HashMap::new(),
                  discrims: @mut HashMap::new(),
                  discrim_symbols: @mut HashMap::new(),
                  tydescs: @mut HashMap::new(),
                  finished_tydescs: @mut false,
                  external: @mut HashMap::new(),
                  monomorphized: @mut HashMap::new(),
                  monomorphizing: @mut HashMap::new(),
                  type_use_cache: @mut HashMap::new(),
                  vtables: @mut HashMap::new(),
                  const_cstr_cache: @mut HashMap::new(),
                  const_globals: @mut HashMap::new(),
                  const_values: @mut HashMap::new(),
                  extern_const_values: @mut HashMap::new(),
                  module_data: @mut HashMap::new(),
                  lltypes: @mut HashMap::new(),
                  llsizingtypes: @mut HashMap::new(),
                  adt_reprs: @mut HashMap::new(),
                  names: new_namegen(),
                  next_addrspace: new_addrspace_gen(),
                  symbol_hasher: symbol_hasher,
                  type_hashcodes: @mut HashMap::new(),
                  type_short_names: @mut HashMap::new(),
                  all_llvm_symbols: @mut HashSet::new(),
                  tcx: tcx,
                  maps: maps,
                  stats: @mut Stats {
                    n_static_tydescs: 0u,
                    n_glues_created: 0u,
                    n_null_glues: 0u,
                    n_real_glues: 0u,
                    n_fns: 0u,
                    n_monos: 0u,
                    n_inlines: 0u,
                    n_closures: 0u,
                    llvm_insn_ctxt: @mut ~[],
                    llvm_insns: @mut HashMap::new(),
                    fn_times: @mut ~[]
                  },
                  upcalls: upcall::declare_upcalls(targ_cfg, llmod),
                  tydesc_type: tydesc_type,
                  int_type: int_type,
                  float_type: float_type,
                  opaque_vec_type: T_opaque_vec(targ_cfg),
                  builder: BuilderRef_res(unsafe {
                      llvm::LLVMCreateBuilderInContext(llcx)
                  }),
                  shape_cx: mk_ctxt(llmod),
                  crate_map: crate_map,
                  uses_gc: @mut false,
                  dbg_cx: dbg_cx,
                  do_not_commit_warning_issued: @mut false
            }
        }
    }
}

#[unsafe_destructor]
impl Drop for CrateContext {
    fn finalize(&self) {
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

