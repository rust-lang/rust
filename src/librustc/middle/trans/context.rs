// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::session;
use lib::llvm::{ContextRef, ModuleRef, ValueRef};
use lib::llvm::{llvm, TargetData, TypeNames};
use lib::llvm::mk_target_data;
use metadata::common::LinkMeta;
use middle::astencode;
use middle::resolve;
use middle::trans::adt;
use middle::trans::base;
use middle::trans::builder::Builder;
use middle::trans::debuginfo;
use middle::trans::common::{C_i32, C_null};
use middle::ty;

use middle::trans::type_::Type;

use util::sha2::Sha256;

use std::cell::{Cell, RefCell};
use std::c_str::ToCStr;
use std::hashmap::{HashMap, HashSet};
use std::local_data;
use std::libc::c_uint;
use syntax::ast;

use middle::trans::common::{mono_id,ExternMap,tydesc_info,BuilderRef_res,Stats};

use middle::trans::base::{decl_crate_map};

pub struct CrateContext {
     sess: session::Session,
     llmod: ModuleRef,
     llcx: ContextRef,
     metadata_llmod: ModuleRef,
     td: TargetData,
     tn: TypeNames,
     externs: RefCell<ExternMap>,
     intrinsics: HashMap<&'static str, ValueRef>,
     item_vals: RefCell<HashMap<ast::NodeId, ValueRef>>,
     exp_map2: resolve::ExportMap2,
     reachable: @RefCell<HashSet<ast::NodeId>>,
     item_symbols: RefCell<HashMap<ast::NodeId, ~str>>,
     link_meta: LinkMeta,
     tydescs: RefCell<HashMap<ty::t, @tydesc_info>>,
     // Set when running emit_tydescs to enforce that no more tydescs are
     // created.
     finished_tydescs: Cell<bool>,
     // Track mapping of external ids to local items imported for inlining
     external: RefCell<HashMap<ast::DefId, Option<ast::NodeId>>>,
     // Backwards version of the `external` map (inlined items to where they
     // came from)
     external_srcs: RefCell<HashMap<ast::NodeId, ast::DefId>>,
     // A set of static items which cannot be inlined into other crates. This
     // will pevent in IIItem() structures from being encoded into the metadata
     // that is generated
     non_inlineable_statics: RefCell<HashSet<ast::NodeId>>,
     // Cache instances of monomorphized functions
     monomorphized: RefCell<HashMap<mono_id, ValueRef>>,
     monomorphizing: RefCell<HashMap<ast::DefId, uint>>,
     // Cache generated vtables
     vtables: RefCell<HashMap<(ty::t, mono_id), ValueRef>>,
     // Cache of constant strings,
     const_cstr_cache: RefCell<HashMap<@str, ValueRef>>,

     // Reverse-direction for const ptrs cast from globals.
     // Key is an int, cast from a ValueRef holding a *T,
     // Val is a ValueRef holding a *[T].
     //
     // Needed because LLVM loses pointer->pointee association
     // when we ptrcast, and we have to ptrcast during translation
     // of a [T] const because we form a slice, a [*T,int] pair, not
     // a pointer to an LLVM array type.
     const_globals: RefCell<HashMap<int, ValueRef>>,

     // Cache of emitted const values
     const_values: RefCell<HashMap<ast::NodeId, ValueRef>>,

     // Cache of external const values
     extern_const_values: RefCell<HashMap<ast::DefId, ValueRef>>,

     impl_method_cache: RefCell<HashMap<(ast::DefId, ast::Name), ast::DefId>>,

     module_data: RefCell<HashMap<~str, ValueRef>>,
     lltypes: RefCell<HashMap<ty::t, Type>>,
     llsizingtypes: RefCell<HashMap<ty::t, Type>>,
     adt_reprs: RefCell<HashMap<ty::t, @adt::Repr>>,
     symbol_hasher: RefCell<Sha256>,
     type_hashcodes: RefCell<HashMap<ty::t, @str>>,
     all_llvm_symbols: RefCell<HashSet<@str>>,
     tcx: ty::ctxt,
     maps: astencode::Maps,
     stats: @Stats,
     tydesc_type: Type,
     int_type: Type,
     opaque_vec_type: Type,
     builder: BuilderRef_res,
     crate_map: ValueRef,
     crate_map_name: ~str,
     // Set when at least one function uses GC. Needed so that
     // decl_gc_metadata knows whether to link to the module metadata, which
     // is not emitted by LLVM's GC pass when no functions use GC.
     uses_gc: bool,
     dbg_cx: Option<debuginfo::CrateDebugContext>,
     do_not_commit_warning_issued: Cell<bool>,
}

impl CrateContext {
    pub fn new(sess: session::Session,
               name: &str,
               tcx: ty::ctxt,
               emap2: resolve::ExportMap2,
               maps: astencode::Maps,
               symbol_hasher: Sha256,
               link_meta: LinkMeta,
               reachable: @RefCell<HashSet<ast::NodeId>>)
               -> CrateContext {
        unsafe {
            let llcx = llvm::LLVMContextCreate();
            set_task_llcx(llcx);
            let llmod = name.with_c_str(|buf| {
                llvm::LLVMModuleCreateWithNameInContext(buf, llcx)
            });
            let metadata_llmod = format!("{}_metadata", name).with_c_str(|buf| {
                llvm::LLVMModuleCreateWithNameInContext(buf, llcx)
            });
            let data_layout: &str = sess.targ_cfg.target_strs.data_layout;
            let targ_triple: &str = sess.targ_cfg.target_strs.target_triple;
            data_layout.with_c_str(|buf| {
                llvm::LLVMSetDataLayout(llmod, buf);
                llvm::LLVMSetDataLayout(metadata_llmod, buf);
            });
            targ_triple.with_c_str(|buf| {
                llvm::LLVMRustSetNormalizedTarget(llmod, buf);
                llvm::LLVMRustSetNormalizedTarget(metadata_llmod, buf);
            });
            let targ_cfg = sess.targ_cfg;

            let td = mk_target_data(sess.targ_cfg.target_strs.data_layout);
            let tn = TypeNames::new();

            let mut intrinsics = base::declare_intrinsics(llmod);
            if sess.opts.extra_debuginfo {
                base::declare_dbg_intrinsics(llmod, &mut intrinsics);
            }
            let int_type = Type::int(targ_cfg.arch);
            let tydesc_type = Type::tydesc(targ_cfg.arch);
            let opaque_vec_type = Type::opaque_vec(targ_cfg.arch);

            let mut str_slice_ty = Type::named_struct("str_slice");
            str_slice_ty.set_struct_body([Type::i8p(), int_type], false);

            tn.associate_type("tydesc", &tydesc_type);
            tn.associate_type("str_slice", &str_slice_ty);

            let (crate_map_name, crate_map) = decl_crate_map(sess, link_meta.clone(), llmod);
            let dbg_cx = if sess.opts.debuginfo {
                Some(debuginfo::CrateDebugContext::new(llmod, name.to_owned()))
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
                  metadata_llmod: metadata_llmod,
                  td: td,
                  tn: tn,
                  externs: RefCell::new(HashMap::new()),
                  intrinsics: intrinsics,
                  item_vals: RefCell::new(HashMap::new()),
                  exp_map2: emap2,
                  reachable: reachable,
                  item_symbols: RefCell::new(HashMap::new()),
                  link_meta: link_meta,
                  tydescs: RefCell::new(HashMap::new()),
                  finished_tydescs: Cell::new(false),
                  external: RefCell::new(HashMap::new()),
                  external_srcs: RefCell::new(HashMap::new()),
                  non_inlineable_statics: RefCell::new(HashSet::new()),
                  monomorphized: RefCell::new(HashMap::new()),
                  monomorphizing: RefCell::new(HashMap::new()),
                  vtables: RefCell::new(HashMap::new()),
                  const_cstr_cache: RefCell::new(HashMap::new()),
                  const_globals: RefCell::new(HashMap::new()),
                  const_values: RefCell::new(HashMap::new()),
                  extern_const_values: RefCell::new(HashMap::new()),
                  impl_method_cache: RefCell::new(HashMap::new()),
                  module_data: RefCell::new(HashMap::new()),
                  lltypes: RefCell::new(HashMap::new()),
                  llsizingtypes: RefCell::new(HashMap::new()),
                  adt_reprs: RefCell::new(HashMap::new()),
                  symbol_hasher: RefCell::new(symbol_hasher),
                  type_hashcodes: RefCell::new(HashMap::new()),
                  all_llvm_symbols: RefCell::new(HashSet::new()),
                  tcx: tcx,
                  maps: maps,
                  stats: @Stats {
                    n_static_tydescs: Cell::new(0u),
                    n_glues_created: Cell::new(0u),
                    n_null_glues: Cell::new(0u),
                    n_real_glues: Cell::new(0u),
                    n_fns: Cell::new(0u),
                    n_monos: Cell::new(0u),
                    n_inlines: Cell::new(0u),
                    n_closures: Cell::new(0u),
                    n_llvm_insns: Cell::new(0u),
                    llvm_insns: RefCell::new(HashMap::new()),
                    fn_stats: RefCell::new(~[]),
                  },
                  tydesc_type: tydesc_type,
                  int_type: int_type,
                  opaque_vec_type: opaque_vec_type,
                  builder: BuilderRef_res(llvm::LLVMCreateBuilderInContext(llcx)),
                  crate_map: crate_map,
                  crate_map_name: crate_map_name,
                  uses_gc: false,
                  dbg_cx: dbg_cx,
                  do_not_commit_warning_issued: Cell::new(false),
            }
        }
    }

    pub fn builder(@self) -> Builder {
        Builder::new(self)
    }

    pub fn const_inbounds_gepi(&self,
                               pointer: ValueRef,
                               indices: &[uint]) -> ValueRef {
        debug!("const_inbounds_gepi: pointer={} indices={:?}",
               self.tn.val_to_str(pointer), indices);
        let v: ~[ValueRef] =
            indices.iter().map(|i| C_i32(*i as i32)).collect();
        unsafe {
            llvm::LLVMConstInBoundsGEP(pointer,
                                       v.as_ptr(),
                                       indices.len() as c_uint)
        }
    }

    pub fn offsetof_gep(&self,
                        llptr_ty: Type,
                        indices: &[uint]) -> ValueRef {
        /*!
         * Returns the offset of applying the given GEP indices
         * to an instance of `llptr_ty`. Similar to `offsetof` in C,
         * except that `llptr_ty` must be a pointer type.
         */

        unsafe {
            let null = C_null(llptr_ty);
            llvm::LLVMConstPtrToInt(self.const_inbounds_gepi(null, indices),
                                    self.int_type.to_ref())
        }
    }
}

#[unsafe_destructor]
impl Drop for CrateContext {
    fn drop(&mut self) {
        unset_task_llcx();
    }
}

local_data_key!(task_local_llcx_key: @ContextRef)

pub fn task_llcx() -> ContextRef {
    let opt = local_data::get(task_local_llcx_key, |k| k.map(|k| *k));
    *opt.expect("task-local LLVMContextRef wasn't ever set!")
}

fn set_task_llcx(c: ContextRef) {
    local_data::set(task_local_llcx_key, @c);
}

fn unset_task_llcx() {
    local_data::pop(task_local_llcx_key);
}
