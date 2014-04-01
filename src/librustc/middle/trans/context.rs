// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::session::NoDebugInfo;
use driver::session::Session;
use lib::llvm::{ContextRef, ModuleRef, ValueRef};
use lib::llvm::{llvm, TargetData, TypeNames};
use lib::llvm::mk_target_data;
use metadata::common::LinkMeta;
use middle::astencode;
use middle::resolve;
use middle::trans::adt;
use middle::trans::base;
use middle::trans::builder::Builder;
use middle::trans::common::{mono_id,ExternMap,tydesc_info,BuilderRef_res,Stats};
use middle::trans::debuginfo;
use middle::trans::type_::Type;
use middle::ty;
use util::sha2::Sha256;
use util::nodemap::{NodeMap, NodeSet, DefIdMap};

use std::cell::{Cell, RefCell};
use std::c_str::ToCStr;
use std::ptr;
use collections::{HashMap, HashSet};
use syntax::ast;
use syntax::parse::token::InternedString;

pub struct CrateContext {
    pub llmod: ModuleRef,
    pub llcx: ContextRef,
    pub metadata_llmod: ModuleRef,
    pub td: TargetData,
    pub tn: TypeNames,
    pub externs: RefCell<ExternMap>,
    pub intrinsics: HashMap<&'static str, ValueRef>,
    pub item_vals: RefCell<NodeMap<ValueRef>>,
    pub exp_map2: resolve::ExportMap2,
    pub reachable: NodeSet,
    pub item_symbols: RefCell<NodeMap<~str>>,
    pub link_meta: LinkMeta,
    pub drop_glues: RefCell<HashMap<ty::t, ValueRef>>,
    pub tydescs: RefCell<HashMap<ty::t, @tydesc_info>>,
    /// Set when running emit_tydescs to enforce that no more tydescs are
    /// created.
    pub finished_tydescs: Cell<bool>,
    /// Track mapping of external ids to local items imported for inlining
    pub external: RefCell<DefIdMap<Option<ast::NodeId>>>,
    /// Backwards version of the `external` map (inlined items to where they
    /// came from)
    pub external_srcs: RefCell<NodeMap<ast::DefId>>,
    /// A set of static items which cannot be inlined into other crates. This
    /// will pevent in IIItem() structures from being encoded into the metadata
    /// that is generated
    pub non_inlineable_statics: RefCell<NodeSet>,
    /// Cache instances of monomorphized functions
    pub monomorphized: RefCell<HashMap<mono_id, ValueRef>>,
    pub monomorphizing: RefCell<DefIdMap<uint>>,
    /// Cache generated vtables
    pub vtables: RefCell<HashMap<(ty::t, mono_id), ValueRef>>,
    /// Cache of constant strings,
    pub const_cstr_cache: RefCell<HashMap<InternedString, ValueRef>>,

    /// Reverse-direction for const ptrs cast from globals.
    /// Key is an int, cast from a ValueRef holding a *T,
    /// Val is a ValueRef holding a *[T].
    ///
    /// Needed because LLVM loses pointer->pointee association
    /// when we ptrcast, and we have to ptrcast during translation
    /// of a [T] const because we form a slice, a [*T,int] pair, not
    /// a pointer to an LLVM array type.
    pub const_globals: RefCell<HashMap<int, ValueRef>>,

    /// Cache of emitted const values
    pub const_values: RefCell<NodeMap<ValueRef>>,

    /// Cache of external const values
    pub extern_const_values: RefCell<DefIdMap<ValueRef>>,

    pub impl_method_cache: RefCell<HashMap<(ast::DefId, ast::Name), ast::DefId>>,

    /// Cache of closure wrappers for bare fn's.
    pub closure_bare_wrapper_cache: RefCell<HashMap<ValueRef, ValueRef>>,

    pub lltypes: RefCell<HashMap<ty::t, Type>>,
    pub llsizingtypes: RefCell<HashMap<ty::t, Type>>,
    pub adt_reprs: RefCell<HashMap<ty::t, @adt::Repr>>,
    pub symbol_hasher: RefCell<Sha256>,
    pub type_hashcodes: RefCell<HashMap<ty::t, ~str>>,
    pub all_llvm_symbols: RefCell<HashSet<~str>>,
    pub tcx: ty::ctxt,
    pub maps: astencode::Maps,
    pub stats: @Stats,
    pub int_type: Type,
    pub opaque_vec_type: Type,
    pub builder: BuilderRef_res,
    /// Set when at least one function uses GC. Needed so that
    /// decl_gc_metadata knows whether to link to the module metadata, which
    /// is not emitted by LLVM's GC pass when no functions use GC.
    pub uses_gc: bool,
    pub dbg_cx: Option<debuginfo::CrateDebugContext>,
}

impl CrateContext {
    pub fn new(name: &str,
               tcx: ty::ctxt,
               emap2: resolve::ExportMap2,
               maps: astencode::Maps,
               symbol_hasher: Sha256,
               link_meta: LinkMeta,
               reachable: NodeSet)
               -> CrateContext {
        unsafe {
            let llcx = llvm::LLVMContextCreate();
            let llmod = name.with_c_str(|buf| {
                llvm::LLVMModuleCreateWithNameInContext(buf, llcx)
            });
            let metadata_llmod = format!("{}_metadata", name).with_c_str(|buf| {
                llvm::LLVMModuleCreateWithNameInContext(buf, llcx)
            });
            tcx.sess.targ_cfg.target_strs.data_layout.with_c_str(|buf| {
                llvm::LLVMSetDataLayout(llmod, buf);
                llvm::LLVMSetDataLayout(metadata_llmod, buf);
            });
            tcx.sess.targ_cfg.target_strs.target_triple.with_c_str(|buf| {
                llvm::LLVMRustSetNormalizedTarget(llmod, buf);
                llvm::LLVMRustSetNormalizedTarget(metadata_llmod, buf);
            });

            let td = mk_target_data(tcx.sess.targ_cfg.target_strs.data_layout);

            let dbg_cx = if tcx.sess.opts.debuginfo != NoDebugInfo {
                Some(debuginfo::CrateDebugContext::new(llmod))
            } else {
                None
            };

            let mut ccx = CrateContext {
                llmod: llmod,
                llcx: llcx,
                metadata_llmod: metadata_llmod,
                td: td,
                tn: TypeNames::new(),
                externs: RefCell::new(HashMap::new()),
                intrinsics: HashMap::new(),
                item_vals: RefCell::new(NodeMap::new()),
                exp_map2: emap2,
                reachable: reachable,
                item_symbols: RefCell::new(NodeMap::new()),
                link_meta: link_meta,
                drop_glues: RefCell::new(HashMap::new()),
                tydescs: RefCell::new(HashMap::new()),
                finished_tydescs: Cell::new(false),
                external: RefCell::new(DefIdMap::new()),
                external_srcs: RefCell::new(NodeMap::new()),
                non_inlineable_statics: RefCell::new(NodeSet::new()),
                monomorphized: RefCell::new(HashMap::new()),
                monomorphizing: RefCell::new(DefIdMap::new()),
                vtables: RefCell::new(HashMap::new()),
                const_cstr_cache: RefCell::new(HashMap::new()),
                const_globals: RefCell::new(HashMap::new()),
                const_values: RefCell::new(NodeMap::new()),
                extern_const_values: RefCell::new(DefIdMap::new()),
                impl_method_cache: RefCell::new(HashMap::new()),
                closure_bare_wrapper_cache: RefCell::new(HashMap::new()),
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
                    fn_stats: RefCell::new(Vec::new()),
                },
                int_type: Type::from_ref(ptr::null()),
                opaque_vec_type: Type::from_ref(ptr::null()),
                builder: BuilderRef_res(llvm::LLVMCreateBuilderInContext(llcx)),
                uses_gc: false,
                dbg_cx: dbg_cx,
            };

            ccx.int_type = Type::int(&ccx);
            ccx.opaque_vec_type = Type::opaque_vec(&ccx);

            ccx.tn.associate_type("tydesc", &Type::tydesc(&ccx));

            let mut str_slice_ty = Type::named_struct(&ccx, "str_slice");
            str_slice_ty.set_struct_body([Type::i8p(&ccx), ccx.int_type], false);
            ccx.tn.associate_type("str_slice", &str_slice_ty);

            base::declare_intrinsics(&mut ccx);

            if ccx.sess().count_llvm_insns() {
                base::init_insn_ctxt()
            }

            ccx
        }
    }

    pub fn tcx<'a>(&'a self) -> &'a ty::ctxt {
        &self.tcx
    }

    pub fn sess<'a>(&'a self) -> &'a Session {
        &self.tcx.sess
    }

    pub fn builder<'a>(&'a self) -> Builder<'a> {
        Builder::new(self)
    }

    pub fn tydesc_type(&self) -> Type {
        self.tn.find_type("tydesc").unwrap()
    }
}
