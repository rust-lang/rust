// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::config::NoDebugInfo;
use driver::session::Session;
use lib::llvm::{ContextRef, ModuleRef, ValueRef};
use lib::llvm::{llvm, TargetData, TypeNames};
use lib::llvm::mk_target_data;
use metadata::common::LinkMeta;
use middle::resolve;
use middle::trans::adt;
use middle::trans::base;
use middle::trans::builder::Builder;
use middle::trans::common::{ExternMap,tydesc_info,BuilderRef_res};
use middle::trans::debuginfo;
use middle::trans::monomorphize::MonoId;
use middle::trans::type_::Type;
use middle::ty;
use util::sha2::Sha256;
use util::nodemap::{NodeMap, NodeSet, DefIdMap};

use std::cell::{Cell, RefCell};
use std::c_str::ToCStr;
use std::ptr;
use std::rc::Rc;
use collections::{HashMap, HashSet};
use syntax::ast;
use syntax::parse::token::InternedString;

pub struct Stats {
    pub n_static_tydescs: Cell<uint>,
    pub n_glues_created: Cell<uint>,
    pub n_null_glues: Cell<uint>,
    pub n_real_glues: Cell<uint>,
    pub n_fns: Cell<uint>,
    pub n_monos: Cell<uint>,
    pub n_inlines: Cell<uint>,
    pub n_closures: Cell<uint>,
    pub n_llvm_insns: Cell<uint>,
    pub llvm_insns: RefCell<HashMap<StrBuf, uint>>,
    // (ident, time-in-ms, llvm-instructions)
    pub fn_stats: RefCell<Vec<(StrBuf, uint, uint)> >,
}

pub struct CrateContext {
    pub llmod: ModuleRef,
    pub llcx: ContextRef,
    pub metadata_llmod: ModuleRef,
    pub td: TargetData,
    pub tn: TypeNames,
    pub externs: RefCell<ExternMap>,
    pub item_vals: RefCell<NodeMap<ValueRef>>,
    pub exp_map2: resolve::ExportMap2,
    pub reachable: NodeSet,
    pub item_symbols: RefCell<NodeMap<StrBuf>>,
    pub link_meta: LinkMeta,
    pub drop_glues: RefCell<HashMap<ty::t, ValueRef>>,
    pub tydescs: RefCell<HashMap<ty::t, Rc<tydesc_info>>>,
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
    pub monomorphized: RefCell<HashMap<MonoId, ValueRef>>,
    pub monomorphizing: RefCell<DefIdMap<uint>>,
    /// Cache generated vtables
    pub vtables: RefCell<HashMap<(ty::t, MonoId), ValueRef>>,
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
    pub adt_reprs: RefCell<HashMap<ty::t, Rc<adt::Repr>>>,
    pub symbol_hasher: RefCell<Sha256>,
    pub type_hashcodes: RefCell<HashMap<ty::t, StrBuf>>,
    pub all_llvm_symbols: RefCell<HashSet<StrBuf>>,
    pub tcx: ty::ctxt,
    pub stats: Stats,
    pub int_type: Type,
    pub opaque_vec_type: Type,
    pub builder: BuilderRef_res,
    /// Set when at least one function uses GC. Needed so that
    /// decl_gc_metadata knows whether to link to the module metadata, which
    /// is not emitted by LLVM's GC pass when no functions use GC.
    pub uses_gc: bool,
    pub dbg_cx: Option<debuginfo::CrateDebugContext>,

    intrinsics: RefCell<HashMap<&'static str, ValueRef>>,
}

impl CrateContext {
    pub fn new(name: &str,
               tcx: ty::ctxt,
               emap2: resolve::ExportMap2,
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
            tcx.sess
               .targ_cfg
               .target_strs
               .data_layout
               .as_slice()
               .with_c_str(|buf| {
                llvm::LLVMSetDataLayout(llmod, buf);
                llvm::LLVMSetDataLayout(metadata_llmod, buf);
            });
            tcx.sess
               .targ_cfg
               .target_strs
               .target_triple
               .as_slice()
               .with_c_str(|buf| {
                llvm::LLVMRustSetNormalizedTarget(llmod, buf);
                llvm::LLVMRustSetNormalizedTarget(metadata_llmod, buf);
            });

            let td = mk_target_data(tcx.sess
                                       .targ_cfg
                                       .target_strs
                                       .data_layout
                                       .as_slice());

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
                stats: Stats {
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
                intrinsics: RefCell::new(HashMap::new()),
            };

            ccx.int_type = Type::int(&ccx);
            ccx.opaque_vec_type = Type::opaque_vec(&ccx);

            ccx.tn.associate_type("tydesc", &Type::tydesc(&ccx));

            let mut str_slice_ty = Type::named_struct(&ccx, "str_slice");
            str_slice_ty.set_struct_body([Type::i8p(&ccx), ccx.int_type], false);
            ccx.tn.associate_type("str_slice", &str_slice_ty);

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

    pub fn get_intrinsic(&self, key: & &'static str) -> ValueRef {
        match self.intrinsics.borrow().find_copy(key) {
            Some(v) => return v,
            _ => {}
        }
        match declare_intrinsic(self, key) {
            Some(v) => return v,
            None => fail!()
        }
    }
}

fn declare_intrinsic(ccx: &CrateContext, key: & &'static str) -> Option<ValueRef> {
    macro_rules! ifn (
        ($name:expr fn() -> $ret:expr) => (
            if *key == $name {
                let f = base::decl_cdecl_fn(ccx.llmod, $name, Type::func([], &$ret), ty::mk_nil());
                ccx.intrinsics.borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        );
        ($name:expr fn($($arg:expr),*) -> $ret:expr) => (
            if *key == $name {
                let f = base::decl_cdecl_fn(ccx.llmod, $name,
                                  Type::func([$($arg),*], &$ret), ty::mk_nil());
                ccx.intrinsics.borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        )
    )
    macro_rules! mk_struct (
        ($($field_ty:expr),*) => (Type::struct_(ccx, [$($field_ty),*], false))
    )

    let i8p = Type::i8p(ccx);
    let void = Type::void(ccx);
    let i1 = Type::i1(ccx);
    let t_i8 = Type::i8(ccx);
    let t_i16 = Type::i16(ccx);
    let t_i32 = Type::i32(ccx);
    let t_i64 = Type::i64(ccx);
    let t_f32 = Type::f32(ccx);
    let t_f64 = Type::f64(ccx);

    ifn!("llvm.memcpy.p0i8.p0i8.i32" fn(i8p, i8p, t_i32, t_i32, i1) -> void);
    ifn!("llvm.memcpy.p0i8.p0i8.i64" fn(i8p, i8p, t_i64, t_i32, i1) -> void);
    ifn!("llvm.memmove.p0i8.p0i8.i32" fn(i8p, i8p, t_i32, t_i32, i1) -> void);
    ifn!("llvm.memmove.p0i8.p0i8.i64" fn(i8p, i8p, t_i64, t_i32, i1) -> void);
    ifn!("llvm.memset.p0i8.i32" fn(i8p, t_i8, t_i32, t_i32, i1) -> void);
    ifn!("llvm.memset.p0i8.i64" fn(i8p, t_i8, t_i64, t_i32, i1) -> void);

    ifn!("llvm.trap" fn() -> void);
    ifn!("llvm.debugtrap" fn() -> void);
    ifn!("llvm.frameaddress" fn(t_i32) -> i8p);

    ifn!("llvm.powi.f32" fn(t_f32, t_i32) -> t_f32);
    ifn!("llvm.powi.f64" fn(t_f64, t_i32) -> t_f64);
    ifn!("llvm.pow.f32" fn(t_f32, t_f32) -> t_f32);
    ifn!("llvm.pow.f64" fn(t_f64, t_f64) -> t_f64);

    ifn!("llvm.sqrt.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.sqrt.f64" fn(t_f64) -> t_f64);
    ifn!("llvm.sin.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.sin.f64" fn(t_f64) -> t_f64);
    ifn!("llvm.cos.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.cos.f64" fn(t_f64) -> t_f64);
    ifn!("llvm.exp.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.exp.f64" fn(t_f64) -> t_f64);
    ifn!("llvm.exp2.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.exp2.f64" fn(t_f64) -> t_f64);
    ifn!("llvm.log.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.log.f64" fn(t_f64) -> t_f64);
    ifn!("llvm.log10.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.log10.f64" fn(t_f64) -> t_f64);
    ifn!("llvm.log2.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.log2.f64" fn(t_f64) -> t_f64);

    ifn!("llvm.fma.f32" fn(t_f32, t_f32, t_f32) -> t_f32);
    ifn!("llvm.fma.f64" fn(t_f64, t_f64, t_f64) -> t_f64);

    ifn!("llvm.fabs.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.fabs.f64" fn(t_f64) -> t_f64);

    ifn!("llvm.floor.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.floor.f64" fn(t_f64) -> t_f64);
    ifn!("llvm.ceil.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.ceil.f64" fn(t_f64) -> t_f64);
    ifn!("llvm.trunc.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.trunc.f64" fn(t_f64) -> t_f64);

    ifn!("llvm.rint.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.rint.f64" fn(t_f64) -> t_f64);
    ifn!("llvm.nearbyint.f32" fn(t_f32) -> t_f32);
    ifn!("llvm.nearbyint.f64" fn(t_f64) -> t_f64);

    ifn!("llvm.ctpop.i8" fn(t_i8) -> t_i8);
    ifn!("llvm.ctpop.i16" fn(t_i16) -> t_i16);
    ifn!("llvm.ctpop.i32" fn(t_i32) -> t_i32);
    ifn!("llvm.ctpop.i64" fn(t_i64) -> t_i64);

    ifn!("llvm.ctlz.i8" fn(t_i8 , i1) -> t_i8);
    ifn!("llvm.ctlz.i16" fn(t_i16, i1) -> t_i16);
    ifn!("llvm.ctlz.i32" fn(t_i32, i1) -> t_i32);
    ifn!("llvm.ctlz.i64" fn(t_i64, i1) -> t_i64);

    ifn!("llvm.cttz.i8" fn(t_i8 , i1) -> t_i8);
    ifn!("llvm.cttz.i16" fn(t_i16, i1) -> t_i16);
    ifn!("llvm.cttz.i32" fn(t_i32, i1) -> t_i32);
    ifn!("llvm.cttz.i64" fn(t_i64, i1) -> t_i64);

    ifn!("llvm.bswap.i16" fn(t_i16) -> t_i16);
    ifn!("llvm.bswap.i32" fn(t_i32) -> t_i32);
    ifn!("llvm.bswap.i64" fn(t_i64) -> t_i64);

    ifn!("llvm.sadd.with.overflow.i8" fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.sadd.with.overflow.i16" fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.sadd.with.overflow.i32" fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.sadd.with.overflow.i64" fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});

    ifn!("llvm.uadd.with.overflow.i8" fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.uadd.with.overflow.i16" fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.uadd.with.overflow.i32" fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.uadd.with.overflow.i64" fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});

    ifn!("llvm.ssub.with.overflow.i8" fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.ssub.with.overflow.i16" fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.ssub.with.overflow.i32" fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.ssub.with.overflow.i64" fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});

    ifn!("llvm.usub.with.overflow.i8" fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.usub.with.overflow.i16" fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.usub.with.overflow.i32" fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.usub.with.overflow.i64" fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});

    ifn!("llvm.smul.with.overflow.i8" fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.smul.with.overflow.i16" fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.smul.with.overflow.i32" fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.smul.with.overflow.i64" fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});

    ifn!("llvm.umul.with.overflow.i8" fn(t_i8, t_i8) -> mk_struct!{t_i8, i1});
    ifn!("llvm.umul.with.overflow.i16" fn(t_i16, t_i16) -> mk_struct!{t_i16, i1});
    ifn!("llvm.umul.with.overflow.i32" fn(t_i32, t_i32) -> mk_struct!{t_i32, i1});
    ifn!("llvm.umul.with.overflow.i64" fn(t_i64, t_i64) -> mk_struct!{t_i64, i1});

    ifn!("llvm.expect.i1" fn(i1, i1) -> i1);

    // Some intrinsics were introduced in later versions of LLVM, but they have
    // fallbacks in libc or libm and such. Currently, all of these intrinsics
    // were introduced in LLVM 3.4, so we case on that.
    macro_rules! compatible_ifn (
        ($name:expr, $cname:ident ($($arg:expr),*) -> $ret:expr) => (
            if unsafe { llvm::LLVMVersionMinor() >= 4 } {
                // The `if key == $name` is already in ifn!
                ifn!($name fn($($arg),*) -> $ret);
            } else if *key == $name {
                let f = base::decl_cdecl_fn(ccx.llmod, stringify!($cname),
                                      Type::func([$($arg),*], &$ret),
                                      ty::mk_nil());
                ccx.intrinsics.borrow_mut().insert($name, f.clone());
                return Some(f);
            }
        )
    )

    compatible_ifn!("llvm.copysign.f32", copysignf(t_f32, t_f32) -> t_f32);
    compatible_ifn!("llvm.copysign.f64", copysign(t_f64, t_f64) -> t_f64);
    compatible_ifn!("llvm.round.f32", roundf(t_f32) -> t_f32);
    compatible_ifn!("llvm.round.f64", round(t_f64) -> t_f64);


    if ccx.sess().opts.debuginfo != NoDebugInfo {
        ifn!("llvm.dbg.declare" fn(Type::metadata(ccx), Type::metadata(ccx)) -> void);
        ifn!("llvm.dbg.value" fn(Type::metadata(ccx), t_i64, Type::metadata(ccx)) -> void);
    }
    return None;
}
