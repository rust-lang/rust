// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(non_camel_case_types)];

//! Code that is useful in various trans modules.

use driver::session::Session;
use lib::llvm::{ValueRef, BasicBlockRef, BuilderRef};
use lib::llvm::{True, False, Bool};
use lib::llvm::llvm;
use lib;
use middle::lang_items::LangItem;
use middle::trans::build;
use middle::trans::cleanup;
use middle::trans::datum;
use middle::trans::datum::{Datum, Lvalue};
use middle::trans::debuginfo;
use middle::trans::type_::Type;
use middle::ty::substs;
use middle::ty;
use middle::typeck;
use util::ppaux::Repr;
use util::nodemap::NodeMap;

use arena::TypedArena;
use collections::HashMap;
use std::c_str::ToCStr;
use std::cell::{Cell, RefCell};
use std::libc::{c_uint, c_longlong, c_ulonglong, c_char};
use std::vec_ng::Vec;
use syntax::ast::Ident;
use syntax::ast;
use syntax::ast_map::{PathElem, PathName};
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use syntax::parse::token;

pub use middle::trans::context::CrateContext;

fn type_is_newtype_immediate(ccx: &CrateContext, ty: ty::t) -> bool {
    match ty::get(ty).sty {
        ty::ty_struct(def_id, ref substs) => {
            let fields = ty::struct_fields(ccx.tcx(), def_id, substs);
            fields.len() == 1 &&
                fields.get(0).ident.name ==
                    token::special_idents::unnamed_field.name &&
                type_is_immediate(ccx, fields.get(0).mt.ty)
        }
        _ => false
    }
}

pub fn type_is_immediate(ccx: &CrateContext, ty: ty::t) -> bool {
    use middle::trans::machine::llsize_of_alloc;
    use middle::trans::type_of::sizing_type_of;
    let tcx = ccx.tcx();
    let simple = ty::type_is_scalar(ty) || ty::type_is_boxed(ty) ||
        ty::type_is_unique(ty) || ty::type_is_region_ptr(ty) ||
        type_is_newtype_immediate(ccx, ty) || ty::type_is_bot(ty) ||
        ty::type_is_simd(tcx, ty);
    if simple {
        return true;
    }
    match ty::get(ty).sty {
        ty::ty_bot => true,
        ty::ty_struct(..) | ty::ty_enum(..) | ty::ty_tup(..) => {
            let llty = sizing_type_of(ccx, ty);
            llsize_of_alloc(ccx, llty) <= llsize_of_alloc(ccx, ccx.int_type)
        }
        _ => type_is_zero_size(ccx, ty)
    }
}

pub fn type_is_zero_size(ccx: &CrateContext, ty: ty::t) -> bool {
    /*!
     * Identify types which have size zero at runtime.
     */

    use middle::trans::machine::llsize_of_alloc;
    use middle::trans::type_of::sizing_type_of;
    let llty = sizing_type_of(ccx, ty);
    llsize_of_alloc(ccx, llty) == 0
}

pub fn return_type_is_void(ccx: &CrateContext, ty: ty::t) -> bool {
    /*!
     * Identifies types which we declare to be equivalent to `void`
     * in C for the purpose of function return types. These are
     * `()`, bot, and uninhabited enums. Note that all such types
     * are also zero-size, but not all zero-size types use a `void`
     * return type (in order to aid with C ABI compatibility).
     */

    ty::type_is_nil(ty) || ty::type_is_bot(ty) || ty::type_is_empty(ccx.tcx(), ty)
}

/// Generates a unique symbol based off the name given. This is used to create
/// unique symbols for things like closures.
pub fn gensym_name(name: &str) -> PathElem {
    let num = token::gensym(name);
    // use one colon which will get translated to a period by the mangler, and
    // we're guaranteed that `num` is globally unique for this crate.
    PathName(token::gensym(format!("{}:{}", name, num)))
}

pub struct tydesc_info {
    ty: ty::t,
    tydesc: ValueRef,
    size: ValueRef,
    align: ValueRef,
    name: ValueRef,
    visit_glue: Cell<Option<ValueRef>>,
}

/*
 * A note on nomenclature of linking: "extern", "foreign", and "upcall".
 *
 * An "extern" is an LLVM symbol we wind up emitting an undefined external
 * reference to. This means "we don't have the thing in this compilation unit,
 * please make sure you link it in at runtime". This could be a reference to
 * C code found in a C library, or rust code found in a rust crate.
 *
 * Most "externs" are implicitly declared (automatically) as a result of a
 * user declaring an extern _module_ dependency; this causes the rust driver
 * to locate an extern crate, scan its compilation metadata, and emit extern
 * declarations for any symbols used by the declaring crate.
 *
 * A "foreign" is an extern that references C (or other non-rust ABI) code.
 * There is no metadata to scan for extern references so in these cases either
 * a header-digester like bindgen, or manual function prototypes, have to
 * serve as declarators. So these are usually given explicitly as prototype
 * declarations, in rust code, with ABI attributes on them noting which ABI to
 * link via.
 *
 * An "upcall" is a foreign call generated by the compiler (not corresponding
 * to any user-written call in the code) into the runtime library, to perform
 * some helper task such as bringing a task to life, allocating memory, etc.
 *
 */

pub struct NodeInfo {
    id: ast::NodeId,
    span: Span,
}

pub fn expr_info(expr: &ast::Expr) -> NodeInfo {
    NodeInfo { id: expr.id, span: expr.span }
}

pub struct Stats {
    n_static_tydescs: Cell<uint>,
    n_glues_created: Cell<uint>,
    n_null_glues: Cell<uint>,
    n_real_glues: Cell<uint>,
    n_fns: Cell<uint>,
    n_monos: Cell<uint>,
    n_inlines: Cell<uint>,
    n_closures: Cell<uint>,
    n_llvm_insns: Cell<uint>,
    llvm_insns: RefCell<HashMap<~str, uint>>,
    // (ident, time-in-ms, llvm-instructions)
    fn_stats: RefCell<Vec<(~str, uint, uint)> >,
}

pub struct BuilderRef_res {
    b: BuilderRef,
}

impl Drop for BuilderRef_res {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMDisposeBuilder(self.b);
        }
    }
}

pub fn BuilderRef_res(b: BuilderRef) -> BuilderRef_res {
    BuilderRef_res {
        b: b
    }
}

pub type ExternMap = HashMap<~str, ValueRef>;

// Here `self_ty` is the real type of the self parameter to this method. It
// will only be set in the case of default methods.
pub struct param_substs {
    tys: Vec<ty::t> ,
    self_ty: Option<ty::t>,
    vtables: Option<typeck::vtable_res>,
    self_vtables: Option<typeck::vtable_param_res>
}

impl param_substs {
    pub fn validate(&self) {
        for t in self.tys.iter() { assert!(!ty::type_needs_infer(*t)); }
        for t in self.self_ty.iter() { assert!(!ty::type_needs_infer(*t)); }
    }
}

fn param_substs_to_str(this: &param_substs, tcx: &ty::ctxt) -> ~str {
    format!("param_substs \\{tys:{}, vtables:{}\\}",
         this.tys.repr(tcx),
         this.vtables.repr(tcx))
}

impl Repr for param_substs {
    fn repr(&self, tcx: &ty::ctxt) -> ~str {
        param_substs_to_str(self, tcx)
    }
}

// work around bizarre resolve errors
pub type RvalueDatum = datum::Datum<datum::Rvalue>;
pub type LvalueDatum = datum::Datum<datum::Lvalue>;

// Function context.  Every LLVM function we create will have one of
// these.
pub struct FunctionContext<'a> {
    // The ValueRef returned from a call to llvm::LLVMAddFunction; the
    // address of the first instruction in the sequence of
    // instructions for this function that will go in the .text
    // section of the executable we're generating.
    llfn: ValueRef,

    // The environment argument in a closure.
    llenv: Option<ValueRef>,

    // The place to store the return value. If the return type is immediate,
    // this is an alloca in the function. Otherwise, it's the hidden first
    // parameter to the function. After function construction, this should
    // always be Some.
    llretptr: Cell<Option<ValueRef>>,

    entry_bcx: RefCell<Option<&'a Block<'a>>>,

    // These elements: "hoisted basic blocks" containing
    // administrative activities that have to happen in only one place in
    // the function, due to LLVM's quirks.
    // A marker for the place where we want to insert the function's static
    // allocas, so that LLVM will coalesce them into a single alloca call.
    alloca_insert_pt: Cell<Option<ValueRef>>,
    llreturn: Cell<Option<BasicBlockRef>>,

    // The a value alloca'd for calls to upcalls.rust_personality. Used when
    // outputting the resume instruction.
    personality: Cell<Option<ValueRef>>,

    // True if the caller expects this fn to use the out pointer to
    // return. Either way, your code should write into llretptr, but if
    // this value is false, llretptr will be a local alloca.
    caller_expects_out_pointer: bool,

    // Maps arguments to allocas created for them in llallocas.
    llargs: RefCell<NodeMap<LvalueDatum>>,

    // Maps the def_ids for local variables to the allocas created for
    // them in llallocas.
    lllocals: RefCell<NodeMap<LvalueDatum>>,

    // Same as above, but for closure upvars
    llupvars: RefCell<NodeMap<ValueRef>>,

    // The NodeId of the function, or -1 if it doesn't correspond to
    // a user-defined function.
    id: ast::NodeId,

    // If this function is being monomorphized, this contains the type
    // substitutions used.
    param_substs: Option<@param_substs>,

    // The source span and nesting context where this function comes from, for
    // error reporting and symbol generation.
    span: Option<Span>,

    // The arena that blocks are allocated from.
    block_arena: &'a TypedArena<Block<'a>>,

    // This function's enclosing crate context.
    ccx: &'a CrateContext,

    // Used and maintained by the debuginfo module.
    debug_context: debuginfo::FunctionDebugContext,

    // Cleanup scopes.
    scopes: RefCell<Vec<cleanup::CleanupScope<'a>> >,
}

impl<'a> FunctionContext<'a> {
    pub fn arg_pos(&self, arg: uint) -> uint {
        let arg = self.env_arg_pos() + arg;
        if self.llenv.is_some() {
            arg + 1
        } else {
            arg
        }
    }

    pub fn out_arg_pos(&self) -> uint {
        assert!(self.caller_expects_out_pointer);
        0u
    }

    pub fn env_arg_pos(&self) -> uint {
        if self.caller_expects_out_pointer {
            1u
        } else {
            0u
        }
    }

    pub fn cleanup(&self) {
        unsafe {
            llvm::LLVMInstructionEraseFromParent(self.alloca_insert_pt
                                                     .get()
                                                     .unwrap());
        }
        // Remove the cycle between fcx and bcx, so memory can be freed
        self.entry_bcx.set(None);
    }

    pub fn get_llreturn(&self) -> BasicBlockRef {
        if self.llreturn.get().is_none() {

            self.llreturn.set(Some(unsafe {
                "return".with_c_str(|buf| {
                    llvm::LLVMAppendBasicBlockInContext(self.ccx.llcx, self.llfn, buf)
                })
            }))
        }

        self.llreturn.get().unwrap()
    }

    pub fn new_block(&'a self,
                     is_lpad: bool,
                     name: &str,
                     opt_node_id: Option<ast::NodeId>)
                     -> &'a Block<'a> {
        unsafe {
            let llbb = name.with_c_str(|buf| {
                    llvm::LLVMAppendBasicBlockInContext(self.ccx.llcx,
                                                        self.llfn,
                                                        buf)
                });
            Block::new(llbb, is_lpad, opt_node_id, self)
        }
    }

    pub fn new_id_block(&'a self,
                        name: &str,
                        node_id: ast::NodeId)
                        -> &'a Block<'a> {
        self.new_block(false, name, Some(node_id))
    }

    pub fn new_temp_block(&'a self,
                          name: &str)
                          -> &'a Block<'a> {
        self.new_block(false, name, None)
    }

    pub fn join_blocks(&'a self,
                       id: ast::NodeId,
                       in_cxs: &[&'a Block<'a>])
                       -> &'a Block<'a> {
        let out = self.new_id_block("join", id);
        let mut reachable = false;
        for bcx in in_cxs.iter() {
            if !bcx.unreachable.get() {
                build::Br(*bcx, out.llbb);
                reachable = true;
            }
        }
        if !reachable {
            build::Unreachable(out);
        }
        return out;
    }
}

// Heap selectors. Indicate which heap something should go on.
#[deriving(Eq)]
pub enum heap {
    heap_managed,
    heap_exchange,
    heap_exchange_closure
}

// Basic block context.  We create a block context for each basic block
// (single-entry, single-exit sequence of instructions) we generate from Rust
// code.  Each basic block we generate is attached to a function, typically
// with many basic blocks per function.  All the basic blocks attached to a
// function are organized as a directed graph.
pub struct Block<'a> {
    // The BasicBlockRef returned from a call to
    // llvm::LLVMAppendBasicBlock(llfn, name), which adds a basic
    // block to the function pointed to by llfn.  We insert
    // instructions into that block by way of this block context.
    // The block pointing to this one in the function's digraph.
    llbb: BasicBlockRef,
    terminated: Cell<bool>,
    unreachable: Cell<bool>,

    // Is this block part of a landing pad?
    is_lpad: bool,

    // AST node-id associated with this block, if any. Used for
    // debugging purposes only.
    opt_node_id: Option<ast::NodeId>,

    // The function context for the function to which this block is
    // attached.
    fcx: &'a FunctionContext<'a>,
}

impl<'a> Block<'a> {
    pub fn new<'a>(
               llbb: BasicBlockRef,
               is_lpad: bool,
               opt_node_id: Option<ast::NodeId>,
               fcx: &'a FunctionContext<'a>)
               -> &'a Block<'a> {
        fcx.block_arena.alloc(Block {
            llbb: llbb,
            terminated: Cell::new(false),
            unreachable: Cell::new(false),
            is_lpad: is_lpad,
            opt_node_id: opt_node_id,
            fcx: fcx
        })
    }

    pub fn ccx(&self) -> &'a CrateContext { self.fcx.ccx }
    pub fn tcx(&self) -> &'a ty::ctxt {
        &self.fcx.ccx.tcx
    }
    pub fn sess(&self) -> &'a Session { self.fcx.ccx.sess() }

    pub fn ident(&self, ident: Ident) -> ~str {
        token::get_ident(ident).get().to_str()
    }

    pub fn node_id_to_str(&self, id: ast::NodeId) -> ~str {
        self.tcx().map.node_to_str(id)
    }

    pub fn expr_to_str(&self, e: &ast::Expr) -> ~str {
        e.repr(self.tcx())
    }

    pub fn expr_is_lval(&self, e: &ast::Expr) -> bool {
        ty::expr_is_lval(self.tcx(), self.ccx().maps.method_map, e)
    }

    pub fn expr_kind(&self, e: &ast::Expr) -> ty::ExprKind {
        ty::expr_kind(self.tcx(), self.ccx().maps.method_map, e)
    }

    pub fn def(&self, nid: ast::NodeId) -> ast::Def {
        let def_map = self.tcx().def_map.borrow();
        match def_map.get().find(&nid) {
            Some(&v) => v,
            None => {
                self.tcx().sess.bug(format!(
                    "no def associated with node id {:?}", nid));
            }
        }
    }

    pub fn val_to_str(&self, val: ValueRef) -> ~str {
        self.ccx().tn.val_to_str(val)
    }

    pub fn llty_str(&self, ty: Type) -> ~str {
        self.ccx().tn.type_to_str(ty)
    }

    pub fn ty_to_str(&self, t: ty::t) -> ~str {
        t.repr(self.tcx())
    }

    pub fn to_str(&self) -> ~str {
        let blk: *Block = self;
        format!("[block {}]", blk)
    }
}

pub struct Result<'a> {
    bcx: &'a Block<'a>,
    val: ValueRef
}

pub fn rslt<'a>(bcx: &'a Block<'a>, val: ValueRef) -> Result<'a> {
    Result {
        bcx: bcx,
        val: val,
    }
}

impl<'a> Result<'a> {
    pub fn unpack(&self, bcx: &mut &'a Block<'a>) -> ValueRef {
        *bcx = self.bcx;
        return self.val;
    }
}

pub fn val_ty(v: ValueRef) -> Type {
    unsafe {
        Type::from_ref(llvm::LLVMTypeOf(v))
    }
}

// LLVM constant constructors.
pub fn C_null(t: Type) -> ValueRef {
    unsafe {
        llvm::LLVMConstNull(t.to_ref())
    }
}

pub fn C_undef(t: Type) -> ValueRef {
    unsafe {
        llvm::LLVMGetUndef(t.to_ref())
    }
}

pub fn C_integral(t: Type, u: u64, sign_extend: bool) -> ValueRef {
    unsafe {
        llvm::LLVMConstInt(t.to_ref(), u, sign_extend as Bool)
    }
}

pub fn C_floating(s: &str, t: Type) -> ValueRef {
    unsafe {
        s.with_c_str(|buf| llvm::LLVMConstRealOfString(t.to_ref(), buf))
    }
}

pub fn C_nil(ccx: &CrateContext) -> ValueRef {
    C_struct(ccx, [], false)
}

pub fn C_bool(ccx: &CrateContext, val: bool) -> ValueRef {
    C_integral(Type::bool(ccx), val as u64, false)
}

pub fn C_i1(ccx: &CrateContext, val: bool) -> ValueRef {
    C_integral(Type::i1(ccx), val as u64, false)
}

pub fn C_i32(ccx: &CrateContext, i: i32) -> ValueRef {
    C_integral(Type::i32(ccx), i as u64, true)
}

pub fn C_i64(ccx: &CrateContext, i: i64) -> ValueRef {
    C_integral(Type::i64(ccx), i as u64, true)
}

pub fn C_u64(ccx: &CrateContext, i: u64) -> ValueRef {
    C_integral(Type::i64(ccx), i, false)
}

pub fn C_int(ccx: &CrateContext, i: int) -> ValueRef {
    C_integral(ccx.int_type, i as u64, true)
}

pub fn C_uint(ccx: &CrateContext, i: uint) -> ValueRef {
    C_integral(ccx.int_type, i as u64, false)
}

pub fn C_u8(ccx: &CrateContext, i: uint) -> ValueRef {
    C_integral(Type::i8(ccx), i as u64, false)
}


// This is a 'c-like' raw string, which differs from
// our boxed-and-length-annotated strings.
pub fn C_cstr(cx: &CrateContext, s: InternedString) -> ValueRef {
    unsafe {
        {
            let const_cstr_cache = cx.const_cstr_cache.borrow();
            match const_cstr_cache.get().find(&s) {
                Some(&llval) => return llval,
                None => ()
            }
        }

        let sc = llvm::LLVMConstStringInContext(cx.llcx,
                                                s.get().as_ptr() as *c_char,
                                                s.get().len() as c_uint,
                                                False);

        let gsym = token::gensym("str");
        let g = format!("str{}", gsym).with_c_str(|buf| {
            llvm::LLVMAddGlobal(cx.llmod, val_ty(sc).to_ref(), buf)
        });
        llvm::LLVMSetInitializer(g, sc);
        llvm::LLVMSetGlobalConstant(g, True);
        lib::llvm::SetLinkage(g, lib::llvm::InternalLinkage);

        let mut const_cstr_cache = cx.const_cstr_cache.borrow_mut();
        const_cstr_cache.get().insert(s, g);
        g
    }
}

// NB: Do not use `do_spill_noroot` to make this into a constant string, or
// you will be kicked off fast isel. See issue #4352 for an example of this.
pub fn C_str_slice(cx: &CrateContext, s: InternedString) -> ValueRef {
    unsafe {
        let len = s.get().len();
        let cs = llvm::LLVMConstPointerCast(C_cstr(cx, s), Type::i8p(cx).to_ref());
        C_struct(cx, [cs, C_uint(cx, len)], false)
    }
}

pub fn C_binary_slice(cx: &CrateContext, data: &[u8]) -> ValueRef {
    unsafe {
        let len = data.len();
        let lldata = C_bytes(cx, data);

        let gsym = token::gensym("binary");
        let g = format!("binary{}", gsym).with_c_str(|buf| {
            llvm::LLVMAddGlobal(cx.llmod, val_ty(lldata).to_ref(), buf)
        });
        llvm::LLVMSetInitializer(g, lldata);
        llvm::LLVMSetGlobalConstant(g, True);
        lib::llvm::SetLinkage(g, lib::llvm::InternalLinkage);

        let cs = llvm::LLVMConstPointerCast(g, Type::i8p(cx).to_ref());
        C_struct(cx, [cs, C_uint(cx, len)], false)
    }
}

pub fn C_struct(ccx: &CrateContext, elts: &[ValueRef], packed: bool) -> ValueRef {
    unsafe {
        llvm::LLVMConstStructInContext(ccx.llcx,
                                       elts.as_ptr(), elts.len() as c_uint,
                                       packed as Bool)
    }
}

pub fn C_named_struct(t: Type, elts: &[ValueRef]) -> ValueRef {
    unsafe {
        llvm::LLVMConstNamedStruct(t.to_ref(), elts.as_ptr(), elts.len() as c_uint)
    }
}

pub fn C_array(ty: Type, elts: &[ValueRef]) -> ValueRef {
    unsafe {
        return llvm::LLVMConstArray(ty.to_ref(), elts.as_ptr(), elts.len() as c_uint);
    }
}

pub fn C_bytes(ccx: &CrateContext, bytes: &[u8]) -> ValueRef {
    unsafe {
        let ptr = bytes.as_ptr() as *c_char;
        return llvm::LLVMConstStringInContext(ccx.llcx, ptr, bytes.len() as c_uint, True);
    }
}

pub fn get_param(fndecl: ValueRef, param: uint) -> ValueRef {
    unsafe {
        llvm::LLVMGetParam(fndecl, param as c_uint)
    }
}

pub fn const_get_elt(cx: &CrateContext, v: ValueRef, us: &[c_uint])
                  -> ValueRef {
    unsafe {
        let r = llvm::LLVMConstExtractValue(v, us.as_ptr(), us.len() as c_uint);

        debug!("const_get_elt(v={}, us={:?}, r={})",
               cx.tn.val_to_str(v), us, cx.tn.val_to_str(r));

        return r;
    }
}

pub fn is_const(v: ValueRef) -> bool {
    unsafe {
        llvm::LLVMIsConstant(v) == True
    }
}

pub fn const_to_int(v: ValueRef) -> c_longlong {
    unsafe {
        llvm::LLVMConstIntGetSExtValue(v)
    }
}

pub fn const_to_uint(v: ValueRef) -> c_ulonglong {
    unsafe {
        llvm::LLVMConstIntGetZExtValue(v)
    }
}

pub fn is_undef(val: ValueRef) -> bool {
    unsafe {
        llvm::LLVMIsUndef(val) != False
    }
}

pub fn is_null(val: ValueRef) -> bool {
    unsafe {
        llvm::LLVMIsNull(val) != False
    }
}

// Used to identify cached monomorphized functions and vtables
#[deriving(Eq, Hash)]
pub enum mono_param_id {
    mono_precise(ty::t, Option<@Vec<mono_id> >),
    mono_any,
    mono_repr(uint /* size */,
              uint /* align */,
              MonoDataClass,
              datum::RvalueMode),
}

#[deriving(Eq, Hash)]
pub enum MonoDataClass {
    MonoBits,    // Anything not treated differently from arbitrary integer data
    MonoNonNull, // Non-null pointers (used for optional-pointer optimization)
    // FIXME(#3547)---scalars and floats are
    // treated differently in most ABIs.  But we
    // should be doing something more detailed
    // here.
    MonoFloat
}

pub fn mono_data_classify(t: ty::t) -> MonoDataClass {
    match ty::get(t).sty {
        ty::ty_float(_) => MonoFloat,
        ty::ty_rptr(..) | ty::ty_uniq(..) | ty::ty_box(..) |
        ty::ty_str(ty::vstore_uniq) | ty::ty_vec(_, ty::vstore_uniq) |
        ty::ty_bare_fn(..) => MonoNonNull,
        // Is that everything?  Would closures or slices qualify?
        _ => MonoBits
    }
}

#[deriving(Eq, Hash)]
pub struct mono_id_ {
    def: ast::DefId,
    params: Vec<mono_param_id> }

pub type mono_id = @mono_id_;

pub fn umax(cx: &Block, a: ValueRef, b: ValueRef) -> ValueRef {
    let cond = build::ICmp(cx, lib::llvm::IntULT, a, b);
    return build::Select(cx, cond, b, a);
}

pub fn umin(cx: &Block, a: ValueRef, b: ValueRef) -> ValueRef {
    let cond = build::ICmp(cx, lib::llvm::IntULT, a, b);
    return build::Select(cx, cond, a, b);
}

pub fn align_to(cx: &Block, off: ValueRef, align: ValueRef) -> ValueRef {
    let mask = build::Sub(cx, align, C_int(cx.ccx(), 1));
    let bumped = build::Add(cx, off, mask);
    return build::And(cx, bumped, build::Not(cx, mask));
}

pub fn monomorphize_type(bcx: &Block, t: ty::t) -> ty::t {
    match bcx.fcx.param_substs {
        Some(substs) => {
            ty::subst_tps(bcx.tcx(), substs.tys.as_slice(), substs.self_ty, t)
        }
        _ => {
            assert!(!ty::type_has_params(t));
            assert!(!ty::type_has_self(t));
            t
        }
    }
}

pub fn node_id_type(bcx: &Block, id: ast::NodeId) -> ty::t {
    let tcx = bcx.tcx();
    let t = ty::node_id_to_type(tcx, id);
    monomorphize_type(bcx, t)
}

pub fn expr_ty(bcx: &Block, ex: &ast::Expr) -> ty::t {
    node_id_type(bcx, ex.id)
}

pub fn expr_ty_adjusted(bcx: &Block, ex: &ast::Expr) -> ty::t {
    let tcx = bcx.tcx();
    let t = ty::expr_ty_adjusted(tcx, ex, bcx.ccx().maps.method_map.borrow().get());
    monomorphize_type(bcx, t)
}

// Key used to lookup values supplied for type parameters in an expr.
#[deriving(Eq)]
pub enum ExprOrMethodCall {
    // Type parameters for a path like `None::<int>`
    ExprId(ast::NodeId),

    // Type parameters for a method call like `a.foo::<int>()`
    MethodCall(typeck::MethodCall)
}

pub fn node_id_type_params(bcx: &Block, node: ExprOrMethodCall) -> Vec<ty::t> {
    let tcx = bcx.tcx();
    let params = match node {
        ExprId(id) => ty::node_id_to_type_params(tcx, id),
        MethodCall(method_call) => {
            bcx.ccx().maps.method_map.borrow().get().get(&method_call).substs.tps.clone()
        }
    };

    if !params.iter().all(|t| !ty::type_needs_infer(*t)) {
        bcx.sess().bug(
            format!("type parameters for node {:?} include inference types: {}",
                 node, params.map(|t| bcx.ty_to_str(*t)).connect(",")));
    }

    match bcx.fcx.param_substs {
      Some(substs) => {
        params.iter().map(|t| {
            ty::subst_tps(tcx, substs.tys.as_slice(), substs.self_ty, *t)
        }).collect()
      }
      _ => params
    }
}

pub fn node_vtables(bcx: &Block, id: ast::NodeId)
                 -> Option<typeck::vtable_res> {
    let vtable_map = bcx.ccx().maps.vtable_map.borrow();
    let raw_vtables = vtable_map.get().find(&id);
    raw_vtables.map(|vts| resolve_vtables_in_fn_ctxt(bcx.fcx, *vts))
}

// Apply the typaram substitutions in the FunctionContext to some
// vtables. This should eliminate any vtable_params.
pub fn resolve_vtables_in_fn_ctxt(fcx: &FunctionContext, vts: typeck::vtable_res)
    -> typeck::vtable_res {
    resolve_vtables_under_param_substs(fcx.ccx.tcx(),
                                       fcx.param_substs,
                                       vts)
}

pub fn resolve_vtables_under_param_substs(tcx: &ty::ctxt,
                                          param_substs: Option<@param_substs>,
                                          vts: typeck::vtable_res)
    -> typeck::vtable_res {
    @vts.iter().map(|ds|
      resolve_param_vtables_under_param_substs(tcx,
                                               param_substs,
                                               *ds))
        .collect()
}

pub fn resolve_param_vtables_under_param_substs(
    tcx: &ty::ctxt,
    param_substs: Option<@param_substs>,
    ds: typeck::vtable_param_res)
    -> typeck::vtable_param_res {
    @ds.iter().map(
        |d| resolve_vtable_under_param_substs(tcx,
                                              param_substs,
                                              d))
        .collect()
}



pub fn resolve_vtable_under_param_substs(tcx: &ty::ctxt,
                                         param_substs: Option<@param_substs>,
                                         vt: &typeck::vtable_origin)
                                         -> typeck::vtable_origin {
    match *vt {
        typeck::vtable_static(trait_id, ref tys, sub) => {
            let tys = match param_substs {
                Some(substs) => {
                    tys.iter().map(|t| {
                        ty::subst_tps(tcx,
                                      substs.tys.as_slice(),
                                      substs.self_ty,
                                      *t)
                    }).collect()
                }
                _ => Vec::from_slice(tys.as_slice())
            };
            typeck::vtable_static(
                trait_id, tys,
                resolve_vtables_under_param_substs(tcx, param_substs, sub))
        }
        typeck::vtable_param(n_param, n_bound) => {
            match param_substs {
                Some(substs) => {
                    find_vtable(tcx, substs, n_param, n_bound)
                }
                _ => {
                    tcx.sess.bug(format!(
                        "resolve_vtable_under_param_substs: asked to lookup \
                         but no vtables in the fn_ctxt!"))
                }
            }
        }
    }
}

pub fn find_vtable(tcx: &ty::ctxt,
                   ps: &param_substs,
                   n_param: typeck::param_index,
                   n_bound: uint)
                   -> typeck::vtable_origin {
    debug!("find_vtable(n_param={:?}, n_bound={}, ps={})",
           n_param, n_bound, ps.repr(tcx));

    let param_bounds = match n_param {
        typeck::param_self => ps.self_vtables.expect("self vtables missing"),
        typeck::param_numbered(n) => {
            let tables = ps.vtables
                .expect("vtables missing where they are needed");
            *tables.get(n)
        }
    };
    param_bounds.get(n_bound).clone()
}

pub fn dummy_substs(tps: Vec<ty::t> ) -> ty::substs {
    substs {
        regions: ty::ErasedRegions,
        self_ty: None,
        tps: tps
    }
}

pub fn filename_and_line_num_from_span(bcx: &Block, span: Span)
                                       -> (ValueRef, ValueRef) {
    let loc = bcx.sess().codemap().lookup_char_pos(span.lo);
    let filename_cstr = C_cstr(bcx.ccx(),
                               token::intern_and_get_ident(loc.file.deref().name));
    let filename = build::PointerCast(bcx, filename_cstr, Type::i8p(bcx.ccx()));
    let line = C_int(bcx.ccx(), loc.line as int);
    (filename, line)
}

// Casts a Rust bool value to an i1.
pub fn bool_to_i1(bcx: &Block, llval: ValueRef) -> ValueRef {
    build::ICmp(bcx, lib::llvm::IntNE, llval, C_bool(bcx.ccx(), false))
}

pub fn langcall(bcx: &Block,
                span: Option<Span>,
                msg: &str,
                li: LangItem)
                -> ast::DefId {
    match bcx.tcx().lang_items.require(li) {
        Ok(id) => id,
        Err(s) => {
            let msg = format!("{} {}", msg, s);
            match span {
                Some(span) => { bcx.tcx().sess.span_fatal(span, msg); }
                None => { bcx.tcx().sess.fatal(msg); }
            }
        }
    }
}
