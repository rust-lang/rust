// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types, non_snake_case)]

//! Code that is useful in various trans modules.

use driver::session::Session;
use llvm;
use llvm::{ValueRef, BasicBlockRef, BuilderRef, ContextRef};
use llvm::{True, False, Bool};
use middle::def;
use middle::lang_items::LangItem;
use middle::mem_categorization as mc;
use middle::subst;
use middle::subst::Subst;
use middle::trans::base;
use middle::trans::build;
use middle::trans::cleanup;
use middle::trans::datum;
use middle::trans::debuginfo;
use middle::trans::machine;
use middle::trans::type_::Type;
use middle::trans::type_of;
use middle::traits;
use middle::ty;
use middle::ty_fold;
use middle::ty_fold::TypeFoldable;
use middle::typeck;
use middle::typeck::infer;
use util::ppaux::Repr;
use util::nodemap::{DefIdMap, NodeMap};

use arena::TypedArena;
use std::collections::HashMap;
use libc::{c_uint, c_char};
use std::c_str::ToCStr;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::vec::Vec;
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
    let simple = ty::type_is_scalar(ty) ||
        ty::type_is_unique(ty) || ty::type_is_region_ptr(ty) ||
        type_is_newtype_immediate(ccx, ty) || ty::type_is_bot(ty) ||
        ty::type_is_simd(tcx, ty);
    if simple && !ty::type_is_fat_ptr(tcx, ty) {
        return true;
    }
    if !ty::type_is_sized(tcx, ty) {
        return false;
    }
    match ty::get(ty).sty {
        ty::ty_bot => true,
        ty::ty_struct(..) | ty::ty_enum(..) | ty::ty_tup(..) |
        ty::ty_unboxed_closure(..) => {
            let llty = sizing_type_of(ccx, ty);
            llsize_of_alloc(ccx, llty) <= llsize_of_alloc(ccx, ccx.int_type())
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
    let num = token::gensym(name).uint();
    // use one colon which will get translated to a period by the mangler, and
    // we're guaranteed that `num` is globally unique for this crate.
    PathName(token::gensym(format!("{}:{}", name, num).as_slice()))
}

pub struct tydesc_info {
    pub ty: ty::t,
    pub tydesc: ValueRef,
    pub size: ValueRef,
    pub align: ValueRef,
    pub name: ValueRef,
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
    pub id: ast::NodeId,
    pub span: Span,
}

pub fn expr_info(expr: &ast::Expr) -> NodeInfo {
    NodeInfo { id: expr.id, span: expr.span }
}

pub struct BuilderRef_res {
    pub b: BuilderRef,
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

pub type ExternMap = HashMap<String, ValueRef>;

// Here `self_ty` is the real type of the self parameter to this method. It
// will only be set in the case of default methods.
pub struct param_substs {
    pub substs: subst::Substs,
}

impl param_substs {
    pub fn empty() -> param_substs {
        param_substs {
            substs: subst::Substs::trans_empty(),
        }
    }

    pub fn validate(&self) {
        assert!(self.substs.types.all(|t| !ty::type_needs_infer(*t)));
    }
}

impl Repr for param_substs {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        self.substs.repr(tcx)
    }
}

pub trait SubstP {
    fn substp(&self, tcx: &ty::ctxt, param_substs: &param_substs)
              -> Self;
}

impl<T:Subst+Clone> SubstP for T {
    fn substp(&self, tcx: &ty::ctxt, substs: &param_substs) -> T {
        self.subst(tcx, &substs.substs)
    }
}

// work around bizarre resolve errors
pub type RvalueDatum = datum::Datum<datum::Rvalue>;
pub type LvalueDatum = datum::Datum<datum::Lvalue>;

// Function context.  Every LLVM function we create will have one of
// these.
pub struct FunctionContext<'a, 'tcx: 'a> {
    // The ValueRef returned from a call to llvm::LLVMAddFunction; the
    // address of the first instruction in the sequence of
    // instructions for this function that will go in the .text
    // section of the executable we're generating.
    pub llfn: ValueRef,

    // The environment argument in a closure.
    pub llenv: Option<ValueRef>,

    // A pointer to where to store the return value. If the return type is
    // immediate, this points to an alloca in the function. Otherwise, it's a
    // pointer to the hidden first parameter of the function. After function
    // construction, this should always be Some.
    pub llretslotptr: Cell<Option<ValueRef>>,

    // These pub elements: "hoisted basic blocks" containing
    // administrative activities that have to happen in only one place in
    // the function, due to LLVM's quirks.
    // A marker for the place where we want to insert the function's static
    // allocas, so that LLVM will coalesce them into a single alloca call.
    pub alloca_insert_pt: Cell<Option<ValueRef>>,
    pub llreturn: Cell<Option<BasicBlockRef>>,

    // If the function has any nested return's, including something like:
    // fn foo() -> Option<Foo> { Some(Foo { x: return None }) }, then
    // we use a separate alloca for each return
    pub needs_ret_allocas: bool,

    // The a value alloca'd for calls to upcalls.rust_personality. Used when
    // outputting the resume instruction.
    pub personality: Cell<Option<ValueRef>>,

    // True if the caller expects this fn to use the out pointer to
    // return. Either way, your code should write into the slot llretslotptr
    // points to, but if this value is false, that slot will be a local alloca.
    pub caller_expects_out_pointer: bool,

    // Maps the DefId's for local variables to the allocas created for
    // them in llallocas.
    pub lllocals: RefCell<NodeMap<LvalueDatum>>,

    // Same as above, but for closure upvars
    pub llupvars: RefCell<NodeMap<ValueRef>>,

    // The NodeId of the function, or -1 if it doesn't correspond to
    // a user-defined function.
    pub id: ast::NodeId,

    // If this function is being monomorphized, this contains the type
    // substitutions used.
    pub param_substs: &'a param_substs,

    // The source span and nesting context where this function comes from, for
    // error reporting and symbol generation.
    pub span: Option<Span>,

    // The arena that blocks are allocated from.
    pub block_arena: &'a TypedArena<BlockS<'a, 'tcx>>,

    // This function's enclosing crate context.
    pub ccx: &'a CrateContext<'a, 'tcx>,

    // Used and maintained by the debuginfo module.
    pub debug_context: debuginfo::FunctionDebugContext,

    // Cleanup scopes.
    pub scopes: RefCell<Vec<cleanup::CleanupScope<'a, 'tcx>>>,
}

impl<'a, 'tcx> FunctionContext<'a, 'tcx> {
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
    }

    pub fn get_llreturn(&self) -> BasicBlockRef {
        if self.llreturn.get().is_none() {

            self.llreturn.set(Some(unsafe {
                "return".with_c_str(|buf| {
                    llvm::LLVMAppendBasicBlockInContext(self.ccx.llcx(), self.llfn, buf)
                })
            }))
        }

        self.llreturn.get().unwrap()
    }

    pub fn get_ret_slot(&self, bcx: Block, ty: ty::t, name: &str) -> ValueRef {
        if self.needs_ret_allocas {
            base::alloca_no_lifetime(bcx, type_of::type_of(bcx.ccx(), ty), name)
        } else {
            self.llretslotptr.get().unwrap()
        }
    }

    pub fn new_block(&'a self,
                     is_lpad: bool,
                     name: &str,
                     opt_node_id: Option<ast::NodeId>)
                     -> Block<'a, 'tcx> {
        unsafe {
            let llbb = name.with_c_str(|buf| {
                    llvm::LLVMAppendBasicBlockInContext(self.ccx.llcx(),
                                                        self.llfn,
                                                        buf)
                });
            BlockS::new(llbb, is_lpad, opt_node_id, self)
        }
    }

    pub fn new_id_block(&'a self,
                        name: &str,
                        node_id: ast::NodeId)
                        -> Block<'a, 'tcx> {
        self.new_block(false, name, Some(node_id))
    }

    pub fn new_temp_block(&'a self,
                          name: &str)
                          -> Block<'a, 'tcx> {
        self.new_block(false, name, None)
    }

    pub fn join_blocks(&'a self,
                       id: ast::NodeId,
                       in_cxs: &[Block<'a, 'tcx>])
                       -> Block<'a, 'tcx> {
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

// Basic block context.  We create a block context for each basic block
// (single-entry, single-exit sequence of instructions) we generate from Rust
// code.  Each basic block we generate is attached to a function, typically
// with many basic blocks per function.  All the basic blocks attached to a
// function are organized as a directed graph.
pub struct BlockS<'blk, 'tcx: 'blk> {
    // The BasicBlockRef returned from a call to
    // llvm::LLVMAppendBasicBlock(llfn, name), which adds a basic
    // block to the function pointed to by llfn.  We insert
    // instructions into that block by way of this block context.
    // The block pointing to this one in the function's digraph.
    pub llbb: BasicBlockRef,
    pub terminated: Cell<bool>,
    pub unreachable: Cell<bool>,

    // Is this block part of a landing pad?
    pub is_lpad: bool,

    // AST node-id associated with this block, if any. Used for
    // debugging purposes only.
    pub opt_node_id: Option<ast::NodeId>,

    // The function context for the function to which this block is
    // attached.
    pub fcx: &'blk FunctionContext<'blk, 'tcx>,
}

pub type Block<'blk, 'tcx> = &'blk BlockS<'blk, 'tcx>;

impl<'blk, 'tcx> BlockS<'blk, 'tcx> {
    pub fn new(llbb: BasicBlockRef,
               is_lpad: bool,
               opt_node_id: Option<ast::NodeId>,
               fcx: &'blk FunctionContext<'blk, 'tcx>)
               -> Block<'blk, 'tcx> {
        fcx.block_arena.alloc(BlockS {
            llbb: llbb,
            terminated: Cell::new(false),
            unreachable: Cell::new(false),
            is_lpad: is_lpad,
            opt_node_id: opt_node_id,
            fcx: fcx
        })
    }

    pub fn ccx(&self) -> &'blk CrateContext<'blk, 'tcx> {
        self.fcx.ccx
    }
    pub fn tcx(&self) -> &'blk ty::ctxt<'tcx> {
        self.fcx.ccx.tcx()
    }
    pub fn sess(&self) -> &'blk Session { self.fcx.ccx.sess() }

    pub fn ident(&self, ident: Ident) -> String {
        token::get_ident(ident).get().to_string()
    }

    pub fn node_id_to_string(&self, id: ast::NodeId) -> String {
        self.tcx().map.node_to_string(id).to_string()
    }

    pub fn expr_to_string(&self, e: &ast::Expr) -> String {
        e.repr(self.tcx())
    }

    pub fn def(&self, nid: ast::NodeId) -> def::Def {
        match self.tcx().def_map.borrow().find(&nid) {
            Some(v) => v.clone(),
            None => {
                self.tcx().sess.bug(format!(
                    "no def associated with node id {}", nid).as_slice());
            }
        }
    }

    pub fn val_to_string(&self, val: ValueRef) -> String {
        self.ccx().tn().val_to_string(val)
    }

    pub fn llty_str(&self, ty: Type) -> String {
        self.ccx().tn().type_to_string(ty)
    }

    pub fn ty_to_string(&self, t: ty::t) -> String {
        t.repr(self.tcx())
    }

    pub fn to_str(&self) -> String {
        format!("[block {:p}]", self)
    }
}

impl<'blk, 'tcx> mc::Typer<'tcx> for BlockS<'blk, 'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        self.tcx()
    }

    fn node_ty(&self, id: ast::NodeId) -> mc::McResult<ty::t> {
        Ok(node_id_type(self, id))
    }

    fn node_method_ty(&self, method_call: typeck::MethodCall) -> Option<ty::t> {
        self.tcx().method_map.borrow().find(&method_call).map(|method| method.ty)
    }

    fn adjustments<'a>(&'a self) -> &'a RefCell<NodeMap<ty::AutoAdjustment>> {
        &self.tcx().adjustments
    }

    fn is_method_call(&self, id: ast::NodeId) -> bool {
        self.tcx().method_map.borrow().contains_key(&typeck::MethodCall::expr(id))
    }

    fn temporary_scope(&self, rvalue_id: ast::NodeId) -> Option<ast::NodeId> {
        self.tcx().region_maps.temporary_scope(rvalue_id)
    }

    fn unboxed_closures<'a>(&'a self)
                        -> &'a RefCell<DefIdMap<ty::UnboxedClosure>> {
        &self.tcx().unboxed_closures
    }

    fn upvar_borrow(&self, upvar_id: ty::UpvarId) -> ty::UpvarBorrow {
        self.tcx().upvar_borrow_map.borrow().get_copy(&upvar_id)
    }

    fn capture_mode(&self, closure_expr_id: ast::NodeId)
                    -> ast::CaptureClause {
        self.tcx().capture_modes.borrow().get_copy(&closure_expr_id)
    }
}

pub struct Result<'blk, 'tcx: 'blk> {
    pub bcx: Block<'blk, 'tcx>,
    pub val: ValueRef
}

impl<'b, 'tcx> Result<'b, 'tcx> {
    pub fn new(bcx: Block<'b, 'tcx>, val: ValueRef) -> Result<'b, 'tcx> {
        Result {
            bcx: bcx,
            val: val,
        }
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

pub fn C_int<I: AsI64>(ccx: &CrateContext, i: I) -> ValueRef {
    let v = i.as_i64();

    match machine::llbitsize_of_real(ccx, ccx.int_type()) {
        32 => assert!(v < (1<<31) && v >= -(1<<31)),
        64 => {},
        n => fail!("unsupported target size: {}", n)
    }

    C_integral(ccx.int_type(), v as u64, true)
}

pub fn C_uint<I: AsU64>(ccx: &CrateContext, i: I) -> ValueRef {
    let v = i.as_u64();

    match machine::llbitsize_of_real(ccx, ccx.int_type()) {
        32 => assert!(v < (1<<32)),
        64 => {},
        n => fail!("unsupported target size: {}", n)
    }

    C_integral(ccx.int_type(), v, false)
}

pub trait AsI64 { fn as_i64(self) -> i64; }
pub trait AsU64 { fn as_u64(self) -> u64; }

// FIXME: remove the intptr conversions, because they
// are host-architecture-dependent
impl AsI64 for i64 { fn as_i64(self) -> i64 { self as i64 }}
impl AsI64 for i32 { fn as_i64(self) -> i64 { self as i64 }}
impl AsI64 for int { fn as_i64(self) -> i64 { self as i64 }}

impl AsU64 for u64  { fn as_u64(self) -> u64 { self as u64 }}
impl AsU64 for u32  { fn as_u64(self) -> u64 { self as u64 }}
impl AsU64 for uint { fn as_u64(self) -> u64 { self as u64 }}

pub fn C_u8(ccx: &CrateContext, i: uint) -> ValueRef {
    C_integral(Type::i8(ccx), i as u64, false)
}


// This is a 'c-like' raw string, which differs from
// our boxed-and-length-annotated strings.
pub fn C_cstr(cx: &CrateContext, s: InternedString, null_terminated: bool) -> ValueRef {
    unsafe {
        match cx.const_cstr_cache().borrow().find(&s) {
            Some(&llval) => return llval,
            None => ()
        }

        let sc = llvm::LLVMConstStringInContext(cx.llcx(),
                                                s.get().as_ptr() as *const c_char,
                                                s.get().len() as c_uint,
                                                !null_terminated as Bool);

        let gsym = token::gensym("str");
        let g = format!("str{}", gsym.uint()).with_c_str(|buf| {
            llvm::LLVMAddGlobal(cx.llmod(), val_ty(sc).to_ref(), buf)
        });
        llvm::LLVMSetInitializer(g, sc);
        llvm::LLVMSetGlobalConstant(g, True);
        llvm::SetLinkage(g, llvm::InternalLinkage);

        cx.const_cstr_cache().borrow_mut().insert(s, g);
        g
    }
}

// NB: Do not use `do_spill_noroot` to make this into a constant string, or
// you will be kicked off fast isel. See issue #4352 for an example of this.
pub fn C_str_slice(cx: &CrateContext, s: InternedString) -> ValueRef {
    unsafe {
        let len = s.get().len();
        let cs = llvm::LLVMConstPointerCast(C_cstr(cx, s, false),
                                            Type::i8p(cx).to_ref());
        C_named_struct(cx.tn().find_type("str_slice").unwrap(), [cs, C_uint(cx, len)])
    }
}

pub fn C_binary_slice(cx: &CrateContext, data: &[u8]) -> ValueRef {
    unsafe {
        let len = data.len();
        let lldata = C_bytes(cx, data);

        let gsym = token::gensym("binary");
        let g = format!("binary{}", gsym.uint()).with_c_str(|buf| {
            llvm::LLVMAddGlobal(cx.llmod(), val_ty(lldata).to_ref(), buf)
        });
        llvm::LLVMSetInitializer(g, lldata);
        llvm::LLVMSetGlobalConstant(g, True);
        llvm::SetLinkage(g, llvm::InternalLinkage);

        let cs = llvm::LLVMConstPointerCast(g, Type::i8p(cx).to_ref());
        C_struct(cx, [cs, C_uint(cx, len)], false)
    }
}

pub fn C_struct(cx: &CrateContext, elts: &[ValueRef], packed: bool) -> ValueRef {
    C_struct_in_context(cx.llcx(), elts, packed)
}

pub fn C_struct_in_context(llcx: ContextRef, elts: &[ValueRef], packed: bool) -> ValueRef {
    unsafe {
        llvm::LLVMConstStructInContext(llcx,
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

pub fn C_bytes(cx: &CrateContext, bytes: &[u8]) -> ValueRef {
    C_bytes_in_context(cx.llcx(), bytes)
}

pub fn C_bytes_in_context(llcx: ContextRef, bytes: &[u8]) -> ValueRef {
    unsafe {
        let ptr = bytes.as_ptr() as *const c_char;
        return llvm::LLVMConstStringInContext(llcx, ptr, bytes.len() as c_uint, True);
    }
}

pub fn const_get_elt(cx: &CrateContext, v: ValueRef, us: &[c_uint])
                  -> ValueRef {
    unsafe {
        let r = llvm::LLVMConstExtractValue(v, us.as_ptr(), us.len() as c_uint);

        debug!("const_get_elt(v={}, us={}, r={})",
               cx.tn().val_to_string(v), us, cx.tn().val_to_string(r));

        return r;
    }
}

pub fn is_const(v: ValueRef) -> bool {
    unsafe {
        llvm::LLVMIsConstant(v) == True
    }
}

pub fn const_to_int(v: ValueRef) -> i64 {
    unsafe {
        llvm::LLVMConstIntGetSExtValue(v)
    }
}

pub fn const_to_uint(v: ValueRef) -> u64 {
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

pub fn monomorphize_type(bcx: &BlockS, t: ty::t) -> ty::t {
    t.subst(bcx.tcx(), &bcx.fcx.param_substs.substs)
}

pub fn node_id_type(bcx: &BlockS, id: ast::NodeId) -> ty::t {
    let tcx = bcx.tcx();
    let t = ty::node_id_to_type(tcx, id);
    monomorphize_type(bcx, t)
}

pub fn expr_ty(bcx: Block, ex: &ast::Expr) -> ty::t {
    node_id_type(bcx, ex.id)
}

pub fn expr_ty_adjusted(bcx: Block, ex: &ast::Expr) -> ty::t {
    monomorphize_type(bcx, ty::expr_ty_adjusted(bcx.tcx(), ex))
}

pub fn fulfill_obligation(ccx: &CrateContext,
                          span: Span,
                          trait_ref: Rc<ty::TraitRef>)
                          -> traits::Vtable<()>
{
    /*!
     * Attempts to resolve an obligation. The result is a shallow
     * vtable resolution -- meaning that we do not (necessarily) resolve
     * all nested obligations on the impl. Note that type check should
     * guarantee to us that all nested obligations *could be* resolved
     * if we wanted to.
     */

    let tcx = ccx.tcx();

    // Remove any references to regions; this helps improve caching.
    let trait_ref = ty_fold::erase_regions(tcx, trait_ref);

    // First check the cache.
    match ccx.trait_cache().borrow().find(&trait_ref) {
        Some(vtable) => {
            info!("Cache hit: {}", trait_ref.repr(ccx.tcx()));
            return (*vtable).clone();
        }
        None => { }
    }

    ty::populate_implementations_for_trait_if_necessary(tcx, trait_ref.def_id);
    let infcx = infer::new_infer_ctxt(tcx);

    // Parameter environment is used to give details about type parameters,
    // but since we are in trans, everything is fully monomorphized.
    let param_env = ty::empty_parameter_environment();

    // Do the initial selection for the obligation. This yields the
    // shallow result we are looking for -- that is, what specific impl.
    let mut selcx = traits::SelectionContext::new(&infcx, &param_env, tcx);
    let obligation = traits::Obligation::misc(span, trait_ref.clone());
    let selection = match selcx.select(&obligation) {
        Ok(Some(selection)) => selection,
        Ok(None) => {
            // Ambiguity can happen when monomorphizing during trans
            // expands to some humongo type that never occurred
            // statically -- this humongo type can then overflow,
            // leading to an ambiguous result. So report this as an
            // overflow bug, since I believe this is the only case
            // where ambiguity can result.
            debug!("Encountered ambiguity selecting `{}` during trans, \
                    presuming due to overflow",
                   trait_ref.repr(tcx));
            ccx.sess().span_fatal(
                span,
                "reached the recursion limit during monomorphization");
        }
        Err(e) => {
            tcx.sess.span_bug(
                span,
                format!("Encountered error `{}` selecting `{}` during trans",
                        e.repr(tcx),
                        trait_ref.repr(tcx)).as_slice())
        }
    };

    // Currently, we use a fulfillment context to completely resolve
    // all nested obligations. This is because they can inform the
    // inference of the impl's type parameters. However, in principle,
    // we only need to do this until the impl's type parameters are
    // fully bound. It could be a slight optimization to stop
    // iterating early.
    let mut fulfill_cx = traits::FulfillmentContext::new();
    let vtable = selection.map_move_nested(|obligation| {
        fulfill_cx.register_obligation(tcx, obligation);
    });
    match fulfill_cx.select_all_or_error(&infcx, &param_env, tcx) {
        Ok(()) => { }
        Err(errors) => {
            if errors.iter().all(|e| e.is_overflow()) {
                // See Ok(None) case above.
                ccx.sess().span_fatal(
                    span,
                    "reached the recursion limit during monomorphization");
            } else {
                tcx.sess.span_bug(
                    span,
                    format!("Encountered errors `{}` fulfilling `{}` during trans",
                            errors.repr(tcx),
                            trait_ref.repr(tcx)).as_slice());
            }
        }
    }

    // Use skolemize to simultaneously replace all type variables with
    // their bindings and replace all regions with 'static.  This is
    // sort of overkill because we do not expect there to be any
    // unbound type variables, hence no skolemized types should ever
    // be inserted.
    let vtable = vtable.fold_with(&mut infcx.skolemizer());

    info!("Cache miss: {}", trait_ref.repr(ccx.tcx()));
    ccx.trait_cache().borrow_mut().insert(trait_ref,
                                          vtable.clone());

    vtable
}

// Key used to lookup values supplied for type parameters in an expr.
#[deriving(PartialEq, Show)]
pub enum ExprOrMethodCall {
    // Type parameters for a path like `None::<int>`
    ExprId(ast::NodeId),

    // Type parameters for a method call like `a.foo::<int>()`
    MethodCall(typeck::MethodCall)
}

pub fn node_id_substs(bcx: Block,
                      node: ExprOrMethodCall)
                      -> subst::Substs
{
    let tcx = bcx.tcx();

    let substs = match node {
        ExprId(id) => {
            ty::node_id_item_substs(tcx, id).substs
        }
        MethodCall(method_call) => {
            tcx.method_map.borrow().get(&method_call).substs.clone()
        }
    };

    if substs.types.any(|t| ty::type_needs_infer(*t)) {
        bcx.sess().bug(
            format!("type parameters for node {} include inference types: \
                     {}",
                    node,
                    substs.repr(bcx.tcx())).as_slice());
    }

    let substs = substs.erase_regions();
    substs.substp(tcx, bcx.fcx.param_substs)
}

pub fn langcall(bcx: Block,
                span: Option<Span>,
                msg: &str,
                li: LangItem)
                -> ast::DefId {
    match bcx.tcx().lang_items.require(li) {
        Ok(id) => id,
        Err(s) => {
            let msg = format!("{} {}", msg, s);
            match span {
                Some(span) => bcx.tcx().sess.span_fatal(span, msg.as_slice()),
                None => bcx.tcx().sess.fatal(msg.as_slice()),
            }
        }
    }
}
