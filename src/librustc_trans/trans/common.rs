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

pub use self::ExprOrMethodCall::*;

use session::Session;
use llvm;
use llvm::{ValueRef, BasicBlockRef, BuilderRef, ContextRef};
use llvm::{True, False, Bool};
use middle::cfg;
use middle::def;
use middle::infer;
use middle::lang_items::LangItem;
use middle::mem_categorization as mc;
use middle::region;
use middle::subst::{self, Subst, Substs};
use trans::base;
use trans::build;
use trans::cleanup;
use trans::consts;
use trans::datum;
use trans::debuginfo::{self, DebugLoc};
use trans::machine;
use trans::monomorphize;
use trans::type_::Type;
use trans::type_of;
use middle::traits;
use middle::ty::{self, HasProjectionTypes, Ty};
use middle::ty_fold;
use middle::ty_fold::{TypeFolder, TypeFoldable};
use util::ppaux::Repr;
use util::nodemap::{FnvHashMap, NodeMap};

use arena::TypedArena;
use libc::{c_uint, c_char};
use std::ffi::CString;
use std::cell::{Cell, RefCell};
use std::vec::Vec;
use syntax::ast::Ident;
use syntax::ast;
use syntax::ast_map::{PathElem, PathName};
use syntax::codemap::{DUMMY_SP, Span};
use syntax::parse::token::InternedString;
use syntax::parse::token;
use util::common::memoized;
use util::nodemap::FnvHashSet;

pub use trans::context::CrateContext;

/// Returns an equivalent value with all free regions removed (note
/// that late-bound regions remain, because they are important for
/// subtyping, but they are anonymized and normalized as well). This
/// is a stronger, caching version of `ty_fold::erase_regions`.
pub fn erase_regions<'tcx,T>(cx: &ty::ctxt<'tcx>, value: &T) -> T
    where T : TypeFoldable<'tcx> + Repr<'tcx>
{
    let value1 = value.fold_with(&mut RegionEraser(cx));
    debug!("erase_regions({}) = {}",
           value.repr(cx), value1.repr(cx));
    return value1;

    struct RegionEraser<'a, 'tcx: 'a>(&'a ty::ctxt<'tcx>);

    impl<'a, 'tcx> TypeFolder<'tcx> for RegionEraser<'a, 'tcx> {
        fn tcx(&self) -> &ty::ctxt<'tcx> { self.0 }

        fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
            match self.tcx().normalized_cache.borrow().get(&ty).cloned() {
                None => {}
                Some(u) => return u
            }

            let t_norm = ty_fold::super_fold_ty(self, ty);
            self.tcx().normalized_cache.borrow_mut().insert(ty, t_norm);
            return t_norm;
        }

        fn fold_binder<T>(&mut self, t: &ty::Binder<T>) -> ty::Binder<T>
            where T : TypeFoldable<'tcx> + Repr<'tcx>
        {
            let u = ty::anonymize_late_bound_regions(self.tcx(), t);
            ty_fold::super_fold_binder(self, &u)
        }

        fn fold_region(&mut self, r: ty::Region) -> ty::Region {
            // because late-bound regions affect subtyping, we can't
            // erase the bound/free distinction, but we can replace
            // all free regions with 'static.
            //
            // Note that we *CAN* replace early-bound regions -- the
            // type system never "sees" those, they get substituted
            // away. In trans, they will always be erased to 'static
            // whenever a substitution occurs.
            match r {
                ty::ReLateBound(..) => r,
                _ => ty::ReStatic
            }
        }

        fn fold_substs(&mut self,
                       substs: &subst::Substs<'tcx>)
                       -> subst::Substs<'tcx> {
            subst::Substs { regions: subst::ErasedRegions,
                            types: substs.types.fold_with(self) }
        }
    }
}

// Is the type's representation size known at compile time?
pub fn type_is_sized<'tcx>(tcx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> bool {
    let param_env = ty::empty_parameter_environment(tcx);
    ty::type_is_sized(&param_env, DUMMY_SP, ty)
}

pub fn lltype_is_sized<'tcx>(cx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.sty {
        ty::ty_open(_) => true,
        _ => type_is_sized(cx, ty),
    }
}

pub fn type_is_fat_ptr<'tcx>(cx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.sty {
        ty::ty_ptr(ty::mt{ty, ..}) |
        ty::ty_rptr(_, ty::mt{ty, ..}) |
        ty::ty_uniq(ty) => {
            !type_is_sized(cx, ty)
        }
        _ => {
            false
        }
    }
}

// Return the smallest part of `ty` which is unsized. Fails if `ty` is sized.
// 'Smallest' here means component of the static representation of the type; not
// the size of an object at runtime.
pub fn unsized_part_of_type<'tcx>(cx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    match ty.sty {
        ty::ty_str | ty::ty_trait(..) | ty::ty_vec(..) => ty,
        ty::ty_struct(def_id, substs) => {
            let unsized_fields: Vec<_> =
                ty::struct_fields(cx, def_id, substs)
                .iter()
                .map(|f| f.mt.ty)
                .filter(|ty| !type_is_sized(cx, *ty))
                .collect();

            // Exactly one of the fields must be unsized.
            assert!(unsized_fields.len() == 1);

            unsized_part_of_type(cx, unsized_fields[0])
        }
        _ => {
            assert!(type_is_sized(cx, ty),
                    "unsized_part_of_type failed even though ty is unsized");
            panic!("called unsized_part_of_type with sized ty");
        }
    }
}

// Some things don't need cleanups during unwinding because the
// task can free them all at once later. Currently only things
// that only contain scalars and shared boxes can avoid unwind
// cleanups.
pub fn type_needs_unwind_cleanup<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    return memoized(ccx.needs_unwind_cleanup_cache(), ty, |ty| {
        type_needs_unwind_cleanup_(ccx.tcx(), ty, &mut FnvHashSet())
    });

    fn type_needs_unwind_cleanup_<'tcx>(tcx: &ty::ctxt<'tcx>,
                                        ty: Ty<'tcx>,
                                        tycache: &mut FnvHashSet<Ty<'tcx>>)
                                        -> bool
    {
        // Prevent infinite recursion
        if !tycache.insert(ty) {
            return false;
        }

        let mut needs_unwind_cleanup = false;
        ty::maybe_walk_ty(ty, |ty| {
            needs_unwind_cleanup |= match ty.sty {
                ty::ty_bool | ty::ty_int(_) | ty::ty_uint(_) |
                ty::ty_float(_) | ty::ty_tup(_) | ty::ty_ptr(_) => false,

                ty::ty_enum(did, substs) =>
                    ty::enum_variants(tcx, did).iter().any(|v|
                        v.args.iter().any(|&aty| {
                            let t = aty.subst(tcx, substs);
                            type_needs_unwind_cleanup_(tcx, t, tycache)
                        })
                    ),

                _ => true
            };
            !needs_unwind_cleanup
        });
        needs_unwind_cleanup
    }
}

pub fn type_needs_drop<'tcx>(cx: &ty::ctxt<'tcx>,
                         ty: Ty<'tcx>)
                         -> bool {
    ty::type_contents(cx, ty).needs_drop(cx)
}

fn type_is_newtype_immediate<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.sty {
        ty::ty_struct(def_id, substs) => {
            let fields = ty::lookup_struct_fields(ccx.tcx(), def_id);
            fields.len() == 1 && {
                let ty = ty::lookup_field_type(ccx.tcx(), def_id, fields[0].id, substs);
                let ty = monomorphize::normalize_associated_type(ccx.tcx(), &ty);
                type_is_immediate(ccx, ty)
            }
        }
        _ => false
    }
}

pub fn type_is_immediate<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    use trans::machine::llsize_of_alloc;
    use trans::type_of::sizing_type_of;

    let tcx = ccx.tcx();
    let simple = ty::type_is_scalar(ty) ||
        ty::type_is_unique(ty) || ty::type_is_region_ptr(ty) ||
        type_is_newtype_immediate(ccx, ty) ||
        ty::type_is_simd(tcx, ty);
    if simple && !type_is_fat_ptr(tcx, ty) {
        return true;
    }
    if !type_is_sized(tcx, ty) {
        return false;
    }
    match ty.sty {
        ty::ty_struct(..) | ty::ty_enum(..) | ty::ty_tup(..) | ty::ty_vec(_, Some(_)) |
        ty::ty_closure(..) => {
            let llty = sizing_type_of(ccx, ty);
            llsize_of_alloc(ccx, llty) <= llsize_of_alloc(ccx, ccx.int_type())
        }
        _ => type_is_zero_size(ccx, ty)
    }
}

/// Identify types which have size zero at runtime.
pub fn type_is_zero_size<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    use trans::machine::llsize_of_alloc;
    use trans::type_of::sizing_type_of;
    let llty = sizing_type_of(ccx, ty);
    llsize_of_alloc(ccx, llty) == 0
}

/// Identifies types which we declare to be equivalent to `void` in C for the purpose of function
/// return types. These are `()`, bot, and uninhabited enums. Note that all such types are also
/// zero-size, but not all zero-size types use a `void` return type (in order to aid with C ABI
/// compatibility).
pub fn return_type_is_void(ccx: &CrateContext, ty: Ty) -> bool {
    ty::type_is_nil(ty) || ty::type_is_empty(ccx.tcx(), ty)
}

/// Generates a unique symbol based off the name given. This is used to create
/// unique symbols for things like closures.
pub fn gensym_name(name: &str) -> PathElem {
    let num = token::gensym(name).usize();
    // use one colon which will get translated to a period by the mangler, and
    // we're guaranteed that `num` is globally unique for this crate.
    PathName(token::gensym(&format!("{}:{}", name, num)[]))
}

#[derive(Copy)]
pub struct tydesc_info<'tcx> {
    pub ty: Ty<'tcx>,
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

#[derive(Copy)]
pub struct NodeIdAndSpan {
    pub id: ast::NodeId,
    pub span: Span,
}

pub fn expr_info(expr: &ast::Expr) -> NodeIdAndSpan {
    NodeIdAndSpan { id: expr.id, span: expr.span }
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

pub type ExternMap = FnvHashMap<String, ValueRef>;

pub fn validate_substs(substs: &Substs) {
    assert!(substs.types.all(|t| !ty::type_needs_infer(*t)));
}

// work around bizarre resolve errors
type RvalueDatum<'tcx> = datum::Datum<'tcx, datum::Rvalue>;
type LvalueDatum<'tcx> = datum::Datum<'tcx, datum::Lvalue>;

// Function context.  Every LLVM function we create will have one of
// these.
pub struct FunctionContext<'a, 'tcx: 'a> {
    // The ValueRef returned from a call to llvm::LLVMAddFunction; the
    // address of the first instruction in the sequence of
    // instructions for this function that will go in the .text
    // section of the executable we're generating.
    pub llfn: ValueRef,

    // always an empty parameter-environment
    pub param_env: ty::ParameterEnvironment<'a, 'tcx>,

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
    pub lllocals: RefCell<NodeMap<LvalueDatum<'tcx>>>,

    // Same as above, but for closure upvars
    pub llupvars: RefCell<NodeMap<ValueRef>>,

    // The NodeId of the function, or -1 if it doesn't correspond to
    // a user-defined function.
    pub id: ast::NodeId,

    // If this function is being monomorphized, this contains the type
    // substitutions used.
    pub param_substs: &'tcx Substs<'tcx>,

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

    pub cfg: Option<cfg::CFG>,
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

    pub fn env_arg_pos(&self) -> uint {
        if self.caller_expects_out_pointer {
            1
        } else {
            0
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
                llvm::LLVMAppendBasicBlockInContext(self.ccx.llcx(), self.llfn,
                                                    "return\0".as_ptr() as *const _)
            }))
        }

        self.llreturn.get().unwrap()
    }

    pub fn get_ret_slot(&self, bcx: Block<'a, 'tcx>,
                        output: ty::FnOutput<'tcx>,
                        name: &str) -> ValueRef {
        if self.needs_ret_allocas {
            base::alloca_no_lifetime(bcx, match output {
                ty::FnConverging(output_type) => type_of::type_of(bcx.ccx(), output_type),
                ty::FnDiverging => Type::void(bcx.ccx())
            }, name)
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
            let name = CString::new(name).unwrap();
            let llbb = llvm::LLVMAppendBasicBlockInContext(self.ccx.llcx(),
                                                           self.llfn,
                                                           name.as_ptr());
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
        for bcx in in_cxs {
            if !bcx.unreachable.get() {
                build::Br(*bcx, out.llbb, DebugLoc::None);
                reachable = true;
            }
        }
        if !reachable {
            build::Unreachable(out);
        }
        return out;
    }

    pub fn monomorphize<T>(&self, value: &T) -> T
        where T : TypeFoldable<'tcx> + Repr<'tcx> + HasProjectionTypes + Clone
    {
        monomorphize::apply_param_substs(self.ccx.tcx(),
                                         self.param_substs,
                                         value)
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
        token::get_ident(ident).to_string()
    }

    pub fn node_id_to_string(&self, id: ast::NodeId) -> String {
        self.tcx().map.node_to_string(id).to_string()
    }

    pub fn expr_to_string(&self, e: &ast::Expr) -> String {
        e.repr(self.tcx())
    }

    pub fn def(&self, nid: ast::NodeId) -> def::Def {
        match self.tcx().def_map.borrow().get(&nid) {
            Some(v) => v.clone(),
            None => {
                self.tcx().sess.bug(&format!(
                    "no def associated with node id {}", nid)[]);
            }
        }
    }

    pub fn val_to_string(&self, val: ValueRef) -> String {
        self.ccx().tn().val_to_string(val)
    }

    pub fn llty_str(&self, ty: Type) -> String {
        self.ccx().tn().type_to_string(ty)
    }

    pub fn ty_to_string(&self, t: Ty<'tcx>) -> String {
        t.repr(self.tcx())
    }

    pub fn to_str(&self) -> String {
        format!("[block {:p}]", self)
    }

    pub fn monomorphize<T>(&self, value: &T) -> T
        where T : TypeFoldable<'tcx> + Repr<'tcx> + HasProjectionTypes + Clone
    {
        monomorphize::apply_param_substs(self.tcx(),
                                         self.fcx.param_substs,
                                         value)
    }
}

impl<'blk, 'tcx> mc::Typer<'tcx> for BlockS<'blk, 'tcx> {
    fn node_ty(&self, id: ast::NodeId) -> mc::McResult<Ty<'tcx>> {
        Ok(node_id_type(self, id))
    }

    fn expr_ty_adjusted(&self, expr: &ast::Expr) -> mc::McResult<Ty<'tcx>> {
        Ok(expr_ty_adjusted(self, expr))
    }

    fn node_method_ty(&self, method_call: ty::MethodCall) -> Option<Ty<'tcx>> {
        self.tcx()
            .method_map
            .borrow()
            .get(&method_call)
            .map(|method| monomorphize_type(self, method.ty))
    }

    fn node_method_origin(&self, method_call: ty::MethodCall)
                          -> Option<ty::MethodOrigin<'tcx>>
    {
        self.tcx()
            .method_map
            .borrow()
            .get(&method_call)
            .map(|method| method.origin.clone())
    }

    fn adjustments<'a>(&'a self) -> &'a RefCell<NodeMap<ty::AutoAdjustment<'tcx>>> {
        &self.tcx().adjustments
    }

    fn is_method_call(&self, id: ast::NodeId) -> bool {
        self.tcx().method_map.borrow().contains_key(&ty::MethodCall::expr(id))
    }

    fn temporary_scope(&self, rvalue_id: ast::NodeId) -> Option<region::CodeExtent> {
        self.tcx().region_maps.temporary_scope(rvalue_id)
    }

    fn upvar_capture(&self, upvar_id: ty::UpvarId) -> Option<ty::UpvarCapture> {
        Some(self.tcx().upvar_capture_map.borrow()[upvar_id].clone())
    }

    fn type_moves_by_default(&self, span: Span, ty: Ty<'tcx>) -> bool {
        self.fcx.param_env.type_moves_by_default(span, ty)
    }
}

impl<'blk, 'tcx> ty::ClosureTyper<'tcx> for BlockS<'blk, 'tcx> {
    fn param_env<'a>(&'a self) -> &'a ty::ParameterEnvironment<'a, 'tcx> {
        &self.fcx.param_env
    }

    fn closure_kind(&self,
                    def_id: ast::DefId)
                    -> Option<ty::ClosureKind>
    {
        let typer = NormalizingClosureTyper::new(self.tcx());
        typer.closure_kind(def_id)
    }

    fn closure_type(&self,
                    def_id: ast::DefId,
                    substs: &subst::Substs<'tcx>)
                    -> ty::ClosureTy<'tcx>
    {
        let typer = NormalizingClosureTyper::new(self.tcx());
        typer.closure_type(def_id, substs)
    }

    fn closure_upvars(&self,
                      def_id: ast::DefId,
                      substs: &Substs<'tcx>)
                      -> Option<Vec<ty::ClosureUpvar<'tcx>>>
    {
        let typer = NormalizingClosureTyper::new(self.tcx());
        typer.closure_upvars(def_id, substs)
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
        let s = CString::new(s).unwrap();
        llvm::LLVMConstRealOfString(t.to_ref(), s.as_ptr())
    }
}

pub fn C_nil(ccx: &CrateContext) -> ValueRef {
    C_struct(ccx, &[], false)
}

pub fn C_bool(ccx: &CrateContext, val: bool) -> ValueRef {
    C_integral(Type::i1(ccx), val as u64, false)
}

pub fn C_i32(ccx: &CrateContext, i: i32) -> ValueRef {
    C_integral(Type::i32(ccx), i as u64, true)
}

pub fn C_u64(ccx: &CrateContext, i: u64) -> ValueRef {
    C_integral(Type::i64(ccx), i, false)
}

pub fn C_int<I: AsI64>(ccx: &CrateContext, i: I) -> ValueRef {
    let v = i.as_i64();

    match machine::llbitsize_of_real(ccx, ccx.int_type()) {
        32 => assert!(v < (1<<31) && v >= -(1<<31)),
        64 => {},
        n => panic!("unsupported target size: {}", n)
    }

    C_integral(ccx.int_type(), v as u64, true)
}

pub fn C_uint<I: AsU64>(ccx: &CrateContext, i: I) -> ValueRef {
    let v = i.as_u64();

    match machine::llbitsize_of_real(ccx, ccx.int_type()) {
        32 => assert!(v < (1<<32)),
        64 => {},
        n => panic!("unsupported target size: {}", n)
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
        match cx.const_cstr_cache().borrow().get(&s) {
            Some(&llval) => return llval,
            None => ()
        }

        let sc = llvm::LLVMConstStringInContext(cx.llcx(),
                                                s.as_ptr() as *const c_char,
                                                s.len() as c_uint,
                                                !null_terminated as Bool);

        let gsym = token::gensym("str");
        let buf = CString::new(format!("str{}", gsym.usize()));
        let buf = buf.unwrap();
        let g = llvm::LLVMAddGlobal(cx.llmod(), val_ty(sc).to_ref(), buf.as_ptr());
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
    let len = s.len();
    let cs = consts::ptrcast(C_cstr(cx, s, false), Type::i8p(cx));
    C_named_struct(cx.tn().find_type("str_slice").unwrap(), &[cs, C_uint(cx, len)])
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

pub fn C_vector(elts: &[ValueRef]) -> ValueRef {
    unsafe {
        return llvm::LLVMConstVector(elts.as_ptr(), elts.len() as c_uint);
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

        debug!("const_get_elt(v={}, us={:?}, r={})",
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

#[allow(dead_code)] // potentially useful
pub fn is_null(val: ValueRef) -> bool {
    unsafe {
        llvm::LLVMIsNull(val) != False
    }
}

pub fn monomorphize_type<'blk, 'tcx>(bcx: &BlockS<'blk, 'tcx>, t: Ty<'tcx>) -> Ty<'tcx> {
    bcx.fcx.monomorphize(&t)
}

pub fn node_id_type<'blk, 'tcx>(bcx: &BlockS<'blk, 'tcx>, id: ast::NodeId) -> Ty<'tcx> {
    let tcx = bcx.tcx();
    let t = ty::node_id_to_type(tcx, id);
    monomorphize_type(bcx, t)
}

pub fn expr_ty<'blk, 'tcx>(bcx: &BlockS<'blk, 'tcx>, ex: &ast::Expr) -> Ty<'tcx> {
    node_id_type(bcx, ex.id)
}

pub fn expr_ty_adjusted<'blk, 'tcx>(bcx: &BlockS<'blk, 'tcx>, ex: &ast::Expr) -> Ty<'tcx> {
    monomorphize_type(bcx, ty::expr_ty_adjusted(bcx.tcx(), ex))
}

/// Attempts to resolve an obligation. The result is a shallow vtable resolution -- meaning that we
/// do not (necessarily) resolve all nested obligations on the impl. Note that type check should
/// guarantee to us that all nested obligations *could be* resolved if we wanted to.
pub fn fulfill_obligation<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                span: Span,
                                trait_ref: ty::PolyTraitRef<'tcx>)
                                -> traits::Vtable<'tcx, ()>
{
    let tcx = ccx.tcx();

    // Remove any references to regions; this helps improve caching.
    let trait_ref = erase_regions(tcx, &trait_ref);

    // First check the cache.
    match ccx.trait_cache().borrow().get(&trait_ref) {
        Some(vtable) => {
            info!("Cache hit: {}", trait_ref.repr(ccx.tcx()));
            return (*vtable).clone();
        }
        None => { }
    }

    debug!("trans fulfill_obligation: trait_ref={}", trait_ref.repr(ccx.tcx()));

    ty::populate_implementations_for_trait_if_necessary(tcx, trait_ref.def_id());
    let infcx = infer::new_infer_ctxt(tcx);

    // Do the initial selection for the obligation. This yields the
    // shallow result we are looking for -- that is, what specific impl.
    let typer = NormalizingClosureTyper::new(tcx);
    let mut selcx = traits::SelectionContext::new(&infcx, &typer);
    let obligation = traits::Obligation::new(traits::ObligationCause::dummy(),
                                             trait_ref.to_poly_trait_predicate());
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
                &format!("Encountered error `{}` selecting `{}` during trans",
                        e.repr(tcx),
                        trait_ref.repr(tcx))[])
        }
    };

    // Currently, we use a fulfillment context to completely resolve
    // all nested obligations. This is because they can inform the
    // inference of the impl's type parameters.
    let mut fulfill_cx = traits::FulfillmentContext::new();
    let vtable = selection.map_move_nested(|predicate| {
        fulfill_cx.register_predicate_obligation(&infcx, predicate);
    });
    let vtable = drain_fulfillment_cx(span, &infcx, &mut fulfill_cx, &vtable);

    info!("Cache miss: {}", trait_ref.repr(ccx.tcx()));
    ccx.trait_cache().borrow_mut().insert(trait_ref,
                                          vtable.clone());

    vtable
}

pub struct NormalizingClosureTyper<'a,'tcx:'a> {
    param_env: ty::ParameterEnvironment<'a, 'tcx>
}

impl<'a,'tcx> NormalizingClosureTyper<'a,'tcx> {
    pub fn new(tcx: &'a ty::ctxt<'tcx>) -> NormalizingClosureTyper<'a,'tcx> {
        // Parameter environment is used to give details about type parameters,
        // but since we are in trans, everything is fully monomorphized.
        NormalizingClosureTyper { param_env: ty::empty_parameter_environment(tcx) }
    }
}

impl<'a,'tcx> ty::ClosureTyper<'tcx> for NormalizingClosureTyper<'a,'tcx> {
    fn param_env<'b>(&'b self) -> &'b ty::ParameterEnvironment<'b,'tcx> {
        &self.param_env
    }

    fn closure_kind(&self,
                    def_id: ast::DefId)
                    -> Option<ty::ClosureKind>
    {
        self.param_env.closure_kind(def_id)
    }

    fn closure_type(&self,
                    def_id: ast::DefId,
                    substs: &subst::Substs<'tcx>)
                    -> ty::ClosureTy<'tcx>
    {
        // the substitutions in `substs` are already monomorphized,
        // but we still must normalize associated types
        let closure_ty = self.param_env.tcx.closure_type(def_id, substs);
        monomorphize::normalize_associated_type(self.param_env.tcx, &closure_ty)
    }

    fn closure_upvars(&self,
                      def_id: ast::DefId,
                      substs: &Substs<'tcx>)
                      -> Option<Vec<ty::ClosureUpvar<'tcx>>>
    {
        // the substitutions in `substs` are already monomorphized,
        // but we still must normalize associated types
        let result = ty::closure_upvars(&self.param_env, def_id, substs);
        monomorphize::normalize_associated_type(self.param_env.tcx, &result)
    }
}

pub fn drain_fulfillment_cx<'a,'tcx,T>(span: Span,
                                   infcx: &infer::InferCtxt<'a,'tcx>,
                                   fulfill_cx: &mut traits::FulfillmentContext<'tcx>,
                                   result: &T)
                                   -> T
    where T : TypeFoldable<'tcx> + Repr<'tcx>
{
    debug!("drain_fulfillment_cx(result={})",
           result.repr(infcx.tcx));

    // In principle, we only need to do this so long as `result`
    // contains unbound type parameters. It could be a slight
    // optimization to stop iterating early.
    let typer = NormalizingClosureTyper::new(infcx.tcx);
    match fulfill_cx.select_all_or_error(infcx, &typer) {
        Ok(()) => { }
        Err(errors) => {
            if errors.iter().all(|e| e.is_overflow()) {
                // See Ok(None) case above.
                infcx.tcx.sess.span_fatal(
                    span,
                    "reached the recursion limit during monomorphization");
            } else {
                infcx.tcx.sess.span_bug(
                    span,
                    &format!("Encountered errors `{}` fulfilling during trans",
                            errors.repr(infcx.tcx))[]);
            }
        }
    }

    // Use freshen to simultaneously replace all type variables with
    // their bindings and replace all regions with 'static.  This is
    // sort of overkill because we do not expect there to be any
    // unbound type variables, hence no `TyFresh` types should ever be
    // inserted.
    result.fold_with(&mut infcx.freshener())
}

// Key used to lookup values supplied for type parameters in an expr.
#[derive(Copy, PartialEq, Debug)]
pub enum ExprOrMethodCall {
    // Type parameters for a path like `None::<int>`
    ExprId(ast::NodeId),

    // Type parameters for a method call like `a.foo::<int>()`
    MethodCallKey(ty::MethodCall)
}

pub fn node_id_substs<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                            node: ExprOrMethodCall,
                            param_substs: &subst::Substs<'tcx>)
                            -> subst::Substs<'tcx> {
    let tcx = ccx.tcx();

    let substs = match node {
        ExprId(id) => {
            ty::node_id_item_substs(tcx, id).substs
        }
        MethodCallKey(method_call) => {
            (*tcx.method_map.borrow())[method_call].substs.clone()
        }
    };

    if substs.types.any(|t| ty::type_needs_infer(*t)) {
            tcx.sess.bug(&format!("type parameters for node {:?} include inference types: {:?}",
                                 node, substs.repr(tcx))[]);
        }

        monomorphize::apply_param_substs(tcx,
                                         param_substs,
                                         &substs.erase_regions())
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
                Some(span) => bcx.tcx().sess.span_fatal(span, &msg[..]),
                None => bcx.tcx().sess.fatal(&msg[..]),
            }
        }
    }
}
