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
use llvm::{ValueRef, BasicBlockRef, BuilderRef, ContextRef, TypeKind};
use llvm::{True, False, Bool};
use middle::cfg;
use middle::def;
use middle::def_id::DefId;
use middle::infer;
use middle::lang_items::LangItem;
use middle::subst::{self, Substs};
use trans::base;
use trans::build;
use trans::callee;
use trans::cleanup;
use trans::consts;
use trans::datum;
use trans::debuginfo::{self, DebugLoc};
use trans::declare;
use trans::machine;
use trans::monomorphize;
use trans::type_::Type;
use trans::type_of;
use middle::traits;
use middle::ty::{self, HasTypeFlags, Ty};
use middle::ty::fold::{TypeFolder, TypeFoldable};
use rustc_front::hir;
use rustc_mir::repr::Mir;
use util::nodemap::{FnvHashMap, NodeMap};

use arena::TypedArena;
use libc::{c_uint, c_char};
use std::ffi::CString;
use std::cell::{Cell, RefCell};
use std::vec::Vec;
use syntax::ast;
use syntax::codemap::{DUMMY_SP, Span};
use syntax::parse::token::InternedString;
use syntax::parse::token;

pub use trans::context::CrateContext;

/// Is the type's representation size known at compile time?
pub fn type_is_sized<'tcx>(tcx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> bool {
    ty.is_sized(&tcx.empty_parameter_environment(), DUMMY_SP)
}

pub fn type_is_fat_ptr<'tcx>(cx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.sty {
        ty::TyRawPtr(ty::TypeAndMut{ty, ..}) |
        ty::TyRef(_, ty::TypeAndMut{ty, ..}) |
        ty::TyBox(ty) => {
            !type_is_sized(cx, ty)
        }
        _ => {
            false
        }
    }
}

/// If `type_needs_drop` returns true, then `ty` is definitely
/// non-copy and *might* have a destructor attached; if it returns
/// false, then `ty` definitely has no destructor (i.e. no drop glue).
///
/// (Note that this implies that if `ty` has a destructor attached,
/// then `type_needs_drop` will definitely return `true` for `ty`.)
pub fn type_needs_drop<'tcx>(cx: &ty::ctxt<'tcx>, ty: Ty<'tcx>) -> bool {
    type_needs_drop_given_env(cx, ty, &cx.empty_parameter_environment())
}

/// Core implementation of type_needs_drop, potentially making use of
/// and/or updating caches held in the `param_env`.
fn type_needs_drop_given_env<'a,'tcx>(cx: &ty::ctxt<'tcx>,
                                      ty: Ty<'tcx>,
                                      param_env: &ty::ParameterEnvironment<'a,'tcx>) -> bool {
    // Issue #22536: We first query type_moves_by_default.  It sees a
    // normalized version of the type, and therefore will definitely
    // know whether the type implements Copy (and thus needs no
    // cleanup/drop/zeroing) ...
    let implements_copy = !ty.moves_by_default(param_env, DUMMY_SP);

    if implements_copy { return false; }

    // ... (issue #22536 continued) but as an optimization, still use
    // prior logic of asking if the `needs_drop` bit is set; we need
    // not zero non-Copy types if they have no destructor.

    // FIXME(#22815): Note that calling `ty::type_contents` is a
    // conservative heuristic; it may report that `needs_drop` is set
    // when actual type does not actually have a destructor associated
    // with it. But since `ty` absolutely did not have the `Copy`
    // bound attached (see above), it is sound to treat it as having a
    // destructor (e.g. zero its memory on move).

    let contents = ty.type_contents(cx);
    debug!("type_needs_drop ty={:?} contents={:?}", ty, contents);
    contents.needs_drop(cx)
}

fn type_is_newtype_immediate<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.sty {
        ty::TyStruct(def, substs) => {
            let fields = &def.struct_variant().fields;
            fields.len() == 1 && {
                type_is_immediate(ccx, monomorphize::field_ty(ccx.tcx(), substs, &fields[0]))
            }
        }
        _ => false
    }
}

pub fn type_is_immediate<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    use trans::machine::llsize_of_alloc;
    use trans::type_of::sizing_type_of;

    let tcx = ccx.tcx();
    let simple = ty.is_scalar() ||
        ty.is_unique() || ty.is_region_ptr() ||
        type_is_newtype_immediate(ccx, ty) ||
        ty.is_simd();
    if simple && !type_is_fat_ptr(tcx, ty) {
        return true;
    }
    if !type_is_sized(tcx, ty) {
        return false;
    }
    match ty.sty {
        ty::TyStruct(..) | ty::TyEnum(..) | ty::TyTuple(..) | ty::TyArray(_, _) |
        ty::TyClosure(..) => {
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
    ty.is_nil() || ty.is_empty(ccx.tcx())
}

/// Generates a unique symbol based off the name given. This is used to create
/// unique symbols for things like closures.
pub fn gensym_name(name: &str) -> ast::Name {
    let num = token::gensym(name).0;
    // use one colon which will get translated to a period by the mangler, and
    // we're guaranteed that `num` is globally unique for this crate.
    token::gensym(&format!("{}:{}", name, num))
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

#[derive(Copy, Clone)]
pub struct NodeIdAndSpan {
    pub id: ast::NodeId,
    pub span: Span,
}

pub fn expr_info(expr: &hir::Expr) -> NodeIdAndSpan {
    NodeIdAndSpan { id: expr.id, span: expr.span }
}

/// The concrete version of ty::FieldDef. The name is the field index if
/// the field is numeric.
pub struct Field<'tcx>(pub ast::Name, pub Ty<'tcx>);

/// The concrete version of ty::VariantDef
pub struct VariantInfo<'tcx> {
    pub discr: ty::Disr,
    pub fields: Vec<Field<'tcx>>
}

impl<'tcx> VariantInfo<'tcx> {
    pub fn from_ty(tcx: &ty::ctxt<'tcx>,
                   ty: Ty<'tcx>,
                   opt_def: Option<def::Def>)
                   -> Self
    {
        match ty.sty {
            ty::TyStruct(adt, substs) | ty::TyEnum(adt, substs) => {
                let variant = match opt_def {
                    None => adt.struct_variant(),
                    Some(def) => adt.variant_of_def(def)
                };

                VariantInfo {
                    discr: variant.disr_val,
                    fields: variant.fields.iter().map(|f| {
                        Field(f.name, monomorphize::field_ty(tcx, substs, f))
                    }).collect()
                }
            }

            ty::TyTuple(ref v) => {
                VariantInfo {
                    discr: 0,
                    fields: v.iter().enumerate().map(|(i, &t)| {
                        Field(token::intern(&i.to_string()), t)
                    }).collect()
                }
            }

            _ => {
                tcx.sess.bug(&format!(
                    "cannot get field types from the type {:?}",
                    ty));
            }
        }
    }

    /// Return the variant corresponding to a given node (e.g. expr)
    pub fn of_node(tcx: &ty::ctxt<'tcx>, ty: Ty<'tcx>, id: ast::NodeId) -> Self {
        let node_def = tcx.def_map.borrow().get(&id).map(|v| v.full_def());
        Self::from_ty(tcx, ty, node_def)
    }

    pub fn field_index(&self, name: ast::Name) -> usize {
        self.fields.iter().position(|&Field(n,_)| n == name).unwrap_or_else(|| {
            panic!("unknown field `{}`", name)
        })
    }
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
    assert!(!substs.types.needs_infer());
}

// work around bizarre resolve errors
type RvalueDatum<'tcx> = datum::Datum<'tcx, datum::Rvalue>;
pub type LvalueDatum<'tcx> = datum::Datum<'tcx, datum::Lvalue>;

#[derive(Clone, Debug)]
struct HintEntry<'tcx> {
    // The datum for the dropflag-hint itself; note that many
    // source-level Lvalues will be associated with the same
    // dropflag-hint datum.
    datum: cleanup::DropHintDatum<'tcx>,
}

pub struct DropFlagHintsMap<'tcx> {
    // Maps NodeId for expressions that read/write unfragmented state
    // to that state's drop-flag "hint."  (A stack-local hint
    // indicates either that (1.) it is certain that no-drop is
    // needed, or (2.)  inline drop-flag must be consulted.)
    node_map: NodeMap<HintEntry<'tcx>>,
}

impl<'tcx> DropFlagHintsMap<'tcx> {
    pub fn new() -> DropFlagHintsMap<'tcx> { DropFlagHintsMap { node_map: NodeMap() } }
    pub fn has_hint(&self, id: ast::NodeId) -> bool { self.node_map.contains_key(&id) }
    pub fn insert(&mut self, id: ast::NodeId, datum: cleanup::DropHintDatum<'tcx>) {
        self.node_map.insert(id, HintEntry { datum: datum });
    }
    pub fn hint_datum(&self, id: ast::NodeId) -> Option<cleanup::DropHintDatum<'tcx>> {
        self.node_map.get(&id).map(|t|t.datum)
    }
}

// Function context.  Every LLVM function we create will have one of
// these.
pub struct FunctionContext<'a, 'tcx: 'a> {
    // The MIR for this function. At present, this is optional because
    // we only have MIR available for things that are local to the
    // crate.
    pub mir: Option<&'a Mir<'tcx>>,

    // The ValueRef returned from a call to llvm::LLVMAddFunction; the
    // address of the first instruction in the sequence of
    // instructions for this function that will go in the .text
    // section of the executable we're generating.
    pub llfn: ValueRef,

    // always an empty parameter-environment NOTE: @jroesch another use of ParamEnv
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

    // Carries info about drop-flags for local bindings (longer term,
    // paths) for the code being compiled.
    pub lldropflag_hints: RefCell<DropFlagHintsMap<'tcx>>,

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
    pub fn mir(&self) -> &'a Mir<'tcx> {
        self.mir.unwrap()
    }

    pub fn arg_offset(&self) -> usize {
        self.env_arg_pos() + if self.llenv.is_some() { 1 } else { 0 }
    }

    pub fn env_arg_pos(&self) -> usize {
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
            base::alloca(bcx, match output {
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
        where T : TypeFoldable<'tcx> + HasTypeFlags
    {
        monomorphize::apply_param_substs(self.ccx.tcx(),
                                         self.param_substs,
                                         value)
    }

    /// This is the same as `common::type_needs_drop`, except that it
    /// may use or update caches within this `FunctionContext`.
    pub fn type_needs_drop(&self, ty: Ty<'tcx>) -> bool {
        type_needs_drop_given_env(self.ccx.tcx(), ty, &self.param_env)
    }

    pub fn eh_personality(&self) -> ValueRef {
        // The exception handling personality function.
        //
        // If our compilation unit has the `eh_personality` lang item somewhere
        // within it, then we just need to translate that. Otherwise, we're
        // building an rlib which will depend on some upstream implementation of
        // this function, so we just codegen a generic reference to it. We don't
        // specify any of the types for the function, we just make it a symbol
        // that LLVM can later use.
        //
        // Note that MSVC is a little special here in that we don't use the
        // `eh_personality` lang item at all. Currently LLVM has support for
        // both Dwarf and SEH unwind mechanisms for MSVC targets and uses the
        // *name of the personality function* to decide what kind of unwind side
        // tables/landing pads to emit. It looks like Dwarf is used by default,
        // injecting a dependency on the `_Unwind_Resume` symbol for resuming
        // an "exception", but for MSVC we want to force SEH. This means that we
        // can't actually have the personality function be our standard
        // `rust_eh_personality` function, but rather we wired it up to the
        // CRT's custom personality function, which forces LLVM to consider
        // landing pads as "landing pads for SEH".
        let target = &self.ccx.sess().target.target;
        match self.ccx.tcx().lang_items.eh_personality() {
            Some(def_id) if !base::wants_msvc_seh(self.ccx.sess()) => {
                callee::trans_fn_ref(self.ccx, def_id, ExprId(0),
                                     self.param_substs).val
            }
            _ => {
                let mut personality = self.ccx.eh_personality().borrow_mut();
                match *personality {
                    Some(llpersonality) => llpersonality,
                    None => {
                        let name = if !base::wants_msvc_seh(self.ccx.sess()) {
                            "rust_eh_personality"
                        } else if target.arch == "x86" {
                            "_except_handler3"
                        } else {
                            "__C_specific_handler"
                        };
                        let fty = Type::variadic_func(&[], &Type::i32(self.ccx));
                        let f = declare::declare_cfn(self.ccx, name, fty,
                                                     self.ccx.tcx().types.i32);
                        *personality = Some(f);
                        f
                    }
                }
            }
        }
    }

    // Returns a ValueRef of the "eh_unwind_resume" lang item if one is defined,
    // otherwise declares it as an external funtion.
    pub fn eh_unwind_resume(&self) -> ValueRef {
        use trans::attributes;
        assert!(self.ccx.sess().target.target.options.custom_unwind_resume);
        match self.ccx.tcx().lang_items.eh_unwind_resume() {
            Some(def_id) => {
                callee::trans_fn_ref(self.ccx, def_id, ExprId(0),
                                     self.param_substs).val
            }
            None => {
                let mut unwresume = self.ccx.eh_unwind_resume().borrow_mut();
                match *unwresume {
                    Some(llfn) => llfn,
                    None => {
                        let fty = Type::func(&[Type::i8p(self.ccx)], &Type::void(self.ccx));
                        let llfn = declare::declare_fn(self.ccx,
                                                       "rust_eh_unwind_resume",
                                                       llvm::CCallConv,
                                                       fty, ty::FnDiverging);
                        attributes::unwind(llfn, true);
                        *unwresume = Some(llfn);
                        llfn
                    }
                }
            }
        }
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

    pub fn mir(&self) -> &'blk Mir<'tcx> {
        self.fcx.mir()
    }

    pub fn name(&self, name: ast::Name) -> String {
        name.to_string()
    }

    pub fn node_id_to_string(&self, id: ast::NodeId) -> String {
        self.tcx().map.node_to_string(id).to_string()
    }

    pub fn def(&self, nid: ast::NodeId) -> def::Def {
        match self.tcx().def_map.borrow().get(&nid) {
            Some(v) => v.full_def(),
            None => {
                self.tcx().sess.bug(&format!(
                    "no def associated with node id {}", nid));
            }
        }
    }

    pub fn val_to_string(&self, val: ValueRef) -> String {
        self.ccx().tn().val_to_string(val)
    }

    pub fn llty_str(&self, ty: Type) -> String {
        self.ccx().tn().type_to_string(ty)
    }

    pub fn to_str(&self) -> String {
        format!("[block {:p}]", self)
    }

    pub fn monomorphize<T>(&self, value: &T) -> T
        where T : TypeFoldable<'tcx> + HasTypeFlags
    {
        monomorphize::apply_param_substs(self.tcx(),
                                         self.fcx.param_substs,
                                         value)
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

pub fn C_u32(ccx: &CrateContext, i: u32) -> ValueRef {
    C_integral(Type::i32(ccx), i as u64, false)
}

pub fn C_u64(ccx: &CrateContext, i: u64) -> ValueRef {
    C_integral(Type::i64(ccx), i, false)
}

pub fn C_int<I: AsI64>(ccx: &CrateContext, i: I) -> ValueRef {
    let v = i.as_i64();

    let bit_size = machine::llbitsize_of_real(ccx, ccx.int_type());

    if bit_size < 64 {
        // make sure it doesn't overflow
        assert!(v < (1<<(bit_size-1)) && v >= -(1<<(bit_size-1)));
    }

    C_integral(ccx.int_type(), v as u64, true)
}

pub fn C_uint<I: AsU64>(ccx: &CrateContext, i: I) -> ValueRef {
    let v = i.as_u64();

    let bit_size = machine::llbitsize_of_real(ccx, ccx.int_type());

    if bit_size < 64 {
        // make sure it doesn't overflow
        assert!(v < (1<<bit_size));
    }

    C_integral(ccx.int_type(), v, false)
}

pub trait AsI64 { fn as_i64(self) -> i64; }
pub trait AsU64 { fn as_u64(self) -> u64; }

// FIXME: remove the intptr conversions, because they
// are host-architecture-dependent
impl AsI64 for i64 { fn as_i64(self) -> i64 { self as i64 }}
impl AsI64 for i32 { fn as_i64(self) -> i64 { self as i64 }}
impl AsI64 for isize { fn as_i64(self) -> i64 { self as i64 }}

impl AsU64 for u64  { fn as_u64(self) -> u64 { self as u64 }}
impl AsU64 for u32  { fn as_u64(self) -> u64 { self as u64 }}
impl AsU64 for usize { fn as_u64(self) -> u64 { self as u64 }}

pub fn C_u8(ccx: &CrateContext, i: u8) -> ValueRef {
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
        let sym = format!("str{}", gsym.0);
        let g = declare::define_global(cx, &sym[..], val_ty(sc)).unwrap_or_else(||{
            cx.sess().bug(&format!("symbol `{}` is already defined", sym));
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

fn is_const_integral(v: ValueRef) -> bool {
    unsafe {
        !llvm::LLVMIsAConstantInt(v).is_null()
    }
}

pub fn const_to_opt_int(v: ValueRef) -> Option<i64> {
    unsafe {
        if is_const_integral(v) {
            Some(llvm::LLVMConstIntGetSExtValue(v))
        } else {
            None
        }
    }
}

pub fn const_to_opt_uint(v: ValueRef) -> Option<u64> {
    unsafe {
        if is_const_integral(v) {
            Some(llvm::LLVMConstIntGetZExtValue(v))
        } else {
            None
        }
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
    let t = tcx.node_id_to_type(id);
    monomorphize_type(bcx, t)
}

pub fn expr_ty<'blk, 'tcx>(bcx: &BlockS<'blk, 'tcx>, ex: &hir::Expr) -> Ty<'tcx> {
    node_id_type(bcx, ex.id)
}

pub fn expr_ty_adjusted<'blk, 'tcx>(bcx: &BlockS<'blk, 'tcx>, ex: &hir::Expr) -> Ty<'tcx> {
    monomorphize_type(bcx, bcx.tcx().expr_ty_adjusted(ex))
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
    let trait_ref = tcx.erase_regions(&trait_ref);

    // First check the cache.
    match ccx.trait_cache().borrow().get(&trait_ref) {
        Some(vtable) => {
            info!("Cache hit: {:?}", trait_ref);
            return (*vtable).clone();
        }
        None => { }
    }

    debug!("trans fulfill_obligation: trait_ref={:?} def_id={:?}",
           trait_ref, trait_ref.def_id());


    // Do the initial selection for the obligation. This yields the
    // shallow result we are looking for -- that is, what specific impl.
    let infcx = infer::normalizing_infer_ctxt(tcx, &tcx.tables);
    let mut selcx = traits::SelectionContext::new(&infcx);

    let obligation =
        traits::Obligation::new(traits::ObligationCause::misc(span, ast::DUMMY_NODE_ID),
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
            debug!("Encountered ambiguity selecting `{:?}` during trans, \
                    presuming due to overflow",
                   trait_ref);
            ccx.sess().span_fatal(
                span,
                "reached the recursion limit during monomorphization (selection ambiguity)");
        }
        Err(e) => {
            tcx.sess.span_bug(
                span,
                &format!("Encountered error `{:?}` selecting `{:?}` during trans",
                        e,
                        trait_ref))
        }
    };

    // Currently, we use a fulfillment context to completely resolve
    // all nested obligations. This is because they can inform the
    // inference of the impl's type parameters.
    let mut fulfill_cx = infcx.fulfillment_cx.borrow_mut();
    let vtable = selection.map(|predicate| {
        fulfill_cx.register_predicate_obligation(&infcx, predicate);
    });
    let vtable = infer::drain_fulfillment_cx_or_panic(
        span, &infcx, &mut fulfill_cx, &vtable
    );

    info!("Cache miss: {:?} => {:?}", trait_ref, vtable);

    ccx.trait_cache().borrow_mut().insert(trait_ref, vtable.clone());

    vtable
}

/// Normalizes the predicates and checks whether they hold.  If this
/// returns false, then either normalize encountered an error or one
/// of the predicates did not hold. Used when creating vtables to
/// check for unsatisfiable methods.
pub fn normalize_and_test_predicates<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                               predicates: Vec<ty::Predicate<'tcx>>)
                                               -> bool
{
    debug!("normalize_and_test_predicates(predicates={:?})",
           predicates);

    let tcx = ccx.tcx();
    let infcx = infer::normalizing_infer_ctxt(tcx, &tcx.tables);
    let mut selcx = traits::SelectionContext::new(&infcx);
    let mut fulfill_cx = infcx.fulfillment_cx.borrow_mut();
    let cause = traits::ObligationCause::dummy();
    let traits::Normalized { value: predicates, obligations } =
        traits::normalize(&mut selcx, cause.clone(), &predicates);
    for obligation in obligations {
        fulfill_cx.register_predicate_obligation(&infcx, obligation);
    }
    for predicate in predicates {
        let obligation = traits::Obligation::new(cause.clone(), predicate);
        fulfill_cx.register_predicate_obligation(&infcx, obligation);
    }

    infer::drain_fulfillment_cx(&infcx, &mut fulfill_cx, &()).is_ok()
}

// Key used to lookup values supplied for type parameters in an expr.
#[derive(Copy, Clone, PartialEq, Debug)]
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
            tcx.node_id_item_substs(id).substs
        }
        MethodCallKey(method_call) => {
            tcx.tables.borrow().method_map[&method_call].substs.clone()
        }
    };

    if substs.types.needs_infer() {
            tcx.sess.bug(&format!("type parameters for node {:?} include inference types: {:?}",
                                 node, substs));
        }

        monomorphize::apply_param_substs(tcx,
                                         param_substs,
                                         &substs.erase_regions())
}

pub fn langcall(bcx: Block,
                span: Option<Span>,
                msg: &str,
                li: LangItem)
                -> DefId {
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

/// Return the VariantDef corresponding to an inlined variant node
pub fn inlined_variant_def<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                     inlined_vid: ast::NodeId)
                                     -> ty::VariantDef<'tcx>
{

    let ctor_ty = ccx.tcx().node_id_to_type(inlined_vid);
    debug!("inlined_variant_def: ctor_ty={:?} inlined_vid={:?}", ctor_ty,
           inlined_vid);
    let adt_def = match ctor_ty.sty {
        ty::TyBareFn(_, &ty::BareFnTy { sig: ty::Binder(ty::FnSig {
            output: ty::FnConverging(ty), ..
        }), ..}) => ty,
        _ => ctor_ty
    }.ty_adt_def().unwrap();
    let inlined_vid_def_id = ccx.tcx().map.local_def_id(inlined_vid);
    adt_def.variants.iter().find(|v| {
        inlined_vid_def_id == v.did ||
            ccx.external().borrow().get(&v.did) == Some(&Some(inlined_vid))
    }).unwrap_or_else(|| {
        ccx.sess().bug(&format!("no variant for {:?}::{}", adt_def, inlined_vid))
    })
}

// To avoid UB from LLVM, these two functions mask RHS with an
// appropriate mask unconditionally (i.e. the fallback behavior for
// all shifts). For 32- and 64-bit types, this matches the semantics
// of Java. (See related discussion on #1877 and #10183.)

pub fn build_unchecked_lshift<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                          lhs: ValueRef,
                                          rhs: ValueRef,
                                          binop_debug_loc: DebugLoc) -> ValueRef {
    let rhs = base::cast_shift_expr_rhs(bcx, hir::BinOp_::BiShl, lhs, rhs);
    // #1877, #10183: Ensure that input is always valid
    let rhs = shift_mask_rhs(bcx, rhs, binop_debug_loc);
    build::Shl(bcx, lhs, rhs, binop_debug_loc)
}

pub fn build_unchecked_rshift<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                          lhs_t: Ty<'tcx>,
                                          lhs: ValueRef,
                                          rhs: ValueRef,
                                          binop_debug_loc: DebugLoc) -> ValueRef {
    let rhs = base::cast_shift_expr_rhs(bcx, hir::BinOp_::BiShr, lhs, rhs);
    // #1877, #10183: Ensure that input is always valid
    let rhs = shift_mask_rhs(bcx, rhs, binop_debug_loc);
    let is_signed = lhs_t.is_signed();
    if is_signed {
        build::AShr(bcx, lhs, rhs, binop_debug_loc)
    } else {
        build::LShr(bcx, lhs, rhs, binop_debug_loc)
    }
}

fn shift_mask_rhs<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              rhs: ValueRef,
                              debug_loc: DebugLoc) -> ValueRef {
    let rhs_llty = val_ty(rhs);
    build::And(bcx, rhs, shift_mask_val(bcx, rhs_llty, rhs_llty, false), debug_loc)
}

pub fn shift_mask_val<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              llty: Type,
                              mask_llty: Type,
                              invert: bool) -> ValueRef {
    let kind = llty.kind();
    match kind {
        TypeKind::Integer => {
            // i8/u8 can shift by at most 7, i16/u16 by at most 15, etc.
            let val = llty.int_width() - 1;
            if invert {
                C_integral(mask_llty, !val, true)
            } else {
                C_integral(mask_llty, val, false)
            }
        },
        TypeKind::Vector => {
            let mask = shift_mask_val(bcx, llty.element_type(), mask_llty.element_type(), invert);
            build::VectorSplat(bcx, mask_llty.vector_length(), mask)
        },
        _ => panic!("shift_mask_val: expected Integer or Vector, found {:?}", kind),
    }
}

