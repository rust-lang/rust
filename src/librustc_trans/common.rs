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

use session::Session;
use llvm;
use llvm::{ValueRef, BasicBlockRef, BuilderRef, ContextRef, TypeKind};
use llvm::{True, False, Bool, OperandBundleDef};
use rustc::cfg;
use rustc::hir::def::Def;
use rustc::hir::def_id::DefId;
use rustc::infer::TransNormalize;
use rustc::util::common::MemoizationMap;
use middle::lang_items::LangItem;
use rustc::ty::subst::Substs;
use abi::{Abi, FnType};
use base;
use build;
use builder::Builder;
use callee::Callee;
use cleanup;
use consts;
use datum;
use debuginfo::{self, DebugLoc};
use declare;
use machine;
use mir::CachedMir;
use monomorphize;
use type_::Type;
use value::Value;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::layout::Layout;
use rustc::traits::{self, SelectionContext, ProjectionMode};
use rustc::ty::fold::TypeFoldable;
use rustc::hir;
use util::nodemap::NodeMap;

use arena::TypedArena;
use libc::{c_uint, c_char};
use std::ops::Deref;
use std::ffi::CString;
use std::cell::{Cell, RefCell};

use syntax::ast;
use syntax::parse::token::InternedString;
use syntax::parse::token;
use syntax_pos::{DUMMY_SP, Span};

pub use context::{CrateContext, SharedCrateContext};

/// Is the type's representation size known at compile time?
pub fn type_is_sized<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> bool {
    ty.is_sized(tcx, &tcx.empty_parameter_environment(), DUMMY_SP)
}

pub fn type_is_fat_ptr<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.sty {
        ty::TyRawPtr(ty::TypeAndMut{ty, ..}) |
        ty::TyRef(_, ty::TypeAndMut{ty, ..}) |
        ty::TyBox(ty) => {
            !type_is_sized(tcx, ty)
        }
        _ => {
            false
        }
    }
}

pub fn type_is_immediate<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    use machine::llsize_of_alloc;
    use type_of::sizing_type_of;

    let tcx = ccx.tcx();
    let simple = ty.is_scalar() ||
        ty.is_unique() || ty.is_region_ptr() ||
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

/// Returns Some([a, b]) if the type has a pair of fields with types a and b.
pub fn type_pair_fields<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>)
                                  -> Option<[Ty<'tcx>; 2]> {
    match ty.sty {
        ty::TyEnum(adt, substs) | ty::TyStruct(adt, substs) => {
            assert_eq!(adt.variants.len(), 1);
            let fields = &adt.variants[0].fields;
            if fields.len() != 2 {
                return None;
            }
            Some([monomorphize::field_ty(ccx.tcx(), substs, &fields[0]),
                  monomorphize::field_ty(ccx.tcx(), substs, &fields[1])])
        }
        ty::TyClosure(_, ty::ClosureSubsts { upvar_tys: tys, .. }) |
        ty::TyTuple(tys) => {
            if tys.len() != 2 {
                return None;
            }
            Some([tys[0], tys[1]])
        }
        _ => None
    }
}

/// Returns true if the type is represented as a pair of immediates.
pub fn type_is_imm_pair<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>)
                                  -> bool {
    let tcx = ccx.tcx();
    let layout = tcx.normalizing_infer_ctxt(ProjectionMode::Any).enter(|infcx| {
        match ty.layout(&infcx) {
            Ok(layout) => layout,
            Err(err) => {
                bug!("type_is_imm_pair: layout for `{:?}` failed: {}",
                     ty, err);
            }
        }
    });

    match *layout {
        Layout::FatPointer { .. } => true,
        Layout::Univariant { ref variant, .. } => {
            // There must be only 2 fields.
            if variant.offset_after_field.len() != 2 {
                return false;
            }

            match type_pair_fields(ccx, ty) {
                Some([a, b]) => {
                    type_is_immediate(ccx, a) && type_is_immediate(ccx, b)
                }
                None => false
            }
        }
        _ => false
    }
}

/// Identify types which have size zero at runtime.
pub fn type_is_zero_size<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> bool {
    use machine::llsize_of_alloc;
    use type_of::sizing_type_of;
    let llty = sizing_type_of(ccx, ty);
    llsize_of_alloc(ccx, llty) == 0
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

use Disr;

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
    pub discr: Disr,
    pub fields: Vec<Field<'tcx>>
}

impl<'a, 'tcx> VariantInfo<'tcx> {
    pub fn from_ty(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                   ty: Ty<'tcx>,
                   opt_def: Option<Def>)
                   -> Self
    {
        match ty.sty {
            ty::TyStruct(adt, substs) | ty::TyEnum(adt, substs) => {
                let variant = match opt_def {
                    None => adt.struct_variant(),
                    Some(def) => adt.variant_of_def(def)
                };

                VariantInfo {
                    discr: Disr::from(variant.disr_val),
                    fields: variant.fields.iter().map(|f| {
                        Field(f.name, monomorphize::field_ty(tcx, substs, f))
                    }).collect()
                }
            }

            ty::TyTuple(ref v) => {
                VariantInfo {
                    discr: Disr(0),
                    fields: v.iter().enumerate().map(|(i, &t)| {
                        Field(token::intern(&i.to_string()), t)
                    }).collect()
                }
            }

            _ => {
                bug!("cannot get field types from the type {:?}", ty);
            }
        }
    }

    /// Return the variant corresponding to a given node (e.g. expr)
    pub fn of_node(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>, id: ast::NodeId) -> Self {
        Self::from_ty(tcx, ty, Some(tcx.expect_def(id)))
    }

    pub fn field_index(&self, name: ast::Name) -> usize {
        self.fields.iter().position(|&Field(n,_)| n == name).unwrap_or_else(|| {
            bug!("unknown field `{}`", name)
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
    pub mir: Option<CachedMir<'a, 'tcx>>,

    // The ValueRef returned from a call to llvm::LLVMAddFunction; the
    // address of the first instruction in the sequence of
    // instructions for this function that will go in the .text
    // section of the executable we're generating.
    pub llfn: ValueRef,

    // always an empty parameter-environment NOTE: @jroesch another use of ParamEnv
    pub param_env: ty::ParameterEnvironment<'tcx>,

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

    // When working with landingpad-based exceptions this value is alloca'd and
    // later loaded when using the resume instruction. This ends up being
    // critical to chaining landing pads and resuing already-translated
    // cleanups.
    //
    // Note that for cleanuppad-based exceptions this is not used.
    pub landingpad_alloca: Cell<Option<ValueRef>>,

    // Maps the DefId's for local variables to the allocas created for
    // them in llallocas.
    pub lllocals: RefCell<NodeMap<LvalueDatum<'tcx>>>,

    // Same as above, but for closure upvars
    pub llupvars: RefCell<NodeMap<ValueRef>>,

    // Carries info about drop-flags for local bindings (longer term,
    // paths) for the code being compiled.
    pub lldropflag_hints: RefCell<DropFlagHintsMap<'tcx>>,

    // Describes the return/argument LLVM types and their ABI handling.
    pub fn_ty: FnType,

    // If this function is being monomorphized, this contains the type
    // substitutions used.
    pub param_substs: &'tcx Substs<'tcx>,

    // The source span and nesting context where this function comes from, for
    // error reporting and symbol generation.
    pub span: Option<Span>,

    // The arena that blocks are allocated from.
    pub block_arena: &'a TypedArena<BlockS<'a, 'tcx>>,

    // The arena that landing pads are allocated from.
    pub lpad_arena: TypedArena<LandingPad>,

    // This function's enclosing crate context.
    pub ccx: &'a CrateContext<'a, 'tcx>,

    // Used and maintained by the debuginfo module.
    pub debug_context: debuginfo::FunctionDebugContext,

    // Cleanup scopes.
    pub scopes: RefCell<Vec<cleanup::CleanupScope<'a, 'tcx>>>,

    pub cfg: Option<cfg::CFG>,
}

impl<'a, 'tcx> FunctionContext<'a, 'tcx> {
    pub fn mir(&self) -> CachedMir<'a, 'tcx> {
        self.mir.clone().expect("fcx.mir was empty")
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

    pub fn get_ret_slot(&self, bcx: Block<'a, 'tcx>, name: &str) -> ValueRef {
        if self.needs_ret_allocas {
            base::alloca(bcx, self.fn_ty.ret.memory_ty(self.ccx), name)
        } else {
            self.llretslotptr.get().unwrap()
        }
    }

    pub fn new_block(&'a self,
                     name: &str,
                     opt_node_id: Option<ast::NodeId>)
                     -> Block<'a, 'tcx> {
        unsafe {
            let name = CString::new(name).unwrap();
            let llbb = llvm::LLVMAppendBasicBlockInContext(self.ccx.llcx(),
                                                           self.llfn,
                                                           name.as_ptr());
            BlockS::new(llbb, opt_node_id, self)
        }
    }

    pub fn new_id_block(&'a self,
                        name: &str,
                        node_id: ast::NodeId)
                        -> Block<'a, 'tcx> {
        self.new_block(name, Some(node_id))
    }

    pub fn new_temp_block(&'a self,
                          name: &str)
                          -> Block<'a, 'tcx> {
        self.new_block(name, None)
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
        where T: TransNormalize<'tcx>
    {
        monomorphize::apply_param_substs(self.ccx.tcx(),
                                         self.param_substs,
                                         value)
    }

    /// This is the same as `common::type_needs_drop`, except that it
    /// may use or update caches within this `FunctionContext`.
    pub fn type_needs_drop(&self, ty: Ty<'tcx>) -> bool {
        self.ccx.tcx().type_needs_drop_given_env(ty, &self.param_env)
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
        let ccx = self.ccx;
        let tcx = ccx.tcx();
        match tcx.lang_items.eh_personality() {
            Some(def_id) if !base::wants_msvc_seh(ccx.sess()) => {
                Callee::def(ccx, def_id, tcx.mk_substs(Substs::empty())).reify(ccx).val
            }
            _ => {
                if let Some(llpersonality) = ccx.eh_personality().get() {
                    return llpersonality
                }
                let name = if base::wants_msvc_seh(ccx.sess()) {
                    "__CxxFrameHandler3"
                } else {
                    "rust_eh_personality"
                };
                let fty = Type::variadic_func(&[], &Type::i32(ccx));
                let f = declare::declare_cfn(ccx, name, fty);
                ccx.eh_personality().set(Some(f));
                f
            }
        }
    }

    // Returns a ValueRef of the "eh_unwind_resume" lang item if one is defined,
    // otherwise declares it as an external function.
    pub fn eh_unwind_resume(&self) -> Callee<'tcx> {
        use attributes;
        let ccx = self.ccx;
        let tcx = ccx.tcx();
        assert!(ccx.sess().target.target.options.custom_unwind_resume);
        if let Some(def_id) = tcx.lang_items.eh_unwind_resume() {
            return Callee::def(ccx, def_id, tcx.mk_substs(Substs::empty()));
        }

        let ty = tcx.mk_fn_ptr(tcx.mk_bare_fn(ty::BareFnTy {
            unsafety: hir::Unsafety::Unsafe,
            abi: Abi::C,
            sig: ty::Binder(ty::FnSig {
                inputs: vec![tcx.mk_mut_ptr(tcx.types.u8)],
                output: ty::FnDiverging,
                variadic: false
            }),
        }));

        let unwresume = ccx.eh_unwind_resume();
        if let Some(llfn) = unwresume.get() {
            return Callee::ptr(datum::immediate_rvalue(llfn, ty));
        }
        let llfn = declare::declare_fn(ccx, "rust_eh_unwind_resume", ty);
        attributes::unwind(llfn, true);
        unwresume.set(Some(llfn));
        Callee::ptr(datum::immediate_rvalue(llfn, ty))
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

    // If this block part of a landing pad, then this is `Some` indicating what
    // kind of landing pad its in, otherwise this is none.
    pub lpad: Cell<Option<&'blk LandingPad>>,

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
               opt_node_id: Option<ast::NodeId>,
               fcx: &'blk FunctionContext<'blk, 'tcx>)
               -> Block<'blk, 'tcx> {
        fcx.block_arena.alloc(BlockS {
            llbb: llbb,
            terminated: Cell::new(false),
            unreachable: Cell::new(false),
            lpad: Cell::new(None),
            opt_node_id: opt_node_id,
            fcx: fcx
        })
    }

    pub fn ccx(&self) -> &'blk CrateContext<'blk, 'tcx> {
        self.fcx.ccx
    }
    pub fn fcx(&self) -> &'blk FunctionContext<'blk, 'tcx> {
        self.fcx
    }
    pub fn tcx(&self) -> TyCtxt<'blk, 'tcx, 'tcx> {
        self.fcx.ccx.tcx()
    }
    pub fn sess(&self) -> &'blk Session { self.fcx.ccx.sess() }

    pub fn lpad(&self) -> Option<&'blk LandingPad> {
        self.lpad.get()
    }

    pub fn set_lpad_ref(&self, lpad: Option<&'blk LandingPad>) {
        // FIXME: use an IVar?
        self.lpad.set(lpad);
    }

    pub fn set_lpad(&self, lpad: Option<LandingPad>) {
        self.set_lpad_ref(lpad.map(|p| &*self.fcx().lpad_arena.alloc(p)))
    }

    pub fn mir(&self) -> CachedMir<'blk, 'tcx> {
        self.fcx.mir()
    }

    pub fn name(&self, name: ast::Name) -> String {
        name.to_string()
    }

    pub fn node_id_to_string(&self, id: ast::NodeId) -> String {
        self.tcx().map.node_to_string(id).to_string()
    }

    pub fn to_str(&self) -> String {
        format!("[block {:p}]", self)
    }

    pub fn monomorphize<T>(&self, value: &T) -> T
        where T: TransNormalize<'tcx>
    {
        monomorphize::apply_param_substs(self.tcx(),
                                         self.fcx.param_substs,
                                         value)
    }

    pub fn build(&'blk self) -> BlockAndBuilder<'blk, 'tcx> {
        BlockAndBuilder::new(self, OwnedBuilder::new_with_ccx(self.ccx()))
    }
}

pub struct OwnedBuilder<'blk, 'tcx: 'blk> {
    builder: Builder<'blk, 'tcx>
}

impl<'blk, 'tcx> OwnedBuilder<'blk, 'tcx> {
    pub fn new_with_ccx(ccx: &'blk CrateContext<'blk, 'tcx>) -> Self {
        // Create a fresh builder from the crate context.
        let llbuilder = unsafe {
            llvm::LLVMCreateBuilderInContext(ccx.llcx())
        };
        OwnedBuilder {
            builder: Builder {
                llbuilder: llbuilder,
                ccx: ccx,
            }
        }
    }
}

impl<'blk, 'tcx> Drop for OwnedBuilder<'blk, 'tcx> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMDisposeBuilder(self.builder.llbuilder);
        }
    }
}

pub struct BlockAndBuilder<'blk, 'tcx: 'blk> {
    bcx: Block<'blk, 'tcx>,
    owned_builder: OwnedBuilder<'blk, 'tcx>,
}

impl<'blk, 'tcx> BlockAndBuilder<'blk, 'tcx> {
    pub fn new(bcx: Block<'blk, 'tcx>, owned_builder: OwnedBuilder<'blk, 'tcx>) -> Self {
        // Set the builder's position to this block's end.
        owned_builder.builder.position_at_end(bcx.llbb);
        BlockAndBuilder {
            bcx: bcx,
            owned_builder: owned_builder,
        }
    }

    pub fn with_block<F, R>(&self, f: F) -> R
        where F: FnOnce(Block<'blk, 'tcx>) -> R
    {
        let result = f(self.bcx);
        self.position_at_end(self.bcx.llbb);
        result
    }

    pub fn map_block<F>(self, f: F) -> Self
        where F: FnOnce(Block<'blk, 'tcx>) -> Block<'blk, 'tcx>
    {
        let BlockAndBuilder { bcx, owned_builder } = self;
        let bcx = f(bcx);
        BlockAndBuilder::new(bcx, owned_builder)
    }

    pub fn at_start<F, R>(&self, f: F) -> R
        where F: FnOnce(&BlockAndBuilder<'blk, 'tcx>) -> R
    {
        self.position_at_start(self.bcx.llbb);
        let r = f(self);
        self.position_at_end(self.bcx.llbb);
        r
    }

    // Methods delegated to bcx

    pub fn is_unreachable(&self) -> bool {
        self.bcx.unreachable.get()
    }

    pub fn ccx(&self) -> &'blk CrateContext<'blk, 'tcx> {
        self.bcx.ccx()
    }
    pub fn fcx(&self) -> &'blk FunctionContext<'blk, 'tcx> {
        self.bcx.fcx()
    }
    pub fn tcx(&self) -> TyCtxt<'blk, 'tcx, 'tcx> {
        self.bcx.tcx()
    }
    pub fn sess(&self) -> &'blk Session {
        self.bcx.sess()
    }

    pub fn llbb(&self) -> BasicBlockRef {
        self.bcx.llbb
    }

    pub fn mir(&self) -> CachedMir<'blk, 'tcx> {
        self.bcx.mir()
    }

    pub fn monomorphize<T>(&self, value: &T) -> T
        where T: TransNormalize<'tcx>
    {
        self.bcx.monomorphize(value)
    }

    pub fn set_lpad(&self, lpad: Option<LandingPad>) {
        self.bcx.set_lpad(lpad)
    }

    pub fn set_lpad_ref(&self, lpad: Option<&'blk LandingPad>) {
        // FIXME: use an IVar?
        self.bcx.set_lpad_ref(lpad);
    }

    pub fn lpad(&self) -> Option<&'blk LandingPad> {
        self.bcx.lpad()
    }
}

impl<'blk, 'tcx> Deref for BlockAndBuilder<'blk, 'tcx> {
    type Target = Builder<'blk, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.owned_builder.builder
    }
}

/// A structure representing an active landing pad for the duration of a basic
/// block.
///
/// Each `Block` may contain an instance of this, indicating whether the block
/// is part of a landing pad or not. This is used to make decision about whether
/// to emit `invoke` instructions (e.g. in a landing pad we don't continue to
/// use `invoke`) and also about various function call metadata.
///
/// For GNU exceptions (`landingpad` + `resume` instructions) this structure is
/// just a bunch of `None` instances (not too interesting), but for MSVC
/// exceptions (`cleanuppad` + `cleanupret` instructions) this contains data.
/// When inside of a landing pad, each function call in LLVM IR needs to be
/// annotated with which landing pad it's a part of. This is accomplished via
/// the `OperandBundleDef` value created for MSVC landing pads.
pub struct LandingPad {
    cleanuppad: Option<ValueRef>,
    operand: Option<OperandBundleDef>,
}

impl LandingPad {
    pub fn gnu() -> LandingPad {
        LandingPad { cleanuppad: None, operand: None }
    }

    pub fn msvc(cleanuppad: ValueRef) -> LandingPad {
        LandingPad {
            cleanuppad: Some(cleanuppad),
            operand: Some(OperandBundleDef::new("funclet", &[cleanuppad])),
        }
    }

    pub fn bundle(&self) -> Option<&OperandBundleDef> {
        self.operand.as_ref()
    }

    pub fn cleanuppad(&self) -> Option<ValueRef> {
        self.cleanuppad
    }
}

impl Clone for LandingPad {
    fn clone(&self) -> LandingPad {
        LandingPad {
            cleanuppad: self.cleanuppad,
            operand: self.cleanuppad.map(|p| {
                OperandBundleDef::new("funclet", &[p])
            }),
        }
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

pub fn C_floating_f64(f: f64, t: Type) -> ValueRef {
    unsafe {
        llvm::LLVMConstReal(t.to_ref(), f)
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
        if let Some(&llval) = cx.const_cstr_cache().borrow().get(&s) {
            return llval;
        }

        let sc = llvm::LLVMConstStringInContext(cx.llcx(),
                                                s.as_ptr() as *const c_char,
                                                s.len() as c_uint,
                                                !null_terminated as Bool);

        let gsym = token::gensym("str");
        let sym = format!("str{}", gsym.0);
        let g = declare::define_global(cx, &sym[..], val_ty(sc)).unwrap_or_else(||{
            bug!("symbol `{}` is already defined", sym);
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

pub fn const_get_elt(v: ValueRef, us: &[c_uint])
              -> ValueRef {
    unsafe {
        let r = llvm::LLVMConstExtractValue(v, us.as_ptr(), us.len() as c_uint);

        debug!("const_get_elt(v={:?}, us={:?}, r={:?})",
               Value(v), us, Value(r));

        r
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
pub fn fulfill_obligation<'a, 'tcx>(scx: &SharedCrateContext<'a, 'tcx>,
                                    span: Span,
                                    trait_ref: ty::PolyTraitRef<'tcx>)
                                    -> traits::Vtable<'tcx, ()>
{
    let tcx = scx.tcx();

    // Remove any references to regions; this helps improve caching.
    let trait_ref = tcx.erase_regions(&trait_ref);

    scx.trait_cache().memoize(trait_ref, || {
        debug!("trans::fulfill_obligation(trait_ref={:?}, def_id={:?})",
               trait_ref, trait_ref.def_id());

        // Do the initial selection for the obligation. This yields the
        // shallow result we are looking for -- that is, what specific impl.
        tcx.normalizing_infer_ctxt(ProjectionMode::Any).enter(|infcx| {
            let mut selcx = SelectionContext::new(&infcx);

            let obligation_cause = traits::ObligationCause::misc(span,
                                                             ast::DUMMY_NODE_ID);
            let obligation = traits::Obligation::new(obligation_cause,
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
                    tcx.sess.span_fatal(span,
                        "reached the recursion limit during monomorphization \
                         (selection ambiguity)");
                }
                Err(e) => {
                    span_bug!(span, "Encountered error `{:?}` selecting `{:?}` during trans",
                              e, trait_ref)
                }
            };

            debug!("fulfill_obligation: selection={:?}", selection);

            // Currently, we use a fulfillment context to completely resolve
            // all nested obligations. This is because they can inform the
            // inference of the impl's type parameters.
            let mut fulfill_cx = traits::FulfillmentContext::new();
            let vtable = selection.map(|predicate| {
                debug!("fulfill_obligation: register_predicate_obligation {:?}", predicate);
                fulfill_cx.register_predicate_obligation(&infcx, predicate);
            });
            let vtable = infcx.drain_fulfillment_cx_or_panic(span, &mut fulfill_cx, &vtable);

            info!("Cache miss: {:?} => {:?}", trait_ref, vtable);
            vtable
        })
    })
}

/// Normalizes the predicates and checks whether they hold.  If this
/// returns false, then either normalize encountered an error or one
/// of the predicates did not hold. Used when creating vtables to
/// check for unsatisfiable methods.
pub fn normalize_and_test_predicates<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                               predicates: Vec<ty::Predicate<'tcx>>)
                                               -> bool
{
    debug!("normalize_and_test_predicates(predicates={:?})",
           predicates);

    tcx.normalizing_infer_ctxt(ProjectionMode::Any).enter(|infcx| {
        let mut selcx = SelectionContext::new(&infcx);
        let mut fulfill_cx = traits::FulfillmentContext::new();
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

        infcx.drain_fulfillment_cx(&mut fulfill_cx, &()).is_ok()
    })
}

pub fn langcall(tcx: TyCtxt,
                span: Option<Span>,
                msg: &str,
                li: LangItem)
                -> DefId {
    match tcx.lang_items.require(li) {
        Ok(id) => id,
        Err(s) => {
            let msg = format!("{} {}", msg, s);
            match span {
                Some(span) => tcx.sess.span_fatal(span, &msg[..]),
                None => tcx.sess.fatal(&msg[..]),
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
        ty::TyFnDef(_, _, &ty::BareFnTy { sig: ty::Binder(ty::FnSig {
            output: ty::FnConverging(ty), ..
        }), ..}) => ty,
        _ => ctor_ty
    }.ty_adt_def().unwrap();
    let inlined_vid_def_id = ccx.tcx().map.local_def_id(inlined_vid);
    adt_def.variants.iter().find(|v| {
        inlined_vid_def_id == v.did ||
            ccx.external().borrow().get(&v.did) == Some(&Some(inlined_vid))
    }).unwrap_or_else(|| {
        bug!("no variant for {:?}::{}", adt_def, inlined_vid)
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
        _ => bug!("shift_mask_val: expected Integer or Vector, found {:?}", kind),
    }
}
