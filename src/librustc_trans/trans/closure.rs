// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::ClosureKind::*;

use back::abi;
use back::link::mangle_internal_name_by_path_and_seq;
use llvm::ValueRef;
use middle::mem_categorization::Typer;
use trans::adt;
use trans::base::*;
use trans::build::*;
use trans::cleanup::{CleanupMethods, ScopeId};
use trans::common::*;
use trans::datum::{Datum, DatumBlock, Expr, Lvalue, rvalue_scratch_datum};
use trans::debuginfo;
use trans::expr;
use trans::monomorphize::MonoId;
use trans::type_of::*;
use trans::type_::Type;
use middle::ty::{mod, Ty};
use middle::subst::{Subst, Substs};
use session::config::FullDebugInfo;
use util::ppaux::Repr;
use util::ppaux::ty_to_string;

use arena::TypedArena;
use syntax::ast;
use syntax::ast_util;

// ___Good to know (tm)__________________________________________________
//
// The layout of a closure environment in memory is
// roughly as follows:
//
// struct rust_opaque_box {         // see rust_internal.h
//   unsigned ref_count;            // obsolete (part of @T's header)
//   fn(void*) *drop_glue;          // destructor (for proc)
//   rust_opaque_box *prev;         // obsolete (part of @T's header)
//   rust_opaque_box *next;         // obsolete (part of @T's header)
//   struct closure_data {
//       upvar1_t upvar1;
//       ...
//       upvarN_t upvarN;
//    }
// };
//
// Note that the closure is itself a rust_opaque_box.  This is true
// even for ~fn and ||, because we wish to keep binary compatibility
// between all kinds of closures.  The allocation strategy for this
// closure depends on the closure type.  For a sendfn, the closure
// (and the referenced type descriptors) will be allocated in the
// exchange heap.  For a fn, the closure is allocated in the task heap
// and is reference counted.  For a block, the closure is allocated on
// the stack.
//
// ## Opaque closures and the embedded type descriptor ##
//
// One interesting part of closures is that they encapsulate the data
// that they close over.  So when I have a ptr to a closure, I do not
// know how many type descriptors it contains nor what upvars are
// captured within.  That means I do not know precisely how big it is
// nor where its fields are located.  This is called an "opaque
// closure".
//
// Typically an opaque closure suffices because we only manipulate it
// by ptr.  The routine Type::at_box().ptr_to() returns an appropriate
// type for such an opaque closure; it allows access to the box fields,
// but not the closure_data itself.
//
// But sometimes, such as when cloning or freeing a closure, we need
// to know the full information.  That is where the type descriptor
// that defines the closure comes in handy.  We can use its take and
// drop glue functions to allocate/free data as needed.
//
// ## Subtleties concerning alignment ##
//
// It is important that we be able to locate the closure data *without
// knowing the kind of data that is being bound*.  This can be tricky
// because the alignment requirements of the bound data affects the
// alignment requires of the closure_data struct as a whole.  However,
// right now this is a non-issue in any case, because the size of the
// rust_opaque_box header is always a multiple of 16-bytes, which is
// the maximum alignment requirement we ever have to worry about.
//
// The only reason alignment matters is that, in order to learn what data
// is bound, we would normally first load the type descriptors: but their
// location is ultimately depend on their content!  There is, however, a
// workaround.  We can load the tydesc from the rust_opaque_box, which
// describes the closure_data struct and has self-contained derived type
// descriptors, and read the alignment from there.   It's just annoying to
// do.  Hopefully should this ever become an issue we'll have monomorphized
// and type descriptors will all be a bad dream.
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[deriving(Copy)]
pub struct EnvValue<'tcx> {
    action: ast::CaptureClause,
    datum: Datum<'tcx, Lvalue>
}

impl<'tcx> EnvValue<'tcx> {
    pub fn to_string<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> String {
        format!("{}({})", self.action, self.datum.to_string(ccx))
    }
}

// Given a closure ty, emits a corresponding tuple ty
pub fn mk_closure_tys<'tcx>(tcx: &ty::ctxt<'tcx>,
                            bound_values: &[EnvValue<'tcx>])
                            -> Ty<'tcx> {
    // determine the types of the values in the env.  Note that this
    // is the actual types that will be stored in the map, not the
    // logical types as the user sees them, so by-ref upvars must be
    // converted to ptrs.
    let bound_tys = bound_values.iter().map(|bv| {
        match bv.action {
            ast::CaptureByValue => bv.datum.ty,
            ast::CaptureByRef => ty::mk_mut_ptr(tcx, bv.datum.ty)
        }
    }).collect();
    let cdata_ty = ty::mk_tup(tcx, bound_tys);
    debug!("cdata_ty={}", ty_to_string(tcx, cdata_ty));
    return cdata_ty;
}

fn tuplify_box_ty<'tcx>(tcx: &ty::ctxt<'tcx>, t: Ty<'tcx>) -> Ty<'tcx> {
    let ptr = ty::mk_imm_ptr(tcx, ty::mk_i8());
    ty::mk_tup(tcx, vec!(ty::mk_uint(), ty::mk_nil_ptr(tcx), ptr, ptr, t))
}

fn allocate_cbox<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                             store: ty::TraitStore,
                             cdata_ty: Ty<'tcx>)
                             -> Result<'blk, 'tcx> {
    let _icx = push_ctxt("closure::allocate_cbox");
    let tcx = bcx.tcx();

    // Allocate and initialize the box:
    let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
    match store {
        ty::UniqTraitStore => {
            malloc_raw_dyn_proc(bcx, cbox_ty)
        }
        ty::RegionTraitStore(..) => {
            let llbox = alloc_ty(bcx, cbox_ty, "__closure");
            Result::new(bcx, llbox)
        }
    }
}

pub struct ClosureResult<'blk, 'tcx: 'blk> {
    llbox: ValueRef,        // llvalue of ptr to closure
    cdata_ty: Ty<'tcx>,     // type of the closure data
    bcx: Block<'blk, 'tcx>  // final bcx
}

// Given a block context and a list of tydescs and values to bind
// construct a closure out of them. If copying is true, it is a
// heap allocated closure that copies the upvars into environment.
// Otherwise, it is stack allocated and copies pointers to the upvars.
pub fn store_environment<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                     bound_values: Vec<EnvValue<'tcx>> ,
                                     store: ty::TraitStore)
                                     -> ClosureResult<'blk, 'tcx> {
    let _icx = push_ctxt("closure::store_environment");
    let ccx = bcx.ccx();
    let tcx = ccx.tcx();

    // compute the type of the closure
    let cdata_ty = mk_closure_tys(tcx, bound_values[]);

    // cbox_ty has the form of a tuple: (a, b, c) we want a ptr to a
    // tuple.  This could be a ptr in uniq or a box or on stack,
    // whatever.
    let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
    let cboxptr_ty = ty::mk_ptr(tcx, ty::mt {ty:cbox_ty, mutbl:ast::MutImmutable});
    let llboxptr_ty = type_of(ccx, cboxptr_ty);

    // If there are no bound values, no point in allocating anything.
    if bound_values.is_empty() {
        return ClosureResult {llbox: C_null(llboxptr_ty),
                              cdata_ty: cdata_ty,
                              bcx: bcx};
    }

    // allocate closure in the heap
    let Result {bcx, val: llbox} = allocate_cbox(bcx, store, cdata_ty);

    let llbox = PointerCast(bcx, llbox, llboxptr_ty);
    debug!("tuplify_box_ty = {}", ty_to_string(tcx, cbox_ty));

    // Copy expr values into boxed bindings.
    let mut bcx = bcx;
    for (i, bv) in bound_values.into_iter().enumerate() {
        debug!("Copy {} into closure", bv.to_string(ccx));

        if ccx.sess().asm_comments() {
            add_comment(bcx, format!("Copy {} into closure",
                                     bv.to_string(ccx))[]);
        }

        let bound_data = GEPi(bcx, llbox, &[0u, abi::BOX_FIELD_BODY, i]);

        match bv.action {
            ast::CaptureByValue => {
                bcx = bv.datum.store_to(bcx, bound_data);
            }
            ast::CaptureByRef => {
                Store(bcx, bv.datum.to_llref(), bound_data);
            }
        }
    }

    ClosureResult { llbox: llbox, cdata_ty: cdata_ty, bcx: bcx }
}

// Given a context and a list of upvars, build a closure. This just
// collects the upvars and packages them up for store_environment.
fn build_closure<'blk, 'tcx>(bcx0: Block<'blk, 'tcx>,
                             freevar_mode: ast::CaptureClause,
                             freevars: &Vec<ty::Freevar>,
                             store: ty::TraitStore)
                             -> ClosureResult<'blk, 'tcx> {
    let _icx = push_ctxt("closure::build_closure");

    // If we need to, package up the iterator body to call
    let bcx = bcx0;

    // Package up the captured upvars
    let mut env_vals = Vec::new();
    for freevar in freevars.iter() {
        let datum = expr::trans_local_var(bcx, freevar.def);
        env_vals.push(EnvValue {action: freevar_mode, datum: datum});
    }

    store_environment(bcx, env_vals, store)
}

// Given an enclosing block context, a new function context, a closure type,
// and a list of upvars, generate code to load and populate the environment
// with the upvars and type descriptors.
fn load_environment<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                cdata_ty: Ty<'tcx>,
                                freevars: &[ty::Freevar],
                                store: ty::TraitStore)
                                -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("closure::load_environment");

    // Load a pointer to the closure data, skipping over the box header:
    let llcdata = at_box_body(bcx, cdata_ty, bcx.fcx.llenv.unwrap());

    // Store the pointer to closure data in an alloca for debug info because that's what the
    // llvm.dbg.declare intrinsic expects
    let env_pointer_alloca = if bcx.sess().opts.debuginfo == FullDebugInfo {
        let alloc = alloc_ty(bcx, ty::mk_mut_ptr(bcx.tcx(), cdata_ty), "__debuginfo_env_ptr");
        Store(bcx, llcdata, alloc);
        Some(alloc)
    } else {
        None
    };

    // Populate the upvars from the environment
    let mut i = 0u;
    for freevar in freevars.iter() {
        let mut upvarptr = GEPi(bcx, llcdata, &[0u, i]);
        let captured_by_ref = match store {
            ty::RegionTraitStore(..) => {
                upvarptr = Load(bcx, upvarptr);
                true
            }
            ty::UniqTraitStore => false
        };
        let def_id = freevar.def.def_id();

        bcx.fcx.llupvars.borrow_mut().insert(def_id.node, upvarptr);
        if let Some(env_pointer_alloca) = env_pointer_alloca {
            debuginfo::create_captured_var_metadata(
                bcx,
                def_id.node,
                env_pointer_alloca,
                i,
                captured_by_ref,
                freevar.span);
        }

        i += 1u;
    }

    bcx
}

fn load_unboxed_closure_environment<'blk, 'tcx>(
                                    bcx: Block<'blk, 'tcx>,
                                    arg_scope_id: ScopeId,
                                    freevar_mode: ast::CaptureClause,
                                    freevars: &[ty::Freevar])
                                    -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("closure::load_environment");

    // Special case for small by-value selfs.
    let closure_id = ast_util::local_def(bcx.fcx.id);
    let self_type = self_type_for_unboxed_closure(bcx.ccx(), closure_id,
                                                  node_id_type(bcx, closure_id.node));
    let kind = kind_for_unboxed_closure(bcx.ccx(), closure_id);
    let llenv = if kind == ty::FnOnceUnboxedClosureKind &&
            !arg_is_indirect(bcx.ccx(), self_type) {
        let datum = rvalue_scratch_datum(bcx,
                                         self_type,
                                         "unboxed_closure_env");
        store_ty(bcx, bcx.fcx.llenv.unwrap(), datum.val, self_type);
        datum.val
    } else {
        bcx.fcx.llenv.unwrap()
    };

    // Store the pointer to closure data in an alloca for debug info because that's what the
    // llvm.dbg.declare intrinsic expects
    let env_pointer_alloca = if bcx.sess().opts.debuginfo == FullDebugInfo {
        let alloc = alloca(bcx, val_ty(llenv), "__debuginfo_env_ptr");
        Store(bcx, llenv, alloc);
        Some(alloc)
    } else {
        None
    };

    for (i, freevar) in freevars.iter().enumerate() {
        let mut upvar_ptr = GEPi(bcx, llenv, &[0, i]);
        let captured_by_ref = match freevar_mode {
            ast::CaptureByRef => {
                upvar_ptr = Load(bcx, upvar_ptr);
                true
            }
            ast::CaptureByValue => false
        };
        let def_id = freevar.def.def_id();
        bcx.fcx.llupvars.borrow_mut().insert(def_id.node, upvar_ptr);

        if kind == ty::FnOnceUnboxedClosureKind && freevar_mode == ast::CaptureByValue {
            bcx.fcx.schedule_drop_mem(arg_scope_id,
                                      upvar_ptr,
                                      node_id_type(bcx, def_id.node))
        }

        if let Some(env_pointer_alloca) = env_pointer_alloca {
            debuginfo::create_captured_var_metadata(
                bcx,
                def_id.node,
                env_pointer_alloca,
                i,
                captured_by_ref,
                freevar.span);
        }
    }

    bcx
}

fn fill_fn_pair(bcx: Block, pair: ValueRef, llfn: ValueRef, llenvptr: ValueRef) {
    Store(bcx, llfn, GEPi(bcx, pair, &[0u, abi::FAT_PTR_ADDR]));
    let llenvptr = PointerCast(bcx, llenvptr, Type::i8p(bcx.ccx()));
    Store(bcx, llenvptr, GEPi(bcx, pair, &[0u, abi::FAT_PTR_EXTRA]));
}

#[deriving(PartialEq)]
pub enum ClosureKind<'tcx> {
    NotClosure,
    // See load_environment.
    BoxedClosure(Ty<'tcx>, ty::TraitStore),
    // See load_unboxed_closure_environment.
    UnboxedClosure(ast::CaptureClause)
}

pub struct ClosureEnv<'a, 'tcx> {
    freevars: &'a [ty::Freevar],
    pub kind: ClosureKind<'tcx>
}

impl<'a, 'tcx> ClosureEnv<'a, 'tcx> {
    pub fn new(freevars: &'a [ty::Freevar], kind: ClosureKind<'tcx>)
               -> ClosureEnv<'a, 'tcx> {
        ClosureEnv {
            freevars: freevars,
            kind: kind
        }
    }

    pub fn load<'blk>(self, bcx: Block<'blk, 'tcx>, arg_scope: ScopeId)
                      -> Block<'blk, 'tcx> {
        // Don't bother to create the block if there's nothing to load
        if self.freevars.is_empty() {
            return bcx;
        }

        match self.kind {
            NotClosure => bcx,
            BoxedClosure(cdata_ty, store) => {
                load_environment(bcx, cdata_ty, self.freevars, store)
            }
            UnboxedClosure(freevar_mode) => {
                load_unboxed_closure_environment(bcx, arg_scope, freevar_mode, self.freevars)
            }
        }
    }
}

/// Translates the body of a closure expression.
///
/// - `store`
/// - `decl`
/// - `body`
/// - `id`: The id of the closure expression.
/// - `cap_clause`: information about captured variables, if any.
/// - `dest`: where to write the closure value, which must be a
///   (fn ptr, env) pair
pub fn trans_expr_fn<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                 store: ty::TraitStore,
                                 decl: &ast::FnDecl,
                                 body: &ast::Block,
                                 id: ast::NodeId,
                                 dest: expr::Dest)
                                 -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("closure::trans_expr_fn");

    let dest_addr = match dest {
        expr::SaveIn(p) => p,
        expr::Ignore => {
            return bcx; // closure construction is non-side-effecting
        }
    };

    let ccx = bcx.ccx();
    let tcx = bcx.tcx();
    let fty = node_id_type(bcx, id);
    let s = tcx.map.with_path(id, |path| {
        mangle_internal_name_by_path_and_seq(path, "closure")
    });
    let llfn = decl_internal_rust_fn(ccx, fty, s[]);

    // set an inline hint for all closures
    set_inline_hint(llfn);

    let freevar_mode = tcx.capture_mode(id);
    let freevars: Vec<ty::Freevar> =
        ty::with_freevars(tcx, id, |fv| fv.iter().map(|&fv| fv).collect());

    let ClosureResult {
        llbox,
        cdata_ty,
        bcx
    } = build_closure(bcx, freevar_mode, &freevars, store);

    trans_closure(ccx,
                  decl,
                  body,
                  llfn,
                  bcx.fcx.param_substs,
                  id,
                  &[],
                  ty::ty_fn_ret(fty),
                  ty::ty_fn_abi(fty),
                  ClosureEnv::new(freevars[],
                                  BoxedClosure(cdata_ty, store)));
    fill_fn_pair(bcx, dest_addr, llfn, llbox);
    bcx
}

/// Returns the LLVM function declaration for an unboxed closure, creating it
/// if necessary. If the ID does not correspond to a closure ID, returns None.
pub fn get_or_create_declaration_if_unboxed_closure<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                                                closure_id: ast::DefId,
                                                                substs: &Substs<'tcx>)
                                                                -> Option<ValueRef> {
    let ccx = bcx.ccx();
    if !ccx.tcx().unboxed_closures.borrow().contains_key(&closure_id) {
        // Not an unboxed closure.
        return None
    }

    let function_type = ty::node_id_to_type(bcx.tcx(), closure_id.node);
    let function_type = function_type.subst(bcx.tcx(), substs);

    // Normalize type so differences in regions and typedefs don't cause
    // duplicate declarations
    let function_type = ty::normalize_ty(bcx.tcx(), function_type);
    let params = match function_type.sty {
        ty::ty_unboxed_closure(_, _, ref substs) => substs.types.clone(),
        _ => unreachable!()
    };
    let mono_id = MonoId {
        def: closure_id,
        params: params
    };

    match ccx.unboxed_closure_vals().borrow().get(&mono_id) {
        Some(llfn) => {
            debug!("get_or_create_declaration_if_unboxed_closure(): found \
                    closure");
            return Some(*llfn)
        }
        None => {}
    }

    let symbol = ccx.tcx().map.with_path(closure_id.node, |path| {
        mangle_internal_name_by_path_and_seq(path, "unboxed_closure")
    });

    let llfn = decl_internal_rust_fn(ccx, function_type, symbol[]);

    // set an inline hint for all closures
    set_inline_hint(llfn);

    debug!("get_or_create_declaration_if_unboxed_closure(): inserting new \
            closure {} (type {})",
           mono_id,
           ccx.tn().type_to_string(val_ty(llfn)));
    ccx.unboxed_closure_vals().borrow_mut().insert(mono_id, llfn);

    Some(llfn)
}

pub fn trans_unboxed_closure<'blk, 'tcx>(
                             mut bcx: Block<'blk, 'tcx>,
                             decl: &ast::FnDecl,
                             body: &ast::Block,
                             id: ast::NodeId,
                             dest: expr::Dest)
                             -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("closure::trans_unboxed_closure");

    debug!("trans_unboxed_closure()");

    let closure_id = ast_util::local_def(id);
    let llfn = get_or_create_declaration_if_unboxed_closure(
        bcx,
        closure_id,
        bcx.fcx.param_substs).unwrap();

    let function_type = (*bcx.tcx().unboxed_closures.borrow())[closure_id]
                                                              .closure_type
                                                              .clone();
    let function_type = ty::mk_closure(bcx.tcx(), function_type);

    let freevars: Vec<ty::Freevar> =
        ty::with_freevars(bcx.tcx(), id, |fv| fv.iter().map(|&fv| fv).collect());
    let freevar_mode = bcx.tcx().capture_mode(id);

    trans_closure(bcx.ccx(),
                  decl,
                  body,
                  llfn,
                  bcx.fcx.param_substs,
                  id,
                  &[],
                  ty::ty_fn_ret(function_type),
                  ty::ty_fn_abi(function_type),
                  ClosureEnv::new(freevars[],
                                  UnboxedClosure(freevar_mode)));

    // Don't hoist this to the top of the function. It's perfectly legitimate
    // to have a zero-size unboxed closure (in which case dest will be
    // `Ignore`) and we must still generate the closure body.
    let dest_addr = match dest {
        expr::SaveIn(p) => p,
        expr::Ignore => {
            debug!("trans_unboxed_closure() ignoring result");
            return bcx
        }
    };

    let repr = adt::represent_type(bcx.ccx(), node_id_type(bcx, id));

    // Create the closure.
    for (i, freevar) in freevars.iter().enumerate() {
        let datum = expr::trans_local_var(bcx, freevar.def);
        let upvar_slot_dest = adt::trans_field_ptr(bcx,
                                                   &*repr,
                                                   dest_addr,
                                                   0,
                                                   i);
        match freevar_mode {
            ast::CaptureByValue => {
                bcx = datum.store_to(bcx, upvar_slot_dest);
            }
            ast::CaptureByRef => {
                Store(bcx, datum.to_llref(), upvar_slot_dest);
            }
        }
    }
    adt::trans_set_discr(bcx, &*repr, dest_addr, 0);

    bcx
}

pub fn get_wrapper_for_bare_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                         closure_ty: Ty<'tcx>,
                                         def_id: ast::DefId,
                                         fn_ptr: ValueRef,
                                         is_local: bool) -> ValueRef {

    match ccx.closure_bare_wrapper_cache().borrow().get(&fn_ptr) {
        Some(&llval) => return llval,
        None => {}
    }

    let tcx = ccx.tcx();

    debug!("get_wrapper_for_bare_fn(closure_ty={})", closure_ty.repr(tcx));

    let f = match closure_ty.sty {
        ty::ty_closure(ref f) => f,
        _ => {
            ccx.sess().bug(format!("get_wrapper_for_bare_fn: \
                                    expected a closure ty, got {}",
                                    closure_ty.repr(tcx))[]);
        }
    };

    let name = ty::with_path(tcx, def_id, |path| {
        mangle_internal_name_by_path_and_seq(path, "as_closure")
    });
    let llfn = if is_local {
        decl_internal_rust_fn(ccx, closure_ty, name[])
    } else {
        decl_rust_fn(ccx, closure_ty, name[])
    };

    ccx.closure_bare_wrapper_cache().borrow_mut().insert(fn_ptr, llfn);

    // This is only used by statics inlined from a different crate.
    if !is_local {
        // Don't regenerate the wrapper, just reuse the original one.
        return llfn;
    }

    let _icx = push_ctxt("closure::get_wrapper_for_bare_fn");

    let arena = TypedArena::new();
    let empty_param_substs = Substs::trans_empty();
    let fcx = new_fn_ctxt(ccx, llfn, ast::DUMMY_NODE_ID, true, f.sig.0.output,
                          &empty_param_substs, None, &arena);
    let bcx = init_function(&fcx, true, f.sig.0.output);

    let args = create_datums_for_fn_args(&fcx,
                                         ty::ty_fn_args(closure_ty)
                                            []);
    let mut llargs = Vec::new();
    match fcx.llretslotptr.get() {
        Some(llretptr) => {
            assert!(!fcx.needs_ret_allocas);
            llargs.push(llretptr);
        }
        None => {}
    }
    llargs.extend(args.iter().map(|arg| arg.val));

    let retval = Call(bcx, fn_ptr, llargs.as_slice(), None);
    match f.sig.0.output {
        ty::FnConverging(output_type) => {
            if return_type_is_void(ccx, output_type) || fcx.llretslotptr.get().is_some() {
                RetVoid(bcx);
            } else {
                Ret(bcx, retval);
            }
        }
        ty::FnDiverging => {
            RetVoid(bcx);
        }
    }

    // HACK(eddyb) finish_fn cannot be used here, we returned directly.
    debuginfo::clear_source_location(&fcx);
    fcx.cleanup();

    llfn
}

pub fn make_closure_from_bare_fn<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                             closure_ty: Ty<'tcx>,
                                             def_id: ast::DefId,
                                             fn_ptr: ValueRef)
                                             -> DatumBlock<'blk, 'tcx, Expr>  {
    let scratch = rvalue_scratch_datum(bcx, closure_ty, "__adjust");
    let wrapper = get_wrapper_for_bare_fn(bcx.ccx(), closure_ty, def_id, fn_ptr, true);
    fill_fn_pair(bcx, scratch.val, wrapper, C_null(Type::i8p(bcx.ccx())));

    DatumBlock::new(bcx, scratch.to_expr_datum())
}
