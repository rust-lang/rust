// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::abi;
use back::link::mangle_internal_name_by_path_and_seq;
use driver::session::FullDebugInfo;
use lib::llvm::ValueRef;
use middle::moves;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::common::*;
use middle::trans::datum::{Datum, DatumBlock, Expr, Lvalue, rvalue_scratch_datum};
use middle::trans::debuginfo;
use middle::trans::expr;
use middle::trans::type_of::*;
use middle::trans::type_::Type;
use middle::ty;
use util::ppaux::Repr;
use util::ppaux::ty_to_str;

use arena::TypedArena;
use syntax::ast;
use syntax::ast_util;

// ___Good to know (tm)__________________________________________________
//
// The layout of a closure environment in memory is
// roughly as follows:
//
// struct rust_opaque_box {         // see rust_internal.h
//   unsigned ref_count;            // only used for @fn()
//   type_desc *tydesc;             // describes closure_data struct
//   rust_opaque_box *prev;         // (used internally by memory alloc)
//   rust_opaque_box *next;         // (used internally by memory alloc)
//   struct closure_data {
//       type_desc *bound_tdescs[]; // bound descriptors
//       struct {
//         upvar1_t upvar1;
//         ...
//         upvarN_t upvarN;
//       } bound_data;
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
// rust_opaque_box header is always a mutiple of 16-bytes, which is
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

pub enum EnvAction {
    /// Copy the value from this llvm ValueRef into the environment.
    EnvCopy,

    /// Move the value from this llvm ValueRef into the environment.
    EnvMove,

    /// Access by reference (used for stack closures).
    EnvRef
}

pub struct EnvValue {
    action: EnvAction,
    datum: Datum<Lvalue>
}

impl EnvAction {
    pub fn to_str(&self) -> ~str {
        match *self {
            EnvCopy => ~"EnvCopy",
            EnvMove => ~"EnvMove",
            EnvRef => ~"EnvRef"
        }
    }
}

impl EnvValue {
    pub fn to_str(&self, ccx: &CrateContext) -> ~str {
        format!("{}({})", self.action.to_str(), self.datum.to_str(ccx))
    }
}

// Given a closure ty, emits a corresponding tuple ty
pub fn mk_closure_tys(tcx: ty::ctxt,
                      bound_values: &[EnvValue])
                   -> ty::t {
    // determine the types of the values in the env.  Note that this
    // is the actual types that will be stored in the map, not the
    // logical types as the user sees them, so by-ref upvars must be
    // converted to ptrs.
    let bound_tys = bound_values.map(|bv| {
        match bv.action {
            EnvCopy | EnvMove => bv.datum.ty,
            EnvRef => ty::mk_mut_ptr(tcx, bv.datum.ty)
        }
    });
    let cdata_ty = ty::mk_tup(tcx, bound_tys);
    debug!("cdata_ty={}", ty_to_str(tcx, cdata_ty));
    return cdata_ty;
}

fn tuplify_box_ty(tcx: ty::ctxt, t: ty::t) -> ty::t {
    let ptr = ty::mk_imm_ptr(tcx, ty::mk_i8());
    ty::mk_tup(tcx, ~[ty::mk_uint(), ty::mk_nil_ptr(tcx), ptr, ptr, t])
}

fn allocate_cbox<'a>(bcx: &'a Block<'a>,
                     sigil: ast::Sigil,
                     cdata_ty: ty::t)
                     -> Result<'a> {
    let _icx = push_ctxt("closure::allocate_cbox");
    let ccx = bcx.ccx();
    let tcx = ccx.tcx;

    // Allocate and initialize the box:
    match sigil {
        ast::ManagedSigil => {
            tcx.sess.bug("trying to trans allocation of @fn")
        }
        ast::OwnedSigil => {
            malloc_raw(bcx, cdata_ty, heap_exchange_closure)
        }
        ast::BorrowedSigil => {
            let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
            let llbox = alloc_ty(bcx, cbox_ty, "__closure");
            rslt(bcx, llbox)
        }
    }
}

pub struct ClosureResult<'a> {
    llbox: ValueRef,    // llvalue of ptr to closure
    cdata_ty: ty::t,    // type of the closure data
    bcx: &'a Block<'a>  // final bcx
}

// Given a block context and a list of tydescs and values to bind
// construct a closure out of them. If copying is true, it is a
// heap allocated closure that copies the upvars into environment.
// Otherwise, it is stack allocated and copies pointers to the upvars.
pub fn store_environment<'a>(
                         bcx: &'a Block<'a>,
                         bound_values: ~[EnvValue],
                         sigil: ast::Sigil)
                         -> ClosureResult<'a> {
    let _icx = push_ctxt("closure::store_environment");
    let ccx = bcx.ccx();
    let tcx = ccx.tcx;

    // compute the type of the closure
    let cdata_ty = mk_closure_tys(tcx, bound_values);

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
    let Result {bcx: bcx, val: llbox} = allocate_cbox(bcx, sigil, cdata_ty);

    let llbox = PointerCast(bcx, llbox, llboxptr_ty);
    debug!("tuplify_box_ty = {}", ty_to_str(tcx, cbox_ty));

    // Copy expr values into boxed bindings.
    let mut bcx = bcx;
    for (i, bv) in bound_values.move_iter().enumerate() {
        debug!("Copy {} into closure", bv.to_str(ccx));

        if ccx.sess.asm_comments() {
            add_comment(bcx, format!("Copy {} into closure",
                                  bv.to_str(ccx)));
        }

        let bound_data = GEPi(bcx, llbox, [0u, abi::box_field_body, i]);

        match bv.action {
            EnvCopy | EnvMove => {
                bcx = bv.datum.store_to(bcx, bound_data);
            }
            EnvRef => {
                Store(bcx, bv.datum.to_llref(), bound_data);
            }
        }
    }

    ClosureResult { llbox: llbox, cdata_ty: cdata_ty, bcx: bcx }
}

// Given a context and a list of upvars, build a closure. This just
// collects the upvars and packages them up for store_environment.
fn build_closure<'a>(bcx0: &'a Block<'a>,
                     cap_vars: &[moves::CaptureVar],
                     sigil: ast::Sigil)
                     -> ClosureResult<'a> {
    let _icx = push_ctxt("closure::build_closure");

    // If we need to, package up the iterator body to call
    let bcx = bcx0;

    // Package up the captured upvars
    let mut env_vals = ~[];
    for cap_var in cap_vars.iter() {
        debug!("Building closure: captured variable {:?}", *cap_var);
        let datum = expr::trans_local_var(bcx, cap_var.def);
        match cap_var.mode {
            moves::CapRef => {
                assert_eq!(sigil, ast::BorrowedSigil);
                env_vals.push(EnvValue {action: EnvRef,
                                        datum: datum});
            }
            moves::CapCopy => {
                env_vals.push(EnvValue {action: EnvCopy,
                                        datum: datum});
            }
            moves::CapMove => {
                env_vals.push(EnvValue {action: EnvMove,
                                        datum: datum});
            }
        }
    }

    return store_environment(bcx, env_vals, sigil);
}

// Given an enclosing block context, a new function context, a closure type,
// and a list of upvars, generate code to load and populate the environment
// with the upvars and type descriptors.
fn load_environment<'a>(bcx: &'a Block<'a>, cdata_ty: ty::t,
                        cap_vars: &[moves::CaptureVar],
                        sigil: ast::Sigil) -> &'a Block<'a> {
    let _icx = push_ctxt("closure::load_environment");

    // Don't bother to create the block if there's nothing to load
    if cap_vars.len() == 0 {
        return bcx;
    }

    // Load a pointer to the closure data, skipping over the box header:
    let llcdata = at_box_body(bcx, cdata_ty, bcx.fcx.llenv.unwrap());

    // Store the pointer to closure data in an alloca for debug info because that's what the
    // llvm.dbg.declare intrinsic expects
    let env_pointer_alloca = if bcx.ccx().sess.opts.debuginfo == FullDebugInfo {
        let alloc = alloc_ty(bcx, ty::mk_mut_ptr(bcx.tcx(), cdata_ty), "__debuginfo_env_ptr");
        Store(bcx, llcdata, alloc);
        Some(alloc)
    } else {
        None
    };

    // Populate the upvars from the environment
    let mut i = 0u;
    for cap_var in cap_vars.iter() {
        let mut upvarptr = GEPi(bcx, llcdata, [0u, i]);
        match sigil {
            ast::BorrowedSigil => { upvarptr = Load(bcx, upvarptr); }
            ast::ManagedSigil | ast::OwnedSigil => {}
        }
        let def_id = ast_util::def_id_of_def(cap_var.def);

        {
            let mut llupvars = bcx.fcx.llupvars.borrow_mut();
            llupvars.get().insert(def_id.node, upvarptr);
        }

        for &env_pointer_alloca in env_pointer_alloca.iter() {
            debuginfo::create_captured_var_metadata(
                bcx,
                def_id.node,
                cdata_ty,
                env_pointer_alloca,
                i,
                sigil,
                cap_var.span);
        }

        i += 1u;
    }

    bcx
}

fn fill_fn_pair(bcx: &Block, pair: ValueRef, llfn: ValueRef, llenvptr: ValueRef) {
    Store(bcx, llfn, GEPi(bcx, pair, [0u, abi::fn_field_code]));
    let llenvptr = PointerCast(bcx, llenvptr, Type::i8p());
    Store(bcx, llenvptr, GEPi(bcx, pair, [0u, abi::fn_field_box]));
}

pub fn trans_expr_fn<'a>(
                     bcx: &'a Block<'a>,
                     sigil: ast::Sigil,
                     decl: &ast::FnDecl,
                     body: &ast::Block,
                     id: ast::NodeId,
                     dest: expr::Dest)
                     -> &'a Block<'a> {
    /*!
     *
     * Translates the body of a closure expression.
     *
     * - `sigil`
     * - `decl`
     * - `body`
     * - `id`: The id of the closure expression.
     * - `cap_clause`: information about captured variables, if any.
     * - `dest`: where to write the closure value, which must be a
         (fn ptr, env) pair
     */

    let _icx = push_ctxt("closure::trans_expr_fn");

    let dest_addr = match dest {
        expr::SaveIn(p) => p,
        expr::Ignore => {
            return bcx; // closure construction is non-side-effecting
        }
    };

    let ccx = bcx.ccx();
    let fty = node_id_type(bcx, id);
    let f = match ty::get(fty).sty {
        ty::ty_closure(ref f) => f,
        _ => fail!("expected closure")
    };

    let tcx = bcx.tcx();
    let s = tcx.map.with_path(id, |path| {
        mangle_internal_name_by_path_and_seq(path, "closure")
    });
    let llfn = decl_internal_rust_fn(ccx, true, f.sig.inputs, f.sig.output, s);

    // set an inline hint for all closures
    set_inline_hint(llfn);

    let cap_vars = {
        let capture_map = ccx.maps.capture_map.borrow();
        capture_map.get().get_copy(&id)
    };
    let ClosureResult {llbox, cdata_ty, bcx} = build_closure(bcx, *cap_vars.borrow(), sigil);
    trans_closure(ccx, decl, body, llfn,
                  bcx.fcx.param_substs, id,
                  [], ty::ty_fn_ret(fty),
                  |bcx| load_environment(bcx, cdata_ty, *cap_vars.borrow(), sigil));
    fill_fn_pair(bcx, dest_addr, llfn, llbox);

    bcx
}

pub fn get_wrapper_for_bare_fn(ccx: @CrateContext,
                               closure_ty: ty::t,
                               def: ast::Def,
                               fn_ptr: ValueRef,
                               is_local: bool) -> ValueRef {

    let def_id = match def {
        ast::DefFn(did, _) | ast::DefStaticMethod(did, _, _) |
        ast::DefVariant(_, did, _) | ast::DefStruct(did) => did,
        _ => {
            ccx.sess.bug(format!("get_wrapper_for_bare_fn: \
                                  expected a statically resolved fn, got {:?}",
                                  def));
        }
    };

    {
        let cache = ccx.closure_bare_wrapper_cache.borrow();
        match cache.get().find(&fn_ptr) {
            Some(&llval) => return llval,
            None => {}
        }
    }

    let tcx = ccx.tcx;

    debug!("get_wrapper_for_bare_fn(closure_ty={})", closure_ty.repr(tcx));

    let f = match ty::get(closure_ty).sty {
        ty::ty_closure(ref f) => f,
        _ => {
            ccx.sess.bug(format!("get_wrapper_for_bare_fn: \
                                  expected a closure ty, got {}",
                                  closure_ty.repr(tcx)));
        }
    };

    let name = ty::with_path(tcx, def_id, |path| {
        mangle_internal_name_by_path_and_seq(path, "as_closure")
    });
    let llfn = if is_local {
        decl_internal_rust_fn(ccx, true, f.sig.inputs, f.sig.output, name)
    } else {
        decl_rust_fn(ccx, true, f.sig.inputs, f.sig.output, name)
    };

    {
        let mut cache = ccx.closure_bare_wrapper_cache.borrow_mut();
        cache.get().insert(fn_ptr, llfn);
    }

    // This is only used by statics inlined from a different crate.
    if !is_local {
        // Don't regenerate the wrapper, just reuse the original one.
        return llfn;
    }

    let _icx = push_ctxt("closure::get_wrapper_for_bare_fn");

    let arena = TypedArena::new();
    let fcx = new_fn_ctxt(ccx, llfn, -1, true, f.sig.output, None, None, &arena);
    init_function(&fcx, true, f.sig.output, None);
    let bcx = fcx.entry_bcx.get().unwrap();

    let args = create_datums_for_fn_args(&fcx, ty::ty_fn_args(closure_ty));
    let mut llargs = ~[];
    match fcx.llretptr.get() {
        Some(llretptr) => {
            llargs.push(llretptr);
        }
        None => {}
    }
    llargs.extend(&mut args.iter().map(|arg| arg.val));

    let retval = Call(bcx, fn_ptr, llargs, []);
    if type_is_zero_size(ccx, f.sig.output) || fcx.llretptr.get().is_some() {
        RetVoid(bcx);
    } else {
        Ret(bcx, retval);
    }

    // HACK(eddyb) finish_fn cannot be used here, we returned directly.
    debuginfo::clear_source_location(&fcx);
    fcx.cleanup();

    llfn
}

pub fn make_closure_from_bare_fn<'a>(bcx: &'a Block<'a>,
                                     closure_ty: ty::t,
                                     def: ast::Def,
                                     fn_ptr: ValueRef)
                                     -> DatumBlock<'a, Expr>  {
    let scratch = rvalue_scratch_datum(bcx, closure_ty, "__adjust");
    let wrapper = get_wrapper_for_bare_fn(bcx.ccx(), closure_ty, def, fn_ptr, true);
    fill_fn_pair(bcx, scratch.val, wrapper, C_null(Type::i8p()));

    DatumBlock(bcx, scratch.to_expr_datum())
}
