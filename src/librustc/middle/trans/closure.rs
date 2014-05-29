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
use driver::config::FullDebugInfo;
use lib::llvm::ValueRef;
use middle::freevars;
use middle::lang_items::ClosureExchangeMallocFnLangItem;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::common::*;
use middle::trans::datum::{Datum, DatumBlock, Expr, Lvalue, rvalue_scratch_datum};
use middle::trans::debuginfo;
use middle::trans::expr;
use middle::trans::machine::llsize_of;
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

pub struct EnvValue {
    action: freevars::CaptureMode,
    datum: Datum<Lvalue>
}

impl EnvValue {
    pub fn to_str(&self, ccx: &CrateContext) -> String {
        format!("{}({})", self.action, self.datum.to_str(ccx))
    }
}

// Given a closure ty, emits a corresponding tuple ty
pub fn mk_closure_tys(tcx: &ty::ctxt,
                      bound_values: &[EnvValue])
                   -> ty::t {
    // determine the types of the values in the env.  Note that this
    // is the actual types that will be stored in the map, not the
    // logical types as the user sees them, so by-ref upvars must be
    // converted to ptrs.
    let bound_tys = bound_values.iter().map(|bv| {
        match bv.action {
            freevars::CaptureByValue => bv.datum.ty,
            freevars::CaptureByRef => ty::mk_mut_ptr(tcx, bv.datum.ty)
        }
    }).collect();
    let cdata_ty = ty::mk_tup(tcx, bound_tys);
    debug!("cdata_ty={}", ty_to_str(tcx, cdata_ty));
    return cdata_ty;
}

fn tuplify_box_ty(tcx: &ty::ctxt, t: ty::t) -> ty::t {
    let ptr = ty::mk_imm_ptr(tcx, ty::mk_i8());
    ty::mk_tup(tcx, vec!(ty::mk_uint(), ty::mk_nil_ptr(tcx), ptr, ptr, t))
}

fn allocate_cbox<'a>(bcx: &'a Block<'a>,
                     store: ty::TraitStore,
                     cdata_ty: ty::t)
                     -> Result<'a> {
    let _icx = push_ctxt("closure::allocate_cbox");
    let tcx = bcx.tcx();

    // Allocate and initialize the box:
    match store {
        ty::UniqTraitStore => {
            let ty = type_of(bcx.ccx(), cdata_ty);
            let size = llsize_of(bcx.ccx(), ty);
            // we treat proc as @ here, which isn't ideal
            malloc_raw_dyn_managed(bcx, cdata_ty, ClosureExchangeMallocFnLangItem, size)
        }
        ty::RegionTraitStore(..) => {
            let cbox_ty = tuplify_box_ty(tcx, cdata_ty);
            let llbox = alloc_ty(bcx, cbox_ty, "__closure");
            Result::new(bcx, llbox)
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
                         bound_values: Vec<EnvValue> ,
                         store: ty::TraitStore)
                         -> ClosureResult<'a> {
    let _icx = push_ctxt("closure::store_environment");
    let ccx = bcx.ccx();
    let tcx = ccx.tcx();

    // compute the type of the closure
    let cdata_ty = mk_closure_tys(tcx, bound_values.as_slice());

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
    let Result {bcx: bcx, val: llbox} = allocate_cbox(bcx, store, cdata_ty);

    let llbox = PointerCast(bcx, llbox, llboxptr_ty);
    debug!("tuplify_box_ty = {}", ty_to_str(tcx, cbox_ty));

    // Copy expr values into boxed bindings.
    let mut bcx = bcx;
    for (i, bv) in bound_values.move_iter().enumerate() {
        debug!("Copy {} into closure", bv.to_str(ccx));

        if ccx.sess().asm_comments() {
            add_comment(bcx, format!("Copy {} into closure",
                                     bv.to_str(ccx)).as_slice());
        }

        let bound_data = GEPi(bcx, llbox, [0u, abi::box_field_body, i]);

        match bv.action {
            freevars::CaptureByValue => {
                bcx = bv.datum.store_to(bcx, bound_data);
            }
            freevars::CaptureByRef => {
                Store(bcx, bv.datum.to_llref(), bound_data);
            }
        }
    }

    ClosureResult { llbox: llbox, cdata_ty: cdata_ty, bcx: bcx }
}

// Given a context and a list of upvars, build a closure. This just
// collects the upvars and packages them up for store_environment.
fn build_closure<'a>(bcx0: &'a Block<'a>,
                     freevar_mode: freevars::CaptureMode,
                     freevars: &Vec<freevars::freevar_entry>,
                     store: ty::TraitStore)
                     -> ClosureResult<'a>
{
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
fn load_environment<'a>(bcx: &'a Block<'a>,
                        cdata_ty: ty::t,
                        freevars: &Vec<freevars::freevar_entry>,
                        store: ty::TraitStore)
                        -> &'a Block<'a> {
    let _icx = push_ctxt("closure::load_environment");

    // Don't bother to create the block if there's nothing to load
    if freevars.len() == 0 {
        return bcx;
    }

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
        let mut upvarptr = GEPi(bcx, llcdata, [0u, i]);
        match store {
            ty::RegionTraitStore(..) => { upvarptr = Load(bcx, upvarptr); }
            ty::UniqTraitStore => {}
        }
        let def_id = ast_util::def_id_of_def(freevar.def);

        bcx.fcx.llupvars.borrow_mut().insert(def_id.node, upvarptr);

        for &env_pointer_alloca in env_pointer_alloca.iter() {
            debuginfo::create_captured_var_metadata(
                bcx,
                def_id.node,
                cdata_ty,
                env_pointer_alloca,
                i,
                store,
                freevar.span);
        }

        i += 1u;
    }

    bcx
}

fn fill_fn_pair(bcx: &Block, pair: ValueRef, llfn: ValueRef, llenvptr: ValueRef) {
    Store(bcx, llfn, GEPi(bcx, pair, [0u, abi::fn_field_code]));
    let llenvptr = PointerCast(bcx, llenvptr, Type::i8p(bcx.ccx()));
    Store(bcx, llenvptr, GEPi(bcx, pair, [0u, abi::fn_field_box]));
}

pub fn trans_expr_fn<'a>(
                     bcx: &'a Block<'a>,
                     store: ty::TraitStore,
                     decl: &ast::FnDecl,
                     body: &ast::Block,
                     id: ast::NodeId,
                     dest: expr::Dest)
                     -> &'a Block<'a> {
    /*!
     *
     * Translates the body of a closure expression.
     *
     * - `store`
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
    let tcx = bcx.tcx();
    let fty = node_id_type(bcx, id);
    let s = tcx.map.with_path(id, |path| {
        mangle_internal_name_by_path_and_seq(path, "closure")
    });
    let llfn = decl_internal_rust_fn(ccx, fty, s.as_slice());

    // set an inline hint for all closures
    set_inline_hint(llfn);

    let freevar_mode = freevars::get_capture_mode(tcx, id);
    let freevars: Vec<freevars::freevar_entry> =
        freevars::with_freevars(
            tcx, id,
            |fv| fv.iter().map(|&fv| fv).collect());

    let ClosureResult {llbox, cdata_ty, bcx} =
        build_closure(bcx, freevar_mode, &freevars, store);
    trans_closure(ccx, decl, body, llfn,
                  bcx.fcx.param_substs, id,
                  [], ty::ty_fn_ret(fty),
                  |bcx| load_environment(bcx, cdata_ty, &freevars, store));
    fill_fn_pair(bcx, dest_addr, llfn, llbox);
    bcx
}

pub fn get_wrapper_for_bare_fn(ccx: &CrateContext,
                               closure_ty: ty::t,
                               def: ast::Def,
                               fn_ptr: ValueRef,
                               is_local: bool) -> ValueRef {

    let def_id = match def {
        ast::DefFn(did, _) | ast::DefStaticMethod(did, _, _) |
        ast::DefVariant(_, did, _) | ast::DefStruct(did) => did,
        _ => {
            ccx.sess().bug(format!("get_wrapper_for_bare_fn: \
                                    expected a statically resolved fn, got \
                                    {:?}",
                                    def).as_slice());
        }
    };

    match ccx.closure_bare_wrapper_cache.borrow().find(&fn_ptr) {
        Some(&llval) => return llval,
        None => {}
    }

    let tcx = ccx.tcx();

    debug!("get_wrapper_for_bare_fn(closure_ty={})", closure_ty.repr(tcx));

    let f = match ty::get(closure_ty).sty {
        ty::ty_closure(ref f) => f,
        _ => {
            ccx.sess().bug(format!("get_wrapper_for_bare_fn: \
                                    expected a closure ty, got {}",
                                    closure_ty.repr(tcx)).as_slice());
        }
    };

    let name = ty::with_path(tcx, def_id, |path| {
        mangle_internal_name_by_path_and_seq(path, "as_closure")
    });
    let llfn = if is_local {
        decl_internal_rust_fn(ccx, closure_ty, name.as_slice())
    } else {
        decl_rust_fn(ccx, closure_ty, name.as_slice())
    };

    ccx.closure_bare_wrapper_cache.borrow_mut().insert(fn_ptr, llfn);

    // This is only used by statics inlined from a different crate.
    if !is_local {
        // Don't regenerate the wrapper, just reuse the original one.
        return llfn;
    }

    let _icx = push_ctxt("closure::get_wrapper_for_bare_fn");

    let arena = TypedArena::new();
    let fcx = new_fn_ctxt(ccx, llfn, -1, true, f.sig.output, None, None, &arena);
    init_function(&fcx, true, f.sig.output);
    let bcx = fcx.entry_bcx.borrow().clone().unwrap();

    let args = create_datums_for_fn_args(&fcx,
                                         ty::ty_fn_args(closure_ty)
                                            .as_slice());
    let mut llargs = Vec::new();
    match fcx.llretptr.get() {
        Some(llretptr) => {
            llargs.push(llretptr);
        }
        None => {}
    }
    llargs.extend(args.iter().map(|arg| arg.val));

    let retval = Call(bcx, fn_ptr, llargs.as_slice(), []);
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
    fill_fn_pair(bcx, scratch.val, wrapper, C_null(Type::i8p(bcx.ccx())));

    DatumBlock(bcx, scratch.to_expr_datum())
}
