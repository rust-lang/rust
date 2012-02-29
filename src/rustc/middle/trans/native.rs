import driver::session::session;
import syntax::codemap::span;
import ctypes::c_uint;
import front::attr;
import lib::llvm::{ llvm, TypeRef, ValueRef };
import syntax::ast;
import back::link;
import common::*;
import build::*;
import base::*;
import type_of::*;

export link_name, trans_native_mod, register_crust_fn, trans_crust_fn;

fn link_name(i: @ast::native_item) -> str {
    alt attr::get_meta_item_value_str_by_name(i.attrs, "link_name") {
      none { ret i.ident; }
      option::some(ln) { ret ln; }
    }
}

type c_stack_tys = {
    arg_tys: [TypeRef],
    ret_ty: TypeRef,
    ret_def: bool,
    bundle_ty: TypeRef,
    shim_fn_ty: TypeRef
};

fn c_arg_and_ret_lltys(ccx: crate_ctxt,
                       id: ast::node_id) -> ([TypeRef], TypeRef, ty::t) {
    alt ty::get(ty::node_id_to_type(ccx.tcx, id)).struct {
      ty::ty_fn({inputs: arg_tys, output: ret_ty, _}) {
        let llargtys = type_of_explicit_args(ccx, arg_tys);
        let llretty = type_of::type_of(ccx, ret_ty);
        (llargtys, llretty, ret_ty)
      }
      _ { ccx.sess.bug("c_arg_and_ret_lltys called on non-function type"); }
    }
}

fn c_stack_tys(ccx: crate_ctxt,
               id: ast::node_id) -> @c_stack_tys {
    let (llargtys, llretty, ret_ty) = c_arg_and_ret_lltys(ccx, id);
    let bundle_ty = T_struct(llargtys + [T_ptr(llretty)]);
    ret @{
        arg_tys: llargtys,
        ret_ty: llretty,
        ret_def: !ty::type_is_bot(ret_ty) && !ty::type_is_nil(ret_ty),
        bundle_ty: bundle_ty,
        shim_fn_ty: T_fn([T_ptr(bundle_ty)], T_void())
    };
}

type shim_arg_builder = fn(bcx: block, tys: @c_stack_tys,
                           llargbundle: ValueRef) -> [ValueRef];

type shim_ret_builder = fn(bcx: block, tys: @c_stack_tys,
                           llargbundle: ValueRef, llretval: ValueRef);

fn build_shim_fn_(ccx: crate_ctxt,
                  shim_name: str,
                  llbasefn: ValueRef,
                  tys: @c_stack_tys,
                  cc: lib::llvm::CallConv,
                  arg_builder: shim_arg_builder,
                  ret_builder: shim_ret_builder) -> ValueRef {

    let llshimfn = decl_internal_cdecl_fn(
        ccx.llmod, shim_name, tys.shim_fn_ty);

    // Declare the body of the shim function:
    let fcx = new_fn_ctxt(ccx, [], llshimfn, none);
    let bcx = top_scope_block(fcx, none);
    let lltop = bcx.llbb;
    let llargbundle = llvm::LLVMGetParam(llshimfn, 0 as c_uint);
    let llargvals = arg_builder(bcx, tys, llargbundle);

    // Create the call itself and store the return value:
    let llretval = CallWithConv(bcx, llbasefn,
                                llargvals, cc); // r

    ret_builder(bcx, tys, llargbundle, llretval);

    build_return(bcx);
    finish_fn(fcx, lltop);

    ret llshimfn;
}

type wrap_arg_builder = fn(bcx: block, tys: @c_stack_tys,
                           llwrapfn: ValueRef,
                           llargbundle: ValueRef);

type wrap_ret_builder = fn(bcx: block, tys: @c_stack_tys,
                           llargbundle: ValueRef);

fn build_wrap_fn_(ccx: crate_ctxt,
                  tys: @c_stack_tys,
                  llshimfn: ValueRef,
                  llwrapfn: ValueRef,
                  shim_upcall: ValueRef,
                  arg_builder: wrap_arg_builder,
                  ret_builder: wrap_ret_builder) {

    let fcx = new_fn_ctxt(ccx, [], llwrapfn, none);
    let bcx = top_scope_block(fcx, none);
    let lltop = bcx.llbb;

    // Allocate the struct and write the arguments into it.
    let llargbundle = alloca(bcx, tys.bundle_ty);
    arg_builder(bcx, tys, llwrapfn, llargbundle);

    // Create call itself.
    let llshimfnptr = PointerCast(bcx, llshimfn, T_ptr(T_i8()));
    let llrawargbundle = PointerCast(bcx, llargbundle, T_ptr(T_i8()));
    Call(bcx, shim_upcall, [llrawargbundle, llshimfnptr]);
    ret_builder(bcx, tys, llargbundle);

    tie_up_header_blocks(fcx, lltop);

    // Make sure our standard return block (that we didn't use) is terminated
    let ret_cx = raw_block(fcx, fcx.llreturn);
    Unreachable(ret_cx);
}

// For each native function F, we generate a wrapper function W and a shim
// function S that all work together.  The wrapper function W is the function
// that other rust code actually invokes.  Its job is to marshall the
// arguments into a struct.  It then uses a small bit of assembly to switch
// over to the C stack and invoke the shim function.  The shim function S then
// unpacks the arguments from the struct and invokes the actual function F
// according to its specified calling convention.
//
// Example: Given a native c-stack function F(x: X, y: Y) -> Z,
// we generate a wrapper function W that looks like:
//
//    void W(Z* dest, void *env, X x, Y y) {
//        struct { X x; Y y; Z *z; } args = { x, y, z };
//        call_on_c_stack_shim(S, &args);
//    }
//
// The shim function S then looks something like:
//
//     void S(struct { X x; Y y; Z *z; } *args) {
//         *args->z = F(args->x, args->y);
//     }
//
// However, if the return type of F is dynamically sized or of aggregate type,
// the shim function looks like:
//
//     void S(struct { X x; Y y; Z *z; } *args) {
//         F(args->z, args->x, args->y);
//     }
//
// Note: on i386, the layout of the args struct is generally the same as the
// desired layout of the arguments on the C stack.  Therefore, we could use
// upcall_alloc_c_stack() to allocate the `args` structure and switch the
// stack pointer appropriately to avoid a round of copies.  (In fact, the shim
// function itself is unnecessary). We used to do this, in fact, and will
// perhaps do so in the future.
fn trans_native_mod(ccx: crate_ctxt,
                    native_mod: ast::native_mod, abi: ast::native_abi) {
    fn build_shim_fn(ccx: crate_ctxt,
                     native_item: @ast::native_item,
                     tys: @c_stack_tys,
                     cc: lib::llvm::CallConv) -> ValueRef {

        fn build_args(bcx: block, tys: @c_stack_tys,
                      llargbundle: ValueRef) -> [ValueRef] {
            let llargvals = [];
            let i = 0u;
            let n = vec::len(tys.arg_tys);
            while i < n {
                let llargval = load_inbounds(bcx, llargbundle, [0, i as int]);
                llargvals += [llargval];
                i += 1u;
            }
            ret llargvals;
        }

        fn build_ret(bcx: block, tys: @c_stack_tys,
                     llargbundle: ValueRef, llretval: ValueRef)  {
            if tys.ret_def {
                let n = vec::len(tys.arg_tys);
                // R** llretptr = &args->r;
                let llretptr = GEPi(bcx, llargbundle, [0, n as int]);
                // R* llretloc = *llretptr; /* (args->r) */
                let llretloc = Load(bcx, llretptr);
                // *args->r = r;
                Store(bcx, llretval, llretloc);
            }
        }

        let lname = link_name(native_item);
        // Declare the "prototype" for the base function F:
        let llbasefnty = T_fn(tys.arg_tys, tys.ret_ty);
        let llbasefn = decl_fn(ccx.llmod, lname, cc, llbasefnty);
        // Name the shim function
        let shim_name = lname + "__c_stack_shim";
        ret build_shim_fn_(ccx, shim_name, llbasefn, tys, cc,
                           build_args, build_ret);
    }

    fn build_wrap_fn(ccx: crate_ctxt,
                     tys: @c_stack_tys,
                     num_tps: uint,
                     llshimfn: ValueRef,
                     llwrapfn: ValueRef) {

        fn build_args(bcx: block, tys: @c_stack_tys,
                      llwrapfn: ValueRef, llargbundle: ValueRef,
                      num_tps: uint) {
            let i = 0u, n = vec::len(tys.arg_tys);
            let implicit_args = first_tp_arg + num_tps; // ret + env
            while i < n {
                let llargval = llvm::LLVMGetParam(
                    llwrapfn,
                    (i + implicit_args) as c_uint);
                store_inbounds(bcx, llargval, llargbundle, [0, i as int]);
                i += 1u;
            }
            let llretptr = llvm::LLVMGetParam(llwrapfn, 0 as c_uint);
            store_inbounds(bcx, llretptr, llargbundle, [0, n as int]);
        }

        fn build_ret(bcx: block, _tys: @c_stack_tys,
                     _llargbundle: ValueRef) {
            RetVoid(bcx);
        }

        build_wrap_fn_(ccx, tys, llshimfn, llwrapfn,
                       ccx.upcalls.call_shim_on_c_stack,
                       bind build_args(_, _ ,_ , _, num_tps),
                       build_ret);
    }

    let cc = lib::llvm::CCallConv;
    alt abi {
      ast::native_abi_rust_intrinsic { ret; }
      ast::native_abi_cdecl { cc = lib::llvm::CCallConv; }
      ast::native_abi_stdcall { cc = lib::llvm::X86StdcallCallConv; }
    }

    for native_item in native_mod.items {
      alt native_item.node {
        ast::native_item_fn(fn_decl, tps) {
          let id = native_item.id;
          let tys = c_stack_tys(ccx, id);
          alt ccx.item_ids.find(id) {
            some(llwrapfn) {
              let llshimfn = build_shim_fn(ccx, native_item, tys, cc);
              build_wrap_fn(ccx, tys, vec::len(tps), llshimfn, llwrapfn);
            }
            none {
              ccx.sess.span_bug(
                  native_item.span,
                  "unbound function item in trans_native_mod");
            }
          }
        }
      }
    }
}

fn trans_crust_fn(ccx: crate_ctxt, path: ast_map::path, decl: ast::fn_decl,
                  body: ast::blk, llwrapfn: ValueRef, id: ast::node_id) {

    fn build_rust_fn(ccx: crate_ctxt, path: ast_map::path,
                     decl: ast::fn_decl, body: ast::blk,
                     id: ast::node_id) -> ValueRef {
        let t = ty::node_id_to_type(ccx.tcx, id);
        let ps = link::mangle_internal_name_by_path(
            ccx, path + [ast_map::path_name("__rust_abi")]);
        let llty = type_of_fn_from_ty(ccx, t, []);
        let llfndecl = decl_internal_cdecl_fn(ccx.llmod, ps, llty);
        trans_fn(ccx, path, decl, body, llfndecl, no_self, [], none, id);
        ret llfndecl;
    }

    fn build_shim_fn(ccx: crate_ctxt, path: ast_map::path,
                     llrustfn: ValueRef, tys: @c_stack_tys) -> ValueRef {

        fn build_args(bcx: block, tys: @c_stack_tys,
                      llargbundle: ValueRef) -> [ValueRef] {
            let llargvals = [];
            let i = 0u;
            let n = vec::len(tys.arg_tys);
            let llretptr = load_inbounds(bcx, llargbundle, [0, n as int]);
            llargvals += [llretptr];
            let llenvptr = C_null(T_opaque_box_ptr(bcx.ccx()));
            llargvals += [llenvptr];
            while i < n {
                let llargval = load_inbounds(bcx, llargbundle, [0, i as int]);
                llargvals += [llargval];
                i += 1u;
            }
            ret llargvals;
        }

        fn build_ret(_bcx: block, _tys: @c_stack_tys,
                     _llargbundle: ValueRef, _llretval: ValueRef)  {
            // Nop. The return pointer in the Rust ABI function
            // is wired directly into the return slot in the shim struct
        }

        let shim_name = link::mangle_internal_name_by_path(
            ccx, path + [ast_map::path_name("__rust_stack_shim")]);
        ret build_shim_fn_(ccx, shim_name, llrustfn, tys,
                           lib::llvm::CCallConv,
                           build_args, build_ret);
    }

    fn build_wrap_fn(ccx: crate_ctxt, llshimfn: ValueRef,
                     llwrapfn: ValueRef, tys: @c_stack_tys) {

        fn build_args(bcx: block, tys: @c_stack_tys,
                      llwrapfn: ValueRef, llargbundle: ValueRef) {
            let llretptr = alloca(bcx, tys.ret_ty);
            let i = 0u, n = vec::len(tys.arg_tys);
            while i < n {
                let llargval = llvm::LLVMGetParam(
                    llwrapfn, i as c_uint);
                store_inbounds(bcx, llargval, llargbundle, [0, i as int]);
                i += 1u;
            }
            store_inbounds(bcx, llretptr, llargbundle, [0, n as int]);
        }

        fn build_ret(bcx: block, tys: @c_stack_tys,
                     llargbundle: ValueRef) {
            let n = vec::len(tys.arg_tys);
            let llretval = load_inbounds(bcx, llargbundle, [0, n as int]);
            let llretval = Load(bcx, llretval);
            Ret(bcx, llretval);
        }

        build_wrap_fn_(ccx, tys, llshimfn, llwrapfn,
                       ccx.upcalls.call_shim_on_rust_stack,
                       build_args, build_ret);
    }

    let tys = c_stack_tys(ccx, id);
    // The internal Rust ABI function - runs on the Rust stack
    let llrustfn = build_rust_fn(ccx, path, decl, body, id);
    // The internal shim function - runs on the Rust stack
    let llshimfn = build_shim_fn(ccx, path, llrustfn, tys);
    // The external C function - runs on the C stack
    build_wrap_fn(ccx, llshimfn, llwrapfn, tys)
}

fn register_crust_fn(ccx: crate_ctxt, sp: span,
                     path: ast_map::path, node_id: ast::node_id) {
    let t = ty::node_id_to_type(ccx.tcx, node_id);
    let (llargtys, llretty, _) = c_arg_and_ret_lltys(ccx, node_id);
    let llfty = T_fn(llargtys, llretty);
    register_fn_fuller(ccx, sp, path, "crust fn", node_id,
                       t, lib::llvm::CCallConv, llfty);
}