import driver::session::session;
import ctypes::c_uint;
import front::attr;
import lib::llvm::{ llvm, TypeRef, ValueRef };
import syntax::ast;
import common::*;
import build::*;
import base::*;

export link_name, trans_native_mod;

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
    base_fn_ty: TypeRef,
    bundle_ty: TypeRef,
    shim_fn_ty: TypeRef
};

fn c_stack_tys(ccx: @crate_ctxt,
               id: ast::node_id) -> @c_stack_tys {
    alt ty::get(ty::node_id_to_type(ccx.tcx, id)).struct {
      ty::ty_fn({inputs: arg_tys, output: ret_ty, _}) {
        let llargtys = type_of_explicit_args(ccx, arg_tys);
        let llretty = type_of(ccx, ret_ty);
        let bundle_ty = T_struct(llargtys + [T_ptr(llretty)]);
        ret @{
            arg_tys: llargtys,
            ret_ty: llretty,
            ret_def: !ty::type_is_bot(ret_ty) && !ty::type_is_nil(ret_ty),
            base_fn_ty: T_fn(llargtys, llretty),
            bundle_ty: bundle_ty,
            shim_fn_ty: T_fn([T_ptr(bundle_ty)], T_void())
        };
      }
      _ {
          // Precondition?
          ccx.tcx.sess.bug("c_stack_tys called on non-function type");
      }
    }
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
fn trans_native_mod(ccx: @crate_ctxt,
                    native_mod: ast::native_mod, abi: ast::native_abi) {
    fn build_shim_fn(ccx: @crate_ctxt,
                     native_item: @ast::native_item,
                     tys: @c_stack_tys,
                     cc: lib::llvm::CallConv) -> ValueRef {
        let lname = link_name(native_item);

        // Declare the "prototype" for the base function F:
        let llbasefn = decl_fn(ccx.llmod, lname, cc, tys.base_fn_ty);

        // Create the shim function:
        let shim_name = lname + "__c_stack_shim";
        let llshimfn = decl_internal_cdecl_fn(
            ccx.llmod, shim_name, tys.shim_fn_ty);

        // Declare the body of the shim function:
        let fcx = new_fn_ctxt(ccx, [], llshimfn, none);
        let bcx = new_top_block_ctxt(fcx, none);
        let lltop = bcx.llbb;
        let llargbundle = llvm::LLVMGetParam(llshimfn, 0 as c_uint);
        let i = 0u, n = vec::len(tys.arg_tys);
        let llargvals = [];
        while i < n {
            let llargval = load_inbounds(bcx, llargbundle, [0, i as int]);
            llargvals += [llargval];
            i += 1u;
        }

        // Create the call itself and store the return value:
        let llretval = CallWithConv(bcx, llbasefn,
                                    llargvals, cc); // r
        if tys.ret_def {
            // R** llretptr = &args->r;
            let llretptr = GEPi(bcx, llargbundle, [0, n as int]);
            // R* llretloc = *llretptr; /* (args->r) */
            let llretloc = Load(bcx, llretptr);
            // *args->r = r;
            Store(bcx, llretval, llretloc);
        }

        // Finish up:
        build_return(bcx);
        finish_fn(fcx, lltop);

        ret llshimfn;
    }

    fn build_wrap_fn(ccx: @crate_ctxt,
                     tys: @c_stack_tys,
                     num_tps: uint,
                     llshimfn: ValueRef,
                     llwrapfn: ValueRef) {
        let fcx = new_fn_ctxt(ccx, [], llwrapfn, none);
        let bcx = new_top_block_ctxt(fcx, none);
        let lltop = bcx.llbb;

        // Allocate the struct and write the arguments into it.
        let llargbundle = alloca(bcx, tys.bundle_ty);
        let i = 0u, n = vec::len(tys.arg_tys);
        let implicit_args = 2u + num_tps; // ret + env
        while i < n {
            let llargval = llvm::LLVMGetParam(llwrapfn,
                                              (i + implicit_args) as c_uint);
            store_inbounds(bcx, llargval, llargbundle, [0, i as int]);
            i += 1u;
        }
        let llretptr = llvm::LLVMGetParam(llwrapfn, 0 as c_uint);
        store_inbounds(bcx, llretptr, llargbundle, [0, n as int]);

        // Create call itself.
        let call_shim_on_c_stack = ccx.upcalls.call_shim_on_c_stack;
        let llshimfnptr = PointerCast(bcx, llshimfn, T_ptr(T_i8()));
        let llrawargbundle = PointerCast(bcx, llargbundle, T_ptr(T_i8()));
        Call(bcx, call_shim_on_c_stack, [llrawargbundle, llshimfnptr]);
        build_return(bcx);
        finish_fn(fcx, lltop);
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
              ccx.sess.span_fatal(
                  native_item.span,
                  "unbound function item in trans_native_mod");
            }
          }
        }
      }
    }
}
