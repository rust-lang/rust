
import std::str;
import middle::trans;
import trans::decl_cdecl_fn;
import middle::trans_common::{T_f32, T_f64, T_fn, T_bool, T_i1, T_i8, T_i32,
                              T_int, T_vec, T_nil, T_opaque_chan_ptr,
                              T_opaque_vec, T_opaque_port_ptr, T_ptr,
                              T_size_t, T_void};
import lib::llvm::type_names;
import lib::llvm::llvm::ModuleRef;
import lib::llvm::llvm::ValueRef;
import lib::llvm::llvm::TypeRef;

type upcalls =
    {_fail: ValueRef,
     malloc: ValueRef,
     free: ValueRef,
     shared_malloc: ValueRef,
     shared_free: ValueRef,
     mark: ValueRef,
     get_type_desc: ValueRef,
     vec_grow: ValueRef,
     vec_push: ValueRef,
     cmp_type: ValueRef,
     log_type: ValueRef,
     dynastack_mark: ValueRef,
     dynastack_alloc: ValueRef,
     dynastack_free: ValueRef,
     alloc_c_stack: ValueRef,
     call_c_stack: ValueRef,
     rust_personality: ValueRef};

fn declare_upcalls(_tn: type_names, tydesc_type: TypeRef,
                   taskptr_type: TypeRef, llmod: ModuleRef) -> @upcalls {
    fn decl(llmod: ModuleRef, name: str, tys: [TypeRef], rv: TypeRef) ->
       ValueRef {
        let arg_tys: [TypeRef] = [];
        for t: TypeRef in tys { arg_tys += [t]; }
        let fn_ty = T_fn(arg_tys, rv);
        ret trans::decl_cdecl_fn(llmod, "upcall_" + name, fn_ty);
    }
    fn decl_with_taskptr(taskptr_type: TypeRef, llmod: ModuleRef, name: str,
                         tys: [TypeRef], rv: TypeRef) -> ValueRef {
        ret decl(llmod, name, [taskptr_type] + tys, rv);
    }
    let dv = bind decl_with_taskptr(taskptr_type, llmod, _, _, T_void());
    let d = bind decl_with_taskptr(taskptr_type, llmod, _, _, _);
    let dr = bind decl(llmod, _, _, _);

    ret @{_fail: dv("fail", [T_ptr(T_i8()), T_ptr(T_i8()), T_size_t()]),
          malloc:
              d("malloc", [T_size_t(), T_ptr(tydesc_type)], T_ptr(T_i8())),
          free: dv("free", [T_ptr(T_i8()), T_int()]),
          shared_malloc:
              d("shared_malloc", [T_size_t(), T_ptr(tydesc_type)],
                T_ptr(T_i8())),
          shared_free: dv("shared_free", [T_ptr(T_i8())]),
          mark: d("mark", [T_ptr(T_i8())], T_int()),
          get_type_desc:
              d("get_type_desc",
                [T_ptr(T_nil()), T_size_t(), T_size_t(), T_size_t(),
                 T_ptr(T_ptr(tydesc_type)), T_int()], T_ptr(tydesc_type)),
          vec_grow:
              d("vec_grow", [T_ptr(T_ptr(T_opaque_vec())), T_int()],
                T_void()),
          vec_push:
              d("vec_push",
                [T_ptr(T_ptr(T_opaque_vec())), T_ptr(tydesc_type),
                 T_ptr(T_i8())], T_void()),
          cmp_type:
              dr("cmp_type",
                 [T_ptr(T_i1()), taskptr_type, T_ptr(tydesc_type),
                  T_ptr(T_ptr(tydesc_type)), T_ptr(T_i8()), T_ptr(T_i8()),
                  T_i8()], T_void()),
          log_type:
              dr("log_type",
                 [taskptr_type, T_ptr(tydesc_type), T_ptr(T_i8()), T_i32()],
                 T_void()),
          dynastack_mark: d("dynastack_mark", [], T_ptr(T_i8())),
          dynastack_alloc:
              d("dynastack_alloc_2", [T_size_t(), T_ptr(tydesc_type)],
                T_ptr(T_i8())),
          dynastack_free: d("dynastack_free", [T_ptr(T_i8())], T_void()),
          alloc_c_stack: dr("alloc_c_stack", [T_size_t()], T_ptr(T_i8())),
          call_c_stack: dr("call_c_stack",
                           [T_ptr(T_fn([], T_void())), T_ptr(T_i8())],
                           T_int()),
          rust_personality: dr("rust_personality", [], T_i32())
         };
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
