
import driver::session;
import middle::trans::base;
import middle::trans::common::{T_fn, T_i1, T_i8, T_i32,
                               T_int, T_nil, T_dict,
                               T_opaque_vec, T_ptr,
                               T_size_t, T_void};
import lib::llvm::{type_names, ModuleRef, ValueRef, TypeRef};

type upcalls =
    {_fail: ValueRef,
     malloc: ValueRef,
     free: ValueRef,
     validate_box: ValueRef,
     shared_malloc: ValueRef,
     shared_free: ValueRef,
     mark: ValueRef,
     create_shared_type_desc: ValueRef,
     free_shared_type_desc: ValueRef,
     get_type_desc: ValueRef,
     intern_dict: ValueRef,
     vec_grow: ValueRef,
     vec_push: ValueRef,
     cmp_type: ValueRef,
     log_type: ValueRef,
     dynastack_mark: ValueRef,
     dynastack_alloc: ValueRef,
     dynastack_free: ValueRef,
     alloc_c_stack: ValueRef,
     call_shim_on_c_stack: ValueRef,
     rust_personality: ValueRef,
     reset_stack_limit: ValueRef};

fn declare_upcalls(targ_cfg: @session::config,
                   _tn: type_names,
                   tydesc_type: TypeRef,
                   llmod: ModuleRef) -> @upcalls {
    fn decl(llmod: ModuleRef, name: str, tys: [TypeRef], rv: TypeRef) ->
       ValueRef {
        let arg_tys: [TypeRef] = [];
        for t: TypeRef in tys { arg_tys += [t]; }
        let fn_ty = T_fn(arg_tys, rv);
        ret base::decl_cdecl_fn(llmod, "upcall_" + name, fn_ty);
    }
    let d = bind decl(llmod, _, _, _);
    let dv = bind decl(llmod, _, _, T_void());

    let int_t = T_int(targ_cfg);
    let size_t = T_size_t(targ_cfg);
    let opaque_vec_t = T_opaque_vec(targ_cfg);

    ret @{_fail: dv("fail", [T_ptr(T_i8()),
                             T_ptr(T_i8()),
                             size_t]),
          malloc:
              d("malloc", [T_ptr(tydesc_type)],
                T_ptr(T_i8())),
          free:
              dv("free", [T_ptr(T_i8())]),
          validate_box:
              dv("validate_box", [T_ptr(T_i8())]),
          shared_malloc:
              d("shared_malloc", [size_t, T_ptr(tydesc_type)],
                T_ptr(T_i8())),
          shared_free:
              dv("shared_free", [T_ptr(T_i8())]),
          mark:
              d("mark", [T_ptr(T_i8())], int_t),
          create_shared_type_desc:
              d("create_shared_type_desc", [T_ptr(tydesc_type)],
                T_ptr(tydesc_type)),
          free_shared_type_desc:
              dv("free_shared_type_desc", [T_ptr(tydesc_type)]),
          get_type_desc:
              d("get_type_desc",
                [T_ptr(T_nil()), size_t,
                 size_t, size_t,
                 T_ptr(T_ptr(tydesc_type)), int_t],
                T_ptr(tydesc_type)),
          intern_dict:
              d("intern_dict", [size_t, T_ptr(T_dict())], T_ptr(T_dict())),
          vec_grow:
              dv("vec_grow", [T_ptr(T_ptr(opaque_vec_t)), int_t]),
          vec_push:
              dv("vec_push",
                [T_ptr(T_ptr(opaque_vec_t)), T_ptr(tydesc_type),
                 T_ptr(T_i8())]),
          cmp_type:
              dv("cmp_type",
                 [T_ptr(T_i1()), T_ptr(tydesc_type),
                  T_ptr(T_ptr(tydesc_type)), T_ptr(T_i8()), T_ptr(T_i8()),
                  T_i8()]),
          log_type:
              dv("log_type", [T_ptr(tydesc_type), T_ptr(T_i8()), T_i32()]),
          dynastack_mark:
              d("dynastack_mark", [], T_ptr(T_i8())),
          dynastack_alloc:
              d("dynastack_alloc_2", [size_t, T_ptr(tydesc_type)],
                T_ptr(T_i8())),
          dynastack_free:
              dv("dynastack_free", [T_ptr(T_i8())]),
          alloc_c_stack:
              d("alloc_c_stack", [size_t], T_ptr(T_i8())),
          call_shim_on_c_stack:
              d("call_shim_on_c_stack",
                // arguments: void *args, void *fn_ptr
                [T_ptr(T_i8()), T_ptr(T_i8())],
                int_t),
          rust_personality:
              d("rust_personality", [], T_i32()),
          reset_stack_limit:
              dv("reset_stack_limit", [])
         };
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
