
import driver::session;
import middle::trans::base;
import middle::trans::common::{T_fn, T_i1, T_i8, T_i32,
                               T_int, T_nil,
                               T_opaque_vec, T_ptr, T_unique_ptr,
                               T_size_t, T_void, T_vec2};
import lib::llvm::{type_names, ModuleRef, ValueRef, TypeRef};

type upcalls =
    {_fail: ValueRef,
     trace: ValueRef,
     malloc_dyn: ValueRef,
     free: ValueRef,
     exchange_malloc_dyn: ValueRef,
     exchange_free: ValueRef,
     validate_box: ValueRef,
     mark: ValueRef,
     vec_grow: ValueRef,
     str_new_uniq: ValueRef,
     str_new_shared: ValueRef,
     str_concat: ValueRef,
     cmp_type: ValueRef,
     log_type: ValueRef,
     alloc_c_stack: ValueRef,
     call_shim_on_c_stack: ValueRef,
     call_shim_on_rust_stack: ValueRef,
     rust_personality: ValueRef,
     reset_stack_limit: ValueRef};

fn declare_upcalls(targ_cfg: @session::config,
                   _tn: type_names,
                   tydesc_type: TypeRef,
                   llmod: ModuleRef) -> @upcalls {
    fn decl(llmod: ModuleRef, prefix: str, name: str,
            tys: [TypeRef], rv: TypeRef) ->
       ValueRef {
        let mut arg_tys: [TypeRef] = [];
        for tys.each {|t| arg_tys += [t]; }
        let fn_ty = T_fn(arg_tys, rv);
        ret base::decl_cdecl_fn(llmod, prefix + name, fn_ty);
    }
    fn nothrow(f: ValueRef) -> ValueRef {
        base::set_no_unwind(f); f
    }
    let d = {|a,b,c|decl(llmod, "upcall_", a, b, c)};
    let dv = {|a,b|decl(llmod, "upcall_", a, b, T_void())};

    let int_t = T_int(targ_cfg);
    let size_t = T_size_t(targ_cfg);

    ret @{_fail: dv("fail", [T_ptr(T_i8()),
                             T_ptr(T_i8()),
                             size_t]),
          trace: dv("trace", [T_ptr(T_i8()),
                              T_ptr(T_i8()),
                              int_t]),
          malloc_dyn:
              nothrow(d("malloc_dyn",
                        [T_ptr(tydesc_type), int_t],
                        T_ptr(T_i8()))),
          free:
              nothrow(dv("free", [T_ptr(T_i8())])),
          exchange_malloc_dyn:
              nothrow(d("exchange_malloc_dyn",
                        [T_ptr(tydesc_type), int_t],
                        T_ptr(T_i8()))),
          exchange_free:
              nothrow(dv("exchange_free", [T_ptr(T_i8())])),
          validate_box:
              nothrow(dv("validate_box", [T_ptr(T_i8())])),
          mark:
              d("mark", [T_ptr(T_i8())], int_t),
          vec_grow:
              nothrow(dv("vec_grow", [T_ptr(T_ptr(T_i8())), int_t])),
          str_new_uniq:
              nothrow(d("str_new_uniq", [T_ptr(T_i8()), int_t],
                        T_ptr(T_i8()))),
          str_new_shared:
              nothrow(d("str_new_shared", [T_ptr(T_i8()), int_t],
                        T_ptr(T_i8()))),
          str_concat:
              nothrow(d("str_concat", [T_ptr(T_i8()),
                                       T_ptr(T_i8())],
                        T_ptr(T_i8()))),
          cmp_type:
              dv("cmp_type",
                 [T_ptr(T_i1()), T_ptr(tydesc_type),
                  T_ptr(T_ptr(tydesc_type)), T_ptr(T_i8()),
                  T_ptr(T_i8()),
                  T_i8()]),
          log_type:
              dv("log_type", [T_ptr(tydesc_type),
                              T_ptr(T_i8()), T_i32()]),
          alloc_c_stack:
              d("alloc_c_stack", [size_t], T_ptr(T_i8())),
          call_shim_on_c_stack:
              d("call_shim_on_c_stack",
                // arguments: void *args, void *fn_ptr
                [T_ptr(T_i8()), T_ptr(T_i8())],
                int_t),
          call_shim_on_rust_stack:
              d("call_shim_on_rust_stack",
                [T_ptr(T_i8()), T_ptr(T_i8())], int_t),
          rust_personality:
              nothrow(d("rust_personality", [], T_i32())),
          reset_stack_limit:
              nothrow(dv("reset_stack_limit", []))
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
