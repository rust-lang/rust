
import middle::trans;
import trans::decl_cdecl_fn;
import middle::trans_common::T_f32;
import middle::trans_common::T_f64;
import middle::trans_common::T_fn;
import middle::trans_common::T_bool;
import middle::trans_common::T_i1;
import middle::trans_common::T_i8;
import middle::trans_common::T_i32;
import middle::trans_common::T_int;
import middle::trans_common::T_ivec;
import middle::trans_common::T_nil;
import middle::trans_common::T_opaque_chan_ptr;
import middle::trans_common::T_opaque_ivec;
import middle::trans_common::T_opaque_port_ptr;
import middle::trans_common::T_opaque_vec_ptr;
import middle::trans_common::T_ptr;
import middle::trans_common::T_size_t;
import middle::trans_common::T_str;
import middle::trans_common::T_void;
import lib::llvm::type_names;
import lib::llvm::llvm::ModuleRef;
import lib::llvm::llvm::ValueRef;
import lib::llvm::llvm::TypeRef;

type upcalls =
    {grow_task: ValueRef,
     log_int: ValueRef,
     log_float: ValueRef,
     log_double: ValueRef,
     log_str: ValueRef,
     log_istr: ValueRef,
     trace_word: ValueRef,
     trace_str: ValueRef,
     new_port: ValueRef,
     del_port: ValueRef,
     new_chan: ValueRef,
     flush_chan: ValueRef,
     drop_chan: ValueRef,
     take_chan: ValueRef,
     del_chan: ValueRef,
     clone_chan: ValueRef,
     chan_target_task: ValueRef,
     _yield: ValueRef,
     sleep: ValueRef,
     send: ValueRef,
     recv: ValueRef,
     _fail: ValueRef,
     kill: ValueRef,
     exit: ValueRef,
     malloc: ValueRef,
     free: ValueRef,
     shared_malloc: ValueRef,
     shared_free: ValueRef,
     mark: ValueRef,
     new_str: ValueRef,
     dup_str: ValueRef,
     new_vec: ValueRef,
     vec_append: ValueRef,
     get_type_desc: ValueRef,
     new_task: ValueRef,
     take_task: ValueRef,
     drop_task: ValueRef,
     start_task: ValueRef,
     ivec_resize: ValueRef,
     ivec_spill: ValueRef,
     ivec_resize_shared: ValueRef,
     ivec_spill_shared: ValueRef,
     cmp_type: ValueRef};

fn declare_upcalls(tn: type_names, tydesc_type: TypeRef,
                   taskptr_type: TypeRef, llmod: ModuleRef) -> @upcalls {
    fn decl(tn: type_names, llmod: ModuleRef, name: str, tys: TypeRef[],
          rv: TypeRef) -> ValueRef {
        let arg_tys: TypeRef[] = ~[];
        for t: TypeRef  in tys { arg_tys += ~[t]; }
        let fn_ty = T_fn(arg_tys, rv);
        ret trans::decl_cdecl_fn(llmod, "upcall_" + name, fn_ty);
    }
    fn decl_with_taskptr(taskptr_type: TypeRef, tn: type_names,
                         llmod: ModuleRef, name: str, tys: TypeRef[],
                         rv: TypeRef) -> ValueRef {
        ret decl(tn, llmod, name, ~[taskptr_type] + tys, rv);
    }
    let dv = bind decl_with_taskptr(taskptr_type, tn, llmod, _, _, T_void());
    let d = bind decl_with_taskptr(taskptr_type, tn, llmod, _, _, _);
    let dr = bind decl(tn, llmod, _, _, _);

    let empty_vec: TypeRef[] = ~[];
    ret @{grow_task: dv("grow_task", ~[T_size_t()]),
          log_int: dv("log_int", ~[T_i32(), T_i32()]),
          log_float: dv("log_float", ~[T_i32(), T_f32()]),
          log_double: dv("log_double", ~[T_i32(), T_ptr(T_f64())]),
          log_str: dv("log_str", ~[T_i32(), T_ptr(T_str())]),
          log_istr: dv("log_istr", ~[T_i32(), T_ptr(T_ivec(T_i8()))]),
          trace_word: dv("trace_word", ~[T_int()]),
          trace_str: dv("trace_str", ~[T_ptr(T_i8())]),
          new_port: d("new_port", ~[T_size_t()], T_opaque_port_ptr()),
          del_port: dv("del_port", ~[T_opaque_port_ptr()]),
          new_chan:
              d("new_chan", ~[T_opaque_port_ptr()], T_opaque_chan_ptr()),
          flush_chan: dv("flush_chan", ~[T_opaque_chan_ptr()]),
          drop_chan: dv("drop_chan", ~[T_opaque_chan_ptr()]),
          take_chan: dv("take_chan", ~[T_opaque_chan_ptr()]),
          del_chan: dv("del_chan", ~[T_opaque_chan_ptr()]),
          clone_chan:
              d("clone_chan", ~[taskptr_type, T_opaque_chan_ptr()],
                T_opaque_chan_ptr()),
          chan_target_task:
              d("chan_target_task", ~[T_opaque_chan_ptr()], taskptr_type),
          _yield: dv("yield", empty_vec),
          sleep: dv("sleep", ~[T_size_t()]),
          send: dv("send", ~[T_opaque_chan_ptr(), T_ptr(T_i8())]),
          recv: dv("recv", ~[T_ptr(T_ptr(T_i8())), T_opaque_port_ptr()]),
          _fail: dv("fail", ~[T_ptr(T_i8()), T_ptr(T_i8()), T_size_t()]),
          kill: dv("kill", ~[taskptr_type]),
          exit: dv("exit", empty_vec),
          malloc:
              d("malloc", ~[T_size_t(), T_ptr(tydesc_type)], T_ptr(T_i8())),
          free: dv("free", ~[T_ptr(T_i8()), T_int()]),
          shared_malloc:
              d("shared_malloc", ~[T_size_t(), T_ptr(tydesc_type)],
                T_ptr(T_i8())),
          shared_free: dv("shared_free", ~[T_ptr(T_i8())]),
          mark: d("mark", ~[T_ptr(T_i8())], T_int()),
          new_str: d("new_str", ~[T_ptr(T_i8()), T_size_t()], T_ptr(T_str())),
          dup_str:
              d("dup_str", ~[taskptr_type, T_ptr(T_str())], T_ptr(T_str())),
          new_vec:
              d("new_vec", ~[T_size_t(), T_ptr(tydesc_type)],
                T_opaque_vec_ptr()),
          vec_append:
              d("vec_append",
                ~[T_ptr(tydesc_type), T_ptr(tydesc_type),
                  T_ptr(T_opaque_vec_ptr()), T_opaque_vec_ptr(), T_bool()],
                T_void()),
          get_type_desc:
              d("get_type_desc",
                ~[T_ptr(T_nil()), T_size_t(), T_size_t(), T_size_t(),
                  T_ptr(T_ptr(tydesc_type))], T_ptr(tydesc_type)),
          new_task: d("new_task", ~[T_ptr(T_str())], taskptr_type),
          take_task: dv("take_task", ~[taskptr_type]),
          drop_task: dv("drop_task", ~[taskptr_type]),
          start_task:
              d("start_task", ~[taskptr_type, T_int(), T_int(), T_size_t()],
                taskptr_type),
          ivec_resize:
              d("ivec_resize", ~[T_ptr(T_opaque_ivec()), T_int()], T_void()),
          ivec_spill:
              d("ivec_spill", ~[T_ptr(T_opaque_ivec()), T_int()], T_void()),
          ivec_resize_shared:
              d("ivec_resize_shared", ~[T_ptr(T_opaque_ivec()), T_int()],
                T_void()),
          ivec_spill_shared:
              d("ivec_spill_shared", ~[T_ptr(T_opaque_ivec()), T_int()],
                T_void()),
          cmp_type:
              dr("cmp_type", ~[T_ptr(T_i1()), taskptr_type,
                 T_ptr(tydesc_type), T_ptr(T_ptr(tydesc_type)),
                 T_ptr(T_i8()), T_ptr(T_i8()), T_i8()],
                 T_void())};
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
