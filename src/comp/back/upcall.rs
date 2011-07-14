
import middle::trans;
import trans::decl_cdecl_fn;
import trans::T_f32;
import trans::T_f64;
import trans::T_fn;
import trans::T_bool;
import trans::T_i8;
import trans::T_i32;
import trans::T_int;
import trans::T_nil;
import trans::T_opaque_chan_ptr;
import trans::T_opaque_ivec;
import trans::T_opaque_port_ptr;
import trans::T_opaque_vec_ptr;
import trans::T_ptr;
import trans::T_size_t;
import trans::T_str;
import trans::T_void;
import lib::llvm::type_names;
import lib::llvm::llvm::ModuleRef;
import lib::llvm::llvm::ValueRef;
import lib::llvm::llvm::TypeRef;

type upcalls =
    rec(ValueRef grow_task,
        ValueRef log_int,
        ValueRef log_float,
        ValueRef log_double,
        ValueRef log_str,
        ValueRef trace_word,
        ValueRef trace_str,
        ValueRef new_port,
        ValueRef del_port,
        ValueRef new_chan,
        ValueRef flush_chan,
        ValueRef del_chan,
        ValueRef clone_chan,
        ValueRef _yield,
        ValueRef sleep,
        ValueRef send,
        ValueRef recv,
        ValueRef _fail,
        ValueRef kill,
        ValueRef exit,
        ValueRef malloc,
        ValueRef free,
        ValueRef shared_malloc,
        ValueRef shared_free,
        ValueRef mark,
        ValueRef new_str,
        ValueRef dup_str,
        ValueRef new_vec,
        ValueRef vec_append,
        ValueRef get_type_desc,
        ValueRef new_task,
        ValueRef start_task,
        ValueRef ivec_resize,
        ValueRef ivec_spill,
        ValueRef ivec_resize_shared,
        ValueRef ivec_spill_shared);

fn declare_upcalls(type_names tn, TypeRef tydesc_type, TypeRef taskptr_type,
                   ModuleRef llmod)
   -> @upcalls {
    fn decl(type_names tn, TypeRef tydesc_type, TypeRef taskptr_type,
            ModuleRef llmod, str name, vec[TypeRef] tys,
            TypeRef rv) -> ValueRef {
        let TypeRef[] arg_tys = ~[taskptr_type];
        for (TypeRef t in tys) { arg_tys += ~[t]; }
        auto fn_ty = T_fn(arg_tys, rv);
        ret trans::decl_cdecl_fn(llmod, "upcall_" + name, fn_ty);
    }
    auto dv = bind decl(tn, tydesc_type, taskptr_type, llmod, _, _, T_void());
    auto d = bind decl(tn, tydesc_type, taskptr_type, llmod, _, _, _);
    // FIXME: Sigh:.. remove this when I fix the typechecker pushdown.
    // --pcwalton

    let vec[TypeRef] empty_vec = [];
    ret @rec(grow_task=dv("grow_task", [T_size_t()]),
             log_int=dv("log_int", [T_i32(), T_i32()]),
             log_float=dv("log_float", [T_i32(), T_f32()]),
             log_double=dv("log_double", [T_i32(), T_ptr(T_f64())]),
             log_str=dv("log_str", [T_i32(), T_ptr(T_str())]),
             trace_word=dv("trace_word", [T_int()]),
             trace_str=dv("trace_str", [T_ptr(T_i8())]),
             new_port=d("new_port", [T_size_t()], T_opaque_port_ptr()),
             del_port=dv("del_port", [T_opaque_port_ptr()]),
             new_chan=d("new_chan", [T_opaque_port_ptr()],
                        T_opaque_chan_ptr()),
             flush_chan=dv("flush_chan", [T_opaque_chan_ptr()]),
             del_chan=dv("del_chan", [T_opaque_chan_ptr()]),
             clone_chan=d("clone_chan", [taskptr_type, T_opaque_chan_ptr()],
                          T_opaque_chan_ptr()),
             _yield=dv("yield", empty_vec),
             sleep=dv("sleep", [T_size_t()]),
             send=dv("send", [T_opaque_chan_ptr(), T_ptr(T_i8())]),
             recv=dv("recv", [T_ptr(T_ptr(T_i8())), T_opaque_port_ptr()]),
             _fail=dv("fail", [T_ptr(T_i8()), T_ptr(T_i8()), T_size_t()]),
             kill=dv("kill", [taskptr_type]),
             exit=dv("exit", empty_vec),
             malloc=d("malloc", [T_size_t(), T_ptr(tydesc_type)],
                      T_ptr(T_i8())),
             free=dv("free", [T_ptr(T_i8()), T_int()]),
             shared_malloc=d("shared_malloc",
                      [T_size_t(), T_ptr(tydesc_type)], T_ptr(T_i8())),
             shared_free=dv("shared_free", [T_ptr(T_i8())]),
             mark=d("mark", [T_ptr(T_i8())], T_int()),
             new_str=d("new_str", [T_ptr(T_i8()), T_size_t()],
                       T_ptr(T_str())),
                     dup_str=d("dup_str", [taskptr_type, T_ptr(T_str())],
                       T_ptr(T_str())),
             new_vec=d("new_vec", [T_size_t(), T_ptr(tydesc_type)],
                       T_opaque_vec_ptr()),
             vec_append=d("vec_append",
                          [T_ptr(tydesc_type), T_ptr(tydesc_type),
                           T_ptr(T_opaque_vec_ptr()), T_opaque_vec_ptr(),
                           T_bool()], T_void()),
             get_type_desc=d("get_type_desc",
                             [T_ptr(T_nil()), T_size_t(), T_size_t(),
                              T_size_t(), T_ptr(T_ptr(tydesc_type))],
                             T_ptr(tydesc_type)),
             new_task=d("new_task", [T_ptr(T_str())], taskptr_type),
             start_task=d("start_task",
                          [taskptr_type, T_int(), T_int(), T_size_t()],
                          taskptr_type),
             ivec_resize=d("ivec_resize", [T_ptr(T_opaque_ivec()), T_int()],
                           T_void()),
             ivec_spill=d("ivec_spill", [T_ptr(T_opaque_ivec()), T_int()],
                          T_void()),
             ivec_resize_shared=d("ivec_resize_shared",
                           [T_ptr(T_opaque_ivec()), T_int()], T_void()),
             ivec_spill_shared=d("ivec_spill_shared",
                          [T_ptr(T_opaque_ivec()), T_int()], T_void()));
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
