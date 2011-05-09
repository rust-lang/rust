import middle.trans;

import trans.decl_cdecl_fn;
import trans.type_names;
import trans.ModuleRef;
import trans.TypeRef;
import trans.ValueRef;

import trans.T_crate;
import trans.T_f32;
import trans.T_f64;
import trans.T_fn;
import trans.T_i8;
import trans.T_i32;
import trans.T_int;
import trans.T_opaque_chan_ptr;
import trans.T_opaque_port_ptr;
import trans.T_opaque_vec_ptr;
import trans.T_ptr;
import trans.T_size_t;
import trans.T_str;
import trans.T_taskptr;
import trans.T_tydesc;
import trans.T_void;

type upcalls = rec(
    ValueRef grow_task,
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
    ValueRef _join,
    ValueRef send,
    ValueRef recv,
    ValueRef _fail,
    ValueRef kill,
    ValueRef exit,
    ValueRef malloc,
    ValueRef free,
    ValueRef mark,
    ValueRef new_str,
    ValueRef new_vec,
    ValueRef vec_grow,
    ValueRef require_rust_sym,
    ValueRef require_c_sym,
    ValueRef get_type_desc,
    ValueRef new_task,
    ValueRef start_task,
    ValueRef new_thread,
    ValueRef start_thread
);

fn declare_upcalls(type_names tn, ModuleRef llmod) -> upcalls {
    fn decl(type_names tn, ModuleRef llmod, str name, vec[TypeRef] tys)
            -> ValueRef {
        let vec[TypeRef] arg_tys = vec(T_taskptr(tn));
        for (TypeRef t in tys) { arg_tys += vec(t); }
        auto fn_ty = T_fn(arg_tys, T_void());
        ret trans.decl_cdecl_fn(llmod, "upcall_" + name, fn_ty);
    }

    auto d = bind decl(tn, llmod, _, _);

    // FIXME: Sigh... remove this when I fix the typechecker pushdown.
    // --pcwalton
    let vec[TypeRef] empty_vec = vec();

    ret rec(
        grow_task=d("grow_task", vec(T_size_t())),
        log_int=d("log_int", vec(T_i32(), T_i32())),
        log_float=d("log_float", vec(T_i32(), T_f32())),
        log_double=d("log_double", vec(T_i32(), T_ptr(T_f64()))),
        log_str=d("log_str", vec(T_i32(), T_ptr(T_str()))),
        trace_word=d("trace_word", vec(T_int())),
        trace_str=d("trace_str", vec(T_ptr(T_i8()))),
        new_port=d("new_port", vec(T_size_t())),
        del_port=d("del_port", vec(T_opaque_port_ptr(tn))),
        new_chan=d("new_chan", vec(T_opaque_port_ptr(tn))),
        flush_chan=d("flush_chan", vec(T_opaque_chan_ptr(tn))),
        del_chan=d("del_chan", vec(T_opaque_chan_ptr(tn))),
        clone_chan=d("clone_chan", vec(T_taskptr(tn), T_opaque_chan_ptr(tn))),
        _yield=d("yield", empty_vec),
        sleep=d("sleep", vec(T_size_t())),
        _join=d("join", vec(T_taskptr(tn))),
        send=d("send", vec(T_opaque_chan_ptr(tn), T_ptr(T_i8()))),
        recv=d("recv", vec(T_ptr(T_int()), T_opaque_port_ptr(tn))),
        _fail=d("fail", vec(T_ptr(T_i8()), T_ptr(T_i8()), T_size_t())),
        kill=d("kill", vec(T_taskptr(tn))),
        exit=d("exit", empty_vec),
        malloc=d("malloc", vec(T_size_t(), T_ptr(T_tydesc(tn)))),
        free=d("free", vec(T_ptr(T_i8()), T_int())),
        mark=d("mark", vec(T_ptr(T_i8()))),
        new_str=d("new_str", vec(T_ptr(T_i8()), T_size_t())),
        new_vec=d("new_vec", vec(T_size_t(), T_ptr(T_tydesc(tn)))),
        vec_grow=d("vec_grow", vec(T_opaque_vec_ptr(), T_size_t(),
                                   T_ptr(T_int()), T_ptr(T_tydesc(tn)))),
        require_rust_sym=d("require_rust_sym",
                           vec(T_ptr(T_crate(tn)), T_size_t(), T_size_t(),
                               T_size_t(), T_ptr(T_i8()),
                               T_ptr(T_ptr(T_i8())))),
        require_c_sym=d("require_c_sym",
                        vec(T_ptr(T_crate(tn)), T_size_t(), T_size_t(),
                            T_ptr(T_i8()), T_ptr(T_i8()))),
        get_type_desc=d("get_type_desc",
                        vec(T_ptr(T_crate(tn)), T_size_t(), T_size_t(),
                            T_size_t(), T_ptr(T_ptr(T_tydesc(tn))))),
        new_task=d("new_task", vec(T_ptr(T_i8()))),
        start_task=d("start_task", vec(T_taskptr(tn), T_int(), T_int(),
                                       T_int(), T_size_t())),
        new_thread=d("new_thread", vec(T_ptr(T_i8()))),
        start_thread=d("start_thread", vec(T_taskptr(tn), T_int(), T_int(),
                                           T_int(), T_size_t()))
    );
}

