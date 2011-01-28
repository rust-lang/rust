import std._str;
import std._uint;
import std._vec;
import std._str.rustrt.sbuf;
import std._vec.rustrt.vbuf;
import std.map;
import std.map.hashmap;
import std.option;
import std.option.some;
import std.option.none;

import front.ast;
import driver.session;
import middle.ty;
import back.x86;
import back.abi;

import middle.ty.pat_ty;

import util.common;
import util.common.append;
import util.common.istr;
import util.common.new_def_hash;
import util.common.new_str_hash;

import lib.llvm.llvm;
import lib.llvm.builder;
import lib.llvm.target_data;
import lib.llvm.type_handle;
import lib.llvm.mk_pass_manager;
import lib.llvm.mk_target_data;
import lib.llvm.mk_type_handle;
import lib.llvm.llvm.ModuleRef;
import lib.llvm.llvm.ValueRef;
import lib.llvm.llvm.TypeRef;
import lib.llvm.llvm.TypeHandleRef;
import lib.llvm.llvm.BuilderRef;
import lib.llvm.llvm.BasicBlockRef;

import lib.llvm.False;
import lib.llvm.True;

state obj namegen(mutable int i) {
    fn next(str prefix) -> str {
        i += 1;
        ret prefix + istr(i);
    }
}

type glue_fns = rec(ValueRef activate_glue,
                    ValueRef yield_glue,
                    ValueRef exit_task_glue,
                    vec[ValueRef] upcall_glues,
                    ValueRef no_op_type_glue,
                    ValueRef memcpy_glue,
                    ValueRef bzero_glue);

tag arity { nullary; n_ary; }
type tag_info = rec(type_handle th,
                    mutable vec[tup(ast.def_id,arity)] variants,
                    mutable uint size);

state type crate_ctxt = rec(session.session sess,
                            ModuleRef llmod,
                            target_data td,
                            hashmap[str, ValueRef] upcalls,
                            hashmap[str, ValueRef] intrinsics,
                            hashmap[str, ValueRef] item_names,
                            hashmap[ast.def_id, ValueRef] item_ids,
                            hashmap[ast.def_id, @ast.item] items,
                            hashmap[ast.def_id, @tag_info] tags,
                            hashmap[ast.def_id, ValueRef] fn_pairs,
                            hashmap[ast.def_id, ValueRef] consts,
                            hashmap[ast.def_id,()] obj_methods,
                            hashmap[@ty.t, ValueRef] tydescs,
                            vec[ast.obj_field] obj_fields,
                            @glue_fns glues,
                            namegen names,
                            str path);

state type fn_ctxt = rec(ValueRef llfn,
                         ValueRef lltaskptr,
                         ValueRef llclosure,
                         mutable option.t[ValueRef] llself,
                         mutable option.t[ValueRef] llretptr,
                         hashmap[ast.def_id, ValueRef] llargs,
                         hashmap[ast.def_id, ValueRef] llobjfields,
                         hashmap[ast.def_id, ValueRef] lllocals,
                         hashmap[ast.def_id, ValueRef] lltydescs,
                         @crate_ctxt ccx);

tag cleanup {
    clean(fn(@block_ctxt cx) -> result);
}


tag block_kind {
    SCOPE_BLOCK;
    NON_SCOPE_BLOCK;
}

state type block_ctxt = rec(BasicBlockRef llbb,
                            builder build,
                            block_parent parent,
                            block_kind kind,
                            mutable vec[cleanup] cleanups,
                            @fn_ctxt fcx);

// FIXME: we should be able to use option.t[@block_parent] here but
// the infinite-tag check in rustboot gets upset.

tag block_parent {
    parent_none;
    parent_some(@block_ctxt);
}


state type result = rec(mutable @block_ctxt bcx,
                        mutable ValueRef val);

fn res(@block_ctxt bcx, ValueRef val) -> result {
    ret rec(mutable bcx = bcx,
            mutable val = val);
}

fn ty_str(TypeRef t) -> str {
    ret lib.llvm.type_to_str(t);
}

fn val_ty(ValueRef v) -> TypeRef {
    ret llvm.LLVMTypeOf(v);
}

fn val_str(ValueRef v) -> str {
    ret ty_str(val_ty(v));
}


// LLVM type constructors.

fn T_void() -> TypeRef {
    // Note: For the time being llvm is kinda busted here, it has the notion
    // of a 'void' type that can only occur as part of the signature of a
    // function, but no general unit type of 0-sized value. This is, afaict,
    // vestigial from its C heritage, and we'll be attempting to submit a
    // patch upstream to fix it. In the mean time we only model function
    // outputs (Rust functions and C functions) using T_void, and model the
    // Rust general purpose nil type you can construct as 1-bit (always
    // zero). This makes the result incorrect for now -- things like a tuple
    // of 10 nil values will have 10-bit size -- but it doesn't seem like we
    // have any other options until it's fixed upstream.
    ret llvm.LLVMVoidType();
}

fn T_nil() -> TypeRef {
    // NB: See above in T_void().
    ret llvm.LLVMInt1Type();
}

fn T_i1() -> TypeRef {
    ret llvm.LLVMInt1Type();
}

fn T_i8() -> TypeRef {
    ret llvm.LLVMInt8Type();
}

fn T_i16() -> TypeRef {
    ret llvm.LLVMInt16Type();
}

fn T_i32() -> TypeRef {
    ret llvm.LLVMInt32Type();
}

fn T_i64() -> TypeRef {
    ret llvm.LLVMInt64Type();
}

fn T_f32() -> TypeRef {
    ret llvm.LLVMFloatType();
}

fn T_f64() -> TypeRef {
    ret llvm.LLVMDoubleType();
}

fn T_bool() -> TypeRef {
    ret T_i1();
}

fn T_int() -> TypeRef {
    // FIXME: switch on target type.
    ret T_i32();
}

fn T_char() -> TypeRef {
    ret T_i32();
}

fn T_fn(vec[TypeRef] inputs, TypeRef output) -> TypeRef {
    ret llvm.LLVMFunctionType(output,
                              _vec.buf[TypeRef](inputs),
                              _vec.len[TypeRef](inputs),
                              False);
}

fn T_fn_pair(TypeRef tfn) -> TypeRef {
    ret T_struct(vec(T_ptr(tfn),
                     T_opaque_closure_ptr()));
}

fn T_ptr(TypeRef t) -> TypeRef {
    ret llvm.LLVMPointerType(t, 0u);
}

fn T_struct(vec[TypeRef] elts) -> TypeRef {
    ret llvm.LLVMStructType(_vec.buf[TypeRef](elts),
                            _vec.len[TypeRef](elts),
                            False);
}

fn T_opaque() -> TypeRef {
    ret llvm.LLVMOpaqueType();
}

fn T_task() -> TypeRef {
    ret T_struct(vec(T_int(),      // Refcount
                     T_int(),      // Delegate pointer
                     T_int(),      // Stack segment pointer
                     T_int(),      // Runtime SP
                     T_int(),      // Rust SP
                     T_int(),      // GC chain
                     T_int(),      // Domain pointer
                     T_int()       // Crate cache pointer
                     ));
}

fn T_tydesc() -> TypeRef {

    auto th = mk_type_handle();
    auto abs_tydesc = llvm.LLVMResolveTypeHandle(th.llth);

    auto pvoid = T_ptr(T_i8());
    auto glue_fn_ty = T_ptr(T_fn(vec(T_taskptr(),
                                     T_ptr(abs_tydesc),
                                     pvoid), T_void()));
    auto tydesc = T_struct(vec(T_ptr(abs_tydesc), // first_param
                               T_int(),           // size
                               T_int(),           // align
                               glue_fn_ty,        // take_glue_off
                               glue_fn_ty,        // drop_glue_off
                               glue_fn_ty,        // free_glue_off
                               glue_fn_ty,        // sever_glue_off
                               glue_fn_ty,        // mark_glue_off
                               glue_fn_ty,        // obj_drop_glue_off
                               glue_fn_ty));      // is_stateful

    llvm.LLVMRefineType(abs_tydesc, tydesc);
    ret llvm.LLVMResolveTypeHandle(th.llth);
}

fn T_array(TypeRef t, uint n) -> TypeRef {
    ret llvm.LLVMArrayType(t, n);
}

fn T_vec(TypeRef t) -> TypeRef {
    ret T_struct(vec(T_int(),       // Refcount
                     T_int(),       // Alloc
                     T_int(),       // Fill
                     T_array(t, 0u) // Body elements
                     ));
}

fn T_str() -> TypeRef {
    ret T_vec(T_i8());
}

fn T_box(TypeRef t) -> TypeRef {
    ret T_struct(vec(T_int(), t));
}

fn T_crate() -> TypeRef {
    ret T_struct(vec(T_int(),      // ptrdiff_t image_base_off
                     T_int(),      // uintptr_t self_addr
                     T_int(),      // ptrdiff_t debug_abbrev_off
                     T_int(),      // size_t debug_abbrev_sz
                     T_int(),      // ptrdiff_t debug_info_off
                     T_int(),      // size_t debug_info_sz
                     T_int(),      // size_t activate_glue_off
                     T_int(),      // size_t yield_glue_off
                     T_int(),      // size_t unwind_glue_off
                     T_int(),      // size_t gc_glue_off
                     T_int(),      // size_t main_exit_task_glue_off
                     T_int(),      // int n_rust_syms
                     T_int(),      // int n_c_syms
                     T_int()       // int n_libs
                     ));
}

fn T_double() -> TypeRef {
    ret llvm.LLVMDoubleType();
}

fn T_taskptr() -> TypeRef {
    ret T_ptr(T_task());
}

fn T_typaram_ptr() -> TypeRef {
    ret T_ptr(T_i8());
}

fn T_closure_ptr(TypeRef lltarget_ty,
                 TypeRef llbindings_ty) -> TypeRef {
    ret T_ptr(T_box(T_struct(vec(T_ptr(T_tydesc()),
                                 lltarget_ty,
                                 llbindings_ty)
                             // FIXME: add captured typarams.
                             )));
}

fn T_opaque_closure_ptr() -> TypeRef {
    ret T_closure_ptr(T_struct(vec(T_ptr(T_nil()),
                                   T_ptr(T_nil()))),
                      T_nil());
}


fn type_of(@crate_ctxt cx, @ty.t t) -> TypeRef {
    let TypeRef llty = type_of_inner(cx, t);
    check (llty as int != 0);
    llvm.LLVMAddTypeName(cx.llmod, _str.buf(ty.ty_to_str(t)), llty);
    ret llty;
}

// NB: this must match trans_args and create_llargs_for_fn_args.
fn type_of_fn_full(@crate_ctxt cx,
                   option.t[TypeRef] obj_self,
                   vec[ty.arg] inputs,
                   @ty.t output) -> TypeRef {
    let vec[TypeRef] atys = vec(T_taskptr());

    auto fn_ty = ty.plain_ty(ty.ty_fn(inputs, output));
    auto ty_param_count = ty.count_ty_params(fn_ty);
    auto i = 0u;
    while (i < ty_param_count) {
        atys += T_ptr(T_tydesc());
        i += 1u;
    }

    if (ty.type_has_dynamic_size(output)) {
        atys += T_typaram_ptr();
    }

    alt (obj_self) {
        case (some[TypeRef](?t)) {
            check (t as int != 0);
            atys += t;
        }
        case (_) {
            atys += T_opaque_closure_ptr();
        }
    }

    for (ty.arg arg in inputs) {
        if (ty.type_has_dynamic_size(arg.ty)) {
            check (arg.mode == ast.alias);
            atys += T_typaram_ptr();
        } else {
            let TypeRef t = type_of(cx, arg.ty);
            alt (arg.mode) {
                case (ast.alias) {
                    t = T_ptr(t);
                }
                case (_) { /* fall through */  }
            }
            atys += t;
        }
    }

    auto ret_ty;
    if (ty.type_is_nil(output) || ty.type_has_dynamic_size(output)) {
        ret_ty = llvm.LLVMVoidType();
    } else {
        ret_ty = type_of(cx, output);
    }

    ret T_fn(atys, ret_ty);
}

fn type_of_fn(@crate_ctxt cx, vec[ty.arg] inputs, @ty.t output) -> TypeRef {
    ret type_of_fn_full(cx, none[TypeRef], inputs, output);
}

fn type_of_inner(@crate_ctxt cx, @ty.t t) -> TypeRef {
    alt (t.struct) {
        case (ty.ty_nil) { ret T_nil(); }
        case (ty.ty_bool) { ret T_bool(); }
        case (ty.ty_int) { ret T_int(); }
        case (ty.ty_uint) { ret T_int(); }
        case (ty.ty_machine(?tm)) {
            alt (tm) {
                case (common.ty_i8) { ret T_i8(); }
                case (common.ty_u8) { ret T_i8(); }
                case (common.ty_i16) { ret T_i16(); }
                case (common.ty_u16) { ret T_i16(); }
                case (common.ty_i32) { ret T_i32(); }
                case (common.ty_u32) { ret T_i32(); }
                case (common.ty_i64) { ret T_i64(); }
                case (common.ty_u64) { ret T_i64(); }
                case (common.ty_f32) { ret T_f32(); }
                case (common.ty_f64) { ret T_f64(); }
            }
        }
        case (ty.ty_char) { ret T_char(); }
        case (ty.ty_str) { ret T_ptr(T_str()); }
        case (ty.ty_tag(?tag_id)) {
            ret llvm.LLVMResolveTypeHandle(cx.tags.get(tag_id).th.llth);
        }
        case (ty.ty_box(?t)) {
            ret T_ptr(T_box(type_of(cx, t)));
        }
        case (ty.ty_vec(?t)) {
            ret T_ptr(T_vec(type_of(cx, t)));
        }
        case (ty.ty_tup(?elts)) {
            let vec[TypeRef] tys = vec();
            for (@ty.t elt in elts) {
                tys += type_of(cx, elt);
            }
            ret T_struct(tys);
        }
        case (ty.ty_rec(?fields)) {
            let vec[TypeRef] tys = vec();
            for (ty.field f in fields) {
                tys += type_of(cx, f.ty);
            }
            ret T_struct(tys);
        }
        case (ty.ty_fn(?args, ?out)) {
            ret T_fn_pair(type_of_fn(cx, args, out));
        }
        case (ty.ty_obj(?meths)) {
            auto th = mk_type_handle();
            auto self_ty = llvm.LLVMResolveTypeHandle(th.llth);

            let vec[TypeRef] mtys = vec();
            for (ty.method m in meths) {
                let TypeRef mty =
                    type_of_fn_full(cx,
                                    some[TypeRef](self_ty),
                                    m.inputs, m.output);
                mtys += T_ptr(mty);
            }
            let TypeRef vtbl = T_struct(mtys);
            let TypeRef body = T_struct(vec(T_ptr(T_tydesc()),
                                            T_nil()));
            let TypeRef pair =
                T_struct(vec(T_ptr(vtbl),
                             T_ptr(T_box(body))));
            auto abs_pair = llvm.LLVMResolveTypeHandle(th.llth);
            llvm.LLVMRefineType(abs_pair, pair);
            abs_pair = llvm.LLVMResolveTypeHandle(th.llth);
            ret abs_pair;
        }
        case (ty.ty_var(_)) {
            log "ty_var in trans.type_of";
            fail;
        }
        case (ty.ty_param(_)) {
            ret T_typaram_ptr();
        }
    }
    fail;
}

fn type_of_arg(@crate_ctxt cx, &ty.arg arg) -> TypeRef {
    auto ty = type_of(cx, arg.ty);
    if (arg.mode == ast.alias) {
        ty = T_ptr(ty);
    }
    ret ty;
}

// Name sanitation. LLVM will happily accept identifiers with weird names, but
// gas doesn't!

fn sanitize(str s) -> str {
    auto result = "";
    for (u8 c in s) {
        if (c == ('@' as u8)) {
            result += "boxed_";
        } else {
            if (c == (',' as u8)) {
                result += "_";
            } else {
                if (c == ('{' as u8) || c == ('(' as u8)) {
                    result += "_of_";
                } else {
                    if (c != 10u8 && c != ('}' as u8) && c != (')' as u8) &&
                        c != (' ' as u8) && c != ('\t' as u8) &&
                        c != (';' as u8)) {
                        auto v = vec(c);
                        result += _str.from_bytes(v);
                    }
                }
            }
        }
    }
    ret result;
}

// LLVM constant constructors.

fn C_null(TypeRef t) -> ValueRef {
    ret llvm.LLVMConstNull(t);
}

fn C_integral(int i, TypeRef t) -> ValueRef {
    // FIXME. We can't use LLVM.ULongLong with our existing minimal native
    // API, which only knows word-sized args.  Lucky for us LLVM has a "take a
    // string encoding" version.  Hilarious. Please fix to handle:
    //
    // ret llvm.LLVMConstInt(T_int(), t as LLVM.ULongLong, False);
    //
    ret llvm.LLVMConstIntOfString(t, _str.buf(istr(i)), 10);
}

fn C_nil() -> ValueRef {
    // NB: See comment above in T_void().
    ret C_integral(0, T_i1());
}

fn C_bool(bool b) -> ValueRef {
    if (b) {
        ret C_integral(1, T_bool());
    } else {
        ret C_integral(0, T_bool());
    }
}

fn C_int(int i) -> ValueRef {
    ret C_integral(i, T_int());
}

// This is a 'c-like' raw string, which differs from
// our boxed-and-length-annotated strings.
fn C_cstr(@crate_ctxt cx, str s) -> ValueRef {
    auto sc = llvm.LLVMConstString(_str.buf(s), _str.byte_len(s), False);
    auto g = llvm.LLVMAddGlobal(cx.llmod, val_ty(sc),
                                _str.buf(cx.names.next("str")));
    llvm.LLVMSetInitializer(g, sc);
    llvm.LLVMSetGlobalConstant(g, True);
    llvm.LLVMSetLinkage(g, lib.llvm.LLVMPrivateLinkage
                        as llvm.Linkage);
    ret g;
}

// A rust boxed-and-length-annotated string.
fn C_str(@crate_ctxt cx, str s) -> ValueRef {
    auto len = _str.byte_len(s);
    auto box = C_struct(vec(C_int(abi.const_refcount as int),
                            C_int(len + 1u as int), // 'alloc'
                            C_int(len + 1u as int), // 'fill'
                            llvm.LLVMConstString(_str.buf(s),
                                                 len, False)));
    auto g = llvm.LLVMAddGlobal(cx.llmod, val_ty(box),
                                _str.buf(cx.names.next("str")));
    llvm.LLVMSetInitializer(g, box);
    llvm.LLVMSetGlobalConstant(g, True);
    llvm.LLVMSetLinkage(g, lib.llvm.LLVMPrivateLinkage
                        as llvm.Linkage);
    ret llvm.LLVMConstPointerCast(g, T_ptr(T_str()));
}

fn C_zero_byte_arr(uint size) -> ValueRef {
    auto i = 0u;
    let vec[ValueRef] elts = vec();
    while (i < size) {
        elts += vec(C_integral(0, T_i8()));
        i += 1u;
    }
    ret llvm.LLVMConstArray(T_i8(), _vec.buf[ValueRef](elts),
                            _vec.len[ValueRef](elts));
}

fn C_struct(vec[ValueRef] elts) -> ValueRef {
    ret llvm.LLVMConstStruct(_vec.buf[ValueRef](elts),
                             _vec.len[ValueRef](elts),
                             False);
}

fn decl_fn(ModuleRef llmod, str name, uint cc, TypeRef llty) -> ValueRef {
    let ValueRef llfn =
        llvm.LLVMAddFunction(llmod, _str.buf(name), llty);
    llvm.LLVMSetFunctionCallConv(llfn, cc);
    ret llfn;
}

fn decl_cdecl_fn(ModuleRef llmod, str name, TypeRef llty) -> ValueRef {
    ret decl_fn(llmod, name, lib.llvm.LLVMCCallConv, llty);
}

fn decl_fastcall_fn(ModuleRef llmod, str name, TypeRef llty) -> ValueRef {
    ret decl_fn(llmod, name, lib.llvm.LLVMFastCallConv, llty);
}

fn decl_glue(ModuleRef llmod, str s) -> ValueRef {
    ret decl_cdecl_fn(llmod, s, T_fn(vec(T_taskptr()), T_void()));
}

fn decl_upcall(ModuleRef llmod, uint _n) -> ValueRef {
    // It doesn't actually matter what type we come up with here, at the
    // moment, as we cast the upcall function pointers to int before passing
    // them to the indirect upcall-invocation glue.  But eventually we'd like
    // to call them directly, once we have a calling convention worked out.
    let int n = _n as int;
    let str s = abi.upcall_glue_name(n);
    let vec[TypeRef] args =
        vec(T_taskptr(), // taskptr
            T_int())     // callee
        + _vec.init_elt[TypeRef](T_int(), n as uint);

    ret decl_fastcall_fn(llmod, s, T_fn(args, T_int()));
}

fn get_upcall(@crate_ctxt cx, str name, int n_args) -> ValueRef {
    if (cx.upcalls.contains_key(name)) {
        ret cx.upcalls.get(name);
    }
    auto inputs = vec(T_taskptr());
    inputs += _vec.init_elt[TypeRef](T_int(), n_args as uint);
    auto output = T_int();
    auto f = decl_cdecl_fn(cx.llmod, name, T_fn(inputs, output));
    cx.upcalls.insert(name, f);
    ret f;
}

fn trans_upcall(@block_ctxt cx, str name, vec[ValueRef] args) -> result {
    let int n = _vec.len[ValueRef](args) as int;
    let ValueRef llupcall = get_upcall(cx.fcx.ccx, name, n);
    llupcall = llvm.LLVMConstPointerCast(llupcall, T_int());

    let ValueRef llglue = cx.fcx.ccx.glues.upcall_glues.(n);
    let vec[ValueRef] call_args = vec(cx.fcx.lltaskptr, llupcall);
    for (ValueRef a in args) {
        call_args += cx.build.ZExtOrBitCast(a, T_int());
    }
    ret res(cx, cx.build.FastCall(llglue, call_args));
}

fn trans_non_gc_free(@block_ctxt cx, ValueRef v) -> result {
    ret trans_upcall(cx, "upcall_free", vec(cx.build.PtrToInt(v, T_int()),
                                            C_int(0)));
}

fn find_scope_cx(@block_ctxt cx) -> @block_ctxt {
    if (cx.kind == SCOPE_BLOCK) {
        ret cx;
    }
    alt (cx.parent) {
        case (parent_some(?b)) {
            be find_scope_cx(b);
        }
        case (parent_none) {
            fail;
        }
    }
}

fn umax(@block_ctxt cx, ValueRef a, ValueRef b) -> ValueRef {
    auto cond = cx.build.ICmp(lib.llvm.LLVMIntULT, a, b);
    ret cx.build.Select(cond, b, a);
}

fn align_to(@block_ctxt cx, ValueRef off, ValueRef align) -> ValueRef {
    auto mask = cx.build.Sub(align, C_int(1));
    auto bumped = cx.build.Add(off, mask);
    ret cx.build.And(bumped, cx.build.Not(mask));
}

fn llsize_of(TypeRef t) -> ValueRef {
    ret llvm.LLVMConstIntCast(lib.llvm.llvm.LLVMSizeOf(t), T_int(), False);
}

fn llalign_of(TypeRef t) -> ValueRef {
    ret llvm.LLVMConstIntCast(lib.llvm.llvm.LLVMAlignOf(t), T_int(), False);
}

fn size_of(@block_ctxt cx, @ty.t t) -> ValueRef {
    if (!ty.type_has_dynamic_size(t)) {
        ret llsize_of(type_of(cx.fcx.ccx, t));
    }
    ret dynamic_size_of(cx, t);
}

fn align_of(@block_ctxt cx, @ty.t t) -> ValueRef {
    if (!ty.type_has_dynamic_size(t)) {
        ret llalign_of(type_of(cx.fcx.ccx, t));
    }
    ret dynamic_align_of(cx, t);
}

fn dynamic_size_of(@block_ctxt cx, @ty.t t) -> ValueRef {
    alt (t.struct) {
        case (ty.ty_param(?p)) {
            auto szptr = field_of_tydesc(cx, t, abi.tydesc_field_size);
            ret cx.build.Load(szptr);
        }
        case (ty.ty_tup(?elts)) {
            //
            // C padding rules:
            //
            //
            //   - Pad after each element so that next element is aligned.
            //   - Pad after final structure member so that whole structure
            //     is aligned to max alignment of interior.
            //
            auto off = C_int(0);
            auto max_align = C_int(1);
            for (@ty.t e in elts) {
                auto elt_align = align_of(cx, e);
                auto elt_size = size_of(cx, e);
                auto aligned_off = align_to(cx, off, elt_align);
                off = cx.build.Add(aligned_off, elt_size);
                max_align = umax(cx, max_align, elt_align);
            }
            off = align_to(cx, off, max_align);
            ret off;
        }
        case (ty.ty_rec(?flds)) {
            auto off = C_int(0);
            auto max_align = C_int(1);
            for (ty.field f in flds) {
                auto elt_align = align_of(cx, f.ty);
                auto elt_size = size_of(cx, f.ty);
                auto aligned_off = align_to(cx, off, elt_align);
                off = cx.build.Add(aligned_off, elt_size);
                max_align = umax(cx, max_align, elt_align);
            }
            off = align_to(cx, off, max_align);
            ret off;
        }
    }
}

fn dynamic_align_of(@block_ctxt cx, @ty.t t) -> ValueRef {
    alt (t.struct) {
        case (ty.ty_param(?p)) {
            auto aptr = field_of_tydesc(cx, t, abi.tydesc_field_align);
            ret cx.build.Load(aptr);
        }
        case (ty.ty_tup(?elts)) {
            auto a = C_int(1);
            for (@ty.t e in elts) {
                a = umax(cx, a, align_of(cx, e));
            }
            ret a;
        }
        case (ty.ty_rec(?flds)) {
            auto a = C_int(1);
            for (ty.field f in flds) {
                a = umax(cx, a, align_of(cx, f.ty));
            }
            ret a;
        }
    }
}

// Replacement for the LLVM 'GEP' instruction when field-indexing into a
// tuple-like structure (tup, rec, tag) with a static index. This one is
// driven off ty.struct and knows what to do when it runs into a ty_param
// stuck in the middle of the thing it's GEP'ing into. Much like size_of and
// align_of, above.

fn GEP_tup_like(@block_ctxt cx, @ty.t t,
                ValueRef base, vec[int] ixs) -> ValueRef {

    check (ty.type_is_tup_like(t));

    // It might be a static-known type. Handle this.

    if (! ty.type_has_dynamic_size(t)) {
        let vec[ValueRef] v = vec();
        for (int i in ixs) {
            v += C_int(i);
        }
        ret cx.build.GEP(base, v);
    }

    // It is a dynamic-containing type that, if we convert directly to an LLVM
    // TypeRef, will be all wrong; there's no proper LLVM type to represent
    // it, and the lowering function will stick in i8* values for each
    // ty_param, which is not right; the ty_params are all of some dynamic
    // size.
    //
    // What we must do instead is sadder. We must look through the indices
    // manually and split the input type into a prefix and a target. We then
    // measure the prefix size, bump the input pointer by that amount, and
    // cast to a pointer-to-target type.


    // Given a type, an index vector and an element number N in that vector,
    // calculate index X and the type that results by taking the first X-1
    // elements of the type and splitting the Xth off. Return the prefix as
    // well as the innermost Xth type.

    fn split_type(@ty.t t, vec[int] ixs, uint n)
        -> rec(vec[@ty.t] prefix, @ty.t target) {

        let uint len = _vec.len[int](ixs);

        // We don't support 0-index or 1-index GEPs. The former is nonsense
        // and the latter would only be meaningful if we supported non-0
        // values for the 0th index (we don't).

        check (len > 1u);

        if (n == 0u) {
            // Since we're starting from a value that's a pointer to a
            // *single* structure, the first index (in GEP-ese) should just be
            // 0, to yield the pointee.
            check (ixs.(n) == 0);
            ret split_type(t, ixs, n+1u);
        }

        check (n < len);

        let int ix = ixs.(n);
        let vec[@ty.t] prefix = vec();
        let int i = 0;
        while (i < ix) {
            append[@ty.t](prefix, ty.get_element_type(t, i as uint));
            i +=1 ;
        }

        auto selected = ty.get_element_type(t, i as uint);

        if (n == len-1u) {
            // We are at the innermost index.
            ret rec(prefix=prefix, target=selected);

        } else {
            // Not the innermost index; call self recursively to dig deeper.
            // Once we get an inner result, append it current prefix and
            // return to caller.
            auto inner = split_type(selected, ixs, n+1u);
            prefix += inner.prefix;
            ret rec(prefix=prefix with inner);
        }
    }

    // We make a fake prefix tuple-type here; luckily for measuring sizes
    // the tuple parens are associative so it doesn't matter that we've
    // flattened the incoming structure.

    auto s = split_type(t, ixs, 0u);
    auto prefix_ty = ty.plain_ty(ty.ty_tup(s.prefix));
    auto sz = size_of(cx, prefix_ty);
    auto raw = cx.build.PointerCast(base, T_ptr(T_i8()));
    auto bumped = cx.build.GEP(raw, vec(sz));
    alt (s.target.struct) {
        case (ty.ty_param(_)) { ret bumped; }
        case (_) {
            auto ty = T_ptr(type_of(cx.fcx.ccx, s.target));
            ret cx.build.PointerCast(bumped, ty);
        }
    }
}


fn trans_malloc_inner(@block_ctxt cx, TypeRef llptr_ty) -> result {
    auto llbody_ty = lib.llvm.llvm.LLVMGetElementType(llptr_ty);
    // FIXME: need a table to collect tydesc globals.
    auto tydesc = C_int(0);
    auto sz = llsize_of(llbody_ty);
    auto sub = trans_upcall(cx, "upcall_malloc", vec(sz, tydesc));
    sub.val = sub.bcx.build.IntToPtr(sub.val, llptr_ty);
    ret sub;
}

fn trans_malloc(@block_ctxt cx, @ty.t t) -> result {
    auto scope_cx = find_scope_cx(cx);
    auto llptr_ty = type_of(cx.fcx.ccx, t);
    auto sub = trans_malloc_inner(cx, llptr_ty);
    scope_cx.cleanups += clean(bind drop_ty(_, sub.val, t));
    ret sub;
}


// Type descriptor and type glue stuff

// Given a type and a field index into its corresponding type descriptor,
// returns an LLVM ValueRef of that field from the tydesc, generating the
// tydesc if necessary.
fn field_of_tydesc(@block_ctxt cx, @ty.t t, int field) -> ValueRef {
    auto tydesc = get_tydesc(cx, t);
    ret cx.build.GEP(tydesc, vec(C_int(0), C_int(field)));
}

// Given a type containing ty params, build a vector containing a ValueRef for
// each of the ty params it uses (from the current frame), as well as a vec
// containing a def_id for each such param. This is used solely for
// constructing derived tydescs.
fn linearize_ty_params(@block_ctxt cx, @ty.t t)
    -> tup(vec[ast.def_id], vec[ValueRef]) {
    let vec[ValueRef] param_vals = vec();
    let vec[ast.def_id] param_defs = vec();
    type rr = rec(@block_ctxt cx,
                 mutable vec[ValueRef] vals,
                 mutable vec[ast.def_id] defs);

    state obj folder(@rr r) {
        fn fold_simple_ty(@ty.t t) -> @ty.t {
            alt(t.struct) {
                case (ty.ty_param(?pid)) {
                    let bool seen = false;
                    for (ast.def_id d in r.defs) {
                        if (d == pid) {
                            seen = true;
                        }
                    }
                    if (!seen) {
                        r.vals += cx.fcx.lltydescs.get(pid);
                        r.defs += pid;
                    }
                }
            }
            ret t;
        }
    }


    auto x = @rec(cx = cx,
                  mutable vals = param_vals,
                  mutable defs = param_defs);

    ty.fold_ty(folder(x), t);

    ret tup(x.defs, x.vals);
}

fn get_tydesc(&@block_ctxt cx, @ty.t t) -> ValueRef {
    // Is the supplied type a type param? If so, return the passed-in tydesc.
    alt (ty.type_param(t)) {
        case (some[ast.def_id](?id)) { ret cx.fcx.lltydescs.get(id); }
        case (none[ast.def_id])      { /* fall through */ }
    }

    // Does it contain a type param? If so, generate a derived tydesc.
    let uint n_params = ty.count_ty_params(t);
    if (ty.count_ty_params(t) > 0u) {
        auto tys = linearize_ty_params(cx, t);
        cx.fcx.ccx.sess.unimpl("derived type descriptors");
    }

    // Otherwise, generate a tydesc if necessary, and return it.
    if (!cx.fcx.ccx.tydescs.contains_key(t)) {
        make_tydesc(cx.fcx.ccx, t);
    }
    ret cx.fcx.ccx.tydescs.get(t);
}

fn make_tydesc(@crate_ctxt cx, @ty.t t) {
    auto tg = make_take_glue;
    auto take_glue = make_generic_glue(cx, t, "take", tg);
    auto dg = make_drop_glue;
    auto drop_glue = make_generic_glue(cx, t, "drop", dg);

    auto llty = type_of(cx, t);
    auto pvoid = T_ptr(T_i8());
    auto glue_fn_ty = T_ptr(T_fn(vec(T_taskptr(),
                                     T_ptr(T_tydesc()),
                                     pvoid), T_void()));
    auto tydesc = C_struct(vec(C_null(T_ptr(T_tydesc())),
                               llsize_of(llty),
                               llalign_of(llty),
                               take_glue,             // take_glue_off
                               drop_glue,             // drop_glue_off
                               C_null(glue_fn_ty),    // free_glue_off
                               C_null(glue_fn_ty),    // sever_glue_off
                               C_null(glue_fn_ty),    // mark_glue_off
                               C_null(glue_fn_ty),    // obj_drop_glue_off
                               C_null(glue_fn_ty)));  // is_stateful

    auto name = sanitize(cx.names.next("tydesc_" + ty.ty_to_str(t)));
    auto gvar = llvm.LLVMAddGlobal(cx.llmod, val_ty(tydesc), _str.buf(name));
    llvm.LLVMSetInitializer(gvar, tydesc);
    llvm.LLVMSetGlobalConstant(gvar, True);
    llvm.LLVMSetLinkage(gvar, lib.llvm.LLVMPrivateLinkage
                        as llvm.Linkage);
    cx.tydescs.insert(t, gvar);
}

fn make_generic_glue(@crate_ctxt cx, @ty.t t, str name,
                     val_and_ty_fn helper) -> ValueRef {
    auto llfnty = T_fn(vec(T_taskptr(),
                           T_ptr(T_tydesc()),
                           T_ptr(T_i8())), T_void());

    auto fn_name = cx.names.next("_rust_" + name) + "." + ty.ty_to_str(t);
    fn_name = sanitize(fn_name);
    auto llfn = decl_fastcall_fn(cx.llmod, fn_name, llfnty);

    auto fcx = new_fn_ctxt(cx, fn_name, llfn);
    auto bcx = new_top_block_ctxt(fcx);

    auto re;
    if (!ty.type_is_scalar(t)) {
        auto llty;
        if (ty.type_is_structural(t)) {
            llty = T_ptr(type_of(cx, t));
        } else {
            llty = type_of(cx, t);
        }

        auto llrawptr = llvm.LLVMGetParam(llfn, 2u);
        auto llval = bcx.build.BitCast(llrawptr, llty);

        re = helper(bcx, llval, t);
    } else {
        re = res(bcx, C_nil());
    }

    re.bcx.build.RetVoid();
    ret llfn;
}

fn make_take_glue(@block_ctxt cx, ValueRef v, @ty.t t) -> result {
    if (ty.type_is_boxed(t)) {
        ret incr_refcnt_of_boxed(cx, v);

    } else if (ty.type_is_structural(t)) {
        ret iter_structural_ty(cx, v, t,
                               bind incr_all_refcnts(_, _, _));
    }
    ret res(cx, C_nil());
}

fn incr_refcnt_of_boxed(@block_ctxt cx, ValueRef box_ptr) -> result {
    auto rc_ptr = cx.build.GEP(box_ptr, vec(C_int(0),
                                            C_int(abi.box_rc_field_refcnt)));
    auto rc = cx.build.Load(rc_ptr);

    auto rc_adj_cx = new_sub_block_ctxt(cx, "rc++");
    auto next_cx = new_sub_block_ctxt(cx, "next");

    auto const_test = cx.build.ICmp(lib.llvm.LLVMIntEQ,
                                    C_int(abi.const_refcount as int), rc);
    cx.build.CondBr(const_test, next_cx.llbb, rc_adj_cx.llbb);

    rc = rc_adj_cx.build.Add(rc, C_int(1));
    rc_adj_cx.build.Store(rc, rc_ptr);
    rc_adj_cx.build.Br(next_cx.llbb);

    ret res(next_cx, C_nil());
}

fn make_drop_glue(@block_ctxt cx, ValueRef v, @ty.t t) -> result {
    alt (t.struct) {
        case (ty.ty_str) {
            ret decr_refcnt_and_if_zero
                (cx, v, bind trans_non_gc_free(_, v),
                 "free string",
                 T_int(), C_int(0));
        }

        case (ty.ty_vec(_)) {
            fn hit_zero(@block_ctxt cx, ValueRef v,
                        @ty.t t) -> result {
                auto res = iter_sequence(cx, v, t,
                                         bind drop_ty(_,_,_));
                // FIXME: switch gc/non-gc on layer of the type.
                ret trans_non_gc_free(res.bcx, v);
            }
            ret decr_refcnt_and_if_zero(cx, v,
                                        bind hit_zero(_, v, t),
                                        "free vector",
                                        T_int(), C_int(0));
        }

        case (ty.ty_box(?body_ty)) {
            fn hit_zero(@block_ctxt cx, ValueRef v,
                        @ty.t body_ty) -> result {
                auto body = cx.build.GEP(v,
                                         vec(C_int(0),
                                             C_int(abi.box_rc_field_body)));

                auto body_val = load_scalar_or_boxed(cx, body, body_ty);
                auto res = drop_ty(cx, body_val, body_ty);
                // FIXME: switch gc/non-gc on layer of the type.
                ret trans_non_gc_free(res.bcx, v);
            }
            ret decr_refcnt_and_if_zero(cx, v,
                                        bind hit_zero(_, v, body_ty),
                                        "free box",
                                        T_int(), C_int(0));
        }

        case (ty.ty_obj(_)) {
            fn hit_zero(@block_ctxt cx, ValueRef v) -> result {

                // Call through the obj's own fields-drop glue first.
                auto body =
                    cx.build.GEP(v,
                                 vec(C_int(0),
                                     C_int(abi.box_rc_field_body)));

                auto fields =
                    cx.build.GEP(body,
                                 vec(C_int(0),
                                     C_int(abi.obj_body_elt_fields)));
                auto tydescptr =
                    cx.build.GEP(body,
                                 vec(C_int(0),
                                     C_int(abi.obj_body_elt_tydesc)));

                call_tydesc_glue_full(cx, fields, cx.build.Load(tydescptr),
                                      abi.tydesc_field_drop_glue_off);

                // Then free the body.
                // FIXME: switch gc/non-gc on layer of the type.
                ret trans_non_gc_free(cx, v);
            }
            auto box_cell =
                cx.build.GEP(v,
                             vec(C_int(0),
                                 C_int(abi.obj_field_box)));

            auto boxptr = cx.build.Load(box_cell);

            ret decr_refcnt_and_if_zero(cx, boxptr,
                                        bind hit_zero(_, boxptr),
                                        "free obj",
                                        T_int(), C_int(0));
        }

        case (ty.ty_fn(_,_)) {
            fn hit_zero(@block_ctxt cx, ValueRef v) -> result {

                // Call through the closure's own fields-drop glue first.
                auto body =
                    cx.build.GEP(v,
                                 vec(C_int(0),
                                     C_int(abi.box_rc_field_body)));

                auto bindings =
                    cx.build.GEP(body,
                                 vec(C_int(0),
                                     C_int(abi.closure_elt_bindings)));

                auto tydescptr =
                    cx.build.GEP(body,
                                 vec(C_int(0),
                                     C_int(abi.closure_elt_tydesc)));

                call_tydesc_glue_full(cx, bindings, cx.build.Load(tydescptr),
                                      abi.tydesc_field_drop_glue_off);


                // Then free the body.
                // FIXME: switch gc/non-gc on layer of the type.
                ret trans_non_gc_free(cx, v);
            }
            auto box_cell =
                cx.build.GEP(v,
                             vec(C_int(0),
                                 C_int(abi.fn_field_box)));

            auto boxptr = cx.build.Load(box_cell);

            ret decr_refcnt_and_if_zero(cx, boxptr,
                                        bind hit_zero(_, boxptr),
                                        "free fn",
                                        T_int(), C_int(0));
        }

        case (_) {
            if (ty.type_is_structural(t)) {
                ret iter_structural_ty(cx, v, t,
                                       bind drop_ty(_, _, _));

            } else if (ty.type_is_scalar(t) ||
                       ty.type_is_nil(t)) {
                ret res(cx, C_nil());
            }
        }
    }
    cx.fcx.ccx.sess.bug("bad type in trans.make_drop_glue_inner: " +
                        ty.ty_to_str(t));
    fail;
}

fn decr_refcnt_and_if_zero(@block_ctxt cx,
                           ValueRef box_ptr,
                           fn(@block_ctxt cx) -> result inner,
                           str inner_name,
                           TypeRef t_else, ValueRef v_else) -> result {

    auto load_rc_cx = new_sub_block_ctxt(cx, "load rc");
    auto rc_adj_cx = new_sub_block_ctxt(cx, "rc--");
    auto inner_cx = new_sub_block_ctxt(cx, inner_name);
    auto next_cx = new_sub_block_ctxt(cx, "next");

    auto null_test = cx.build.IsNull(box_ptr);
    cx.build.CondBr(null_test, next_cx.llbb, load_rc_cx.llbb);


    auto rc_ptr = load_rc_cx.build.GEP(box_ptr,
                                       vec(C_int(0),
                                           C_int(abi.box_rc_field_refcnt)));

    auto rc = load_rc_cx.build.Load(rc_ptr);
    auto const_test =
        load_rc_cx.build.ICmp(lib.llvm.LLVMIntEQ,
                              C_int(abi.const_refcount as int), rc);
    load_rc_cx.build.CondBr(const_test, next_cx.llbb, rc_adj_cx.llbb);

    rc = rc_adj_cx.build.Sub(rc, C_int(1));
    rc_adj_cx.build.Store(rc, rc_ptr);
    auto zero_test = rc_adj_cx.build.ICmp(lib.llvm.LLVMIntEQ, C_int(0), rc);
    rc_adj_cx.build.CondBr(zero_test, inner_cx.llbb, next_cx.llbb);

    auto inner_res = inner(inner_cx);
    inner_res.bcx.build.Br(next_cx.llbb);

    auto phi = next_cx.build.Phi(t_else,
                                 vec(v_else, v_else, v_else, inner_res.val),
                                 vec(cx.llbb,
                                     load_rc_cx.llbb,
                                     rc_adj_cx.llbb,
                                     inner_res.bcx.llbb));

    ret res(next_cx, phi);
}

fn type_of_variant(@crate_ctxt cx, &ast.variant v) -> TypeRef {
    let vec[TypeRef] lltys = vec();
    alt (ty.ann_to_type(v.ann).struct) {
        case (ty.ty_fn(?args, _)) {
            for (ty.arg arg in args) {
                lltys += vec(type_of(cx, arg.ty));
            }
        }
        case (_) { fail; }
    }
    ret T_struct(lltys);
}

type val_and_ty_fn = fn(@block_ctxt cx, ValueRef v, @ty.t t) -> result;

// Iterates through the elements of a structural type.
fn iter_structural_ty(@block_ctxt cx,
                      ValueRef v,
                      @ty.t t,
                      val_and_ty_fn f)
    -> result {
    let result r = res(cx, C_nil());

    fn iter_boxpp(@block_ctxt cx,
                  ValueRef box_cell,
                  val_and_ty_fn f) -> result {
        auto box_ptr = cx.build.Load(box_cell);
        auto tnil = ty.plain_ty(ty.ty_nil);
        auto tbox = ty.plain_ty(ty.ty_box(tnil));

        auto inner_cx = new_sub_block_ctxt(cx, "iter box");
        auto next_cx = new_sub_block_ctxt(cx, "next");
        auto null_test = cx.build.IsNull(box_ptr);
        cx.build.CondBr(null_test, next_cx.llbb, inner_cx.llbb);

        auto r = f(inner_cx, box_ptr, tbox);
        r.bcx.build.Br(next_cx.llbb);
        ret res(next_cx, r.val);
    }

    alt (t.struct) {
        case (ty.ty_tup(?args)) {
            let int i = 0;
            for (@ty.t arg in args) {
                auto elt = r.bcx.build.GEP(v, vec(C_int(0), C_int(i)));
                r = f(r.bcx,
                      load_scalar_or_boxed(r.bcx, elt, arg),
                      arg);
                i += 1;
            }
        }
        case (ty.ty_rec(?fields)) {
            let int i = 0;
            for (ty.field fld in fields) {
                auto llfld = r.bcx.build.GEP(v, vec(C_int(0), C_int(i)));
                r = f(r.bcx,
                      load_scalar_or_boxed(r.bcx, llfld, fld.ty),
                      fld.ty);
                i += 1;
            }
        }
        case (ty.ty_tag(?tid)) {
            check (cx.fcx.ccx.tags.contains_key(tid));
            auto info = cx.fcx.ccx.tags.get(tid);
            auto n_variants = _vec.len[tup(ast.def_id,arity)](info.variants);

            // Look up the tag in the typechecked AST.
            check (cx.fcx.ccx.items.contains_key(tid));
            auto tag_item = cx.fcx.ccx.items.get(tid);
            let vec[ast.variant] variants = vec();  // FIXME: typestate bug
            alt (tag_item.node) {
                case (ast.item_tag(_, ?vs, _, _)) {
                    variants = vs;
                }
                case (_) {
                    log "trans: ty_tag doesn't actually refer to a tag";
                    fail;
                }
            }

            auto lldiscrim_ptr = cx.build.GEP(v, vec(C_int(0), C_int(0)));
            auto llunion_ptr = cx.build.GEP(v, vec(C_int(0), C_int(1)));
            auto lldiscrim = cx.build.Load(lldiscrim_ptr);

            auto unr_cx = new_sub_block_ctxt(cx, "tag-iter-unr");
            unr_cx.build.Unreachable();

            auto llswitch = cx.build.Switch(lldiscrim, unr_cx.llbb,
                                            n_variants);

            auto next_cx = new_sub_block_ctxt(cx, "tag-iter-next");

            auto i = 0u;
            for (tup(ast.def_id,arity) variant in info.variants) {
                auto variant_cx = new_sub_block_ctxt(cx, "tag-iter-variant-" +
                                                     _uint.to_str(i, 10u));
                llvm.LLVMAddCase(llswitch, C_int(i as int), variant_cx.llbb);

                alt (variant._1) {
                    case (n_ary) {
                        let vec[ValueRef] vals = vec(C_int(0), C_int(1),
                                                     C_int(i as int));
                        auto llvar = variant_cx.build.GEP(v, vals);
                        auto llvarty = type_of_variant(cx.fcx.ccx,
                                                       variants.(i));

                        auto fn_ty = ty.ann_to_type(variants.(i).ann);
                        alt (fn_ty.struct) {
                            case (ty.ty_fn(?args, _)) {
                                auto llvarp = variant_cx.build.
                                    TruncOrBitCast(llunion_ptr,
                                                   T_ptr(llvarty));

                                auto j = 0u;
                                for (ty.arg a in args) {
                                    auto v = vec(C_int(0),
                                                 C_int(j as int));
                                    auto llfldp =
                                        variant_cx.build.GEP(llvarp, v);

                                    auto llfld =
                                        load_scalar_or_boxed(variant_cx,
                                                             llfldp, a.ty);

                                    auto res = f(variant_cx, llfld, a.ty);
                                    variant_cx = res.bcx;
                                    j += 1u;
                                }
                            }
                            case (_) { fail; }
                        }

                        variant_cx.build.Br(next_cx.llbb);
                    }
                    case (nullary) {
                        // Nothing to do.
                        variant_cx.build.Br(next_cx.llbb);
                    }
                }

                i += 1u;
            }

            ret res(next_cx, C_nil());
        }
        case (ty.ty_fn(_,_)) {
            auto box_cell =
                cx.build.GEP(v,
                             vec(C_int(0),
                                 C_int(abi.fn_field_box)));
            ret iter_boxpp(cx, box_cell, f);
        }
        case (ty.ty_obj(_)) {
            auto box_cell =
                cx.build.GEP(v,
                             vec(C_int(0),
                                 C_int(abi.obj_field_box)));
            ret iter_boxpp(cx, box_cell, f);
        }
        case (_) {
            cx.fcx.ccx.sess.unimpl("type in iter_structural_ty");
        }
    }
    ret r;
}

// Iterates through the elements of a vec or str.
fn iter_sequence(@block_ctxt cx,
                 ValueRef v,
                 @ty.t t,
                 val_and_ty_fn f) -> result {

    fn iter_sequence_body(@block_ctxt cx,
                          ValueRef v,
                          @ty.t elt_ty,
                          val_and_ty_fn f,
                          bool trailing_null) -> result {

        auto p0 = cx.build.GEP(v, vec(C_int(0),
                                      C_int(abi.vec_elt_data)));
        auto lenptr = cx.build.GEP(v, vec(C_int(0),
                                          C_int(abi.vec_elt_fill)));

        auto llunit_ty = type_of(cx.fcx.ccx, elt_ty);
        auto unit_sz = size_of(cx, elt_ty);

        auto len = cx.build.Load(lenptr);
        if (trailing_null) {
            len = cx.build.Sub(len, unit_sz);
        }

        auto r = res(cx, C_nil());

        auto cond_cx = new_scope_block_ctxt(cx, "sequence-iter cond");
        auto body_cx = new_scope_block_ctxt(cx, "sequence-iter body");
        auto next_cx = new_sub_block_ctxt(cx, "next");

        cx.build.Br(cond_cx.llbb);

        auto ix = cond_cx.build.Phi(T_int(), vec(C_int(0)), vec(cx.llbb));
        auto scaled_ix = cond_cx.build.Phi(T_int(),
                                           vec(C_int(0)), vec(cx.llbb));

        auto end_test = cond_cx.build.ICmp(lib.llvm.LLVMIntNE,
                                           scaled_ix, len);
        cond_cx.build.CondBr(end_test, body_cx.llbb, next_cx.llbb);

        auto elt = body_cx.build.GEP(p0, vec(C_int(0), ix));
        auto body_res = f(body_cx,
                          load_scalar_or_boxed(body_cx, elt, elt_ty),
                          elt_ty);
        auto next_ix = body_res.bcx.build.Add(ix, C_int(1));
        auto next_scaled_ix = body_res.bcx.build.Add(scaled_ix, unit_sz);

        cond_cx.build.AddIncomingToPhi(ix, vec(next_ix),
                                       vec(body_res.bcx.llbb));

        cond_cx.build.AddIncomingToPhi(scaled_ix, vec(next_scaled_ix),
                                       vec(body_res.bcx.llbb));

        body_res.bcx.build.Br(cond_cx.llbb);
        ret res(next_cx, C_nil());
    }

    alt (t.struct) {
        case (ty.ty_vec(?et)) {
            ret iter_sequence_body(cx, v, et, f, false);
        }
        case (ty.ty_str) {
            auto et = ty.plain_ty(ty.ty_machine(common.ty_u8));
            ret iter_sequence_body(cx, v, et, f, true);
        }
        case (_) { fail; }
    }
    cx.fcx.ccx.sess.bug("bad type in trans.iter_sequence");
    fail;
}

fn call_tydesc_glue_full(@block_ctxt cx, ValueRef v,
                         ValueRef tydesc, int field) {
    auto llrawptr = cx.build.BitCast(v, T_ptr(T_i8()));
    auto lltydescs = cx.build.GEP(tydesc,
                                  vec(C_int(0),
                                      C_int(abi.tydesc_field_first_param)));
    lltydescs = cx.build.Load(lltydescs);
    auto llfnptr = cx.build.GEP(tydesc, vec(C_int(0), C_int(field)));
    auto llfn = cx.build.Load(llfnptr);
    cx.build.FastCall(llfn, vec(cx.fcx.lltaskptr, lltydescs, llrawptr));
}

fn call_tydesc_glue(@block_ctxt cx, ValueRef v, @ty.t t, int field) {
    call_tydesc_glue_full(cx, v, get_tydesc(cx, t), field);
}

fn incr_all_refcnts(@block_ctxt cx,
                    ValueRef v,
                    @ty.t t) -> result {

    if (!ty.type_is_scalar(t)) {
        call_tydesc_glue(cx, v, t, abi.tydesc_field_take_glue_off);
    }
    ret res(cx, C_nil());
}

fn drop_slot(@block_ctxt cx,
             ValueRef slot,
             @ty.t t) -> result {
    auto llptr = load_scalar_or_boxed(cx, slot, t);
    auto re = drop_ty(cx, llptr, t);

    auto llty = val_ty(slot);
    auto llelemty = lib.llvm.llvm.LLVMGetElementType(llty);
    re.bcx.build.Store(C_null(llelemty), slot);
    ret re;
}

fn drop_ty(@block_ctxt cx,
           ValueRef v,
           @ty.t t) -> result {

    if (!ty.type_is_scalar(t)) {
        call_tydesc_glue(cx, v, t, abi.tydesc_field_drop_glue_off);
    }
    ret res(cx, C_nil());
}

fn call_memcpy(@block_ctxt cx,
               ValueRef dst,
               ValueRef src,
               ValueRef n_bytes) -> result {
    auto src_ptr = cx.build.PointerCast(src, T_ptr(T_i8()));
    auto dst_ptr = cx.build.PointerCast(dst, T_ptr(T_i8()));
    auto size = cx.build.IntCast(n_bytes, T_int());
    ret res(cx, cx.build.FastCall(cx.fcx.ccx.glues.memcpy_glue,
                                  vec(dst_ptr, src_ptr, size)));
}

fn call_bzero(@block_ctxt cx,
              ValueRef dst,
              ValueRef n_bytes) -> result {
    auto dst_ptr = cx.build.PointerCast(dst, T_ptr(T_i8()));
    auto size = cx.build.IntCast(n_bytes, T_int());
    ret res(cx, cx.build.FastCall(cx.fcx.ccx.glues.bzero_glue,
                                  vec(dst_ptr, size)));
}

fn memcpy_ty(@block_ctxt cx,
             ValueRef dst,
             ValueRef src,
             @ty.t t) -> result {
    if (ty.type_has_dynamic_size(t)) {
        auto llszptr = field_of_tydesc(cx, t, abi.tydesc_field_size);
        auto llsz = cx.build.Load(llszptr);
        ret call_memcpy(cx, dst, src, llsz);

    } else {
        ret res(cx, cx.build.Store(cx.build.Load(src), dst));
    }
}

tag copy_action {
    INIT;
    DROP_EXISTING;
}

fn copy_ty(@block_ctxt cx,
           copy_action action,
           ValueRef dst,
           ValueRef src,
           @ty.t t) -> result {
    if (ty.type_is_scalar(t)) {
        ret res(cx, cx.build.Store(src, dst));

    } else if (ty.type_is_nil(t)) {
        ret res(cx, C_nil());

    } else if (ty.type_is_boxed(t)) {
        auto r = incr_all_refcnts(cx, src, t);
        if (action == DROP_EXISTING) {
            r = drop_ty(r.bcx, r.bcx.build.Load(dst), t);
        }
        ret res(r.bcx, r.bcx.build.Store(src, dst));

    } else if (ty.type_is_structural(t) ||
               ty.type_has_dynamic_size(t)) {
        auto r = incr_all_refcnts(cx, src, t);
        if (action == DROP_EXISTING) {
            r = drop_ty(r.bcx, dst, t);
        }
        ret memcpy_ty(r.bcx, dst, src, t);
    }

    cx.fcx.ccx.sess.bug("unexpected type in trans.copy_ty: " +
                        ty.ty_to_str(t));
    fail;
}

fn trans_lit(@crate_ctxt cx, &ast.lit lit, &ast.ann ann) -> ValueRef {
    alt (lit.node) {
        case (ast.lit_int(?i)) {
            ret C_int(i);
        }
        case (ast.lit_uint(?u)) {
            ret C_int(u as int);
        }
        case (ast.lit_mach_int(?tm, ?i)) {
            // FIXME: the entire handling of mach types falls apart
            // if target int width is larger than host, at the moment;
            // re-do the mach-int types using 'big' when that works.
            auto t = T_int();
            alt (tm) {
                case (common.ty_u8) { t = T_i8(); }
                case (common.ty_u16) { t = T_i16(); }
                case (common.ty_u32) { t = T_i32(); }
                case (common.ty_u64) { t = T_i64(); }

                case (common.ty_i8) { t = T_i8(); }
                case (common.ty_i16) { t = T_i16(); }
                case (common.ty_i32) { t = T_i32(); }
                case (common.ty_i64) { t = T_i64(); }
            }
            ret C_integral(i, t);
        }
        case (ast.lit_char(?c)) {
            ret C_integral(c as int, T_char());
        }
        case (ast.lit_bool(?b)) {
            ret C_bool(b);
        }
        case (ast.lit_nil) {
            ret C_nil();
        }
        case (ast.lit_str(?s)) {
            ret C_str(cx, s);
        }
    }
}

fn target_type(@crate_ctxt cx, @ty.t t) -> @ty.t {
    alt (t.struct) {
        case (ty.ty_int) {
            auto tm = ty.ty_machine(cx.sess.get_targ_cfg().int_type);
            ret @rec(struct=tm with *t);
        }
        case (ty.ty_uint) {
            auto tm = ty.ty_machine(cx.sess.get_targ_cfg().uint_type);
            ret @rec(struct=tm with *t);
        }
        case (_) { /* fall through */ }
    }
    ret t;
}

fn node_ann_type(@crate_ctxt cx, &ast.ann a) -> @ty.t {
    alt (a) {
        case (ast.ann_none) {
            cx.sess.bug("missing type annotation");
        }
        case (ast.ann_type(?t)) {
            ret target_type(cx, t);
        }
    }
}

fn node_type(@crate_ctxt cx, &ast.ann a) -> TypeRef {
    ret type_of(cx, node_ann_type(cx, a));
}

fn trans_unary(@block_ctxt cx, ast.unop op,
               @ast.expr e, &ast.ann a) -> result {

    auto sub = trans_expr(cx, e);

    alt (op) {
        case (ast.bitnot) {
            sub = autoderef(sub.bcx, sub.val, ty.expr_ty(e));
            ret res(sub.bcx, cx.build.Not(sub.val));
        }
        case (ast.not) {
            sub = autoderef(sub.bcx, sub.val, ty.expr_ty(e));
            ret res(sub.bcx, cx.build.Not(sub.val));
        }
        case (ast.neg) {
            sub = autoderef(sub.bcx, sub.val, ty.expr_ty(e));
            ret res(sub.bcx, cx.build.Neg(sub.val));
        }
        case (ast.box) {
            auto e_ty = ty.expr_ty(e);
            auto e_val = sub.val;
            sub = trans_malloc(sub.bcx, node_ann_type(sub.bcx.fcx.ccx, a));
            auto box = sub.val;
            auto rc = sub.bcx.build.GEP(box,
                                        vec(C_int(0),
                                            C_int(abi.box_rc_field_refcnt)));
            auto body = sub.bcx.build.GEP(box,
                                          vec(C_int(0),
                                              C_int(abi.box_rc_field_body)));
            sub.bcx.build.Store(C_int(1), rc);
            sub = copy_ty(sub.bcx, INIT, body, e_val, e_ty);
            ret res(sub.bcx, box);
        }
        case (ast.deref) {
            auto val = sub.bcx.build.GEP(sub.val,
                                         vec(C_int(0),
                                             C_int(abi.box_rc_field_body)));
            auto e_ty = node_ann_type(sub.bcx.fcx.ccx, a);
            if (ty.type_is_scalar(e_ty) ||
                ty.type_is_nil(e_ty)) {
                val = sub.bcx.build.Load(val);
            }
            ret res(sub.bcx, val);
        }
    }
    fail;
}

fn trans_eager_binop(@block_ctxt cx, ast.binop op,
                     ValueRef lhs, ValueRef rhs) -> ValueRef {

    alt (op) {
        case (ast.add) { ret cx.build.Add(lhs, rhs); }
        case (ast.sub) { ret cx.build.Sub(lhs, rhs); }

        // FIXME (issue #57): switch by signedness.
        case (ast.mul) { ret cx.build.Mul(lhs, rhs); }
        case (ast.div) { ret cx.build.SDiv(lhs, rhs); }
        case (ast.rem) { ret cx.build.SRem(lhs, rhs); }

        case (ast.bitor) { ret cx.build.Or(lhs, rhs); }
        case (ast.bitand) { ret cx.build.And(lhs, rhs); }
        case (ast.bitxor) { ret cx.build.Xor(lhs, rhs); }
        case (ast.lsl) { ret cx.build.Shl(lhs, rhs); }
        case (ast.lsr) { ret cx.build.LShr(lhs, rhs); }
        case (ast.asr) { ret cx.build.AShr(lhs, rhs); }
        case (_) {
            auto cmp = lib.llvm.LLVMIntEQ;
            alt (op) {
                case (ast.eq) { cmp = lib.llvm.LLVMIntEQ; }
                case (ast.ne) { cmp = lib.llvm.LLVMIntNE; }

                // FIXME (issue #57): switch by signedness.
                case (ast.lt) { cmp = lib.llvm.LLVMIntSLT; }
                case (ast.le) { cmp = lib.llvm.LLVMIntSLE; }
                case (ast.ge) { cmp = lib.llvm.LLVMIntSGE; }
                case (ast.gt) { cmp = lib.llvm.LLVMIntSGT; }
            }
            ret cx.build.ICmp(cmp, lhs, rhs);
        }
    }
    fail;
}

fn autoderef(@block_ctxt cx, ValueRef v, @ty.t t) -> result {
    let ValueRef v1 = v;
    let @ty.t t1 = t;

    while (true) {
        alt (t1.struct) {
            case (ty.ty_box(?inner)) {
                auto body = cx.build.GEP(v1,
                                         vec(C_int(0),
                                             C_int(abi.box_rc_field_body)));
                t1 = inner;
                v1 = load_scalar_or_boxed(cx, body, inner);
            }
            case (_) {
                ret res(cx, v1);
            }
        }
    }
}

fn trans_binary(@block_ctxt cx, ast.binop op,
                @ast.expr a, @ast.expr b) -> result {

    // First couple cases are lazy:

    alt (op) {
        case (ast.and) {
            // Lazy-eval and
            auto lhs_res = trans_expr(cx, a);
            lhs_res = autoderef(lhs_res.bcx, lhs_res.val, ty.expr_ty(a));

            auto rhs_cx = new_scope_block_ctxt(cx, "rhs");
            auto rhs_res = trans_expr(rhs_cx, b);
            rhs_res = autoderef(rhs_res.bcx, rhs_res.val, ty.expr_ty(b));

            auto lhs_false_cx = new_scope_block_ctxt(cx, "lhs false");
            auto lhs_false_res = res(lhs_false_cx, C_bool(false));

            lhs_res.bcx.build.CondBr(lhs_res.val,
                                     rhs_cx.llbb,
                                     lhs_false_cx.llbb);

            ret join_results(cx, T_bool(),
                             vec(lhs_false_res, rhs_res));
        }

        case (ast.or) {
            // Lazy-eval or
            auto lhs_res = trans_expr(cx, a);
            lhs_res = autoderef(lhs_res.bcx, lhs_res.val, ty.expr_ty(a));

            auto rhs_cx = new_scope_block_ctxt(cx, "rhs");
            auto rhs_res = trans_expr(rhs_cx, b);
            rhs_res = autoderef(rhs_res.bcx, rhs_res.val, ty.expr_ty(b));

            auto lhs_true_cx = new_scope_block_ctxt(cx, "lhs true");
            auto lhs_true_res = res(lhs_true_cx, C_bool(true));

            lhs_res.bcx.build.CondBr(lhs_res.val,
                                     lhs_true_cx.llbb,
                                     rhs_cx.llbb);

            ret join_results(cx, T_bool(),
                             vec(lhs_true_res, rhs_res));
        }

        case (_) {
            // Remaining cases are eager:
            auto lhs = trans_expr(cx, a);
            lhs = autoderef(lhs.bcx, lhs.val, ty.expr_ty(a));
            auto rhs = trans_expr(lhs.bcx, b);
            rhs = autoderef(rhs.bcx, rhs.val, ty.expr_ty(b));
            ret res(rhs.bcx, trans_eager_binop(rhs.bcx, op,
                                               lhs.val, rhs.val));
        }
    }
    fail;
}

fn join_results(@block_ctxt parent_cx,
                TypeRef t,
                vec[result] ins)
    -> result {

    let vec[result] live = vec();
    let vec[ValueRef] vals = vec();
    let vec[BasicBlockRef] bbs = vec();

    for (result r in ins) {
        if (! is_terminated(r.bcx)) {
            live += r;
            vals += r.val;
            bbs += r.bcx.llbb;
        }
    }

    alt (_vec.len[result](live)) {
        case (0u) {
            // No incoming edges are live, so we're in dead-code-land.
            // Arbitrarily pick the first dead edge, since the caller
            // is just going to propagate it outward.
            check (_vec.len[result](ins) >= 1u);
            ret ins.(0);
        }

        case (1u) {
            // Only one incoming edge is live, so we just feed that block
            // onward.
            ret live.(0);
        }

        case (_) { /* fall through */ }
    }

    // We have >1 incoming edges. Make a join block and br+phi them into it.
    auto join_cx = new_sub_block_ctxt(parent_cx, "join");
    for (result r in live) {
        r.bcx.build.Br(join_cx.llbb);
    }
    auto phi = join_cx.build.Phi(t, vals, bbs);
    ret res(join_cx, phi);
}

fn trans_if(@block_ctxt cx, @ast.expr cond,
            &ast.block thn, &option.t[ast.block] els) -> result {

    auto cond_res = trans_expr(cx, cond);

    auto then_cx = new_scope_block_ctxt(cx, "then");
    auto then_res = trans_block(then_cx, thn);

    auto else_cx = new_scope_block_ctxt(cx, "else");
    auto else_res = res(else_cx, C_nil());

    alt (els) {
        case (some[ast.block](?eblk)) {
            else_res = trans_block(else_cx, eblk);
        }
        case (_) { /* fall through */ }
    }

    cond_res.bcx.build.CondBr(cond_res.val,
                              then_cx.llbb,
                              else_cx.llbb);

    // FIXME: use inferred type when available.
    ret join_results(cx, T_nil(),
                     vec(then_res, else_res));
}

fn trans_for(@block_ctxt cx,
             @ast.decl decl,
             @ast.expr seq,
             &ast.block body) -> result {

    fn inner(@block_ctxt cx,
             @ast.local local, ValueRef curr,
             @ty.t t, ast.block body) -> result {

        auto scope_cx = new_scope_block_ctxt(cx, "for loop scope");
        auto next_cx = new_sub_block_ctxt(cx, "next");

        cx.build.Br(scope_cx.llbb);
        auto local_res = alloc_local(scope_cx, local);
        auto bcx = copy_ty(local_res.bcx, INIT, local_res.val, curr, t).bcx;
        bcx = trans_block(bcx, body).bcx;
        bcx.build.Br(next_cx.llbb);
        ret res(next_cx, C_nil());
    }


    let @ast.local local;
    alt (decl.node) {
        case (ast.decl_local(?loc)) {
            local = loc;
        }
    }

    auto seq_ty = ty.expr_ty(seq);
    auto seq_res = trans_expr(cx, seq);
    ret iter_sequence(seq_res.bcx, seq_res.val, seq_ty,
                      bind inner(_, local, _, _, body));
}

fn trans_while(@block_ctxt cx, @ast.expr cond,
               &ast.block body) -> result {

    auto cond_cx = new_scope_block_ctxt(cx, "while cond");
    auto body_cx = new_scope_block_ctxt(cx, "while loop body");
    auto next_cx = new_sub_block_ctxt(cx, "next");

    auto body_res = trans_block(body_cx, body);
    auto cond_res = trans_expr(cond_cx, cond);

    body_res.bcx.build.Br(cond_cx.llbb);
    cond_res.bcx.build.CondBr(cond_res.val,
                              body_cx.llbb,
                              next_cx.llbb);

    cx.build.Br(cond_cx.llbb);
    ret res(next_cx, C_nil());
}

fn trans_do_while(@block_ctxt cx, &ast.block body,
                  @ast.expr cond) -> result {

    auto body_cx = new_scope_block_ctxt(cx, "do-while loop body");
    auto next_cx = new_sub_block_ctxt(cx, "next");

    auto body_res = trans_block(body_cx, body);
    auto cond_res = trans_expr(body_res.bcx, cond);

    cond_res.bcx.build.CondBr(cond_res.val,
                              body_cx.llbb,
                              next_cx.llbb);
    cx.build.Br(body_cx.llbb);
    ret res(next_cx, body_res.val);
}

// Pattern matching translation

// Returns a pointer to the union part of the LLVM representation of a tag
// type, cast to the appropriate type.
fn get_pat_union_ptr(@block_ctxt cx, vec[@ast.pat] subpats, ValueRef llval)
    -> ValueRef {
    auto llblobptr = cx.build.GEP(llval, vec(C_int(0), C_int(1)));

    // Generate the union type.
    let vec[TypeRef] llsubpattys = vec();
    for (@ast.pat subpat in subpats) {
        llsubpattys += vec(type_of(cx.fcx.ccx, pat_ty(subpat)));
    }

    // Recursively check subpatterns.
    auto llunionty = T_struct(llsubpattys);
    ret cx.build.TruncOrBitCast(llblobptr, T_ptr(llunionty));
}

fn trans_pat_match(@block_ctxt cx, @ast.pat pat, ValueRef llval,
                   @block_ctxt next_cx) -> result {
    alt (pat.node) {
        case (ast.pat_wild(_)) { ret res(cx, llval); }
        case (ast.pat_bind(_, _, _)) { ret res(cx, llval); }
        case (ast.pat_tag(?id, ?subpats, ?vdef_opt, ?ann)) {
            auto lltagptr = cx.build.GEP(llval, vec(C_int(0), C_int(0)));
            auto lltag = cx.build.Load(lltagptr);

            auto vdef = option.get[ast.variant_def](vdef_opt);
            auto variant_id = vdef._1;
            auto tinfo = cx.fcx.ccx.tags.get(vdef._0);
            auto variant_tag = 0;
            auto i = 0;
            for (tup(ast.def_id,arity) vinfo in tinfo.variants) {
                auto this_variant_id = vinfo._0;
                if (variant_id._0 == this_variant_id._0 &&
                    variant_id._1 == this_variant_id._1) {
                    variant_tag = i;
                }
                i += 1;
            }

            auto matched_cx = new_sub_block_ctxt(cx, "matched_cx");

            auto lleq = cx.build.ICmp(lib.llvm.LLVMIntEQ, lltag,
                                      C_int(variant_tag));
            cx.build.CondBr(lleq, matched_cx.llbb, next_cx.llbb);

            if (_vec.len[@ast.pat](subpats) > 0u) {
                auto llunionptr = get_pat_union_ptr(matched_cx, subpats,
                                                    llval);
                auto i = 0;
                for (@ast.pat subpat in subpats) {
                    auto llsubvalptr = matched_cx.build.GEP(llunionptr,
                                                            vec(C_int(0),
                                                                C_int(i)));
                    auto llsubval = load_scalar_or_boxed(matched_cx,
                                                         llsubvalptr,
                                                         pat_ty(subpat));
                    auto subpat_res = trans_pat_match(matched_cx, subpat,
                                                      llsubval, next_cx);
                    matched_cx = subpat_res.bcx;
                }
            }

            ret res(matched_cx, llval);
        }
    }

    fail;
}

fn trans_pat_binding(@block_ctxt cx, @ast.pat pat, ValueRef llval)
    -> result {
    alt (pat.node) {
        case (ast.pat_wild(_)) { ret res(cx, llval); }
        case (ast.pat_bind(?id, ?def_id, ?ann)) {
            auto ty = node_ann_type(cx.fcx.ccx, ann);
            auto llty = type_of(cx.fcx.ccx, ty);

            auto dst = cx.build.Alloca(llty);
            llvm.LLVMSetValueName(dst, _str.buf(id));
            cx.fcx.lllocals.insert(def_id, dst);
            cx.cleanups += clean(bind drop_slot(_, dst, ty));

            ret copy_ty(cx, INIT, dst, llval, ty);
        }
        case (ast.pat_tag(_, ?subpats, _, _)) {
            if (_vec.len[@ast.pat](subpats) == 0u) { ret res(cx, llval); }

            auto llunionptr = get_pat_union_ptr(cx, subpats, llval);

            auto this_cx = cx;
            auto i = 0;
            for (@ast.pat subpat in subpats) {
                auto llsubvalptr = this_cx.build.GEP(llunionptr,
                                                     vec(C_int(0), C_int(i)));
                auto llsubval = load_scalar_or_boxed(this_cx, llsubvalptr,
                                                     pat_ty(subpat));
                auto subpat_res = trans_pat_binding(this_cx, subpat,
                                                    llsubval);
                this_cx = subpat_res.bcx;
                i += 1;
            }

            ret res(this_cx, llval);
        }
    }
}

fn trans_alt(@block_ctxt cx, @ast.expr expr, vec[ast.arm] arms)
    -> result {
    auto expr_res = trans_expr(cx, expr);

    auto last_cx = new_sub_block_ctxt(expr_res.bcx, "last");

    auto this_cx = expr_res.bcx;
    for (ast.arm arm in arms) {
        auto next_cx = new_sub_block_ctxt(expr_res.bcx, "next");
        auto match_res = trans_pat_match(this_cx, arm.pat, expr_res.val,
                                         next_cx);

        auto binding_cx = new_scope_block_ctxt(match_res.bcx, "binding");
        match_res.bcx.build.Br(binding_cx.llbb);

        auto binding_res = trans_pat_binding(binding_cx, arm.pat,
                                             expr_res.val);

        auto block_res = trans_block(binding_res.bcx, arm.block);
        if (!is_terminated(block_res.bcx)) {
            block_res.bcx.build.Br(last_cx.llbb);
        }

        this_cx = next_cx;
    }

    // FIXME: This is executed when none of the patterns match; it should fail
    // instead!
    this_cx.build.Br(last_cx.llbb);

    // FIXME: This is very wrong; we should phi together all the arm blocks,
    // since this is an expression.
    ret res(last_cx, C_nil());
}

type generic_info = rec(@ty.t item_type,
                        vec[ValueRef] tydescs);

type lval_result = rec(result res,
                       bool is_mem,
                       option.t[generic_info] generic,
                       option.t[ValueRef] llobj);

fn lval_mem(@block_ctxt cx, ValueRef val) -> lval_result {
    ret rec(res=res(cx, val),
            is_mem=true,
            generic=none[generic_info],
            llobj=none[ValueRef]);
}

fn lval_val(@block_ctxt cx, ValueRef val) -> lval_result {
    ret rec(res=res(cx, val),
            is_mem=false,
            generic=none[generic_info],
            llobj=none[ValueRef]);
}

fn trans_path(@block_ctxt cx, &ast.path p, &option.t[ast.def] dopt,
              &ast.ann ann) -> lval_result {
    alt (dopt) {
        case (some[ast.def](?def)) {
            alt (def) {
                case (ast.def_arg(?did)) {
                    check (cx.fcx.llargs.contains_key(did));
                    ret lval_mem(cx, cx.fcx.llargs.get(did));
                }
                case (ast.def_local(?did)) {
                    check (cx.fcx.lllocals.contains_key(did));
                    ret lval_mem(cx, cx.fcx.lllocals.get(did));
                }
                case (ast.def_binding(?did)) {
                    check (cx.fcx.lllocals.contains_key(did));
                    ret lval_mem(cx, cx.fcx.lllocals.get(did));
                }
                case (ast.def_obj_field(?did)) {
                    check (cx.fcx.llobjfields.contains_key(did));
                    ret lval_mem(cx, cx.fcx.llobjfields.get(did));
                }
                case (ast.def_fn(?did)) {
                    check (cx.fcx.ccx.fn_pairs.contains_key(did));
                    check (cx.fcx.ccx.item_ids.contains_key(did));

                    auto fn_item = cx.fcx.ccx.items.get(did);
                    auto lv = lval_val(cx, cx.fcx.ccx.fn_pairs.get(did));
                    auto monoty = node_ann_type(cx.fcx.ccx, ann);
                    auto tys = ty.resolve_ty_params(fn_item, monoty);

                    if (_vec.len[@ty.t](tys) != 0u) {
                        let vec[ValueRef] tydescs = vec();
                        for (@ty.t t in tys) {
                            append[ValueRef](tydescs,
                                             get_tydesc(cx, t));
                        }
                        auto gen = rec( item_type = ty.item_ty(fn_item)._1,
                                        tydescs = tydescs );
                        lv = rec(generic = some[generic_info](gen)
                                 with lv);
                    }

                    ret lv;
                }
                case (ast.def_obj(?did)) {
                    check (cx.fcx.ccx.fn_pairs.contains_key(did));
                    ret lval_val(cx, cx.fcx.ccx.fn_pairs.get(did));
                }
                case (ast.def_variant(?tid, ?vid)) {
                    check (cx.fcx.ccx.tags.contains_key(tid));
                    if (cx.fcx.ccx.fn_pairs.contains_key(vid)) {
                        ret lval_val(cx, cx.fcx.ccx.fn_pairs.get(vid));
                    } else {
                        // Nullary variants are just scalar constants.
                        check (cx.fcx.ccx.item_ids.contains_key(vid));
                        ret lval_val(cx, cx.fcx.ccx.item_ids.get(vid));
                    }
                }
                case (ast.def_const(?did)) {
                    check (cx.fcx.ccx.consts.contains_key(did));
                    ret lval_mem(cx, cx.fcx.ccx.consts.get(did));
                }
                case (_) {
                    cx.fcx.ccx.sess.unimpl("def variant in trans");
                }
            }
        }
        case (none[ast.def]) {
            cx.fcx.ccx.sess.err("unresolved expr_path in trans");
        }
    }
    fail;
}

fn trans_field(@block_ctxt cx, &ast.span sp, @ast.expr base,
               &ast.ident field, &ast.ann ann) -> lval_result {
    auto lv = trans_lval(cx, base);
    auto r = lv.res;
    r = autoderef(r.bcx, r.val, ty.expr_ty(base));
    check (lv.is_mem);
    auto t = ty.expr_ty(base);
    alt (t.struct) {
        case (ty.ty_tup(?fields)) {
            let uint ix = ty.field_num(cx.fcx.ccx.sess, sp, field);
            auto v = GEP_tup_like(r.bcx, t, r.val, vec(0, ix as int));
            ret lval_mem(r.bcx, v);
        }
        case (ty.ty_rec(?fields)) {
            let uint ix = ty.field_idx(cx.fcx.ccx.sess, sp, field, fields);
            auto v = GEP_tup_like(r.bcx, t, r.val, vec(0, ix as int));
            ret lval_mem(r.bcx, v);
        }
        case (ty.ty_obj(?methods)) {
            let uint ix = ty.method_idx(cx.fcx.ccx.sess, sp, field, methods);
            auto vtbl = r.bcx.build.GEP(r.val,
                                        vec(C_int(0),
                                            C_int(abi.obj_field_vtbl)));
            vtbl = r.bcx.build.Load(vtbl);
            auto v =  r.bcx.build.GEP(vtbl, vec(C_int(0),
                                                C_int(ix as int)));

            auto lvo = lval_mem(r.bcx, v);
            ret rec(llobj = some[ValueRef](r.val) with lvo);
        }
        case (_) { cx.fcx.ccx.sess.unimpl("field variant in trans_field"); }
    }
    fail;
}

fn trans_index(@block_ctxt cx, &ast.span sp, @ast.expr base,
               @ast.expr idx, &ast.ann ann) -> lval_result {

    auto lv = trans_expr(cx, base);
    lv = autoderef(lv.bcx, lv.val, ty.expr_ty(base));
    auto ix = trans_expr(lv.bcx, idx);
    auto v = lv.val;

    auto llunit_ty = node_type(cx.fcx.ccx, ann);
    auto unit_sz = size_of(cx, node_ann_type(cx.fcx.ccx, ann));
    auto scaled_ix = ix.bcx.build.Mul(ix.val, unit_sz);

    auto lim = ix.bcx.build.GEP(v, vec(C_int(0), C_int(abi.vec_elt_fill)));
    lim = ix.bcx.build.Load(lim);

    auto bounds_check = ix.bcx.build.ICmp(lib.llvm.LLVMIntULT,
                                          scaled_ix, lim);

    auto fail_cx = new_sub_block_ctxt(ix.bcx, "fail");
    auto next_cx = new_sub_block_ctxt(ix.bcx, "next");
    ix.bcx.build.CondBr(bounds_check, next_cx.llbb, fail_cx.llbb);

    // fail: bad bounds check.
    auto fail_res = trans_fail(fail_cx, sp, "bounds check");
    fail_res.bcx.build.Br(next_cx.llbb);

    auto body = next_cx.build.GEP(v, vec(C_int(0), C_int(abi.vec_elt_data)));
    auto elt = next_cx.build.GEP(body, vec(C_int(0), ix.val));
    ret lval_mem(next_cx, elt);
}

// The additional bool returned indicates whether it's mem (that is
// represented as an alloca or heap, hence needs a 'load' to be used as an
// immediate).

fn trans_lval(@block_ctxt cx, @ast.expr e) -> lval_result {
    alt (e.node) {
        case (ast.expr_path(?p, ?dopt, ?ann)) {
            ret trans_path(cx, p, dopt, ann);
        }
        case (ast.expr_field(?base, ?ident, ?ann)) {
            ret trans_field(cx, e.span, base, ident, ann);
        }
        case (ast.expr_index(?base, ?idx, ?ann)) {
            ret trans_index(cx, e.span, base, idx, ann);
        }
        case (_) { cx.fcx.ccx.sess.unimpl("expr variant in trans_lval"); }
    }
    fail;
}

fn trans_cast(@block_ctxt cx, @ast.expr e, &ast.ann ann) -> result {
    auto e_res = trans_expr(cx, e);
    auto llsrctype = val_ty(e_res.val);
    auto t = node_ann_type(cx.fcx.ccx, ann);
    auto lldsttype = type_of(cx.fcx.ccx, t);
    if (!ty.type_is_fp(t)) {
        if (llvm.LLVMGetIntTypeWidth(lldsttype) >
            llvm.LLVMGetIntTypeWidth(llsrctype)) {
            if (ty.type_is_signed(t)) {
                // Widening signed cast.
                e_res.val =
                    e_res.bcx.build.SExtOrBitCast(e_res.val,
                                                  lldsttype);
            } else {
                // Widening unsigned cast.
                e_res.val =
                    e_res.bcx.build.ZExtOrBitCast(e_res.val,
                                                  lldsttype);
            }
        } else {
            // Narrowing cast.
            e_res.val =
                e_res.bcx.build.TruncOrBitCast(e_res.val,
                                               lldsttype);
        }
    } else {
        cx.fcx.ccx.sess.unimpl("fp cast");
    }
    ret e_res;
}


// NB: this must match type_of_fn_full and create_llargs_for_fn_args.
fn trans_args(@block_ctxt cx,
              ValueRef llclosure,
              option.t[ValueRef] llobj,
              option.t[generic_info] gen,
              &vec[@ast.expr] es,
              @ty.t fn_ty)
    -> tup(@block_ctxt, vec[ValueRef], option.t[ValueRef]) {
    let vec[ValueRef] vs = vec(cx.fcx.lltaskptr);
    let @block_ctxt bcx = cx;

    let vec[ty.arg] args = ty.ty_fn_args(fn_ty);

    let option.t[ValueRef] llretslot_opt = none[ValueRef];

    alt (gen) {
        case (some[generic_info](?g)) {
            for (ValueRef t in g.tydescs) {
                vs += t;
            }
            args = ty.ty_fn_args(g.item_type);
            if (ty.type_has_dynamic_size(ty.ty_fn_ret(g.item_type))) {
                auto retty = ty.ty_fn_ret(fn_ty);
                auto llretty = type_of(cx.fcx.ccx, retty);
                auto llretslot = cx.build.Alloca(llretty);
                vs += cx.build.PointerCast(llretslot, T_ptr(T_i8()));
                llretslot_opt = some[ValueRef](llretslot);
            }
        }
        case (_) { }
    }

    alt (llobj) {
        case (some[ValueRef](?ob)) {
            // Every object is always found in memory,
            // and not-yet-loaded (as part of an lval x.y
            // doted method-call).
            vs += cx.build.Load(ob);
        }
        case (_) {
            vs += llclosure;
        }
    }

    auto i = 0u;
    for (@ast.expr e in es) {
        auto mode = args.(i).mode;

        auto val;
        if (ty.type_is_structural(ty.expr_ty(e))) {
            auto re = trans_expr(bcx, e);
            val = re.val;
            bcx = re.bcx;
            if (mode == ast.val) {
                // Until here we've been treating structures by pointer;
                // we are now passing it as an arg, so need to load it.
                val = bcx.build.Load(val);
            }
        } else if (mode == ast.alias) {
            let lval_result lv;
            if (ty.is_lval(e)) {
                lv = trans_lval(bcx, e);
            } else {
                auto r = trans_expr(bcx, e);
                lv = lval_val(r.bcx, r.val);
            }
            bcx = lv.res.bcx;

            if (lv.is_mem) {
                val = lv.res.val;
            } else {
                // Non-mem but we're trying to alias; synthesize an
                // alloca, spill to it and pass its address.
                auto llty = val_ty(lv.res.val);
                auto llptr = lv.res.bcx.build.Alloca(llty);
                lv.res.bcx.build.Store(lv.res.val, llptr);
                val = llptr;
            }

        } else {
            auto re = trans_expr(bcx, e);
            val = re.val;
            bcx = re.bcx;
        }

        if (ty.type_has_dynamic_size(args.(i).ty)) {
            val = bcx.build.PointerCast(val, T_typaram_ptr());
        }

        vs += val;
        i += 1u;
    }

    ret tup(bcx, vs, llretslot_opt);
}

fn trans_bind_thunk(@crate_ctxt cx,
                    @ty.t incoming_fty,
                    @ty.t outgoing_fty,
                    vec[option.t[@ast.expr]] args,
                    TypeRef llclosure_ty,
                    vec[@ty.t] bound_tys) -> ValueRef {
    // Construct a thunk-call with signature incoming_fty, and that copies
    // args forward into a call to outgoing_fty.

    let str s = cx.names.next("_rust_thunk") + "." + cx.path;
    let TypeRef llthunk_ty = get_pair_fn_ty(type_of(cx, incoming_fty));
    let ValueRef llthunk = decl_fastcall_fn(cx.llmod, s, llthunk_ty);

    let @ty.t rty = ret_ty_of_fn_ty(incoming_fty);

    // FIXME: handle ty params properly.
    let vec[ast.ty_param] ty_params = vec();

    auto fcx = new_fn_ctxt(cx, s, llthunk);
    auto bcx = new_top_block_ctxt(fcx);

    auto llclosure = bcx.build.PointerCast(fcx.llclosure, llclosure_ty);

    auto llbody = bcx.build.GEP(llclosure,
                                vec(C_int(0),
                                    C_int(abi.box_rc_field_body)));

    auto lltarget = bcx.build.GEP(llbody,
                                  vec(C_int(0),
                                      C_int(abi.closure_elt_target)));

    auto llbound = bcx.build.GEP(llbody,
                                 vec(C_int(0),
                                     C_int(abi.closure_elt_bindings)));

    auto lltargetclosure = bcx.build.GEP(lltarget,
                                         vec(C_int(0),
                                             C_int(abi.fn_field_box)));
    lltargetclosure = bcx.build.Load(lltargetclosure);
    let vec[ValueRef] llargs = vec(fcx.lltaskptr,
                                   lltargetclosure);
    let uint a = 0u;
    let int b = 0;
    for (option.t[@ast.expr] arg in args) {
        alt (arg) {

            // Arg provided at binding time; thunk copies it from closure.
            case (some[@ast.expr](_)) {
                let ValueRef bound_arg = bcx.build.GEP(llbound,
                                                       vec(C_int(0),
                                                           C_int(b)));
                // FIXME: possibly support passing aliases someday.
                llargs += bcx.build.Load(bound_arg);
                b += 1;
            }

            // Arg will be provided when the thunk is invoked.
            case (none[@ast.expr]) {
                let ValueRef passed_arg = llvm.LLVMGetParam(llthunk, a);
                llargs += passed_arg;
                a += 1u;
            }
        }
    }

    // FIXME: turn this call + ret into a tail call.
    auto lltargetfn = bcx.build.GEP(lltarget,
                                    vec(C_int(0),
                                        C_int(abi.fn_field_code)));
    lltargetfn = bcx.build.Load(lltargetfn);
    auto r = bcx.build.FastCall(lltargetfn, llargs);

    alt (fcx.llretptr) {
        case (some[ValueRef](?llptr)) {
            bcx.build.Store(bcx.build.Load(r), llptr);
            bcx.build.RetVoid();
        }
        case (none[ValueRef]) {
            if (ty.type_is_nil(rty)) {
                bcx.build.RetVoid();
            } else {
                bcx.build.Ret(r);
            }
        }
    }

    ret llthunk;
}

fn trans_bind(@block_ctxt cx, @ast.expr f,
              vec[option.t[@ast.expr]] args,
              &ast.ann ann) -> result {
    auto f_res = trans_lval(cx, f);
    if (f_res.is_mem) {
        cx.fcx.ccx.sess.unimpl("re-binding existing function");
    } else {
        let vec[@ast.expr] bound = vec();

        for (option.t[@ast.expr] argopt in args) {
            alt (argopt) {
                case (none[@ast.expr]) {
                }
                case (some[@ast.expr](?e)) {
                    append[@ast.expr](bound, e);
                }
            }
        }
        if (_vec.len[@ast.expr](bound) == 0u) {
            // Trivial 'binding': just return the static pair-ptr.
            ret f_res.res;
        } else {
            auto bcx = f_res.res.bcx;
            auto pair_t = node_type(cx.fcx.ccx, ann);
            auto pair_v = bcx.build.Alloca(pair_t);

            // Translate the bound expressions.
            let vec[@ty.t] bound_tys = vec();
            let vec[ValueRef] bound_vals = vec();
            for (@ast.expr e in bound) {
                auto arg = trans_expr(bcx, e);
                bcx = arg.bcx;
                append[ValueRef](bound_vals, arg.val);
                append[@ty.t](bound_tys, ty.expr_ty(e));
            }

            // Synthesize a closure type.
            let @ty.t bindings_ty = ty.plain_ty(ty.ty_tup(bound_tys));
            let TypeRef lltarget_ty = type_of(bcx.fcx.ccx, ty.expr_ty(f));
            let TypeRef llbindings_ty = type_of(bcx.fcx.ccx, bindings_ty);
            let TypeRef llclosure_ty = T_closure_ptr(lltarget_ty,
                                                     llbindings_ty);

            // Malloc a box for the body.
            auto r = trans_malloc_inner(bcx, llclosure_ty);
            auto box = r.val;
            bcx = r.bcx;
            auto rc = bcx.build.GEP(box,
                                    vec(C_int(0),
                                        C_int(abi.box_rc_field_refcnt)));
            auto closure =
                bcx.build.GEP(box,
                              vec(C_int(0),
                                  C_int(abi.box_rc_field_body)));
            bcx.build.Store(C_int(1), rc);

            // Store bindings tydesc.
            auto bound_tydesc =
                bcx.build.GEP(closure,
                              vec(C_int(0),
                                  C_int(abi.closure_elt_tydesc)));
            auto bindings_tydesc = get_tydesc(bcx, bindings_ty);
            bcx.build.Store(bindings_tydesc, bound_tydesc);

            // Store thunk-target.
            auto bound_target =
                bcx.build.GEP(closure,
                              vec(C_int(0),
                                  C_int(abi.closure_elt_target)));
            bcx.build.Store(bcx.build.Load(f_res.res.val), bound_target);

            // Copy expr values into boxed bindings.
            let int i = 0;
            auto bindings =
                bcx.build.GEP(closure,
                              vec(C_int(0),
                                  C_int(abi.closure_elt_bindings)));
            for (ValueRef v in bound_vals) {
                auto bound = bcx.build.GEP(bindings,
                                           vec(C_int(0),C_int(i)));
                bcx = copy_ty(r.bcx, INIT, bound, v, bound_tys.(i)).bcx;
                i += 1;
            }

            // Make thunk and store thunk-ptr in outer pair's code slot.
            auto pair_code = bcx.build.GEP(pair_v,
                                           vec(C_int(0),
                                               C_int(abi.fn_field_code)));

            let @ty.t pair_ty = node_ann_type(cx.fcx.ccx, ann);
            let ValueRef llthunk =
                trans_bind_thunk(cx.fcx.ccx, pair_ty, ty.expr_ty(f),
                                 args, llclosure_ty, bound_tys);

            bcx.build.Store(llthunk, pair_code);

            // Store box ptr in outer pair's box slot.
            auto pair_box = bcx.build.GEP(pair_v,
                                          vec(C_int(0),
                                              C_int(abi.fn_field_box)));
            bcx.build.Store(bcx.build.PointerCast(box,
                                                  T_opaque_closure_ptr()),
                            pair_box);

            find_scope_cx(cx).cleanups +=
                clean(bind drop_slot(_, pair_v, pair_ty));

            ret res(bcx, pair_v);
        }
    }
}

fn trans_call(@block_ctxt cx, @ast.expr f,
              vec[@ast.expr] args, &ast.ann ann) -> result {
    auto f_res = trans_lval(cx, f);
    auto faddr = f_res.res.val;
    auto llclosure = C_null(T_opaque_closure_ptr());

    alt (f_res.llobj) {
        case (some[ValueRef](_)) {
            // It's a vtbl entry.
            faddr = f_res.res.bcx.build.Load(faddr);
        }
        case (none[ValueRef]) {
            // It's a closure.
            auto bcx = f_res.res.bcx;
            auto pair = faddr;
            faddr = bcx.build.GEP(pair, vec(C_int(0),
                                            C_int(abi.fn_field_code)));
            faddr = bcx.build.Load(faddr);

            llclosure = bcx.build.GEP(pair, vec(C_int(0),
                                                C_int(abi.fn_field_box)));
            llclosure = bcx.build.Load(llclosure);
        }
    }
    auto fn_ty = ty.expr_ty(f);
    auto ret_ty = ty.ann_to_type(ann);
    auto args_res = trans_args(f_res.res.bcx,
                               llclosure, f_res.llobj,
                               f_res.generic,
                               args, fn_ty);

    auto bcx = args_res._0;
    auto real_retval = bcx.build.FastCall(faddr, args_res._1);
    auto retval = real_retval;

    if (ty.type_is_nil(ret_ty)) {
        retval = C_nil();
    }

    // Check for a generic retslot.
    alt (args_res._2) {

        case (some[ValueRef](?llretslot)) {
            retval = load_scalar_or_boxed(bcx, llretslot, ret_ty);
        }

        case (none[ValueRef]) {
            if (! (ty.type_is_scalar(ret_ty) ||
                   ty.type_is_boxed(ret_ty))) {
                // Structured returns come back as first-class values. This is
                // nice for LLVM but wrong for us; we treat structured values
                // by pointer in most of our code here. So spill it to an
                // alloca.
                auto local = bcx.build.Alloca(type_of(cx.fcx.ccx, ret_ty));
                bcx.build.Store(retval, local);
                retval = local;
            }
        }
    }

    // Retval doesn't correspond to anything really tangible in the frame, but
    // it's a ref all the same, so we put a note here to drop it when we're
    // done in this scope.
    find_scope_cx(cx).cleanups += clean(bind drop_ty(_, retval, ret_ty));

    ret res(bcx, retval);
}

fn trans_tup(@block_ctxt cx, vec[ast.elt] elts,
             &ast.ann ann) -> result {
    auto t = node_ann_type(cx.fcx.ccx, ann);
    auto llty = type_of(cx.fcx.ccx, t);
    auto tup_val = cx.build.Alloca(llty);
    find_scope_cx(cx).cleanups += clean(bind drop_ty(_, tup_val, t));
    let int i = 0;
    auto r = res(cx, C_nil());
    for (ast.elt e in elts) {
        auto t = ty.expr_ty(e.expr);
        auto src_res = trans_expr(r.bcx, e.expr);
        auto dst_elt = r.bcx.build.GEP(tup_val, vec(C_int(0), C_int(i)));
        r = copy_ty(src_res.bcx, INIT, dst_elt, src_res.val, t);
        i += 1;
    }
    ret res(r.bcx, tup_val);
}

fn trans_vec(@block_ctxt cx, vec[@ast.expr] args,
             &ast.ann ann) -> result {
    auto t = node_ann_type(cx.fcx.ccx, ann);
    auto unit_ty = t;
    alt (t.struct) {
        case (ty.ty_vec(?t)) {
            unit_ty = t;
        }
        case (_) {
            cx.fcx.ccx.sess.bug("non-vec type in trans_vec");
        }
    }

    auto llunit_ty = type_of(cx.fcx.ccx, unit_ty);
    auto unit_sz = size_of(cx, unit_ty);
    auto data_sz = llvm.LLVMConstMul(C_int(_vec.len[@ast.expr](args) as int),
                                     unit_sz);

    // FIXME: pass tydesc properly.
    auto sub = trans_upcall(cx, "upcall_new_vec", vec(data_sz, C_int(0)));

    auto llty = type_of(cx.fcx.ccx, t);
    auto vec_val = sub.bcx.build.IntToPtr(sub.val, llty);
    find_scope_cx(cx).cleanups += clean(bind drop_ty(_, vec_val, t));

    auto body = sub.bcx.build.GEP(vec_val, vec(C_int(0),
                                               C_int(abi.vec_elt_data)));
    let int i = 0;
    for (@ast.expr e in args) {
        auto src_res = trans_expr(sub.bcx, e);
        auto dst_elt = sub.bcx.build.GEP(body, vec(C_int(0), C_int(i)));
        sub = copy_ty(src_res.bcx, INIT, dst_elt, src_res.val, unit_ty);
        i += 1;
    }
    auto fill = sub.bcx.build.GEP(vec_val,
                                  vec(C_int(0), C_int(abi.vec_elt_fill)));
    sub.bcx.build.Store(data_sz, fill);

    ret res(sub.bcx, vec_val);
}

fn trans_rec(@block_ctxt cx, vec[ast.field] fields,
             &ast.ann ann) -> result {
    auto t = node_ann_type(cx.fcx.ccx, ann);
    auto llty = type_of(cx.fcx.ccx, t);
    auto rec_val = cx.build.Alloca(llty);
    find_scope_cx(cx).cleanups += clean(bind drop_ty(_, rec_val, t));
    let int i = 0;
    auto r = res(cx, C_nil());
    for (ast.field f in fields) {
        auto t = ty.expr_ty(f.expr);
        auto src_res = trans_expr(r.bcx, f.expr);
        auto dst_elt = r.bcx.build.GEP(rec_val, vec(C_int(0), C_int(i)));
        // FIXME: calculate copy init-ness in typestate.
        r = copy_ty(src_res.bcx, INIT, dst_elt, src_res.val, t);
        i += 1;
    }
    ret res(r.bcx, rec_val);
}



fn trans_expr(@block_ctxt cx, @ast.expr e) -> result {
    alt (e.node) {
        case (ast.expr_lit(?lit, ?ann)) {
            ret res(cx, trans_lit(cx.fcx.ccx, *lit, ann));
        }

        case (ast.expr_unary(?op, ?x, ?ann)) {
            ret trans_unary(cx, op, x, ann);
        }

        case (ast.expr_binary(?op, ?x, ?y, _)) {
            ret trans_binary(cx, op, x, y);
        }

        case (ast.expr_if(?cond, ?thn, ?els, _)) {
            ret trans_if(cx, cond, thn, els);
        }

        case (ast.expr_for(?decl, ?seq, ?body, _)) {
            ret trans_for(cx, decl, seq, body);
        }

        case (ast.expr_while(?cond, ?body, _)) {
            ret trans_while(cx, cond, body);
        }

        case (ast.expr_do_while(?body, ?cond, _)) {
            ret trans_do_while(cx, body, cond);
        }

        case (ast.expr_alt(?expr, ?arms, _)) {
            ret trans_alt(cx, expr, arms);
        }

        case (ast.expr_block(?blk, _)) {
            auto sub_cx = new_scope_block_ctxt(cx, "block-expr body");
            auto next_cx = new_sub_block_ctxt(cx, "next");
            auto sub = trans_block(sub_cx, blk);

            cx.build.Br(sub_cx.llbb);
            sub.bcx.build.Br(next_cx.llbb);

            ret res(next_cx, sub.val);
        }

        case (ast.expr_assign(?dst, ?src, ?ann)) {
            auto lhs_res = trans_lval(cx, dst);
            check (lhs_res.is_mem);
            auto rhs_res = trans_expr(lhs_res.res.bcx, src);
            auto t = node_ann_type(cx.fcx.ccx, ann);
            // FIXME: calculate copy init-ness in typestate.
            ret copy_ty(rhs_res.bcx, DROP_EXISTING,
                        lhs_res.res.val, rhs_res.val, t);
        }

        case (ast.expr_assign_op(?op, ?dst, ?src, ?ann)) {
            auto t = node_ann_type(cx.fcx.ccx, ann);
            auto lhs_res = trans_lval(cx, dst);
            check (lhs_res.is_mem);
            auto lhs_val = load_scalar_or_boxed(lhs_res.res.bcx,
                                                lhs_res.res.val, t);
            auto rhs_res = trans_expr(lhs_res.res.bcx, src);
            auto v = trans_eager_binop(rhs_res.bcx, op, lhs_val, rhs_res.val);
            // FIXME: calculate copy init-ness in typestate.
            ret copy_ty(rhs_res.bcx, DROP_EXISTING,
                        lhs_res.res.val, v, t);
        }

        case (ast.expr_bind(?f, ?args, ?ann)) {
            ret trans_bind(cx, f, args, ann);
        }

        case (ast.expr_call(?f, ?args, ?ann)) {
            ret trans_call(cx, f, args, ann);
        }

        case (ast.expr_cast(?e, _, ?ann)) {
            ret trans_cast(cx, e, ann);
        }

        case (ast.expr_vec(?args, ?ann)) {
            ret trans_vec(cx, args, ann);
        }

        case (ast.expr_tup(?args, ?ann)) {
            ret trans_tup(cx, args, ann);
        }

        case (ast.expr_rec(?args, ?ann)) {
            ret trans_rec(cx, args, ann);
        }

        // lval cases fall through to trans_lval and then
        // possibly load the result (if it's non-structural).

        case (_) {
            auto t = ty.expr_ty(e);
            auto sub = trans_lval(cx, e);
            ret res(sub.res.bcx,
                    load_scalar_or_boxed(sub.res.bcx, sub.res.val, t));
        }
    }
    cx.fcx.ccx.sess.unimpl("expr variant in trans_expr");
    fail;
}

// We pass structural values around the compiler "by pointer" and
// non-structural values (scalars and boxes) "by value". This function selects
// whether to load a pointer or pass it.

fn load_scalar_or_boxed(@block_ctxt cx,
                        ValueRef v,
                        @ty.t t) -> ValueRef {
    if (ty.type_is_scalar(t) || ty.type_is_boxed(t)) {
        ret cx.build.Load(v);
    } else {
        ret v;
    }
}

fn trans_log(@block_ctxt cx, @ast.expr e) -> result {

    auto sub = trans_expr(cx, e);
    auto e_ty = ty.expr_ty(e);
    alt (e_ty.struct) {
        case (ty.ty_str) {
            auto v = sub.bcx.build.PtrToInt(sub.val, T_int());
            ret trans_upcall(sub.bcx,
                             "upcall_log_str",
                             vec(v));
        }
        case (_) {
            ret trans_upcall(sub.bcx,
                             "upcall_log_int",
                             vec(sub.val));
        }
    }
    fail;
}

fn trans_check_expr(@block_ctxt cx, @ast.expr e) -> result {
    auto cond_res = trans_expr(cx, e);

    // FIXME: need pretty-printer.
    auto expr_str = "<expr>";
    auto fail_cx = new_sub_block_ctxt(cx, "fail");
    auto fail_res = trans_fail(fail_cx, e.span, expr_str);

    auto next_cx = new_sub_block_ctxt(cx, "next");
    fail_res.bcx.build.Br(next_cx.llbb);
    cond_res.bcx.build.CondBr(cond_res.val,
                              next_cx.llbb,
                              fail_cx.llbb);
    ret res(next_cx, C_nil());
}

fn trans_fail(@block_ctxt cx, common.span sp, str fail_str) -> result {
    auto V_fail_str = p2i(C_cstr(cx.fcx.ccx, fail_str));
    auto V_filename = p2i(C_cstr(cx.fcx.ccx, sp.filename));
    auto V_line = sp.lo.line as int;
    auto args = vec(V_fail_str, V_filename, C_int(V_line));

    ret trans_upcall(cx, "upcall_fail", args);
}

fn trans_ret(@block_ctxt cx, &option.t[@ast.expr] e) -> result {
    auto bcx = cx;
    auto val = C_nil();

    alt (e) {
        case (some[@ast.expr](?x)) {
            auto t = ty.expr_ty(x);
            auto r = trans_expr(cx, x);
            bcx = r.bcx;
            val = r.val;

            // A return is an implicit copy into a newborn anonymous
            // 'return value' in the caller frame.
            bcx = incr_all_refcnts(bcx, val, t).bcx;

            if (ty.type_is_structural(t)) {
                // We usually treat structurals by-pointer; in particular,
                // trans_expr will have given us a structure pointer. But in
                // this case we're about to return. LLVM wants a first-class
                // value here (which makes sense; the frame is going away!)
                val = r.bcx.build.Load(val);
            }
        }
        case (_) { /* fall through */  }
    }

    // Run all cleanups and back out.
    let bool more_cleanups = true;
    auto cleanup_cx = cx;
    while (more_cleanups) {
        bcx = trans_block_cleanups(bcx, cleanup_cx);
        alt (cleanup_cx.parent) {
            case (parent_some(?b)) {
                cleanup_cx = b;
            }
            case (parent_none) {
                more_cleanups = false;
            }
        }
    }

    alt (e) {
        case (some[@ast.expr](?ex)) {
            auto t = ty.expr_ty(ex);

            if (ty.type_is_nil(t)) {
                bcx.build.RetVoid();
                val = C_nil();
                ret res(bcx, val);  // FIXME: early return needed due to
                                    // typestate bug
            }

            alt (cx.fcx.llretptr) {
                case (some[ValueRef](?llptr)) {
                    // Generic return via tydesc + retptr.
                    bcx = copy_ty(bcx, INIT, llptr, val, t).bcx;
                    bcx.build.RetVoid();
                }
                case (none[ValueRef]) {
                    val = bcx.build.Ret(val);
                }
            }
            ret res(bcx, val);
        }
        case (_) { /* fall through */  }
    }

    // FIXME: until LLVM has a unit type, we are moving around
    // C_nil values rather than their void type.
    bcx.build.RetVoid();
    ret res(bcx, C_nil());
}

fn init_local(@block_ctxt cx, @ast.local local) -> result {

    // Make a note to drop this slot on the way out.
    check (cx.fcx.lllocals.contains_key(local.id));
    auto llptr = cx.fcx.lllocals.get(local.id);
    auto ty = node_ann_type(cx.fcx.ccx, local.ann);
    auto bcx = cx;

    find_scope_cx(cx).cleanups +=
        clean(bind drop_slot(_, llptr, ty));

    alt (local.init) {
        case (some[@ast.expr](?e)) {
            auto sub = trans_expr(bcx, e);
            bcx = copy_ty(sub.bcx, INIT, llptr, sub.val, ty).bcx;
        }
        case (_) {
            if (middle.ty.type_has_dynamic_size(ty)) {
                auto llsz = size_of(cx, ty);
                bcx = call_bzero(cx, llptr, llsz).bcx;

            } else {
                auto llty = type_of(bcx.fcx.ccx, ty);
                auto null = lib.llvm.llvm.LLVMConstNull(llty);
                bcx.build.Store(null, llptr);
            }
        }
    }
    ret res(bcx, llptr);
}

fn trans_stmt(@block_ctxt cx, &ast.stmt s) -> result {
    auto bcx = cx;
    alt (s.node) {
        case (ast.stmt_log(?a)) {
            bcx = trans_log(cx, a).bcx;
        }

        case (ast.stmt_check_expr(?a)) {
            bcx = trans_check_expr(cx, a).bcx;
        }

        case (ast.stmt_fail) {
            bcx = trans_fail(cx, s.span, "explicit failure").bcx;
        }

        case (ast.stmt_ret(?e)) {
            bcx = trans_ret(cx, e).bcx;
        }

        case (ast.stmt_expr(?e)) {
            bcx = trans_expr(cx, e).bcx;
        }

        case (ast.stmt_decl(?d)) {
            alt (d.node) {
                case (ast.decl_local(?local)) {
                    bcx = init_local(bcx, local).bcx;
                }
                case (ast.decl_item(?i)) {
                    trans_item(cx.fcx.ccx, *i);
                }
            }
        }
        case (_) {
            cx.fcx.ccx.sess.unimpl("stmt variant");
        }
    }
    ret res(bcx, C_nil());
}

fn new_builder(BasicBlockRef llbb) -> builder {
    let BuilderRef llbuild = llvm.LLVMCreateBuilder();
    llvm.LLVMPositionBuilderAtEnd(llbuild, llbb);
    ret builder(llbuild);
}

// You probably don't want to use this one. See the
// next three functions instead.
fn new_block_ctxt(@fn_ctxt cx, block_parent parent,
                  block_kind kind,
                  str name) -> @block_ctxt {
    let vec[cleanup] cleanups = vec();
    let BasicBlockRef llbb =
        llvm.LLVMAppendBasicBlock(cx.llfn,
                                  _str.buf(cx.ccx.names.next(name)));

    ret @rec(llbb=llbb,
             build=new_builder(llbb),
             parent=parent,
             kind=kind,
             mutable cleanups=cleanups,
             fcx=cx);
}

// Use this when you're at the top block of a function or the like.
fn new_top_block_ctxt(@fn_ctxt fcx) -> @block_ctxt {
    ret new_block_ctxt(fcx, parent_none, SCOPE_BLOCK, "function top level");
}

// Use this when you're at a curly-brace or similar lexical scope.
fn new_scope_block_ctxt(@block_ctxt bcx, str n) -> @block_ctxt {
    ret new_block_ctxt(bcx.fcx, parent_some(bcx), SCOPE_BLOCK, n);
}

// Use this when you're making a general CFG BB within a scope.
fn new_sub_block_ctxt(@block_ctxt bcx, str n) -> @block_ctxt {
    ret new_block_ctxt(bcx.fcx, parent_some(bcx), NON_SCOPE_BLOCK, n);
}


fn trans_block_cleanups(@block_ctxt cx,
                        @block_ctxt cleanup_cx) -> @block_ctxt {
    auto bcx = cx;

    if (cleanup_cx.kind != SCOPE_BLOCK) {
        check (_vec.len[cleanup](cleanup_cx.cleanups) == 0u);
    }

    for (cleanup c in cleanup_cx.cleanups) {
        alt (c) {
            case (clean(?cfn)) {
                bcx = cfn(bcx).bcx;
            }
        }
    }
    ret bcx;
}

iter block_locals(&ast.block b) -> @ast.local {
    // FIXME: putting from inside an iter block doesn't work, so we can't
    // use the index here.
    for (@ast.stmt s in b.node.stmts) {
        alt (s.node) {
            case (ast.stmt_decl(?d)) {
                alt (d.node) {
                    case (ast.decl_local(?local)) {
                        put local;
                    }
                    case (_) { /* fall through */ }
                }
            }
            case (_) { /* fall through */ }
        }
    }
}

fn alloc_local(@block_ctxt cx, @ast.local local) -> result {
    auto t = node_ann_type(cx.fcx.ccx, local.ann);
    auto val = C_int(0);
    if (ty.type_has_dynamic_size(t)) {
        auto n = size_of(cx, t);
        val = cx.build.ArrayAlloca(T_i8(), n);
    } else {
        val = cx.build.Alloca(type_of(cx.fcx.ccx, t));
    }
    cx.fcx.lllocals.insert(local.id, val);
    ret res(cx, val);
}

fn trans_block(@block_ctxt cx, &ast.block b) -> result {
    auto bcx = cx;

    for each (@ast.local local in block_locals(b)) {
        bcx = alloc_local(bcx, local).bcx;
    }
    auto r = res(bcx, C_nil());

    for (@ast.stmt s in b.node.stmts) {
        r = trans_stmt(bcx, *s);
        bcx = r.bcx;
        // If we hit a terminator, control won't go any further so
        // we're in dead-code land. Stop here.
        if (is_terminated(bcx)) {
            ret r;
        }
    }

    alt (b.node.expr) {
        case (some[@ast.expr](?e)) {
            r = trans_expr(bcx, e);
            bcx = r.bcx;
            if (is_terminated(bcx)) {
                ret r;
            }
        }
        case (none[@ast.expr]) {
            r = res(bcx, C_nil());
        }
    }

    bcx = trans_block_cleanups(bcx, find_scope_cx(bcx));
    ret res(bcx, r.val);
}

fn new_fn_ctxt(@crate_ctxt cx,
               str name,
               ValueRef llfndecl) -> @fn_ctxt {

    let ValueRef lltaskptr = llvm.LLVMGetParam(llfndecl, 0u);
    let ValueRef llclosure = llvm.LLVMGetParam(llfndecl, 1u);

    let hashmap[ast.def_id, ValueRef] llargs = new_def_hash[ValueRef]();
    let hashmap[ast.def_id, ValueRef] llobjfields = new_def_hash[ValueRef]();
    let hashmap[ast.def_id, ValueRef] lllocals = new_def_hash[ValueRef]();
    let hashmap[ast.def_id, ValueRef] lltydescs = new_def_hash[ValueRef]();

    ret @rec(llfn=llfndecl,
             lltaskptr=lltaskptr,
             llclosure=llclosure,
             mutable llself=none[ValueRef],
             mutable llretptr=none[ValueRef],
             llargs=llargs,
             llobjfields=llobjfields,
             lllocals=lllocals,
             lltydescs=lltydescs,
             ccx=cx);
}

// NB: this must match trans_args and type_of_fn_full.
fn create_llargs_for_fn_args(&@fn_ctxt cx,
                             option.t[TypeRef] ty_self,
                             @ty.t ret_ty,
                             &vec[ast.arg] args,
                             &vec[ast.ty_param] ty_params) {
    let uint arg_n = 1u;

    for (ast.ty_param tp in ty_params) {
        auto llarg = llvm.LLVMGetParam(cx.llfn, arg_n);
        check (llarg as int != 0);
        cx.lltydescs.insert(tp.id, llarg);
        arg_n += 1u;
    }

    if (ty.type_has_dynamic_size(ret_ty)) {
        cx.llretptr = some[ValueRef](llvm.LLVMGetParam(cx.llfn, arg_n));
        arg_n += 1u;
    }

    alt (ty_self) {
        case (some[TypeRef](_)) {
            auto llself = llvm.LLVMGetParam(cx.llfn, arg_n);
            check (llself as int != 0);
            cx.llself = some[ValueRef](llself);
            arg_n += 1u;
        }
        case (_) {
            // llclosure, we don't know what it is.
            arg_n += 1u;
        }
    }

    for (ast.arg arg in args) {
        auto llarg = llvm.LLVMGetParam(cx.llfn, arg_n);
        check (llarg as int != 0);
        cx.llargs.insert(arg.id, llarg);
        arg_n += 1u;
    }
}

// Recommended LLVM style, strange though this is, is to copy from args to
// allocas immediately upon entry; this permits us to GEP into structures we
// were passed and whatnot. Apparently mem2reg will mop up.

fn copy_args_to_allocas(@block_ctxt cx,
                        option.t[TypeRef] ty_self,
                        vec[ast.arg] args,
                        vec[ty.arg] arg_tys) {

    let uint arg_n = 0u;

    alt (cx.fcx.llself) {
        case (some[ValueRef](?self_v)) {
            alt (ty_self) {
                case (some[TypeRef](?self_t)) {
                    auto alloca = cx.build.Alloca(self_t);
                    cx.build.Store(self_v, alloca);
                    cx.fcx.llself = some[ValueRef](alloca);
                }
            }
        }
        case (_) {
        }
    }

    for (ast.arg aarg in args) {
        if (aarg.mode != ast.alias) {
            auto arg_t = type_of_arg(cx.fcx.ccx, arg_tys.(arg_n));
            auto alloca = cx.build.Alloca(arg_t);
            auto argval = cx.fcx.llargs.get(aarg.id);
            cx.build.Store(argval, alloca);
            // Overwrite the llargs entry for this arg with its alloca.
            cx.fcx.llargs.insert(aarg.id, alloca);
        }

        arg_n += 1u;
    }
}

fn is_terminated(@block_ctxt cx) -> bool {
    auto inst = llvm.LLVMGetLastInstruction(cx.llbb);
    ret llvm.LLVMIsATerminatorInst(inst) as int != 0;
}

fn arg_tys_of_fn(ast.ann ann) -> vec[ty.arg] {
    alt (ty.ann_to_type(ann).struct) {
        case (ty.ty_fn(?arg_tys, _)) {
            ret arg_tys;
        }
    }
    fail;
}

fn ret_ty_of_fn_ty(@ty.t t) -> @ty.t {
    alt (t.struct) {
        case (ty.ty_fn(_, ?ret_ty)) {
            ret ret_ty;
        }
    }
    fail;
}


fn ret_ty_of_fn(ast.ann ann) -> @ty.t {
    ret ret_ty_of_fn_ty(ty.ann_to_type(ann));
}

fn create_llobjfields_for_fields(@block_ctxt cx, ValueRef llself) {

    let vec[TypeRef] llfield_tys = vec();

    for (ast.obj_field f in cx.fcx.ccx.obj_fields) {
        llfield_tys += node_type(cx.fcx.ccx, f.ann);
    }

    let TypeRef llfields_ty = T_struct(llfield_tys);
    let TypeRef lltydesc_ty = T_ptr(T_tydesc());
    let TypeRef llobj_body_ty = T_struct(vec(lltydesc_ty,
                                             llfields_ty));
    let TypeRef llobj_box_ty = T_ptr(T_box(llobj_body_ty));

    auto box_cell =
        cx.build.GEP(llself,
                     vec(C_int(0),
                         C_int(abi.obj_field_box)));

    auto box_ptr = cx.build.Load(box_cell);

    box_ptr = cx.build.PointerCast(box_ptr, llobj_box_ty);

    auto obj_fields = cx.build.GEP(box_ptr,
                                   vec(C_int(0),
                                       C_int(abi.box_rc_field_body),
                                       C_int(abi.obj_body_elt_fields)));

    let int i = 0;
    for (ast.obj_field f in cx.fcx.ccx.obj_fields) {
        let ValueRef llfield = cx.build.GEP(obj_fields,
                                            vec(C_int(0),
                                                C_int(i)));
        cx.fcx.llobjfields.insert(f.id, llfield);
        i += 1;
    }
}

fn trans_fn(@crate_ctxt cx, &ast._fn f, ast.def_id fid,
            option.t[TypeRef] ty_self,
            &vec[ast.ty_param] ty_params, &ast.ann ann) {

    auto llfndecl = cx.item_ids.get(fid);
    cx.item_names.insert(cx.path, llfndecl);

    auto fcx = new_fn_ctxt(cx, cx.path, llfndecl);
    create_llargs_for_fn_args(fcx, ty_self, ret_ty_of_fn(ann),
                              f.inputs, ty_params);
    auto bcx = new_top_block_ctxt(fcx);

    copy_args_to_allocas(bcx, ty_self, f.inputs,
                         arg_tys_of_fn(ann));

    alt (fcx.llself) {
        case (some[ValueRef](?llself)) {
            create_llobjfields_for_fields(bcx, llself);
        }
        case (_) {
        }
    }

    auto res = trans_block(bcx, f.body);
    if (!is_terminated(res.bcx)) {
        // FIXME: until LLVM has a unit type, we are moving around
        // C_nil values rather than their void type.
        res.bcx.build.RetVoid();
    }
}

fn trans_vtbl(@crate_ctxt cx, TypeRef self_ty,
              &ast._obj ob,
              &vec[ast.ty_param] ty_params) -> ValueRef {
    let vec[ValueRef] methods = vec();

    fn meth_lteq(&@ast.method a, &@ast.method b) -> bool {
        ret _str.lteq(a.node.ident, b.node.ident);
    }

    auto meths = std.sort.merge_sort[@ast.method](bind meth_lteq(_,_),
                                                  ob.methods);

    for (@ast.method m in meths) {

        auto llfnty = T_nil();
        alt (node_ann_type(cx, m.node.ann).struct) {
            case (ty.ty_fn(?inputs, ?output)) {
                llfnty = type_of_fn_full(cx,
                                         some[TypeRef](self_ty),
                                         inputs, output);
            }
        }

        let @crate_ctxt mcx = @rec(path=cx.path + "." + m.node.ident
                                   with *cx);

        let str s = cx.names.next("_rust_method") + "." + mcx.path;
        let ValueRef llfn = decl_fastcall_fn(cx.llmod, s, llfnty);
        cx.item_ids.insert(m.node.id, llfn);

        trans_fn(mcx, m.node.meth, m.node.id, some[TypeRef](self_ty),
                 ty_params, m.node.ann);
        methods += llfn;
    }
    auto vtbl = C_struct(methods);
    auto gvar = llvm.LLVMAddGlobal(cx.llmod,
                                   val_ty(vtbl),
                                   _str.buf("_rust_vtbl" + "." + cx.path));
    llvm.LLVMSetInitializer(gvar, vtbl);
    llvm.LLVMSetGlobalConstant(gvar, True);
    llvm.LLVMSetLinkage(gvar, lib.llvm.LLVMPrivateLinkage
                        as llvm.Linkage);
    ret gvar;
}

fn trans_obj(@crate_ctxt cx, &ast._obj ob, ast.def_id oid,
             &vec[ast.ty_param] ty_params, &ast.ann ann) {

    auto llctor_decl = cx.item_ids.get(oid);
    cx.item_names.insert(cx.path, llctor_decl);

    // Translate obj ctor args to function arguments.
    let vec[ast.arg] fn_args = vec();
    for (ast.obj_field f in ob.fields) {
        fn_args += vec(rec(mode=ast.alias,
                           ty=f.ty,
                           ident=f.ident,
                           id=f.id));
    }

    auto fcx = new_fn_ctxt(cx, cx.path, llctor_decl);
    create_llargs_for_fn_args(fcx, none[TypeRef], ret_ty_of_fn(ann),
                              fn_args, ty_params);

    auto bcx = new_top_block_ctxt(fcx);

    let vec[ty.arg] arg_tys = arg_tys_of_fn(ann);
    copy_args_to_allocas(bcx, none[TypeRef], fn_args, arg_tys);

    auto llself_ty = type_of(cx, ret_ty_of_fn(ann));
    auto pair = bcx.build.Alloca(llself_ty);
    auto vtbl = trans_vtbl(cx, llself_ty, ob, ty_params);
    auto pair_vtbl = bcx.build.GEP(pair,
                                   vec(C_int(0),
                                       C_int(abi.obj_field_vtbl)));
    auto pair_box = bcx.build.GEP(pair,
                                  vec(C_int(0),
                                      C_int(abi.obj_field_box)));
    bcx.build.Store(vtbl, pair_vtbl);

    let TypeRef llbox_ty = T_ptr(T_box(T_struct(vec(T_ptr(T_tydesc()),
                                                    T_nil()))));
    if (_vec.len[ty.arg](arg_tys) == 0u) {
        // Store null into pair, if no args.
        bcx.build.Store(C_null(llbox_ty), pair_box);
    } else {
        // Malloc a box for the body and copy args in.
        let vec[@ty.t] obj_fields = vec();
        for (ty.arg a in arg_tys) {
            append[@ty.t](obj_fields, a.ty);
        }

        // Synthesize an obj body type.
        let @ty.t fields_ty = ty.plain_ty(ty.ty_tup(obj_fields));
        let TypeRef llfields_ty = type_of(bcx.fcx.ccx, fields_ty);
        let TypeRef llobj_body_ty =
            T_ptr(T_box(T_struct(vec(T_ptr(T_tydesc()),
                                     llfields_ty))));

        // Malloc a box for the body.
        auto r = trans_malloc_inner(bcx, llobj_body_ty);
        auto box = r.val;
        auto rc = r.bcx.build.GEP(box,
                                  vec(C_int(0),
                                      C_int(abi.box_rc_field_refcnt)));
        auto body = r.bcx.build.GEP(box,
                                    vec(C_int(0),
                                        C_int(abi.box_rc_field_body)));
        r.bcx.build.Store(C_int(1), rc);

        // Store body tydesc.
        auto body_tydesc =
            r.bcx.build.GEP(body,
                            vec(C_int(0),
                                C_int(abi.obj_body_elt_tydesc)));

        auto fields_tydesc = get_tydesc(r.bcx, fields_ty);
        r.bcx.build.Store(fields_tydesc, body_tydesc);

        // Copy args into body fields.
        auto body_fields =
            r.bcx.build.GEP(body,
                            vec(C_int(0),
                                C_int(abi.obj_body_elt_fields)));

        let int i = 0;
        for (ast.obj_field f in ob.fields) {
            auto arg = r.bcx.fcx.llargs.get(f.id);
            arg = load_scalar_or_boxed(r.bcx, arg, arg_tys.(i).ty);
            auto field = r.bcx.build.GEP(body_fields,
                                         vec(C_int(0),C_int(i)));
            r = copy_ty(r.bcx, INIT, field, arg, arg_tys.(i).ty);
            i += 1;
        }

        // Store box ptr in outer pair.
        auto p = r.bcx.build.PointerCast(box, llbox_ty);
        r.bcx.build.Store(p, pair_box);
    }
    bcx.build.Ret(bcx.build.Load(pair));
}

fn trans_tag_variant(@crate_ctxt cx, ast.def_id tag_id,
                     &ast.variant variant, int index,
                     &vec[ast.ty_param] ty_params) {
    if (_vec.len[ast.variant_arg](variant.args) == 0u) {
        ret;    // nullary constructors are just constants
    }

    // Translate variant arguments to function arguments.
    let vec[ast.arg] fn_args = vec();
    auto i = 0u;
    for (ast.variant_arg varg in variant.args) {
        fn_args += vec(rec(mode=ast.alias,
                           ty=varg.ty,
                           ident="arg" + _uint.to_str(i, 10u),
                           id=varg.id));
    }

    check (cx.item_ids.contains_key(variant.id));
    let ValueRef llfndecl = cx.item_ids.get(variant.id);

    auto fcx = new_fn_ctxt(cx, cx.path, llfndecl);
    create_llargs_for_fn_args(fcx, none[TypeRef], ret_ty_of_fn(variant.ann),
                              fn_args, ty_params);

    auto bcx = new_top_block_ctxt(fcx);

    auto arg_tys = arg_tys_of_fn(variant.ann);
    copy_args_to_allocas(bcx, none[TypeRef], fn_args, arg_tys);

    auto info = cx.tags.get(tag_id);

    auto lltagty = T_struct(vec(T_int(), T_array(T_i8(), info.size)));

    // FIXME: better name.
    llvm.LLVMAddTypeName(cx.llmod, _str.buf("tag"), lltagty);

    auto lltagptr = bcx.build.Alloca(lltagty);
    auto lldiscrimptr = bcx.build.GEP(lltagptr, vec(C_int(0), C_int(0)));
    bcx.build.Store(C_int(index), lldiscrimptr);

    auto llblobptr = bcx.build.GEP(lltagptr, vec(C_int(0), C_int(1)));

    // First, generate the union type.
    let vec[TypeRef] llargtys = vec();
    for (ty.arg arg in arg_tys) {
        llargtys += vec(type_of(cx, arg.ty));
    }

    auto llunionty = T_struct(llargtys);
    auto llunionptr = bcx.build.TruncOrBitCast(llblobptr, T_ptr(llunionty));

    i = 0u;
    for (ast.variant_arg va in variant.args) {
        auto llargval = bcx.build.Load(fcx.llargs.get(va.id));
        auto lldestptr = bcx.build.GEP(llunionptr,
                                       vec(C_int(0), C_int(i as int)));

        bcx.build.Store(llargval, lldestptr);
        i += 1u;
    }

    auto lltagval = bcx.build.Load(lltagptr);
    bcx = trans_block_cleanups(bcx, find_scope_cx(bcx));
    bcx.build.Ret(lltagval);
}

// FIXME: this should do some structural hash-consing to avoid
// duplicate constants. I think. Maybe LLVM has a magical mode
// that does so later on?

fn trans_const_expr(@crate_ctxt cx, @ast.expr e) -> ValueRef {
    alt (e.node) {
        case (ast.expr_lit(?lit, ?ann)) {
            ret trans_lit(cx, *lit, ann);
        }
    }
}

fn trans_const(@crate_ctxt cx, @ast.expr e,
               &ast.def_id cid, &ast.ann ann) {
    auto t = node_ann_type(cx, ann);
    auto v = trans_const_expr(cx, e);
    if (ty.type_is_scalar(t)) {
        // The scalars come back as 1st class LLVM vals
        // which we have to stick into global constants.
        auto g = llvm.LLVMAddGlobal(cx.llmod, val_ty(v),
                                    _str.buf(cx.names.next(cx.path)));
        llvm.LLVMSetInitializer(g, v);
        llvm.LLVMSetGlobalConstant(g, True);
        llvm.LLVMSetLinkage(g, lib.llvm.LLVMPrivateLinkage
                            as llvm.Linkage);
        cx.consts.insert(cid, g);
    } else {
        cx.consts.insert(cid, v);
    }
}

fn trans_item(@crate_ctxt cx, &ast.item item) {
    alt (item.node) {
        case (ast.item_fn(?name, ?f, ?tps, ?fid, ?ann)) {
            auto sub_cx = @rec(path=cx.path + "." + name with *cx);
            trans_fn(sub_cx, f, fid, none[TypeRef], tps, ann);
        }
        case (ast.item_obj(?name, ?ob, ?tps, ?oid, ?ann)) {
            auto sub_cx = @rec(path=cx.path + "." + name,
                               obj_fields=ob.fields with *cx);
            trans_obj(sub_cx, ob, oid, tps, ann);
        }
        case (ast.item_mod(?name, ?m, _)) {
            auto sub_cx = @rec(path=cx.path + "." + name with *cx);
            trans_mod(sub_cx, m);
        }
        case (ast.item_tag(?name, ?variants, ?tps, ?tag_id)) {
            auto sub_cx = @rec(path=cx.path + "." + name with *cx);
            auto i = 0;
            for (ast.variant variant in variants) {
                trans_tag_variant(sub_cx, tag_id, variant, i, tps);
                i += 1;
            }
        }
        case (ast.item_const(?name, _, ?expr, ?cid, ?ann)) {
            auto sub_cx = @rec(path=cx.path + "." + name with *cx);
            trans_const(sub_cx, expr, cid, ann);
        }
        case (_) { /* fall through */ }
    }
}

fn trans_mod(@crate_ctxt cx, &ast._mod m) {
    for (@ast.item item in m.items) {
        trans_item(cx, *item);
    }
}

fn get_pair_fn_ty(TypeRef llpairty) -> TypeRef {
    // Bit of a kludge: pick the fn typeref out of the pair.
    let vec[TypeRef] pair_tys = vec(T_nil(), T_nil());
    llvm.LLVMGetStructElementTypes(llpairty,
                                   _vec.buf[TypeRef](pair_tys));
    ret llvm.LLVMGetElementType(pair_tys.(0));
}

fn decl_fn_and_pair(@crate_ctxt cx,
                    str kind,
                    str name,
                    &ast.ann ann,
                    ast.def_id id) {

    auto llpairty = node_type(cx, ann);
    auto llfty = get_pair_fn_ty(llpairty);

    // Declare the function itself.
    let str s = cx.names.next("_rust_" + kind) + "." + name;
    let ValueRef llfn = decl_fastcall_fn(cx.llmod, s, llfty);

    // Declare the global constant pair that points to it.
    let str ps = cx.names.next("_rust_" + kind + "_pair") + "." + name;
    let ValueRef gvar = llvm.LLVMAddGlobal(cx.llmod, llpairty,
                                           _str.buf(ps));
    auto pair = C_struct(vec(llfn,
                             C_null(T_opaque_closure_ptr())));

    llvm.LLVMSetInitializer(gvar, pair);
    llvm.LLVMSetGlobalConstant(gvar, True);
    llvm.LLVMSetLinkage(gvar,
                        lib.llvm.LLVMPrivateLinkage
                        as llvm.Linkage);

    cx.item_ids.insert(id, llfn);
    cx.fn_pairs.insert(id, gvar);
}


fn collect_item(&@crate_ctxt cx, @ast.item i) -> @crate_ctxt {

    alt (i.node) {
        case (ast.item_fn(?name, ?f, _, ?fid, ?ann)) {
            // TODO: type-params
            cx.items.insert(fid, i);
            if (! cx.obj_methods.contains_key(fid)) {
                decl_fn_and_pair(cx, "fn", name, ann, fid);
            }
        }

        case (ast.item_obj(?name, ?ob, _, ?oid, ?ann)) {
            // TODO: type-params
            cx.items.insert(oid, i);
            decl_fn_and_pair(cx, "obj_ctor", name, ann, oid);
            for (@ast.method m in ob.methods) {
                cx.obj_methods.insert(m.node.id, ());
            }
        }

        case (ast.item_const(?name, _, _, ?cid, _)) {
            cx.items.insert(cid, i);
        }

        case (ast.item_mod(?name, ?m, ?mid)) {
            cx.items.insert(mid, i);
        }

        case (ast.item_tag(_, ?variants, _, ?tag_id)) {
            auto vi = new_def_hash[uint]();
            auto navi = new_def_hash[uint]();
            let vec[tup(ast.def_id,arity)] variant_info = vec();
            cx.tags.insert(tag_id, @rec(th=mk_type_handle(),
                                        mutable variants=variant_info,
                                        mutable size=0u));
            cx.items.insert(tag_id, i);
        }

        case (_) { /* fall through */ }
    }
    ret cx;
}


fn collect_items(@crate_ctxt cx, @ast.crate crate) {

    let fold.ast_fold[@crate_ctxt] fld =
        fold.new_identity_fold[@crate_ctxt]();

    fld = @rec( update_env_for_item = bind collect_item(_,_)
                with *fld );

    fold.fold_crate[@crate_ctxt](cx, fld, crate);
}

fn collect_tag_ctor(&@crate_ctxt cx, @ast.item i) -> @crate_ctxt {

    alt (i.node) {

        case (ast.item_tag(_, ?variants, _, _)) {
            for (ast.variant variant in variants) {
                if (_vec.len[ast.variant_arg](variant.args) != 0u) {
                    decl_fn_and_pair(cx, "tag", variant.name,
                                     variant.ann, variant.id);
                }
            }
        }

        case (_) { /* fall through */ }
    }
    ret cx;
}

fn collect_tag_ctors(@crate_ctxt cx, @ast.crate crate) {

    let fold.ast_fold[@crate_ctxt] fld =
        fold.new_identity_fold[@crate_ctxt]();

    fld = @rec( update_env_for_item = bind collect_tag_ctor(_,_)
                with *fld );

    fold.fold_crate[@crate_ctxt](cx, fld, crate);
}


// The tag type resolution pass, which determines all the LLVM types that
// correspond to each tag type in the crate.

fn resolve_tag_types_for_item(&@crate_ctxt cx, @ast.item i) -> @crate_ctxt {
    alt (i.node) {
        case (ast.item_tag(_, ?variants, _, ?tag_id)) {
            auto max_align = 0u;
            auto max_size = 0u;

            auto info = cx.tags.get(tag_id);
            let vec[tup(ast.def_id,arity)] variant_info = vec();

            for (ast.variant variant in variants) {
                auto arity_info;
                if (_vec.len[ast.variant_arg](variant.args) > 0u) {
                    auto llvariantty = type_of_variant(cx, variant);
                    auto align =
                        llvm.LLVMPreferredAlignmentOfType(cx.td.lltd,
                                                          llvariantty);
                    auto size =
                        llvm.LLVMStoreSizeOfType(cx.td.lltd,
                                                 llvariantty) as uint;
                    if (max_align < align) { max_align = align; }
                    if (max_size < size) { max_size = size; }

                    arity_info = n_ary;
                } else {
                    arity_info = nullary;
                }

                variant_info += vec(tup(variant.id, arity_info));
            }

            info.variants = variant_info;
            info.size = max_size;

            // FIXME: alignment is wrong here, manually insert padding I
            // guess :(
            auto tag_ty = T_struct(vec(T_int(), T_array(T_i8(), max_size)));
            auto th = cx.tags.get(tag_id).th.llth;
            llvm.LLVMRefineType(llvm.LLVMResolveTypeHandle(th), tag_ty);
        }
        case (_) {
            // fall through
        }
    }

    ret cx;
}

fn resolve_tag_types(@crate_ctxt cx, @ast.crate crate) {
    let fold.ast_fold[@crate_ctxt] fld =
        fold.new_identity_fold[@crate_ctxt]();

    fld = @rec( update_env_for_item = bind resolve_tag_types_for_item(_,_)
                with *fld );

    fold.fold_crate[@crate_ctxt](cx, fld, crate);
}

// The constant translation pass.

fn trans_constant(&@crate_ctxt cx, @ast.item it) -> @crate_ctxt {
    alt (it.node) {
        case (ast.item_tag(_, ?variants, _, ?tag_id)) {
            auto info = cx.tags.get(tag_id);

            auto tag_ty = llvm.LLVMResolveTypeHandle(info.th.llth);
            check (llvm.LLVMCountStructElementTypes(tag_ty) == 2u);
            auto elts = vec(0 as TypeRef, 0 as TypeRef);
            llvm.LLVMGetStructElementTypes(tag_ty, _vec.buf[TypeRef](elts));
            auto union_ty = elts.(1);

            auto i = 0u;
            while (i < _vec.len[tup(ast.def_id,arity)](info.variants)) {
                auto variant_info = info.variants.(i);
                alt (variant_info._1) {
                    case (nullary) {
                        // Nullary tags become constants.
                        auto union_val = C_zero_byte_arr(info.size as uint);
                        auto val = C_struct(vec(C_int(i as int), union_val));

                        // FIXME: better name
                        auto gvar = llvm.LLVMAddGlobal(cx.llmod, val_ty(val),
                                                       _str.buf("tag"));
                        llvm.LLVMSetInitializer(gvar, val);
                        llvm.LLVMSetGlobalConstant(gvar, True);
                        llvm.LLVMSetLinkage(gvar,
                                            lib.llvm.LLVMPrivateLinkage
                                            as llvm.Linkage);
                        cx.item_ids.insert(variant_info._0, gvar);
                    }
                    case (n_ary) {
                        // N-ary tags are treated as functions and generated
                        // later.
                    }
                }

                i += 1u;
            }
        }

        case (ast.item_const(?name, _, ?expr, ?cid, ?ann)) {
            // FIXME: The whole expr-translation system needs cloning to deal
            // with consts.
            auto v = C_int(1);
            cx.item_ids.insert(cid, v);
        }

        case (_) {
            // empty
        }
    }

    ret cx;
}

fn trans_constants(@crate_ctxt cx, @ast.crate crate) {
    let fold.ast_fold[@crate_ctxt] fld =
        fold.new_identity_fold[@crate_ctxt]();

    fld = @rec(update_env_for_item = bind trans_constant(_,_) with *fld);

    fold.fold_crate[@crate_ctxt](cx, fld, crate);
}

fn p2i(ValueRef v) -> ValueRef {
    ret llvm.LLVMConstPtrToInt(v, T_int());
}

fn trans_exit_task_glue(@crate_ctxt cx) {
    let vec[TypeRef] T_args = vec();
    let vec[ValueRef] V_args = vec();

    auto llfn = cx.glues.exit_task_glue;
    let ValueRef lltaskptr = llvm.LLVMGetParam(llfn, 0u);
    auto fcx = @rec(llfn=llfn,
                    lltaskptr=lltaskptr,
                    llclosure=C_null(T_opaque_closure_ptr()),
                    mutable llself=none[ValueRef],
                    mutable llretptr=none[ValueRef],
                    llargs=new_def_hash[ValueRef](),
                    llobjfields=new_def_hash[ValueRef](),
                    lllocals=new_def_hash[ValueRef](),
                    lltydescs=new_def_hash[ValueRef](),
                    ccx=cx);

    auto bcx = new_top_block_ctxt(fcx);
    trans_upcall(bcx, "upcall_exit", V_args);
    bcx.build.RetVoid();
}

fn create_typedefs(@crate_ctxt cx) {
    llvm.LLVMAddTypeName(cx.llmod, _str.buf("rust_crate"), T_crate());
    llvm.LLVMAddTypeName(cx.llmod, _str.buf("rust_task"), T_task());
    llvm.LLVMAddTypeName(cx.llmod, _str.buf("rust_tydesc"), T_tydesc());
}

fn crate_constant(@crate_ctxt cx) -> ValueRef {

    let ValueRef crate_ptr =
        llvm.LLVMAddGlobal(cx.llmod, T_crate(),
                           _str.buf("rust_crate"));

    let ValueRef crate_addr = p2i(crate_ptr);

    let ValueRef activate_glue_off =
        llvm.LLVMConstSub(p2i(cx.glues.activate_glue), crate_addr);

    let ValueRef yield_glue_off =
        llvm.LLVMConstSub(p2i(cx.glues.yield_glue), crate_addr);

    let ValueRef exit_task_glue_off =
        llvm.LLVMConstSub(p2i(cx.glues.exit_task_glue), crate_addr);

    let ValueRef crate_val =
        C_struct(vec(C_null(T_int()),     // ptrdiff_t image_base_off
                     p2i(crate_ptr),      // uintptr_t self_addr
                     C_null(T_int()),     // ptrdiff_t debug_abbrev_off
                     C_null(T_int()),     // size_t debug_abbrev_sz
                     C_null(T_int()),     // ptrdiff_t debug_info_off
                     C_null(T_int()),     // size_t debug_info_sz
                     activate_glue_off,   // size_t activate_glue_off
                     yield_glue_off,      // size_t yield_glue_off
                     C_null(T_int()),     // size_t unwind_glue_off
                     C_null(T_int()),     // size_t gc_glue_off
                     exit_task_glue_off,  // size_t main_exit_task_glue_off
                     C_null(T_int()),     // int n_rust_syms
                     C_null(T_int()),     // int n_c_syms
                     C_null(T_int())      // int n_libs
                     ));

    llvm.LLVMSetInitializer(crate_ptr, crate_val);
    ret crate_ptr;
}

fn trans_main_fn(@crate_ctxt cx, ValueRef llcrate) {
    auto T_main_args = vec(T_int(), T_int());
    auto T_rust_start_args = vec(T_int(), T_int(), T_int(), T_int());

    auto main_name;
    if (_str.eq(std.os.target_os(), "win32")) {
        main_name = "WinMain@16";
    } else {
        main_name = "main";
    }

    auto llmain =
        decl_cdecl_fn(cx.llmod, main_name, T_fn(T_main_args, T_int()));

    auto llrust_start = decl_cdecl_fn(cx.llmod, "rust_start",
                                      T_fn(T_rust_start_args, T_int()));

    auto llargc = llvm.LLVMGetParam(llmain, 0u);
    auto llargv = llvm.LLVMGetParam(llmain, 1u);
    check (cx.item_names.contains_key("_rust.main"));
    auto llrust_main = cx.item_names.get("_rust.main");

    //
    // Emit the moral equivalent of:
    //
    // main(int argc, char **argv) {
    //     rust_start(&_rust.main, &crate, argc, argv);
    // }
    //

    let BasicBlockRef llbb =
        llvm.LLVMAppendBasicBlock(llmain, _str.buf(""));
    auto b = new_builder(llbb);

    auto start_args = vec(p2i(llrust_main), p2i(llcrate), llargc, llargv);

    b.Ret(b.Call(llrust_start, start_args));
}

fn declare_intrinsics(ModuleRef llmod) -> hashmap[str,ValueRef] {

    let vec[TypeRef] T_trap_args = vec();
    auto trap = decl_cdecl_fn(llmod, "llvm.trap",
                              T_fn(T_trap_args, T_void()));

    auto intrinsics = new_str_hash[ValueRef]();
    intrinsics.insert("llvm.trap", trap);
    ret intrinsics;
}

fn check_module(ModuleRef llmod) {
    auto pm = mk_pass_manager();
    llvm.LLVMAddVerifierPass(pm.llpm);
    llvm.LLVMRunPassManager(pm.llpm, llmod);

    // TODO: run the linter here also, once there are llvm-c bindings for it.
}

fn make_no_op_type_glue(ModuleRef llmod) -> ValueRef {
    auto ty = T_fn(vec(T_taskptr(), T_ptr(T_i8())), T_void());
    auto fun = decl_fastcall_fn(llmod, abi.no_op_type_glue_name(), ty);
    auto bb_name = _str.buf("_rust_no_op_type_glue_bb");
    auto llbb = llvm.LLVMAppendBasicBlock(fun, bb_name);
    new_builder(llbb).RetVoid();
    ret fun;
}

fn make_memcpy_glue(ModuleRef llmod) -> ValueRef {

    // We're not using the LLVM memcpy intrinsic. It appears to call through
    // to the platform memcpy in some cases, which is not terribly safe to run
    // on a rust stack.

    auto p8 = T_ptr(T_i8());

    auto ty = T_fn(vec(p8, p8, T_int()), T_void());
    auto fun = decl_fastcall_fn(llmod, abi.memcpy_glue_name(), ty);

    auto initbb = llvm.LLVMAppendBasicBlock(fun, _str.buf("init"));
    auto hdrbb = llvm.LLVMAppendBasicBlock(fun, _str.buf("hdr"));
    auto loopbb = llvm.LLVMAppendBasicBlock(fun, _str.buf("loop"));
    auto endbb = llvm.LLVMAppendBasicBlock(fun, _str.buf("end"));

    auto dst = llvm.LLVMGetParam(fun, 0u);
    auto src = llvm.LLVMGetParam(fun, 1u);
    auto count = llvm.LLVMGetParam(fun, 2u);

    // Init block.
    auto ib = new_builder(initbb);
    auto ip = ib.Alloca(T_int());
    ib.Store(C_int(0), ip);
    ib.Br(hdrbb);

    // Loop-header block
    auto hb = new_builder(hdrbb);
    auto i = hb.Load(ip);
    hb.CondBr(hb.ICmp(lib.llvm.LLVMIntEQ, count, i), endbb, loopbb);

    // Loop-body block
    auto lb = new_builder(loopbb);
    i = lb.Load(ip);
    lb.Store(lb.Load(lb.GEP(src, vec(i))),
             lb.GEP(dst, vec(i)));
    lb.Store(lb.Add(i, C_int(1)), ip);
    lb.Br(hdrbb);

    // End block
    auto eb = new_builder(endbb);
    eb.RetVoid();
    ret fun;
}

fn make_bzero_glue(ModuleRef llmod) -> ValueRef {

    // We're not using the LLVM memset intrinsic. Same as with memcpy.

    auto p8 = T_ptr(T_i8());

    auto ty = T_fn(vec(p8, T_int()), T_void());
    auto fun = decl_fastcall_fn(llmod, abi.bzero_glue_name(), ty);

    auto initbb = llvm.LLVMAppendBasicBlock(fun, _str.buf("init"));
    auto hdrbb = llvm.LLVMAppendBasicBlock(fun, _str.buf("hdr"));
    auto loopbb = llvm.LLVMAppendBasicBlock(fun, _str.buf("loop"));
    auto endbb = llvm.LLVMAppendBasicBlock(fun, _str.buf("end"));

    auto dst = llvm.LLVMGetParam(fun, 0u);
    auto count = llvm.LLVMGetParam(fun, 1u);

    // Init block.
    auto ib = new_builder(initbb);
    auto ip = ib.Alloca(T_int());
    ib.Store(C_int(0), ip);
    ib.Br(hdrbb);

    // Loop-header block
    auto hb = new_builder(hdrbb);
    auto i = hb.Load(ip);
    hb.CondBr(hb.ICmp(lib.llvm.LLVMIntEQ, count, i), endbb, loopbb);

    // Loop-body block
    auto lb = new_builder(loopbb);
    i = lb.Load(ip);
    lb.Store(C_integral(0, T_i8()), lb.GEP(dst, vec(i)));
    lb.Store(lb.Add(i, C_int(1)), ip);
    lb.Br(hdrbb);

    // End block
    auto eb = new_builder(endbb);
    eb.RetVoid();
    ret fun;
}

fn make_glues(ModuleRef llmod) -> @glue_fns {
    ret @rec(activate_glue = decl_glue(llmod, abi.activate_glue_name()),
             yield_glue = decl_glue(llmod, abi.yield_glue_name()),
             /*
              * Note: the signature passed to decl_cdecl_fn here looks unusual
              * because it is. It corresponds neither to an upcall signature
              * nor a normal rust-ABI signature. In fact it is a fake
              * signature, that exists solely to acquire the task pointer as
              * an argument to the upcall. It so happens that the runtime sets
              * up the task pointer as the sole incoming argument to the frame
              * that we return into when returning to the exit task glue. So
              * this is the signature required to retrieve it.
              */
             exit_task_glue = decl_cdecl_fn(llmod, abi.exit_task_glue_name(),
                                            T_fn(vec(T_taskptr()), T_void())),

             upcall_glues =
             _vec.init_fn[ValueRef](bind decl_upcall(llmod, _),
                                    abi.n_upcall_glues as uint),
             no_op_type_glue = make_no_op_type_glue(llmod),
             memcpy_glue = make_memcpy_glue(llmod),
             bzero_glue = make_bzero_glue(llmod));
}

fn trans_crate(session.session sess, @ast.crate crate, str output,
               bool shared) {
    auto llmod =
        llvm.LLVMModuleCreateWithNameInContext(_str.buf("rust_out"),
                                               llvm.LLVMGetGlobalContext());

    llvm.LLVMSetDataLayout(llmod, _str.buf(x86.get_data_layout()));
    llvm.LLVMSetTarget(llmod, _str.buf(x86.get_target_triple()));
    auto td = mk_target_data(x86.get_data_layout());

    llvm.LLVMSetModuleInlineAsm(llmod, _str.buf(x86.get_module_asm()));

    auto intrinsics = declare_intrinsics(llmod);

    auto glues = make_glues(llmod);
    auto hasher = ty.hash_ty;
    auto eqer = ty.eq_ty;
    auto tydescs = map.mk_hashmap[@ty.t,ValueRef](hasher, eqer);
    let vec[ast.obj_field] obj_fields = vec();

    auto cx = @rec(sess = sess,
                   llmod = llmod,
                   td = td,
                   upcalls = new_str_hash[ValueRef](),
                   intrinsics = intrinsics,
                   item_names = new_str_hash[ValueRef](),
                   item_ids = new_def_hash[ValueRef](),
                   items = new_def_hash[@ast.item](),
                   tags = new_def_hash[@tag_info](),
                   fn_pairs = new_def_hash[ValueRef](),
                   consts = new_def_hash[ValueRef](),
                   obj_methods = new_def_hash[()](),
                   tydescs = tydescs,
                   obj_fields = obj_fields,
                   glues = glues,
                   names = namegen(0),
                   path = "_rust");

    create_typedefs(cx);

    collect_items(cx, crate);
    resolve_tag_types(cx, crate);
    collect_tag_ctors(cx, crate);
    trans_constants(cx, crate);

    trans_mod(cx, crate.node.module);
    trans_exit_task_glue(cx);
    if (!shared) {
        trans_main_fn(cx, crate_constant(cx));
    }

    check_module(llmod);

    llvm.LLVMWriteBitcodeToFile(llmod, _str.buf(output));
    llvm.LLVMDisposeModule(llmod);
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
