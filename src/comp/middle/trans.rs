import std._int;
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
import middle.ty.plain_ty;

import util.common;
import util.common.append;
import util.common.istr;
import util.common.new_def_hash;
import util.common.new_str_hash;

import lib.llvm.llvm;
import lib.llvm.builder;
import lib.llvm.target_data;
import lib.llvm.type_handle;
import lib.llvm.type_names;
import lib.llvm.mk_pass_manager;
import lib.llvm.mk_target_data;
import lib.llvm.mk_type_handle;
import lib.llvm.mk_type_names;
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
type tag_info = rec(type_handle th, mutable uint size);

state type crate_ctxt = rec(session.session sess,
                            ModuleRef llmod,
                            target_data td,
                            type_names tn,
                            ValueRef crate_ptr,
                            hashmap[str, ValueRef] upcalls,
                            hashmap[str, ValueRef] intrinsics,
                            hashmap[str, ValueRef] item_names,
                            hashmap[ast.def_id, ValueRef] item_ids,
                            hashmap[ast.def_id, @ast.item] items,
                            hashmap[ast.def_id,
                                    @ast.native_item] native_items,
                            hashmap[@ty.t, @tag_info] tags,
                            hashmap[ast.def_id, ValueRef] fn_pairs,
                            hashmap[ast.def_id, ValueRef] consts,
                            hashmap[ast.def_id,()] obj_methods,
                            hashmap[@ty.t, ValueRef] tydescs,
                            vec[ast.ty_param] obj_typarams,
                            vec[ast.obj_field] obj_fields,
                            @glue_fns glues,
                            namegen names,
                            str path);

state type fn_ctxt = rec(ValueRef llfn,
                         ValueRef lltaskptr,
                         ValueRef llenv,
                         ValueRef llretptr,
                         mutable option.t[ValueRef] llself,
                         mutable option.t[ValueRef] lliterbody,
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

fn sep() -> str {
    ret "_";
}

fn res(@block_ctxt bcx, ValueRef val) -> result {
    ret rec(mutable bcx = bcx,
            mutable val = val);
}

fn ty_str(type_names tn, TypeRef t) -> str {
    ret lib.llvm.type_to_str(tn, t);
}

fn val_ty(ValueRef v) -> TypeRef {
    ret llvm.LLVMTypeOf(v);
}

fn val_str(type_names tn, ValueRef v) -> str {
    ret ty_str(tn, val_ty(v));
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

fn T_fn_pair(type_names tn, TypeRef tfn) -> TypeRef {
    ret T_struct(vec(T_ptr(tfn),
                     T_opaque_closure_ptr(tn)));
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

fn T_task(type_names tn) -> TypeRef {
    auto s = "task";
    if (tn.name_has_type(s)) {
        ret tn.get_type(s);
    }

    auto t = T_struct(vec(T_int(),      // Refcount
                          T_int(),      // Delegate pointer
                          T_int(),      // Stack segment pointer
                          T_int(),      // Runtime SP
                          T_int(),      // Rust SP
                          T_int(),      // GC chain
                          T_int(),      // Domain pointer
                          T_int()       // Crate cache pointer
                          ));
    tn.associate(s, t);
    ret t;
}

fn T_glue_fn(type_names tn) -> TypeRef {
    auto s = "glue_fn";
    if (tn.name_has_type(s)) {
        ret tn.get_type(s);
    }

    // Bit of a kludge: pick the fn typeref out of the tydesc..
    let vec[TypeRef] tydesc_elts = _vec.init_elt[TypeRef](T_nil(), 10u);
    llvm.LLVMGetStructElementTypes(T_tydesc(tn),
                                   _vec.buf[TypeRef](tydesc_elts));
    auto t =
        llvm.LLVMGetElementType
        (tydesc_elts.(abi.tydesc_field_drop_glue_off));
    tn.associate(s, t);
    ret t;
}

fn T_tydesc(type_names tn) -> TypeRef {

    auto s = "tydesc";
    if (tn.name_has_type(s)) {
        ret tn.get_type(s);
    }

    auto th = mk_type_handle();
    auto abs_tydesc = llvm.LLVMResolveTypeHandle(th.llth);
    auto tydescpp = T_ptr(T_ptr(abs_tydesc));
    auto pvoid = T_ptr(T_i8());
    auto glue_fn_ty = T_ptr(T_fn(vec(T_ptr(T_nil()),
                                     T_taskptr(tn),
                                     T_ptr(T_nil()),
                                     tydescpp,
                                     pvoid), T_void()));
    auto tydesc = T_struct(vec(tydescpp,          // first_param
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
    auto t = llvm.LLVMResolveTypeHandle(th.llth);
    tn.associate(s, t);
    ret t;
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

fn T_crate(type_names tn) -> TypeRef {
    auto s = "crate";
    if (tn.name_has_type(s)) {
        ret tn.get_type(s);
    }

    auto t = T_struct(vec(T_int(),      // ptrdiff_t image_base_off
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
                          T_int(),      // int n_libs
                          T_int()       // uintptr_t abi_tag
                          ));
    tn.associate(s, t);
    ret t;
}

fn T_double() -> TypeRef {
    ret llvm.LLVMDoubleType();
}

fn T_taskptr(type_names tn) -> TypeRef {
    ret T_ptr(T_task(tn));
}

fn T_typaram_ptr(type_names tn) -> TypeRef {
    auto s = "typaram";
    if (tn.name_has_type(s)) {
        ret tn.get_type(s);
    }

    auto t = T_ptr(T_i8());
    tn.associate(s, t);
    ret t;
}

fn T_closure_ptr(type_names tn,
                 TypeRef lltarget_ty,
                 TypeRef llbindings_ty,
                 uint n_ty_params) -> TypeRef {
    ret T_ptr(T_box(T_struct(vec(T_ptr(T_tydesc(tn)),
                                 lltarget_ty,
                                 llbindings_ty,
                                 T_captured_tydescs(tn, n_ty_params))
                             )));
}

fn T_opaque_closure_ptr(type_names tn) -> TypeRef {
    auto s = "*closure";
    if (tn.name_has_type(s)) {
        ret tn.get_type(s);
    }
    auto t = T_closure_ptr(tn, T_struct(vec(T_ptr(T_nil()),
                                            T_ptr(T_nil()))),
                           T_nil(),
                           0u);
    tn.associate(s, t);
    ret t;
}

fn T_captured_tydescs(type_names tn, uint n) -> TypeRef {
    ret T_struct(_vec.init_elt[TypeRef](T_ptr(T_tydesc(tn)), n));
}

fn T_obj(type_names tn, uint n_captured_tydescs,
         TypeRef llfields_ty) -> TypeRef {
    ret T_struct(vec(T_ptr(T_tydesc(tn)),
                     T_captured_tydescs(tn, n_captured_tydescs),
                     llfields_ty));
}

fn T_obj_ptr(type_names tn, uint n_captured_tydescs,
             TypeRef llfields_ty) -> TypeRef {
    ret T_ptr(T_box(T_obj(tn, n_captured_tydescs, llfields_ty)));
}

fn T_opaque_obj_ptr(type_names tn) -> TypeRef {
    ret T_obj_ptr(tn, 0u, T_nil());
}


fn type_of(@crate_ctxt cx, @ty.t t) -> TypeRef {
    ret type_of_inner(cx, t);
}

fn type_of_explicit_args(@crate_ctxt cx,
                     vec[ty.arg] inputs) -> vec[TypeRef] {
    let vec[TypeRef] atys = vec();
    for (ty.arg arg in inputs) {
        if (ty.type_has_dynamic_size(arg.ty)) {
            check (arg.mode == ast.alias);
            atys += T_typaram_ptr(cx.tn);
        } else {
            let TypeRef t = type_of_inner(cx, arg.ty);
            alt (arg.mode) {
                case (ast.alias) {
                    t = T_ptr(t);
                }
                case (_) { /* fall through */  }
            }
            atys += t;
        }
    }
    ret atys;
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn_full
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args

fn type_of_fn_full(@crate_ctxt cx,
                   ast.proto proto,
                   option.t[TypeRef] obj_self,
                   vec[ty.arg] inputs,
                   @ty.t output) -> TypeRef {
    let vec[TypeRef] atys = vec();

    // Arg 0: Output pointer.
    if (ty.type_has_dynamic_size(output)) {
        atys += T_typaram_ptr(cx.tn);
    } else {
        atys += T_ptr(type_of_inner(cx, output));
    }

    // Arg 1: Task pointer.
    atys += T_taskptr(cx.tn);

    // Arg 2: Env (closure-bindings / self-obj)
    alt (obj_self) {
        case (some[TypeRef](?t)) {
            check (t as int != 0);
            atys += t;
        }
        case (_) {
            atys += T_opaque_closure_ptr(cx.tn);
        }
    }

    // Args >3: ty params, if not acquired via capture...
    if (obj_self == none[TypeRef]) {
        auto ty_param_count =
            ty.count_ty_params(plain_ty(ty.ty_fn(proto,
                                                 inputs,
                                                 output)));
        auto i = 0u;
        while (i < ty_param_count) {
            atys += T_ptr(T_tydesc(cx.tn));
            i += 1u;
        }
    }

    if (proto == ast.proto_iter) {
        // If it's an iter, the 'output' type of the iter is actually the
        // *input* type of the function we're given as our iter-block
        // argument.
        atys += T_fn_pair(cx.tn,
                          type_of_fn_full(cx, ast.proto_fn, none[TypeRef],
                                          vec(rec(mode=ast.val, ty=output)),
                                          plain_ty(ty.ty_nil)));
    }

    // ... then explicit args.
    atys += type_of_explicit_args(cx, inputs);

    ret T_fn(atys, llvm.LLVMVoidType());
}

fn type_of_fn(@crate_ctxt cx,
              ast.proto proto,
              vec[ty.arg] inputs, @ty.t output) -> TypeRef {
    ret type_of_fn_full(cx, proto, none[TypeRef], inputs, output);
}

fn type_of_native_fn(@crate_ctxt cx, ast.native_abi abi,
                     vec[ty.arg] inputs,
                     @ty.t output) -> TypeRef {
    let vec[TypeRef] atys = vec();
    if (abi == ast.native_abi_rust) {
        atys += T_taskptr(cx.tn);
        auto t = ty.ty_native_fn(abi, inputs, output);
        auto ty_param_count = ty.count_ty_params(plain_ty(t));
        auto i = 0u;
        while (i < ty_param_count) {
            atys += T_ptr(T_tydesc(cx.tn));
            i += 1u;
        }
    }
    atys += type_of_explicit_args(cx, inputs);
    ret T_fn(atys, type_of_inner(cx, output));
}

fn type_of_inner(@crate_ctxt cx, @ty.t t) -> TypeRef {
    let TypeRef llty = 0 as TypeRef;

    alt (t.struct) {
        case (ty.ty_native) { llty = T_ptr(T_i8()); }
        case (ty.ty_nil) { llty = T_nil(); }
        case (ty.ty_bool) { llty = T_bool(); }
        case (ty.ty_int) { llty = T_int(); }
        case (ty.ty_uint) { llty = T_int(); }
        case (ty.ty_machine(?tm)) {
            alt (tm) {
                case (common.ty_i8) { llty = T_i8(); }
                case (common.ty_u8) { llty = T_i8(); }
                case (common.ty_i16) { llty = T_i16(); }
                case (common.ty_u16) { llty = T_i16(); }
                case (common.ty_i32) { llty = T_i32(); }
                case (common.ty_u32) { llty = T_i32(); }
                case (common.ty_i64) { llty = T_i64(); }
                case (common.ty_u64) { llty = T_i64(); }
                case (common.ty_f32) { llty = T_f32(); }
                case (common.ty_f64) { llty = T_f64(); }
            }
        }
        case (ty.ty_char) { llty = T_char(); }
        case (ty.ty_str) { llty = T_ptr(T_str()); }
        case (ty.ty_tag(?tag_id, _)) {
            llty = llvm.LLVMResolveTypeHandle(cx.tags.get(t).th.llth);
        }
        case (ty.ty_box(?t)) {
            llty = T_ptr(T_box(type_of_inner(cx, t)));
        }
        case (ty.ty_vec(?t)) {
            llty = T_ptr(T_vec(type_of_inner(cx, t)));
        }
        case (ty.ty_tup(?elts)) {
            let vec[TypeRef] tys = vec();
            for (@ty.t elt in elts) {
                tys += type_of_inner(cx, elt);
            }
            llty = T_struct(tys);
        }
        case (ty.ty_rec(?fields)) {
            let vec[TypeRef] tys = vec();
            for (ty.field f in fields) {
                tys += type_of_inner(cx, f.ty);
            }
            llty = T_struct(tys);
        }
        case (ty.ty_fn(?proto, ?args, ?out)) {
            llty = T_fn_pair(cx.tn, type_of_fn(cx, proto, args, out));
        }
        case (ty.ty_native_fn(?abi, ?args, ?out)) {
            llty = T_fn_pair(cx.tn, type_of_native_fn(cx, abi, args, out));
        }
        case (ty.ty_obj(?meths)) {
            auto th = mk_type_handle();
            auto self_ty = llvm.LLVMResolveTypeHandle(th.llth);

            let vec[TypeRef] mtys = vec();
            for (ty.method m in meths) {
                let TypeRef mty =
                    type_of_fn_full(cx, m.proto,
                                    some[TypeRef](self_ty),
                                    m.inputs, m.output);
                mtys += T_ptr(mty);
            }
            let TypeRef vtbl = T_struct(mtys);
            let TypeRef pair = T_struct(vec(T_ptr(vtbl),
                                            T_opaque_obj_ptr(cx.tn)));

            auto abs_pair = llvm.LLVMResolveTypeHandle(th.llth);
            llvm.LLVMRefineType(abs_pair, pair);
            abs_pair = llvm.LLVMResolveTypeHandle(th.llth);
            llty = abs_pair;
        }
        case (ty.ty_var(_)) {
            log "ty_var in trans.type_of";
            fail;
        }
        case (ty.ty_param(_)) {
            llty = T_typaram_ptr(cx.tn);
        }
        case (ty.ty_type) { llty = T_ptr(T_tydesc(cx.tn)); }
    }

    check (llty as int != 0);
    llvm.LLVMAddTypeName(cx.llmod, _str.buf(ty.ty_to_str(t)), llty);
    ret llty;
}

fn type_of_arg(@crate_ctxt cx, &ty.arg arg) -> TypeRef {
    auto ty = type_of_inner(cx, arg.ty);
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

fn decl_glue(ModuleRef llmod, type_names tn, str s) -> ValueRef {
    ret decl_cdecl_fn(llmod, s, T_fn(vec(T_taskptr(tn)), T_void()));
}

fn decl_upcall_glue(ModuleRef llmod, type_names tn, uint _n) -> ValueRef {
    // It doesn't actually matter what type we come up with here, at the
    // moment, as we cast the upcall function pointers to int before passing
    // them to the indirect upcall-invocation glue.  But eventually we'd like
    // to call them directly, once we have a calling convention worked out.
    let int n = _n as int;
    let str s = abi.upcall_glue_name(n);
    let vec[TypeRef] args =
        vec(T_taskptr(tn), // taskptr
            T_int())     // callee
        + _vec.init_elt[TypeRef](T_int(), n as uint);

    ret decl_fastcall_fn(llmod, s, T_fn(args, T_int()));
}

fn get_upcall(@crate_ctxt cx, str name, int n_args) -> ValueRef {
    if (cx.upcalls.contains_key(name)) {
        ret cx.upcalls.get(name);
    }
    auto inputs = vec(T_taskptr(cx.tn));
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

fn size_of(@block_ctxt cx, @ty.t t) -> result {
    if (!ty.type_has_dynamic_size(t)) {
        ret res(cx, llsize_of(type_of(cx.fcx.ccx, t)));
    }
    ret dynamic_size_of(cx, t);
}

fn align_of(@block_ctxt cx, @ty.t t) -> result {
    if (!ty.type_has_dynamic_size(t)) {
        ret res(cx, llalign_of(type_of(cx.fcx.ccx, t)));
    }
    ret dynamic_align_of(cx, t);
}

fn dynamic_size_of(@block_ctxt cx, @ty.t t) -> result {
    alt (t.struct) {
        case (ty.ty_param(?p)) {
            auto szptr = field_of_tydesc(cx, t, abi.tydesc_field_size);
            ret res(szptr.bcx, szptr.bcx.build.Load(szptr.val));
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
            auto bcx = cx;
            for (@ty.t e in elts) {
                auto elt_align = align_of(bcx, e);
                bcx = elt_align.bcx;
                auto elt_size = size_of(bcx, e);
                bcx = elt_size.bcx;
                auto aligned_off = align_to(bcx, off, elt_align.val);
                off = cx.build.Add(aligned_off, elt_size.val);
                max_align = umax(bcx, max_align, elt_align.val);
            }
            off = align_to(bcx, off, max_align);
            ret res(bcx, off);
        }
        case (ty.ty_rec(?flds)) {
            auto off = C_int(0);
            auto max_align = C_int(1);
            auto bcx = cx;
            for (ty.field f in flds) {
                auto elt_align = align_of(bcx, f.ty);
                bcx = elt_align.bcx;
                auto elt_size = size_of(bcx, f.ty);
                bcx = elt_size.bcx;
                auto aligned_off = align_to(bcx, off, elt_align.val);
                off = cx.build.Add(aligned_off, elt_size.val);
                max_align = umax(bcx, max_align, elt_align.val);
            }
            off = align_to(bcx, off, max_align);
            ret res(bcx, off);
        }
    }
}

fn dynamic_align_of(@block_ctxt cx, @ty.t t) -> result {
    alt (t.struct) {
        case (ty.ty_param(?p)) {
            auto aptr = field_of_tydesc(cx, t, abi.tydesc_field_align);
            ret res(aptr.bcx, aptr.bcx.build.Load(aptr.val));
        }
        case (ty.ty_tup(?elts)) {
            auto a = C_int(1);
            auto bcx = cx;
            for (@ty.t e in elts) {
                auto align = align_of(bcx, e);
                bcx = align.bcx;
                a = umax(bcx, a, align.val);
            }
            ret res(bcx, a);
        }
        case (ty.ty_rec(?flds)) {
            auto a = C_int(1);
            auto bcx = cx;
            for (ty.field f in flds) {
                auto align = align_of(bcx, f.ty);
                bcx = align.bcx;
                a = umax(bcx, a, align.val);
            }
            ret res(bcx, a);
        }
    }
}

// Replacement for the LLVM 'GEP' instruction when field-indexing into a
// tuple-like structure (tup, rec, tag) with a static index. This one is
// driven off ty.struct and knows what to do when it runs into a ty_param
// stuck in the middle of the thing it's GEP'ing into. Much like size_of and
// align_of, above.

fn GEP_tup_like(@block_ctxt cx, @ty.t t,
                ValueRef base, vec[int] ixs) -> result {

    check (ty.type_is_tup_like(t));

    // It might be a static-known type. Handle this.

    if (! ty.type_has_dynamic_size(t)) {
        let vec[ValueRef] v = vec();
        for (int i in ixs) {
            v += C_int(i);
        }
        ret res(cx, cx.build.GEP(base, v));
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
    auto prefix_ty = plain_ty(ty.ty_tup(s.prefix));
    auto bcx = cx;
    auto sz = size_of(bcx, prefix_ty);
    bcx = sz.bcx;
    auto raw = bcx.build.PointerCast(base, T_ptr(T_i8()));
    auto bumped = bcx.build.GEP(raw, vec(sz.val));
    alt (s.target.struct) {
        case (ty.ty_param(_)) { ret res(bcx, bumped); }
        case (_) {
            auto ty = T_ptr(type_of(bcx.fcx.ccx, s.target));
            ret res(bcx, bcx.build.PointerCast(bumped, ty));
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
fn field_of_tydesc(@block_ctxt cx, @ty.t t, int field) -> result {
    auto tydesc = get_tydesc(cx, t);
    ret res(tydesc.bcx,
            tydesc.bcx.build.GEP(tydesc.val, vec(C_int(0), C_int(field))));
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
                        r.vals += r.cx.fcx.lltydescs.get(pid);
                        r.defs += pid;
                    }
                }
                case (_) { }
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

fn get_tydesc(&@block_ctxt cx, @ty.t t) -> result {
    // Is the supplied type a type param? If so, return the passed-in tydesc.
    alt (ty.type_param(t)) {
        case (some[ast.def_id](?id)) {
            check (cx.fcx.lltydescs.contains_key(id));
            ret res(cx, cx.fcx.lltydescs.get(id));
        }
        case (none[ast.def_id])      { /* fall through */ }
    }

    // Does it contain a type param? If so, generate a derived tydesc.
    let uint n_params = ty.count_ty_params(t);

    if (ty.count_ty_params(t) > 0u) {
        auto tys = linearize_ty_params(cx, t);

        check (n_params == _vec.len[ast.def_id](tys._0));
        check (n_params == _vec.len[ValueRef](tys._1));

        if (!cx.fcx.ccx.tydescs.contains_key(t)) {
            make_tydesc(cx.fcx.ccx, t, tys._0);
        }

        auto root = cx.fcx.ccx.tydescs.get(t);

        auto tydescs = cx.build.Alloca(T_array(T_ptr(T_tydesc(cx.fcx.ccx.tn)),
                                               n_params));

        auto i = 0;
        auto tdp = cx.build.GEP(tydescs, vec(C_int(0), C_int(i)));
        cx.build.Store(root, tdp);
        i += 1;
        for (ValueRef td in tys._1) {
            auto tdp = cx.build.GEP(tydescs, vec(C_int(0), C_int(i)));
            cx.build.Store(td, tdp);
            i += 1;
        }

        auto bcx = cx;
        auto sz = size_of(bcx, t);
        bcx = sz.bcx;
        auto align = align_of(bcx, t);
        bcx = align.bcx;

        auto v = trans_upcall(bcx, "upcall_get_type_desc",
                              vec(p2i(bcx.fcx.ccx.crate_ptr),
                                  sz.val,
                                  align.val,
                                  C_int((1u + n_params) as int),
                                  bcx.build.PtrToInt(tydescs, T_int())));

        ret res(v.bcx, v.bcx.build.IntToPtr(v.val,
                                            T_ptr(T_tydesc(cx.fcx.ccx.tn))));
    }

    // Otherwise, generate a tydesc if necessary, and return it.
    if (!cx.fcx.ccx.tydescs.contains_key(t)) {
        let vec[ast.def_id] defs = vec();
        make_tydesc(cx.fcx.ccx, t, defs);
    }
    ret res(cx, cx.fcx.ccx.tydescs.get(t));
}

fn make_tydesc(@crate_ctxt cx, @ty.t t, vec[ast.def_id] typaram_defs) {
    auto tg = make_take_glue;
    auto take_glue = make_generic_glue(cx, t, "take", tg, typaram_defs);
    auto dg = make_drop_glue;
    auto drop_glue = make_generic_glue(cx, t, "drop", dg, typaram_defs);

    auto llty = type_of(cx, t);
    auto glue_fn_ty = T_ptr(T_glue_fn(cx.tn));

    // FIXME: this adjustment has to do with the ridiculous encoding of
    // glue-pointer-constants in the tydesc records: They are tydesc-relative
    // displacements.  This is purely for compatibility with rustboot and
    // should go when it is discarded.
    fn off(ValueRef tydescp,
           ValueRef gluefn) -> ValueRef {
        ret i2p(llvm.LLVMConstSub(p2i(gluefn), p2i(tydescp)),
                val_ty(gluefn));
    }

    auto name = sanitize(cx.names.next("tydesc_" + ty.ty_to_str(t)));
    auto gvar = llvm.LLVMAddGlobal(cx.llmod, T_tydesc(cx.tn),
                                   _str.buf(name));
    auto tydesc = C_struct(vec(C_null(T_ptr(T_ptr(T_tydesc(cx.tn)))),
                               llsize_of(llty),
                               llalign_of(llty),
                               off(gvar, take_glue),  // take_glue_off
                               off(gvar, drop_glue),  // drop_glue_off
                               C_null(glue_fn_ty),    // free_glue_off
                               C_null(glue_fn_ty),    // sever_glue_off
                               C_null(glue_fn_ty),    // mark_glue_off
                               C_null(glue_fn_ty),    // obj_drop_glue_off
                               C_null(glue_fn_ty)));  // is_stateful

    llvm.LLVMSetInitializer(gvar, tydesc);
    llvm.LLVMSetGlobalConstant(gvar, True);
    llvm.LLVMSetLinkage(gvar, lib.llvm.LLVMPrivateLinkage
                        as llvm.Linkage);
    cx.tydescs.insert(t, gvar);
}

fn make_generic_glue(@crate_ctxt cx, @ty.t t, str name,
                     val_and_ty_fn helper,
                     vec[ast.def_id] typaram_defs) -> ValueRef {
    auto llfnty = T_glue_fn(cx.tn);

    auto fn_name = cx.names.next("_rust_" + name) + sep() + ty.ty_to_str(t);
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

        auto lltyparams = llvm.LLVMGetParam(llfn, 3u);
        auto p = 0;
        for (ast.def_id d in typaram_defs) {
            auto llparam = bcx.build.GEP(lltyparams, vec(C_int(p)));
            llparam = bcx.build.Load(llparam);
            bcx.fcx.lltydescs.insert(d, llparam);
            p += 1;
        }

        auto llrawptr = llvm.LLVMGetParam(llfn, 4u);
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

                auto tydescptr =
                    cx.build.GEP(body,
                                 vec(C_int(0),
                                     C_int(abi.obj_body_elt_tydesc)));

                call_tydesc_glue_full(cx, body, cx.build.Load(tydescptr),
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

        case (ty.ty_fn(_,_,_)) {
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

// Tag information

fn type_of_variant(@crate_ctxt cx, &ast.variant v) -> TypeRef {
    let vec[TypeRef] lltys = vec();
    alt (ty.ann_to_type(v.ann).struct) {
        case (ty.ty_fn(_, ?args, _)) {
            for (ty.arg arg in args) {
                lltys += vec(type_of(cx, arg.ty));
            }
        }
        case (_) { fail; }
    }
    ret T_struct(lltys);
}

// Returns the type parameters of a tag.
fn tag_ty_params(@crate_ctxt cx, ast.def_id id) -> vec[ast.ty_param] {
    check (cx.items.contains_key(id));
    alt (cx.items.get(id).node) {
        case (ast.item_tag(_, _, ?tps, _)) { ret tps; }
    }
    fail;   // not reached
}

// Returns the variants in a tag.
fn tag_variants(@crate_ctxt cx, ast.def_id id) -> vec[ast.variant] {
    check (cx.items.contains_key(id));
    alt (cx.items.get(id).node) {
        case (ast.item_tag(_, ?variants, _, _)) { ret variants; }
    }
    fail;   // not reached
}

// Returns a new plain tag type of the given ID with no type parameters. Don't
// use this function in new code; it's a hack to keep things working for now.
fn mk_plain_tag(ast.def_id tid) -> @ty.t {
    let vec[@ty.t] tps = vec();
    ret ty.plain_ty(ty.ty_tag(tid, tps));
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
        auto tnil = plain_ty(ty.ty_nil);
        auto tbox = plain_ty(ty.ty_box(tnil));

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
        case (ty.ty_tag(?tid, ?tps)) {
            auto info = cx.fcx.ccx.tags.get(mk_plain_tag(tid));

            auto variants = tag_variants(cx.fcx.ccx, tid);
            auto n_variants = _vec.len[ast.variant](variants);

            auto lldiscrim_ptr = cx.build.GEP(v, vec(C_int(0), C_int(0)));
            auto llunion_ptr = cx.build.GEP(v, vec(C_int(0), C_int(1)));
            auto lldiscrim = cx.build.Load(lldiscrim_ptr);

            auto unr_cx = new_sub_block_ctxt(cx, "tag-iter-unr");
            unr_cx.build.Unreachable();

            auto llswitch = cx.build.Switch(lldiscrim, unr_cx.llbb,
                                            n_variants);

            auto next_cx = new_sub_block_ctxt(cx, "tag-iter-next");

            auto i = 0u;
            for (ast.variant variant in variants) {
                auto variant_cx = new_sub_block_ctxt(cx, "tag-iter-variant-" +
                                                     _uint.to_str(i, 10u));
                llvm.LLVMAddCase(llswitch, C_int(i as int), variant_cx.llbb);

                if (_vec.len[ast.variant_arg](variant.args) > 0u) {
                    // N-ary variant.
                    let vec[ValueRef] vals = vec(C_int(0), C_int(1),
                                                 C_int(i as int));
                    auto llvar = variant_cx.build.GEP(v, vals);
                    auto llvarty = type_of_variant(cx.fcx.ccx, variants.(i));

                    auto fn_ty = ty.ann_to_type(variants.(i).ann);
                    alt (fn_ty.struct) {
                        case (ty.ty_fn(_, ?args, _)) {
                            auto llvarp = variant_cx.build.
                                TruncOrBitCast(llunion_ptr, T_ptr(llvarty));

                            auto ty_params = tag_ty_params(cx.fcx.ccx, tid);

                            auto j = 0u;
                            for (ty.arg a in args) {
                                auto v = vec(C_int(0), C_int(j as int));
                                auto llfldp = variant_cx.build.GEP(llvarp, v);

                                auto ty_subst = ty.substitute_ty_params(
                                    ty_params, tps, a.ty);

                                auto llfld =
                                    load_scalar_or_boxed(variant_cx,
                                                         llfldp,
                                                         ty_subst);

                                auto res = f(variant_cx, llfld, ty_subst);
                                variant_cx = res.bcx;
                                j += 1u;
                            }
                        }
                        case (_) { fail; }
                    }

                    variant_cx.build.Br(next_cx.llbb);
                } else {
                    // Nullary variant; nothing to do.
                    variant_cx.build.Br(next_cx.llbb);
                }

                i += 1u;
            }

            ret res(next_cx, C_nil());
        }
        case (ty.ty_fn(_,_,_)) {
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
        auto bcx = cx;
        auto unit_sz = size_of(bcx, elt_ty);
        bcx = unit_sz.bcx;

        auto len = bcx.build.Load(lenptr);
        if (trailing_null) {
            len = bcx.build.Sub(len, unit_sz.val);
        }

        auto cond_cx = new_scope_block_ctxt(cx, "sequence-iter cond");
        auto body_cx = new_scope_block_ctxt(cx, "sequence-iter body");
        auto next_cx = new_sub_block_ctxt(cx, "next");

        bcx.build.Br(cond_cx.llbb);

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
        auto next_scaled_ix = body_res.bcx.build.Add(scaled_ix, unit_sz.val);

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
            auto et = plain_ty(ty.ty_machine(common.ty_u8));
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

    // FIXME: this adjustment has to do with the ridiculous encoding of
    // glue-pointer-constants in the tydesc records: They are tydesc-relative
    // displacements.  This is purely for compatibility with rustboot and
    // should go when it is discarded.
    llfn = cx.build.IntToPtr(cx.build.Add(cx.build.PtrToInt(llfn, T_int()),
                                          cx.build.PtrToInt(tydesc, T_int())),
                             val_ty(llfn));

    cx.build.FastCall(llfn, vec(C_null(T_ptr(T_nil())),
                                cx.fcx.lltaskptr,
                                C_null(T_ptr(T_nil())),
                                lltydescs,
                                llrawptr));
}

fn call_tydesc_glue(@block_ctxt cx, ValueRef v, @ty.t t, int field) {
    auto td = get_tydesc(cx, t);
    call_tydesc_glue_full(td.bcx, v, td.val, field);
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
        auto llsz = llszptr.bcx.build.Load(llszptr.val);
        ret call_memcpy(llszptr.bcx, dst, src, llsz);

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
        case (ast._mutable) {
            ret trans_expr(cx, e);
        }
    }
    fail;
}

// FIXME: implement proper structural comparison.

fn trans_compare(@block_ctxt cx, ast.binop op, @ty.t intype,
                 ValueRef lhs, ValueRef rhs) -> ValueRef {
    auto cmp = lib.llvm.LLVMIntEQ;
    alt (op) {
        case (ast.eq) { cmp = lib.llvm.LLVMIntEQ; }
        case (ast.ne) { cmp = lib.llvm.LLVMIntNE; }

        case (ast.lt) {
            if (ty.type_is_signed(intype)) {
                cmp = lib.llvm.LLVMIntSLT;
            } else {
                cmp = lib.llvm.LLVMIntULT;
            }
        }
        case (ast.le) {
            if (ty.type_is_signed(intype)) {
                cmp = lib.llvm.LLVMIntSLE;
            } else {
                cmp = lib.llvm.LLVMIntULE;
            }
        }
        case (ast.gt) {
            if (ty.type_is_signed(intype)) {
                cmp = lib.llvm.LLVMIntSGT;
            } else {
                cmp = lib.llvm.LLVMIntUGT;
            }
        }
        case (ast.ge) {
            if (ty.type_is_signed(intype)) {
                cmp = lib.llvm.LLVMIntSGE;
            } else {
                cmp = lib.llvm.LLVMIntUGE;
            }
        }
    }
    ret cx.build.ICmp(cmp, lhs, rhs);
}

fn trans_eager_binop(@block_ctxt cx, ast.binop op, @ty.t intype,
                     ValueRef lhs, ValueRef rhs) -> ValueRef {

    alt (op) {
        case (ast.add) { ret cx.build.Add(lhs, rhs); }
        case (ast.sub) { ret cx.build.Sub(lhs, rhs); }

        case (ast.mul) { ret cx.build.Mul(lhs, rhs); }
        case (ast.div) {
            if (ty.type_is_signed(intype)) {
                ret cx.build.SDiv(lhs, rhs);
            } else {
                ret cx.build.UDiv(lhs, rhs);
            }
        }
        case (ast.rem) {
            if (ty.type_is_signed(intype)) {
                ret cx.build.SRem(lhs, rhs);
            } else {
                ret cx.build.URem(lhs, rhs);
            }
        }

        case (ast.bitor) { ret cx.build.Or(lhs, rhs); }
        case (ast.bitand) { ret cx.build.And(lhs, rhs); }
        case (ast.bitxor) { ret cx.build.Xor(lhs, rhs); }
        case (ast.lsl) { ret cx.build.Shl(lhs, rhs); }
        case (ast.lsr) { ret cx.build.LShr(lhs, rhs); }
        case (ast.asr) { ret cx.build.AShr(lhs, rhs); }
        case (_) {
            ret trans_compare(cx, op, intype, lhs, rhs);
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
            auto lhty = ty.expr_ty(a);
            lhs = autoderef(lhs.bcx, lhs.val, lhty);
            auto rhs = trans_expr(lhs.bcx, b);
            auto rhty = ty.expr_ty(b);
            rhs = autoderef(rhs.bcx, rhs.val, rhty);
            ret res(rhs.bcx, trans_eager_binop(rhs.bcx, op, lhty,
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

fn trans_if(@block_ctxt cx, @ast.expr cond, &ast.block thn,
            &vec[tup(@ast.expr, ast.block)] elifs,
            &option.t[ast.block] els) -> result {

    auto cond_res = trans_expr(cx, cond);

    auto then_cx = new_scope_block_ctxt(cx, "then");
    auto then_res = trans_block(then_cx, thn);

    auto else_cx = new_scope_block_ctxt(cx, "else");
    auto else_res = res(else_cx, C_nil());

    auto num_elifs = _vec.len[tup(@ast.expr, ast.block)](elifs);
    if (num_elifs > 0u) {
        auto next_elif = elifs.(0u);
        auto next_elifthn = next_elif._0;
        auto next_elifcnd = next_elif._1;
        auto rest_elifs = _vec.shift[tup(@ast.expr, ast.block)](elifs);
        else_res = trans_if(else_cx, next_elifthn, next_elifcnd,
                            rest_elifs, els);
    }

    /* else: FIXME: rustboot has a problem here
       with preconditions inside an else block */
    if (num_elifs == 0u)  {
        alt (els) {
            case (some[ast.block](?eblk)) {
                else_res = trans_block(else_cx, eblk);
            }
            case (_) { /* fall through */ }
        }
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
        scope_cx.cleanups += clean(bind drop_slot(_, local_res.val, t));
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

fn trans_for_each(@block_ctxt cx,
                  @ast.decl decl,
                  @ast.expr seq,
                  &ast.block body) -> result {

    /*
     * The translation is a little .. complex here. Code like:
     *
     *    let ty1 p = ...;
     *
     *    let ty1 q = ...;
     *
     *    foreach (ty v in foo(a,b)) { body(p,q,v) }
     *
     *
     * Turns into a something like so (C/Rust mishmash):
     *
     *    type env = { *ty1 p, *ty2 q, ... };
     *
     *    let env e = { &p, &q, ... };
     *
     *    fn foreach123_body(env* e, ty v) { body(*(e->p),*(e->q),v) }
     *
     *    foo([foreach123_body, env*], a, b);
     *
     */

    // Step 1: walk body and figure out which references it makes
    // escape. This could be determined upstream, and probably ought
    // to be so, eventualy. For first cut, skip this. Null env.

    auto env_ty = T_opaque_closure_ptr(cx.fcx.ccx.tn);


    // Step 2: Declare foreach body function.

    // FIXME: possibly support alias-mode here?
    auto decl_ty = plain_ty(ty.ty_nil);
    alt (decl.node) {
        case (ast.decl_local(?local)) {
            decl_ty = node_ann_type(cx.fcx.ccx, local.ann);
        }
    }

    let str s =
        cx.fcx.ccx.names.next("_rust_foreach")
        + sep() + cx.fcx.ccx.path;

    // The 'env' arg entering the body function is a fake env member (as in
    // the env-part of the normal rust calling convention) that actually
    // points to a stack allocated env in this frame. We bundle that env
    // pointer along with the foreach-body-fn pointer into a 'normal' fn pair
    // and pass it in as a first class fn-arg to the iterator.

    auto iter_body_llty = type_of_fn_full(cx.fcx.ccx, ast.proto_fn,
                                          none[TypeRef],
                                          vec(rec(mode=ast.val, ty=decl_ty)),
                                          plain_ty(ty.ty_nil));

    let ValueRef lliterbody = decl_fastcall_fn(cx.fcx.ccx.llmod,
                                               s, iter_body_llty);

    // FIXME: handle ty params properly.
    let vec[ast.ty_param] ty_params = vec();

    auto fcx = new_fn_ctxt(cx.fcx.ccx, s, lliterbody);
    auto bcx = new_top_block_ctxt(fcx);

    // FIXME: populate lllocals from llenv here.
    auto res = trans_block(bcx, body);
    res.bcx.build.RetVoid();


    // Step 3: Call iter passing [lliterbody, llenv], plus other args.

    alt (seq.node) {

        case (ast.expr_call(?f, ?args, ?ann)) {

            auto pair = cx.build.Alloca(T_fn_pair(cx.fcx.ccx.tn,
                                                  iter_body_llty));
            auto code_cell = cx.build.GEP(pair,
                                          vec(C_int(0),
                                              C_int(abi.fn_field_code)));
            cx.build.Store(lliterbody, code_cell);

            // log "lliterbody: " + val_str(cx.fcx.ccx.tn, lliterbody);
            ret trans_call(cx, f,
                           some[ValueRef](cx.build.Load(pair)),
                           args,
                           ann);
        }
    }
    fail;
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

        case (ast.pat_lit(?lt, ?ann)) {
            auto lllit = trans_lit(cx.fcx.ccx, *lt, ann);
            auto lltype = ty.ann_to_type(ann);
            auto lleq = trans_compare(cx, ast.eq, lltype, llval, lllit);

            auto matched_cx = new_sub_block_ctxt(cx, "matched_cx");
            cx.build.CondBr(lleq, matched_cx.llbb, next_cx.llbb);
            ret res(matched_cx, llval);
        }

        case (ast.pat_tag(?id, ?subpats, ?vdef_opt, ?ann)) {
            auto lltagptr = cx.build.GEP(llval, vec(C_int(0), C_int(0)));
            auto lltag = cx.build.Load(lltagptr);

            auto vdef = option.get[ast.variant_def](vdef_opt);
            auto variant_id = vdef._1;
            auto variant_tag = 0;

            auto variants = tag_variants(cx.fcx.ccx, vdef._0);
            auto i = 0;
            for (ast.variant v in variants) {
                auto this_variant_id = v.id;
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
        case (ast.pat_lit(_, _)) { ret res(cx, llval); }
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

fn lval_generic_fn(@block_ctxt cx,
                   ty.ty_params_and_ty tpt,
                   ast.def_id fn_id,
                   &ast.ann ann)
    -> lval_result {

    check (cx.fcx.ccx.fn_pairs.contains_key(fn_id));
    auto lv = lval_val(cx, cx.fcx.ccx.fn_pairs.get(fn_id));
    auto monoty = node_ann_type(cx.fcx.ccx, ann);
    auto tys = ty.resolve_ty_params(tpt, monoty);

    if (_vec.len[@ty.t](tys) != 0u) {
        auto bcx = cx;
        let vec[ValueRef] tydescs = vec();
        for (@ty.t t in tys) {
            auto td = get_tydesc(bcx, t);
            bcx = td.bcx;
            append[ValueRef](tydescs, td.val);
        }
        auto gen = rec( item_type = tpt._1,
                        tydescs = tydescs );
        lv = rec(res = res(bcx, lv.res.val),
                 generic = some[generic_info](gen)
                 with lv);
    }
    ret lv;
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
                    check (cx.fcx.ccx.items.contains_key(did));
                    auto fn_item = cx.fcx.ccx.items.get(did);
                    ret lval_generic_fn(cx, ty.item_ty(fn_item), did, ann);
                }
                case (ast.def_obj(?did)) {
                    check (cx.fcx.ccx.items.contains_key(did));
                    auto fn_item = cx.fcx.ccx.items.get(did);
                    ret lval_generic_fn(cx, ty.item_ty(fn_item), did, ann);
                }
                case (ast.def_variant(?tid, ?vid)) {
                    if (cx.fcx.ccx.fn_pairs.contains_key(vid)) {
                        check (cx.fcx.ccx.items.contains_key(tid));
                        auto tag_item = cx.fcx.ccx.items.get(tid);
                        auto params = ty.item_ty(tag_item)._0;
                        auto fty = plain_ty(ty.ty_nil);
                        alt (tag_item.node) {
                            case (ast.item_tag(_, ?variants, _, _)) {
                                for (ast.variant v in variants) {
                                    if (v.id == vid) {
                                        fty = node_ann_type(cx.fcx.ccx,
                                                            v.ann);
                                    }
                                }
                            }
                        }
                        ret lval_generic_fn(cx, tup(params, fty), vid, ann);
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
                case (ast.def_native_fn(?did)) {
                    check (cx.fcx.ccx.native_items.contains_key(did));
                    auto fn_item = cx.fcx.ccx.native_items.get(did);
                    ret lval_generic_fn(cx, ty.native_item_ty(fn_item),
                                        did, ann);
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
    auto r = trans_expr(cx, base);
    r = autoderef(r.bcx, r.val, ty.expr_ty(base));
    auto t = ty.expr_ty(base);
    alt (t.struct) {
        case (ty.ty_tup(?fields)) {
            let uint ix = ty.field_num(cx.fcx.ccx.sess, sp, field);
            auto v = GEP_tup_like(r.bcx, t, r.val, vec(0, ix as int));
            ret lval_mem(v.bcx, v.val);
        }
        case (ty.ty_rec(?fields)) {
            let uint ix = ty.field_idx(cx.fcx.ccx.sess, sp, field, fields);
            auto v = GEP_tup_like(r.bcx, t, r.val, vec(0, ix as int));
            ret lval_mem(v.bcx, v.val);
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
    auto bcx = ix.bcx;

    auto llunit_ty = node_type(cx.fcx.ccx, ann);
    auto unit_sz = size_of(bcx, node_ann_type(cx.fcx.ccx, ann));
    bcx = unit_sz.bcx;
    auto scaled_ix = bcx.build.Mul(ix.val, unit_sz.val);

    auto lim = bcx.build.GEP(v, vec(C_int(0), C_int(abi.vec_elt_fill)));
    lim = bcx.build.Load(lim);

    auto bounds_check = bcx.build.ICmp(lib.llvm.LLVMIntULT,
                                       scaled_ix, lim);

    auto fail_cx = new_sub_block_ctxt(bcx, "fail");
    auto next_cx = new_sub_block_ctxt(bcx, "next");
    bcx.build.CondBr(bounds_check, next_cx.llbb, fail_cx.llbb);

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

fn trans_bind_thunk(@crate_ctxt cx,
                    @ty.t incoming_fty,
                    @ty.t outgoing_fty,
                    vec[option.t[@ast.expr]] args,
                    TypeRef llclosure_ty,
                    vec[@ty.t] bound_tys,
                    uint ty_param_count) -> ValueRef {
    // Construct a thunk-call with signature incoming_fty, and that copies
    // args forward into a call to outgoing_fty.

    let str s = cx.names.next("_rust_thunk") + sep() + cx.path;
    let TypeRef llthunk_ty = get_pair_fn_ty(type_of(cx, incoming_fty));
    let ValueRef llthunk = decl_fastcall_fn(cx.llmod, s, llthunk_ty);

    auto fcx = new_fn_ctxt(cx, s, llthunk);
    auto bcx = new_top_block_ctxt(fcx);

    auto llclosure = bcx.build.PointerCast(fcx.llenv, llclosure_ty);

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

    auto outgoing_ret_ty = ty.ty_fn_ret(outgoing_fty);
    auto outgoing_arg_tys = ty.ty_fn_args(outgoing_fty);

    auto llretptr = fcx.llretptr;
    if (ty.type_has_dynamic_size(outgoing_ret_ty)) {
        llretptr = bcx.build.PointerCast(llretptr, T_typaram_ptr(cx.tn));
    }

    let vec[ValueRef] llargs = vec(llretptr,
                                   fcx.lltaskptr,
                                   lltargetclosure);

    // Copy in the type parameters.
    let uint i = 0u;
    while (i < ty_param_count) {
        auto lltyparam_ptr =
            bcx.build.GEP(llbody, vec(C_int(0),
                                      C_int(abi.closure_elt_ty_params),
                                      C_int(i as int)));
        llargs += vec(bcx.build.Load(lltyparam_ptr));
        i += 1u;
    }

    let uint a = 2u + i;    // retptr, task ptr, env come first
    let int b = 0;
    let uint outgoing_arg_index = 0u;
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
                if (ty.type_has_dynamic_size(outgoing_arg_tys.
                        (outgoing_arg_index).ty)) {
                    // Cast to a generic typaram pointer in order to make a
                    // type-compatible call.
                    passed_arg = bcx.build.PointerCast(passed_arg,
                                                       T_typaram_ptr(cx.tn));
                }
                llargs += passed_arg;
                a += 1u;
            }
        }

        outgoing_arg_index += 0u;
    }

    // FIXME: turn this call + ret into a tail call.
    auto lltargetfn = bcx.build.GEP(lltarget,
                                    vec(C_int(0),
                                        C_int(abi.fn_field_code)));
    lltargetfn = bcx.build.Load(lltargetfn);

    auto r = bcx.build.FastCall(lltargetfn, llargs);
    bcx.build.RetVoid();

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

        // Figure out which tydescs we need to pass, if any.
        // FIXME: typestate botch
        let @ty.t outgoing_fty = ty.plain_ty(ty.ty_nil);
        let vec[ValueRef] lltydescs = vec();
        alt (f_res.generic) {
            case (none[generic_info]) {
                outgoing_fty = ty.expr_ty(f);
            }
            case (some[generic_info](?ginfo)) {
                outgoing_fty = ginfo.item_type;
                lltydescs = ginfo.tydescs;
            }
        }
        auto ty_param_count = _vec.len[ValueRef](lltydescs);

        if (_vec.len[@ast.expr](bound) == 0u && ty_param_count == 0u) {
            // Trivial 'binding': just return the static pair-ptr.
            ret f_res.res;
        } else {
            auto bcx = f_res.res.bcx;
            auto pair_t = node_type(cx.fcx.ccx, ann);
            auto pair_v = bcx.build.Alloca(pair_t);

            // Translate the bound expressions.
            let vec[@ty.t] bound_tys = vec();
            let vec[ValueRef] bound_vals = vec();
            auto i = 0u;
            for (@ast.expr e in bound) {
                auto arg = trans_expr(bcx, e);
                bcx = arg.bcx;

                append[ValueRef](bound_vals, arg.val);
                append[@ty.t](bound_tys, ty.expr_ty(e));

                i += 1u;
            }

            // Get the type of the bound function.
            let TypeRef lltarget_ty = type_of(bcx.fcx.ccx, outgoing_fty);

            // Synthesize a closure type.
            let @ty.t bindings_ty = plain_ty(ty.ty_tup(bound_tys));
            let TypeRef llbindings_ty = type_of(bcx.fcx.ccx, bindings_ty);
            let TypeRef llclosure_ty = T_closure_ptr(cx.fcx.ccx.tn,
                                                     lltarget_ty,
                                                     llbindings_ty,
                                                     ty_param_count);

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
            bcx = bindings_tydesc.bcx;
            bcx.build.Store(bindings_tydesc.val, bound_tydesc);

            // Store thunk-target.
            auto bound_target =
                bcx.build.GEP(closure,
                              vec(C_int(0),
                                  C_int(abi.closure_elt_target)));
            auto src = bcx.build.Load(f_res.res.val);
            bcx.build.Store(src, bound_target);

            // Copy expr values into boxed bindings.
            i = 0u;
            auto bindings =
                bcx.build.GEP(closure,
                              vec(C_int(0),
                                  C_int(abi.closure_elt_bindings)));
            for (ValueRef v in bound_vals) {
                auto bound = bcx.build.GEP(bindings,
                                           vec(C_int(0), C_int(i as int)));
                bcx = copy_ty(r.bcx, INIT, bound, v, bound_tys.(i)).bcx;
                i += 1u;
            }

            // If necessary, copy tydescs describing type parameters into the
            // appropriate slot in the closure.
            alt (f_res.generic) {
                case (none[generic_info]) { /* nothing to do */ }
                case (some[generic_info](?ginfo)) {
                    auto ty_params_slot =
                        bcx.build.GEP(closure,
                                      vec(C_int(0),
                                          C_int(abi.closure_elt_ty_params)));
                    auto i = 0;
                    for (ValueRef td in ginfo.tydescs) {
                        auto ty_param_slot = bcx.build.GEP(ty_params_slot,
                                                           vec(C_int(0),
                                                               C_int(i)));
                        bcx.build.Store(td, ty_param_slot);
                        i += 1;
                    }
                }
            }

            // Make thunk and store thunk-ptr in outer pair's code slot.
            auto pair_code = bcx.build.GEP(pair_v,
                                           vec(C_int(0),
                                               C_int(abi.fn_field_code)));

            let @ty.t pair_ty = node_ann_type(cx.fcx.ccx, ann);
            let ValueRef llthunk =
                trans_bind_thunk(cx.fcx.ccx, pair_ty, outgoing_fty,
                                 args, llclosure_ty, bound_tys,
                                 ty_param_count);

            bcx.build.Store(llthunk, pair_code);

            // Store box ptr in outer pair's box slot.
            auto pair_box = bcx.build.GEP(pair_v,
                                          vec(C_int(0),
                                              C_int(abi.fn_field_box)));
            bcx.build.Store
                (bcx.build.PointerCast
                 (box,
                  T_opaque_closure_ptr(bcx.fcx.ccx.tn)),
                 pair_box);

            find_scope_cx(cx).cleanups +=
                clean(bind drop_slot(_, pair_v, pair_ty));

            ret res(bcx, pair_v);
        }
    }
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn_full
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args

fn trans_args(@block_ctxt cx,
              ValueRef llenv,
              option.t[ValueRef] llobj,
              option.t[generic_info] gen,
              option.t[ValueRef] lliterbody,
              &vec[@ast.expr] es,
              @ty.t fn_ty)
    -> tup(@block_ctxt, vec[ValueRef], ValueRef) {

    let vec[ty.arg] args = ty.ty_fn_args(fn_ty);
    let vec[ValueRef] llargs = vec();
    let vec[ValueRef] lltydescs = vec();
    let @block_ctxt bcx = cx;


    // Arg 0: Output pointer.
    auto retty = ty.ty_fn_ret(fn_ty);
    auto llretslot_res = alloc_ty(bcx, retty);
    bcx = llretslot_res.bcx;
    auto llretslot = llretslot_res.val;

    alt (gen) {
        case (some[generic_info](?g)) {
            lltydescs = g.tydescs;
            args = ty.ty_fn_args(g.item_type);
            retty = ty.ty_fn_ret(g.item_type);
        }
        case (_) {
        }
    }
    if (ty.type_has_dynamic_size(retty)) {
        llargs += bcx.build.PointerCast(llretslot,
                                        T_typaram_ptr(cx.fcx.ccx.tn));
    } else if (ty.count_ty_params(retty) != 0u) {
        // It's possible that the callee has some generic-ness somewhere in
        // its return value -- say a method signature within an obj or a fn
        // type deep in a structure -- which the caller has a concrete view
        // of. If so, cast the caller's view of the restlot to the callee's
        // view, for the sake of making a type-compatible call.
        llargs += cx.build.PointerCast(llretslot,
                                       T_ptr(type_of(bcx.fcx.ccx, retty)));
    } else {
        llargs += llretslot;
    }


    // Arg 1: Task pointer.
    llargs += bcx.fcx.lltaskptr;

    // Arg 2: Env (closure-bindings / self-obj)
    alt (llobj) {
        case (some[ValueRef](?ob)) {
            // Every object is always found in memory,
            // and not-yet-loaded (as part of an lval x.y
            // doted method-call).
            llargs += bcx.build.Load(ob);
        }
        case (_) {
            llargs += llenv;
        }
    }

    // Args >3: ty_params ...
    llargs += lltydescs;

    // ... then possibly an lliterbody argument.
    alt (lliterbody) {
        case (none[ValueRef]) {}
        case (some[ValueRef](?lli)) {
            llargs += lli;
        }
    }

    // ... then explicit args.

    // First we figure out the caller's view of the types of the arguments.
    // This will be needed if this is a generic call, because the callee has
    // to cast her view of the arguments to the caller's view.
    auto arg_tys = type_of_explicit_args(cx.fcx.ccx, args);

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

        if (ty.count_ty_params(args.(i).ty) > 0u) {
            auto lldestty = arg_tys.(i);
            val = bcx.build.PointerCast(val, lldestty);
        }

        llargs += val;
        i += 1u;
    }

    ret tup(bcx, llargs, llretslot);
}

fn trans_call(@block_ctxt cx, @ast.expr f,
              option.t[ValueRef] lliterbody,
              vec[@ast.expr] args,
              &ast.ann ann) -> result {
    auto f_res = trans_lval(cx, f);
    auto faddr = f_res.res.val;
    auto llenv = C_null(T_opaque_closure_ptr(cx.fcx.ccx.tn));

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

            auto llclosure = bcx.build.GEP(pair,
                                           vec(C_int(0),
                                               C_int(abi.fn_field_box)));
            llenv = bcx.build.Load(llclosure);
        }
    }
    auto fn_ty = ty.expr_ty(f);
    auto ret_ty = ty.ann_to_type(ann);
    auto args_res = trans_args(f_res.res.bcx,
                               llenv, f_res.llobj,
                               f_res.generic,
                               lliterbody,
                               args, fn_ty);

    auto bcx = args_res._0;
    auto llargs = args_res._1;
    auto llretslot = args_res._2;

    /*
    log "calling: " + val_str(cx.fcx.ccx.tn, faddr);

    for (ValueRef arg in llargs) {
        log "arg: " + val_str(cx.fcx.ccx.tn, arg);
    }
    */

    bcx.build.FastCall(faddr, llargs);
    auto retval = C_nil();

    if (!ty.type_is_nil(ret_ty)) {
        retval = load_scalar_or_boxed(bcx, llretslot, ret_ty);
        // Retval doesn't correspond to anything really tangible in the frame,
        // but it's a ref all the same, so we put a note here to drop it when
        // we're done in this scope.
        find_scope_cx(cx).cleanups += clean(bind drop_ty(_, retval, ret_ty));
    }

    ret res(bcx, retval);
}

fn trans_tup(@block_ctxt cx, vec[ast.elt] elts,
             &ast.ann ann) -> result {
    auto bcx = cx;
    auto t = node_ann_type(bcx.fcx.ccx, ann);
    auto llty = type_of(bcx.fcx.ccx, t);
    auto tup_res = alloc_ty(bcx, t);
    auto tup_val = tup_res.val;
    bcx = tup_res.bcx;

    find_scope_cx(cx).cleanups += clean(bind drop_ty(_, tup_val, t));
    let int i = 0;

    for (ast.elt e in elts) {
        auto e_ty = ty.expr_ty(e.expr);
        auto src_res = trans_expr(bcx, e.expr);
        bcx = src_res.bcx;
        auto dst_res = GEP_tup_like(bcx, t, tup_val, vec(0, i));
        bcx = dst_res.bcx;
        bcx = copy_ty(src_res.bcx, INIT, dst_res.val, src_res.val, e_ty).bcx;
        i += 1;
    }
    ret res(bcx, tup_val);
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
    auto bcx = cx;
    auto unit_sz = size_of(bcx, unit_ty);
    bcx = unit_sz.bcx;
    auto data_sz = llvm.LLVMConstMul(C_int(_vec.len[@ast.expr](args) as int),
                                     unit_sz.val);

    // FIXME: pass tydesc properly.
    auto sub = trans_upcall(bcx, "upcall_new_vec", vec(data_sz, C_int(0)));
    bcx = sub.bcx;

    auto llty = type_of(bcx.fcx.ccx, t);
    auto vec_val = bcx.build.IntToPtr(sub.val, llty);
    find_scope_cx(bcx).cleanups += clean(bind drop_ty(_, vec_val, t));

    auto body = bcx.build.GEP(vec_val, vec(C_int(0),
                                           C_int(abi.vec_elt_data)));

    auto pseudo_tup_ty =
        plain_ty(ty.ty_tup(_vec.init_elt[@ty.t](unit_ty,
                                                _vec.len[@ast.expr](args))));
    let int i = 0;

    for (@ast.expr e in args) {
        auto src_res = trans_expr(bcx, e);
        bcx = src_res.bcx;
        auto dst_res = GEP_tup_like(bcx, pseudo_tup_ty, body, vec(0, i));
        bcx = dst_res.bcx;
        bcx = copy_ty(bcx, INIT, dst_res.val, src_res.val, unit_ty).bcx;
        i += 1;
    }
    auto fill = bcx.build.GEP(vec_val,
                              vec(C_int(0), C_int(abi.vec_elt_fill)));
    bcx.build.Store(data_sz, fill);

    ret res(bcx, vec_val);
}

fn trans_rec(@block_ctxt cx, vec[ast.field] fields,
             option.t[@ast.expr] base, &ast.ann ann) -> result {

    auto bcx = cx;
    auto t = node_ann_type(bcx.fcx.ccx, ann);
    auto llty = type_of(bcx.fcx.ccx, t);
    auto rec_res = alloc_ty(bcx, t);
    auto rec_val = rec_res.val;
    bcx = rec_res.bcx;

    find_scope_cx(cx).cleanups += clean(bind drop_ty(_, rec_val, t));
    let int i = 0;

    auto base_val = C_nil();

    alt (base) {
        case (none[@ast.expr]) { }
        case (some[@ast.expr](?bexp)) {
            auto base_res = trans_expr(bcx, bexp);
            bcx = base_res.bcx;
            base_val = base_res.val;
        }
    }

    let vec[ty.field] ty_fields = vec();
    alt (t.struct) {
        case (ty.ty_rec(?flds)) { ty_fields = flds; }
    }

    for (ty.field tf in ty_fields) {
        auto e_ty = tf.ty;
        auto dst_res = GEP_tup_like(bcx, t, rec_val, vec(0, i));
        bcx = dst_res.bcx;

        auto expr_provided = false;
        auto src_res = res(bcx, C_nil());

        for (ast.field f in fields) {
            if (_str.eq(f.ident, tf.ident)) {
                expr_provided = true;
                src_res = trans_expr(bcx, f.expr);
            }
        }
        if (!expr_provided) {
            src_res = GEP_tup_like(bcx, t, base_val, vec(0, i));
            src_res = res(src_res.bcx,
                          load_scalar_or_boxed(bcx, src_res.val, e_ty));
        }

        bcx = src_res.bcx;
        bcx = copy_ty(bcx, INIT, dst_res.val, src_res.val, e_ty).bcx;
        i += 1;
    }
    ret res(bcx, rec_val);
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

        case (ast.expr_if(?cond, ?thn, ?elifs, ?els, _)) {
            ret trans_if(cx, cond, thn, elifs, els);
        }

        case (ast.expr_for(?decl, ?seq, ?body, _)) {
            ret trans_for(cx, decl, seq, body);
        }

        case (ast.expr_for_each(?decl, ?seq, ?body, _)) {
            ret trans_for_each(cx, decl, seq, body);
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
            auto v = trans_eager_binop(rhs_res.bcx, op, t,
                                       lhs_val, rhs_res.val);
            // FIXME: calculate copy init-ness in typestate.
            ret copy_ty(rhs_res.bcx, DROP_EXISTING,
                        lhs_res.res.val, v, t);
        }

        case (ast.expr_bind(?f, ?args, ?ann)) {
            ret trans_bind(cx, f, args, ann);
        }

        case (ast.expr_call(?f, ?args, ?ann)) {
            ret trans_call(cx, f, none[ValueRef], args, ann);
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

        case (ast.expr_rec(?args, ?base, ?ann)) {
            ret trans_rec(cx, args, base, ann);
        }

        case (ast.expr_fail) {
            ret trans_fail(cx, e.span, "explicit failure");
        }

        case (ast.expr_log(?a)) {
            ret trans_log(cx, a);
        }

        case (ast.expr_check_expr(?a)) {
            ret trans_check_expr(cx, a);
        }

        case (ast.expr_ret(?e)) {
            ret trans_ret(cx, e);
        }

        case (ast.expr_put(?e)) {
            ret trans_put(cx, e);
        }

        case (ast.expr_be(?e)) {
            ret trans_be(cx, e);
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

fn trans_put(@block_ctxt cx, &option.t[@ast.expr] e) -> result {
    auto llcallee = C_nil();
    auto llenv = C_nil();

    alt (cx.fcx.lliterbody) {
        case (some[ValueRef](?lli)) {
            auto slot = cx.build.Alloca(val_ty(lli));
            cx.build.Store(lli, slot);

            llcallee = cx.build.GEP(slot, vec(C_int(0),
                                              C_int(abi.fn_field_code)));
            llcallee = cx.build.Load(llcallee);

            llenv = cx.build.GEP(slot, vec(C_int(0),
                                           C_int(abi.fn_field_box)));
            llenv = cx.build.Load(llenv);
        }
    }
    auto bcx = cx;
    auto dummy_retslot = bcx.build.Alloca(T_nil());
    let vec[ValueRef] llargs = vec(dummy_retslot, cx.fcx.lltaskptr, llenv);
    alt (e) {
        case (none[@ast.expr]) { }
        case (some[@ast.expr](?x)) {
            auto r = trans_expr(bcx, x);
            llargs += r.val;
            bcx = r.bcx;
        }
    }
    ret res(bcx, bcx.build.FastCall(llcallee, llargs));
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
            bcx = copy_ty(bcx, INIT, cx.fcx.llretptr, val, t).bcx;
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

    bcx.build.RetVoid();
    ret res(bcx, C_nil());
}

fn trans_be(@block_ctxt cx, @ast.expr e) -> result {
    // FIXME: This should be a typestate precondition
    check (ast.is_call_expr(e));
    // FIXME: Turn this into a real tail call once
    // calling convention issues are settled
    ret trans_ret(cx, some(e));
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
                auto llsz = size_of(bcx, ty);
                bcx = call_bzero(llsz.bcx, llptr, llsz.val).bcx;

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
    auto cx = new_block_ctxt(fcx, parent_none, SCOPE_BLOCK,
                             "function top level");

    // FIXME: hack to give us some spill room to make up for an LLVM
    // bug where it destroys its own callee-saves.
    cx.build.Alloca(T_array(T_int(), 10u));
    ret cx;
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

    auto i = _vec.len[cleanup](cleanup_cx.cleanups);
    while (i > 0u) {
        i -= 1u;
        auto c = cleanup_cx.cleanups.(i);
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

fn alloc_ty(@block_ctxt cx, @ty.t t) -> result {
    auto val = C_int(0);
    auto bcx = cx;
    if (ty.type_has_dynamic_size(t)) {
        auto n = size_of(bcx, t);
        bcx = n.bcx;
        val = bcx.build.ArrayAlloca(T_i8(), n.val);
    } else {
        val = bcx.build.Alloca(type_of(cx.fcx.ccx, t));
    }
    ret res(bcx, val);
}

fn alloc_local(@block_ctxt cx, @ast.local local) -> result {
    auto t = node_ann_type(cx.fcx.ccx, local.ann);
    auto r = alloc_ty(cx, t);
    r.bcx.fcx.lllocals.insert(local.id, r.val);
    ret r;
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

// NB: must keep 4 fns in sync:
//
//  - type_of_fn_full
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args

fn new_fn_ctxt(@crate_ctxt cx,
               str name,
               ValueRef llfndecl) -> @fn_ctxt {

    let ValueRef llretptr = llvm.LLVMGetParam(llfndecl, 0u);
    let ValueRef lltaskptr = llvm.LLVMGetParam(llfndecl, 1u);
    let ValueRef llenv = llvm.LLVMGetParam(llfndecl, 2u);

    let hashmap[ast.def_id, ValueRef] llargs = new_def_hash[ValueRef]();
    let hashmap[ast.def_id, ValueRef] llobjfields = new_def_hash[ValueRef]();
    let hashmap[ast.def_id, ValueRef] lllocals = new_def_hash[ValueRef]();
    let hashmap[ast.def_id, ValueRef] lltydescs = new_def_hash[ValueRef]();

    ret @rec(llfn=llfndecl,
             lltaskptr=lltaskptr,
             llenv=llenv,
             llretptr=llretptr,
             mutable llself=none[ValueRef],
             mutable lliterbody=none[ValueRef],
             llargs=llargs,
             llobjfields=llobjfields,
             lllocals=lllocals,
             lltydescs=lltydescs,
             ccx=cx);
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn_full
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args

fn create_llargs_for_fn_args(&@fn_ctxt cx,
                             ast.proto proto,
                             option.t[TypeRef] ty_self,
                             @ty.t ret_ty,
                             &vec[ast.arg] args,
                             &vec[ast.ty_param] ty_params) {

    alt (ty_self) {
        case (some[TypeRef](_)) {
            cx.llself = some[ValueRef](cx.llenv);
        }
        case (_) {
        }
    }

    auto arg_n = 3u;

    if (ty_self == none[TypeRef]) {
        for (ast.ty_param tp in ty_params) {
            auto llarg = llvm.LLVMGetParam(cx.llfn, arg_n);
            check (llarg as int != 0);
            cx.lltydescs.insert(tp.id, llarg);
            arg_n += 1u;
        }
    }

    if (proto == ast.proto_iter) {
        auto llarg = llvm.LLVMGetParam(cx.llfn, arg_n);
        check (llarg as int != 0);
        cx.lliterbody = some[ValueRef](llarg);
        arg_n += 1u;
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
        case (ty.ty_fn(_, ?arg_tys, _)) {
            ret arg_tys;
        }
    }
    fail;
}

fn ret_ty_of_fn_ty(@ty.t t) -> @ty.t {
    alt (t.struct) {
        case (ty.ty_fn(_, _, ?ret_ty)) {
            ret ret_ty;
        }
    }
    fail;
}


fn ret_ty_of_fn(ast.ann ann) -> @ty.t {
    ret ret_ty_of_fn_ty(ty.ann_to_type(ann));
}

fn populate_fn_ctxt_from_llself(@block_ctxt cx, ValueRef llself) {

    let vec[TypeRef] llfield_tys = vec();

    for (ast.obj_field f in cx.fcx.ccx.obj_fields) {
        llfield_tys += node_type(cx.fcx.ccx, f.ann);
    }

    auto n_typarams = _vec.len[ast.ty_param](cx.fcx.ccx.obj_typarams);
    let TypeRef llobj_box_ty = T_obj_ptr(cx.fcx.ccx.tn, n_typarams,
                                         T_struct(llfield_tys));

    auto box_cell =
        cx.build.GEP(llself,
                     vec(C_int(0),
                         C_int(abi.obj_field_box)));

    auto box_ptr = cx.build.Load(box_cell);

    box_ptr = cx.build.PointerCast(box_ptr, llobj_box_ty);

    auto obj_typarams = cx.build.GEP(box_ptr,
                                     vec(C_int(0),
                                         C_int(abi.box_rc_field_body),
                                         C_int(abi.obj_body_elt_typarams)));

    auto obj_fields = cx.build.GEP(box_ptr,
                                   vec(C_int(0),
                                       C_int(abi.box_rc_field_body),
                                       C_int(abi.obj_body_elt_fields)));

    let int i = 0;

    for (ast.ty_param p in cx.fcx.ccx.obj_typarams) {
        let ValueRef lltyparam = cx.build.GEP(obj_typarams,
                                              vec(C_int(0),
                                                  C_int(i)));
        lltyparam = cx.build.Load(lltyparam);
        cx.fcx.lltydescs.insert(p.id, lltyparam);
        i += 1;
    }

    i = 0;
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
    create_llargs_for_fn_args(fcx, f.proto,
                              ty_self, ret_ty_of_fn(ann),
                              f.decl.inputs, ty_params);
    auto bcx = new_top_block_ctxt(fcx);

    copy_args_to_allocas(bcx, ty_self, f.decl.inputs,
                         arg_tys_of_fn(ann));

    alt (fcx.llself) {
        case (some[ValueRef](?llself)) {
            populate_fn_ctxt_from_llself(bcx, llself);
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
            case (ty.ty_fn(?proto, ?inputs, ?output)) {
                llfnty = type_of_fn_full(cx, proto,
                                         some[TypeRef](self_ty),
                                         inputs, output);
            }
        }

        let @crate_ctxt mcx = @rec(path=cx.path + sep() + m.node.ident
                                   with *cx);

        let str s = cx.names.next("_rust_method") + sep() + mcx.path;
        let ValueRef llfn = decl_fastcall_fn(cx.llmod, s, llfnty);
        cx.item_ids.insert(m.node.id, llfn);

        trans_fn(mcx, m.node.meth, m.node.id, some[TypeRef](self_ty),
                 ty_params, m.node.ann);
        methods += llfn;
    }
    auto vtbl = C_struct(methods);
    auto gvar = llvm.LLVMAddGlobal(cx.llmod,
                                   val_ty(vtbl),
                                   _str.buf("_rust_vtbl" + sep() + cx.path));
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
    create_llargs_for_fn_args(fcx, ast.proto_fn,
                              none[TypeRef], ret_ty_of_fn(ann),
                              fn_args, ty_params);

    auto bcx = new_top_block_ctxt(fcx);

    let vec[ty.arg] arg_tys = arg_tys_of_fn(ann);
    copy_args_to_allocas(bcx, none[TypeRef], fn_args, arg_tys);

    auto llself_ty = type_of(cx, ret_ty_of_fn(ann));
    auto pair = bcx.fcx.llretptr;
    auto vtbl = trans_vtbl(cx, llself_ty, ob, ty_params);
    auto pair_vtbl = bcx.build.GEP(pair,
                                   vec(C_int(0),
                                       C_int(abi.obj_field_vtbl)));
    auto pair_box = bcx.build.GEP(pair,
                                  vec(C_int(0),
                                      C_int(abi.obj_field_box)));
    bcx.build.Store(vtbl, pair_vtbl);

    let TypeRef llbox_ty = T_opaque_obj_ptr(cx.tn);

    if (_vec.len[ast.ty_param](ty_params) == 0u &&
        _vec.len[ty.arg](arg_tys) == 0u) {
        // Store null into pair, if no args or typarams.
        bcx.build.Store(C_null(llbox_ty), pair_box);
    } else {
        // Malloc a box for the body and copy args in.
        let vec[@ty.t] obj_fields = vec();
        for (ty.arg a in arg_tys) {
            append[@ty.t](obj_fields, a.ty);
        }

        // Synthesize an obj body type.
        auto tydesc_ty = plain_ty(ty.ty_type);
        let vec[@ty.t] tps = vec();
        for (ast.ty_param tp in ty_params) {
            append[@ty.t](tps, tydesc_ty);
        }

        let @ty.t typarams_ty = plain_ty(ty.ty_tup(tps));
        let @ty.t fields_ty = plain_ty(ty.ty_tup(obj_fields));
        let @ty.t body_ty = plain_ty(ty.ty_tup(vec(tydesc_ty,
                                                   typarams_ty,
                                                   fields_ty)));
        let @ty.t boxed_body_ty = plain_ty(ty.ty_box(body_ty));

        let TypeRef llboxed_body_ty = type_of(cx, boxed_body_ty);

        // Malloc a box for the body.
        auto box = trans_malloc_inner(bcx, llboxed_body_ty);
        bcx = box.bcx;
        auto rc = GEP_tup_like(bcx, boxed_body_ty, box.val,
                               vec(0, abi.box_rc_field_refcnt));
        bcx = rc.bcx;
        auto body = GEP_tup_like(bcx, boxed_body_ty, box.val,
                                 vec(0, abi.box_rc_field_body));
        bcx = body.bcx;
        bcx.build.Store(C_int(1), rc.val);

        // Store body tydesc.
        auto body_tydesc =
            GEP_tup_like(bcx, body_ty, body.val,
                         vec(0, abi.obj_body_elt_tydesc));
        bcx = body_tydesc.bcx;

        auto body_td = get_tydesc(bcx, body_ty);
        bcx = body_td.bcx;
        bcx.build.Store(body_td.val, body_tydesc.val);

        // Copy typarams into captured typarams.
        auto body_typarams =
            GEP_tup_like(bcx, body_ty, body.val,
                         vec(0, abi.obj_body_elt_typarams));
        bcx = body_typarams.bcx;
        let int i = 0;
        for (ast.ty_param tp in ty_params) {
            auto typaram = bcx.fcx.lltydescs.get(tp.id);
            auto capture = GEP_tup_like(bcx, typarams_ty, body_typarams.val,
                                        vec(0, i));
            bcx = capture.bcx;
            bcx = copy_ty(bcx, INIT, capture.val, typaram, tydesc_ty).bcx;
            i += 1;
        }

        // Copy args into body fields.
        auto body_fields =
            GEP_tup_like(bcx, body_ty, body.val,
                         vec(0, abi.obj_body_elt_fields));
        bcx = body_fields.bcx;

        i = 0;
        for (ast.obj_field f in ob.fields) {
            auto arg = bcx.fcx.llargs.get(f.id);
            arg = load_scalar_or_boxed(bcx, arg, arg_tys.(i).ty);
            auto field = GEP_tup_like(bcx, fields_ty, body_fields.val,
                                      vec(0, i));
            bcx = field.bcx;
            bcx = copy_ty(bcx, INIT, field.val, arg, arg_tys.(i).ty).bcx;
            i += 1;
        }
        // Store box ptr in outer pair.
        auto p = bcx.build.PointerCast(box.val, llbox_ty);
        bcx.build.Store(p, pair_box);
    }
    bcx.build.RetVoid();
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
    create_llargs_for_fn_args(fcx, ast.proto_fn,
                              none[TypeRef], ret_ty_of_fn(variant.ann),
                              fn_args, ty_params);

    auto bcx = new_top_block_ctxt(fcx);

    auto arg_tys = arg_tys_of_fn(variant.ann);
    copy_args_to_allocas(bcx, none[TypeRef], fn_args, arg_tys);

    // FIXME: This is wrong for generic tags. We should be dynamically
    // computing "size" below based on the tydescs passed in.
    auto info = cx.tags.get(mk_plain_tag(tag_id));

    auto lltagty = T_struct(vec(T_int(), T_array(T_i8(), info.size)));

    // FIXME: better name.
    llvm.LLVMAddTypeName(cx.llmod, _str.buf("tag"), lltagty);

    auto lldiscrimptr = bcx.build.GEP(fcx.llretptr,
                                      vec(C_int(0), C_int(0)));
    bcx.build.Store(C_int(index), lldiscrimptr);

    auto llblobptr = bcx.build.GEP(fcx.llretptr,
                                   vec(C_int(0), C_int(1)));

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

    bcx = trans_block_cleanups(bcx, find_scope_cx(bcx));
    bcx.build.RetVoid();
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
            auto sub_cx = @rec(path=cx.path + sep() + name with *cx);
            trans_fn(sub_cx, f, fid, none[TypeRef], tps, ann);
        }
        case (ast.item_obj(?name, ?ob, ?tps, ?oid, ?ann)) {
            auto sub_cx = @rec(path=cx.path + sep() + name,
                               obj_typarams=tps,
                               obj_fields=ob.fields with *cx);
            trans_obj(sub_cx, ob, oid, tps, ann);
        }
        case (ast.item_mod(?name, ?m, _)) {
            auto sub_cx = @rec(path=cx.path + sep() + name with *cx);
            trans_mod(sub_cx, m);
        }
        case (ast.item_tag(?name, ?variants, ?tps, ?tag_id)) {
            auto sub_cx = @rec(path=cx.path + sep() + name with *cx);
            auto i = 0;
            for (ast.variant variant in variants) {
                trans_tag_variant(sub_cx, tag_id, variant, i, tps);
                i += 1;
            }
        }
        case (ast.item_const(?name, _, ?expr, ?cid, ?ann)) {
            auto sub_cx = @rec(path=cx.path + sep() + name with *cx);
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
    let str s = cx.names.next("_rust_" + kind) + sep() + name;
    let ValueRef llfn = decl_fastcall_fn(cx.llmod, s, llfty);

    // Declare the global constant pair that points to it.
    let str ps = cx.names.next("_rust_" + kind + "_pair") + sep() + name;

    register_fn_pair(cx, ps, llpairty, llfn, id);
}

fn register_fn_pair(@crate_ctxt cx, str ps, TypeRef llpairty, ValueRef llfn,
                    ast.def_id id) {
    let ValueRef gvar = llvm.LLVMAddGlobal(cx.llmod, llpairty,
                                           _str.buf(ps));
    auto pair = C_struct(vec(llfn,
                             C_null(T_opaque_closure_ptr(cx.tn))));

    llvm.LLVMSetInitializer(gvar, pair);
    llvm.LLVMSetGlobalConstant(gvar, True);
    llvm.LLVMSetLinkage(gvar,
                        lib.llvm.LLVMPrivateLinkage
                        as llvm.Linkage);

    cx.item_ids.insert(id, llfn);
    cx.fn_pairs.insert(id, gvar);
}

fn native_fn_wrapper_type(@crate_ctxt cx, &ast.ann ann) -> TypeRef {
    auto x = node_ann_type(cx, ann);
    alt (x.struct) {
        case (ty.ty_native_fn(?abi, ?args, ?out)) {
            ret type_of_fn(cx, ast.proto_fn, args, out);
        }
    }
    fail;
}

fn decl_native_fn_and_pair(@crate_ctxt cx,
                           str name,
                           &ast.ann ann,
                           ast.def_id id) {
    // Declare the wrapper.
    auto wrapper_type = native_fn_wrapper_type(cx, ann);
    let str s = cx.names.next("_rust_wrapper") + sep() + name;
    let ValueRef wrapper_fn = decl_fastcall_fn(cx.llmod, s, wrapper_type);

    // Declare the global constant pair that points to it.
    auto wrapper_pair_type = T_fn_pair(cx.tn, wrapper_type);
    let str ps = cx.names.next("_rust_wrapper_pair") + sep() + name;

    register_fn_pair(cx, ps, wrapper_pair_type, wrapper_fn, id);

    // Declare the function itself.
    auto llfty = get_pair_fn_ty(node_type(cx, ann));
    decl_cdecl_fn(cx.llmod, name, llfty);
}

fn collect_native_item(&@crate_ctxt cx, @ast.native_item i) -> @crate_ctxt {
    alt (i.node) {
        case (ast.native_item_fn(?name, _, _, ?fid, ?ann)) {
            cx.native_items.insert(fid, i);
            if (! cx.obj_methods.contains_key(fid)) {
                decl_native_fn_and_pair(cx, name, ann, fid);
            }
        }
        case (_) { /* fall through */ }
    }
    ret cx;
}

fn collect_item(&@crate_ctxt cx, @ast.item i) -> @crate_ctxt {

    alt (i.node) {
        case (ast.item_fn(?name, ?f, _, ?fid, ?ann)) {
            cx.items.insert(fid, i);
            if (! cx.obj_methods.contains_key(fid)) {
                decl_fn_and_pair(cx, "fn", name, ann, fid);
            }
        }

        case (ast.item_obj(?name, ?ob, _, ?oid, ?ann)) {
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

        case (ast.item_tag(_, ?variants, ?tps, ?tag_id)) {
            auto vi = new_def_hash[uint]();
            auto navi = new_def_hash[uint]();
            cx.tags.insert(mk_plain_tag(tag_id), @rec(th=mk_type_handle(),
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

    fld = @rec( update_env_for_item = bind collect_item(_,_),
                update_env_for_native_item = bind collect_native_item(_,_)
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

            auto info = cx.tags.get(mk_plain_tag(tag_id));

            for (ast.variant variant in variants) {
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
                }
            }

            info.size = max_size;

            // FIXME: alignment is wrong here, manually insert padding I
            // guess :(
            auto tag_ty = T_struct(vec(T_int(), T_array(T_i8(), max_size)));
            auto th = info.th.llth;
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
            auto info = cx.tags.get(mk_plain_tag(tag_id));

            auto tag_ty = llvm.LLVMResolveTypeHandle(info.th.llth);
            check (llvm.LLVMCountStructElementTypes(tag_ty) == 2u);
            auto elts = vec(0 as TypeRef, 0 as TypeRef);
            llvm.LLVMGetStructElementTypes(tag_ty, _vec.buf[TypeRef](elts));
            auto union_ty = elts.(1);

            auto i = 0u;
            auto n_variants = _vec.len[ast.variant](variants);
            while (i < n_variants) {
                auto variant = variants.(i);
                if (_vec.len[ast.variant_arg](variant.args) == 0u) {
                    // Nullary tags become constants. (N-ary tags are treated
                    // as functions and generated later.)

                    auto union_val = C_zero_byte_arr(info.size as uint);
                    auto val = C_struct(vec(C_int(i as int), union_val));

                    // FIXME: better name
                    auto gvar = llvm.LLVMAddGlobal(cx.llmod, val_ty(val),
                                                   _str.buf("tag"));
                    llvm.LLVMSetInitializer(gvar, val);
                    llvm.LLVMSetGlobalConstant(gvar, True);
                    llvm.LLVMSetLinkage(gvar, lib.llvm.LLVMPrivateLinkage
                                        as llvm.Linkage);
                    cx.item_ids.insert(variant.id, gvar);
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

fn i2p(ValueRef v, TypeRef t) -> ValueRef {
    ret llvm.LLVMConstIntToPtr(v, t);
}

fn trans_exit_task_glue(@crate_ctxt cx) {
    let vec[TypeRef] T_args = vec();
    let vec[ValueRef] V_args = vec();

    auto llfn = cx.glues.exit_task_glue;
    let ValueRef lltaskptr = llvm.LLVMGetParam(llfn, 3u);
    auto fcx = @rec(llfn=llfn,
                    lltaskptr=lltaskptr,
                    llenv=C_null(T_opaque_closure_ptr(cx.tn)),
                    llretptr=C_null(T_ptr(T_nil())),
                    mutable llself=none[ValueRef],
                    mutable lliterbody=none[ValueRef],
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
    llvm.LLVMAddTypeName(cx.llmod, _str.buf("crate"), T_crate(cx.tn));
    llvm.LLVMAddTypeName(cx.llmod, _str.buf("task"), T_task(cx.tn));
    llvm.LLVMAddTypeName(cx.llmod, _str.buf("tydesc"), T_tydesc(cx.tn));
}

fn create_crate_constant(@crate_ctxt cx) {

    let ValueRef crate_addr = p2i(cx.crate_ptr);

    let ValueRef activate_glue_off =
        llvm.LLVMConstSub(p2i(cx.glues.activate_glue), crate_addr);

    let ValueRef yield_glue_off =
        llvm.LLVMConstSub(p2i(cx.glues.yield_glue), crate_addr);

    let ValueRef exit_task_glue_off =
        llvm.LLVMConstSub(p2i(cx.glues.exit_task_glue), crate_addr);

    let ValueRef crate_val =
        C_struct(vec(C_null(T_int()),     // ptrdiff_t image_base_off
                     p2i(cx.crate_ptr),   // uintptr_t self_addr
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
                     C_null(T_int()),     // int n_libs
                     C_int(abi.abi_x86_rustc_fastcall) // uintptr_t abi_tag
                     ));

    llvm.LLVMSetInitializer(cx.crate_ptr, crate_val);
}

fn find_main_fn(@crate_ctxt cx) -> ValueRef {
    auto e = sep() + "main";
    let ValueRef v = C_nil();
    let uint n = 0u;
    for each (tup(str,ValueRef) i in cx.item_names.items()) {
        if (_str.ends_with(i._0, e)) {
            n += 1u;
            v = i._1;
        }
    }
    alt (n) {
        case (0u) {
            cx.sess.err("main fn not found");
        }
        case (1u) {
            ret v;
        }
        case (_) {
            cx.sess.err("multiple main fns found");
        }
    }
    fail;
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
    auto llrust_main = find_main_fn(cx);

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


fn trace_str(@block_ctxt cx, str s) {
    trans_upcall(cx, "upcall_trace_str", vec(p2i(C_cstr(cx.fcx.ccx, s))));
}

fn trace_word(@block_ctxt cx, ValueRef v) {
    trans_upcall(cx, "upcall_trace_word", vec(v));
}

fn trace_ptr(@block_ctxt cx, ValueRef v) {
    trace_word(cx, cx.build.PtrToInt(v, T_int()));
}

fn trap(@block_ctxt bcx) {
    let vec[ValueRef] v = vec();
    bcx.build.Call(bcx.fcx.ccx.intrinsics.get("llvm.trap"), v);
}

fn check_module(ModuleRef llmod) {
    auto pm = mk_pass_manager();
    llvm.LLVMAddVerifierPass(pm.llpm);
    llvm.LLVMRunPassManager(pm.llpm, llmod);

    // TODO: run the linter here also, once there are llvm-c bindings for it.
}

fn make_no_op_type_glue(ModuleRef llmod, type_names tn) -> ValueRef {
    auto ty = T_fn(vec(T_taskptr(tn), T_ptr(T_i8())), T_void());
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

fn make_glues(ModuleRef llmod, type_names tn) -> @glue_fns {
    ret @rec(activate_glue = decl_glue(llmod, tn, abi.activate_glue_name()),
             yield_glue = decl_glue(llmod, tn, abi.yield_glue_name()),
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
                                            T_fn(vec(T_int(),
                                                     T_int(),
                                                     T_int(),
                                                     T_taskptr(tn)),
                                                 T_void())),

             upcall_glues =
             _vec.init_fn[ValueRef](bind decl_upcall_glue(llmod, tn, _),
                                    abi.n_upcall_glues as uint),
             no_op_type_glue = make_no_op_type_glue(llmod, tn),
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
    auto tn = mk_type_names();
    let ValueRef crate_ptr =
        llvm.LLVMAddGlobal(llmod, T_crate(tn), _str.buf("rust_crate"));

    llvm.LLVMSetModuleInlineAsm(llmod, _str.buf(x86.get_module_asm()));

    auto intrinsics = declare_intrinsics(llmod);

    auto glues = make_glues(llmod, tn);
    auto hasher = ty.hash_ty;
    auto eqer = ty.eq_ty;
    auto tags = map.mk_hashmap[@ty.t,@tag_info](hasher, eqer);
    auto tydescs = map.mk_hashmap[@ty.t,ValueRef](hasher, eqer);
    let vec[ast.ty_param] obj_typarams = vec();
    let vec[ast.obj_field] obj_fields = vec();

    auto cx = @rec(sess = sess,
                   llmod = llmod,
                   td = td,
                   tn = tn,
                   crate_ptr = crate_ptr,
                   upcalls = new_str_hash[ValueRef](),
                   intrinsics = intrinsics,
                   item_names = new_str_hash[ValueRef](),
                   item_ids = new_def_hash[ValueRef](),
                   items = new_def_hash[@ast.item](),
                   native_items = new_def_hash[@ast.native_item](),
                   tags = tags,
                   fn_pairs = new_def_hash[ValueRef](),
                   consts = new_def_hash[ValueRef](),
                   obj_methods = new_def_hash[()](),
                   tydescs = tydescs,
                   obj_typarams = obj_typarams,
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
    create_crate_constant(cx);
    if (!shared) {
        trans_main_fn(cx, cx.crate_ptr);
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
