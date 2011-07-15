/**
   Code that is useful in various trans modules.

*/

import std::int;
import std::str;
import std::uint;
import std::str::rustrt::sbuf;
import std::map;
import std::map::hashmap;
import std::option;
import std::option::some;
import std::option::none;
import std::fs;
import syntax::ast;
import syntax::walk;
import driver::session;
import middle::ty;
import back::link;
import back::x86;
import back::abi;
import back::upcall;
import syntax::visit;
import visit::vt;
import util::common;
import util::common::*;
import std::map::new_int_hash;
import std::map::new_str_hash;
import syntax::codemap::span;
import lib::llvm::llvm;
import lib::llvm::builder;
import lib::llvm::target_data;
import lib::llvm::type_names;
import lib::llvm::mk_target_data;
import lib::llvm::mk_type_names;
import lib::llvm::llvm::ModuleRef;
import lib::llvm::llvm::ValueRef;
import lib::llvm::llvm::TypeRef;
import lib::llvm::llvm::TypeHandleRef;
import lib::llvm::llvm::BuilderRef;
import lib::llvm::llvm::BasicBlockRef;
import lib::llvm::False;
import lib::llvm::True;
import lib::llvm::Bool;
import link::mangle_internal_name_by_type_only;
import link::mangle_internal_name_by_seq;
import link::mangle_internal_name_by_path;
import link::mangle_internal_name_by_path_and_seq;
import link::mangle_exported_name;
import metadata::creader;
import metadata::csearch;
import metadata::cstore;
import util::ppaux::ty_to_str;
import util::ppaux::ty_to_short_str;
import syntax::print::pprust::expr_to_str;
import syntax::print::pprust::path_to_str;

// FIXME: These should probably be pulled in here too.
import trans::crate_ctxt;
import trans::type_of_fn_full;
import trans::val_ty;

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

    ret llvm::LLVMVoidType();
}

fn T_nil() -> TypeRef {
    // NB: See above in T_void().

    ret llvm::LLVMInt1Type();
}

fn T_i1() -> TypeRef { ret llvm::LLVMInt1Type(); }

fn T_i8() -> TypeRef { ret llvm::LLVMInt8Type(); }

fn T_i16() -> TypeRef { ret llvm::LLVMInt16Type(); }

fn T_i32() -> TypeRef { ret llvm::LLVMInt32Type(); }

fn T_i64() -> TypeRef { ret llvm::LLVMInt64Type(); }

fn T_f32() -> TypeRef { ret llvm::LLVMFloatType(); }

fn T_f64() -> TypeRef { ret llvm::LLVMDoubleType(); }

fn T_bool() -> TypeRef { ret T_i1(); }

fn T_int() -> TypeRef {
    // FIXME: switch on target type.

    ret T_i32();
}

fn T_float() -> TypeRef {
    // FIXME: switch on target type.

    ret T_f64();
}

fn T_char() -> TypeRef { ret T_i32(); }

fn T_size_t() -> TypeRef {
    // FIXME: switch on target type.

    ret T_i32();
}

fn T_fn(&TypeRef[] inputs, TypeRef output) -> TypeRef {
    ret llvm::LLVMFunctionType(output, std::ivec::to_ptr(inputs),
                               std::ivec::len[TypeRef](inputs), False);
}

fn T_fn_pair(&crate_ctxt cx, TypeRef tfn) -> TypeRef {
    ret T_struct(~[T_ptr(tfn), T_opaque_closure_ptr(cx)]);
}

fn T_ptr(TypeRef t) -> TypeRef { ret llvm::LLVMPointerType(t, 0u); }

fn T_struct(&TypeRef[] elts) -> TypeRef {
    ret llvm::LLVMStructType(std::ivec::to_ptr(elts), std::ivec::len(elts),
                             False);
}

fn T_named_struct(&str name) -> TypeRef {
    auto c = llvm::LLVMGetGlobalContext();
    ret llvm::LLVMStructCreateNamed(c, str::buf(name));
}

fn set_struct_body(TypeRef t, &TypeRef[] elts) {
    llvm::LLVMStructSetBody(t, std::ivec::to_ptr(elts), std::ivec::len(elts),
                            False);
}

fn T_empty_struct() -> TypeRef { ret T_struct(~[]); }

fn T_rust_object() -> TypeRef {
    auto t = T_named_struct("rust_object");
    auto e = T_ptr(T_empty_struct());
    set_struct_body(t, ~[e,e]);
    ret t;
}

fn T_task() -> TypeRef {
    auto t = T_named_struct("task");

    auto elems = ~[T_int(), // Refcount
                   T_int(), // Delegate pointer
                   T_int(), // Stack segment pointer
                   T_int(), // Runtime SP
                   T_int(), // Rust SP
                   T_int(), // GC chain

                   T_int(), // Domain pointer
                            // Crate cache pointer
                   T_int()];
    set_struct_body(t, elems);
    ret t;
}

fn T_tydesc_field(&crate_ctxt cx, int field) -> TypeRef {
    // Bit of a kludge: pick the fn typeref out of the tydesc..

    let TypeRef[] tydesc_elts =
        std::ivec::init_elt[TypeRef](T_nil(), abi::n_tydesc_fields as uint);
    llvm::LLVMGetStructElementTypes(cx.tydesc_type,
                                    std::ivec::to_ptr[TypeRef](tydesc_elts));
    auto t = llvm::LLVMGetElementType(tydesc_elts.(field));
    ret t;
}

fn T_glue_fn(&crate_ctxt cx) -> TypeRef {
    auto s = "glue_fn";
    if (cx.tn.name_has_type(s)) { ret cx.tn.get_type(s); }
    auto t = T_tydesc_field(cx, abi::tydesc_field_drop_glue);
    cx.tn.associate(s, t);
    ret t;
}

fn T_dtor(&@crate_ctxt ccx, &span sp) -> TypeRef {
    ret type_of_fn_full(ccx, sp, ast::proto_fn, true,
                        ~[], ty::mk_nil(ccx.tcx), 0u);
}

fn T_cmp_glue_fn(&crate_ctxt cx) -> TypeRef {
    auto s = "cmp_glue_fn";
    if (cx.tn.name_has_type(s)) { ret cx.tn.get_type(s); }
    auto t = T_tydesc_field(cx, abi::tydesc_field_cmp_glue);
    cx.tn.associate(s, t);
    ret t;
}

fn T_tydesc(TypeRef taskptr_type) -> TypeRef {
    auto tydesc = T_named_struct("tydesc");
    auto tydescpp = T_ptr(T_ptr(tydesc));
    auto pvoid = T_ptr(T_i8());
    auto glue_fn_ty =
        T_ptr(T_fn(~[T_ptr(T_nil()), taskptr_type, T_ptr(T_nil()), tydescpp,
                     pvoid], T_void()));
    auto cmp_glue_fn_ty =
        T_ptr(T_fn(~[T_ptr(T_i1()), taskptr_type, T_ptr(T_nil()), tydescpp,
                     pvoid, pvoid, T_i8()], T_void()));

    auto elems = ~[tydescpp,   // first_param
                   T_int(),    // size
                   T_int(),    // align
                   glue_fn_ty, // copy_glue
                   glue_fn_ty, // drop_glue
                   glue_fn_ty, // free_glue
                   glue_fn_ty, // sever_glue
                   glue_fn_ty, // mark_glue
                   glue_fn_ty, // obj_drop_glue
                   glue_fn_ty, // is_stateful
                   cmp_glue_fn_ty];
    set_struct_body(tydesc, elems);
    ret tydesc;
}

fn T_array(TypeRef t, uint n) -> TypeRef { ret llvm::LLVMArrayType(t, n); }

fn T_vec(TypeRef t) -> TypeRef {
    ret T_struct(~[T_int(), // Refcount
                   T_int(), // Alloc
                   T_int(), // Fill

                   T_int(), // Pad
                           // Body elements
                            T_array(t, 0u)]);
}

fn T_opaque_vec_ptr() -> TypeRef { ret T_ptr(T_vec(T_int())); }


// Interior vector.
//
// TODO: Support user-defined vector sizes.
fn T_ivec(TypeRef t) -> TypeRef {
    ret T_struct(~[T_int(), // Length ("fill"; if zero, heapified)
                   T_int(), // Alloc
                   T_array(t, abi::ivec_default_length)]); // Body elements

}


// Note that the size of this one is in bytes.
fn T_opaque_ivec() -> TypeRef {
    ret T_struct(~[T_int(), // Length ("fill"; if zero, heapified)
                   T_int(), // Alloc
                   T_array(T_i8(), 0u)]); // Body elements

}

fn T_ivec_heap_part(TypeRef t) -> TypeRef {
    ret T_struct(~[T_int(), // Real length
                   T_array(t, 0u)]); // Body elements

}


// Interior vector on the heap, also known as the "stub". Cast to this when
// the allocated length (second element of T_ivec above) is zero.
fn T_ivec_heap(TypeRef t) -> TypeRef {
    ret T_struct(~[T_int(), // Length (zero)
                   T_int(), // Alloc
                   T_ptr(T_ivec_heap_part(t))]); // Pointer

}

fn T_opaque_ivec_heap_part() -> TypeRef {
    ret T_struct(~[T_int(), // Real length
                   T_array(T_i8(), 0u)]); // Body elements

}

fn T_opaque_ivec_heap() -> TypeRef {
    ret T_struct(~[T_int(), // Length (zero)
                   T_int(), // Alloc
                   T_ptr(T_opaque_ivec_heap_part())]); // Pointer

}

fn T_str() -> TypeRef { ret T_vec(T_i8()); }

fn T_box(TypeRef t) -> TypeRef { ret T_struct(~[T_int(), t]); }

fn T_port(TypeRef t) -> TypeRef {
    ret T_struct(~[T_int()]); // Refcount

}

fn T_chan(TypeRef t) -> TypeRef {
    ret T_struct(~[T_int()]); // Refcount

}

fn T_taskptr(&crate_ctxt cx) -> TypeRef { ret T_ptr(cx.task_type); }


// This type must never be used directly; it must always be cast away.
fn T_typaram(&type_names tn) -> TypeRef {
    auto s = "typaram";
    if (tn.name_has_type(s)) { ret tn.get_type(s); }
    auto t = T_i8();
    tn.associate(s, t);
    ret t;
}

fn T_typaram_ptr(&type_names tn) -> TypeRef { ret T_ptr(T_typaram(tn)); }

fn T_closure_ptr(&crate_ctxt cx, TypeRef lltarget_ty, TypeRef llbindings_ty,
                 uint n_ty_params) -> TypeRef {
    // NB: keep this in sync with code in trans_bind; we're making
    // an LLVM typeref structure that has the same "shape" as the ty::t
    // it constructs.

    ret T_ptr(T_box(T_struct(~[T_ptr(cx.tydesc_type), lltarget_ty,
                               llbindings_ty,
                               T_captured_tydescs(cx, n_ty_params)])));
}

fn T_opaque_closure_ptr(&crate_ctxt cx) -> TypeRef {
    auto s = "*closure";
    if (cx.tn.name_has_type(s)) { ret cx.tn.get_type(s); }
    auto t =
        T_closure_ptr(cx,
                      T_struct(~[T_ptr(T_nil()), T_ptr(T_nil())]),
                      T_nil(),
                      0u);
    cx.tn.associate(s, t);
    ret t;
}

fn T_tag(&type_names tn, uint size) -> TypeRef {
    auto s = "tag_" + uint::to_str(size, 10u);
    if (tn.name_has_type(s)) { ret tn.get_type(s); }
    auto t = T_struct(~[T_int(), T_array(T_i8(), size)]);
    tn.associate(s, t);
    ret t;
}

fn T_opaque_tag(&type_names tn) -> TypeRef {
    auto s = "opaque_tag";
    if (tn.name_has_type(s)) { ret tn.get_type(s); }
    auto t = T_struct(~[T_int(), T_i8()]);
    tn.associate(s, t);
    ret t;
}

fn T_opaque_tag_ptr(&type_names tn) -> TypeRef {
    ret T_ptr(T_opaque_tag(tn));
}

fn T_captured_tydescs(&crate_ctxt cx, uint n) -> TypeRef {
    ret T_struct(std::ivec::init_elt[TypeRef](T_ptr(cx.tydesc_type), n));
}

fn T_obj_ptr(&crate_ctxt cx, uint n_captured_tydescs) -> TypeRef {
    // This function is not publicly exposed because it returns an incomplete
    // type. The dynamically-sized fields follow the captured tydescs.

    fn T_obj(&crate_ctxt cx, uint n_captured_tydescs) -> TypeRef {
        ret T_struct(~[T_ptr(cx.tydesc_type),
                       T_captured_tydescs(cx, n_captured_tydescs)]);
    }
    ret T_ptr(T_box(T_obj(cx, n_captured_tydescs)));
}

fn T_opaque_obj_ptr(&crate_ctxt cx) -> TypeRef { ret T_obj_ptr(cx, 0u); }

fn T_opaque_port_ptr() -> TypeRef { ret T_ptr(T_i8()); }

fn T_opaque_chan_ptr() -> TypeRef { ret T_ptr(T_i8()); }


// LLVM constant constructors.
fn C_null(TypeRef t) -> ValueRef { ret llvm::LLVMConstNull(t); }

fn C_integral(TypeRef t, uint u, Bool sign_extend) -> ValueRef {
    // FIXME: We can't use LLVM::ULongLong with our existing minimal native
    // API, which only knows word-sized args.
    //
    // ret llvm::LLVMConstInt(T_int(), t as LLVM::ULongLong, False);
    //

    ret llvm::LLVMRustConstSmallInt(t, u, sign_extend);
}

fn C_float(&str s) -> ValueRef {
    ret llvm::LLVMConstRealOfString(T_float(), str::buf(s));
}

fn C_floating(&str s, TypeRef t) -> ValueRef {
    ret llvm::LLVMConstRealOfString(t, str::buf(s));
}

fn C_nil() -> ValueRef {
    // NB: See comment above in T_void().

    ret C_integral(T_i1(), 0u, False);
}

fn C_bool(bool b) -> ValueRef {
    if (b) {
        ret C_integral(T_bool(), 1u, False);
    } else { ret C_integral(T_bool(), 0u, False); }
}

fn C_int(int i) -> ValueRef { ret C_integral(T_int(), i as uint, True); }

fn C_uint(uint i) -> ValueRef { ret C_integral(T_int(), i, False); }

fn C_u8(uint i) -> ValueRef { ret C_integral(T_i8(), i, False); }


// This is a 'c-like' raw string, which differs from
// our boxed-and-length-annotated strings.
fn C_cstr(&@crate_ctxt cx, &str s) -> ValueRef {
    auto sc = llvm::LLVMConstString(str::buf(s), str::byte_len(s), False);
    auto g =
        llvm::LLVMAddGlobal(cx.llmod, val_ty(sc),
                            str::buf(cx.names.next("str")));
    llvm::LLVMSetInitializer(g, sc);
    llvm::LLVMSetGlobalConstant(g, True);
    llvm::LLVMSetLinkage(g, lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    ret g;
}


// A rust boxed-and-length-annotated string.
fn C_str(&@crate_ctxt cx, &str s) -> ValueRef {
    auto len = str::byte_len(s);
    auto box =
        C_struct(~[C_int(abi::const_refcount as int),
                   C_int(len + 1u as int), // 'alloc'
                   C_int(len + 1u as int), // 'fill'
                   C_int(0), // 'pad'
                   llvm::LLVMConstString(str::buf(s), len, False)]);
    auto g =
        llvm::LLVMAddGlobal(cx.llmod, val_ty(box),
                            str::buf(cx.names.next("str")));
    llvm::LLVMSetInitializer(g, box);
    llvm::LLVMSetGlobalConstant(g, True);
    llvm::LLVMSetLinkage(g, lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    ret llvm::LLVMConstPointerCast(g, T_ptr(T_str()));
}

// Returns a Plain Old LLVM String:
fn C_postr(&str s) -> ValueRef {
    ret llvm::LLVMConstString(str::buf(s), str::byte_len(s), False);
}

fn C_zero_byte_arr(uint size) -> ValueRef {
    auto i = 0u;
    let ValueRef[] elts = ~[];
    while (i < size) { elts += ~[C_u8(0u)]; i += 1u; }
    ret llvm::LLVMConstArray(T_i8(), std::ivec::to_ptr(elts),
                             std::ivec::len(elts));
}

fn C_struct(&ValueRef[] elts) -> ValueRef {
    ret llvm::LLVMConstStruct(std::ivec::to_ptr(elts), std::ivec::len(elts),
                              False);
}

fn C_named_struct(TypeRef T, &ValueRef[] elts) -> ValueRef {
    ret llvm::LLVMConstNamedStruct(T, std::ivec::to_ptr(elts),
                                   std::ivec::len(elts));
}

fn C_array(TypeRef ty, &ValueRef[] elts) -> ValueRef {
    ret llvm::LLVMConstArray(ty, std::ivec::to_ptr(elts),
                             std::ivec::len(elts));
}

