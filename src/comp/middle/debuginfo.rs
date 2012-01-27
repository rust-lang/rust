import core::{vec, str, option, sys, ctypes, unsafe};
import std::fs;
import std::map::hashmap;
import lib::llvm::llvm;
import lib::llvm::llvm::ValueRef;
import trans::common::*;
import trans::base;
import trans::build::B;
import middle::ty;
import syntax::{ast, codemap};
import codemap::span;
import ast::ty;
import pat_util::*;
import util::ppaux::ty_to_str;

export create_local_var;
export create_function;
export create_arg;
export update_source_pos;
export debug_ctxt;

const LLVMDebugVersion: int = (9 << 16);

const DW_LANG_RUST: int = 0x9000;
const DW_VIRTUALITY_none: int = 0;

const CompileUnitTag: int = 17;
const FileDescriptorTag: int = 41;
const SubprogramTag: int = 46;
const SubroutineTag: int = 21;
const BasicTypeDescriptorTag: int = 36;
const AutoVariableTag: int = 256;
const ArgVariableTag: int = 257;
const ReturnVariableTag: int = 258;
const LexicalBlockTag: int = 11;
const PointerTypeTag: int = 15;
const StructureTypeTag: int = 19;
const MemberTag: int = 13;
const ArrayTypeTag: int = 1;
const SubrangeTag: int = 33;

const DW_ATE_boolean: int = 0x02;
const DW_ATE_float: int = 0x04;
const DW_ATE_signed: int = 0x05;
const DW_ATE_signed_char: int = 0x06;
const DW_ATE_unsigned: int = 0x07;
const DW_ATE_unsigned_char: int = 0x08;

fn llstr(s: str) -> ValueRef {
    str::as_buf(s, {|sbuf|
        llvm::LLVMMDString(sbuf, str::byte_len(s) as ctypes::c_uint)
    })
}
fn lltag(lltag: int) -> ValueRef {
    lli32(LLVMDebugVersion | lltag)
}
fn lli32(val: int) -> ValueRef {
    C_i32(val as i32)
}
fn lli64(val: int) -> ValueRef {
    C_i64(val as i64)
}
fn lli1(bval: bool) -> ValueRef {
    C_bool(bval)
}
fn llmdnode(elems: [ValueRef]) -> ValueRef unsafe {
    llvm::LLVMMDNode(vec::unsafe::to_ptr(elems),
                     vec::len(elems) as ctypes::c_uint)
}
fn llunused() -> ValueRef {
    lli32(0x0)
}
fn llnull() -> ValueRef unsafe {
    unsafe::reinterpret_cast(ptr::null::<ValueRef>())
}

fn add_named_metadata(cx: @crate_ctxt, name: str, val: ValueRef) {
    str::as_buf(name, {|sbuf|
        llvm::LLVMAddNamedMetadataOperand(cx.llmod, sbuf,
                                          val)
    })
}

////////////////

type debug_ctxt = {
    llmetadata: metadata_cache,
    names: namegen
};

fn update_cache(cache: metadata_cache, mdtag: int, val: debug_metadata) {
    let existing = if cache.contains_key(mdtag) {
        cache.get(mdtag)
    } else {
        []
    };
    cache.insert(mdtag, existing + [val]);
}

type metadata<T> = {node: ValueRef, data: T};

type file_md = {path: str};
type compile_unit_md = {path: str};
type subprogram_md = {path: str};
type local_var_md = {id: ast::node_id};
type tydesc_md = {hash: uint};
type block_md = {start: codemap::loc, end: codemap::loc};
type argument_md = {id: ast::node_id};
type retval_md = {id: ast::node_id};

type metadata_cache = hashmap<int, [debug_metadata]>;

enum debug_metadata {
    file_metadata(@metadata<file_md>),
    compile_unit_metadata(@metadata<compile_unit_md>),
    subprogram_metadata(@metadata<subprogram_md>),
    local_var_metadata(@metadata<local_var_md>),
    tydesc_metadata(@metadata<tydesc_md>),
    block_metadata(@metadata<block_md>),
    argument_metadata(@metadata<argument_md>),
    retval_metadata(@metadata<retval_md>),
}

fn cast_safely<T: copy, U>(val: T) -> U unsafe {
    let val2 = val;
    let val3 = unsafe::reinterpret_cast(val2);
    unsafe::leak(val2);
    ret val3;
}

fn md_from_metadata<T>(val: debug_metadata) -> T unsafe {
    alt val {
      file_metadata(md) { cast_safely(md) }
      compile_unit_metadata(md) { cast_safely(md) }
      subprogram_metadata(md) { cast_safely(md) }
      local_var_metadata(md) { cast_safely(md) }
      tydesc_metadata(md) { cast_safely(md) }
      block_metadata(md) { cast_safely(md) }
      argument_metadata(md) { cast_safely(md) }
      retval_metadata(md) { cast_safely(md) }
    }
}

fn cached_metadata<T: copy>(cache: metadata_cache, mdtag: int,
                           eq: fn(md: T) -> bool) -> option::t<T> unsafe {
    if cache.contains_key(mdtag) {
        let items = cache.get(mdtag);
        for item in items {
            let md: T = md_from_metadata::<T>(item);
            if eq(md) {
                ret option::some(md);
            }
        }
    }
    ret option::none;
}

fn create_compile_unit(cx: @crate_ctxt, full_path: str)
    -> @metadata<compile_unit_md> {
    let cache = get_cache(cx);
    let tg = CompileUnitTag;
    alt cached_metadata::<@metadata<compile_unit_md>>(cache, tg,
                        {|md| md.data.path == full_path}) {
      option::some(md) { ret md; }
      option::none {}
    }

    let work_dir = cx.sess.working_dir;
    let file_path = if str::starts_with(full_path, work_dir) {
        str::slice(full_path, str::byte_len(work_dir),
                   str::byte_len(full_path))
    } else {
        full_path
    };
    let unit_metadata = [lltag(tg),
                         llunused(),
                         lli32(DW_LANG_RUST),
                         llstr(file_path),
                         llstr(work_dir),
                         llstr(#env["CFG_VERSION"]),
                         lli1(false), // main compile unit
                         lli1(cx.sess.opts.optimize != 0u),
                         llstr(""), // flags (???)
                         lli32(0) // runtime version (???)
                         // list of enum types
                         // list of retained values
                         // list of subprograms
                         // list of global variables
                        ];
    let unit_node = llmdnode(unit_metadata);
    add_named_metadata(cx, "llvm.dbg.cu", unit_node);
    let mdval = @{node: unit_node, data: {path: full_path}};
    update_cache(cache, tg, compile_unit_metadata(mdval));
    ret mdval;
}

fn get_cache(cx: @crate_ctxt) -> metadata_cache {
    option::get(cx.dbg_cx).llmetadata
}

fn create_file(cx: @crate_ctxt, full_path: str) -> @metadata<file_md> {
    let cache = get_cache(cx);;
    let tg = FileDescriptorTag;
    alt cached_metadata::<@metadata<file_md>>(
        cache, tg, {|md| md.data.path == full_path}) {
        option::some(md) { ret md; }
        option::none {}
    }

    let fname = fs::basename(full_path);
    let path = fs::dirname(full_path);
    let unit_node = create_compile_unit(cx, full_path).node;
    let file_md = [lltag(tg),
                   llstr(fname),
                   llstr(path),
                   unit_node];
    let val = llmdnode(file_md);
    let mdval = @{node: val, data: {path: full_path}};
    update_cache(cache, tg, file_metadata(mdval));
    ret mdval;
}

fn line_from_span(cm: codemap::codemap, sp: span) -> uint {
    codemap::lookup_char_pos(cm, sp.lo).line
}

fn create_block(cx: @block_ctxt, sp: span) -> @metadata<block_md> {
    let cache = get_cache(bcx_ccx(cx));
    let start = codemap::lookup_char_pos(bcx_ccx(cx).sess.codemap,
                                         sp.lo);
    let fname = start.filename;
    let end = codemap::lookup_char_pos(bcx_ccx(cx).sess.codemap,
                                       sp.hi);
    let tg = LexicalBlockTag;
    alt cached_metadata::<@metadata<block_md>>(
        cache, tg,
        {|md| start == md.data.start && end == md.data.end}) {
      option::some(md) { ret md; }
      option::none {}
    }

    let parent = alt cx.parent {
      parent_none { create_function(cx.fcx, sp).node }
      parent_some(bcx) { create_block(cx, sp).node }
    };
    let file_node = create_file(bcx_ccx(cx), fname);
    let unique_id = alt cache.find(LexicalBlockTag) {
      option::some(v) { vec::len(v) as int }
      option::none { 0 }
    };
    let lldata = [lltag(tg),
                  parent,
                  lli32(start.line as int),
                  lli32(start.col as int),
                  file_node.node,
                  lli32(unique_id)
                 ];
      let val = llmdnode(lldata);
      let mdval = @{node: val, data: {start: start, end: end}};
      update_cache(cache, tg, block_metadata(mdval));
      ret mdval;
}

fn size_and_align_of<T>() -> (int, int) {
    (sys::size_of::<T>() as int, sys::align_of::<T>() as int)
}

fn create_basic_type(cx: @crate_ctxt, t: ty::t, ty: @ast::ty)
    -> @metadata<tydesc_md> {
    let cache = get_cache(cx);
    let tg = BasicTypeDescriptorTag;
    alt cached_metadata::<@metadata<tydesc_md>>(
        cache, tg, {|md| t == md.data.hash}) {
      option::some(md) { ret md; }
      option::none {}
    }

    let (name, (size, align), encoding) = alt ty.node {
      ast::ty_bool {("bool", size_and_align_of::<bool>(), DW_ATE_boolean)}
      ast::ty_int(m) { alt m {
        ast::ty_char {("char", size_and_align_of::<char>(), DW_ATE_unsigned)}
        ast::ty_i {("int", size_and_align_of::<int>(), DW_ATE_signed)}
        ast::ty_i8 {("i8", size_and_align_of::<i8>(), DW_ATE_signed_char)}
        ast::ty_i16 {("i16", size_and_align_of::<i16>(), DW_ATE_signed)}
        ast::ty_i32 {("i32", size_and_align_of::<i32>(), DW_ATE_signed)}
        ast::ty_i64 {("i64", size_and_align_of::<i64>(), DW_ATE_signed)}
      }}
      ast::ty_uint(m) { alt m {
        ast::ty_u {("uint", size_and_align_of::<uint>(), DW_ATE_unsigned)}
        ast::ty_u8 {("u8", size_and_align_of::<u8>(), DW_ATE_unsigned_char)}
        ast::ty_u16 {("u16", size_and_align_of::<u16>(), DW_ATE_unsigned)}
        ast::ty_u32 {("u32", size_and_align_of::<u32>(), DW_ATE_unsigned)}
        ast::ty_u64 {("u64", size_and_align_of::<u64>(), DW_ATE_unsigned)}
      }}
      ast::ty_float(m) { alt m {
        ast::ty_f {("float", size_and_align_of::<float>(), DW_ATE_float)}
        ast::ty_f32 {("f32", size_and_align_of::<f32>(), DW_ATE_float)}
        ast::ty_f64 {("f64", size_and_align_of::<f64>(), DW_ATE_float)}
      }}
    };

    let fname = filename_from_span(cx, ty.span);
    let file_node = create_file(cx, fname);
    let cu_node = create_compile_unit(cx, fname);
    let lldata = [lltag(tg),
                  cu_node.node,
                  llstr(name),
                  file_node.node,
                  lli32(0), //XXX source line
                  lli64(size * 8),  // size in bits
                  lli64(align * 8), // alignment in bits
                  lli64(0), //XXX offset?
                  lli32(0), //XXX flags?
                  lli32(encoding)];
    let llnode = llmdnode(lldata);
    let mdval = @{node: llnode, data: {hash: t}};
    update_cache(cache, tg, tydesc_metadata(mdval));
    add_named_metadata(cx, "llvm.dbg.ty", llnode);
    ret mdval;
}

fn create_pointer_type(cx: @crate_ctxt, t: ty::t, span: span,
                       pointee: @metadata<tydesc_md>)
    -> @metadata<tydesc_md> {
    let tg = PointerTypeTag;
    /*let cache = cx.llmetadata;
    alt cached_metadata::<@metadata<tydesc_md>>(
        cache, tg, {|md| ty::hash_ty(t) == ty::hash_ty(md.data.hash)}) {
      option::some(md) { ret md; }
      option::none {}
    }*/
    let (size, align) = size_and_align_of::<ctypes::intptr_t>();
    let fname = filename_from_span(cx, span);
    let file_node = create_file(cx, fname);
    //let cu_node = create_compile_unit(cx, fname);
    let llnode = create_derived_type(tg, file_node.node, "", 0, size * 8,
                                     align * 8, 0, pointee.node);
    let mdval = @{node: llnode, data: {hash: t}};
    //update_cache(cache, tg, tydesc_metadata(mdval));
    add_named_metadata(cx, "llvm.dbg.ty", llnode);
    ret mdval;
}

type struct_ctxt = {
    file: ValueRef,
    name: str,
    line: int,
    mutable members: [ValueRef],
    mutable total_size: int,
    align: int
};

fn finish_structure(cx: @struct_ctxt) -> ValueRef {
    ret create_composite_type(StructureTypeTag, cx.name, cx.file, cx.line,
                              cx.total_size, cx.align, 0, option::none,
                              option::some(cx.members));
}

fn create_structure(file: @metadata<file_md>, name: str, line: int)
    -> @struct_ctxt {
    let cx = @{file: file.node,
               name: name,
               line: line,
               mutable members: [],
               mutable total_size: 0,
               align: 64 //XXX different alignment per arch?
              };
    ret cx;
}

fn create_derived_type(type_tag: int, file: ValueRef, name: str, line: int,
                       size: int, align: int, offset: int, ty: ValueRef)
    -> ValueRef {
    let lldata = [lltag(type_tag),
                  file,
                  llstr(name),
                  file,
                  lli32(line),
                  lli64(size),
                  lli64(align),
                  lli64(offset),
                  lli32(0),
                  ty];
    ret llmdnode(lldata);
}

fn add_member(cx: @struct_ctxt, name: str, line: int, size: int, align: int,
              ty: ValueRef) {
    cx.members += [create_derived_type(MemberTag, cx.file, name, line,
                                       size * 8, align * 8, cx.total_size,
                                       ty)];
    cx.total_size += size * 8;
}

fn create_record(cx: @crate_ctxt, t: ty::t, fields: [ast::ty_field],
                 span: span) -> @metadata<tydesc_md> {
    let fname = filename_from_span(cx, span);
    let file_node = create_file(cx, fname);
    let scx = create_structure(file_node,
                               option::get(cx.dbg_cx).names("rec"),
                               line_from_span(cx.sess.codemap,
                                              span) as int);
    for field in fields {
        let field_t = ty::get_field(ccx_tcx(cx), t, field.node.ident).mt.ty;
        let ty_md = create_ty(cx, field_t, field.node.mt.ty);
        let (size, align) = member_size_and_align(field.node.mt.ty);
        add_member(scx, field.node.ident,
                   line_from_span(cx.sess.codemap, field.span) as int,
                   size as int, align as int, ty_md.node);
    }
    let mdval = @{node: finish_structure(scx), data:{hash: t}};
    ret mdval;
}

fn create_boxed_type(cx: @crate_ctxt, outer: ty::t, _inner: ty::t,
                     span: span, boxed: @metadata<tydesc_md>)
    -> @metadata<tydesc_md> {
    //let tg = StructureTypeTag;
    /*let cache = cx.llmetadata;
    alt cached_metadata::<@metadata<tydesc_md>>(
        cache, tg, {|md| ty::hash_ty(outer) == ty::hash_ty(md.data.hash)}) {
      option::some(md) { ret md; }
      option::none {}
    }*/
    let fname = filename_from_span(cx, span);
    let file_node = create_file(cx, fname);
    //let cu_node = create_compile_unit_metadata(cx, fname);
    let tcx = ccx_tcx(cx);
    let uint_t = ty::mk_uint(tcx);
    let uint_ty = @{node: ast::ty_uint(ast::ty_u), span: span};
    let refcount_type = create_basic_type(cx, uint_t, uint_ty);
    let scx = create_structure(file_node, ty_to_str(ccx_tcx(cx), outer), 0);
    add_member(scx, "refcnt", 0, sys::size_of::<uint>() as int,
               sys::align_of::<uint>() as int, refcount_type.node);
    add_member(scx, "boxed", 0, 8, //XXX member_size_and_align(??)
               8, //XXX just a guess
               boxed.node);
    let llnode = finish_structure(scx);
    let mdval = @{node: llnode, data: {hash: outer}};
    //update_cache(cache, tg, tydesc_metadata(mdval));
    add_named_metadata(cx, "llvm.dbg.ty", llnode);
    ret mdval;
}

fn create_composite_type(type_tag: int, name: str, file: ValueRef, line: int,
                         size: int, align: int, offset: int,
                         derived: option::t<ValueRef>,
                         members: option::t<[ValueRef]>)
    -> ValueRef {
    let lldata = [lltag(type_tag),
                  file,
                  llstr(name), // type name
                  file, // source file definition
                  lli32(line), // source line definition
                  lli64(size), // size of members
                  lli64(align), // align
                  lli64(offset), // offset
                  lli32(0), // flags
                  option::is_none(derived) ? llnull() : // derived from
                                             option::get(derived),
                  option::is_none(members) ? llnull() : // members
                                             llmdnode(option::get(members)),
                  lli32(0),  // runtime language
                  llnull()
                 ];
    ret llmdnode(lldata);
}

fn create_vec(cx: @crate_ctxt, vec_t: ty::t, elem_t: ty::t, vec_ty: @ast::ty)
    -> @metadata<tydesc_md> {
    let fname = filename_from_span(cx, vec_ty.span);
    let file_node = create_file(cx, fname);
    let elem_ty = alt vec_ty.node { ast::ty_vec(mt) { mt.ty } };
    let elem_ty_md = create_ty(cx, elem_t, elem_ty);
    let tcx = ccx_tcx(cx);
    let scx = create_structure(file_node, ty_to_str(tcx, vec_t), 0);
    let uint_ty = @{node: ast::ty_uint(ast::ty_u), span: vec_ty.span};
    let size_t_type = create_basic_type(cx, ty::mk_uint(tcx), uint_ty);
    add_member(scx, "fill", 0, sys::size_of::<ctypes::size_t>() as int,
               sys::align_of::<ctypes::size_t>() as int, size_t_type.node);
    add_member(scx, "alloc", 0, sys::size_of::<ctypes::size_t>() as int,
               sys::align_of::<ctypes::size_t>() as int, size_t_type.node);
    let subrange = llmdnode([lltag(SubrangeTag), lli64(0), lli64(0)]);
    let (arr_size, arr_align) = member_size_and_align(elem_ty);
    let data_ptr = create_composite_type(ArrayTypeTag, "", file_node.node, 0,
                                         arr_size, arr_align, 0,
                                         option::some(elem_ty_md.node),
                                         option::some([subrange]));
    add_member(scx, "data", 0, 0, // clang says the size should be 0
               sys::align_of::<u8>() as int, data_ptr);
    let llnode = finish_structure(scx);
    ret @{node: llnode, data: {hash: vec_t}};
}

fn member_size_and_align(ty: @ast::ty) -> (int, int) {
    alt ty.node {
      ast::ty_bool { size_and_align_of::<bool>() }
      ast::ty_int(m) { alt m {
        ast::ty_char { size_and_align_of::<char>() }
        ast::ty_i { size_and_align_of::<int>() }
        ast::ty_i8 { size_and_align_of::<i8>() }
        ast::ty_i16 { size_and_align_of::<i16>() }
        ast::ty_i32 { size_and_align_of::<i32>() }
      }}
      ast::ty_uint(m) { alt m {
        ast::ty_u { size_and_align_of::<uint>() }
        ast::ty_u8 { size_and_align_of::<i8>() }
        ast::ty_u16 { size_and_align_of::<u16>() }
        ast::ty_u32 { size_and_align_of::<u32>() }
      }}
      ast::ty_float(m) { alt m {
        ast::ty_f { size_and_align_of::<float>() }
        ast::ty_f32 { size_and_align_of::<f32>() }
        ast::ty_f64 { size_and_align_of::<f64>() }
      }}
      ast::ty_box(_) | ast::ty_uniq(_) {
        size_and_align_of::<ctypes::uintptr_t>()
      }
      ast::ty_rec(fields) {
        let total_size = 0;
        for field in fields {
            let (size, _) = member_size_and_align(field.node.mt.ty);
            total_size += size;
        }
        (total_size, 64) //XXX different align for other arches?
      }
      ast::ty_vec(_) {
        size_and_align_of::<ctypes::uintptr_t>()
      }
    }
}

fn create_ty(cx: @crate_ctxt, t: ty::t, ty: @ast::ty)
    -> @metadata<tydesc_md> {
    /*let cache = get_cache(cx);
    alt cached_metadata::<@metadata<tydesc_md>>(
        cache, tg, {|md| t == md.data.hash}) {
      option::some(md) { ret md; }
      option::none {}
    }*/

    fn t_to_ty(cx: @crate_ctxt, t: ty::t, span: span) -> @ast::ty {
        let ty = alt ty::struct(ccx_tcx(cx), t) {
          ty::ty_nil { ast::ty_nil }
          ty::ty_bot { ast::ty_bot }
          ty::ty_bool { ast::ty_bool }
          ty::ty_int(t) { ast::ty_int(t) }
          ty::ty_float(t) { ast::ty_float(t) }
          ty::ty_uint(t) { ast::ty_uint(t) }
          ty::ty_box(mt) { ast::ty_box({ty: t_to_ty(cx, mt.ty, span),
                                        mut: mt.mut}) }
          ty::ty_uniq(mt) { ast::ty_uniq({ty: t_to_ty(cx, mt.ty, span),
                                          mut: mt.mut}) }
          ty::ty_rec(fields) {
            let fs = [];
            for field in fields {
                fs += [{node: {ident: field.ident,
                               mt: {ty: t_to_ty(cx, field.mt.ty, span),
                                    mut: field.mt.mut}},
                        span: span}];
            }
            ast::ty_rec(fs)
          }
          ty::ty_vec(mt) { ast::ty_vec({ty: t_to_ty(cx, mt.ty, span),
                                        mut: mt.mut}) }
        };
        ret @{node: ty, span: span};
    }

    alt ty.node {
      ast::ty_box(mt) {
        let inner_t = alt ty::struct(ccx_tcx(cx), t) {
          ty::ty_box(boxed) { boxed.ty }
        };
        let md = create_ty(cx, inner_t, mt.ty);
        let box = create_boxed_type(cx, t, inner_t, ty.span, md);
        ret create_pointer_type(cx, t, ty.span, box);
      }

      ast::ty_uniq(mt) {
        let inner_t = alt ty::struct(ccx_tcx(cx), t) {
          ty::ty_uniq(boxed) { boxed.ty }
        };
        let md = create_ty(cx, inner_t, mt.ty);
        ret create_pointer_type(cx, t, ty.span, md);
      }

      ast::ty_infer {
        let inferred = t_to_ty(cx, t, ty.span);
        ret create_ty(cx, t, inferred);
      }

      ast::ty_rec(fields) {
        ret create_record(cx, t, fields, ty.span);
      }

      ast::ty_vec(mt) {
        let inner_t = ty::sequence_element_type(ccx_tcx(cx), t);
        let v = create_vec(cx, t, inner_t, ty);
        ret create_pointer_type(cx, t, ty.span, v);
      }

      _ { ret create_basic_type(cx, t, ty); }
    };
}

fn filename_from_span(cx: @crate_ctxt, sp: span) -> str {
    codemap::lookup_char_pos(cx.sess.codemap, sp.lo).filename
}

fn create_var(type_tag: int, context: ValueRef, name: str, file: ValueRef,
              line: int, ret_ty: ValueRef) -> ValueRef {
    let lldata = [lltag(type_tag),
                  context,
                  llstr(name),
                  file,
                  lli32(line),
                  ret_ty,
                  lli32(0)
                 ];
    ret llmdnode(lldata);
}

fn create_local_var(bcx: @block_ctxt, local: @ast::local)
    -> @metadata<local_var_md> unsafe {
    let cx = bcx_ccx(bcx);
    let cache = get_cache(cx);
    let tg = AutoVariableTag;
    alt cached_metadata::<@metadata<local_var_md>>(
        cache, tg, {|md| md.data.id == local.node.id}) {
      option::some(md) { ret md; }
      option::none {}
    }

    let name = path_to_ident(alt pat_util::normalize_pat(bcx_tcx(bcx),
                                           local.node.pat).node {
      ast::pat_ident(ident, _) { ident /*XXX deal w/ optional node binding*/ }
    });
    let loc = codemap::lookup_char_pos(cx.sess.codemap,
                                       local.span.lo);
    let ty = base::node_id_type(cx, local.node.id);
    let tymd = create_ty(cx, ty, local.node.ty);
    let filemd = create_file(cx, loc.filename);
    let context = alt bcx.parent {
      parent_none { create_function(bcx.fcx, local.span).node }
      parent_some(_) { create_block(bcx, local.span).node }
    };
    let mdnode = create_var(tg, context, name, filemd.node,
                            loc.line as int, tymd.node);
    let mdval = @{node: mdnode, data: {id: local.node.id}};
    update_cache(cache, AutoVariableTag, local_var_metadata(mdval));

    let llptr = alt bcx.fcx.lllocals.find(local.node.id) {
      option::some(local_mem(v)) { v }
      option::none {
        alt bcx.fcx.lllocals.get(local.node.pat.id) {
          local_imm(v) { v }
        }
      }
    };
    let declargs = [llmdnode([llptr]), mdnode];
    trans::build::Call(bcx, cx.intrinsics.get("llvm.dbg.declare"),
                       declargs);
    ret mdval;
}

fn create_arg(bcx: @block_ctxt, arg: ast::arg, sp: span)
    -> @metadata<argument_md> unsafe {
    let fcx = bcx_fcx(bcx);
    let cx = fcx_ccx(fcx);
    let cache = get_cache(cx);
    let tg = ArgVariableTag;
    alt cached_metadata::<@metadata<argument_md>>(
        cache, ArgVariableTag, {|md| md.data.id == arg.id}) {
      option::some(md) { ret md; }
      option::none {}
    }

    /*let arg_n = alt cx.ast_map.get(arg.id) {
      ast_map::node_arg(_, n) { n - 2u }
    };*/
    let loc = codemap::lookup_char_pos(cx.sess.codemap,
                                       sp.lo);
    let ty = base::node_id_type(cx, arg.id);
    let tymd = create_ty(cx, ty, arg.ty);
    let filemd = create_file(cx, loc.filename);
    let context = create_function(bcx.fcx, sp);
    let mdnode = create_var(tg, context.node, arg.ident, filemd.node,
                            loc.line as int, tymd.node);
    let mdval = @{node: mdnode, data: {id: arg.id}};
    update_cache(cache, tg, argument_metadata(mdval));

    let llptr = alt fcx.llargs.get(arg.id) {
      local_mem(v) | local_imm(v) { v }
    };
    let declargs = [llmdnode([llptr]), mdnode];
    trans::build::Call(bcx, cx.intrinsics.get("llvm.dbg.declare"),
                       declargs);
    ret mdval;
}

fn update_source_pos(cx: @block_ctxt, s: span) {
    if !bcx_ccx(cx).sess.opts.debuginfo {
        ret;
    }
    let cm = bcx_ccx(cx).sess.codemap;
    let blockmd = create_block(cx, s);
    let loc = codemap::lookup_char_pos(cm, s.lo);
    let scopedata = [lli32(loc.line as int),
                     lli32(loc.col as int),
                     blockmd.node,
                     llnull()];
    let dbgscope = llmdnode(scopedata);
    llvm::LLVMSetCurrentDebugLocation(trans::build::B(cx), dbgscope);
}

fn create_function(fcx: @fn_ctxt, sp: span) -> @metadata<subprogram_md> {
    let cx = fcx_ccx(fcx);
    let dbg_cx = option::get(cx.dbg_cx);

    #debug("~~");
    log(debug, fcx.id);

    log(debug, codemap::span_to_str(sp, cx.sess.codemap));

    let (ident, ret_ty, id) = alt cx.ast_map.get(fcx.id) {
      ast_map::node_item(item) {
        alt item.node {
          ast::item_fn(decl, _, _) | ast::item_res(decl, _, _, _, _) {
            (item.ident, decl.output, item.id)
          }
        }
      }
      ast_map::node_method(method) {
          (method.ident, method.decl.output, method.id)
      }
      ast_map::node_res_ctor(item) {
        alt item.node { ast::item_res(decl, _, _, _, ctor_id) {
          (item.ident, decl.output, ctor_id)
        }}
      }
      ast_map::node_expr(expr) {
        alt expr.node {
          ast::expr_fn(_, decl, _, _) {
            (dbg_cx.names("fn"), decl.output, expr.id)
          }
          ast::expr_fn_block(decl, _) {
            (dbg_cx.names("fn"), decl.output, expr.id)
          }
        }
      }
    };

    log(debug, ident);
    log(debug, id);

    let path = str::connect(fcx.lcx.path + [ident], "::");

    let cache = get_cache(cx);
    alt cached_metadata::<@metadata<subprogram_md>>(
        cache, SubprogramTag, {|md| md.data.path == path &&
                                    /*md.data.path == ??*/ true}) {
      option::some(md) { ret md; }
      option::none {}
    }

    let loc = codemap::lookup_char_pos(cx.sess.codemap,
                                       sp.lo);
    let file_node = create_file(cx, loc.filename).node;
    let key = cx.item_symbols.contains_key(fcx.id) ? fcx.id : id;
    let mangled = cx.item_symbols.get(key);
    let ty_node = if cx.sess.opts.extra_debuginfo {
        alt ret_ty.node {
          ast::ty_nil { llnull() }
          _ { create_ty(cx, ty::node_id_to_type(ccx_tcx(cx), id),
                        ret_ty).node }
        }
    } else {
        llnull()
    };
    let sub_node = create_composite_type(SubroutineTag, "", file_node, 0, 0,
                                         0, 0, option::none,
                                         option::some([ty_node]));

    let fn_metadata = [lltag(SubprogramTag),
                       llunused(),
                       file_node,
                       llstr(ident),
                       llstr(path), //XXX fully-qualified C++ name
                       llstr(mangled), //XXX MIPS name?????
                       file_node,
                       lli32(loc.line as int),
                       sub_node,
                       lli1(false), //XXX static (check export)
                       lli1(true), // not extern
                       lli32(DW_VIRTUALITY_none), // virtual-ness
                       lli32(0i), //index into virt func
                       llnull(), // base type with vtbl
                       lli1(false), // artificial
                       lli1(cx.sess.opts.optimize != 0u),
                       fcx.llfn
                       //list of template params
                       //func decl descriptor
                       //list of func vars
                      ];
    let val = llmdnode(fn_metadata);
    add_named_metadata(cx, "llvm.dbg.sp", val);
    let mdval = @{node: val, data: {path: path}};
    update_cache(cache, SubprogramTag, subprogram_metadata(mdval));
    ret mdval;
}
