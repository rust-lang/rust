// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use driver::session;
use lib::llvm::ValueRef;
use lib::llvm::llvm;
use middle::trans::common::*;
use middle::trans::machine;
use middle::trans::type_of;
use middle::trans;
use middle::ty;
use util::ppaux::ty_to_str;

use core::cast;
use core::hashmap::HashMap;
use core::libc;
use core::option;
use core::ptr;
use core::str;
use core::sys;
use core::vec;
use syntax::codemap::span;
use syntax::{ast, codemap, ast_util, ast_map};

static LLVMDebugVersion: int = (9 << 16);

static DW_LANG_RUST: int = 0x9000;
static DW_VIRTUALITY_none: int = 0;

static CompileUnitTag: int = 17;
static FileDescriptorTag: int = 41;
static SubprogramTag: int = 46;
static SubroutineTag: int = 21;
static BasicTypeDescriptorTag: int = 36;
static AutoVariableTag: int = 256;
static ArgVariableTag: int = 257;
static ReturnVariableTag: int = 258;
static LexicalBlockTag: int = 11;
static PointerTypeTag: int = 15;
static StructureTypeTag: int = 19;
static MemberTag: int = 13;
static ArrayTypeTag: int = 1;
static SubrangeTag: int = 33;

static DW_ATE_boolean: int = 0x02;
static DW_ATE_float: int = 0x04;
static DW_ATE_signed: int = 0x05;
static DW_ATE_signed_char: int = 0x06;
static DW_ATE_unsigned: int = 0x07;
static DW_ATE_unsigned_char: int = 0x08;

fn llstr(s: &str) -> ValueRef {
    do str::as_c_str(s) |sbuf| {
        unsafe {
            llvm::LLVMMDString(sbuf, s.len() as libc::c_uint)
        }
    }
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
    C_i1(bval)
}
fn llmdnode(elems: &[ValueRef]) -> ValueRef {
    unsafe {
        llvm::LLVMMDNode(vec::raw::to_ptr(elems), elems.len() as libc::c_uint)
    }
}
fn llunused() -> ValueRef {
    lli32(0x0)
}
fn llnull() -> ValueRef {
    unsafe {
        cast::transmute(ptr::null::<ValueRef>())
    }
}

fn add_named_metadata(cx: @CrateContext, name: ~str, val: ValueRef) {
    str::as_c_str(name, |sbuf| {
        unsafe {
            llvm::LLVMAddNamedMetadataOperand(cx.llmod, sbuf, val)
        }
    })
}

////////////////

pub struct DebugContext {
    llmetadata: metadata_cache,
    names: namegen,
    crate_file: ~str
}

pub fn mk_ctxt(crate: ~str) -> DebugContext {
    DebugContext {
        llmetadata: @mut HashMap::new(),
        names: new_namegen(),
        crate_file: crate
    }
}

fn update_cache(cache: metadata_cache, mdtag: int, val: debug_metadata) {
    let mut existing = match cache.pop(&mdtag) {
        Some(arr) => arr, None => ~[]
    };
    existing.push(val);
    cache.insert(mdtag, existing);
}

struct Metadata<T> {
    node: ValueRef,
    data: T
}

struct FileMetadata {
    path: ~str
}
struct CompileUnitMetadata {
    name: ~str
}
struct SubProgramMetadata {
    id: ast::node_id
}
struct LocalVarMetadata {
    id: ast::node_id
}
struct TyDescMetadata {
    hash: uint
}
struct BlockMetadata {
    start: codemap::Loc,
    end: codemap::Loc
}
struct ArgumentMetadata {
    id: ast::node_id
}
struct RetvalMetadata {
    id: ast::node_id
}

type metadata_cache = @mut HashMap<int, ~[debug_metadata]>;

enum debug_metadata {
    file_metadata(@Metadata<FileMetadata>),
    compile_unit_metadata(@Metadata<CompileUnitMetadata>),
    subprogram_metadata(@Metadata<SubProgramMetadata>),
    local_var_metadata(@Metadata<LocalVarMetadata>),
    tydesc_metadata(@Metadata<TyDescMetadata>),
    block_metadata(@Metadata<BlockMetadata>),
    argument_metadata(@Metadata<ArgumentMetadata>),
    retval_metadata(@Metadata<RetvalMetadata>),
}

fn cast_safely<T:Copy,U>(val: T) -> U {
    unsafe {
        let val2 = val;
        return cast::transmute(val2);
    }
}

fn md_from_metadata<T>(val: debug_metadata) -> T {
    match val {
      file_metadata(md) => cast_safely(md),
      compile_unit_metadata(md) => cast_safely(md),
      subprogram_metadata(md) => cast_safely(md),
      local_var_metadata(md) => cast_safely(md),
      tydesc_metadata(md) => cast_safely(md),
      block_metadata(md) => cast_safely(md),
      argument_metadata(md) => cast_safely(md),
      retval_metadata(md) => cast_safely(md)
    }
}

fn cached_metadata<T:Copy>(cache: metadata_cache,
                            mdtag: int,
                            eq_fn: &fn(md: T) -> bool)
                         -> Option<T> {
    if cache.contains_key(&mdtag) {
        let items = cache.get(&mdtag);
        for items.each |item| {
            let md: T = md_from_metadata::<T>(*item);
            if eq_fn(md) {
                return option::Some(md);
            }
        }
    }
    return option::None;
}

fn create_compile_unit(cx: @CrateContext) -> @Metadata<CompileUnitMetadata> {
    let cache = get_cache(cx);
    let crate_name = /*bad*/copy (/*bad*/copy cx.dbg_cx).get().crate_file;
    let tg = CompileUnitTag;
    match cached_metadata::<@Metadata<CompileUnitMetadata>>(cache, tg,
                        |md| md.data.name == crate_name) {
      option::Some(md) => return md,
      option::None => ()
    }

    let (_, work_dir) = get_file_path_and_dir(
        cx.sess.working_dir.to_str(), crate_name);
    let unit_metadata = ~[lltag(tg),
                         llunused(),
                         lli32(DW_LANG_RUST),
                         llstr(crate_name),
                         llstr(work_dir),
                         llstr(env!("CFG_VERSION")),
                         lli1(true), // deprecated: main compile unit
                         lli1(cx.sess.opts.optimize != session::No),
                         llstr(""), // flags (???)
                         lli32(0) // runtime version (???)
                        ];
    let unit_node = llmdnode(unit_metadata);
    add_named_metadata(cx, ~"llvm.dbg.cu", unit_node);
    let mdval = @Metadata {
        node: unit_node,
        data: CompileUnitMetadata {
            name: crate_name
        }
    };
    update_cache(cache, tg, compile_unit_metadata(mdval));

    return mdval;
}

fn get_cache(cx: @CrateContext) -> metadata_cache {
    (/*bad*/copy cx.dbg_cx).get().llmetadata
}

fn get_file_path_and_dir(work_dir: &str, full_path: &str) -> (~str, ~str) {
    (if full_path.starts_with(work_dir) {
        full_path.slice(work_dir.len() + 1u,
                   full_path.len()).to_owned()
    } else {
        full_path.to_owned()
    }, work_dir.to_owned())
}

fn create_file(cx: @CrateContext, full_path: ~str)
    -> @Metadata<FileMetadata> {
    let cache = get_cache(cx);;
    let tg = FileDescriptorTag;
    match cached_metadata::<@Metadata<FileMetadata>>(
        cache, tg, |md| md.data.path == full_path) {
        option::Some(md) => return md,
        option::None => ()
    }

    let (file_path, work_dir) =
        get_file_path_and_dir(cx.sess.working_dir.to_str(),
                              full_path);
    let unit_node = create_compile_unit(cx).node;
    let file_md = ~[lltag(tg),
                   llstr(file_path),
                   llstr(work_dir),
                   unit_node];
    let val = llmdnode(file_md);
    let mdval = @Metadata {
        node: val,
        data: FileMetadata {
            path: full_path
        }
    };
    update_cache(cache, tg, file_metadata(mdval));
    return mdval;
}

fn line_from_span(cm: @codemap::CodeMap, sp: span) -> uint {
    cm.lookup_char_pos(sp.lo).line
}

fn create_block(cx: block) -> @Metadata<BlockMetadata> {
    let cache = get_cache(cx.ccx());
    let mut cx = cx;
    while cx.node_info.is_none() {
        match cx.parent {
          Some(b) => cx = b,
          None => fail!()
        }
    }
    let sp = cx.node_info.get().span;

    let start = cx.sess().codemap.lookup_char_pos(sp.lo);
    let fname = /*bad*/copy start.file.name;
    let end = cx.sess().codemap.lookup_char_pos(sp.hi);
    let tg = LexicalBlockTag;
    /*match cached_metadata::<@Metadata<BlockMetadata>>(
        cache, tg,
        {|md| start == md.data.start && end == md.data.end}) {
      option::Some(md) { return md; }
      option::None {}
    }*/

    let parent = match cx.parent {
        None => create_function(cx.fcx).node,
        Some(bcx) => create_block(bcx).node
    };
    let file_node = create_file(cx.ccx(), fname);
    let unique_id = match cache.find(&LexicalBlockTag) {
      option::Some(v) => v.len() as int,
      option::None => 0
    };
    let lldata = ~[lltag(tg),
                  parent,
                  lli32(start.line.to_int()),
                  lli32(start.col.to_int()),
                  file_node.node,
                  lli32(unique_id)
                 ];
    let val = llmdnode(lldata);
    let mdval = @Metadata {
        node: val,
        data: BlockMetadata {
            start: start,
            end: end
        }
    };
    //update_cache(cache, tg, block_metadata(mdval));
    return mdval;
}

fn size_and_align_of(cx: @CrateContext, t: ty::t) -> (int, int) {
    let llty = type_of::type_of(cx, t);
    (machine::llsize_of_real(cx, llty) as int,
     machine::llalign_of_pref(cx, llty) as int)
}

fn create_basic_type(cx: @CrateContext, t: ty::t, span: span)
    -> @Metadata<TyDescMetadata> {
    let cache = get_cache(cx);
    let tg = BasicTypeDescriptorTag;
    match cached_metadata::<@Metadata<TyDescMetadata>>(
        cache, tg, |md| ty::type_id(t) == md.data.hash) {
      option::Some(md) => return md,
      option::None => ()
    }

    let (name, encoding) = match ty::get(t).sty {
        ty::ty_nil | ty::ty_bot => (~"uint", DW_ATE_unsigned),
        ty::ty_bool => (~"bool", DW_ATE_boolean),
        ty::ty_int(int_ty) => match int_ty {
            ast::ty_i => (~"int", DW_ATE_signed),
            ast::ty_char => (~"char", DW_ATE_signed_char),
            ast::ty_i8 => (~"i8", DW_ATE_signed),
            ast::ty_i16 => (~"i16", DW_ATE_signed),
            ast::ty_i32 => (~"i32", DW_ATE_signed),
            ast::ty_i64 => (~"i64", DW_ATE_signed)
        },
        ty::ty_uint(uint_ty) => match uint_ty {
            ast::ty_u => (~"uint", DW_ATE_unsigned),
            ast::ty_u8 => (~"u8", DW_ATE_unsigned),
            ast::ty_u16 => (~"u16", DW_ATE_unsigned),
            ast::ty_u32 => (~"u32", DW_ATE_unsigned),
            ast::ty_u64 => (~"u64", DW_ATE_unsigned)
        },
        ty::ty_float(float_ty) => match float_ty {
            ast::ty_f => (~"float", DW_ATE_float),
            ast::ty_f32 => (~"f32", DW_ATE_float),
            ast::ty_f64 => (~"f64", DW_ATE_float)
        },
        _ => cx.sess.bug("debuginfo::create_basic_type - t is invalid type")
    };

    let fname = filename_from_span(cx, span);
    let file_node = create_file(cx, fname);
    let cu_node = create_compile_unit(cx);
    let (size, align) = size_and_align_of(cx, t);
    let lldata = ~[lltag(tg),
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
    let mdval = @Metadata {
        node: llnode,
        data: TyDescMetadata {
            hash: ty::type_id(t)
        }
    };
    update_cache(cache, tg, tydesc_metadata(mdval));
    add_named_metadata(cx, ~"llvm.dbg.ty", llnode);
    return mdval;
}

fn create_pointer_type(cx: @CrateContext, t: ty::t, span: span,
                       pointee: @Metadata<TyDescMetadata>)
    -> @Metadata<TyDescMetadata> {
    let tg = PointerTypeTag;
    /*let cache = cx.llmetadata;
    match cached_metadata::<@Metadata<TyDescMetadata>>(
        cache, tg, {|md| ty::hash_ty(t) == ty::hash_ty(md.data.hash)}) {
      option::Some(md) { return md; }
      option::None {}
    }*/
    let (size, align) = size_and_align_of(cx, t);
    let fname = filename_from_span(cx, span);
    let file_node = create_file(cx, fname);
    //let cu_node = create_compile_unit(cx, fname);
    let name = ty_to_str(cx.tcx, t);
    let llnode = create_derived_type(tg, file_node.node, name, 0, size * 8,
                                     align * 8, 0, pointee.node);
    let mdval = @Metadata {
        node: llnode,
        data: TyDescMetadata {
            hash: ty::type_id(t)
        }
    };
    //update_cache(cache, tg, tydesc_metadata(mdval));
    add_named_metadata(cx, ~"llvm.dbg.ty", llnode);
    return mdval;
}

struct StructCtxt {
    file: ValueRef,
    name: @~str,
    line: int,
    members: ~[ValueRef],
    total_size: int,
    align: int
}

fn finish_structure(cx: @mut StructCtxt) -> ValueRef {
    return create_composite_type(StructureTypeTag,
                                 *cx.name,
                                 cx.file,
                                 cx.line,
                                 cx.total_size,
                                 cx.align,
                                 0,
                                 None,
                                 Some(/*bad*/copy cx.members));
}

fn create_structure(file: @Metadata<FileMetadata>, name: @~str, line: int)
                 -> @mut StructCtxt {
    let cx = @mut StructCtxt {
        file: file.node,
        name: name,
        line: line,
        members: ~[],
        total_size: 0,
        align: 64 //XXX different alignment per arch?
    };
    return cx;
}

fn create_derived_type(type_tag: int, file: ValueRef, name: &str, line: int,
                       size: int, align: int, offset: int, ty: ValueRef)
    -> ValueRef {
    let lldata = ~[lltag(type_tag),
                  file,
                  llstr(name),
                  file,
                  lli32(line),
                  lli64(size),
                  lli64(align),
                  lli64(offset),
                  lli32(0),
                  ty];
    return llmdnode(lldata);
}

fn add_member(cx: @mut StructCtxt,
              name: &str,
              line: int,
              size: int,
              align: int,
              ty: ValueRef) {
    cx.members.push(create_derived_type(MemberTag, cx.file, name, line,
                                        size * 8, align * 8, cx.total_size,
                                        ty));
    cx.total_size += size * 8;
}

fn create_struct(cx: @CrateContext, t: ty::t, fields: ~[ty::field],
                 span: span) -> @Metadata<TyDescMetadata> {
    let fname = filename_from_span(cx, span);
    let file_node = create_file(cx, fname);
    let scx = create_structure(file_node, @ty_to_str(cx.tcx, t),
                               line_from_span(cx.sess.codemap, span) as int);
    for fields.each |field| {
        let field_t = field.mt.ty;
        let ty_md = create_ty(cx, field_t, span);
        let (size, align) = size_and_align_of(cx, field_t);
        add_member(scx, *cx.sess.str_of(field.ident),
                   line_from_span(cx.sess.codemap, span) as int,
                   size as int, align as int, ty_md.node);
    }
    let mdval = @Metadata {
        node: finish_structure(scx),
        data: TyDescMetadata {
            hash: ty::type_id(t)
        }
    };
    return mdval;
}

fn create_tuple(cx: @CrateContext, t: ty::t, elements: &[ty::t], span: span)
    -> @Metadata<TyDescMetadata> {
    let fname = filename_from_span(cx, span);
    let file_node = create_file(cx, fname);
    let scx = create_structure(file_node,
                               cx.sess.str_of(
                                   ((/*bad*/copy cx.dbg_cx).get().names)
                                   ("tuple")),
                               line_from_span(cx.sess.codemap, span) as int);
    for elements.each |element| {
        let ty_md = create_ty(cx, *element, span);
        let (size, align) = size_and_align_of(cx, *element);
        add_member(scx, "", line_from_span(cx.sess.codemap, span) as int,
                   size as int, align as int, ty_md.node);
    }
    let mdval = @Metadata {
        node: finish_structure(scx),
        data: TyDescMetadata {
            hash: ty::type_id(t)
        }
    };
    return mdval;
}

// returns (void* type as a ValueRef, size in bytes, align in bytes)
fn voidptr() -> (ValueRef, int, int) {
    let null = ptr::null();
    let size = sys::size_of::<ValueRef>() as int;
    let align = sys::min_align_of::<ValueRef>() as int;
    let vp = create_derived_type(PointerTypeTag, null, "", 0,
                                 size, align, 0, null);
    return (vp, size, align);
}

fn create_boxed_type(cx: @CrateContext, contents: ty::t,
                     span: span, boxed: @Metadata<TyDescMetadata>)
    -> @Metadata<TyDescMetadata> {
    //let tg = StructureTypeTag;
    /*let cache = cx.llmetadata;
    match cached_metadata::<@Metadata<TyDescMetadata>>(
        cache, tg, {|md| ty::hash_ty(contents) == ty::hash_ty(md.data.hash)}) {
      option::Some(md) { return md; }
      option::None {}
    }*/
    let fname = filename_from_span(cx, span);
    let file_node = create_file(cx, fname);
    //let cu_node = create_compile_unit_metadata(cx, fname);
    let int_t = ty::mk_int();
    let refcount_type = create_basic_type(cx, int_t, span);
    let name = ty_to_str(cx.tcx, contents);
    let scx = create_structure(file_node, @fmt!("box<%s>", name), 0);
    add_member(scx, "refcnt", 0, sys::size_of::<uint>() as int,
               sys::min_align_of::<uint>() as int, refcount_type.node);
    // the tydesc and other pointers should be irrelevant to the
    // debugger, so treat them as void* types
    let (vp, vpsize, vpalign) = voidptr();
    add_member(scx, "tydesc", 0, vpsize, vpalign, vp);
    add_member(scx, "prev", 0, vpsize, vpalign, vp);
    add_member(scx, "next", 0, vpsize, vpalign, vp);
    let (size, align) = size_and_align_of(cx, contents);
    add_member(scx, "boxed", 0, size, align, boxed.node);
    let llnode = finish_structure(scx);
    let mdval = @Metadata {
        node: llnode,
        data: TyDescMetadata {
            hash: ty::type_id(contents)
        }
    };
    //update_cache(cache, tg, tydesc_metadata(mdval));
    add_named_metadata(cx, ~"llvm.dbg.ty", llnode);
    return mdval;
}

fn create_composite_type(type_tag: int, name: &str, file: ValueRef,
                         line: int, size: int, align: int, offset: int,
                         derived: Option<ValueRef>,
                         members: Option<~[ValueRef]>)
    -> ValueRef {
    let lldata = ~[lltag(type_tag),
                  file,
                  llstr(name), // type name
                  file, // source file definition
                  lli32(line), // source line definition
                  lli64(size), // size of members
                  lli64(align), // align
                  lli32/*64*/(offset), // offset
                  lli32(0), // flags
                  if derived.is_none() {
                      llnull()
                  } else { // derived from
                      derived.get()
                  },
                  if members.is_none() {
                      llnull()
                  } else { //members
                      llmdnode(members.get())
                  },
                  lli32(0),  // runtime language
                  llnull()
                 ];
    return llmdnode(lldata);
}

fn create_fixed_vec(cx: @CrateContext, vec_t: ty::t, elem_t: ty::t,
                    len: int, span: span) -> @Metadata<TyDescMetadata> {
    let t_md = create_ty(cx, elem_t, span);
    let fname = filename_from_span(cx, span);
    let file_node = create_file(cx, fname);
    let (size, align) = size_and_align_of(cx, elem_t);
    let subrange = llmdnode([lltag(SubrangeTag), lli64(0), lli64(len - 1)]);
    let name = fmt!("[%s]", ty_to_str(cx.tcx, elem_t));
    let array = create_composite_type(ArrayTypeTag, name, file_node.node, 0,
                                      size * len, align, 0, Some(t_md.node),
                                      Some(~[subrange]));
    @Metadata {
        node: array,
        data: TyDescMetadata {
            hash: ty::type_id(vec_t)
        }
    }
}

fn create_boxed_vec(cx: @CrateContext, vec_t: ty::t, elem_t: ty::t,
                    vec_ty_span: codemap::span)
    -> @Metadata<TyDescMetadata> {
    let fname = filename_from_span(cx, vec_ty_span);
    let file_node = create_file(cx, fname);
    let elem_ty_md = create_ty(cx, elem_t, vec_ty_span);
    let vec_scx = create_structure(file_node,
                               @/*bad*/ copy ty_to_str(cx.tcx, vec_t), 0);
    let size_t_type = create_basic_type(cx, ty::mk_uint(), vec_ty_span);
    add_member(vec_scx, "fill", 0, sys::size_of::<libc::size_t>() as int,
               sys::min_align_of::<libc::size_t>() as int, size_t_type.node);
    add_member(vec_scx, "alloc", 0, sys::size_of::<libc::size_t>() as int,
               sys::min_align_of::<libc::size_t>() as int, size_t_type.node);
    let subrange = llmdnode([lltag(SubrangeTag), lli64(0), lli64(0)]);
    let (arr_size, arr_align) = size_and_align_of(cx, elem_t);
    let name = fmt!("[%s]", ty_to_str(cx.tcx, elem_t));
    let data_ptr = create_composite_type(ArrayTypeTag, name, file_node.node, 0,
                                         arr_size, arr_align, 0,
                                         Some(elem_ty_md.node),
                                         Some(~[subrange]));
    add_member(vec_scx, "data", 0, 0, // clang says the size should be 0
               sys::min_align_of::<u8>() as int, data_ptr);
    let llnode = finish_structure(vec_scx);
    let vec_md = @Metadata {
        node: llnode,
        data: TyDescMetadata {
            hash: ty::type_id(vec_t)
        }
    };

    let box_scx = create_structure(file_node, @fmt!("box<%s>", name), 0);
    let int_t = ty::mk_int();
    let refcount_type = create_basic_type(cx, int_t, vec_ty_span);
    add_member(box_scx, "refcnt", 0, sys::size_of::<uint>() as int,
               sys::min_align_of::<uint>() as int, refcount_type.node);
    let (vp, vpsize, vpalign) = voidptr();
    add_member(box_scx, "tydesc", 0, vpsize, vpalign, vp);
    add_member(box_scx, "prev", 0, vpsize, vpalign, vp);
    add_member(box_scx, "next", 0, vpsize, vpalign, vp);
    let size = 2 * sys::size_of::<int>() as int;
    let align = sys::min_align_of::<int>() as int;
    add_member(box_scx, "boxed", 0, size, align, vec_md.node);
    let llnode = finish_structure(box_scx);
    let mdval = @Metadata {
        node: llnode,
        data: TyDescMetadata {
            hash: ty::type_id(elem_t)
        }
    };
    return mdval;
}

fn create_vec_slice(cx: @CrateContext, vec_t: ty::t, elem_t: ty::t, span: span)
    -> @Metadata<TyDescMetadata> {
    let fname = filename_from_span(cx, span);
    let file_node = create_file(cx, fname);
    let elem_ty_md = create_ty(cx, elem_t, span);
    let uint_type = create_basic_type(cx, ty::mk_uint(), span);
    let elem_ptr = create_pointer_type(cx, elem_t, span, elem_ty_md);
    let scx = create_structure(file_node, @ty_to_str(cx.tcx, vec_t), 0);
    let (_, ptr_size, ptr_align) = voidptr();
    add_member(scx, "vec", 0, ptr_size, ptr_align, elem_ptr.node);
    add_member(scx, "length", 0, sys::size_of::<uint>() as int,
               sys::min_align_of::<uint>() as int, uint_type.node);
    let llnode = finish_structure(scx);
    let mdval = @Metadata {
        node: llnode,
        data: TyDescMetadata {
            hash: ty::type_id(vec_t)
        }
    };
    return mdval;
}

fn create_fn_ty(cx: @CrateContext, fn_ty: ty::t, inputs: ~[ty::t], output: ty::t,
                span: span) -> @Metadata<TyDescMetadata> {
    let fname = filename_from_span(cx, span);
    let file_node = create_file(cx, fname);
    let (vp, _, _) = voidptr();
    let output_md = create_ty(cx, output, span);
    let output_ptr_md = create_pointer_type(cx, output, span, output_md);
    let inputs_vals = do inputs.map |arg| { create_ty(cx, *arg, span).node };
    let members = ~[output_ptr_md.node, vp] + inputs_vals;
    let llnode = create_composite_type(SubroutineTag, "", file_node.node,
                                       0, 0, 0, 0, None, Some(members));
    let mdval = @Metadata {
        node: llnode,
        data: TyDescMetadata {
            hash: ty::type_id(fn_ty)
        }
    };
    return mdval;
}

fn create_ty(cx: @CrateContext, t: ty::t, span: span)
    -> @Metadata<TyDescMetadata> {
    debug!("create_ty: %?", ty::get(t));
    /*let cache = get_cache(cx);
    match cached_metadata::<@Metadata<TyDescMetadata>>(
        cache, tg, {|md| t == md.data.hash}) {
      option::Some(md) { return md; }
      option::None {}
    }*/

    let sty = copy ty::get(t).sty;
    match sty {
        ty::ty_nil | ty::ty_bot | ty::ty_bool | ty::ty_int(_) | ty::ty_uint(_)
        | ty::ty_float(_) => create_basic_type(cx, t, span),
        ty::ty_estr(ref vstore) => {
            let i8_t = ty::mk_i8();
            match *vstore {
                ty::vstore_fixed(len) => {
                    create_fixed_vec(cx, t, i8_t, len as int + 1, span)
                },
                ty::vstore_uniq | ty::vstore_box => {
                    let box_md = create_boxed_vec(cx, t, i8_t, span);
                    create_pointer_type(cx, t, span, box_md)
                }
                ty::vstore_slice(_region) => {
                    create_vec_slice(cx, t, i8_t, span)
                }
            }
        },
        ty::ty_enum(_did, ref _substs) => {
            cx.sess.span_bug(span, "debuginfo for enum NYI")
        }
        ty::ty_box(ref mt) | ty::ty_uniq(ref mt) => {
            let boxed = create_ty(cx, mt.ty, span);
            let box_md = create_boxed_type(cx, mt.ty, span, boxed);
            create_pointer_type(cx, t, span, box_md)
        },
        ty::ty_evec(ref mt, ref vstore) => {
            match *vstore {
                ty::vstore_fixed(len) => {
                    create_fixed_vec(cx, t, mt.ty, len as int, span)
                },
                ty::vstore_uniq | ty::vstore_box => {
                    let box_md = create_boxed_vec(cx, t, mt.ty, span);
                    create_pointer_type(cx, t, span, box_md)
                },
                ty::vstore_slice(_region) => {
                    create_vec_slice(cx, t, mt.ty, span)
                }
            }
        },
        ty::ty_ptr(ref mt) => {
            let pointee = create_ty(cx, mt.ty, span);
            create_pointer_type(cx, t, span, pointee)
        },
        ty::ty_rptr(ref _region, ref _mt) => {
            cx.sess.span_bug(span, "debuginfo for rptr NYI")
        },
        ty::ty_bare_fn(ref barefnty) => {
            let inputs = barefnty.sig.inputs.map(|a| *a);
            let output = barefnty.sig.output;
            create_fn_ty(cx, t, inputs, output, span)
        },
        ty::ty_closure(ref _closurety) => {
            cx.sess.span_bug(span, "debuginfo for closure NYI")
        },
        ty::ty_trait(_did, ref _substs, ref _vstore, _) => {
            cx.sess.span_bug(span, "debuginfo for trait NYI")
        },
        ty::ty_struct(did, ref substs) => {
            let fields = ty::struct_fields(cx.tcx, did, substs);
            create_struct(cx, t, fields, span)
        },
        ty::ty_tup(ref elements) => {
            create_tuple(cx, t, *elements, span)
        },
        _ => cx.sess.bug("debuginfo: unexpected type in create_ty")
    }
}

fn filename_from_span(cx: @CrateContext, sp: codemap::span) -> ~str {
    /*bad*/copy cx.sess.codemap.lookup_char_pos(sp.lo).file.name
}

fn create_var(type_tag: int, context: ValueRef, name: &str, file: ValueRef,
              line: int, ret_ty: ValueRef) -> ValueRef {
    let lldata = ~[lltag(type_tag),
                  context,
                  llstr(name),
                  file,
                  lli32(line),
                  ret_ty,
                  lli32(0)
                 ];
    return llmdnode(lldata);
}

pub fn create_local_var(bcx: block, local: @ast::local)
    -> @Metadata<LocalVarMetadata> {
    let cx = bcx.ccx();
    let cache = get_cache(cx);
    let tg = AutoVariableTag;
    match cached_metadata::<@Metadata<LocalVarMetadata>>(
        cache, tg, |md| md.data.id == local.node.id) {
      option::Some(md) => return md,
      option::None => ()
    }

    let name = match local.node.pat.node {
      ast::pat_ident(_, pth, _) => ast_util::path_to_ident(pth),
      // FIXME this should be handled (#2533)
      _ => fail!("no single variable name for local")
    };
    let loc = cx.sess.codemap.lookup_char_pos(local.span.lo);
    let ty = node_id_type(bcx, local.node.id);
    let tymd = create_ty(cx, ty, local.node.ty.span);
    let filemd = create_file(cx, /*bad*/copy loc.file.name);
    let context = match bcx.parent {
        None => create_function(bcx.fcx).node,
        Some(_) => create_block(bcx).node
    };
    let mdnode = create_var(tg, context, *cx.sess.str_of(name),
                            filemd.node, loc.line as int, tymd.node);
    let mdval = @Metadata {
        node: mdnode,
        data: LocalVarMetadata {
            id: local.node.id
        }
    };
    update_cache(cache, AutoVariableTag, local_var_metadata(mdval));

    // FIXME(#6814) Should use `pat_util::pat_bindings` for pats like (a, b) etc
    let llptr = match bcx.fcx.lllocals.find_copy(&local.node.pat.id) {
        Some(v) => v,
        None => {
            bcx.tcx().sess.span_bug(
                local.span,
                fmt!("No entry in lllocals table for %?", local.node.id));
        }
    };
    let declargs = ~[llmdnode([llptr]), mdnode];
    trans::build::Call(bcx, *cx.intrinsics.get(&~"llvm.dbg.declare"),
                       declargs);
    return mdval;
}

pub fn create_arg(bcx: block, arg: ast::arg, sp: span)
    -> Option<@Metadata<ArgumentMetadata>> {
    let fcx = bcx.fcx;
    let cx = *fcx.ccx;
    let cache = get_cache(cx);
    let tg = ArgVariableTag;
    match cached_metadata::<@Metadata<ArgumentMetadata>>(
        cache, ArgVariableTag, |md| md.data.id == arg.id) {
      option::Some(md) => return Some(md),
      option::None => ()
    }

    let loc = cx.sess.codemap.lookup_char_pos(sp.lo);
    if loc.file.name == ~"<intrinsic>" {
        return None;
    }
    let ty = node_id_type(bcx, arg.id);
    let tymd = create_ty(cx, ty, arg.ty.span);
    let filemd = create_file(cx, /*bad*/copy loc.file.name);
    let context = create_function(bcx.fcx);

    match arg.pat.node {
        ast::pat_ident(_, path, _) => {
            // XXX: This is wrong; it should work for multiple bindings.
            let mdnode = create_var(
                tg,
                context.node,
                *cx.sess.str_of(*path.idents.last()),
                filemd.node,
                loc.line as int,
                tymd.node
            );

            let mdval = @Metadata {
                node: mdnode,
                data: ArgumentMetadata {
                    id: arg.id
                }
            };
            update_cache(cache, tg, argument_metadata(mdval));

            let llptr = fcx.llargs.get_copy(&arg.id);
            let declargs = ~[llmdnode([llptr]), mdnode];
            trans::build::Call(bcx,
                               *cx.intrinsics.get(&~"llvm.dbg.declare"),
                               declargs);
            return Some(mdval);
        }
        _ => {
            return None;
        }
    }
}

pub fn update_source_pos(cx: block, s: span) {
    if !cx.sess().opts.debuginfo || (*s.lo == 0 && *s.hi == 0) {
        return;
    }
    let cm = cx.sess().codemap;
    let blockmd = create_block(cx);
    let loc = cm.lookup_char_pos(s.lo);
    let scopedata = ~[lli32(loc.line.to_int()),
                     lli32(loc.col.to_int()),
                     blockmd.node,
                     llnull()];
    let dbgscope = llmdnode(scopedata);
    unsafe {
        llvm::LLVMSetCurrentDebugLocation(trans::build::B(cx), dbgscope);
    }
}

pub fn create_function(fcx: fn_ctxt) -> @Metadata<SubProgramMetadata> {
    let cx = *fcx.ccx;
    let dbg_cx = (/*bad*/copy cx.dbg_cx).get();

    debug!("~~");

    let fcx = &mut *fcx;

    let sp = fcx.span.get();
    debug!("%s", cx.sess.codemap.span_to_str(sp));

    let (ident, ret_ty, id) = match cx.tcx.items.get_copy(&fcx.id) {
      ast_map::node_item(item, _) => {
        match item.node {
          ast::item_fn(ref decl, _, _, _, _) => {
            (item.ident, decl.output, item.id)
          }
          _ => fcx.ccx.sess.span_bug(item.span, "create_function: item bound to non-function")
        }
      }
      ast_map::node_method(method, _, _) => {
          (method.ident, method.decl.output, method.id)
      }
      ast_map::node_expr(expr) => {
        match expr.node {
          ast::expr_fn_block(ref decl, _) => {
            ((dbg_cx.names)("fn"), decl.output, expr.id)
          }
          _ => fcx.ccx.sess.span_bug(expr.span,
                  "create_function: expected an expr_fn_block here")
        }
      }
      _ => fcx.ccx.sess.bug("create_function: unexpected sort of node")
    };

    debug!("%?", ident);
    debug!("%?", id);

    let cache = get_cache(cx);
    match cached_metadata::<@Metadata<SubProgramMetadata>>(
        cache, SubprogramTag, |md| md.data.id == id) {
      option::Some(md) => return md,
      option::None => ()
    }

    let loc = cx.sess.codemap.lookup_char_pos(sp.lo);
    let file_node = create_file(cx, copy loc.file.name).node;
    let ty_node = if cx.sess.opts.extra_debuginfo {
        match ret_ty.node {
          ast::ty_nil => llnull(),
          _ => create_ty(cx, ty::node_id_to_type(cx.tcx, id),
                         ret_ty.span).node
        }
    } else {
        llnull()
    };
    let sub_node = create_composite_type(SubroutineTag, "", file_node, 0, 0,
                                         0, 0, option::None,
                                         option::Some(~[ty_node]));

    let fn_metadata = ~[lltag(SubprogramTag),
                       llunused(),
                       file_node,
                       llstr(*cx.sess.str_of(ident)),
                        //XXX fully-qualified C++ name:
                       llstr(*cx.sess.str_of(ident)),
                       llstr(""), //XXX MIPS name?????
                       file_node,
                       lli32(loc.line as int),
                       sub_node,
                       lli1(false), //XXX static (check export)
                       lli1(true), // defined in compilation unit
                       lli32(DW_VIRTUALITY_none), // virtual-ness
                       lli32(0i), //index into virt func
                       /*llnull()*/ lli32(0), // base type with vtbl
                       lli32(256), // flags
                       lli1(cx.sess.opts.optimize != session::No),
                       fcx.llfn
                       //list of template params
                       //func decl descriptor
                       //list of func vars
                      ];
    let val = llmdnode(fn_metadata);
    add_named_metadata(cx, ~"llvm.dbg.sp", val);
    let mdval = @Metadata {
        node: val,
        data: SubProgramMetadata {
            id: id
        }
    };
    update_cache(cache, SubprogramTag, subprogram_metadata(mdval));

    return mdval;
}
