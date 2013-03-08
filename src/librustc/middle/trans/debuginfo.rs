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

use core::libc;
use core::option;
use core::sys;
use std::oldmap::HashMap;
use std::oldmap;
use syntax::codemap::{span, CharPos};
use syntax::parse::token::ident_interner;
use syntax::{ast, codemap, ast_util, ast_map};

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
        cast::reinterpret_cast(&ptr::null::<ValueRef>())
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

pub fn mk_ctxt(+crate: ~str, intr: @ident_interner) -> DebugContext {
    DebugContext {
        llmetadata: oldmap::HashMap(),
        names: new_namegen(intr),
        crate_file: crate
    }
}

fn update_cache(cache: metadata_cache, mdtag: int, val: debug_metadata) {
    let existing = if cache.contains_key(&mdtag) {
        cache.get(&mdtag)
    } else {
        ~[]
    };
    cache.insert(mdtag, vec::append_one(existing, val));
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

type metadata_cache = HashMap<int, ~[debug_metadata]>;

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
    unsafe {
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
}

fn cached_metadata<T:Copy>(cache: metadata_cache,
                            mdtag: int,
                            eq_fn: fn(md: T) -> bool)
                         -> Option<T> {
    unsafe {
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
}

fn create_compile_unit(cx: @CrateContext) -> @Metadata<CompileUnitMetadata> {
    unsafe {
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
                             llstr(~""), // flags (???)
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
}

fn get_cache(cx: @CrateContext) -> metadata_cache {
    (/*bad*/copy cx.dbg_cx).get().llmetadata
}

fn get_file_path_and_dir(work_dir: &str, full_path: &str) -> (~str, ~str) {
    (if str::starts_with(full_path, work_dir) {
        str::slice(full_path, str::len(work_dir) + 1u,
                   str::len(full_path))
    } else {
        str::from_slice(full_path)
    }, str::from_slice(work_dir))
}

fn create_file(cx: @CrateContext, +full_path: ~str)
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
      option::Some(v) => vec::len(v) as int,
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

    let (name, encoding) = (~"uint", DW_ATE_unsigned);

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
    let llnode = create_derived_type(tg, file_node.node, ~"", 0, size * 8,
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

fn create_boxed_type(cx: @CrateContext, outer: ty::t, _inner: ty::t,
                     span: span, boxed: @Metadata<TyDescMetadata>)
    -> @Metadata<TyDescMetadata> {
    //let tg = StructureTypeTag;
    /*let cache = cx.llmetadata;
    match cached_metadata::<@Metadata<TyDescMetadata>>(
        cache, tg, {|md| ty::hash_ty(outer) == ty::hash_ty(md.data.hash)}) {
      option::Some(md) { return md; }
      option::None {}
    }*/
    let fname = filename_from_span(cx, span);
    let file_node = create_file(cx, fname);
    //let cu_node = create_compile_unit_metadata(cx, fname);
    let uint_t = ty::mk_uint(cx.tcx);
    let refcount_type = create_basic_type(cx, uint_t, span);
    let scx = create_structure(file_node,
                               @/*bad*/ copy ty_to_str(cx.tcx, outer), 0);
    add_member(scx, ~"refcnt", 0, sys::size_of::<uint>() as int,
               sys::min_align_of::<uint>() as int, refcount_type.node);
    add_member(scx, ~"boxed", 0, 8, //XXX member_size_and_align(??)
               8, //XXX just a guess
               boxed.node);
    let llnode = finish_structure(scx);
    let mdval = @Metadata {
        node: llnode,
        data: TyDescMetadata {
            hash: ty::type_id(outer)
        }
    };
    //update_cache(cache, tg, tydesc_metadata(mdval));
    add_named_metadata(cx, ~"llvm.dbg.ty", llnode);
    return mdval;
}

fn create_composite_type(type_tag: int, name: &str, file: ValueRef,
                         line: int, size: int, align: int, offset: int,
                         derived: Option<ValueRef>,
                         +members: Option<~[ValueRef]>)
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

fn create_vec(cx: @CrateContext, vec_t: ty::t, elem_t: ty::t,
              vec_ty_span: codemap::span, elem_ty: @ast::Ty)
    -> @Metadata<TyDescMetadata> {
    let fname = filename_from_span(cx, vec_ty_span);
    let file_node = create_file(cx, fname);
    let elem_ty_md = create_ty(cx, elem_t, elem_ty);
    let scx = create_structure(file_node,
                               @/*bad*/ copy ty_to_str(cx.tcx, vec_t), 0);
    let size_t_type = create_basic_type(cx, ty::mk_uint(cx.tcx), vec_ty_span);
    add_member(scx, ~"fill", 0, sys::size_of::<libc::size_t>() as int,
               sys::min_align_of::<libc::size_t>() as int, size_t_type.node);
    add_member(scx, ~"alloc", 0, sys::size_of::<libc::size_t>() as int,
               sys::min_align_of::<libc::size_t>() as int, size_t_type.node);
    let subrange = llmdnode(~[lltag(SubrangeTag), lli64(0), lli64(0)]);
    let (arr_size, arr_align) = size_and_align_of(cx, elem_t);
    let data_ptr = create_composite_type(ArrayTypeTag, ~"", file_node.node, 0,
                                         arr_size, arr_align, 0,
                                         Some(elem_ty_md.node),
                                         Some(~[subrange]));
    add_member(scx, ~"data", 0, 0, // clang says the size should be 0
               sys::min_align_of::<u8>() as int, data_ptr);
    let llnode = finish_structure(scx);
    @Metadata {
        node: llnode,
        data: TyDescMetadata {
            hash: ty::type_id(vec_t)
        }
    }
}

fn create_ty(_cx: @CrateContext, _t: ty::t, _ty: @ast::Ty)
    -> @Metadata<TyDescMetadata> {
    /*let cache = get_cache(cx);
    match cached_metadata::<@Metadata<TyDescMetadata>>(
        cache, tg, {|md| t == md.data.hash}) {
      option::Some(md) { return md; }
      option::None {}
    }*/

    /* FIXME (#2012): disabled this code as part of the patch that moves
     * recognition of named builtin types into resolve. I tried to fix
     * it, but it seems to already be broken -- it's only called when
     * --xg is given, and compiling with --xg fails on trivial programs.
     *
     * Generating an ast::ty from a ty::t seems like it should not be
     * needed. It is only done to track spans, but you will not get the
     * right spans anyway -- types tend to refer to stuff defined
     * elsewhere, not be self-contained.
     */

    fail!();
    /*
    fn t_to_ty(cx: CrateContext, t: ty::t, span: span) -> @ast::ty {
        let ty = match ty::get(t).struct {
          ty::ty_nil { ast::ty_nil }
          ty::ty_bot { ast::ty_bot }
          ty::ty_bool { ast::ty_bool }
          ty::ty_int(t) { ast::ty_int(t) }
          ty::ty_float(t) { ast::ty_float(t) }
          ty::ty_uint(t) { ast::ty_uint(t) }
          ty::ty_box(mt) { ast::ty_box({ty: t_to_ty(cx, mt.ty, span),
                                        mutbl: mt.mutbl}) }
          ty::ty_uniq(mt) { ast::ty_uniq({ty: t_to_ty(cx, mt.ty, span),
                                          mutbl: mt.mutbl}) }
          ty::ty_vec(mt) { ast::ty_vec({ty: t_to_ty(cx, mt.ty, span),
                                        mutbl: mt.mutbl}) }
          _ {
            cx.sess.span_bug(span, "t_to_ty: Can't handle this type");
          }
        };
        return @{node: ty, span: span};
    }

    match ty.node {
      ast::ty_box(mt) {
        let inner_t = match ty::get(t).struct {
          ty::ty_box(boxed) { boxed.ty }
          _ { cx.sess.span_bug(ty.span, "t_to_ty was incoherent"); }
        };
        let md = create_ty(cx, inner_t, mt.ty);
        let box = create_boxed_type(cx, t, inner_t, ty.span, md);
        return create_pointer_type(cx, t, ty.span, box);
      }

      ast::ty_uniq(mt) {
        let inner_t = match ty::get(t).struct {
          ty::ty_uniq(boxed) { boxed.ty }
          // Hoping we'll have a way to eliminate this check soon.
          _ { cx.sess.span_bug(ty.span, "t_to_ty was incoherent"); }
        };
        let md = create_ty(cx, inner_t, mt.ty);
        return create_pointer_type(cx, t, ty.span, md);
      }

      ast::ty_infer {
        let inferred = t_to_ty(cx, t, ty.span);
        return create_ty(cx, t, inferred);
      }

      ast::ty_vec(mt) {
        let inner_t = ty::sequence_element_type(cx.tcx, t);
        let inner_ast_t = t_to_ty(cx, inner_t, mt.ty.span);
        let v = create_vec(cx, t, inner_t, ty.span, inner_ast_t);
        return create_pointer_type(cx, t, ty.span, v);
      }

      ast::ty_path(_, id) {
        match cx.tcx.def_map.get(id) {
          ast::def_prim_ty(pty) {
            return create_basic_type(cx, t, pty, ty.span);
          }
          _ {}
        }
      }

      _ {}
    };
    */
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
    unsafe {
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
          _ => fail!(~"no single variable name for local")
        };
        let loc = cx.sess.codemap.lookup_char_pos(local.span.lo);
        let ty = node_id_type(bcx, local.node.id);
        let tymd = create_ty(cx, ty, local.node.ty);
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

        let llptr = match bcx.fcx.lllocals.find(&local.node.id) {
          option::Some(local_mem(v)) => v,
          option::Some(_) => {
            bcx.tcx().sess.span_bug(local.span, ~"local is bound to \
                    something weird");
          }
          option::None => {
            match bcx.fcx.lllocals.get(&local.node.pat.id) {
              local_imm(v) => v,
              _ => bcx.tcx().sess.span_bug(local.span, ~"local is bound to \
                                                         something weird")
            }
          }
        };
        let declargs = ~[llmdnode(~[llptr]), mdnode];
        trans::build::Call(bcx, cx.intrinsics.get(&~"llvm.dbg.declare"),
                           declargs);
        return mdval;
    }
}

pub fn create_arg(bcx: block, arg: ast::arg, sp: span)
    -> Option<@Metadata<ArgumentMetadata>> {
    unsafe {
        let fcx = bcx.fcx, cx = *fcx.ccx;
        let cache = get_cache(cx);
        let tg = ArgVariableTag;
        match cached_metadata::<@Metadata<ArgumentMetadata>>(
            cache, ArgVariableTag, |md| md.data.id == arg.id) {
          option::Some(md) => return Some(md),
          option::None => ()
        }

        let loc = cx.sess.codemap.lookup_char_pos(sp.lo);
        let ty = node_id_type(bcx, arg.id);
        let tymd = create_ty(cx, ty, arg.ty);
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

                let llptr = match fcx.llargs.get(&arg.id) {
                  local_mem(v) | local_imm(v) => v,
                };
                let declargs = ~[llmdnode(~[llptr]), mdnode];
                trans::build::Call(bcx,
                                   cx.intrinsics.get(&~"llvm.dbg.declare"),
                                   declargs);
                return Some(mdval);
            }
            _ => {
                return None;
            }
        }
    }
}

pub fn update_source_pos(cx: block, s: span) {
    if !cx.sess().opts.debuginfo {
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
    log(debug, fcx.id);

    let sp = fcx.span.get();
    log(debug, cx.sess.codemap.span_to_str(sp));

    let (ident, ret_ty, id) = match cx.tcx.items.get(&fcx.id) {
      ast_map::node_item(item, _) => {
        match /*bad*/copy item.node {
          ast::item_fn(decl, _, _, _) => {
            (item.ident, decl.output, item.id)
          }
          _ => fcx.ccx.sess.span_bug(item.span, ~"create_function: item \
                                                  bound to non-function")
        }
      }
      ast_map::node_method(method, _, _) => {
          (method.ident, method.decl.output, method.id)
      }
      ast_map::node_expr(expr) => {
        match /*bad*/copy expr.node {
          ast::expr_fn_block(decl, _) => {
            ((dbg_cx.names)(~"fn"), decl.output, expr.id)
          }
          _ => fcx.ccx.sess.span_bug(expr.span,
                                     ~"create_function: \
                                       expected an expr_fn_block here")
        }
      }
      ast_map::node_dtor(_, _, did, _) => {
        ((dbg_cx.names)(~"dtor"), ast_util::dtor_ty(), did.node)
      }
      _ => fcx.ccx.sess.bug(~"create_function: unexpected \
                              sort of node")
    };

    log(debug, ident);
    log(debug, id);

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
          _ => create_ty(cx, ty::node_id_to_type(cx.tcx, id), ret_ty).node
        }
    } else {
        llnull()
    };
    let sub_node = create_composite_type(SubroutineTag, ~"", file_node, 0, 0,
                                         0, 0, option::None,
                                         option::Some(~[ty_node]));

    let fn_metadata = ~[lltag(SubprogramTag),
                       llunused(),
                       file_node,
                       llstr(*cx.sess.str_of(ident)),
                        //XXX fully-qualified C++ name:
                       llstr(*cx.sess.str_of(ident)),
                       llstr(~""), //XXX MIPS name?????
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
