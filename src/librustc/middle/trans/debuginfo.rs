// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
# Debug Info Module

This module serves the purpose of generating debug symbols. We use LLVM's
[source level debugging](http://llvm.org/docs/SourceLevelDebugging.html) features for generating
the debug information. The general principle is this:

Given the right metadata in the LLVM IR, the LLVM code generator is able to create DWARF debug
symbols for the given code. The [metadata](http://llvm.org/docs/LangRef.html#metadata-type) is
structured much like DWARF *debugging information entries* (DIE), representing type information
such as datatype layout, function signatures, block layout, variable location and scope information,
etc. It is the purpose of this module to generate correct metadata and insert it into the LLVM IR.

As the exact format of metadata trees may change between different LLVM versions, we now use LLVM
[DIBuilder](http://llvm.org/docs/doxygen/html/classllvm_1_1DIBuilder.html) to create metadata
where possible. This will hopefully ease the adaption of this module to future LLVM versions.

The public API of the module is a set of functions that will insert the correct metadata into the
LLVM IR when called with the right parameters. The module is thus driven from an outside client with
functions like `debuginfo::create_local_var(bcx: block, local: @ast::local)`.

Internally the module will try to reuse already created metadata by utilizing a cache. All private
state used by the module is stored within a DebugContext struct, which in turn is contained in the
CrateContext.


This file consists of three conceptual sections:
1. The public interface of the module
2. Module-internal metadata creation functions
3. Minor utility functions

*/


use driver::session;
use lib::llvm::llvm;
use lib::llvm::{ValueRef, ModuleRef, ContextRef};
use lib::llvm::debuginfo::*;
use middle::trans::common::*;
use middle::trans::machine;
use middle::trans::type_of;
use middle::trans::type_::Type;
use middle::trans;
use middle::ty;
use util::ppaux::ty_to_str;

use std::hashmap::HashMap;
use std::libc::{c_uint, c_ulonglong, c_longlong};
use std::ptr;
use std::str::as_c_str;
use std::sys;
use std::vec;
use syntax::codemap::span;
use syntax::{ast, codemap, ast_util, ast_map};

static DW_LANG_RUST: int = 0x9000;

static AutoVariableTag: int = 256;
static ArgVariableTag: int = 257;

static DW_ATE_boolean: int = 0x02;
static DW_ATE_float: int = 0x04;
static DW_ATE_signed: int = 0x05;
static DW_ATE_signed_char: int = 0x06;
static DW_ATE_unsigned: int = 0x07;
static DW_ATE_unsigned_char: int = 0x08;




//=-------------------------------------------------------------------------------------------------
//  Public Interface of debuginfo module
//=-------------------------------------------------------------------------------------------------

/// A context object for maintaining all state needed by the debuginfo module.
pub struct DebugContext {
    crate_file: ~str,
    llcontext: ContextRef,
    builder: DIBuilderRef,
    curr_loc: (uint, uint),
    created_files: HashMap<~str, DIFile>,
    created_functions: HashMap<ast::node_id, DISubprogram>,
    created_blocks: HashMap<ast::node_id, DILexicalBlock>,
    created_types: HashMap<uint, DIType>
}

impl DebugContext {
    pub fn new(llmod: ModuleRef, crate: ~str) -> DebugContext {
        debug!("DebugContext::new");
        let builder = unsafe { llvm::LLVMDIBuilderCreate(llmod) };
        // DIBuilder inherits context from the module, so we'd better use the same one
        let llcontext = unsafe { llvm::LLVMGetModuleContext(llmod) };
        return DebugContext {
            crate_file: crate,
            llcontext: llcontext,
            builder: builder,
            curr_loc: (0, 0),
            created_files: HashMap::new(),
            created_functions: HashMap::new(),
            created_blocks: HashMap::new(),
            created_types: HashMap::new(),
        };
    }
}

/// Create any deferred debug metadata nodes
pub fn finalize(cx: @mut CrateContext) {
    debug!("finalize");
    create_compile_unit(cx);
    unsafe {
        llvm::LLVMDIBuilderFinalize(DIB(cx));
        llvm::LLVMDIBuilderDispose(DIB(cx));
    };
}

/// Creates debug information for the given local variable.
///
/// Adds the created metadata nodes directly to the crate's IR.
/// The return value should be ignored if called from outside of the debuginfo module.
pub fn create_local_var(bcx: block, local: @ast::local) -> DIVariable {
    let cx = bcx.ccx();

    let ident = match local.node.pat.node {
      ast::pat_ident(_, ref pth, _) => ast_util::path_to_ident(pth),
      // FIXME this should be handled (#2533)
      _ => {
        bcx.sess().span_note(local.span, "debuginfo for pattern bindings NYI");
        return ptr::null();
      }
    };
    let name: &str = cx.sess.str_of(ident);
    debug!("create_local_var: %s", name);

    let loc = span_start(cx, local.span);
    let ty = node_id_type(bcx, local.node.id);
    let tymd = get_or_create_type(cx, ty, local.node.ty.span);
    let filemd = get_or_create_file(cx, loc.file.name);
    let context = match bcx.parent {
        None => create_function(bcx.fcx),
        Some(_) => get_or_create_block(bcx)
    };

    let var_md = do as_c_str(name) |name| { unsafe {
        llvm::LLVMDIBuilderCreateLocalVariable(
            DIB(cx), AutoVariableTag as u32,
            context, name, filemd,
            loc.line as c_uint, tymd, false, 0, 0)
        }};

    // FIXME(#6814) Should use `pat_util::pat_bindings` for pats like (a, b) etc
    let llptr = match bcx.fcx.lllocals.find_copy(&local.node.pat.id) {
        Some(v) => v,
        None => {
            bcx.tcx().sess.span_bug(
                local.span,
                fmt!("No entry in lllocals table for %?", local.node.id));
        }
    };

    set_debug_location(cx, get_or_create_block(bcx), loc.line, loc.col.to_uint());
    unsafe {
        let instr = llvm::LLVMDIBuilderInsertDeclareAtEnd(DIB(cx), llptr, var_md, bcx.llbb);
        llvm::LLVMSetInstDebugLocation(trans::build::B(bcx), instr);
    }

    return var_md;
}

/// Creates debug information for the given function argument.
///
/// Adds the created metadata nodes directly to the crate's IR.
/// The return value should be ignored if called from outside of the debuginfo module.
pub fn create_arg(bcx: block, arg: &ast::arg, span: span) -> Option<DIVariable> {
    debug!("create_arg");
    if true {
        // XXX create_arg disabled for now because "node_id_type(bcx, arg.id)" below blows
        // up: "error: internal compiler error: node_id_to_type: no type for node `arg (id=10)`"
        return None;
    }

    let fcx = bcx.fcx;
    let cx = fcx.ccx;

    let loc = span_start(cx, span);
    if "<intrinsic>" == loc.file.name {
        return None;
    }

    let ty = node_id_type(bcx, arg.id);
    let tymd = get_or_create_type(cx, ty, arg.ty.span);
    let filemd = get_or_create_file(cx, loc.file.name);
    let context = create_function(fcx);

    match arg.pat.node {
        ast::pat_ident(_, ref path, _) => {
            // XXX: This is wrong; it should work for multiple bindings.
            let ident = path.idents.last();
            let name: &str = cx.sess.str_of(*ident);
            let mdnode = do as_c_str(name) |name| { unsafe {
                llvm::LLVMDIBuilderCreateLocalVariable(
                    DIB(cx),
                    ArgVariableTag as u32,
                    context,
                    name,
                    filemd,
                    loc.line as c_uint,
                    tymd,
                    false,
                    0,
                    0)
                    // XXX need to pass in a real argument number
            }};

            let llptr = fcx.llargs.get_copy(&arg.id);
            set_debug_location(cx, get_or_create_block(bcx), loc.line, loc.col.to_uint());
            unsafe {
                let instr = llvm::LLVMDIBuilderInsertDeclareAtEnd(
                        DIB(cx), llptr, mdnode, bcx.llbb);
                llvm::LLVMSetInstDebugLocation(trans::build::B(bcx), instr);
            }
            return Some(mdnode);
        }
        _ => {
            return None;
        }
    }
}

/// Sets the current debug location at the beginning of the span
///
/// Maps to a call to llvm::LLVMSetCurrentDebugLocation(...)
pub fn update_source_pos(bcx: block, span: span) {
    if !bcx.sess().opts.debuginfo || (*span.lo == 0 && *span.hi == 0) {
        return;
    }
    debug!("update_source_pos: %s", bcx.sess().codemap.span_to_str(span));
    let loc = span_start(bcx.ccx(), span);
    set_debug_location(bcx.ccx(), get_or_create_block(bcx), loc.line, loc.col.to_uint())
}

/// Creates debug information for the given function.
///
/// Adds the created metadata nodes directly to the crate's IR.
/// The return value should be ignored if called from outside of the debuginfo module.
pub fn create_function(fcx: fn_ctxt) -> DISubprogram {
    let cx = fcx.ccx;
    let fcx = &mut *fcx;
    let span = fcx.span.get();

    let fnitem = cx.tcx.items.get_copy(&fcx.id);
    let (ident, ret_ty, id) = match fnitem {
      ast_map::node_item(ref item, _) => {
        match item.node {
          ast::item_fn(ast::fn_decl { output: ref ty, _}, _, _, _, _) => {
            (item.ident, ty, item.id)
          }
          _ => fcx.ccx.sess.span_bug(item.span, "create_function: item bound to non-function")
        }
      }
      ast_map::node_method(@ast::method { decl: ast::fn_decl { output: ref ty, _ },
                           id: id, ident: ident, _}, _, _) => {
          (ident, ty, id)
      }
      ast_map::node_expr(ref expr) => {
        match expr.node {
          ast::expr_fn_block(ref decl, _) => {
            let name = gensym_name("fn");
            (name, &decl.output, expr.id)
          }
          _ => fcx.ccx.sess.span_bug(expr.span,
                  "create_function: expected an expr_fn_block here")
        }
      }
      _ => fcx.ccx.sess.bug("create_function: unexpected sort of node")
    };

    match dbg_cx(cx).created_functions.find(&id) {
        Some(fn_md) => return *fn_md,
        None => ()
    }

    debug!("create_function: %s, %s", cx.sess.str_of(ident), cx.sess.codemap.span_to_str(span));

    let loc = span_start(cx, span);
    let file_md = get_or_create_file(cx, loc.file.name);

    let ret_ty_md = if cx.sess.opts.extra_debuginfo {
        match ret_ty.node {
          ast::ty_nil => ptr::null(),
          _ => get_or_create_type(cx, ty::node_id_to_type(cx.tcx, id), ret_ty.span)
        }
    } else {
        ptr::null()
    };

    let fn_ty = unsafe {
        llvm::LLVMDIBuilderCreateSubroutineType(
            DIB(cx),
            file_md,
            create_DIArray(DIB(cx), [ret_ty_md]))
    };

    let fn_md =
        do as_c_str(cx.sess.str_of(ident)) |name| {
        do as_c_str(cx.sess.str_of(ident)) |linkage| { unsafe {
            llvm::LLVMDIBuilderCreateFunction(
                DIB(cx),
                file_md,
                name,
                linkage,
                file_md,
                loc.line as c_uint,
                fn_ty,
                false,
                true,
                loc.line as c_uint,
                FlagPrototyped as c_uint,
                cx.sess.opts.optimize != session::No,
                fcx.llfn,
                ptr::null(),
                ptr::null())
            }}};

    dbg_cx(cx).created_functions.insert(id, fn_md);
    return fn_md;
}




//=-------------------------------------------------------------------------------------------------
// Module-Internal debug info creation functions
//=-------------------------------------------------------------------------------------------------

fn create_DIArray(builder: DIBuilderRef, arr: &[DIDescriptor]) -> DIArray {
    return unsafe {
        llvm::LLVMDIBuilderGetOrCreateArray(builder, vec::raw::to_ptr(arr), arr.len() as u32)
    };
}

fn create_compile_unit(cx: @mut CrateContext) {
    let dcx = dbg_cx(cx);
    let crate_name: &str = dcx.crate_file;

    debug!("create_compile_unit: %?", crate_name);

    let work_dir = cx.sess.working_dir.to_str();
    let producer = fmt!("rustc version %s", env!("CFG_VERSION"));

    do as_c_str(crate_name) |crate_name| {
    do as_c_str(work_dir) |work_dir| {
    do as_c_str(producer) |producer| {
    do as_c_str("") |flags| {
    do as_c_str("") |split_name| { unsafe {
        llvm::LLVMDIBuilderCreateCompileUnit(dcx.builder,
            DW_LANG_RUST as c_uint, crate_name, work_dir, producer,
            cx.sess.opts.optimize != session::No,
            flags, 0, split_name);
    }}}}}};
}

fn get_or_create_file(cx: &mut CrateContext, full_path: &str) -> DIFile {
    match dbg_cx(cx).created_files.find_equiv(&full_path) {
        Some(file_md) => return *file_md,
        None => ()
    }

    debug!("get_or_create_file: %s", full_path);

    let work_dir = cx.sess.working_dir.to_str();
    let file_name =
        if full_path.starts_with(work_dir) {
            full_path.slice(work_dir.len() + 1u, full_path.len())
        } else {
            full_path
        };

    let file_md =
        do as_c_str(file_name) |file_name| {
        do as_c_str(work_dir) |work_dir| { unsafe {
            llvm::LLVMDIBuilderCreateFile(DIB(cx), file_name, work_dir)
        }}};

    dbg_cx(cx).created_files.insert(full_path.to_owned(), file_md);
    return file_md;
}



fn get_or_create_block(bcx: block) -> DILexicalBlock {
    let mut bcx = bcx;
    let cx = bcx.ccx();

    while bcx.node_info.is_none() {
        match bcx.parent {
          Some(b) => bcx = b,
          None => fail!()
        }
    }
    let span = bcx.node_info.get().span;
    let id = bcx.node_info.get().id;

    match dbg_cx(cx).created_blocks.find(&id) {
        Some(block) => return *block,
        None => ()
    }

    debug!("get_or_create_block: %s", bcx.sess().codemap.span_to_str(span));

    let parent = match bcx.parent {
        None => create_function(bcx.fcx),
        Some(b) => get_or_create_block(b)
    };
    let cx = bcx.ccx();
    let loc = span_start(cx, span);
    let file_md = get_or_create_file(cx, loc.file.name);

    let block_md = unsafe {
        llvm::LLVMDIBuilderCreateLexicalBlock(
            DIB(cx),
            parent, file_md,
            loc.line as c_uint, loc.col.to_uint() as c_uint)
    };

    dbg_cx(cx).created_blocks.insert(id, block_md);

    return block_md;
}



fn create_basic_type(cx: &mut CrateContext, t: ty::t, _span: span) -> DIType {

    debug!("create_basic_type: %?", ty::get(t));

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

    let (size, align) = size_and_align_of(cx, t);
    let ty_md = do as_c_str(name) |name| { unsafe {
            llvm::LLVMDIBuilderCreateBasicType(
                DIB(cx),
                name,
                bytes_to_bits(size),
                bytes_to_bits(align),
                encoding as c_uint)
        }};

    return ty_md;
}

fn create_pointer_type(cx: &mut CrateContext, t: ty::t, _span: span, pointee: DIType) -> DIType {
    let (size, align) = size_and_align_of(cx, t);
    let name = ty_to_str(cx.tcx, t);
    let ptr_md = do as_c_str(name) |name| { unsafe {
        llvm::LLVMDIBuilderCreatePointerType(
            DIB(cx),
            pointee,
            bytes_to_bits(size),
            bytes_to_bits(align),
            name)
    }};
    return ptr_md;
}

fn create_struct(cx: &mut CrateContext,
                 struct_type: ty::t,
                 fields: ~[ty::field],
                 span: span)
              -> DICompositeType {
    debug!("create_struct: %?", ty::get(struct_type));

    let struct_name = ty_to_str(cx.tcx, struct_type);
    let struct_llvm_type = type_of::type_of(cx, struct_type);

    let field_llvm_types = fields.map(|field| type_of::type_of(cx, field.mt.ty));
    let field_names = fields.map(|field| cx.sess.str_of(field.ident).to_owned());
    let field_types_metadata = fields.map(|field| get_or_create_type(cx, field.mt.ty, span));

    return create_composite_type(
        cx,
        struct_llvm_type,
        struct_name,
        field_llvm_types,
        field_names,
        field_types_metadata,
        span);
}

fn create_tuple(cx: &mut CrateContext,
                tuple_type: ty::t,
                component_types: &[ty::t],
                span: span)
             -> DICompositeType {

    let tuple_name = "tuple"; // this should have a better name
    let tuple_llvm_type = type_of::type_of(cx, tuple_type);
    // Create a vec of empty strings. A vec::build_n() function would be nice for this.
    let mut component_names : ~[~str] = vec::with_capacity(component_types.len());
    component_names.grow_fn(component_types.len(), |_| ~"");

    let component_llvm_types = component_types.map(|it| type_of::type_of(cx, *it));
    let component_types_metadata = component_types.map(|it| get_or_create_type(cx, *it, span));

    return create_composite_type(
        cx,
        tuple_llvm_type,
        tuple_name,
        component_llvm_types,
        component_names,
        component_types_metadata,
        span);
}

fn create_enum_md(cx: &mut CrateContext,
                  enum_type: ty::t,
                  enum_def_id: ast::def_id,
                  span: span) -> DIType {

    let enum_name = ty_to_str(cx.tcx, enum_type);
    let discriminator_llvm_type = Type::enum_discrim(cx);
    let discriminator_size = machine::llsize_of_alloc(cx, discriminator_llvm_type);
    let discriminator_align = machine::llalign_of_min(cx, discriminator_llvm_type);

    assert!(Type::enum_discrim(cx) == cx.int_type);
    let discriminator_type_md = get_or_create_type(cx, ty::mk_int(), span);

    if ty::type_is_empty(cx.tcx, enum_type) {
        // XXX: This should not "rename" the type to nil
        return get_or_create_type(cx, ty::mk_nil(), span);
    }

    if ty::type_is_c_like_enum(cx.tcx, enum_type) {

        let variants : &[ty::VariantInfo] = *ty::enum_variants(cx.tcx, enum_def_id);

        let enumerators : ~[(~str, int)] = variants
            .iter()
            .transform(|v| (cx.sess.str_of(v.name).to_owned(), v.disr_val))
            .collect();

        let enumerators_md : ~[DIDescriptor] =
            do enumerators.iter().transform |&(name,value)| {
                do name.as_c_str |name| { unsafe {
                    llvm::LLVMDIBuilderCreateEnumerator(
                        DIB(cx),
                        name,
                        value as c_ulonglong)
                }}
            }.collect();

        let loc = span_start(cx, span);
        let file_metadata = get_or_create_file(cx, loc.file.name);

        return do enum_name.as_c_str |enum_name| { unsafe {
            llvm::LLVMDIBuilderCreateEnumerationType(
                DIB(cx),
                file_metadata,
                enum_name,
                file_metadata,
                loc.line as c_uint,
                bytes_to_bits(discriminator_size),
                bytes_to_bits(discriminator_align),
                create_DIArray(DIB(cx), enumerators_md),
                discriminator_type_md)
        }};
    }

    cx.sess.bug("");
}




/// Creates debug information for a composite type, that is, anything that results in a LLVM struct.
///
/// Examples of Rust types to use this are: structs, tuples, boxes, vecs, and enums.
fn create_composite_type(cx: &mut CrateContext,
                         composite_llvm_type: Type,
                         composite_type_name: &str,
                         member_llvm_types: &[Type],
                         member_names: &[~str],
                         member_type_metadata: &[DIType],
                         span: span)
                      -> DICompositeType {

    let loc = span_start(cx, span);
    let file_metadata = get_or_create_file(cx, loc.file.name);

    let composite_size = machine::llsize_of_alloc(cx, composite_llvm_type);
    let composite_align = machine::llalign_of_min(cx, composite_llvm_type);

    let member_metadata : ~[DIDescriptor] = member_llvm_types
        .iter()
        .enumerate()
        .transform(|(i, member_llvm_type)| {
            let member_size = machine::llsize_of_alloc(cx, *member_llvm_type);
            let member_align = machine::llalign_of_min(cx, *member_llvm_type);
            let member_offset = machine::llelement_offset(cx, composite_llvm_type, i);
            let member_name : &str = member_names[i];

            do member_name.as_c_str |member_name| { unsafe {
                llvm::LLVMDIBuilderCreateMemberType(
                    DIB(cx),
                    file_metadata,
                    member_name,
                    file_metadata,
                    loc.line as c_uint,
                    bytes_to_bits(member_size),
                    bytes_to_bits(member_align),
                    bytes_to_bits(member_offset),
                    0,
                    member_type_metadata[i])
            }}
        })
        .collect();

    return do composite_type_name.as_c_str |name| { unsafe {
        llvm::LLVMDIBuilderCreateStructType(
            DIB(cx),
            file_metadata,
            name,
            file_metadata,
            loc.line as c_uint,
            bytes_to_bits(composite_size),
            bytes_to_bits(composite_align),
            0,
            ptr::null(),
            create_DIArray(DIB(cx), member_metadata),
            0,
            ptr::null())
    }};
}

// returns (void* type as a ValueRef, size in bytes, align in bytes)
fn voidptr(cx: &mut CrateContext) -> (DIDerivedType, uint, uint) {
    let size = sys::size_of::<ValueRef>();
    let align = sys::min_align_of::<ValueRef>();
    let vp = do as_c_str("*void") |name| { unsafe {
            llvm::LLVMDIBuilderCreatePointerType(
                DIB(cx),
                ptr::null(),
                bytes_to_bits(size),
                bytes_to_bits(align),
                name)
        }};
    return (vp, size, align);
}

fn create_boxed_type(cx: &mut CrateContext,
                     content_llvm_type: Type,
                     content_type_metadata: DIType,
                     span: span)
                  -> DICompositeType {

    let box_llvm_type = Type::box(cx, &content_llvm_type);
    let member_llvm_types = box_llvm_type.field_types();
    let member_names = [~"refcnt", ~"tydesc", ~"prev", ~"next", ~"val"];

    assert!(box_layout_is_as_expected(cx, member_llvm_types, content_llvm_type));

    let int_type = ty::mk_int();
    let nil_pointer_type = ty::mk_nil_ptr(cx.tcx);

    let member_types_metadata = [
        get_or_create_type(cx, int_type, span),
        get_or_create_type(cx, nil_pointer_type, span),
        get_or_create_type(cx, nil_pointer_type, span),
        get_or_create_type(cx, nil_pointer_type, span),
        content_type_metadata
    ];

    return create_composite_type(
        cx,
        box_llvm_type,
        "box name",
        member_llvm_types,
        member_names,
        member_types_metadata,
        span);

    // Unfortunately, we cannot assert anything but the correct types here---and not whether the
    // 'next' and 'prev' pointers are in the order.
    fn box_layout_is_as_expected(cx: &CrateContext,
                                 member_llvm_types: &[Type],
                                 content_llvm_type: Type)
                              -> bool {
        member_llvm_types.len() == 5 &&
        member_llvm_types[0] == cx.int_type &&
        member_llvm_types[1] == cx.tydesc_type.ptr_to() &&
        member_llvm_types[2] == Type::i8().ptr_to() &&
        member_llvm_types[3] == Type::i8().ptr_to() &&
        member_llvm_types[4] == content_llvm_type
    }
}

fn create_fixed_vec(cx: &mut CrateContext, _vec_t: ty::t, elem_t: ty::t,
                    len: uint, span: span) -> DIType {
    debug!("create_fixed_vec: %?", ty::get(_vec_t));

    let elem_ty_md = get_or_create_type(cx, elem_t, span);
    let (size, align) = size_and_align_of(cx, elem_t);

    let subrange = unsafe { llvm::LLVMDIBuilderGetOrCreateSubrange(
        DIB(cx),
        0,
        len as c_longlong
    )};

    let subscripts = create_DIArray(DIB(cx), [subrange]);
    return unsafe { llvm::LLVMDIBuilderCreateArrayType(
            DIB(cx),
            bytes_to_bits(size * len),
            bytes_to_bits(align),
            elem_ty_md,
            subscripts
    )};
}

fn create_boxed_vec(cx: &mut CrateContext,
                    element_type: ty::t,
                    span: span)
                 -> DICompositeType {

    let element_type_metadata = get_or_create_type(cx, element_type, span);
    let element_llvm_type = type_of::type_of(cx, element_type);
    let vec_llvm_type = Type::vec(cx.sess.targ_cfg.arch, &element_llvm_type);
    let vec_type_name = &"vec";

    let member_llvm_types = vec_llvm_type.field_types();
    let member_names = &[~"fill", ~"alloc", ~"elements"];

    let int_type_md = get_or_create_type(cx, ty::mk_int(), span);
    let array_type_md = unsafe { llvm::LLVMDIBuilderCreateArrayType(
        DIB(cx),
        bytes_to_bits(machine::llsize_of_alloc(cx, element_llvm_type)),
        bytes_to_bits(machine::llalign_of_min(cx, element_llvm_type)),
        element_type_metadata,
        create_DIArray(DIB(cx), [llvm::LLVMDIBuilderGetOrCreateSubrange(DIB(cx), 0, 0)]))
    };

    //                           fill         alloc        elements
    let member_type_metadata = &[int_type_md, int_type_md, array_type_md];

    let vec_md = create_composite_type(
        cx,
        vec_llvm_type,
        vec_type_name,
        member_llvm_types,
        member_names,
        member_type_metadata,
        span);

    return create_boxed_type(cx, vec_llvm_type, vec_md, span);
}

fn create_vec_slice(cx: &mut CrateContext,
                    vec_type: ty::t,
                    element_type: ty::t,
                    span: span)
                 -> DICompositeType {

    debug!("create_vec_slice: %?", ty::get(vec_type));

    let slice_llvm_type = type_of::type_of(cx, vec_type);
    let slice_type_name = ty_to_str(cx.tcx, vec_type);

    let member_llvm_types = slice_llvm_type.field_types();
    let member_names = &[~"data_ptr", ~"size_in_bytes"];

    assert!(slice_layout_is_as_expected(cx, member_llvm_types, element_type));

    let data_ptr_type = ty::mk_ptr(cx.tcx, ty::mt { ty: element_type, mutbl: ast::m_const });

    let member_type_metadata = &[
        get_or_create_type(cx, data_ptr_type, span),
        get_or_create_type(cx, ty::mk_uint(), span)
        ];

    return create_composite_type(
        cx,
        slice_llvm_type,
        slice_type_name,
        member_llvm_types,
        member_names,
        member_type_metadata,
        span);

    fn slice_layout_is_as_expected(cx: &mut CrateContext,
                                   member_llvm_types: &[Type],
                                   element_type: ty::t)
                                -> bool {
        member_llvm_types.len() == 2 &&
        member_llvm_types[0] == type_of::type_of(cx, element_type).ptr_to() &&
        member_llvm_types[1] == cx.int_type
    }
}

fn create_fn_ty(cx: &mut CrateContext, _fn_ty: ty::t, inputs: ~[ty::t], output: ty::t,
                span: span) -> DICompositeType {
    debug!("create_fn_ty: %?", ty::get(_fn_ty));

    let loc = span_start(cx, span);
    let file_md = get_or_create_file(cx, loc.file.name);
    let (vp, _, _) = voidptr(cx);
    let output_md = get_or_create_type(cx, output, span);
    let output_ptr_md = create_pointer_type(cx, output, span, output_md);
    let inputs_vals = do inputs.map |arg| { get_or_create_type(cx, *arg, span) };
    let members = ~[output_ptr_md, vp] + inputs_vals;

    return unsafe {
        llvm::LLVMDIBuilderCreateSubroutineType(
            DIB(cx),
            file_md,
            create_DIArray(DIB(cx), members))
    };
}

fn create_unimpl_ty(cx: &mut CrateContext, t: ty::t) -> DIType {
    debug!("create_unimpl_ty: %?", ty::get(t));

    let name = ty_to_str(cx.tcx, t);
    let md = do as_c_str(fmt!("NYI<%s>", name)) |name| { unsafe {
        llvm::LLVMDIBuilderCreateBasicType(
            DIB(cx),
            name,
            0_u64,
            8_u64,
            DW_ATE_unsigned as c_uint)
        }};
    return md;
}

fn get_or_create_type(cx: &mut CrateContext, t: ty::t, span: span) -> DIType {
    let ty_id = ty::type_id(t);
    match dbg_cx(cx).created_types.find(&ty_id) {
        Some(ty_md) => return *ty_md,
        None => ()
    }

    debug!("get_or_create_type: %?", ty::get(t));

    let sty = copy ty::get(t).sty;
    let ty_md = match sty {
        ty::ty_nil      |
        ty::ty_bot      |
        ty::ty_bool     |
        ty::ty_int(_)   |
        ty::ty_uint(_)  |
        ty::ty_float(_) => {
            create_basic_type(cx, t, span)
        },
        ty::ty_estr(ref vstore) => {
            let i8_t = ty::mk_i8();
            match *vstore {
                ty::vstore_fixed(len) => {
                    create_fixed_vec(cx, t, i8_t, len + 1, span)
                },
                ty::vstore_uniq |
                ty::vstore_box => {
                    let box_md = create_boxed_vec(cx, i8_t, span);
                    create_pointer_type(cx, t, span, box_md)
                }
                ty::vstore_slice(_region) => {
                    create_vec_slice(cx, t, i8_t, span)
                }
            }
        },
        ty::ty_enum(def_id, ref _substs) => {
            //cx.sess.span_note(span, "debuginfo for enum NYI");
            //create_unimpl_ty(cx, t)
            create_enum_md(cx, t, def_id, span)
        },
        ty::ty_box(ref mt) |
        ty::ty_uniq(ref mt) => {
            let content_llvm_type = type_of::type_of(cx, mt.ty);
            let content_type_metadata = get_or_create_type(cx, mt.ty, span);

            let box_metadata = create_boxed_type(cx,
                                                 content_llvm_type,
                                                 content_type_metadata,
                                                 span);

            create_pointer_type(cx, t, span, box_metadata)
        },
        ty::ty_evec(ref mt, ref vstore) => {
            match *vstore {
                ty::vstore_fixed(len) => {
                    create_fixed_vec(cx, t, mt.ty, len, span)
                },
                ty::vstore_uniq |
                ty::vstore_box  => {
                    let box_md = create_boxed_vec(cx, mt.ty, span);
                    create_pointer_type(cx, t, span, box_md)
                },
                ty::vstore_slice(_) => {
                    create_vec_slice(cx, t, mt.ty, span)
                }
            }
        },
        ty::ty_ptr(ref mt) |
        ty::ty_rptr(_, ref mt) => {
            let pointee = get_or_create_type(cx, mt.ty, span);
            create_pointer_type(cx, t, span, pointee)
        },
        ty::ty_bare_fn(ref barefnty) => {
            let inputs = barefnty.sig.inputs.map(|a| *a);
            let output = barefnty.sig.output;
            create_fn_ty(cx, t, inputs, output, span)
        },
        ty::ty_closure(ref _closurety) => {
            cx.sess.span_note(span, "debuginfo for closure NYI");
            create_unimpl_ty(cx, t)
        },
        ty::ty_trait(_did, ref _substs, ref _vstore, _, _bounds) => {
            cx.sess.span_note(span, "debuginfo for trait NYI");
            create_unimpl_ty(cx, t)
        },
        ty::ty_struct(did, ref substs) => {
            let fields = ty::struct_fields(cx.tcx, did, substs);
            create_struct(cx, t, fields, span)
        },
        ty::ty_tup(ref elements) => {
            create_tuple(cx, t, *elements, span)
        },
        _ => cx.sess.bug("debuginfo: unexpected type in get_or_create_type")
    };

    dbg_cx(cx).created_types.insert(ty_id, ty_md);
    return ty_md;
}

fn set_debug_location(cx: @mut CrateContext, scope: DIScope, line: uint, col: uint) {
    if dbg_cx(cx).curr_loc == (line, col) {
        return;
    }
    debug!("setting debug location to %u %u", line, col);
    dbg_cx(cx).curr_loc = (line, col);

    let elems = ~[C_i32(line as i32), C_i32(col as i32), scope, ptr::null()];
    unsafe {
        let dbg_loc = llvm::LLVMMDNodeInContext(
                dbg_cx(cx).llcontext,
                vec::raw::to_ptr(elems),
                elems.len() as c_uint);

        llvm::LLVMSetCurrentDebugLocation(cx.builder.B, dbg_loc);
    }
}


//=-------------------------------------------------------------------------------------------------
//  Utility Functions
//=-------------------------------------------------------------------------------------------------

#[inline]
fn roundup(x: uint, a: uint) -> uint {
    ((x + (a - 1)) / a) * a
}

/// Return codemap::Loc corresponding to the beginning of the span
fn span_start(cx: &CrateContext, span: span) -> codemap::Loc {
    cx.sess.codemap.lookup_char_pos(span.lo)
}

fn size_and_align_of(cx: &mut CrateContext, t: ty::t) -> (uint, uint) {
    let llty = type_of::type_of(cx, t);
    (machine::llsize_of_alloc(cx, llty), machine::llalign_of_min(cx, llty))
}

fn bytes_to_bits(bytes: uint) -> c_ulonglong {
    (bytes * 8) as c_ulonglong
}

#[inline]
fn dbg_cx<'a>(cx: &'a mut CrateContext) -> &'a mut DebugContext {
    cx.dbg_cx.get_mut_ref()
}

#[inline]
fn DIB(cx: &CrateContext) -> DIBuilderRef {
    cx.dbg_cx.get_ref().builder
}
