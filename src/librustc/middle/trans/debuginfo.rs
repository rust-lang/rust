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
functions like `debuginfo::local_var_metadata(bcx: block, local: &ast::local)`.

Internally the module will try to reuse already created metadata by utilizing a cache. The way to
get a shared metadata node when needed is thus to just call the corresponding function in this
module:

    let file_metadata = file_metadata(crate_context, path);

The function will take care of probing the cache for an existing node for that exact file path.

All private state used by the module is stored within a DebugContext struct, which in turn is
contained in the CrateContext.


This file consists of three conceptual sections:
1. The public interface of the module
2. Module-internal metadata creation functions
3. Minor utility functions

*/


use driver::session;
use lib::llvm::llvm;
use lib::llvm::{ModuleRef, ContextRef};
use lib::llvm::debuginfo::*;
use middle::trans::common::*;
use middle::trans::machine;
use middle::trans::type_of;
use middle::trans::type_::Type;
use middle::trans::adt;
use middle::trans;
use middle::ty;
use util::ppaux::ty_to_str;

use std::hashmap::HashMap;
use std::libc::{c_uint, c_ulonglong, c_longlong};
use std::ptr;
use std::vec;
use syntax::codemap::span;
use syntax::{ast, codemap, ast_util, ast_map};

static DW_LANG_RUST: int = 0x9000;

static DW_TAG_auto_variable: int = 0x100;
static DW_TAG_arg_variable: int = 0x101;

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
    compile_unit_metadata(cx);
    unsafe {
        llvm::LLVMDIBuilderFinalize(DIB(cx));
        llvm::LLVMDIBuilderDispose(DIB(cx));
    };
}

/// Creates debug information for the given local variable.
///
/// Adds the created metadata nodes directly to the crate's IR.
/// The return value should be ignored if called from outside of the debuginfo module.
pub fn create_local_var_metadata(bcx: @mut Block, local: @ast::Local) -> DIVariable {
    let cx = bcx.ccx();

    let ident = match local.pat.node {
      ast::pat_ident(_, ref pth, _) => ast_util::path_to_ident(pth),
      // FIXME this should be handled (#2533)
      _ => {
        bcx.sess().span_note(local.span, "debuginfo for pattern bindings NYI");
        return ptr::null();
      }
    };

    let name: &str = cx.sess.str_of(ident);
    debug!("create_local_var_metadata: %s", name);

    let loc = span_start(cx, local.span);
    let ty = node_id_type(bcx, local.id);
    let type_metadata = type_metadata(cx, ty, local.ty.span);
    let file_metadata = file_metadata(cx, loc.file.name);

    let context = match bcx.parent {
        None => create_function_metadata(bcx.fcx),
        Some(_) => lexical_block_metadata(bcx)
    };

    let var_metadata = do name.as_c_str |name| {
        unsafe {
            llvm::LLVMDIBuilderCreateLocalVariable(
                DIB(cx),
                DW_TAG_auto_variable as u32,
                context,
                name,
                file_metadata,
                loc.line as c_uint,
                type_metadata,
                false,
                0,
                0)
        }
    };

    // FIXME(#6814) Should use `pat_util::pat_bindings` for pats like (a, b) etc
    let llptr = match bcx.fcx.lllocals.find_copy(&local.pat.id) {
        Some(v) => v,
        None => {
            bcx.tcx().sess.span_bug(
                local.span,
                fmt!("No entry in lllocals table for %?", local.id));
        }
    };

    set_debug_location(cx, lexical_block_metadata(bcx), loc.line, loc.col.to_uint());
    unsafe {
        let instr = llvm::LLVMDIBuilderInsertDeclareAtEnd(DIB(cx), llptr, var_metadata, bcx.llbb);
        llvm::LLVMSetInstDebugLocation(trans::build::B(bcx).llbuilder, instr);
    }

    return var_metadata;
}

/// Creates debug information for the given function argument.
///
/// Adds the created metadata nodes directly to the crate's IR.
/// The return value should be ignored if called from outside of the debuginfo module.
pub fn create_argument_metadata(bcx: @mut Block, arg: &ast::arg, span: span) -> Option<DIVariable> {
    debug!("create_argument_metadata");
    if true {
        // XXX create_argument_metadata disabled for now because "node_id_type(bcx, arg.id)" below
        // blows up:
        // "error: internal compiler error: node_id_to_type: no type for node `arg (id=10)`"
        return None;
    }

    let fcx = bcx.fcx;
    let cx = fcx.ccx;

    let loc = span_start(cx, span);
    if "<intrinsic>" == loc.file.name {
        return None;
    }

    let ty = node_id_type(bcx, arg.id);
    let type_metadata = type_metadata(cx, ty, arg.ty.span);
    let file_metadata = file_metadata(cx, loc.file.name);
    let context = create_function_metadata(fcx);

    match arg.pat.node {
        ast::pat_ident(_, ref path, _) => {
            // XXX: This is wrong; it should work for multiple bindings.
            let ident = path.idents.last();
            let name: &str = cx.sess.str_of(*ident);
            let var_metadata = do name.as_c_str |name| {
                unsafe {
                    llvm::LLVMDIBuilderCreateLocalVariable(
                        DIB(cx),
                        DW_TAG_arg_variable as u32,
                        context,
                        name,
                        file_metadata,
                        loc.line as c_uint,
                        type_metadata,
                        false,
                        0,
                        0)
                    // XXX need to pass in a real argument number
                }
            };

            let llptr = fcx.llargs.get_copy(&arg.id);
            set_debug_location(cx, lexical_block_metadata(bcx), loc.line, loc.col.to_uint());
            unsafe {
                let instr = llvm::LLVMDIBuilderInsertDeclareAtEnd(
                        DIB(cx), llptr, var_metadata, bcx.llbb);
                llvm::LLVMSetInstDebugLocation(trans::build::B(bcx).llbuilder, instr);
            }
            return Some(var_metadata);
        }
        _ => {
            return None;
        }
    }
}

/// Sets the current debug location at the beginning of the span
///
/// Maps to a call to llvm::LLVMSetCurrentDebugLocation(...)
pub fn update_source_pos(bcx: @mut Block, span: span) {
    if !bcx.sess().opts.debuginfo || (*span.lo == 0 && *span.hi == 0) {
        return;
    }
    debug!("update_source_pos: %s", bcx.sess().codemap.span_to_str(span));
    let loc = span_start(bcx.ccx(), span);
    set_debug_location(bcx.ccx(), lexical_block_metadata(bcx), loc.line, loc.col.to_uint())
}

/// Creates debug information for the given function.
///
/// Adds the created metadata nodes directly to the crate's IR.
/// The return value should be ignored if called from outside of the debuginfo module.
pub fn create_function_metadata(fcx: &FunctionContext) -> DISubprogram {
    let cx = fcx.ccx;
    let span = fcx.span.get();

    let fnitem = cx.tcx.items.get_copy(&fcx.id);
    let (ident, ret_ty, id) = match fnitem {
        ast_map::node_item(ref item, _) => {
            match item.node {
                ast::item_fn(ast::fn_decl { output: ref ty, _}, _, _, _, _) => {
                    (item.ident, ty, item.id)
                }
                _ => fcx.ccx.sess.span_bug(item.span,
                                           "create_function_metadata: item bound to non-function")
            }
        }
        ast_map::node_method(
            @ast::method {
                decl: ast::fn_decl { output: ref ty, _ },
                id: id,
                ident: ident,
                _
            },
            _,
            _) => {
            (ident, ty, id)
        }
        ast_map::node_expr(ref expr) => {
            match expr.node {
                ast::expr_fn_block(ref decl, _) => {
                    let name = gensym_name("fn");
                    (name, &decl.output, expr.id)
                }
                _ => fcx.ccx.sess.span_bug(expr.span,
                        "create_function_metadata: expected an expr_fn_block here")
            }
        }
        ast_map::node_trait_method(
            @ast::provided(
                @ast::method {
                    decl: ast::fn_decl { output: ref ty, _ },
                    id: id,
                    ident: ident,
                    _
                }),
            _,
            _) => {
            (ident, ty, id)
        }
        _ => fcx.ccx.sess.bug("create_function_metadata: unexpected sort of node")
    };

    match dbg_cx(cx).created_functions.find(&id) {
        Some(fn_metadata) => return *fn_metadata,
        None => ()
    }

    debug!("create_function_metadata: %s, %s",
           cx.sess.str_of(ident),
           cx.sess.codemap.span_to_str(span));

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, loc.file.name);

    let return_type_metadata = if cx.sess.opts.extra_debuginfo {
        match ret_ty.node {
          ast::ty_nil => ptr::null(),
          _ => type_metadata(cx, ty::node_id_to_type(cx.tcx, id), ret_ty.span)
        }
    } else {
        ptr::null()
    };

    let fn_ty = unsafe {
        llvm::LLVMDIBuilderCreateSubroutineType(
            DIB(cx),
            file_metadata,
            create_DIArray(DIB(cx), [return_type_metadata]))
    };

    let fn_metadata =
        do cx.sess.str_of(ident).as_c_str |name| {
        do cx.sess.str_of(ident).as_c_str |linkage| {
            unsafe {
                llvm::LLVMDIBuilderCreateFunction(
                    DIB(cx),
                    file_metadata,
                    name,
                    linkage,
                    file_metadata,
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
            }
        }};

    dbg_cx(cx).created_functions.insert(id, fn_metadata);
    return fn_metadata;
}




//=-------------------------------------------------------------------------------------------------
// Module-Internal debug info creation functions
//=-------------------------------------------------------------------------------------------------

fn create_DIArray(builder: DIBuilderRef, arr: &[DIDescriptor]) -> DIArray {
    return unsafe {
        llvm::LLVMDIBuilderGetOrCreateArray(builder, vec::raw::to_ptr(arr), arr.len() as u32)
    };
}

fn compile_unit_metadata(cx: @mut CrateContext) {
    let dcx = dbg_cx(cx);
    let crate_name: &str = dcx.crate_file;

    debug!("compile_unit_metadata: %?", crate_name);

    let work_dir = cx.sess.working_dir.to_str();
    let producer = fmt!("rustc version %s", env!("CFG_VERSION"));

    do crate_name.as_c_str |crate_name| {
    do work_dir.as_c_str |work_dir| {
    do producer.as_c_str |producer| {
    do "".as_c_str |flags| {
    do "".as_c_str |split_name| {
        unsafe {
            llvm::LLVMDIBuilderCreateCompileUnit(dcx.builder,
                DW_LANG_RUST as c_uint, crate_name, work_dir, producer,
                cx.sess.opts.optimize != session::No,
                flags, 0, split_name);
        }
    }}}}};
}

fn file_metadata(cx: &mut CrateContext, full_path: &str) -> DIFile {
    match dbg_cx(cx).created_files.find_equiv(&full_path) {
        Some(file_metadata) => return *file_metadata,
        None => ()
    }

    debug!("file_metadata: %s", full_path);

    let work_dir = cx.sess.working_dir.to_str();
    let file_name =
        if full_path.starts_with(work_dir) {
            full_path.slice(work_dir.len() + 1u, full_path.len())
        } else {
            full_path
        };

    let file_metadata =
        do file_name.as_c_str |file_name| {
        do work_dir.as_c_str |work_dir| {
            unsafe {
                llvm::LLVMDIBuilderCreateFile(DIB(cx), file_name, work_dir)
            }
        }};

    dbg_cx(cx).created_files.insert(full_path.to_owned(), file_metadata);
    return file_metadata;
}

/// Get or create the lexical block metadata node for the given LLVM basic block.
fn lexical_block_metadata(bcx: @mut Block) -> DILexicalBlock {
    let cx = bcx.ccx();
    let mut bcx = bcx;

    // Search up the tree of basic blocks until we find one that knows the containing lexical block.
    while bcx.node_info.is_none() {
        match bcx.parent {
            Some(b) => bcx = b,
            None => cx.sess.bug("debuginfo: Could not find lexical block for LLVM basic block.")
        }
    }

    let span = bcx.node_info.get().span;
    let id = bcx.node_info.get().id;

    // Check whether we already have a cache entry for this node id
    match dbg_cx(cx).created_blocks.find(&id) {
        Some(block) => return *block,
        None => ()
    }

    debug!("lexical_block_metadata: %s", bcx.sess().codemap.span_to_str(span));

    let parent = match bcx.parent {
        None => create_function_metadata(bcx.fcx),
        Some(b) => lexical_block_metadata(b)
    };

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, loc.file.name);

    let lexical_block_metadata = unsafe {
        llvm::LLVMDIBuilderCreateLexicalBlock(
            DIB(cx),
            parent,
            file_metadata,
            loc.line as c_uint,
            loc.col.to_uint() as c_uint)
    };

    dbg_cx(cx).created_blocks.insert(id, lexical_block_metadata);

    return lexical_block_metadata;
}

fn basic_type_metadata(cx: &mut CrateContext, t: ty::t) -> DIType {

    debug!("basic_type_metadata: %?", ty::get(t));

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
        _ => cx.sess.bug("debuginfo::basic_type_metadata - t is invalid type")
    };

    let llvm_type = type_of::type_of(cx, t);
    let (size, align) = size_and_align_of(cx, llvm_type);
    let ty_metadata = do name.as_c_str |name| {
        unsafe {
            llvm::LLVMDIBuilderCreateBasicType(
                DIB(cx),
                name,
                bytes_to_bits(size),
                bytes_to_bits(align),
                encoding as c_uint)
        }
    };

    return ty_metadata;
}

fn pointer_type_metadata(cx: &mut CrateContext,
                         pointer_type: ty::t,
                         pointee_type_metadata: DIType)
                      -> DIType {
    let pointer_llvm_type = type_of::type_of(cx, pointer_type);
    let (pointer_size, pointer_align) = size_and_align_of(cx, pointer_llvm_type);
    let name = ty_to_str(cx.tcx, pointer_type);
    let ptr_metadata = do name.as_c_str |name| {
        unsafe {
            llvm::LLVMDIBuilderCreatePointerType(
                DIB(cx),
                pointee_type_metadata,
                bytes_to_bits(pointer_size),
                bytes_to_bits(pointer_align),
                name)
        }
    };
    return ptr_metadata;
}

fn struct_metadata(cx: &mut CrateContext,
                   struct_type: ty::t,
                   fields: ~[ty::field],
                   span: span)
                -> DICompositeType {
    let struct_name = ty_to_str(cx.tcx, struct_type);
    debug!("struct_metadata: %s", struct_name);

    let struct_llvm_type = type_of::type_of(cx, struct_type);

    let field_llvm_types = do fields.map |field| { type_of::type_of(cx, field.mt.ty) };
    let field_names = do fields.map |field| { cx.sess.str_of(field.ident).to_owned() };
    let field_types_metadata = do fields.map |field| {
        type_metadata(cx, field.mt.ty, span)
    };

    return composite_type_metadata(
        cx,
        struct_llvm_type,
        struct_name,
        field_llvm_types,
        field_names,
        field_types_metadata,
        span);
}

fn tuple_metadata(cx: &mut CrateContext,
                  tuple_type: ty::t,
                  component_types: &[ty::t],
                  span: span)
               -> DICompositeType {

    let tuple_name = ty_to_str(cx.tcx, tuple_type);
    let tuple_llvm_type = type_of::type_of(cx, tuple_type);

    let component_names = do component_types.map |_| { ~"" };
    let component_llvm_types = do component_types.map |it| { type_of::type_of(cx, *it) };
    let component_types_metadata = do component_types.map |it| {
        type_metadata(cx, *it, span)
    };

    return composite_type_metadata(
        cx,
        tuple_llvm_type,
        tuple_name,
        component_llvm_types,
        component_names,
        component_types_metadata,
        span);
}

fn enum_metadata(cx: &mut CrateContext,
                 enum_type: ty::t,
                 enum_def_id: ast::def_id,
                 // _substs is only needed in the other version. Will go away with new snapshot.
                 _substs: &ty::substs,
                 span: span)
              -> DIType {

    let enum_name = ty_to_str(cx.tcx, enum_type);

    // For empty enums there is an early exit. Just describe it as an empty struct with the
    // appropriate type name
    if ty::type_is_empty(cx.tcx, enum_type) {
        return composite_type_metadata(cx, Type::nil(), enum_name, [], [], [], span);
    }

    // Prepare some data (llvm type, size, align, ...) about the discriminant. This data will be
    // needed in all of the following cases.
    let discriminant_llvm_type = Type::enum_discrim(cx);
    let (discriminant_size, discriminant_align) = size_and_align_of(cx, discriminant_llvm_type);

    assert!(Type::enum_discrim(cx) == cx.int_type);
    let discriminant_type_metadata = type_metadata(cx, ty::mk_int(), span);

    let variants: &[@ty::VariantInfo] = *ty::enum_variants(cx.tcx, enum_def_id);

    let enumerators_metadata: ~[DIDescriptor] = variants
        .iter()
        .transform(|v| {
            let name: &str = cx.sess.str_of(v.name);
            let discriminant_value = v.disr_val as c_ulonglong;

            do name.as_c_str |name| {
                unsafe {
                    llvm::LLVMDIBuilderCreateEnumerator(
                        DIB(cx),
                        name,
                        discriminant_value)
                }
            }
        })
        .collect();

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, loc.file.name);

    let discriminant_type_metadata = do enum_name.as_c_str |enum_name| {
        unsafe {
            llvm::LLVMDIBuilderCreateEnumerationType(
                DIB(cx),
                file_metadata,
                enum_name,
                file_metadata,
                loc.line as c_uint,
                bytes_to_bits(discriminant_size),
                bytes_to_bits(discriminant_align),
                create_DIArray(DIB(cx), enumerators_metadata),
                discriminant_type_metadata)
        }
    };

    let type_rep = adt::represent_type(cx, enum_type);

    match *type_rep {
        adt::CEnum(*) => {
            return discriminant_type_metadata;
        }
        adt::Univariant(ref struct_def, _) => {
            assert!(variants.len() == 1);
            return adt_struct_metadata(cx, struct_def, variants[0], None, span);
        }
        adt::General(ref struct_defs) => {
            let variants_member_metadata: ~[DIDescriptor] = do struct_defs
                .iter()
                .enumerate()
                .transform |(i, struct_def)| {
                    let variant_type_metadata = adt_struct_metadata(
                        cx,
                        struct_def,
                        variants[i],
                        Some(discriminant_type_metadata),
                        span);

                    do "".as_c_str |name| {
                        unsafe {
                            llvm::LLVMDIBuilderCreateMemberType(
                                DIB(cx),
                                file_metadata,
                                name,
                                file_metadata,
                                loc.line as c_uint,
                                bytes_to_bits(struct_def.size as uint),
                                bytes_to_bits(struct_def.align as uint),
                                bytes_to_bits(0),
                                0,
                                variant_type_metadata)
                        }
                    }
            }.collect();

            let enum_llvm_type = type_of::type_of(cx, enum_type);
            let (enum_type_size, enum_type_align) = size_and_align_of(cx, enum_llvm_type);

            return do enum_name.as_c_str |enum_name| {
                unsafe {
                    llvm::LLVMDIBuilderCreateUnionType(
                    DIB(cx),
                    file_metadata,
                    enum_name,
                    file_metadata,
                    loc.line as c_uint,
                    bytes_to_bits(enum_type_size),
                    bytes_to_bits(enum_type_align),
                    0, // Flags
                    create_DIArray(DIB(cx), variants_member_metadata),
                    0) // RuntimeLang
            }};
        }
        adt::NullablePointer { nonnull: ref struct_def, nndiscr, _ } => {
            return adt_struct_metadata(cx, struct_def, variants[nndiscr], None, span);
        }
    }

    fn adt_struct_metadata(cx: &mut CrateContext,
                                  struct_def: &adt::Struct,
                                  variant_info: &ty::VariantInfo,
                                  discriminant_type_metadata: Option<DIType>,
                                  span: span)
                               -> DICompositeType
    {
        let arg_llvm_types: ~[Type] = do struct_def.fields.map |&ty| { type_of::type_of(cx, ty) };
        let arg_metadata: ~[DIType] = do struct_def.fields.iter().enumerate()
            .transform |(i, &ty)| {
                match discriminant_type_metadata {
                    Some(metadata) if i == 0 => metadata,
                    _                        => type_metadata(cx, ty, span)
                }
        }.collect();

        let mut arg_names = match variant_info.arg_names {
            Some(ref names) => do names.map |ident| { cx.sess.str_of(*ident).to_owned() },
            None => do variant_info.args.map |_| { ~"" }
        };

        if discriminant_type_metadata.is_some() {
            arg_names.insert(0, ~"");
        }

        let variant_llvm_type = Type::struct_(arg_llvm_types, struct_def.packed);
        let variant_name: &str = cx.sess.str_of(variant_info.name);

        return composite_type_metadata(
            cx,
            variant_llvm_type,
            variant_name,
            arg_llvm_types,
            arg_names,
            arg_metadata,
            span);
    }
}

/// Creates debug information for a composite type, that is, anything that results in a LLVM struct.
///
/// Examples of Rust types to use this are: structs, tuples, boxes, vecs, and enums.
fn composite_type_metadata(cx: &mut CrateContext,
                           composite_llvm_type: Type,
                           composite_type_name: &str,
                           member_llvm_types: &[Type],
                           member_names: &[~str],
                           member_type_metadata: &[DIType],
                           span: span)
                        -> DICompositeType {

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, loc.file.name);

    let (composite_size, composite_align) = size_and_align_of(cx, composite_llvm_type);

    let member_metadata: ~[DIDescriptor] = member_llvm_types
        .iter()
        .enumerate()
        .transform(|(i, &member_llvm_type)| {
            let (member_size, member_align) = size_and_align_of(cx, member_llvm_type);
            let member_offset = machine::llelement_offset(cx, composite_llvm_type, i);
            let member_name: &str = member_names[i];

            do member_name.as_c_str |member_name| {
                unsafe {
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
                }
            }
        })
        .collect();

    return do composite_type_name.as_c_str |name| {
        unsafe {
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
    }
    };
}

fn boxed_type_metadata(cx: &mut CrateContext,
                       content_type_name: Option<&str>,
                       content_llvm_type: Type,
                       content_type_metadata: DIType,
                       span: span)
                    -> DICompositeType {

    let box_type_name = match content_type_name {
        Some(content_type_name) => fmt!("Boxed<%s>", content_type_name),
        None                    => ~"BoxedType"
    };

    let box_llvm_type = Type::box(cx, &content_llvm_type);
    let member_llvm_types = box_llvm_type.field_types();
    let member_names = [~"refcnt", ~"tydesc", ~"prev", ~"next", ~"val"];

    assert!(box_layout_is_correct(cx, member_llvm_types, content_llvm_type));

    let int_type = ty::mk_int();
    let nil_pointer_type = ty::mk_nil_ptr(cx.tcx);

    let member_types_metadata = [
        type_metadata(cx, int_type, span),
        type_metadata(cx, nil_pointer_type, span),
        type_metadata(cx, nil_pointer_type, span),
        type_metadata(cx, nil_pointer_type, span),
        content_type_metadata
    ];

    return composite_type_metadata(
        cx,
        box_llvm_type,
        box_type_name,
        member_llvm_types,
        member_names,
        member_types_metadata,
        span);

    // Unfortunately, we cannot assert anything but the correct types here---and not whether the
    // 'next' and 'prev' pointers are in the correct order.
    fn box_layout_is_correct(cx: &CrateContext,
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

fn fixed_vec_metadata(cx: &mut CrateContext,
                      element_type: ty::t,
                      len: uint,
                      span: span)
                   -> DIType {
    let element_type_metadata = type_metadata(cx, element_type, span);
    let element_llvm_type = type_of::type_of(cx, element_type);
    let (element_type_size, element_type_align) = size_and_align_of(cx, element_llvm_type);

    let subrange = unsafe {
        llvm::LLVMDIBuilderGetOrCreateSubrange(
        DIB(cx),
        0,
        len as c_longlong)
    };

    let subscripts = create_DIArray(DIB(cx), [subrange]);
    return unsafe {
        llvm::LLVMDIBuilderCreateArrayType(
            DIB(cx),
            bytes_to_bits(element_type_size * len),
            bytes_to_bits(element_type_align),
            element_type_metadata,
            subscripts)
    };
}

fn vec_metadata(cx: &mut CrateContext,
                element_type: ty::t,
                span: span)
             -> DICompositeType {

    let element_type_metadata = type_metadata(cx, element_type, span);
    let element_llvm_type = type_of::type_of(cx, element_type);
    let (element_size, element_align) = size_and_align_of(cx, element_llvm_type);

    let vec_llvm_type = Type::vec(cx.sess.targ_cfg.arch, &element_llvm_type);
    let vec_type_name: &str = fmt!("[%s]", ty_to_str(cx.tcx, element_type));

    let member_llvm_types = vec_llvm_type.field_types();
    let member_names = &[~"fill", ~"alloc", ~"elements"];

    let int_type_metadata = type_metadata(cx, ty::mk_int(), span);
    let array_type_metadata = unsafe {
        llvm::LLVMDIBuilderCreateArrayType(
            DIB(cx),
            bytes_to_bits(element_size),
            bytes_to_bits(element_align),
            element_type_metadata,
            create_DIArray(DIB(cx), [llvm::LLVMDIBuilderGetOrCreateSubrange(DIB(cx), 0, 0)]))
    };

    //                           fill               alloc              elements
    let member_type_metadata = &[int_type_metadata, int_type_metadata, array_type_metadata];

    return composite_type_metadata(
        cx,
        vec_llvm_type,
        vec_type_name,
        member_llvm_types,
        member_names,
        member_type_metadata,
        span);
}

fn boxed_vec_metadata(cx: &mut CrateContext,
                      element_type: ty::t,
                      span: span)
                   -> DICompositeType {

    let element_llvm_type = type_of::type_of(cx, element_type);
    let vec_llvm_type = Type::vec(cx.sess.targ_cfg.arch, &element_llvm_type);
    let vec_type_name: &str = fmt!("[%s]", ty_to_str(cx.tcx, element_type));
    let vec_metadata = vec_metadata(cx, element_type, span);

    return boxed_type_metadata(
        cx,
        Some(vec_type_name),
        vec_llvm_type,
        vec_metadata,
        span);
}

fn vec_slice_metadata(cx: &mut CrateContext,
                      vec_type: ty::t,
                      element_type: ty::t,
                      span: span)
                   -> DICompositeType {

    debug!("vec_slice_metadata: %?", ty::get(vec_type));

    let slice_llvm_type = type_of::type_of(cx, vec_type);
    let slice_type_name = ty_to_str(cx.tcx, vec_type);

    let member_llvm_types = slice_llvm_type.field_types();
    let member_names = &[~"data_ptr", ~"size_in_bytes"];

    assert!(slice_layout_is_correct(cx, member_llvm_types, element_type));

    let data_ptr_type = ty::mk_ptr(cx.tcx, ty::mt { ty: element_type, mutbl: ast::m_imm });

    let member_type_metadata = &[type_metadata(cx, data_ptr_type, span),
                                 type_metadata(cx, ty::mk_uint(), span)];

    return composite_type_metadata(
        cx,
        slice_llvm_type,
        slice_type_name,
        member_llvm_types,
        member_names,
        member_type_metadata,
        span);

    fn slice_layout_is_correct(cx: &mut CrateContext,
                               member_llvm_types: &[Type],
                               element_type: ty::t)
                            -> bool {
        member_llvm_types.len() == 2 &&
        member_llvm_types[0] == type_of::type_of(cx, element_type).ptr_to() &&
        member_llvm_types[1] == cx.int_type
    }
}

fn bare_fn_metadata(cx: &mut CrateContext,
                    _fn_ty: ty::t,
                    inputs: ~[ty::t],
                    output: ty::t,
                    span: span)
                 -> DICompositeType {

    debug!("bare_fn_metadata: %?", ty::get(_fn_ty));

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, loc.file.name);

    let nil_pointer_type_metadata = type_metadata(cx, ty::mk_nil_ptr(cx.tcx), span);
    let output_metadata = type_metadata(cx, output, span);
    let output_ptr_metadata = pointer_type_metadata(cx, output, output_metadata);

    let inputs_vals = do inputs.map |arg| { type_metadata(cx, *arg, span) };
    let members = ~[output_ptr_metadata, nil_pointer_type_metadata] + inputs_vals;

    return unsafe {
        llvm::LLVMDIBuilderCreateSubroutineType(
            DIB(cx),
            file_metadata,
            create_DIArray(DIB(cx), members))
    };
}

fn unimplemented_type_metadata(cx: &mut CrateContext, t: ty::t) -> DIType {
    debug!("unimplemented_type_metadata: %?", ty::get(t));

    let name = ty_to_str(cx.tcx, t);
    let metadata = do fmt!("NYI<%s>", name).as_c_str |name| {
        unsafe {
            llvm::LLVMDIBuilderCreateBasicType(
                DIB(cx),
                name,
                0_u64,
                8_u64,
                DW_ATE_unsigned as c_uint)
            }
        };

    return metadata;
}

fn type_metadata(cx: &mut CrateContext,
                 t: ty::t,
                 span: span)
              -> DIType {
    let type_id = ty::type_id(t);
    match dbg_cx(cx).created_types.find(&type_id) {
        Some(type_metadata) => return *type_metadata,
        None => ()
    }

    fn create_pointer_to_box_metadata(cx: &mut CrateContext,
                                      pointer_type: ty::t,
                                      type_in_box: ty::t)
                                   -> DIType {

        let content_type_name: &str = ty_to_str(cx.tcx, type_in_box);
        let content_llvm_type = type_of::type_of(cx, type_in_box);
        let content_type_metadata = type_metadata(
            cx,
            type_in_box,
            codemap::dummy_sp());

        let box_metadata = boxed_type_metadata(
            cx,
            Some(content_type_name),
            content_llvm_type,
            content_type_metadata,
            codemap::dummy_sp());

        pointer_type_metadata(cx, pointer_type, box_metadata)
    }

    debug!("type_metadata: %?", ty::get(t));

    let sty = &ty::get(t).sty;
    let type_metadata = match *sty {
        ty::ty_nil      |
        ty::ty_bot      |
        ty::ty_bool     |
        ty::ty_int(_)   |
        ty::ty_uint(_)  |
        ty::ty_float(_) => {
            basic_type_metadata(cx, t)
        },
        ty::ty_estr(ref vstore) => {
            let i8_t = ty::mk_i8();
            match *vstore {
                ty::vstore_fixed(len) => {
                    fixed_vec_metadata(cx, i8_t, len + 1, span)
                },
                ty::vstore_uniq  => {
                    let vec_metadata = vec_metadata(cx, i8_t, span);
                    pointer_type_metadata(cx, t, vec_metadata)
                }
                ty::vstore_box => {
                    let boxed_vec_metadata = boxed_vec_metadata(cx, i8_t, span);
                    pointer_type_metadata(cx, t, boxed_vec_metadata)
                }
                ty::vstore_slice(_region) => {
                    vec_slice_metadata(cx, t, i8_t, span)
                }
            }
        },
        ty::ty_enum(def_id, ref substs) => {
            enum_metadata(cx, t, def_id, substs, span)
        },
        ty::ty_box(ref mt) => {
            create_pointer_to_box_metadata(cx, t, mt.ty)
        },
        ty::ty_evec(ref mt, ref vstore) => {
            match *vstore {
                ty::vstore_fixed(len) => {
                    fixed_vec_metadata(cx, mt.ty, len, span)
                }
                ty::vstore_uniq if ty::type_contents(cx.tcx, mt.ty).contains_managed() => {
                    let boxed_vec_metadata = boxed_vec_metadata(cx, mt.ty, span);
                    pointer_type_metadata(cx, t, boxed_vec_metadata)
                }
                ty::vstore_uniq => {
                    let vec_metadata = vec_metadata(cx, mt.ty, span);
                    pointer_type_metadata(cx, t, vec_metadata)
                }
                ty::vstore_box => {
                    let boxed_vec_metadata = boxed_vec_metadata(cx, mt.ty, span);
                    pointer_type_metadata(cx, t, boxed_vec_metadata)
                }
                ty::vstore_slice(_) => {
                    vec_slice_metadata(cx, t, mt.ty, span)
                }
            }
        },
        ty::ty_uniq(ref mt) if ty::type_contents(cx.tcx, mt.ty).contains_managed() => {
            create_pointer_to_box_metadata(cx, t, mt.ty)
        },
        ty::ty_uniq(ref mt)    |
        ty::ty_ptr(ref mt)     |
        ty::ty_rptr(_, ref mt) => {
            let pointee = type_metadata(cx, mt.ty, span);
            pointer_type_metadata(cx, t, pointee)
        },
        ty::ty_bare_fn(ref barefnty) => {
            let inputs = barefnty.sig.inputs.map(|a| *a);
            let output = barefnty.sig.output;
            bare_fn_metadata(cx, t, inputs, output, span)
        },
        ty::ty_closure(ref _closurety) => {
            cx.sess.span_note(span, "debuginfo for closure NYI");
            unimplemented_type_metadata(cx, t)
        },
        ty::ty_trait(_did, ref _substs, ref _vstore, _, _bounds) => {
            cx.sess.span_note(span, "debuginfo for trait NYI");
            unimplemented_type_metadata(cx, t)
        },
        ty::ty_struct(did, ref substs) => {
            let fields = ty::struct_fields(cx.tcx, did, substs);
            struct_metadata(cx, t, fields, span)
        },
        ty::ty_tup(ref elements) => {
            tuple_metadata(cx, t, *elements, span)
        },
        _ => cx.sess.bug("debuginfo: unexpected type in type_metadata")
    };

    dbg_cx(cx).created_types.insert(type_id, type_metadata);
    return type_metadata;
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

fn size_and_align_of(cx: &mut CrateContext, llvm_type: Type) -> (uint, uint) {
    (machine::llsize_of_alloc(cx, llvm_type), machine::llalign_of_min(cx, llvm_type))
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
