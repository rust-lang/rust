// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::utils::{debug_context, DIB, span_start, bytes_to_bits, size_and_align_of,
                   get_namespace_and_span_for_item};
use super::{UNKNOWN_FILE_METADATA, UNKNOWN_SCOPE_METADATA,
            UniqueTypeId, FLAGS_NONE};
use super::types::compute_debuginfo_type_name;
use super::create::create_DIArray;
use super::adt::{prepare_struct_metadata, prepare_tuple_metadata, prepare_enum_metadata,
                 composite_type_metadata, MemberDescription};
use super::adt::MemberOffset::ComputedMemberOffset;

use llvm;
use llvm::debuginfo::{DIType, DIFile, DIScope, DIDescriptor};
use trans::type_of;
use trans::common::{CrateContext, FunctionContext, NormalizingClosureTyper};
use trans::type_::Type;
use middle::ty::{self, Ty, ClosureTyper};
use session::config;
use util::ppaux;
use util::common::path2cstr;

use libc::{c_uint, c_longlong};
use std::ffi::CString;
use std::path::Path;
use std::ptr;
use syntax::codemap::Span;
use syntax::{ast, codemap};


const DW_LANG_RUST: c_uint = 0x9000;
#[allow(non_upper_case_globals)]
const DW_ATE_boolean: c_uint = 0x02;
#[allow(non_upper_case_globals)]
const DW_ATE_float: c_uint = 0x04;
#[allow(non_upper_case_globals)]
const DW_ATE_signed: c_uint = 0x05;
#[allow(non_upper_case_globals)]
const DW_ATE_unsigned: c_uint = 0x07;
#[allow(non_upper_case_globals)]
const DW_ATE_unsigned_char: c_uint = 0x08;


// Returns from the enclosing function if the type metadata with the given
// unique id can be found in the type map
macro_rules! return_if_metadata_created_in_meantime {
    ($cx: expr, $unique_type_id: expr) => (
        match debug_context($cx).type_map
                                .borrow()
                                .find_metadata_for_unique_id($unique_type_id) {
            Some(metadata) => return MetadataCreationResult::new(metadata, true),
            None => { /* proceed normally */ }
        };
    )
}

fn fixed_vec_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                unique_type_id: UniqueTypeId,
                                element_type: Ty<'tcx>,
                                len: Option<u64>,
                                span: Span)
                                -> MetadataCreationResult {
    let element_type_metadata = type_metadata(cx, element_type, span);

    return_if_metadata_created_in_meantime!(cx, unique_type_id);

    let element_llvm_type = type_of::type_of(cx, element_type);
    let (element_type_size, element_type_align) = size_and_align_of(cx, element_llvm_type);

    let (array_size_in_bytes, upper_bound) = match len {
        Some(len) => (element_type_size * len, len as c_longlong),
        None => (0, -1)
    };

    let subrange = unsafe {
        llvm::LLVMDIBuilderGetOrCreateSubrange(DIB(cx), 0, upper_bound)
    };

    let subscripts = create_DIArray(DIB(cx), &[subrange]);
    let metadata = unsafe {
        llvm::LLVMDIBuilderCreateArrayType(
            DIB(cx),
            bytes_to_bits(array_size_in_bytes),
            bytes_to_bits(element_type_align),
            element_type_metadata,
            subscripts)
    };

    return MetadataCreationResult::new(metadata, false);
}

fn vec_slice_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                vec_type: Ty<'tcx>,
                                element_type: Ty<'tcx>,
                                unique_type_id: UniqueTypeId,
                                span: Span)
                                -> MetadataCreationResult {
    let data_ptr_type = ty::mk_ptr(cx.tcx(), ty::mt {
        ty: element_type,
        mutbl: ast::MutImmutable
    });

    let element_type_metadata = type_metadata(cx, data_ptr_type, span);

    return_if_metadata_created_in_meantime!(cx, unique_type_id);

    let slice_llvm_type = type_of::type_of(cx, vec_type);
    let slice_type_name = compute_debuginfo_type_name(cx, vec_type, true);

    let member_llvm_types = slice_llvm_type.field_types();
    assert!(slice_layout_is_correct(cx,
                                    &member_llvm_types[..],
                                    element_type));
    let member_descriptions = [
        MemberDescription {
            name: "data_ptr".to_string(),
            llvm_type: member_llvm_types[0],
            type_metadata: element_type_metadata,
            offset: ComputedMemberOffset,
            flags: FLAGS_NONE
        },
        MemberDescription {
            name: "length".to_string(),
            llvm_type: member_llvm_types[1],
            type_metadata: type_metadata(cx, cx.tcx().types.usize, span),
            offset: ComputedMemberOffset,
            flags: FLAGS_NONE
        },
    ];

    assert!(member_descriptions.len() == member_llvm_types.len());

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, &loc.file.name);

    let metadata = composite_type_metadata(cx,
                                           slice_llvm_type,
                                           &slice_type_name[..],
                                           unique_type_id,
                                           &member_descriptions,
                                           UNKNOWN_SCOPE_METADATA,
                                           file_metadata,
                                           span);
    return MetadataCreationResult::new(metadata, false);

    fn slice_layout_is_correct<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                         member_llvm_types: &[Type],
                                         element_type: Ty<'tcx>)
                                         -> bool {
        member_llvm_types.len() == 2 &&
        member_llvm_types[0] == type_of::type_of(cx, element_type).ptr_to() &&
        member_llvm_types[1] == cx.int_type()
    }
}

fn subroutine_type_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                      unique_type_id: UniqueTypeId,
                                      signature: &ty::PolyFnSig<'tcx>,
                                      span: Span)
                                      -> MetadataCreationResult
{
    let signature = ty::erase_late_bound_regions(cx.tcx(), signature);

    let mut signature_metadata: Vec<DIType> = Vec::with_capacity(signature.inputs.len() + 1);

    // return type
    signature_metadata.push(match signature.output {
        ty::FnConverging(ret_ty) => match ret_ty.sty {
            ty::ty_tup(ref tys) if tys.is_empty() => ptr::null_mut(),
            _ => type_metadata(cx, ret_ty, span)
        },
        ty::FnDiverging => diverging_type_metadata(cx)
    });

    // regular arguments
    for &argument_type in &signature.inputs {
        signature_metadata.push(type_metadata(cx, argument_type, span));
    }

    return_if_metadata_created_in_meantime!(cx, unique_type_id);

    return MetadataCreationResult::new(
        unsafe {
            llvm::LLVMDIBuilderCreateSubroutineType(
                DIB(cx),
                UNKNOWN_FILE_METADATA,
                create_DIArray(DIB(cx), &signature_metadata[..]))
        },
        false);
}

// FIXME(1563) This is all a bit of a hack because 'trait pointer' is an ill-
// defined concept. For the case of an actual trait pointer (i.e., Box<Trait>,
// &Trait), trait_object_type should be the whole thing (e.g, Box<Trait>) and
// trait_type should be the actual trait (e.g., Trait). Where the trait is part
// of a DST struct, there is no trait_object_type and the results of this
// function will be a little bit weird.
fn trait_pointer_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                    trait_type: Ty<'tcx>,
                                    trait_object_type: Option<Ty<'tcx>>,
                                    unique_type_id: UniqueTypeId)
                                    -> DIType {
    // The implementation provided here is a stub. It makes sure that the trait
    // type is assigned the correct name, size, namespace, and source location.
    // But it does not describe the trait's methods.

    let def_id = match trait_type.sty {
        ty::ty_trait(ref data) => data.principal_def_id(),
        _ => {
            let pp_type_name = ppaux::ty_to_string(cx.tcx(), trait_type);
            cx.sess().bug(&format!("debuginfo: Unexpected trait-object type in \
                                   trait_pointer_metadata(): {}",
                                   &pp_type_name[..]));
        }
    };

    let trait_object_type = trait_object_type.unwrap_or(trait_type);
    let trait_type_name =
        compute_debuginfo_type_name(cx, trait_object_type, false);

    let (containing_scope, _) = get_namespace_and_span_for_item(cx, def_id);

    let trait_llvm_type = type_of::type_of(cx, trait_object_type);

    composite_type_metadata(cx,
                            trait_llvm_type,
                            &trait_type_name[..],
                            unique_type_id,
                            &[],
                            containing_scope,
                            UNKNOWN_FILE_METADATA,
                            codemap::DUMMY_SP)
}

pub fn type_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                               t: Ty<'tcx>,
                               usage_site_span: Span)
                               -> DIType {
    // Get the unique type id of this type.
    let unique_type_id = {
        let mut type_map = debug_context(cx).type_map.borrow_mut();
        // First, try to find the type in TypeMap. If we have seen it before, we
        // can exit early here.
        match type_map.find_metadata_for_type(t) {
            Some(metadata) => {
                return metadata;
            },
            None => {
                // The Ty is not in the TypeMap but maybe we have already seen
                // an equivalent type (e.g. only differing in region arguments).
                // In order to find out, generate the unique type id and look
                // that up.
                let unique_type_id = type_map.get_unique_type_id_of_type(cx, t);
                match type_map.find_metadata_for_unique_id(unique_type_id) {
                    Some(metadata) => {
                        // There is already an equivalent type in the TypeMap.
                        // Register this Ty as an alias in the cache and
                        // return the cached metadata.
                        type_map.register_type_with_metadata(cx, t, metadata);
                        return metadata;
                    },
                    None => {
                        // There really is no type metadata for this type, so
                        // proceed by creating it.
                        unique_type_id
                    }
                }
            }
        }
    };

    debug!("type_metadata: {:?}", t);

    let sty = &t.sty;
    let MetadataCreationResult { metadata, already_stored_in_typemap } = match *sty {
        ty::ty_bool     |
        ty::ty_char     |
        ty::ty_int(_)   |
        ty::ty_uint(_)  |
        ty::ty_float(_) => {
            MetadataCreationResult::new(basic_type_metadata(cx, t), false)
        }
        ty::ty_tup(ref elements) if elements.is_empty() => {
            MetadataCreationResult::new(basic_type_metadata(cx, t), false)
        }
        ty::ty_enum(def_id, _) => {
            prepare_enum_metadata(cx, t, def_id, unique_type_id, usage_site_span).finalize(cx)
        }
        ty::ty_vec(typ, len) => {
            fixed_vec_metadata(cx, unique_type_id, typ, len.map(|x| x as u64), usage_site_span)
        }
        ty::ty_str => {
            fixed_vec_metadata(cx, unique_type_id, cx.tcx().types.i8, None, usage_site_span)
        }
        ty::ty_trait(..) => {
            MetadataCreationResult::new(
                        trait_pointer_metadata(cx, t, None, unique_type_id),
            false)
        }
        ty::ty_uniq(ty) | ty::ty_ptr(ty::mt{ty, ..}) | ty::ty_rptr(_, ty::mt{ty, ..}) => {
            match ty.sty {
                ty::ty_vec(typ, None) => {
                    vec_slice_metadata(cx, t, typ, unique_type_id, usage_site_span)
                }
                ty::ty_str => {
                    vec_slice_metadata(cx, t, cx.tcx().types.u8, unique_type_id, usage_site_span)
                }
                ty::ty_trait(..) => {
                    MetadataCreationResult::new(
                        trait_pointer_metadata(cx, ty, Some(t), unique_type_id),
                        false)
                }
                _ => {
                    let pointee_metadata = type_metadata(cx, ty, usage_site_span);

                    match debug_context(cx).type_map
                                           .borrow()
                                           .find_metadata_for_unique_id(unique_type_id) {
                        Some(metadata) => return metadata,
                        None => { /* proceed normally */ }
                    };

                    MetadataCreationResult::new(pointer_type_metadata(cx, t, pointee_metadata),
                                                false)
                }
            }
        }
        ty::ty_bare_fn(_, ref barefnty) => {
            subroutine_type_metadata(cx, unique_type_id, &barefnty.sig, usage_site_span)
        }
        ty::ty_closure(def_id, substs) => {
            let typer = NormalizingClosureTyper::new(cx.tcx());
            let sig = typer.closure_type(def_id, substs).sig;
            subroutine_type_metadata(cx, unique_type_id, &sig, usage_site_span)
        }
        ty::ty_struct(def_id, substs) => {
            prepare_struct_metadata(cx,
                                    t,
                                    def_id,
                                    substs,
                                    unique_type_id,
                                    usage_site_span).finalize(cx)
        }
        ty::ty_tup(ref elements) => {
            prepare_tuple_metadata(cx,
                                   t,
                                   &elements[..],
                                   unique_type_id,
                                   usage_site_span).finalize(cx)
        }
        _ => {
            cx.sess().bug(&format!("debuginfo: unexpected type in type_metadata: {:?}",
                                  sty))
        }
    };

    {
        let mut type_map = debug_context(cx).type_map.borrow_mut();

        if already_stored_in_typemap {
            // Also make sure that we already have a TypeMap entry entry for the unique type id.
            let metadata_for_uid = match type_map.find_metadata_for_unique_id(unique_type_id) {
                Some(metadata) => metadata,
                None => {
                    let unique_type_id_str =
                        type_map.get_unique_type_id_as_string(unique_type_id);
                    let error_message = format!("Expected type metadata for unique \
                                                 type id '{}' to already be in \
                                                 the debuginfo::TypeMap but it \
                                                 was not. (Ty = {})",
                                                &unique_type_id_str[..],
                                                ppaux::ty_to_string(cx.tcx(), t));
                    cx.sess().span_bug(usage_site_span, &error_message[..]);
                }
            };

            match type_map.find_metadata_for_type(t) {
                Some(metadata) => {
                    if metadata != metadata_for_uid {
                        let unique_type_id_str =
                            type_map.get_unique_type_id_as_string(unique_type_id);
                        let error_message = format!("Mismatch between Ty and \
                                                     UniqueTypeId maps in \
                                                     debuginfo::TypeMap. \
                                                     UniqueTypeId={}, Ty={}",
                            &unique_type_id_str[..],
                            ppaux::ty_to_string(cx.tcx(), t));
                        cx.sess().span_bug(usage_site_span, &error_message[..]);
                    }
                }
                None => {
                    type_map.register_type_with_metadata(cx, t, metadata);
                }
            }
        } else {
            type_map.register_type_with_metadata(cx, t, metadata);
            type_map.register_unique_id_with_metadata(cx, unique_type_id, metadata);
        }
    }

    metadata
}

pub fn file_metadata(cx: &CrateContext, full_path: &str) -> DIFile {
    match debug_context(cx).created_files.borrow().get(full_path) {
        Some(file_metadata) => return *file_metadata,
        None => ()
    }

    debug!("file_metadata: {}", full_path);

    // FIXME (#9639): This needs to handle non-utf8 paths
    let work_dir = cx.sess().working_dir.to_str().unwrap();
    let file_name =
        if full_path.starts_with(work_dir) {
            &full_path[work_dir.len() + 1..full_path.len()]
        } else {
            full_path
        };

    let file_name = CString::new(file_name).unwrap();
    let work_dir = CString::new(work_dir).unwrap();
    let file_metadata = unsafe {
        llvm::LLVMDIBuilderCreateFile(DIB(cx), file_name.as_ptr(),
                                      work_dir.as_ptr())
    };

    let mut created_files = debug_context(cx).created_files.borrow_mut();
    created_files.insert(full_path.to_string(), file_metadata);
    return file_metadata;
}

/// Finds the scope metadata node for the given AST node.
pub fn scope_metadata(fcx: &FunctionContext,
                  node_id: ast::NodeId,
                  error_reporting_span: Span)
               -> DIScope {
    let scope_map = &fcx.debug_context
                        .get_ref(fcx.ccx, error_reporting_span)
                        .scope_map;
    match scope_map.borrow().get(&node_id).cloned() {
        Some(scope_metadata) => scope_metadata,
        None => {
            let node = fcx.ccx.tcx().map.get(node_id);

            fcx.ccx.sess().span_bug(error_reporting_span,
                &format!("debuginfo: Could not find scope info for node {:?}",
                        node));
        }
    }
}

fn diverging_type_metadata(cx: &CrateContext) -> DIType {
    unsafe {
        llvm::LLVMDIBuilderCreateBasicType(
            DIB(cx),
            "!\0".as_ptr() as *const _,
            bytes_to_bits(0),
            bytes_to_bits(0),
            DW_ATE_unsigned)
    }
}

fn basic_type_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                 t: Ty<'tcx>) -> DIType {

    debug!("basic_type_metadata: {:?}", t);

    let (name, encoding) = match t.sty {
        ty::ty_tup(ref elements) if elements.is_empty() =>
            ("()".to_string(), DW_ATE_unsigned),
        ty::ty_bool => ("bool".to_string(), DW_ATE_boolean),
        ty::ty_char => ("char".to_string(), DW_ATE_unsigned_char),
        ty::ty_int(int_ty) => match int_ty {
            ast::TyIs => ("isize".to_string(), DW_ATE_signed),
            ast::TyI8 => ("i8".to_string(), DW_ATE_signed),
            ast::TyI16 => ("i16".to_string(), DW_ATE_signed),
            ast::TyI32 => ("i32".to_string(), DW_ATE_signed),
            ast::TyI64 => ("i64".to_string(), DW_ATE_signed)
        },
        ty::ty_uint(uint_ty) => match uint_ty {
            ast::TyUs => ("usize".to_string(), DW_ATE_unsigned),
            ast::TyU8 => ("u8".to_string(), DW_ATE_unsigned),
            ast::TyU16 => ("u16".to_string(), DW_ATE_unsigned),
            ast::TyU32 => ("u32".to_string(), DW_ATE_unsigned),
            ast::TyU64 => ("u64".to_string(), DW_ATE_unsigned)
        },
        ty::ty_float(float_ty) => match float_ty {
            ast::TyF32 => ("f32".to_string(), DW_ATE_float),
            ast::TyF64 => ("f64".to_string(), DW_ATE_float),
        },
        _ => cx.sess().bug("debuginfo::basic_type_metadata - t is invalid type")
    };

    let llvm_type = type_of::type_of(cx, t);
    let (size, align) = size_and_align_of(cx, llvm_type);
    let name = CString::new(name).unwrap();
    let ty_metadata = unsafe {
        llvm::LLVMDIBuilderCreateBasicType(
            DIB(cx),
            name.as_ptr(),
            bytes_to_bits(size),
            bytes_to_bits(align),
            encoding)
    };

    return ty_metadata;
}

fn pointer_type_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                   pointer_type: Ty<'tcx>,
                                   pointee_type_metadata: DIType)
                                   -> DIType {
    let pointer_llvm_type = type_of::type_of(cx, pointer_type);
    let (pointer_size, pointer_align) = size_and_align_of(cx, pointer_llvm_type);
    let name = compute_debuginfo_type_name(cx, pointer_type, false);
    let name = CString::new(name).unwrap();
    let ptr_metadata = unsafe {
        llvm::LLVMDIBuilderCreatePointerType(
            DIB(cx),
            pointee_type_metadata,
            bytes_to_bits(pointer_size),
            bytes_to_bits(pointer_align),
            name.as_ptr())
    };
    return ptr_metadata;
}

pub fn compile_unit_metadata(cx: &CrateContext) -> DIDescriptor {
    let work_dir = &cx.sess().working_dir;
    let compile_unit_name = match cx.sess().local_crate_source_file {
        None => fallback_path(cx),
        Some(ref abs_path) => {
            if abs_path.is_relative() {
                cx.sess().warn("debuginfo: Invalid path to crate's local root source file!");
                fallback_path(cx)
            } else {
                match abs_path.relative_from(work_dir) {
                    Some(ref p) if p.is_relative() => {
                        if p.starts_with(Path::new("./")) {
                            path2cstr(p)
                        } else {
                            path2cstr(&Path::new(".").join(p))
                        }
                    }
                    _ => fallback_path(cx)
                }
            }
        }
    };

    debug!("compile_unit_metadata: {:?}", compile_unit_name);
    let producer = format!("rustc version {}",
                           (option_env!("CFG_VERSION")).expect("CFG_VERSION"));

    let compile_unit_name = compile_unit_name.as_ptr();
    let work_dir = path2cstr(&work_dir);
    let producer = CString::new(producer).unwrap();
    let flags = "\0";
    let split_name = "\0";
    return unsafe {
        llvm::LLVMDIBuilderCreateCompileUnit(
            debug_context(cx).builder,
            DW_LANG_RUST,
            compile_unit_name,
            work_dir.as_ptr(),
            producer.as_ptr(),
            cx.sess().opts.optimize != config::No,
            flags.as_ptr() as *const _,
            0,
            split_name.as_ptr() as *const _)
    };

    fn fallback_path(cx: &CrateContext) -> CString {
        CString::new(cx.link_meta().crate_name.clone()).unwrap()
    }
}

pub struct MetadataCreationResult {
    metadata: DIType,
    already_stored_in_typemap: bool
}

impl MetadataCreationResult {
    pub fn new(metadata: DIType, already_stored_in_typemap: bool) -> MetadataCreationResult {
        MetadataCreationResult {
            metadata: metadata,
            already_stored_in_typemap: already_stored_in_typemap
        }
    }
}
