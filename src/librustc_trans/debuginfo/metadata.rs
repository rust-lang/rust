// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::RecursiveTypeDescription::*;
use self::MemberOffset::*;
use self::MemberDescriptionFactory::*;
use self::EnumDiscriminantInfo::*;

use super::utils::{debug_context, DIB, span_start, bytes_to_bits, size_and_align_of,
                   get_namespace_and_span_for_item, create_DIArray, is_node_local_to_unit};
use super::namespace::mangled_name_of_item;
use super::type_names::compute_debuginfo_type_name;
use super::{CrateDebugContext};
use context::SharedCrateContext;
use session::Session;

use llvm::{self, ValueRef};
use llvm::debuginfo::{DIType, DIFile, DIScope, DIDescriptor,
                      DICompositeType, DILexicalBlock, DIFlags};

use rustc::hir::def::CtorKind;
use rustc::hir::def_id::DefId;
use rustc::ty::fold::TypeVisitor;
use rustc::ty::subst::Substs;
use rustc::ty::util::TypeIdHasher;
use rustc::hir;
use rustc_data_structures::ToHex;
use {type_of, machine, monomorphize};
use common::CrateContext;
use type_::Type;
use rustc::ty::{self, AdtKind, Ty, layout};
use session::config;
use util::nodemap::FxHashMap;
use util::common::path2cstr;

use libc::{c_uint, c_longlong};
use std::ffi::CString;
use std::path::Path;
use std::ptr;
use syntax::ast;
use syntax::symbol::{Interner, InternedString};
use syntax_pos::{self, Span};


// From DWARF 5.
// See http://www.dwarfstd.org/ShowIssue.php?issue=140129.1
const DW_LANG_RUST: c_uint = 0x1c;
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

pub const UNKNOWN_LINE_NUMBER: c_uint = 0;
pub const UNKNOWN_COLUMN_NUMBER: c_uint = 0;

// ptr::null() doesn't work :(
pub const NO_SCOPE_METADATA: DIScope = (0 as DIScope);

#[derive(Copy, Debug, Hash, Eq, PartialEq, Clone)]
pub struct UniqueTypeId(ast::Name);

// The TypeMap is where the CrateDebugContext holds the type metadata nodes
// created so far. The metadata nodes are indexed by UniqueTypeId, and, for
// faster lookup, also by Ty. The TypeMap is responsible for creating
// UniqueTypeIds.
pub struct TypeMap<'tcx> {
    // The UniqueTypeIds created so far
    unique_id_interner: Interner,
    // A map from UniqueTypeId to debuginfo metadata for that type. This is a 1:1 mapping.
    unique_id_to_metadata: FxHashMap<UniqueTypeId, DIType>,
    // A map from types to debuginfo metadata. This is a N:1 mapping.
    type_to_metadata: FxHashMap<Ty<'tcx>, DIType>,
    // A map from types to UniqueTypeId. This is a N:1 mapping.
    type_to_unique_id: FxHashMap<Ty<'tcx>, UniqueTypeId>
}

impl<'tcx> TypeMap<'tcx> {
    pub fn new() -> TypeMap<'tcx> {
        TypeMap {
            unique_id_interner: Interner::new(),
            type_to_metadata: FxHashMap(),
            unique_id_to_metadata: FxHashMap(),
            type_to_unique_id: FxHashMap(),
        }
    }

    // Adds a Ty to metadata mapping to the TypeMap. The method will fail if
    // the mapping already exists.
    fn register_type_with_metadata<'a>(&mut self,
                                       type_: Ty<'tcx>,
                                       metadata: DIType) {
        if self.type_to_metadata.insert(type_, metadata).is_some() {
            bug!("Type metadata for Ty '{}' is already in the TypeMap!", type_);
        }
    }

    // Adds a UniqueTypeId to metadata mapping to the TypeMap. The method will
    // fail if the mapping already exists.
    fn register_unique_id_with_metadata(&mut self,
                                        unique_type_id: UniqueTypeId,
                                        metadata: DIType) {
        if self.unique_id_to_metadata.insert(unique_type_id, metadata).is_some() {
            bug!("Type metadata for unique id '{}' is already in the TypeMap!",
                 self.get_unique_type_id_as_string(unique_type_id));
        }
    }

    fn find_metadata_for_type(&self, type_: Ty<'tcx>) -> Option<DIType> {
        self.type_to_metadata.get(&type_).cloned()
    }

    fn find_metadata_for_unique_id(&self, unique_type_id: UniqueTypeId) -> Option<DIType> {
        self.unique_id_to_metadata.get(&unique_type_id).cloned()
    }

    // Get the string representation of a UniqueTypeId. This method will fail if
    // the id is unknown.
    fn get_unique_type_id_as_string(&self, unique_type_id: UniqueTypeId) -> &str {
        let UniqueTypeId(interner_key) = unique_type_id;
        self.unique_id_interner.get(interner_key)
    }

    // Get the UniqueTypeId for the given type. If the UniqueTypeId for the given
    // type has been requested before, this is just a table lookup. Otherwise an
    // ID will be generated and stored for later lookup.
    fn get_unique_type_id_of_type<'a>(&mut self, cx: &CrateContext<'a, 'tcx>,
                                      type_: Ty<'tcx>) -> UniqueTypeId {
        // Let's see if we already have something in the cache
        match self.type_to_unique_id.get(&type_).cloned() {
            Some(unique_type_id) => return unique_type_id,
            None => { /* generate one */}
        };

        // The hasher we are using to generate the UniqueTypeId. We want
        // something that provides more than the 64 bits of the DefaultHasher.

        let mut type_id_hasher = TypeIdHasher::<[u8; 20]>::new(cx.tcx());
        type_id_hasher.visit_ty(type_);

        let unique_type_id = type_id_hasher.finish().to_hex();
        let key = self.unique_id_interner.intern(&unique_type_id);
        self.type_to_unique_id.insert(type_, UniqueTypeId(key));

        return UniqueTypeId(key);
    }

    // Get the UniqueTypeId for an enum variant. Enum variants are not really
    // types of their own, so they need special handling. We still need a
    // UniqueTypeId for them, since to debuginfo they *are* real types.
    fn get_unique_type_id_of_enum_variant<'a>(&mut self,
                                              cx: &CrateContext<'a, 'tcx>,
                                              enum_type: Ty<'tcx>,
                                              variant_name: &str)
                                              -> UniqueTypeId {
        let enum_type_id = self.get_unique_type_id_of_type(cx, enum_type);
        let enum_variant_type_id = format!("{}::{}",
                                           self.get_unique_type_id_as_string(enum_type_id),
                                           variant_name);
        let interner_key = self.unique_id_interner.intern(&enum_variant_type_id);
        UniqueTypeId(interner_key)
    }
}

// A description of some recursive type. It can either be already finished (as
// with FinalMetadata) or it is not yet finished, but contains all information
// needed to generate the missing parts of the description. See the
// documentation section on Recursive Types at the top of this file for more
// information.
enum RecursiveTypeDescription<'tcx> {
    UnfinishedMetadata {
        unfinished_type: Ty<'tcx>,
        unique_type_id: UniqueTypeId,
        metadata_stub: DICompositeType,
        llvm_type: Type,
        member_description_factory: MemberDescriptionFactory<'tcx>,
    },
    FinalMetadata(DICompositeType)
}

fn create_and_register_recursive_type_forward_declaration<'a, 'tcx>(
    cx: &CrateContext<'a, 'tcx>,
    unfinished_type: Ty<'tcx>,
    unique_type_id: UniqueTypeId,
    metadata_stub: DICompositeType,
    llvm_type: Type,
    member_description_factory: MemberDescriptionFactory<'tcx>)
 -> RecursiveTypeDescription<'tcx> {

    // Insert the stub into the TypeMap in order to allow for recursive references
    let mut type_map = debug_context(cx).type_map.borrow_mut();
    type_map.register_unique_id_with_metadata(unique_type_id, metadata_stub);
    type_map.register_type_with_metadata(unfinished_type, metadata_stub);

    UnfinishedMetadata {
        unfinished_type: unfinished_type,
        unique_type_id: unique_type_id,
        metadata_stub: metadata_stub,
        llvm_type: llvm_type,
        member_description_factory: member_description_factory,
    }
}

impl<'tcx> RecursiveTypeDescription<'tcx> {
    // Finishes up the description of the type in question (mostly by providing
    // descriptions of the fields of the given type) and returns the final type
    // metadata.
    fn finalize<'a>(&self, cx: &CrateContext<'a, 'tcx>) -> MetadataCreationResult {
        match *self {
            FinalMetadata(metadata) => MetadataCreationResult::new(metadata, false),
            UnfinishedMetadata {
                unfinished_type,
                unique_type_id,
                metadata_stub,
                llvm_type,
                ref member_description_factory,
                ..
            } => {
                // Make sure that we have a forward declaration of the type in
                // the TypeMap so that recursive references are possible. This
                // will always be the case if the RecursiveTypeDescription has
                // been properly created through the
                // create_and_register_recursive_type_forward_declaration()
                // function.
                {
                    let type_map = debug_context(cx).type_map.borrow();
                    if type_map.find_metadata_for_unique_id(unique_type_id).is_none() ||
                       type_map.find_metadata_for_type(unfinished_type).is_none() {
                        bug!("Forward declaration of potentially recursive type \
                              '{:?}' was not found in TypeMap!",
                             unfinished_type);
                    }
                }

                // ... then create the member descriptions ...
                let member_descriptions =
                    member_description_factory.create_member_descriptions(cx);

                // ... and attach them to the stub to complete it.
                set_members_of_composite_type(cx,
                                              metadata_stub,
                                              llvm_type,
                                              &member_descriptions[..]);
                return MetadataCreationResult::new(metadata_stub, true);
            }
        }
    }
}

// Returns from the enclosing function if the type metadata with the given
// unique id can be found in the type map
macro_rules! return_if_metadata_created_in_meantime {
    ($cx: expr, $unique_type_id: expr) => (
        match debug_context($cx).type_map
                                .borrow()
                                .find_metadata_for_unique_id($unique_type_id) {
            Some(metadata) => return MetadataCreationResult::new(metadata, true),
            None => { /* proceed normally */ }
        }
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
        llvm::LLVMRustDIBuilderGetOrCreateSubrange(DIB(cx), 0, upper_bound)
    };

    let subscripts = create_DIArray(DIB(cx), &[subrange]);
    let metadata = unsafe {
        llvm::LLVMRustDIBuilderCreateArrayType(
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
    let data_ptr_type = cx.tcx().mk_ptr(ty::TypeAndMut {
        ty: element_type,
        mutbl: hir::MutImmutable
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
            flags: DIFlags::FlagZero,
        },
        MemberDescription {
            name: "length".to_string(),
            llvm_type: member_llvm_types[1],
            type_metadata: type_metadata(cx, cx.tcx().types.usize, span),
            offset: ComputedMemberOffset,
            flags: DIFlags::FlagZero,
        },
    ];

    assert!(member_descriptions.len() == member_llvm_types.len());

    let loc = span_start(cx, span);
    let file_metadata = file_metadata(cx, &loc.file.name, &loc.file.abs_path);

    let metadata = composite_type_metadata(cx,
                                           slice_llvm_type,
                                           &slice_type_name[..],
                                           unique_type_id,
                                           &member_descriptions,
                                           NO_SCOPE_METADATA,
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
    let signature = cx.tcx().erase_late_bound_regions(signature);

    let mut signature_metadata: Vec<DIType> = Vec::with_capacity(signature.inputs().len() + 1);

    // return type
    signature_metadata.push(match signature.output().sty {
        ty::TyTuple(ref tys) if tys.is_empty() => ptr::null_mut(),
        _ => type_metadata(cx, signature.output(), span)
    });

    // regular arguments
    for &argument_type in signature.inputs() {
        signature_metadata.push(type_metadata(cx, argument_type, span));
    }

    return_if_metadata_created_in_meantime!(cx, unique_type_id);

    return MetadataCreationResult::new(
        unsafe {
            llvm::LLVMRustDIBuilderCreateSubroutineType(
                DIB(cx),
                unknown_file_metadata(cx),
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

    let containing_scope = match trait_type.sty {
        ty::TyDynamic(ref data, ..) => if let Some(principal) = data.principal() {
            let def_id = principal.def_id();
            get_namespace_and_span_for_item(cx, def_id).0
        } else {
            NO_SCOPE_METADATA
        },
        _ => {
            bug!("debuginfo: Unexpected trait-object type in \
                  trait_pointer_metadata(): {:?}",
                 trait_type);
        }
    };

    let trait_object_type = trait_object_type.unwrap_or(trait_type);
    let trait_type_name =
        compute_debuginfo_type_name(cx, trait_object_type, false);

    let trait_llvm_type = type_of::type_of(cx, trait_object_type);
    let file_metadata = unknown_file_metadata(cx);

    composite_type_metadata(cx,
                            trait_llvm_type,
                            &trait_type_name[..],
                            unique_type_id,
                            &[],
                            containing_scope,
                            file_metadata,
                            syntax_pos::DUMMY_SP)
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
                        type_map.register_type_with_metadata(t, metadata);
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
        ty::TyNever    |
        ty::TyBool     |
        ty::TyChar     |
        ty::TyInt(_)   |
        ty::TyUint(_)  |
        ty::TyFloat(_) => {
            MetadataCreationResult::new(basic_type_metadata(cx, t), false)
        }
        ty::TyTuple(ref elements) if elements.is_empty() => {
            MetadataCreationResult::new(basic_type_metadata(cx, t), false)
        }
        ty::TyArray(typ, len) => {
            fixed_vec_metadata(cx, unique_type_id, typ, Some(len as u64), usage_site_span)
        }
        ty::TySlice(typ) => {
            fixed_vec_metadata(cx, unique_type_id, typ, None, usage_site_span)
        }
        ty::TyStr => {
            fixed_vec_metadata(cx, unique_type_id, cx.tcx().types.i8, None, usage_site_span)
        }
        ty::TyDynamic(..) => {
            MetadataCreationResult::new(
                        trait_pointer_metadata(cx, t, None, unique_type_id),
            false)
        }
        ty::TyBox(ty) |
        ty::TyRawPtr(ty::TypeAndMut{ty, ..}) |
        ty::TyRef(_, ty::TypeAndMut{ty, ..}) => {
            match ty.sty {
                ty::TySlice(typ) => {
                    vec_slice_metadata(cx, t, typ, unique_type_id, usage_site_span)
                }
                ty::TyStr => {
                    vec_slice_metadata(cx, t, cx.tcx().types.u8, unique_type_id, usage_site_span)
                }
                ty::TyDynamic(..) => {
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
        ty::TyFnDef(.., ref barefnty) | ty::TyFnPtr(ref barefnty) => {
            let fn_metadata = subroutine_type_metadata(cx,
                                                       unique_type_id,
                                                       &barefnty.sig,
                                                       usage_site_span).metadata;
            match debug_context(cx).type_map
                                   .borrow()
                                   .find_metadata_for_unique_id(unique_type_id) {
                Some(metadata) => return metadata,
                None => { /* proceed normally */ }
            };

            // This is actually a function pointer, so wrap it in pointer DI
            MetadataCreationResult::new(pointer_type_metadata(cx, t, fn_metadata), false)

        }
        ty::TyClosure(def_id, substs) => {
            let upvar_tys : Vec<_> = substs.upvar_tys(def_id, cx.tcx()).collect();
            prepare_tuple_metadata(cx,
                                   t,
                                   &upvar_tys,
                                   unique_type_id,
                                   usage_site_span).finalize(cx)
        }
        ty::TyAdt(def, ..) => match def.adt_kind() {
            AdtKind::Struct => {
                prepare_struct_metadata(cx,
                                        t,
                                        unique_type_id,
                                        usage_site_span).finalize(cx)
            }
            AdtKind::Union => {
                prepare_union_metadata(cx,
                                    t,
                                    unique_type_id,
                                    usage_site_span).finalize(cx)
            }
            AdtKind::Enum => {
                prepare_enum_metadata(cx,
                                    t,
                                    def.did,
                                    unique_type_id,
                                    usage_site_span).finalize(cx)
            }
        },
        ty::TyTuple(ref elements) => {
            prepare_tuple_metadata(cx,
                                   t,
                                   &elements[..],
                                   unique_type_id,
                                   usage_site_span).finalize(cx)
        }
        _ => {
            bug!("debuginfo: unexpected type in type_metadata: {:?}", sty)
        }
    };

    {
        let mut type_map = debug_context(cx).type_map.borrow_mut();

        if already_stored_in_typemap {
            // Also make sure that we already have a TypeMap entry for the unique type id.
            let metadata_for_uid = match type_map.find_metadata_for_unique_id(unique_type_id) {
                Some(metadata) => metadata,
                None => {
                    span_bug!(usage_site_span,
                              "Expected type metadata for unique \
                               type id '{}' to already be in \
                               the debuginfo::TypeMap but it \
                               was not. (Ty = {})",
                              type_map.get_unique_type_id_as_string(unique_type_id),
                              t);
                }
            };

            match type_map.find_metadata_for_type(t) {
                Some(metadata) => {
                    if metadata != metadata_for_uid {
                        span_bug!(usage_site_span,
                                  "Mismatch between Ty and \
                                   UniqueTypeId maps in \
                                   debuginfo::TypeMap. \
                                   UniqueTypeId={}, Ty={}",
                                  type_map.get_unique_type_id_as_string(unique_type_id),
                                  t);
                    }
                }
                None => {
                    type_map.register_type_with_metadata(t, metadata);
                }
            }
        } else {
            type_map.register_type_with_metadata(t, metadata);
            type_map.register_unique_id_with_metadata(unique_type_id, metadata);
        }
    }

    metadata
}

pub fn file_metadata(cx: &CrateContext, path: &str, full_path: &Option<String>) -> DIFile {
    // FIXME (#9639): This needs to handle non-utf8 paths
    let work_dir = cx.sess().working_dir.to_str().unwrap();
    let file_name =
        full_path.as_ref().map(|p| p.as_str()).unwrap_or_else(|| {
            if path.starts_with(work_dir) {
                &path[work_dir.len() + 1..path.len()]
            } else {
                path
            }
        });

    file_metadata_(cx, path, file_name, &work_dir)
}

pub fn unknown_file_metadata(cx: &CrateContext) -> DIFile {
    // Regular filenames should not be empty, so we abuse an empty name as the
    // key for the special unknown file metadata
    file_metadata_(cx, "", "<unknown>", "")

}

fn file_metadata_(cx: &CrateContext, key: &str, file_name: &str, work_dir: &str) -> DIFile {
    if let Some(file_metadata) = debug_context(cx).created_files.borrow().get(key) {
        return *file_metadata;
    }

    debug!("file_metadata: file_name: {}, work_dir: {}", file_name, work_dir);

    let file_name = CString::new(file_name).unwrap();
    let work_dir = CString::new(work_dir).unwrap();
    let file_metadata = unsafe {
        llvm::LLVMRustDIBuilderCreateFile(DIB(cx), file_name.as_ptr(),
                                          work_dir.as_ptr())
    };

    let mut created_files = debug_context(cx).created_files.borrow_mut();
    created_files.insert(key.to_string(), file_metadata);
    file_metadata
}

fn basic_type_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                 t: Ty<'tcx>) -> DIType {

    debug!("basic_type_metadata: {:?}", t);

    let (name, encoding) = match t.sty {
        ty::TyNever => ("!", DW_ATE_unsigned),
        ty::TyTuple(ref elements) if elements.is_empty() =>
            ("()", DW_ATE_unsigned),
        ty::TyBool => ("bool", DW_ATE_boolean),
        ty::TyChar => ("char", DW_ATE_unsigned_char),
        ty::TyInt(int_ty) => {
            (int_ty.ty_to_string(), DW_ATE_signed)
        },
        ty::TyUint(uint_ty) => {
            (uint_ty.ty_to_string(), DW_ATE_unsigned)
        },
        ty::TyFloat(float_ty) => {
            (float_ty.ty_to_string(), DW_ATE_float)
        },
        _ => bug!("debuginfo::basic_type_metadata - t is invalid type")
    };

    let llvm_type = type_of::type_of(cx, t);
    let (size, align) = size_and_align_of(cx, llvm_type);
    let name = CString::new(name).unwrap();
    let ty_metadata = unsafe {
        llvm::LLVMRustDIBuilderCreateBasicType(
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
        llvm::LLVMRustDIBuilderCreatePointerType(
            DIB(cx),
            pointee_type_metadata,
            bytes_to_bits(pointer_size),
            bytes_to_bits(pointer_align),
            name.as_ptr())
    };
    return ptr_metadata;
}

pub fn compile_unit_metadata(scc: &SharedCrateContext,
                             debug_context: &CrateDebugContext,
                             sess: &Session)
                             -> DIDescriptor {
    let work_dir = &sess.working_dir;
    let compile_unit_name = match sess.local_crate_source_file {
        None => fallback_path(scc),
        Some(ref abs_path) => {
            if abs_path.is_relative() {
                sess.warn("debuginfo: Invalid path to crate's local root source file!");
                fallback_path(scc)
            } else {
                match abs_path.strip_prefix(work_dir) {
                    Ok(ref p) if p.is_relative() => {
                        if p.starts_with(Path::new("./")) {
                            path2cstr(p)
                        } else {
                            path2cstr(&Path::new(".").join(p))
                        }
                    }
                    _ => fallback_path(scc)
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
        llvm::LLVMRustDIBuilderCreateCompileUnit(
            debug_context.builder,
            DW_LANG_RUST,
            compile_unit_name,
            work_dir.as_ptr(),
            producer.as_ptr(),
            sess.opts.optimize != config::OptLevel::No,
            flags.as_ptr() as *const _,
            0,
            split_name.as_ptr() as *const _)
    };

    fn fallback_path(scc: &SharedCrateContext) -> CString {
        CString::new(scc.link_meta().crate_name.to_string()).unwrap()
    }
}

struct MetadataCreationResult {
    metadata: DIType,
    already_stored_in_typemap: bool
}

impl MetadataCreationResult {
    fn new(metadata: DIType, already_stored_in_typemap: bool) -> MetadataCreationResult {
        MetadataCreationResult {
            metadata: metadata,
            already_stored_in_typemap: already_stored_in_typemap
        }
    }
}

#[derive(Debug)]
enum MemberOffset {
    FixedMemberOffset { bytes: usize },
    // For ComputedMemberOffset, the offset is read from the llvm type definition.
    ComputedMemberOffset
}

// Description of a type member, which can either be a regular field (as in
// structs or tuples) or an enum variant.
#[derive(Debug)]
struct MemberDescription {
    name: String,
    llvm_type: Type,
    type_metadata: DIType,
    offset: MemberOffset,
    flags: DIFlags,
}

// A factory for MemberDescriptions. It produces a list of member descriptions
// for some record-like type. MemberDescriptionFactories are used to defer the
// creation of type member descriptions in order to break cycles arising from
// recursive type definitions.
enum MemberDescriptionFactory<'tcx> {
    StructMDF(StructMemberDescriptionFactory<'tcx>),
    TupleMDF(TupleMemberDescriptionFactory<'tcx>),
    EnumMDF(EnumMemberDescriptionFactory<'tcx>),
    UnionMDF(UnionMemberDescriptionFactory<'tcx>),
    VariantMDF(VariantMemberDescriptionFactory<'tcx>)
}

impl<'tcx> MemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CrateContext<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        match *self {
            StructMDF(ref this) => {
                this.create_member_descriptions(cx)
            }
            TupleMDF(ref this) => {
                this.create_member_descriptions(cx)
            }
            EnumMDF(ref this) => {
                this.create_member_descriptions(cx)
            }
            UnionMDF(ref this) => {
                this.create_member_descriptions(cx)
            }
            VariantMDF(ref this) => {
                this.create_member_descriptions(cx)
            }
        }
    }
}

//=-----------------------------------------------------------------------------
// Structs
//=-----------------------------------------------------------------------------

// Creates MemberDescriptions for the fields of a struct
struct StructMemberDescriptionFactory<'tcx> {
    ty: Ty<'tcx>,
    variant: &'tcx ty::VariantDef,
    substs: &'tcx Substs<'tcx>,
    span: Span,
}

impl<'tcx> StructMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CrateContext<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        let layout = cx.layout_of(self.ty);

        let tmp;
        let offsets = match *layout {
            layout::Univariant { ref variant, .. } => &variant.offsets,
            layout::Vector { element, count } => {
                let element_size = element.size(&cx.tcx().data_layout).bytes();
                tmp = (0..count).
                  map(|i| layout::Size::from_bytes(i*element_size))
                  .collect::<Vec<layout::Size>>();
                &tmp
            }
            _ => bug!("{} is not a struct", self.ty)
        };

        self.variant.fields.iter().enumerate().map(|(i, f)| {
            let name = if self.variant.ctor_kind == CtorKind::Fn {
                format!("__{}", i)
            } else {
                f.name.to_string()
            };
            let fty = monomorphize::field_ty(cx.tcx(), self.substs, f);

            let offset = FixedMemberOffset { bytes: offsets[i].bytes() as usize};

            MemberDescription {
                name: name,
                llvm_type: type_of::in_memory_type_of(cx, fty),
                type_metadata: type_metadata(cx, fty, self.span),
                offset: offset,
                flags: DIFlags::FlagZero,
            }
        }).collect()
    }
}


fn prepare_struct_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                     struct_type: Ty<'tcx>,
                                     unique_type_id: UniqueTypeId,
                                     span: Span)
                                     -> RecursiveTypeDescription<'tcx> {
    let struct_name = compute_debuginfo_type_name(cx, struct_type, false);
    let struct_llvm_type = type_of::in_memory_type_of(cx, struct_type);

    let (struct_def_id, variant, substs) = match struct_type.sty {
        ty::TyAdt(def, substs) => (def.did, def.struct_variant(), substs),
        _ => bug!("prepare_struct_metadata on a non-ADT")
    };

    let (containing_scope, _) = get_namespace_and_span_for_item(cx, struct_def_id);

    let struct_metadata_stub = create_struct_stub(cx,
                                                  struct_llvm_type,
                                                  &struct_name,
                                                  unique_type_id,
                                                  containing_scope);

    create_and_register_recursive_type_forward_declaration(
        cx,
        struct_type,
        unique_type_id,
        struct_metadata_stub,
        struct_llvm_type,
        StructMDF(StructMemberDescriptionFactory {
            ty: struct_type,
            variant: variant,
            substs: substs,
            span: span,
        })
    )
}

//=-----------------------------------------------------------------------------
// Tuples
//=-----------------------------------------------------------------------------

// Creates MemberDescriptions for the fields of a tuple
struct TupleMemberDescriptionFactory<'tcx> {
    ty: Ty<'tcx>,
    component_types: Vec<Ty<'tcx>>,
    span: Span,
}

impl<'tcx> TupleMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CrateContext<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        let layout = cx.layout_of(self.ty);
        let offsets = if let layout::Univariant { ref variant, .. } = *layout {
            &variant.offsets
        } else {
            bug!("{} is not a tuple", self.ty);
        };

        self.component_types
            .iter()
            .enumerate()
            .map(|(i, &component_type)| {
            MemberDescription {
                name: format!("__{}", i),
                llvm_type: type_of::type_of(cx, component_type),
                type_metadata: type_metadata(cx, component_type, self.span),
                offset: FixedMemberOffset { bytes: offsets[i].bytes() as usize },
                flags: DIFlags::FlagZero,
            }
        }).collect()
    }
}

fn prepare_tuple_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                    tuple_type: Ty<'tcx>,
                                    component_types: &[Ty<'tcx>],
                                    unique_type_id: UniqueTypeId,
                                    span: Span)
                                    -> RecursiveTypeDescription<'tcx> {
    let tuple_name = compute_debuginfo_type_name(cx, tuple_type, false);
    let tuple_llvm_type = type_of::type_of(cx, tuple_type);

    create_and_register_recursive_type_forward_declaration(
        cx,
        tuple_type,
        unique_type_id,
        create_struct_stub(cx,
                           tuple_llvm_type,
                           &tuple_name[..],
                           unique_type_id,
                           NO_SCOPE_METADATA),
        tuple_llvm_type,
        TupleMDF(TupleMemberDescriptionFactory {
            ty: tuple_type,
            component_types: component_types.to_vec(),
            span: span,
        })
    )
}

//=-----------------------------------------------------------------------------
// Unions
//=-----------------------------------------------------------------------------

struct UnionMemberDescriptionFactory<'tcx> {
    variant: &'tcx ty::VariantDef,
    substs: &'tcx Substs<'tcx>,
    span: Span,
}

impl<'tcx> UnionMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CrateContext<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        self.variant.fields.iter().map(|field| {
            let fty = monomorphize::field_ty(cx.tcx(), self.substs, field);
            MemberDescription {
                name: field.name.to_string(),
                llvm_type: type_of::type_of(cx, fty),
                type_metadata: type_metadata(cx, fty, self.span),
                offset: FixedMemberOffset { bytes: 0 },
                flags: DIFlags::FlagZero,
            }
        }).collect()
    }
}

fn prepare_union_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                    union_type: Ty<'tcx>,
                                    unique_type_id: UniqueTypeId,
                                    span: Span)
                                    -> RecursiveTypeDescription<'tcx> {
    let union_name = compute_debuginfo_type_name(cx, union_type, false);
    let union_llvm_type = type_of::in_memory_type_of(cx, union_type);

    let (union_def_id, variant, substs) = match union_type.sty {
        ty::TyAdt(def, substs) => (def.did, def.struct_variant(), substs),
        _ => bug!("prepare_union_metadata on a non-ADT")
    };

    let (containing_scope, _) = get_namespace_and_span_for_item(cx, union_def_id);

    let union_metadata_stub = create_union_stub(cx,
                                                union_llvm_type,
                                                &union_name,
                                                unique_type_id,
                                                containing_scope);

    create_and_register_recursive_type_forward_declaration(
        cx,
        union_type,
        unique_type_id,
        union_metadata_stub,
        union_llvm_type,
        UnionMDF(UnionMemberDescriptionFactory {
            variant: variant,
            substs: substs,
            span: span,
        })
    )
}

//=-----------------------------------------------------------------------------
// Enums
//=-----------------------------------------------------------------------------

// Describes the members of an enum value: An enum is described as a union of
// structs in DWARF. This MemberDescriptionFactory provides the description for
// the members of this union; so for every variant of the given enum, this
// factory will produce one MemberDescription (all with no name and a fixed
// offset of zero bytes).
struct EnumMemberDescriptionFactory<'tcx> {
    enum_type: Ty<'tcx>,
    type_rep: &'tcx layout::Layout,
    discriminant_type_metadata: Option<DIType>,
    containing_scope: DIScope,
    file_metadata: DIFile,
    span: Span,
}

impl<'tcx> EnumMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CrateContext<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        let adt = &self.enum_type.ty_adt_def().unwrap();
        let substs = match self.enum_type.sty {
            ty::TyAdt(def, ref s) if def.adt_kind() == AdtKind::Enum => s,
            _ => bug!("{} is not an enum", self.enum_type)
        };
        match *self.type_rep {
            layout::General { ref variants, .. } => {
                let discriminant_info = RegularDiscriminant(self.discriminant_type_metadata
                    .expect(""));
                variants
                    .iter()
                    .enumerate()
                    .map(|(i, struct_def)| {
                        let (variant_type_metadata,
                             variant_llvm_type,
                             member_desc_factory) =
                            describe_enum_variant(cx,
                                                  self.enum_type,
                                                  struct_def,
                                                  &adt.variants[i],
                                                  discriminant_info,
                                                  self.containing_scope,
                                                  self.span);

                        let member_descriptions = member_desc_factory
                            .create_member_descriptions(cx);

                        set_members_of_composite_type(cx,
                                                      variant_type_metadata,
                                                      variant_llvm_type,
                                                      &member_descriptions);
                        MemberDescription {
                            name: "".to_string(),
                            llvm_type: variant_llvm_type,
                            type_metadata: variant_type_metadata,
                            offset: FixedMemberOffset { bytes: 0 },
                            flags: DIFlags::FlagZero
                        }
                    }).collect()
            },
            layout::Univariant{ ref variant, .. } => {
                assert!(adt.variants.len() <= 1);

                if adt.variants.is_empty() {
                    vec![]
                } else {
                    let (variant_type_metadata,
                         variant_llvm_type,
                         member_description_factory) =
                        describe_enum_variant(cx,
                                              self.enum_type,
                                              variant,
                                              &adt.variants[0],
                                              NoDiscriminant,
                                              self.containing_scope,
                                              self.span);

                    let member_descriptions =
                        member_description_factory.create_member_descriptions(cx);

                    set_members_of_composite_type(cx,
                                                  variant_type_metadata,
                                                  variant_llvm_type,
                                                  &member_descriptions[..]);
                    vec![
                        MemberDescription {
                            name: "".to_string(),
                            llvm_type: variant_llvm_type,
                            type_metadata: variant_type_metadata,
                            offset: FixedMemberOffset { bytes: 0 },
                            flags: DIFlags::FlagZero
                        }
                    ]
                }
            }
            layout::RawNullablePointer { nndiscr: non_null_variant_index, .. } => {
                // As far as debuginfo is concerned, the pointer this enum
                // represents is still wrapped in a struct. This is to make the
                // DWARF representation of enums uniform.

                // First create a description of the artificial wrapper struct:
                let non_null_variant = &adt.variants[non_null_variant_index as usize];
                let non_null_variant_name = non_null_variant.name.as_str();

                // The llvm type and metadata of the pointer
                let nnty = monomorphize::field_ty(cx.tcx(), &substs, &non_null_variant.fields[0] );
                let non_null_llvm_type = type_of::type_of(cx, nnty);
                let non_null_type_metadata = type_metadata(cx, nnty, self.span);

                // The type of the artificial struct wrapping the pointer
                let artificial_struct_llvm_type = Type::struct_(cx,
                                                                &[non_null_llvm_type],
                                                                false);

                // For the metadata of the wrapper struct, we need to create a
                // MemberDescription of the struct's single field.
                let sole_struct_member_description = MemberDescription {
                    name: match non_null_variant.ctor_kind {
                        CtorKind::Fn => "__0".to_string(),
                        CtorKind::Fictive => {
                            non_null_variant.fields[0].name.to_string()
                        }
                        CtorKind::Const => bug!()
                    },
                    llvm_type: non_null_llvm_type,
                    type_metadata: non_null_type_metadata,
                    offset: FixedMemberOffset { bytes: 0 },
                    flags: DIFlags::FlagZero
                };

                let unique_type_id = debug_context(cx).type_map
                                                      .borrow_mut()
                                                      .get_unique_type_id_of_enum_variant(
                                                          cx,
                                                          self.enum_type,
                                                          &non_null_variant_name);

                // Now we can create the metadata of the artificial struct
                let artificial_struct_metadata =
                    composite_type_metadata(cx,
                                            artificial_struct_llvm_type,
                                            &non_null_variant_name,
                                            unique_type_id,
                                            &[sole_struct_member_description],
                                            self.containing_scope,
                                            self.file_metadata,
                                            syntax_pos::DUMMY_SP);

                // Encode the information about the null variant in the union
                // member's name.
                let null_variant_index = (1 - non_null_variant_index) as usize;
                let null_variant_name = adt.variants[null_variant_index].name;
                let union_member_name = format!("RUST$ENCODED$ENUM${}${}",
                                                0,
                                                null_variant_name);

                // Finally create the (singleton) list of descriptions of union
                // members.
                vec![
                    MemberDescription {
                        name: union_member_name,
                        llvm_type: artificial_struct_llvm_type,
                        type_metadata: artificial_struct_metadata,
                        offset: FixedMemberOffset { bytes: 0 },
                        flags: DIFlags::FlagZero
                    }
                ]
            },
            layout::StructWrappedNullablePointer { nonnull: ref struct_def,
                                                nndiscr,
                                                ref discrfield_source, ..} => {
                // Create a description of the non-null variant
                let (variant_type_metadata, variant_llvm_type, member_description_factory) =
                    describe_enum_variant(cx,
                                          self.enum_type,
                                          struct_def,
                                          &adt.variants[nndiscr as usize],
                                          OptimizedDiscriminant,
                                          self.containing_scope,
                                          self.span);

                let variant_member_descriptions =
                    member_description_factory.create_member_descriptions(cx);

                set_members_of_composite_type(cx,
                                              variant_type_metadata,
                                              variant_llvm_type,
                                              &variant_member_descriptions[..]);

                // Encode the information about the null variant in the union
                // member's name.
                let null_variant_index = (1 - nndiscr) as usize;
                let null_variant_name = adt.variants[null_variant_index].name;
                let discrfield_source = discrfield_source.iter()
                                           .skip(1)
                                           .map(|x| x.to_string())
                                           .collect::<Vec<_>>().join("$");
                let union_member_name = format!("RUST$ENCODED$ENUM${}${}",
                                                discrfield_source,
                                                null_variant_name);

                // Create the (singleton) list of descriptions of union members.
                vec![
                    MemberDescription {
                        name: union_member_name,
                        llvm_type: variant_llvm_type,
                        type_metadata: variant_type_metadata,
                        offset: FixedMemberOffset { bytes: 0 },
                        flags: DIFlags::FlagZero
                    }
                ]
            },
            layout::CEnum { .. } => span_bug!(self.span, "This should be unreachable."),
            ref l @ _ => bug!("Not an enum layout: {:#?}", l)
        }
    }
}

// Creates MemberDescriptions for the fields of a single enum variant.
struct VariantMemberDescriptionFactory<'tcx> {
    // Cloned from the layout::Struct describing the variant.
    offsets: &'tcx [layout::Size],
    args: Vec<(String, Ty<'tcx>)>,
    discriminant_type_metadata: Option<DIType>,
    span: Span,
}

impl<'tcx> VariantMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CrateContext<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        self.args.iter().enumerate().map(|(i, &(ref name, ty))| {
            MemberDescription {
                name: name.to_string(),
                llvm_type: type_of::type_of(cx, ty),
                type_metadata: match self.discriminant_type_metadata {
                    Some(metadata) if i == 0 => metadata,
                    _ => type_metadata(cx, ty, self.span)
                },
                offset: FixedMemberOffset { bytes: self.offsets[i].bytes() as usize },
                flags: DIFlags::FlagZero
            }
        }).collect()
    }
}

#[derive(Copy, Clone)]
enum EnumDiscriminantInfo {
    RegularDiscriminant(DIType),
    OptimizedDiscriminant,
    NoDiscriminant
}

// Returns a tuple of (1) type_metadata_stub of the variant, (2) the llvm_type
// of the variant, and (3) a MemberDescriptionFactory for producing the
// descriptions of the fields of the variant. This is a rudimentary version of a
// full RecursiveTypeDescription.
fn describe_enum_variant<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                   enum_type: Ty<'tcx>,
                                   struct_def: &'tcx layout::Struct,
                                   variant: &'tcx ty::VariantDef,
                                   discriminant_info: EnumDiscriminantInfo,
                                   containing_scope: DIScope,
                                   span: Span)
                                   -> (DICompositeType, Type, MemberDescriptionFactory<'tcx>) {
    let substs = match enum_type.sty {
        ty::TyAdt(def, s) if def.adt_kind() == AdtKind::Enum => s,
        ref t @ _ => bug!("{:#?} is not an enum", t)
    };

    let maybe_discr_and_signed: Option<(layout::Integer, bool)> = match *cx.layout_of(enum_type) {
        layout::CEnum {discr, ..} => Some((discr, true)),
        layout::General{discr, ..} => Some((discr, false)),
        layout::Univariant { .. }
        | layout::RawNullablePointer { .. }
        | layout::StructWrappedNullablePointer { .. } => None,
        ref l @ _ => bug!("This should be unreachable. Type is {:#?} layout is {:#?}", enum_type, l)
    };

    let mut field_tys = variant.fields.iter().map(|f| {
        monomorphize::field_ty(cx.tcx(), &substs, f)
    }).collect::<Vec<_>>();

    if let Some((discr, signed)) = maybe_discr_and_signed {
        field_tys.insert(0, discr.to_ty(&cx.tcx(), signed));
    }


    let variant_llvm_type =
        Type::struct_(cx, &field_tys
                                    .iter()
                                    .map(|t| type_of::type_of(cx, t))
                                    .collect::<Vec<_>>()
                                    ,
                      struct_def.packed);
    // Could do some consistency checks here: size, align, field count, discr type

    let variant_name = variant.name.as_str();
    let unique_type_id = debug_context(cx).type_map
                                          .borrow_mut()
                                          .get_unique_type_id_of_enum_variant(
                                              cx,
                                              enum_type,
                                              &variant_name);

    let metadata_stub = create_struct_stub(cx,
                                           variant_llvm_type,
                                           &variant_name,
                                           unique_type_id,
                                           containing_scope);

    // Get the argument names from the enum variant info
    let mut arg_names: Vec<_> = match variant.ctor_kind {
        CtorKind::Const => vec![],
        CtorKind::Fn => {
            variant.fields
                   .iter()
                   .enumerate()
                   .map(|(i, _)| format!("__{}", i))
                   .collect()
        }
        CtorKind::Fictive => {
            variant.fields
                   .iter()
                   .map(|f| f.name.to_string())
                   .collect()
        }
    };

    // If this is not a univariant enum, there is also the discriminant field.
    match discriminant_info {
        RegularDiscriminant(_) => arg_names.insert(0, "RUST$ENUM$DISR".to_string()),
        _ => { /* do nothing */ }
    };

    // Build an array of (field name, field type) pairs to be captured in the factory closure.
    let args: Vec<(String, Ty)> = arg_names.iter()
        .zip(field_tys.iter())
        .map(|(s, &t)| (s.to_string(), t))
        .collect();

    let member_description_factory =
        VariantMDF(VariantMemberDescriptionFactory {
            offsets: &struct_def.offsets[..],
            args: args,
            discriminant_type_metadata: match discriminant_info {
                RegularDiscriminant(discriminant_type_metadata) => {
                    Some(discriminant_type_metadata)
                }
                _ => None
            },
            span: span,
        });

    (metadata_stub, variant_llvm_type, member_description_factory)
}

fn prepare_enum_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                   enum_type: Ty<'tcx>,
                                   enum_def_id: DefId,
                                   unique_type_id: UniqueTypeId,
                                   span: Span)
                                   -> RecursiveTypeDescription<'tcx> {
    let enum_name = compute_debuginfo_type_name(cx, enum_type, false);

    let (containing_scope, _) = get_namespace_and_span_for_item(cx, enum_def_id);
    // FIXME: This should emit actual file metadata for the enum, but we
    // currently can't get the necessary information when it comes to types
    // imported from other crates. Formerly we violated the ODR when performing
    // LTO because we emitted debuginfo for the same type with varying file
    // metadata, so as a workaround we pretend that the type comes from
    // <unknown>
    let file_metadata = unknown_file_metadata(cx);

    let variants = &enum_type.ty_adt_def().unwrap().variants;
    let enumerators_metadata: Vec<DIDescriptor> = variants
        .iter()
        .map(|v| {
            let token = v.name.as_str();
            let name = CString::new(token.as_bytes()).unwrap();
            unsafe {
                llvm::LLVMRustDIBuilderCreateEnumerator(
                    DIB(cx),
                    name.as_ptr(),
                    // FIXME: what if enumeration has i128 discriminant?
                    v.disr_val.to_u128_unchecked() as u64)
            }
        })
        .collect();

    let discriminant_type_metadata = |inttype: layout::Integer, signed: bool| {
        let disr_type_key = (enum_def_id, inttype);
        let cached_discriminant_type_metadata = debug_context(cx).created_enum_disr_types
                                                                 .borrow()
                                                                 .get(&disr_type_key).cloned();
        match cached_discriminant_type_metadata {
            Some(discriminant_type_metadata) => discriminant_type_metadata,
            None => {
                let discriminant_llvm_type = Type::from_integer(cx, inttype);
                let (discriminant_size, discriminant_align) =
                    size_and_align_of(cx, discriminant_llvm_type);
                let discriminant_base_type_metadata =
                    type_metadata(cx,
                                  inttype.to_ty(&cx.tcx(), signed),
                                  syntax_pos::DUMMY_SP);
                let discriminant_name = get_enum_discriminant_name(cx, enum_def_id);

                let name = CString::new(discriminant_name.as_bytes()).unwrap();
                let discriminant_type_metadata = unsafe {
                    llvm::LLVMRustDIBuilderCreateEnumerationType(
                        DIB(cx),
                        containing_scope,
                        name.as_ptr(),
                        file_metadata,
                        UNKNOWN_LINE_NUMBER,
                        bytes_to_bits(discriminant_size),
                        bytes_to_bits(discriminant_align),
                        create_DIArray(DIB(cx), &enumerators_metadata),
                        discriminant_base_type_metadata)
                };

                debug_context(cx).created_enum_disr_types
                                 .borrow_mut()
                                 .insert(disr_type_key, discriminant_type_metadata);

                discriminant_type_metadata
            }
        }
    };

    let type_rep = cx.layout_of(enum_type);

    let discriminant_type_metadata = match *type_rep {
        layout::CEnum { discr, signed, .. } => {
            return FinalMetadata(discriminant_type_metadata(discr, signed))
        },
        layout::RawNullablePointer { .. }           |
        layout::StructWrappedNullablePointer { .. } |
        layout::Univariant { .. }                      => None,
        layout::General { discr, .. } => Some(discriminant_type_metadata(discr, false)),
        ref l @ _ => bug!("Not an enum layout: {:#?}", l)
    };

    let enum_llvm_type = type_of::type_of(cx, enum_type);
    let (enum_type_size, enum_type_align) = size_and_align_of(cx, enum_llvm_type);

    let enum_name = CString::new(enum_name).unwrap();
    let unique_type_id_str = CString::new(
        debug_context(cx).type_map.borrow().get_unique_type_id_as_string(unique_type_id).as_bytes()
    ).unwrap();
    let enum_metadata = unsafe {
        llvm::LLVMRustDIBuilderCreateUnionType(
        DIB(cx),
        containing_scope,
        enum_name.as_ptr(),
        file_metadata,
        UNKNOWN_LINE_NUMBER,
        bytes_to_bits(enum_type_size),
        bytes_to_bits(enum_type_align),
        DIFlags::FlagZero,
        ptr::null_mut(),
        0, // RuntimeLang
        unique_type_id_str.as_ptr())
    };

    return create_and_register_recursive_type_forward_declaration(
        cx,
        enum_type,
        unique_type_id,
        enum_metadata,
        enum_llvm_type,
        EnumMDF(EnumMemberDescriptionFactory {
            enum_type: enum_type,
            type_rep: type_rep,
            discriminant_type_metadata: discriminant_type_metadata,
            containing_scope: containing_scope,
            file_metadata: file_metadata,
            span: span,
        }),
    );

    fn get_enum_discriminant_name(cx: &CrateContext,
                                  def_id: DefId)
                                  -> InternedString {
        cx.tcx().item_name(def_id).as_str()
    }
}

/// Creates debug information for a composite type, that is, anything that
/// results in a LLVM struct.
///
/// Examples of Rust types to use this are: structs, tuples, boxes, vecs, and enums.
fn composite_type_metadata(cx: &CrateContext,
                           composite_llvm_type: Type,
                           composite_type_name: &str,
                           composite_type_unique_id: UniqueTypeId,
                           member_descriptions: &[MemberDescription],
                           containing_scope: DIScope,

                           // Ignore source location information as long as it
                           // can't be reconstructed for non-local crates.
                           _file_metadata: DIFile,
                           _definition_span: Span)
                           -> DICompositeType {
    // Create the (empty) struct metadata node ...
    let composite_type_metadata = create_struct_stub(cx,
                                                     composite_llvm_type,
                                                     composite_type_name,
                                                     composite_type_unique_id,
                                                     containing_scope);
    // ... and immediately create and add the member descriptions.
    set_members_of_composite_type(cx,
                                  composite_type_metadata,
                                  composite_llvm_type,
                                  member_descriptions);

    return composite_type_metadata;
}

fn set_members_of_composite_type(cx: &CrateContext,
                                 composite_type_metadata: DICompositeType,
                                 composite_llvm_type: Type,
                                 member_descriptions: &[MemberDescription]) {
    // In some rare cases LLVM metadata uniquing would lead to an existing type
    // description being used instead of a new one created in
    // create_struct_stub. This would cause a hard to trace assertion in
    // DICompositeType::SetTypeArray(). The following check makes sure that we
    // get a better error message if this should happen again due to some
    // regression.
    {
        let mut composite_types_completed =
            debug_context(cx).composite_types_completed.borrow_mut();
        if composite_types_completed.contains(&composite_type_metadata) {
            bug!("debuginfo::set_members_of_composite_type() - \
                  Already completed forward declaration re-encountered.");
        } else {
            composite_types_completed.insert(composite_type_metadata);
        }
    }

    let member_metadata: Vec<DIDescriptor> = member_descriptions
        .iter()
        .enumerate()
        .map(|(i, member_description)| {
            let (member_size, member_align) = size_and_align_of(cx, member_description.llvm_type);
            let member_offset = match member_description.offset {
                FixedMemberOffset { bytes } => bytes as u64,
                ComputedMemberOffset => machine::llelement_offset(cx, composite_llvm_type, i)
            };

            let member_name = member_description.name.as_bytes();
            let member_name = CString::new(member_name).unwrap();
            unsafe {
                llvm::LLVMRustDIBuilderCreateMemberType(
                    DIB(cx),
                    composite_type_metadata,
                    member_name.as_ptr(),
                    unknown_file_metadata(cx),
                    UNKNOWN_LINE_NUMBER,
                    bytes_to_bits(member_size),
                    bytes_to_bits(member_align),
                    bytes_to_bits(member_offset),
                    member_description.flags,
                    member_description.type_metadata)
            }
        })
        .collect();

    unsafe {
        let type_array = create_DIArray(DIB(cx), &member_metadata[..]);
        llvm::LLVMRustDICompositeTypeSetTypeArray(
            DIB(cx), composite_type_metadata, type_array);
    }
}

// A convenience wrapper around LLVMRustDIBuilderCreateStructType(). Does not do
// any caching, does not add any fields to the struct. This can be done later
// with set_members_of_composite_type().
fn create_struct_stub(cx: &CrateContext,
                      struct_llvm_type: Type,
                      struct_type_name: &str,
                      unique_type_id: UniqueTypeId,
                      containing_scope: DIScope)
                   -> DICompositeType {
    let (struct_size, struct_align) = size_and_align_of(cx, struct_llvm_type);

    let name = CString::new(struct_type_name).unwrap();
    let unique_type_id = CString::new(
        debug_context(cx).type_map.borrow().get_unique_type_id_as_string(unique_type_id).as_bytes()
    ).unwrap();
    let metadata_stub = unsafe {
        // LLVMRustDIBuilderCreateStructType() wants an empty array. A null
        // pointer will lead to hard to trace and debug LLVM assertions
        // later on in llvm/lib/IR/Value.cpp.
        let empty_array = create_DIArray(DIB(cx), &[]);

        llvm::LLVMRustDIBuilderCreateStructType(
            DIB(cx),
            containing_scope,
            name.as_ptr(),
            unknown_file_metadata(cx),
            UNKNOWN_LINE_NUMBER,
            bytes_to_bits(struct_size),
            bytes_to_bits(struct_align),
            DIFlags::FlagZero,
            ptr::null_mut(),
            empty_array,
            0,
            ptr::null_mut(),
            unique_type_id.as_ptr())
    };

    return metadata_stub;
}

fn create_union_stub(cx: &CrateContext,
                     union_llvm_type: Type,
                     union_type_name: &str,
                     unique_type_id: UniqueTypeId,
                     containing_scope: DIScope)
                   -> DICompositeType {
    let (union_size, union_align) = size_and_align_of(cx, union_llvm_type);

    let name = CString::new(union_type_name).unwrap();
    let unique_type_id = CString::new(
        debug_context(cx).type_map.borrow().get_unique_type_id_as_string(unique_type_id).as_bytes()
    ).unwrap();
    let metadata_stub = unsafe {
        // LLVMRustDIBuilderCreateUnionType() wants an empty array. A null
        // pointer will lead to hard to trace and debug LLVM assertions
        // later on in llvm/lib/IR/Value.cpp.
        let empty_array = create_DIArray(DIB(cx), &[]);

        llvm::LLVMRustDIBuilderCreateUnionType(
            DIB(cx),
            containing_scope,
            name.as_ptr(),
            unknown_file_metadata(cx),
            UNKNOWN_LINE_NUMBER,
            bytes_to_bits(union_size),
            bytes_to_bits(union_align),
            DIFlags::FlagZero,
            empty_array,
            0, // RuntimeLang
            unique_type_id.as_ptr())
    };

    return metadata_stub;
}

/// Creates debug information for the given global variable.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_global_var_metadata(cx: &CrateContext,
                                  node_id: ast::NodeId,
                                  global: ValueRef) {
    if cx.dbg_cx().is_none() {
        return;
    }

    let tcx = cx.tcx();

    let node_def_id = tcx.map.local_def_id(node_id);
    let (var_scope, span) = get_namespace_and_span_for_item(cx, node_def_id);

    let (file_metadata, line_number) = if span != syntax_pos::DUMMY_SP {
        let loc = span_start(cx, span);
        (file_metadata(cx, &loc.file.name, &loc.file.abs_path), loc.line as c_uint)
    } else {
        (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER)
    };

    let is_local_to_unit = is_node_local_to_unit(cx, node_id);
    let variable_type = tcx.erase_regions(&tcx.item_type(node_def_id));
    let type_metadata = type_metadata(cx, variable_type, span);
    let var_name = tcx.item_name(node_def_id).to_string();
    let linkage_name = mangled_name_of_item(cx, node_def_id, "");

    let var_name = CString::new(var_name).unwrap();
    let linkage_name = CString::new(linkage_name).unwrap();

    let ty = cx.tcx().item_type(node_def_id);
    let global_align = type_of::align_of(cx, ty);

    unsafe {
        llvm::LLVMRustDIBuilderCreateStaticVariable(DIB(cx),
                                                    var_scope,
                                                    var_name.as_ptr(),
                                                    linkage_name.as_ptr(),
                                                    file_metadata,
                                                    line_number,
                                                    type_metadata,
                                                    is_local_to_unit,
                                                    global,
                                                    ptr::null_mut(),
                                                    global_align as u64,
        );
    }
}

// Creates an "extension" of an existing DIScope into another file.
pub fn extend_scope_to_file(ccx: &CrateContext,
                            scope_metadata: DIScope,
                            file: &syntax_pos::FileMap)
                            -> DILexicalBlock {
    let file_metadata = file_metadata(ccx, &file.name, &file.abs_path);
    unsafe {
        llvm::LLVMRustDIBuilderCreateLexicalBlockFile(
            DIB(ccx),
            scope_metadata,
            file_metadata)
    }
}
