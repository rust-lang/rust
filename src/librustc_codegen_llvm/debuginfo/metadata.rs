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
use self::MemberDescriptionFactory::*;
use self::EnumDiscriminantInfo::*;

use super::utils::{debug_context, DIB, span_start,
                   get_namespace_for_item, create_DIArray, is_node_local_to_unit};
use super::namespace::mangled_name_of_instance;
use super::type_names::compute_debuginfo_type_name;
use super::{CrateDebugContext};
use abi;

use llvm::{self, ValueRef};
use llvm::debuginfo::{DIType, DIFile, DIScope, DIDescriptor,
                      DICompositeType, DILexicalBlock, DIFlags};

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc::hir::CodegenFnAttrFlags;
use rustc::hir::def::CtorKind;
use rustc::hir::def_id::{DefId, CrateNum, LOCAL_CRATE};
use rustc::ich::{Fingerprint, NodeIdHashingMode};
use rustc::ty::Instance;
use common::CodegenCx;
use rustc::ty::{self, AdtKind, ParamEnv, Ty, TyCtxt};
use rustc::ty::layout::{self, Align, LayoutOf, PrimitiveExt, Size, TyLayout};
use rustc::session::config;
use rustc::util::nodemap::FxHashMap;
use rustc::util::common::path2cstr;

use libc::{c_uint, c_longlong};
use std::ffi::CString;
use std::fmt::Write;
use std::ptr;
use std::path::{Path, PathBuf};
use syntax::ast;
use syntax::symbol::{Interner, InternedString, Symbol};
use syntax_pos::{self, Span, FileName};


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
    fn get_unique_type_id_of_type<'a>(&mut self, cx: &CodegenCx<'a, 'tcx>,
                                      type_: Ty<'tcx>) -> UniqueTypeId {
        // Let's see if we already have something in the cache
        match self.type_to_unique_id.get(&type_).cloned() {
            Some(unique_type_id) => return unique_type_id,
            None => { /* generate one */}
        };

        // The hasher we are using to generate the UniqueTypeId. We want
        // something that provides more than the 64 bits of the DefaultHasher.
        let mut hasher = StableHasher::<Fingerprint>::new();
        let mut hcx = cx.tcx.create_stable_hashing_context();
        let type_ = cx.tcx.erase_regions(&type_);
        hcx.while_hashing_spans(false, |hcx| {
            hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
                type_.hash_stable(hcx, &mut hasher);
            });
        });
        let unique_type_id = hasher.finish().to_hex();

        let key = self.unique_id_interner.intern(&unique_type_id);
        self.type_to_unique_id.insert(type_, UniqueTypeId(key));

        return UniqueTypeId(key);
    }

    // Get the UniqueTypeId for an enum variant. Enum variants are not really
    // types of their own, so they need special handling. We still need a
    // UniqueTypeId for them, since to debuginfo they *are* real types.
    fn get_unique_type_id_of_enum_variant<'a>(&mut self,
                                              cx: &CodegenCx<'a, 'tcx>,
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
        member_description_factory: MemberDescriptionFactory<'tcx>,
    },
    FinalMetadata(DICompositeType)
}

fn create_and_register_recursive_type_forward_declaration<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    unfinished_type: Ty<'tcx>,
    unique_type_id: UniqueTypeId,
    metadata_stub: DICompositeType,
    member_description_factory: MemberDescriptionFactory<'tcx>)
 -> RecursiveTypeDescription<'tcx> {

    // Insert the stub into the TypeMap in order to allow for recursive references
    let mut type_map = debug_context(cx).type_map.borrow_mut();
    type_map.register_unique_id_with_metadata(unique_type_id, metadata_stub);
    type_map.register_type_with_metadata(unfinished_type, metadata_stub);

    UnfinishedMetadata {
        unfinished_type,
        unique_type_id,
        metadata_stub,
        member_description_factory,
    }
}

impl<'tcx> RecursiveTypeDescription<'tcx> {
    // Finishes up the description of the type in question (mostly by providing
    // descriptions of the fields of the given type) and returns the final type
    // metadata.
    fn finalize<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> MetadataCreationResult {
        match *self {
            FinalMetadata(metadata) => MetadataCreationResult::new(metadata, false),
            UnfinishedMetadata {
                unfinished_type,
                unique_type_id,
                metadata_stub,
                ref member_description_factory,
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

fn fixed_vec_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                unique_type_id: UniqueTypeId,
                                array_or_slice_type: Ty<'tcx>,
                                element_type: Ty<'tcx>,
                                span: Span)
                                -> MetadataCreationResult {
    let element_type_metadata = type_metadata(cx, element_type, span);

    return_if_metadata_created_in_meantime!(cx, unique_type_id);

    let (size, align) = cx.size_and_align_of(array_or_slice_type);

    let upper_bound = match array_or_slice_type.sty {
        ty::TyArray(_, len) => {
            len.unwrap_usize(cx.tcx) as c_longlong
        }
        _ => -1
    };

    let subrange = unsafe {
        llvm::LLVMRustDIBuilderGetOrCreateSubrange(DIB(cx), 0, upper_bound)
    };

    let subscripts = create_DIArray(DIB(cx), &[subrange]);
    let metadata = unsafe {
        llvm::LLVMRustDIBuilderCreateArrayType(
            DIB(cx),
            size.bits(),
            align.abi_bits() as u32,
            element_type_metadata,
            subscripts)
    };

    return MetadataCreationResult::new(metadata, false);
}

fn vec_slice_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                slice_ptr_type: Ty<'tcx>,
                                element_type: Ty<'tcx>,
                                unique_type_id: UniqueTypeId,
                                span: Span)
                                -> MetadataCreationResult {
    let data_ptr_type = cx.tcx.mk_imm_ptr(element_type);

    let data_ptr_metadata = type_metadata(cx, data_ptr_type, span);

    return_if_metadata_created_in_meantime!(cx, unique_type_id);

    let slice_type_name = compute_debuginfo_type_name(cx, slice_ptr_type, true);

    let (pointer_size, pointer_align) = cx.size_and_align_of(data_ptr_type);
    let (usize_size, usize_align) = cx.size_and_align_of(cx.tcx.types.usize);

    let member_descriptions = [
        MemberDescription {
            name: "data_ptr".to_string(),
            type_metadata: data_ptr_metadata,
            offset: Size::ZERO,
            size: pointer_size,
            align: pointer_align,
            flags: DIFlags::FlagZero,
        },
        MemberDescription {
            name: "length".to_string(),
            type_metadata: type_metadata(cx, cx.tcx.types.usize, span),
            offset: pointer_size,
            size: usize_size,
            align: usize_align,
            flags: DIFlags::FlagZero,
        },
    ];

    let file_metadata = unknown_file_metadata(cx);

    let metadata = composite_type_metadata(cx,
                                           slice_ptr_type,
                                           &slice_type_name[..],
                                           unique_type_id,
                                           &member_descriptions,
                                           NO_SCOPE_METADATA,
                                           file_metadata,
                                           span);
    MetadataCreationResult::new(metadata, false)
}

fn subroutine_type_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                      unique_type_id: UniqueTypeId,
                                      signature: ty::PolyFnSig<'tcx>,
                                      span: Span)
                                      -> MetadataCreationResult
{
    let signature = cx.tcx.normalize_erasing_late_bound_regions(
        ty::ParamEnv::reveal_all(),
        &signature,
    );

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
fn trait_pointer_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
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
            get_namespace_for_item(cx, def_id)
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

    let file_metadata = unknown_file_metadata(cx);

    let layout = cx.layout_of(cx.tcx.mk_mut_ptr(trait_type));

    assert_eq!(abi::FAT_PTR_ADDR, 0);
    assert_eq!(abi::FAT_PTR_EXTRA, 1);

    let data_ptr_field = layout.field(cx, 0);
    let vtable_field = layout.field(cx, 1);
    let member_descriptions = [
        MemberDescription {
            name: "pointer".to_string(),
            type_metadata: type_metadata(cx,
                cx.tcx.mk_mut_ptr(cx.tcx.types.u8),
                syntax_pos::DUMMY_SP),
            offset: layout.fields.offset(0),
            size: data_ptr_field.size,
            align: data_ptr_field.align,
            flags: DIFlags::FlagArtificial,
        },
        MemberDescription {
            name: "vtable".to_string(),
            type_metadata: type_metadata(cx, vtable_field.ty, syntax_pos::DUMMY_SP),
            offset: layout.fields.offset(1),
            size: vtable_field.size,
            align: vtable_field.align,
            flags: DIFlags::FlagArtificial,
        },
    ];

    composite_type_metadata(cx,
                            trait_object_type,
                            &trait_type_name[..],
                            unique_type_id,
                            &member_descriptions,
                            containing_scope,
                            file_metadata,
                            syntax_pos::DUMMY_SP)
}

pub fn type_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
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

    let ptr_metadata = |ty: Ty<'tcx>| {
        match ty.sty {
            ty::TySlice(typ) => {
                Ok(vec_slice_metadata(cx, t, typ, unique_type_id, usage_site_span))
            }
            ty::TyStr => {
                Ok(vec_slice_metadata(cx, t, cx.tcx.types.u8, unique_type_id, usage_site_span))
            }
            ty::TyDynamic(..) => {
                Ok(MetadataCreationResult::new(
                    trait_pointer_metadata(cx, ty, Some(t), unique_type_id),
                    false))
            }
            _ => {
                let pointee_metadata = type_metadata(cx, ty, usage_site_span);

                match debug_context(cx).type_map
                                        .borrow()
                                        .find_metadata_for_unique_id(unique_type_id) {
                    Some(metadata) => return Err(metadata),
                    None => { /* proceed normally */ }
                };

                Ok(MetadataCreationResult::new(pointer_type_metadata(cx, t, pointee_metadata),
                   false))
            }
        }
    };

    let MetadataCreationResult { metadata, already_stored_in_typemap } = match t.sty {
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
        ty::TyArray(typ, _) |
        ty::TySlice(typ) => {
            fixed_vec_metadata(cx, unique_type_id, t, typ, usage_site_span)
        }
        ty::TyStr => {
            fixed_vec_metadata(cx, unique_type_id, t, cx.tcx.types.i8, usage_site_span)
        }
        ty::TyDynamic(..) => {
            MetadataCreationResult::new(
                        trait_pointer_metadata(cx, t, None, unique_type_id),
            false)
        }
        ty::TyForeign(..) => {
            MetadataCreationResult::new(
                        foreign_type_metadata(cx, t, unique_type_id),
            false)
        }
        ty::TyRawPtr(ty::TypeAndMut{ty, ..}) |
        ty::TyRef(_, ty, _) => {
            match ptr_metadata(ty) {
                Ok(res) => res,
                Err(metadata) => return metadata,
            }
        }
        ty::TyAdt(def, _) if def.is_box() => {
            match ptr_metadata(t.boxed_ty()) {
                Ok(res) => res,
                Err(metadata) => return metadata,
            }
        }
        ty::TyFnDef(..) | ty::TyFnPtr(_) => {
            let fn_metadata = subroutine_type_metadata(cx,
                                                       unique_type_id,
                                                       t.fn_sig(cx.tcx),
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
            let upvar_tys : Vec<_> = substs.upvar_tys(def_id, cx.tcx).collect();
            prepare_tuple_metadata(cx,
                                   t,
                                   &upvar_tys,
                                   unique_type_id,
                                   usage_site_span).finalize(cx)
        }
        ty::TyGenerator(def_id, substs,  _) => {
            let upvar_tys : Vec<_> = substs.field_tys(def_id, cx.tcx).map(|t| {
                cx.tcx.normalize_erasing_regions(ParamEnv::reveal_all(), t)
            }).collect();
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
            bug!("debuginfo: unexpected type in type_metadata: {:?}", t)
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

pub fn file_metadata(cx: &CodegenCx,
                     file_name: &FileName,
                     defining_crate: CrateNum) -> DIFile {
    debug!("file_metadata: file_name: {}, defining_crate: {}",
           file_name,
           defining_crate);

    let directory = if defining_crate == LOCAL_CRATE {
        &cx.sess().working_dir.0
    } else {
        // If the path comes from an upstream crate we assume it has been made
        // independent of the compiler's working directory one way or another.
        Path::new("")
    };

    file_metadata_raw(cx, &file_name.to_string(), &directory.to_string_lossy())
}

pub fn unknown_file_metadata(cx: &CodegenCx) -> DIFile {
    file_metadata_raw(cx, "<unknown>", "")
}

fn file_metadata_raw(cx: &CodegenCx,
                     file_name: &str,
                     directory: &str)
                     -> DIFile {
    let key = (Symbol::intern(file_name), Symbol::intern(directory));

    if let Some(file_metadata) = debug_context(cx).created_files.borrow().get(&key) {
        return *file_metadata;
    }

    debug!("file_metadata: file_name: {}, directory: {}", file_name, directory);

    let file_name = CString::new(file_name).unwrap();
    let directory = CString::new(directory).unwrap();

    let file_metadata = unsafe {
        llvm::LLVMRustDIBuilderCreateFile(DIB(cx),
                                          file_name.as_ptr(),
                                          directory.as_ptr())
    };

    let mut created_files = debug_context(cx).created_files.borrow_mut();
    created_files.insert(key, file_metadata);
    file_metadata
}

fn basic_type_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
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

    let (size, align) = cx.size_and_align_of(t);
    let name = CString::new(name).unwrap();
    let ty_metadata = unsafe {
        llvm::LLVMRustDIBuilderCreateBasicType(
            DIB(cx),
            name.as_ptr(),
            size.bits(),
            align.abi_bits() as u32,
            encoding)
    };

    return ty_metadata;
}

fn foreign_type_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                   t: Ty<'tcx>,
                                   unique_type_id: UniqueTypeId) -> DIType {
    debug!("foreign_type_metadata: {:?}", t);

    let name = compute_debuginfo_type_name(cx, t, false);
    create_struct_stub(cx, t, &name, unique_type_id, NO_SCOPE_METADATA)
}

fn pointer_type_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                   pointer_type: Ty<'tcx>,
                                   pointee_type_metadata: DIType)
                                   -> DIType {
    let (pointer_size, pointer_align) = cx.size_and_align_of(pointer_type);
    let name = compute_debuginfo_type_name(cx, pointer_type, false);
    let name = CString::new(name).unwrap();
    unsafe {
        llvm::LLVMRustDIBuilderCreatePointerType(
            DIB(cx),
            pointee_type_metadata,
            pointer_size.bits(),
            pointer_align.abi_bits() as u32,
            name.as_ptr())
    }
}

pub fn compile_unit_metadata(tcx: TyCtxt,
                             codegen_unit_name: &str,
                             debug_context: &CrateDebugContext)
                             -> DIDescriptor {
    let mut name_in_debuginfo = match tcx.sess.local_crate_source_file {
        Some(ref path) => path.clone(),
        None => PathBuf::from(&*tcx.crate_name(LOCAL_CRATE).as_str()),
    };

    // The OSX linker has an idiosyncrasy where it will ignore some debuginfo
    // if multiple object files with the same DW_AT_name are linked together.
    // As a workaround we generate unique names for each object file. Those do
    // not correspond to an actual source file but that should be harmless.
    if tcx.sess.target.target.options.is_like_osx {
        name_in_debuginfo.push("@");
        name_in_debuginfo.push(codegen_unit_name);
    }

    debug!("compile_unit_metadata: {:?}", name_in_debuginfo);
    // FIXME(#41252) Remove "clang LLVM" if we can get GDB and LLVM to play nice.
    let producer = format!("clang LLVM (rustc version {})",
                           (option_env!("CFG_VERSION")).expect("CFG_VERSION"));

    let name_in_debuginfo = name_in_debuginfo.to_string_lossy().into_owned();
    let name_in_debuginfo = CString::new(name_in_debuginfo).unwrap();
    let work_dir = CString::new(&tcx.sess.working_dir.0.to_string_lossy()[..]).unwrap();
    let producer = CString::new(producer).unwrap();
    let flags = "\0";
    let split_name = "\0";

    unsafe {
        let file_metadata = llvm::LLVMRustDIBuilderCreateFile(
            debug_context.builder, name_in_debuginfo.as_ptr(), work_dir.as_ptr());

        let unit_metadata = llvm::LLVMRustDIBuilderCreateCompileUnit(
            debug_context.builder,
            DW_LANG_RUST,
            file_metadata,
            producer.as_ptr(),
            tcx.sess.opts.optimize != config::OptLevel::No,
            flags.as_ptr() as *const _,
            0,
            split_name.as_ptr() as *const _);

        if tcx.sess.opts.debugging_opts.profile {
            let cu_desc_metadata = llvm::LLVMRustMetadataAsValue(debug_context.llcontext,
                                                                 unit_metadata);

            let gcov_cu_info = [
                path_to_mdstring(debug_context.llcontext,
                                 &tcx.output_filenames(LOCAL_CRATE).with_extension("gcno")),
                path_to_mdstring(debug_context.llcontext,
                                 &tcx.output_filenames(LOCAL_CRATE).with_extension("gcda")),
                cu_desc_metadata,
            ];
            let gcov_metadata = llvm::LLVMMDNodeInContext(debug_context.llcontext,
                                                          gcov_cu_info.as_ptr(),
                                                          gcov_cu_info.len() as c_uint);

            let llvm_gcov_ident = CString::new("llvm.gcov").unwrap();
            llvm::LLVMAddNamedMetadataOperand(debug_context.llmod,
                                              llvm_gcov_ident.as_ptr(),
                                              gcov_metadata);
        }

        return unit_metadata;
    };

    fn path_to_mdstring(llcx: llvm::ContextRef, path: &Path) -> llvm::ValueRef {
        let path_str = path2cstr(path);
        unsafe {
            llvm::LLVMMDStringInContext(llcx,
                                        path_str.as_ptr(),
                                        path_str.as_bytes().len() as c_uint)
        }
    }
}

struct MetadataCreationResult {
    metadata: DIType,
    already_stored_in_typemap: bool
}

impl MetadataCreationResult {
    fn new(metadata: DIType, already_stored_in_typemap: bool) -> MetadataCreationResult {
        MetadataCreationResult {
            metadata,
            already_stored_in_typemap,
        }
    }
}

// Description of a type member, which can either be a regular field (as in
// structs or tuples) or an enum variant.
#[derive(Debug)]
struct MemberDescription {
    name: String,
    type_metadata: DIType,
    offset: Size,
    size: Size,
    align: Align,
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
    fn create_member_descriptions<'a>(&self, cx: &CodegenCx<'a, 'tcx>)
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
    span: Span,
}

impl<'tcx> StructMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CodegenCx<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        let layout = cx.layout_of(self.ty);
        self.variant.fields.iter().enumerate().map(|(i, f)| {
            let name = if self.variant.ctor_kind == CtorKind::Fn {
                format!("__{}", i)
            } else {
                f.ident.to_string()
            };
            let field = layout.field(cx, i);
            let (size, align) = field.size_and_align();
            MemberDescription {
                name,
                type_metadata: type_metadata(cx, field.ty, self.span),
                offset: layout.fields.offset(i),
                size,
                align,
                flags: DIFlags::FlagZero,
            }
        }).collect()
    }
}


fn prepare_struct_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                     struct_type: Ty<'tcx>,
                                     unique_type_id: UniqueTypeId,
                                     span: Span)
                                     -> RecursiveTypeDescription<'tcx> {
    let struct_name = compute_debuginfo_type_name(cx, struct_type, false);

    let (struct_def_id, variant) = match struct_type.sty {
        ty::TyAdt(def, _) => (def.did, def.non_enum_variant()),
        _ => bug!("prepare_struct_metadata on a non-ADT")
    };

    let containing_scope = get_namespace_for_item(cx, struct_def_id);

    let struct_metadata_stub = create_struct_stub(cx,
                                                  struct_type,
                                                  &struct_name,
                                                  unique_type_id,
                                                  containing_scope);

    create_and_register_recursive_type_forward_declaration(
        cx,
        struct_type,
        unique_type_id,
        struct_metadata_stub,
        StructMDF(StructMemberDescriptionFactory {
            ty: struct_type,
            variant,
            span,
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
    fn create_member_descriptions<'a>(&self, cx: &CodegenCx<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        let layout = cx.layout_of(self.ty);
        self.component_types.iter().enumerate().map(|(i, &component_type)| {
            let (size, align) = cx.size_and_align_of(component_type);
            MemberDescription {
                name: format!("__{}", i),
                type_metadata: type_metadata(cx, component_type, self.span),
                offset: layout.fields.offset(i),
                size,
                align,
                flags: DIFlags::FlagZero,
            }
        }).collect()
    }
}

fn prepare_tuple_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                    tuple_type: Ty<'tcx>,
                                    component_types: &[Ty<'tcx>],
                                    unique_type_id: UniqueTypeId,
                                    span: Span)
                                    -> RecursiveTypeDescription<'tcx> {
    let tuple_name = compute_debuginfo_type_name(cx, tuple_type, false);

    create_and_register_recursive_type_forward_declaration(
        cx,
        tuple_type,
        unique_type_id,
        create_struct_stub(cx,
                           tuple_type,
                           &tuple_name[..],
                           unique_type_id,
                           NO_SCOPE_METADATA),
        TupleMDF(TupleMemberDescriptionFactory {
            ty: tuple_type,
            component_types: component_types.to_vec(),
            span,
        })
    )
}

//=-----------------------------------------------------------------------------
// Unions
//=-----------------------------------------------------------------------------

struct UnionMemberDescriptionFactory<'tcx> {
    layout: TyLayout<'tcx>,
    variant: &'tcx ty::VariantDef,
    span: Span,
}

impl<'tcx> UnionMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CodegenCx<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        self.variant.fields.iter().enumerate().map(|(i, f)| {
            let field = self.layout.field(cx, i);
            let (size, align) = field.size_and_align();
            MemberDescription {
                name: f.ident.to_string(),
                type_metadata: type_metadata(cx, field.ty, self.span),
                offset: Size::ZERO,
                size,
                align,
                flags: DIFlags::FlagZero,
            }
        }).collect()
    }
}

fn prepare_union_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                    union_type: Ty<'tcx>,
                                    unique_type_id: UniqueTypeId,
                                    span: Span)
                                    -> RecursiveTypeDescription<'tcx> {
    let union_name = compute_debuginfo_type_name(cx, union_type, false);

    let (union_def_id, variant) = match union_type.sty {
        ty::TyAdt(def, _) => (def.did, def.non_enum_variant()),
        _ => bug!("prepare_union_metadata on a non-ADT")
    };

    let containing_scope = get_namespace_for_item(cx, union_def_id);

    let union_metadata_stub = create_union_stub(cx,
                                                union_type,
                                                &union_name,
                                                unique_type_id,
                                                containing_scope);

    create_and_register_recursive_type_forward_declaration(
        cx,
        union_type,
        unique_type_id,
        union_metadata_stub,
        UnionMDF(UnionMemberDescriptionFactory {
            layout: cx.layout_of(union_type),
            variant,
            span,
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
    layout: TyLayout<'tcx>,
    discriminant_type_metadata: Option<DIType>,
    containing_scope: DIScope,
    span: Span,
}

impl<'tcx> EnumMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CodegenCx<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        let adt = &self.enum_type.ty_adt_def().unwrap();
        match self.layout.variants {
            layout::Variants::Single { .. } if adt.variants.is_empty() => vec![],
            layout::Variants::Single { index } => {
                let (variant_type_metadata, member_description_factory) =
                    describe_enum_variant(cx,
                                          self.layout,
                                          &adt.variants[index],
                                          NoDiscriminant,
                                          self.containing_scope,
                                          self.span);

                let member_descriptions =
                    member_description_factory.create_member_descriptions(cx);

                set_members_of_composite_type(cx,
                                              variant_type_metadata,
                                              &member_descriptions[..]);
                vec![
                    MemberDescription {
                        name: "".to_string(),
                        type_metadata: variant_type_metadata,
                        offset: Size::ZERO,
                        size: self.layout.size,
                        align: self.layout.align,
                        flags: DIFlags::FlagZero
                    }
                ]
            }
            layout::Variants::Tagged { ref variants, .. } => {
                let discriminant_info = RegularDiscriminant(self.discriminant_type_metadata
                    .expect(""));
                (0..variants.len()).map(|i| {
                    let variant = self.layout.for_variant(cx, i);
                    let (variant_type_metadata, member_desc_factory) =
                        describe_enum_variant(cx,
                                              variant,
                                              &adt.variants[i],
                                              discriminant_info,
                                              self.containing_scope,
                                              self.span);

                    let member_descriptions = member_desc_factory
                        .create_member_descriptions(cx);

                    set_members_of_composite_type(cx,
                                                  variant_type_metadata,
                                                  &member_descriptions);
                    MemberDescription {
                        name: "".to_string(),
                        type_metadata: variant_type_metadata,
                        offset: Size::ZERO,
                        size: variant.size,
                        align: variant.align,
                        flags: DIFlags::FlagZero
                    }
                }).collect()
            }
            layout::Variants::NicheFilling { dataful_variant, ref niche_variants, .. } => {
                let variant = self.layout.for_variant(cx, dataful_variant);
                // Create a description of the non-null variant
                let (variant_type_metadata, member_description_factory) =
                    describe_enum_variant(cx,
                                          variant,
                                          &adt.variants[dataful_variant],
                                          OptimizedDiscriminant,
                                          self.containing_scope,
                                          self.span);

                let variant_member_descriptions =
                    member_description_factory.create_member_descriptions(cx);

                set_members_of_composite_type(cx,
                                              variant_type_metadata,
                                              &variant_member_descriptions[..]);

                // Encode the information about the null variant in the union
                // member's name.
                let mut name = String::from("RUST$ENCODED$ENUM$");
                // HACK(eddyb) the debuggers should just handle offset+size
                // of discriminant instead of us having to recover its path.
                // Right now it's not even going to work for `niche_start > 0`,
                // and for multiple niche variants it only supports the first.
                fn compute_field_path<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                                name: &mut String,
                                                layout: TyLayout<'tcx>,
                                                offset: Size,
                                                size: Size) {
                    for i in 0..layout.fields.count() {
                        let field_offset = layout.fields.offset(i);
                        if field_offset > offset {
                            continue;
                        }
                        let inner_offset = offset - field_offset;
                        let field = layout.field(cx, i);
                        if inner_offset + size <= field.size {
                            write!(name, "{}$", i).unwrap();
                            compute_field_path(cx, name, field, inner_offset, size);
                        }
                    }
                }
                compute_field_path(cx, &mut name,
                                   self.layout,
                                   self.layout.fields.offset(0),
                                   self.layout.field(cx, 0).size);
                name.push_str(&adt.variants[*niche_variants.start()].name.as_str());

                // Create the (singleton) list of descriptions of union members.
                vec![
                    MemberDescription {
                        name,
                        type_metadata: variant_type_metadata,
                        offset: Size::ZERO,
                        size: variant.size,
                        align: variant.align,
                        flags: DIFlags::FlagZero
                    }
                ]
            }
        }
    }
}

// Creates MemberDescriptions for the fields of a single enum variant.
struct VariantMemberDescriptionFactory<'tcx> {
    // Cloned from the layout::Struct describing the variant.
    offsets: Vec<layout::Size>,
    args: Vec<(String, Ty<'tcx>)>,
    discriminant_type_metadata: Option<DIType>,
    span: Span,
}

impl<'tcx> VariantMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CodegenCx<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        self.args.iter().enumerate().map(|(i, &(ref name, ty))| {
            let (size, align) = cx.size_and_align_of(ty);
            MemberDescription {
                name: name.to_string(),
                type_metadata: match self.discriminant_type_metadata {
                    Some(metadata) if i == 0 => metadata,
                    _ => type_metadata(cx, ty, self.span)
                },
                offset: self.offsets[i],
                size,
                align,
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
fn describe_enum_variant<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                   layout: layout::TyLayout<'tcx>,
                                   variant: &'tcx ty::VariantDef,
                                   discriminant_info: EnumDiscriminantInfo,
                                   containing_scope: DIScope,
                                   span: Span)
                                   -> (DICompositeType, MemberDescriptionFactory<'tcx>) {
    let variant_name = variant.name.as_str();
    let unique_type_id = debug_context(cx).type_map
                                          .borrow_mut()
                                          .get_unique_type_id_of_enum_variant(
                                              cx,
                                              layout.ty,
                                              &variant_name);

    let metadata_stub = create_struct_stub(cx,
                                           layout.ty,
                                           &variant_name,
                                           unique_type_id,
                                           containing_scope);

    // If this is not a univariant enum, there is also the discriminant field.
    let (discr_offset, discr_arg) = match discriminant_info {
        RegularDiscriminant(_) => {
            let enum_layout = cx.layout_of(layout.ty);
            (Some(enum_layout.fields.offset(0)),
             Some(("RUST$ENUM$DISR".to_string(), enum_layout.field(cx, 0).ty)))
        }
        _ => (None, None),
    };
    let offsets = discr_offset.into_iter().chain((0..layout.fields.count()).map(|i| {
        layout.fields.offset(i)
    })).collect();

    // Build an array of (field name, field type) pairs to be captured in the factory closure.
    let args = discr_arg.into_iter().chain((0..layout.fields.count()).map(|i| {
        let name = if variant.ctor_kind == CtorKind::Fn {
            format!("__{}", i)
        } else {
            variant.fields[i].ident.to_string()
        };
        (name, layout.field(cx, i).ty)
    })).collect();

    let member_description_factory =
        VariantMDF(VariantMemberDescriptionFactory {
            offsets,
            args,
            discriminant_type_metadata: match discriminant_info {
                RegularDiscriminant(discriminant_type_metadata) => {
                    Some(discriminant_type_metadata)
                }
                _ => None
            },
            span,
        });

    (metadata_stub, member_description_factory)
}

fn prepare_enum_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                   enum_type: Ty<'tcx>,
                                   enum_def_id: DefId,
                                   unique_type_id: UniqueTypeId,
                                   span: Span)
                                   -> RecursiveTypeDescription<'tcx> {
    let enum_name = compute_debuginfo_type_name(cx, enum_type, false);

    let containing_scope = get_namespace_for_item(cx, enum_def_id);
    // FIXME: This should emit actual file metadata for the enum, but we
    // currently can't get the necessary information when it comes to types
    // imported from other crates. Formerly we violated the ODR when performing
    // LTO because we emitted debuginfo for the same type with varying file
    // metadata, so as a workaround we pretend that the type comes from
    // <unknown>
    let file_metadata = unknown_file_metadata(cx);

    let def = enum_type.ty_adt_def().unwrap();
    let enumerators_metadata: Vec<DIDescriptor> = def.discriminants(cx.tcx)
        .zip(&def.variants)
        .map(|(discr, v)| {
            let token = v.name.as_str();
            let name = CString::new(token.as_bytes()).unwrap();
            unsafe {
                llvm::LLVMRustDIBuilderCreateEnumerator(
                    DIB(cx),
                    name.as_ptr(),
                    // FIXME: what if enumeration has i128 discriminant?
                    discr.val as u64)
            }
        })
        .collect();

    let discriminant_type_metadata = |discr: layout::Primitive| {
        let disr_type_key = (enum_def_id, discr);
        let cached_discriminant_type_metadata = debug_context(cx).created_enum_disr_types
                                                                 .borrow()
                                                                 .get(&disr_type_key).cloned();
        match cached_discriminant_type_metadata {
            Some(discriminant_type_metadata) => discriminant_type_metadata,
            None => {
                let (discriminant_size, discriminant_align) =
                    (discr.size(cx), discr.align(cx));
                let discriminant_base_type_metadata =
                    type_metadata(cx, discr.to_ty(cx.tcx), syntax_pos::DUMMY_SP);
                let discriminant_name = get_enum_discriminant_name(cx, enum_def_id).as_str();

                let name = CString::new(discriminant_name.as_bytes()).unwrap();
                let discriminant_type_metadata = unsafe {
                    llvm::LLVMRustDIBuilderCreateEnumerationType(
                        DIB(cx),
                        containing_scope,
                        name.as_ptr(),
                        file_metadata,
                        UNKNOWN_LINE_NUMBER,
                        discriminant_size.bits(),
                        discriminant_align.abi_bits() as u32,
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

    let layout = cx.layout_of(enum_type);

    let discriminant_type_metadata = match layout.variants {
        layout::Variants::Single { .. } |
        layout::Variants::NicheFilling { .. } => None,
        layout::Variants::Tagged { ref tag, .. } => {
            Some(discriminant_type_metadata(tag.value))
        }
    };

    match (&layout.abi, discriminant_type_metadata) {
        (&layout::Abi::Scalar(_), Some(discr)) => return FinalMetadata(discr),
        _ => {}
    }

    let (enum_type_size, enum_type_align) = layout.size_and_align();

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
        enum_type_size.bits(),
        enum_type_align.abi_bits() as u32,
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
        EnumMDF(EnumMemberDescriptionFactory {
            enum_type,
            layout,
            discriminant_type_metadata,
            containing_scope,
            span,
        }),
    );

    fn get_enum_discriminant_name(cx: &CodegenCx,
                                  def_id: DefId)
                                  -> InternedString {
        cx.tcx.item_name(def_id)
    }
}

/// Creates debug information for a composite type, that is, anything that
/// results in a LLVM struct.
///
/// Examples of Rust types to use this are: structs, tuples, boxes, vecs, and enums.
fn composite_type_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                     composite_type: Ty<'tcx>,
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
                                                     composite_type,
                                                     composite_type_name,
                                                     composite_type_unique_id,
                                                     containing_scope);
    // ... and immediately create and add the member descriptions.
    set_members_of_composite_type(cx,
                                  composite_type_metadata,
                                  member_descriptions);

    return composite_type_metadata;
}

fn set_members_of_composite_type(cx: &CodegenCx,
                                 composite_type_metadata: DICompositeType,
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
        .map(|member_description| {
            let member_name = member_description.name.as_bytes();
            let member_name = CString::new(member_name).unwrap();
            unsafe {
                llvm::LLVMRustDIBuilderCreateMemberType(
                    DIB(cx),
                    composite_type_metadata,
                    member_name.as_ptr(),
                    unknown_file_metadata(cx),
                    UNKNOWN_LINE_NUMBER,
                    member_description.size.bits(),
                    member_description.align.abi_bits() as u32,
                    member_description.offset.bits(),
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
fn create_struct_stub<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                struct_type: Ty<'tcx>,
                                struct_type_name: &str,
                                unique_type_id: UniqueTypeId,
                                containing_scope: DIScope)
                                -> DICompositeType {
    let (struct_size, struct_align) = cx.size_and_align_of(struct_type);

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
            struct_size.bits(),
            struct_align.abi_bits() as u32,
            DIFlags::FlagZero,
            ptr::null_mut(),
            empty_array,
            0,
            ptr::null_mut(),
            unique_type_id.as_ptr())
    };

    return metadata_stub;
}

fn create_union_stub<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                               union_type: Ty<'tcx>,
                               union_type_name: &str,
                               unique_type_id: UniqueTypeId,
                               containing_scope: DIScope)
                               -> DICompositeType {
    let (union_size, union_align) = cx.size_and_align_of(union_type);

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
            union_size.bits(),
            union_align.abi_bits() as u32,
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
pub fn create_global_var_metadata(cx: &CodegenCx,
                                  def_id: DefId,
                                  global: ValueRef) {
    if cx.dbg_cx.is_none() {
        return;
    }

    let tcx = cx.tcx;
    let attrs = tcx.codegen_fn_attrs(def_id);

    if attrs.flags.contains(CodegenFnAttrFlags::NO_DEBUG) {
        return;
    }

    let no_mangle = attrs.flags.contains(CodegenFnAttrFlags::NO_MANGLE);
    // We may want to remove the namespace scope if we're in an extern block, see:
    // https://github.com/rust-lang/rust/pull/46457#issuecomment-351750952
    let var_scope = get_namespace_for_item(cx, def_id);
    let span = tcx.def_span(def_id);

    let (file_metadata, line_number) = if span != syntax_pos::DUMMY_SP {
        let loc = span_start(cx, span);
        (file_metadata(cx, &loc.file.name, LOCAL_CRATE), loc.line as c_uint)
    } else {
        (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER)
    };

    let is_local_to_unit = is_node_local_to_unit(cx, def_id);
    let variable_type = Instance::mono(cx.tcx, def_id).ty(cx.tcx);
    let type_metadata = type_metadata(cx, variable_type, span);
    let var_name = tcx.item_name(def_id).to_string();
    let var_name = CString::new(var_name).unwrap();
    let linkage_name = if no_mangle {
        None
    } else {
        let linkage_name = mangled_name_of_instance(cx, Instance::mono(tcx, def_id));
        Some(CString::new(linkage_name.to_string()).unwrap())
    };

    let global_align = cx.align_of(variable_type);

    unsafe {
        llvm::LLVMRustDIBuilderCreateStaticVariable(DIB(cx),
                                                    var_scope,
                                                    var_name.as_ptr(),
                                                    // If null, linkage_name field is omitted,
                                                    // which is what we want for no_mangle statics
                                                    linkage_name.as_ref()
                                                     .map_or(ptr::null(), |name| name.as_ptr()),
                                                    file_metadata,
                                                    line_number,
                                                    type_metadata,
                                                    is_local_to_unit,
                                                    global,
                                                    ptr::null_mut(),
                                                    global_align.abi() as u32,
        );
    }
}

// Creates an "extension" of an existing DIScope into another file.
pub fn extend_scope_to_file(cx: &CodegenCx,
                            scope_metadata: DIScope,
                            file: &syntax_pos::FileMap,
                            defining_crate: CrateNum)
                            -> DILexicalBlock {
    let file_metadata = file_metadata(cx, &file.name, defining_crate);
    unsafe {
        llvm::LLVMRustDIBuilderCreateLexicalBlockFile(
            DIB(cx),
            scope_metadata,
            file_metadata)
    }
}

/// Creates debug information for the given vtable, which is for the
/// given type.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_vtable_metadata<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                        ty: ty::Ty<'tcx>,
                                        vtable: ValueRef) {
    if cx.dbg_cx.is_none() {
        return;
    }

    let type_metadata = type_metadata(cx, ty, syntax_pos::DUMMY_SP);

    unsafe {
        // LLVMRustDIBuilderCreateStructType() wants an empty array. A null
        // pointer will lead to hard to trace and debug LLVM assertions
        // later on in llvm/lib/IR/Value.cpp.
        let empty_array = create_DIArray(DIB(cx), &[]);

        let name = CString::new("vtable").unwrap();

        // Create a new one each time.  We don't want metadata caching
        // here, because each vtable will refer to a unique containing
        // type.
        let vtable_type = llvm::LLVMRustDIBuilderCreateStructType(
            DIB(cx),
            NO_SCOPE_METADATA,
            name.as_ptr(),
            unknown_file_metadata(cx),
            UNKNOWN_LINE_NUMBER,
            Size::ZERO.bits(),
            cx.tcx.data_layout.pointer_align.abi_bits() as u32,
            DIFlags::FlagArtificial,
            ptr::null_mut(),
            empty_array,
            0,
            type_metadata,
            name.as_ptr()
        );

        llvm::LLVMRustDIBuilderCreateStaticVariable(DIB(cx),
                                                    NO_SCOPE_METADATA,
                                                    name.as_ptr(),
                                                    // LLVM 3.9
                                                    // doesn't accept
                                                    // null here, so
                                                    // pass the name
                                                    // as the linkage
                                                    // name.
                                                    name.as_ptr(),
                                                    unknown_file_metadata(cx),
                                                    UNKNOWN_LINE_NUMBER,
                                                    vtable_type,
                                                    true,
                                                    vtable,
                                                    ptr::null_mut(),
                                                    0);
    }
}
