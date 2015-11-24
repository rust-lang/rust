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
                   get_namespace_and_span_for_item, create_DIArray,
                   fn_should_be_ignored, is_node_local_to_unit};
use super::namespace::namespace_for_item;
use super::type_names::{compute_debuginfo_type_name, push_debuginfo_type_name};
use super::{declare_local, VariableKind, VariableAccess};

use llvm::{self, ValueRef};
use llvm::debuginfo::{DIType, DIFile, DIScope, DIDescriptor, DICompositeType};

use middle::def_id::DefId;
use middle::infer;
use middle::pat_util;
use middle::subst::{self, Substs};
use rustc::front::map as hir_map;
use rustc_front::hir;
use trans::{type_of, adt, machine, monomorphize};
use trans::common::{self, CrateContext, FunctionContext, Block};
use trans::_match::{BindingInfo, TransBindingMode};
use trans::type_::Type;
use middle::ty::{self, Ty};
use session::config::{self, FullDebugInfo};
use util::nodemap::FnvHashMap;
use util::common::path2cstr;

use libc::{c_uint, c_longlong};
use std::ffi::CString;
use std::path::Path;
use std::ptr;
use std::rc::Rc;
use syntax;
use syntax::util::interner::Interner;
use syntax::codemap::Span;
use syntax::{ast, ast_util, codemap};
use syntax::parse::token;


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

pub const UNKNOWN_LINE_NUMBER: c_uint = 0;
pub const UNKNOWN_COLUMN_NUMBER: c_uint = 0;

// ptr::null() doesn't work :(
const NO_FILE_METADATA: DIFile = (0 as DIFile);
const NO_SCOPE_METADATA: DIScope = (0 as DIScope);

const FLAGS_NONE: c_uint = 0;

#[derive(Copy, Debug, Hash, Eq, PartialEq, Clone)]
pub struct UniqueTypeId(ast::Name);

// The TypeMap is where the CrateDebugContext holds the type metadata nodes
// created so far. The metadata nodes are indexed by UniqueTypeId, and, for
// faster lookup, also by Ty. The TypeMap is responsible for creating
// UniqueTypeIds.
pub struct TypeMap<'tcx> {
    // The UniqueTypeIds created so far
    unique_id_interner: Interner<Rc<String>>,
    // A map from UniqueTypeId to debuginfo metadata for that type. This is a 1:1 mapping.
    unique_id_to_metadata: FnvHashMap<UniqueTypeId, DIType>,
    // A map from types to debuginfo metadata. This is a N:1 mapping.
    type_to_metadata: FnvHashMap<Ty<'tcx>, DIType>,
    // A map from types to UniqueTypeId. This is a N:1 mapping.
    type_to_unique_id: FnvHashMap<Ty<'tcx>, UniqueTypeId>
}

impl<'tcx> TypeMap<'tcx> {
    pub fn new() -> TypeMap<'tcx> {
        TypeMap {
            unique_id_interner: Interner::new(),
            type_to_metadata: FnvHashMap(),
            unique_id_to_metadata: FnvHashMap(),
            type_to_unique_id: FnvHashMap(),
        }
    }

    // Adds a Ty to metadata mapping to the TypeMap. The method will fail if
    // the mapping already exists.
    fn register_type_with_metadata<'a>(&mut self,
                                       cx: &CrateContext<'a, 'tcx>,
                                       type_: Ty<'tcx>,
                                       metadata: DIType) {
        if self.type_to_metadata.insert(type_, metadata).is_some() {
            cx.sess().bug(&format!("Type metadata for Ty '{}' is already in the TypeMap!",
                                   type_));
        }
    }

    // Adds a UniqueTypeId to metadata mapping to the TypeMap. The method will
    // fail if the mapping already exists.
    fn register_unique_id_with_metadata(&mut self,
                                        cx: &CrateContext,
                                        unique_type_id: UniqueTypeId,
                                        metadata: DIType) {
        if self.unique_id_to_metadata.insert(unique_type_id, metadata).is_some() {
            let unique_type_id_str = self.get_unique_type_id_as_string(unique_type_id);
            cx.sess().bug(&format!("Type metadata for unique id '{}' is already in the TypeMap!",
                                  &unique_type_id_str[..]));
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
    fn get_unique_type_id_as_string(&self, unique_type_id: UniqueTypeId) -> Rc<String> {
        let UniqueTypeId(interner_key) = unique_type_id;
        self.unique_id_interner.get(interner_key)
    }

    // Get the UniqueTypeId for the given type. If the UniqueTypeId for the given
    // type has been requested before, this is just a table lookup. Otherwise an
    // ID will be generated and stored for later lookup.
    fn get_unique_type_id_of_type<'a>(&mut self, cx: &CrateContext<'a, 'tcx>,
                                      type_: Ty<'tcx>) -> UniqueTypeId {

        // basic type             -> {:name of the type:}
        // tuple                  -> {tuple_(:param-uid:)*}
        // struct                 -> {struct_:svh: / :node-id:_<(:param-uid:),*> }
        // enum                   -> {enum_:svh: / :node-id:_<(:param-uid:),*> }
        // enum variant           -> {variant_:variant-name:_:enum-uid:}
        // reference (&)          -> {& :pointee-uid:}
        // mut reference (&mut)   -> {&mut :pointee-uid:}
        // ptr (*)                -> {* :pointee-uid:}
        // mut ptr (*mut)         -> {*mut :pointee-uid:}
        // unique ptr (box)       -> {box :pointee-uid:}
        // @-ptr (@)              -> {@ :pointee-uid:}
        // sized vec ([T; x])     -> {[:size:] :element-uid:}
        // unsized vec ([T])      -> {[] :element-uid:}
        // trait (T)              -> {trait_:svh: / :node-id:_<(:param-uid:),*> }
        // closure                -> {<unsafe_> <once_> :store-sigil: |(:param-uid:),* <,_...>| -> \
        //                             :return-type-uid: : (:bounds:)*}
        // function               -> {<unsafe_> <abi_> fn( (:param-uid:)* <,_...> ) -> \
        //                             :return-type-uid:}

        match self.type_to_unique_id.get(&type_).cloned() {
            Some(unique_type_id) => return unique_type_id,
            None => { /* generate one */}
        };

        let mut unique_type_id = String::with_capacity(256);
        unique_type_id.push('{');

        match type_.sty {
            ty::TyBool     |
            ty::TyChar     |
            ty::TyStr      |
            ty::TyInt(_)   |
            ty::TyUint(_)  |
            ty::TyFloat(_) => {
                push_debuginfo_type_name(cx, type_, false, &mut unique_type_id);
            },
            ty::TyEnum(def, substs) => {
                unique_type_id.push_str("enum ");
                from_def_id_and_substs(self, cx, def.did, substs, &mut unique_type_id);
            },
            ty::TyStruct(def, substs) => {
                unique_type_id.push_str("struct ");
                from_def_id_and_substs(self, cx, def.did, substs, &mut unique_type_id);
            },
            ty::TyTuple(ref component_types) if component_types.is_empty() => {
                push_debuginfo_type_name(cx, type_, false, &mut unique_type_id);
            },
            ty::TyTuple(ref component_types) => {
                unique_type_id.push_str("tuple ");
                for &component_type in component_types {
                    let component_type_id =
                        self.get_unique_type_id_of_type(cx, component_type);
                    let component_type_id =
                        self.get_unique_type_id_as_string(component_type_id);
                    unique_type_id.push_str(&component_type_id[..]);
                }
            },
            ty::TyBox(inner_type) => {
                unique_type_id.push_str("box ");
                let inner_type_id = self.get_unique_type_id_of_type(cx, inner_type);
                let inner_type_id = self.get_unique_type_id_as_string(inner_type_id);
                unique_type_id.push_str(&inner_type_id[..]);
            },
            ty::TyRawPtr(ty::TypeAndMut { ty: inner_type, mutbl } ) => {
                unique_type_id.push('*');
                if mutbl == hir::MutMutable {
                    unique_type_id.push_str("mut");
                }

                let inner_type_id = self.get_unique_type_id_of_type(cx, inner_type);
                let inner_type_id = self.get_unique_type_id_as_string(inner_type_id);
                unique_type_id.push_str(&inner_type_id[..]);
            },
            ty::TyRef(_, ty::TypeAndMut { ty: inner_type, mutbl }) => {
                unique_type_id.push('&');
                if mutbl == hir::MutMutable {
                    unique_type_id.push_str("mut");
                }

                let inner_type_id = self.get_unique_type_id_of_type(cx, inner_type);
                let inner_type_id = self.get_unique_type_id_as_string(inner_type_id);
                unique_type_id.push_str(&inner_type_id[..]);
            },
            ty::TyArray(inner_type, len) => {
                unique_type_id.push_str(&format!("[{}]", len));

                let inner_type_id = self.get_unique_type_id_of_type(cx, inner_type);
                let inner_type_id = self.get_unique_type_id_as_string(inner_type_id);
                unique_type_id.push_str(&inner_type_id[..]);
            },
            ty::TySlice(inner_type) => {
                unique_type_id.push_str("[]");

                let inner_type_id = self.get_unique_type_id_of_type(cx, inner_type);
                let inner_type_id = self.get_unique_type_id_as_string(inner_type_id);
                unique_type_id.push_str(&inner_type_id[..]);
            },
            ty::TyTrait(ref trait_data) => {
                unique_type_id.push_str("trait ");

                let principal = cx.tcx().erase_late_bound_regions(&trait_data.principal);

                from_def_id_and_substs(self,
                                       cx,
                                       principal.def_id,
                                       principal.substs,
                                       &mut unique_type_id);
            },
            ty::TyBareFn(_, &ty::BareFnTy{ unsafety, abi, ref sig } ) => {
                if unsafety == hir::Unsafety::Unsafe {
                    unique_type_id.push_str("unsafe ");
                }

                unique_type_id.push_str(abi.name());

                unique_type_id.push_str(" fn(");

                let sig = cx.tcx().erase_late_bound_regions(sig);
                let sig = infer::normalize_associated_type(cx.tcx(), &sig);

                for &parameter_type in &sig.inputs {
                    let parameter_type_id =
                        self.get_unique_type_id_of_type(cx, parameter_type);
                    let parameter_type_id =
                        self.get_unique_type_id_as_string(parameter_type_id);
                    unique_type_id.push_str(&parameter_type_id[..]);
                    unique_type_id.push(',');
                }

                if sig.variadic {
                    unique_type_id.push_str("...");
                }

                unique_type_id.push_str(")->");
                match sig.output {
                    ty::FnConverging(ret_ty) => {
                        let return_type_id = self.get_unique_type_id_of_type(cx, ret_ty);
                        let return_type_id = self.get_unique_type_id_as_string(return_type_id);
                        unique_type_id.push_str(&return_type_id[..]);
                    }
                    ty::FnDiverging => {
                        unique_type_id.push_str("!");
                    }
                }
            },
            ty::TyClosure(_, ref substs) if substs.upvar_tys.is_empty() => {
                push_debuginfo_type_name(cx, type_, false, &mut unique_type_id);
            },
            ty::TyClosure(_, ref substs) => {
                unique_type_id.push_str("closure ");
                for upvar_type in &substs.upvar_tys {
                    let upvar_type_id =
                        self.get_unique_type_id_of_type(cx, upvar_type);
                    let upvar_type_id =
                        self.get_unique_type_id_as_string(upvar_type_id);
                    unique_type_id.push_str(&upvar_type_id[..]);
                }
            },
            _ => {
                cx.sess().bug(&format!("get_unique_type_id_of_type() - unexpected type: {:?}",
                                       type_))
            }
        };

        unique_type_id.push('}');

        // Trim to size before storing permanently
        unique_type_id.shrink_to_fit();

        let key = self.unique_id_interner.intern(Rc::new(unique_type_id));
        self.type_to_unique_id.insert(type_, UniqueTypeId(key));

        return UniqueTypeId(key);

        fn from_def_id_and_substs<'a, 'tcx>(type_map: &mut TypeMap<'tcx>,
                                            cx: &CrateContext<'a, 'tcx>,
                                            def_id: DefId,
                                            substs: &subst::Substs<'tcx>,
                                            output: &mut String) {
            // First, find out the 'real' def_id of the type. Items inlined from
            // other crates have to be mapped back to their source.
            let source_def_id = if let Some(node_id) = cx.tcx().map.as_local_node_id(def_id) {
                match cx.external_srcs().borrow().get(&node_id).cloned() {
                    Some(source_def_id) => {
                        // The given def_id identifies the inlined copy of a
                        // type definition, let's take the source of the copy.
                        source_def_id
                    }
                    None => def_id
                }
            } else {
                def_id
            };

            // Get the crate hash as first part of the identifier.
            let crate_hash = if source_def_id.is_local() {
                cx.link_meta().crate_hash.clone()
            } else {
                cx.sess().cstore.get_crate_hash(source_def_id.krate)
            };

            output.push_str(crate_hash.as_str());
            output.push_str("/");
            output.push_str(&format!("{:x}", def_id.index.as_usize()));

            // Maybe check that there is no self type here.

            let tps = substs.types.get_slice(subst::TypeSpace);
            if !tps.is_empty() {
                output.push('<');

                for &type_parameter in tps {
                    let param_type_id =
                        type_map.get_unique_type_id_of_type(cx, type_parameter);
                    let param_type_id =
                        type_map.get_unique_type_id_as_string(param_type_id);
                    output.push_str(&param_type_id[..]);
                    output.push(',');
                }

                output.push('>');
            }
        }
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
                                           &self.get_unique_type_id_as_string(enum_type_id),
                                           variant_name);
        let interner_key = self.unique_id_interner.intern(Rc::new(enum_variant_type_id));
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
    type_map.register_unique_id_with_metadata(cx, unique_type_id, metadata_stub);
    type_map.register_type_with_metadata(cx, unfinished_type, metadata_stub);

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
                        cx.sess().bug(&format!("Forward declaration of potentially recursive type \
                                              '{:?}' was not found in TypeMap!",
                                              unfinished_type)
                                      );
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

    let mut signature_metadata: Vec<DIType> = Vec::with_capacity(signature.inputs.len() + 1);

    // return type
    signature_metadata.push(match signature.output {
        ty::FnConverging(ret_ty) => match ret_ty.sty {
            ty::TyTuple(ref tys) if tys.is_empty() => ptr::null_mut(),
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
                NO_FILE_METADATA,
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
        ty::TyTrait(ref data) => data.principal_def_id(),
        _ => {
            cx.sess().bug(&format!("debuginfo: Unexpected trait-object type in \
                                   trait_pointer_metadata(): {:?}",
                                   trait_type));
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
                            NO_FILE_METADATA,
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
        ty::TyEnum(def, _) => {
            prepare_enum_metadata(cx,
                                  t,
                                  def.did,
                                  unique_type_id,
                                  usage_site_span).finalize(cx)
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
        ty::TyTrait(..) => {
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
                ty::TyTrait(..) => {
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
        ty::TyBareFn(_, ref barefnty) => {
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
        ty::TyClosure(_, ref substs) => {
            prepare_tuple_metadata(cx,
                                   t,
                                   &substs.upvar_tys,
                                   unique_type_id,
                                   usage_site_span).finalize(cx)
        }
        ty::TyStruct(..) => {
            prepare_struct_metadata(cx,
                                    t,
                                    unique_type_id,
                                    usage_site_span).finalize(cx)
        }
        ty::TyTuple(ref elements) => {
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
            // Also make sure that we already have a TypeMap entry for the unique type id.
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
                                                t);
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
                            t);
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
    // FIXME (#9639): This needs to handle non-utf8 paths
    let work_dir = cx.sess().working_dir.to_str().unwrap();
    let file_name =
        if full_path.starts_with(work_dir) {
            &full_path[work_dir.len() + 1..full_path.len()]
        } else {
            full_path
        };

    file_metadata_(cx, full_path, file_name, &work_dir)
}

pub fn unknown_file_metadata(cx: &CrateContext) -> DIFile {
    // Regular filenames should not be empty, so we abuse an empty name as the
    // key for the special unknown file metadata
    file_metadata_(cx, "", "<unknown>", "")

}

fn file_metadata_(cx: &CrateContext, key: &str, file_name: &str, work_dir: &str) -> DIFile {
    match debug_context(cx).created_files.borrow().get(key) {
        Some(file_metadata) => return *file_metadata,
        None => ()
    }

    debug!("file_metadata: file_name: {}, work_dir: {}", file_name, work_dir);

    let file_name = CString::new(file_name).unwrap();
    let work_dir = CString::new(work_dir).unwrap();
    let file_metadata = unsafe {
        llvm::LLVMDIBuilderCreateFile(DIB(cx), file_name.as_ptr(),
                                      work_dir.as_ptr())
    };

    let mut created_files = debug_context(cx).created_files.borrow_mut();
    created_files.insert(key.to_string(), file_metadata);
    file_metadata
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

pub fn diverging_type_metadata(cx: &CrateContext) -> DIType {
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
        ty::TyTuple(ref elements) if elements.is_empty() =>
            ("()", DW_ATE_unsigned),
        ty::TyBool => ("bool", DW_ATE_boolean),
        ty::TyChar => ("char", DW_ATE_unsigned_char),
        ty::TyInt(int_ty) => {
            (ast_util::int_ty_to_string(int_ty), DW_ATE_signed)
        },
        ty::TyUint(uint_ty) => {
            (ast_util::uint_ty_to_string(uint_ty), DW_ATE_unsigned)
        },
        ty::TyFloat(float_ty) => {
            (ast_util::float_ty_to_string(float_ty), DW_ATE_float)
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
    flags: c_uint
}

// A factory for MemberDescriptions. It produces a list of member descriptions
// for some record-like type. MemberDescriptionFactories are used to defer the
// creation of type member descriptions in order to break cycles arising from
// recursive type definitions.
enum MemberDescriptionFactory<'tcx> {
    StructMDF(StructMemberDescriptionFactory<'tcx>),
    TupleMDF(TupleMemberDescriptionFactory<'tcx>),
    EnumMDF(EnumMemberDescriptionFactory<'tcx>),
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
    variant: ty::VariantDef<'tcx>,
    substs: &'tcx subst::Substs<'tcx>,
    is_simd: bool,
    span: Span,
}

impl<'tcx> StructMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CrateContext<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        if let ty::VariantKind::Unit = self.variant.kind() {
            return Vec::new();
        }

        let field_size = if self.is_simd {
            let fty = monomorphize::field_ty(cx.tcx(),
                                             self.substs,
                                             &self.variant.fields[0]);
            Some(machine::llsize_of_alloc(
                cx,
                type_of::type_of(cx, fty)
            ) as usize)
        } else {
            None
        };

        self.variant.fields.iter().enumerate().map(|(i, f)| {
            let name = if let ty::VariantKind::Tuple = self.variant.kind() {
                format!("__{}", i)
            } else {
                f.name.to_string()
            };
            let fty = monomorphize::field_ty(cx.tcx(), self.substs, f);

            let offset = if self.is_simd {
                FixedMemberOffset { bytes: i * field_size.unwrap() }
            } else {
                ComputedMemberOffset
            };

            MemberDescription {
                name: name,
                llvm_type: type_of::type_of(cx, fty),
                type_metadata: type_metadata(cx, fty, self.span),
                offset: offset,
                flags: FLAGS_NONE,
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

    let (variant, substs) = match struct_type.sty {
        ty::TyStruct(def, substs) => (def.struct_variant(), substs),
        _ => cx.tcx().sess.bug("prepare_struct_metadata on a non-struct")
    };

    let (containing_scope, _) = get_namespace_and_span_for_item(cx, variant.did);

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
            variant: variant,
            substs: substs,
            is_simd: struct_type.is_simd(),
            span: span,
        })
    )
}


//=-----------------------------------------------------------------------------
// Tuples
//=-----------------------------------------------------------------------------

// Creates MemberDescriptions for the fields of a tuple
struct TupleMemberDescriptionFactory<'tcx> {
    component_types: Vec<Ty<'tcx>>,
    span: Span,
}

impl<'tcx> TupleMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CrateContext<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        self.component_types
            .iter()
            .enumerate()
            .map(|(i, &component_type)| {
            MemberDescription {
                name: format!("__{}", i),
                llvm_type: type_of::type_of(cx, component_type),
                type_metadata: type_metadata(cx, component_type, self.span),
                offset: ComputedMemberOffset,
                flags: FLAGS_NONE,
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
            component_types: component_types.to_vec(),
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
    type_rep: Rc<adt::Repr<'tcx>>,
    discriminant_type_metadata: Option<DIType>,
    containing_scope: DIScope,
    file_metadata: DIFile,
    span: Span,
}

impl<'tcx> EnumMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CrateContext<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        let adt = &self.enum_type.ty_adt_def().unwrap();
        match *self.type_rep {
            adt::General(_, ref struct_defs, _) => {
                let discriminant_info = RegularDiscriminant(self.discriminant_type_metadata
                    .expect(""));
                struct_defs
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
                            flags: FLAGS_NONE
                        }
                    }).collect()
            },
            adt::Univariant(ref struct_def, _) => {
                assert!(adt.variants.len() <= 1);

                if adt.variants.is_empty() {
                    vec![]
                } else {
                    let (variant_type_metadata,
                         variant_llvm_type,
                         member_description_factory) =
                        describe_enum_variant(cx,
                                              self.enum_type,
                                              struct_def,
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
                            flags: FLAGS_NONE
                        }
                    ]
                }
            }
            adt::RawNullablePointer { nndiscr: non_null_variant_index, nnty, .. } => {
                // As far as debuginfo is concerned, the pointer this enum
                // represents is still wrapped in a struct. This is to make the
                // DWARF representation of enums uniform.

                // First create a description of the artificial wrapper struct:
                let non_null_variant = &adt.variants[non_null_variant_index as usize];
                let non_null_variant_name = non_null_variant.name.as_str();

                // The llvm type and metadata of the pointer
                let non_null_llvm_type = type_of::type_of(cx, nnty);
                let non_null_type_metadata = type_metadata(cx, nnty, self.span);

                // The type of the artificial struct wrapping the pointer
                let artificial_struct_llvm_type = Type::struct_(cx,
                                                                &[non_null_llvm_type],
                                                                false);

                // For the metadata of the wrapper struct, we need to create a
                // MemberDescription of the struct's single field.
                let sole_struct_member_description = MemberDescription {
                    name: match non_null_variant.kind() {
                        ty::VariantKind::Tuple => "__0".to_string(),
                        ty::VariantKind::Struct => {
                            non_null_variant.fields[0].name.to_string()
                        }
                        ty::VariantKind::Unit => unreachable!()
                    },
                    llvm_type: non_null_llvm_type,
                    type_metadata: non_null_type_metadata,
                    offset: FixedMemberOffset { bytes: 0 },
                    flags: FLAGS_NONE
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
                                            codemap::DUMMY_SP);

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
                        flags: FLAGS_NONE
                    }
                ]
            },
            adt::StructWrappedNullablePointer { nonnull: ref struct_def,
                                                nndiscr,
                                                ref discrfield, ..} => {
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
                let discrfield = discrfield.iter()
                                           .skip(1)
                                           .map(|x| x.to_string())
                                           .collect::<Vec<_>>().join("$");
                let union_member_name = format!("RUST$ENCODED$ENUM${}${}",
                                                discrfield,
                                                null_variant_name);

                // Create the (singleton) list of descriptions of union members.
                vec![
                    MemberDescription {
                        name: union_member_name,
                        llvm_type: variant_llvm_type,
                        type_metadata: variant_type_metadata,
                        offset: FixedMemberOffset { bytes: 0 },
                        flags: FLAGS_NONE
                    }
                ]
            },
            adt::CEnum(..) => cx.sess().span_bug(self.span, "This should be unreachable.")
        }
    }
}

// Creates MemberDescriptions for the fields of a single enum variant.
struct VariantMemberDescriptionFactory<'tcx> {
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
                offset: ComputedMemberOffset,
                flags: FLAGS_NONE
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
                                   struct_def: &adt::Struct<'tcx>,
                                   variant: ty::VariantDef<'tcx>,
                                   discriminant_info: EnumDiscriminantInfo,
                                   containing_scope: DIScope,
                                   span: Span)
                                   -> (DICompositeType, Type, MemberDescriptionFactory<'tcx>) {
    let variant_llvm_type =
        Type::struct_(cx, &struct_def.fields
                                    .iter()
                                    .map(|&t| type_of::type_of(cx, t))
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
    let mut arg_names: Vec<_> = match variant.kind() {
        ty::VariantKind::Unit => vec![],
        ty::VariantKind::Tuple => {
            variant.fields
                   .iter()
                   .enumerate()
                   .map(|(i, _)| format!("__{}", i))
                   .collect()
        }
        ty::VariantKind::Struct => {
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
        .zip(&struct_def.fields)
        .map(|(s, &t)| (s.to_string(), t))
        .collect();

    let member_description_factory =
        VariantMDF(VariantMemberDescriptionFactory {
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
                llvm::LLVMDIBuilderCreateEnumerator(
                    DIB(cx),
                    name.as_ptr(),
                    v.disr_val as u64)
            }
        })
        .collect();

    let discriminant_type_metadata = |inttype: syntax::attr::IntType| {
        let disr_type_key = (enum_def_id, inttype);
        let cached_discriminant_type_metadata = debug_context(cx).created_enum_disr_types
                                                                 .borrow()
                                                                 .get(&disr_type_key).cloned();
        match cached_discriminant_type_metadata {
            Some(discriminant_type_metadata) => discriminant_type_metadata,
            None => {
                let discriminant_llvm_type = adt::ll_inttype(cx, inttype);
                let (discriminant_size, discriminant_align) =
                    size_and_align_of(cx, discriminant_llvm_type);
                let discriminant_base_type_metadata =
                    type_metadata(cx,
                                  adt::ty_of_inttype(cx.tcx(), inttype),
                                  codemap::DUMMY_SP);
                let discriminant_name = get_enum_discriminant_name(cx, enum_def_id);

                let name = CString::new(discriminant_name.as_bytes()).unwrap();
                let discriminant_type_metadata = unsafe {
                    llvm::LLVMDIBuilderCreateEnumerationType(
                        DIB(cx),
                        containing_scope,
                        name.as_ptr(),
                        NO_FILE_METADATA,
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

    let type_rep = adt::represent_type(cx, enum_type);

    let discriminant_type_metadata = match *type_rep {
        adt::CEnum(inttype, _, _) => {
            return FinalMetadata(discriminant_type_metadata(inttype))
        },
        adt::RawNullablePointer { .. }           |
        adt::StructWrappedNullablePointer { .. } |
        adt::Univariant(..)                      => None,
        adt::General(inttype, _, _) => Some(discriminant_type_metadata(inttype)),
    };

    let enum_llvm_type = type_of::type_of(cx, enum_type);
    let (enum_type_size, enum_type_align) = size_and_align_of(cx, enum_llvm_type);

    let unique_type_id_str = debug_context(cx)
                             .type_map
                             .borrow()
                             .get_unique_type_id_as_string(unique_type_id);

    let enum_name = CString::new(enum_name).unwrap();
    let unique_type_id_str = CString::new(unique_type_id_str.as_bytes()).unwrap();
    let enum_metadata = unsafe {
        llvm::LLVMDIBuilderCreateUnionType(
        DIB(cx),
        containing_scope,
        enum_name.as_ptr(),
        file_metadata,
        UNKNOWN_LINE_NUMBER,
        bytes_to_bits(enum_type_size),
        bytes_to_bits(enum_type_align),
        0, // Flags
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
            type_rep: type_rep.clone(),
            discriminant_type_metadata: discriminant_type_metadata,
            containing_scope: containing_scope,
            file_metadata: file_metadata,
            span: span,
        }),
    );

    fn get_enum_discriminant_name(cx: &CrateContext,
                                  def_id: DefId)
                                  -> token::InternedString {
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
            cx.sess().bug("debuginfo::set_members_of_composite_type() - \
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
                llvm::LLVMDIBuilderCreateMemberType(
                    DIB(cx),
                    composite_type_metadata,
                    member_name.as_ptr(),
                    NO_FILE_METADATA,
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
        llvm::LLVMDICompositeTypeSetTypeArray(DIB(cx), composite_type_metadata, type_array);
    }
}

// A convenience wrapper around LLVMDIBuilderCreateStructType(). Does not do any
// caching, does not add any fields to the struct. This can be done later with
// set_members_of_composite_type().
fn create_struct_stub(cx: &CrateContext,
                      struct_llvm_type: Type,
                      struct_type_name: &str,
                      unique_type_id: UniqueTypeId,
                      containing_scope: DIScope)
                   -> DICompositeType {
    let (struct_size, struct_align) = size_and_align_of(cx, struct_llvm_type);

    let unique_type_id_str = debug_context(cx).type_map
                                              .borrow()
                                              .get_unique_type_id_as_string(unique_type_id);
    let name = CString::new(struct_type_name).unwrap();
    let unique_type_id = CString::new(unique_type_id_str.as_bytes()).unwrap();
    let metadata_stub = unsafe {
        // LLVMDIBuilderCreateStructType() wants an empty array. A null
        // pointer will lead to hard to trace and debug LLVM assertions
        // later on in llvm/lib/IR/Value.cpp.
        let empty_array = create_DIArray(DIB(cx), &[]);

        llvm::LLVMDIBuilderCreateStructType(
            DIB(cx),
            containing_scope,
            name.as_ptr(),
            NO_FILE_METADATA,
            UNKNOWN_LINE_NUMBER,
            bytes_to_bits(struct_size),
            bytes_to_bits(struct_align),
            0,
            ptr::null_mut(),
            empty_array,
            0,
            ptr::null_mut(),
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

    // Don't create debuginfo for globals inlined from other crates. The other
    // crate should already contain debuginfo for it. More importantly, the
    // global might not even exist in un-inlined form anywhere which would lead
    // to a linker errors.
    if cx.external_srcs().borrow().contains_key(&node_id) {
        return;
    }

    let var_item = cx.tcx().map.get(node_id);

    let (name, span) = match var_item {
        hir_map::NodeItem(item) => {
            match item.node {
                hir::ItemStatic(..) => (item.name, item.span),
                hir::ItemConst(..) => (item.name, item.span),
                _ => {
                    cx.sess()
                      .span_bug(item.span,
                                &format!("debuginfo::\
                                         create_global_var_metadata() -
                                         Captured var-id refers to \
                                         unexpected ast_item variant: {:?}",
                                        var_item))
                }
            }
        },
        _ => cx.sess().bug(&format!("debuginfo::create_global_var_metadata() \
                                    - Captured var-id refers to unexpected \
                                    hir_map variant: {:?}",
                                   var_item))
    };

    let (file_metadata, line_number) = if span != codemap::DUMMY_SP {
        let loc = span_start(cx, span);
        (file_metadata(cx, &loc.file.name), loc.line as c_uint)
    } else {
        (NO_FILE_METADATA, UNKNOWN_LINE_NUMBER)
    };

    let is_local_to_unit = is_node_local_to_unit(cx, node_id);
    let variable_type = cx.tcx().node_id_to_type(node_id);
    let type_metadata = type_metadata(cx, variable_type, span);
    let node_def_id = cx.tcx().map.local_def_id(node_id);
    let namespace_node = namespace_for_item(cx, node_def_id);
    let var_name = name.to_string();
    let linkage_name =
        namespace_node.mangled_name_of_contained_item(&var_name[..]);
    let var_scope = namespace_node.scope;

    let var_name = CString::new(var_name).unwrap();
    let linkage_name = CString::new(linkage_name).unwrap();
    unsafe {
        llvm::LLVMDIBuilderCreateStaticVariable(DIB(cx),
                                                var_scope,
                                                var_name.as_ptr(),
                                                linkage_name.as_ptr(),
                                                file_metadata,
                                                line_number,
                                                type_metadata,
                                                is_local_to_unit,
                                                global,
                                                ptr::null_mut());
    }
}

/// Creates debug information for the given local variable.
///
/// This function assumes that there's a datum for each pattern component of the
/// local in `bcx.fcx.lllocals`.
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_local_var_metadata(bcx: Block, local: &hir::Local) {
    if bcx.unreachable.get() ||
       fn_should_be_ignored(bcx.fcx) ||
       bcx.sess().opts.debuginfo != FullDebugInfo  {
        return;
    }

    let cx = bcx.ccx();
    let def_map = &cx.tcx().def_map;
    let locals = bcx.fcx.lllocals.borrow();

    pat_util::pat_bindings(def_map, &*local.pat, |_, node_id, span, var_name| {
        let datum = match locals.get(&node_id) {
            Some(datum) => datum,
            None => {
                bcx.sess().span_bug(span,
                    &format!("no entry in lllocals table for {}",
                            node_id));
            }
        };

        if unsafe { llvm::LLVMIsAAllocaInst(datum.val) } == ptr::null_mut() {
            cx.sess().span_bug(span, "debuginfo::create_local_var_metadata() - \
                                      Referenced variable location is not an alloca!");
        }

        let scope_metadata = scope_metadata(bcx.fcx, node_id, span);

        declare_local(bcx,
                      var_name.node,
                      datum.ty,
                      scope_metadata,
                      VariableAccess::DirectVariable { alloca: datum.val },
                      VariableKind::LocalVariable,
                      span);
    })
}

/// Creates debug information for a variable captured in a closure.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_captured_var_metadata<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                                node_id: ast::NodeId,
                                                env_pointer: ValueRef,
                                                env_index: usize,
                                                captured_by_ref: bool,
                                                span: Span) {
    if bcx.unreachable.get() ||
       fn_should_be_ignored(bcx.fcx) ||
       bcx.sess().opts.debuginfo != FullDebugInfo {
        return;
    }

    let cx = bcx.ccx();

    let ast_item = cx.tcx().map.find(node_id);

    let variable_name = match ast_item {
        None => {
            cx.sess().span_bug(span, "debuginfo::create_captured_var_metadata: node not found");
        }
        Some(hir_map::NodeLocal(pat)) => {
            match pat.node {
                hir::PatIdent(_, ref path1, _) => {
                    path1.node.name
                }
                _ => {
                    cx.sess()
                      .span_bug(span,
                                &format!(
                                "debuginfo::create_captured_var_metadata() - \
                                 Captured var-id refers to unexpected \
                                 hir_map variant: {:?}",
                                 ast_item));
                }
            }
        }
        _ => {
            cx.sess()
              .span_bug(span,
                        &format!("debuginfo::create_captured_var_metadata() - \
                                 Captured var-id refers to unexpected \
                                 hir_map variant: {:?}",
                                ast_item));
        }
    };

    let variable_type = common::node_id_type(bcx, node_id);
    let scope_metadata = bcx.fcx.debug_context.get_ref(cx, span).fn_metadata;

    // env_pointer is the alloca containing the pointer to the environment,
    // so it's type is **EnvironmentType. In order to find out the type of
    // the environment we have to "dereference" two times.
    let llvm_env_data_type = common::val_ty(env_pointer).element_type()
                                                        .element_type();
    let byte_offset_of_var_in_env = machine::llelement_offset(cx,
                                                              llvm_env_data_type,
                                                              env_index);

    let address_operations = unsafe {
        [llvm::LLVMDIBuilderCreateOpDeref(),
         llvm::LLVMDIBuilderCreateOpPlus(),
         byte_offset_of_var_in_env as i64,
         llvm::LLVMDIBuilderCreateOpDeref()]
    };

    let address_op_count = if captured_by_ref {
        address_operations.len()
    } else {
        address_operations.len() - 1
    };

    let variable_access = VariableAccess::IndirectVariable {
        alloca: env_pointer,
        address_operations: &address_operations[..address_op_count]
    };

    declare_local(bcx,
                  variable_name,
                  variable_type,
                  scope_metadata,
                  variable_access,
                  VariableKind::CapturedVariable,
                  span);
}

/// Creates debug information for a local variable introduced in the head of a
/// match-statement arm.
///
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_match_binding_metadata<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                                 variable_name: ast::Name,
                                                 binding: BindingInfo<'tcx>) {
    if bcx.unreachable.get() ||
       fn_should_be_ignored(bcx.fcx) ||
       bcx.sess().opts.debuginfo != FullDebugInfo {
        return;
    }

    let scope_metadata = scope_metadata(bcx.fcx, binding.id, binding.span);
    let aops = unsafe {
        [llvm::LLVMDIBuilderCreateOpDeref()]
    };
    // Regardless of the actual type (`T`) we're always passed the stack slot
    // (alloca) for the binding. For ByRef bindings that's a `T*` but for ByMove
    // bindings we actually have `T**`. So to get the actual variable we need to
    // dereference once more. For ByCopy we just use the stack slot we created
    // for the binding.
    let var_access = match binding.trmode {
        TransBindingMode::TrByCopy(llbinding) |
        TransBindingMode::TrByMoveIntoCopy(llbinding) => VariableAccess::DirectVariable {
            alloca: llbinding
        },
        TransBindingMode::TrByMoveRef => VariableAccess::IndirectVariable {
            alloca: binding.llmatch,
            address_operations: &aops
        },
        TransBindingMode::TrByRef => VariableAccess::DirectVariable {
            alloca: binding.llmatch
        }
    };

    declare_local(bcx,
                  variable_name,
                  binding.ty,
                  scope_metadata,
                  var_access,
                  VariableKind::LocalVariable,
                  binding.span);
}

/// Creates debug information for the given function argument.
///
/// This function assumes that there's a datum for each pattern component of the
/// argument in `bcx.fcx.lllocals`.
/// Adds the created metadata nodes directly to the crate's IR.
pub fn create_argument_metadata(bcx: Block, arg: &hir::Arg) {
    if bcx.unreachable.get() ||
       fn_should_be_ignored(bcx.fcx) ||
       bcx.sess().opts.debuginfo != FullDebugInfo {
        return;
    }

    let def_map = &bcx.tcx().def_map;
    let scope_metadata = bcx
                         .fcx
                         .debug_context
                         .get_ref(bcx.ccx(), arg.pat.span)
                         .fn_metadata;
    let locals = bcx.fcx.lllocals.borrow();

    pat_util::pat_bindings(def_map, &*arg.pat, |_, node_id, span, var_name| {
        let datum = match locals.get(&node_id) {
            Some(v) => v,
            None => {
                bcx.sess().span_bug(span,
                    &format!("no entry in lllocals table for {}",
                            node_id));
            }
        };

        if unsafe { llvm::LLVMIsAAllocaInst(datum.val) } == ptr::null_mut() {
            bcx.sess().span_bug(span, "debuginfo::create_argument_metadata() - \
                                       Referenced variable location is not an alloca!");
        }

        let argument_index = {
            let counter = &bcx
                          .fcx
                          .debug_context
                          .get_ref(bcx.ccx(), span)
                          .argument_counter;
            let argument_index = counter.get();
            counter.set(argument_index + 1);
            argument_index
        };

        declare_local(bcx,
                      var_name.node,
                      datum.ty,
                      scope_metadata,
                      VariableAccess::DirectVariable { alloca: datum.val },
                      VariableKind::ArgumentVariable(argument_index),
                      span);
    })
}
