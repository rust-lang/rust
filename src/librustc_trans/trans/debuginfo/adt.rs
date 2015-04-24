// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Common facilities for record-like types (structs, enums, tuples)

use self::MemberDescriptionFactory::*;
use self::EnumDiscriminantInfo::*;
use self::MemberOffset::*;

use super::{UNKNOWN_FILE_METADATA, UNKNOWN_SCOPE_METADATA, UNKNOWN_LINE_NUMBER,
            UniqueTypeId, FLAGS_NONE, create_and_register_recursive_type_forward_declaration};
use super::utils::{debug_context, DIB, span_start, bytes_to_bits,
                   size_and_align_of, get_namespace_and_span_for_item};
use super::create::create_DIArray;
use super::types::compute_debuginfo_type_name;
use super::metadata::{type_metadata, file_metadata};

use super::RecursiveTypeDescription::{self, FinalMetadata};

use llvm;
use llvm::debuginfo::{DIType, DIFile, DIScope, DIDescriptor, DICompositeType};
use metadata::csearch;
use middle::subst::{self, Substs};
use trans::{adt, machine, type_of};
use trans::common::CrateContext;
use trans::monomorphize;
use trans::type_::Type;
use middle::ty::{self, Ty, ClosureTyper};

use libc::c_uint;
use std::ffi::CString;
use std::ptr;
use std::rc::Rc;
use syntax::codemap::Span;
use syntax::{ast, codemap};
use syntax::parse::token::{self, special_idents};


pub enum MemberOffset {
    FixedMemberOffset { bytes: usize },
    // For ComputedMemberOffset, the offset is read from the llvm type definition.
    ComputedMemberOffset
}

// Description of a type member, which can either be a regular field (as in
// structs or tuples) or an enum variant.
pub struct MemberDescription {
    pub name: String,
    pub llvm_type: Type,
    pub type_metadata: DIType,
    pub offset: MemberOffset,
    pub flags: c_uint
}

// A factory for MemberDescriptions. It produces a list of member descriptions
// for some record-like type. MemberDescriptionFactories are used to defer the
// creation of type member descriptions in order to break cycles arising from
// recursive type definitions.
pub enum MemberDescriptionFactory<'tcx> {
    StructMDF(StructMemberDescriptionFactory<'tcx>),
    TupleMDF(TupleMemberDescriptionFactory<'tcx>),
    EnumMDF(EnumMemberDescriptionFactory<'tcx>),
    VariantMDF(VariantMemberDescriptionFactory<'tcx>)
}

impl<'tcx> MemberDescriptionFactory<'tcx> {
    pub fn create_member_descriptions<'a>(&self, cx: &CrateContext<'a, 'tcx>)
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
    fields: Vec<ty::field<'tcx>>,
    is_simd: bool,
    span: Span,
}

impl<'tcx> StructMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CrateContext<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
        if self.fields.is_empty() {
            return Vec::new();
        }

        let field_size = if self.is_simd {
            machine::llsize_of_alloc(cx, type_of::type_of(cx, self.fields[0].mt.ty)) as usize
        } else {
            0xdeadbeef
        };

        self.fields.iter().enumerate().map(|(i, field)| {
            let name = if field.name == special_idents::unnamed_field.name {
                format!("__{}", i)
            } else {
                token::get_name(field.name).to_string()
            };

            let offset = if self.is_simd {
                assert!(field_size != 0xdeadbeef);
                FixedMemberOffset { bytes: i * field_size }
            } else {
                ComputedMemberOffset
            };

            MemberDescription {
                name: name,
                llvm_type: type_of::type_of(cx, field.mt.ty),
                type_metadata: type_metadata(cx, field.mt.ty, self.span),
                offset: offset,
                flags: FLAGS_NONE,
            }
        }).collect()
    }
}


pub fn prepare_struct_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                         struct_type: Ty<'tcx>,
                                         def_id: ast::DefId,
                                         substs: &subst::Substs<'tcx>,
                                         unique_type_id: UniqueTypeId,
                                         span: Span)
                                         -> RecursiveTypeDescription<'tcx> {
    let struct_name = compute_debuginfo_type_name(cx, struct_type, false);
    let struct_llvm_type = type_of::type_of(cx, struct_type);

    let (containing_scope, _) = get_namespace_and_span_for_item(cx, def_id);

    let struct_metadata_stub = create_struct_stub(cx,
                                                  struct_llvm_type,
                                                  &struct_name[..],
                                                  unique_type_id,
                                                  containing_scope);

    let mut fields = ty::struct_fields(cx.tcx(), def_id, substs);

    // The `Ty` values returned by `ty::struct_fields` can still contain
    // `ty_projection` variants, so normalize those away.
    for field in &mut fields {
        field.mt.ty = monomorphize::normalize_associated_type(cx.tcx(), &field.mt.ty);
    }

    create_and_register_recursive_type_forward_declaration(
        cx,
        struct_type,
        unique_type_id,
        struct_metadata_stub,
        struct_llvm_type,
        StructMDF(StructMemberDescriptionFactory {
            fields: fields,
            is_simd: ty::type_is_simd(cx.tcx(), struct_type),
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

pub fn prepare_tuple_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
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
                           UNKNOWN_SCOPE_METADATA),
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
    variants: Rc<Vec<Rc<ty::VariantInfo<'tcx>>>>,
    discriminant_type_metadata: Option<DIType>,
    containing_scope: DIScope,
    file_metadata: DIFile,
    span: Span,
}

impl<'tcx> EnumMemberDescriptionFactory<'tcx> {
    fn create_member_descriptions<'a>(&self, cx: &CrateContext<'a, 'tcx>)
                                      -> Vec<MemberDescription> {
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
                                                  &*(*self.variants)[i],
                                                  discriminant_info,
                                                  self.containing_scope,
                                                  self.span);

                        let member_descriptions = member_desc_factory
                            .create_member_descriptions(cx);

                        set_members_of_composite_type(cx,
                                                      variant_type_metadata,
                                                      variant_llvm_type,
                                                      &member_descriptions[..]);
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
                assert!(self.variants.len() <= 1);

                if self.variants.is_empty() {
                    vec![]
                } else {
                    let (variant_type_metadata,
                         variant_llvm_type,
                         member_description_factory) =
                        describe_enum_variant(cx,
                                              self.enum_type,
                                              struct_def,
                                              &*(*self.variants)[0],
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
                let non_null_variant = &(*self.variants)[non_null_variant_index as usize];
                let non_null_variant_name = token::get_name(non_null_variant.name);

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
                    name: match non_null_variant.arg_names {
                        Some(ref names) => token::get_name(names[0]).to_string(),
                        None => "__0".to_string()
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
                let null_variant_name = token::get_name((*self.variants)[null_variant_index].name);
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
                                          &*(*self.variants)[nndiscr as usize],
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
                let null_variant_name = token::get_name((*self.variants)[null_variant_index].name);
                let discrfield = discrfield.iter()
                                           .skip(1)
                                           .map(|x| x.to_string())
                                           .collect::<Vec<_>>().connect("$");
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
                                   variant_info: &ty::VariantInfo<'tcx>,
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

    let variant_name = token::get_name(variant_info.name);
    let variant_name = &variant_name;
    let unique_type_id = debug_context(cx).type_map
                                          .borrow_mut()
                                          .get_unique_type_id_of_enum_variant(
                                              cx,
                                              enum_type,
                                              variant_name);

    let metadata_stub = create_struct_stub(cx,
                                           variant_llvm_type,
                                           variant_name,
                                           unique_type_id,
                                           containing_scope);

    // Get the argument names from the enum variant info
    let mut arg_names: Vec<_> = match variant_info.arg_names {
        Some(ref names) => {
            names.iter()
                 .map(|&name| token::get_name(name).to_string())
                 .collect()
        }
        None => {
            variant_info.args
                        .iter()
                        .enumerate()
                        .map(|(i, _)| format!("__{}", i))
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
        .zip(struct_def.fields.iter())
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

pub fn prepare_enum_metadata<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                       enum_type: Ty<'tcx>,
                                       enum_def_id: ast::DefId,
                                       unique_type_id: UniqueTypeId,
                                       span: Span)
                                       -> RecursiveTypeDescription<'tcx> {
    let enum_name = compute_debuginfo_type_name(cx, enum_type, false);

    let (containing_scope, definition_span) = get_namespace_and_span_for_item(cx, enum_def_id);
    let loc = span_start(cx, definition_span);
    let file_metadata = file_metadata(cx, &loc.file.name);

    let variants = ty::enum_variants(cx.tcx(), enum_def_id);

    let enumerators_metadata: Vec<DIDescriptor> = variants
        .iter()
        .map(|v| {
            let token = token::get_name(v.name);
            let name = CString::new(token.as_bytes()).unwrap();
            unsafe {
                llvm::LLVMDIBuilderCreateEnumerator(
                    DIB(cx),
                    name.as_ptr(),
                    v.disr_val as u64)
            }
        })
        .collect();

    let discriminant_type_metadata = |inttype| {
        // We can reuse the type of the discriminant for all monomorphized
        // instances of an enum because it doesn't depend on any type
        // parameters. The def_id, uniquely identifying the enum's polytype acts
        // as key in this cache.
        let cached_discriminant_type_metadata = debug_context(cx).created_enum_disr_types
                                                                 .borrow()
                                                                 .get(&enum_def_id).cloned();
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
                        UNKNOWN_FILE_METADATA,
                        UNKNOWN_LINE_NUMBER,
                        bytes_to_bits(discriminant_size),
                        bytes_to_bits(discriminant_align),
                        create_DIArray(DIB(cx), &enumerators_metadata),
                        discriminant_base_type_metadata)
                };

                debug_context(cx).created_enum_disr_types
                                 .borrow_mut()
                                 .insert(enum_def_id, discriminant_type_metadata);

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
        UNKNOWN_FILE_METADATA,
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
            variants: variants,
            discriminant_type_metadata: discriminant_type_metadata,
            containing_scope: containing_scope,
            file_metadata: file_metadata,
            span: span,
        }),
    );

    fn get_enum_discriminant_name(cx: &CrateContext,
                                  def_id: ast::DefId)
                                  -> token::InternedString {
        let name = if def_id.krate == ast::LOCAL_CRATE {
            cx.tcx().map.get_path_elem(def_id.node).name()
        } else {
            csearch::get_item_path(cx.tcx(), def_id).last().unwrap().name()
        };

        token::get_name(name)
    }
}

/// Creates debug information for a composite type, that is, anything that
/// results in a LLVM struct.
///
/// Examples of Rust types to use this are: structs, tuples, boxes, vecs, and enums.
pub fn composite_type_metadata(cx: &CrateContext,
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

pub fn set_members_of_composite_type(cx: &CrateContext,
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
                    UNKNOWN_FILE_METADATA,
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
            UNKNOWN_FILE_METADATA,
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
