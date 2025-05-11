use std::borrow::Cow;

use rustc_abi::{FieldIdx, TagEncoding, VariantIdx, Variants};
use rustc_codegen_ssa::debuginfo::type_names::{compute_debuginfo_type_name, cpp_like_debuginfo};
use rustc_codegen_ssa::debuginfo::{tag_base_type, wants_c_like_enum_debuginfo};
use rustc_codegen_ssa::traits::MiscCodegenMethods;
use rustc_hir::def::CtorKind;
use rustc_index::IndexSlice;
use rustc_middle::bug;
use rustc_middle::mir::CoroutineLayout;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, AdtDef, CoroutineArgs, CoroutineArgsExt, Ty, VariantDef};
use rustc_span::Symbol;

use super::type_map::{DINodeCreationResult, UniqueTypeId};
use super::{SmallVec, size_and_align_of};
use crate::common::{AsCCharPtr, CodegenCx};
use crate::debuginfo::metadata::type_map::{self, Stub};
use crate::debuginfo::metadata::{
    UNKNOWN_LINE_NUMBER, build_field_di_node, build_generic_type_param_di_nodes,
    file_metadata_from_def_id, type_di_node, unknown_file_metadata,
};
use crate::debuginfo::utils::{DIB, create_DIArray, get_namespace_for_item};
use crate::llvm::debuginfo::{DIFlags, DIType};
use crate::llvm::{self};

mod cpp_like;
mod native;

pub(super) fn build_enum_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    let enum_type = unique_type_id.expect_ty();
    let &ty::Adt(enum_adt_def, _) = enum_type.kind() else {
        bug!("build_enum_type_di_node() called with non-enum type: `{:?}`", enum_type)
    };

    let enum_type_and_layout = cx.layout_of(enum_type);

    if wants_c_like_enum_debuginfo(cx.tcx, enum_type_and_layout) {
        return build_c_style_enum_di_node(cx, enum_adt_def, enum_type_and_layout);
    }

    if cpp_like_debuginfo(cx.tcx) {
        cpp_like::build_enum_type_di_node(cx, unique_type_id)
    } else {
        native::build_enum_type_di_node(cx, unique_type_id)
    }
}

pub(super) fn build_coroutine_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    if cpp_like_debuginfo(cx.tcx) {
        cpp_like::build_coroutine_di_node(cx, unique_type_id)
    } else {
        native::build_coroutine_di_node(cx, unique_type_id)
    }
}

/// Build the debuginfo node for a C-style enum, i.e. an enum the variants of which have no fields.
///
/// The resulting debuginfo will be a DW_TAG_enumeration_type.
fn build_c_style_enum_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_adt_def: AdtDef<'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
) -> DINodeCreationResult<'ll> {
    let containing_scope = get_namespace_for_item(cx, enum_adt_def.did());
    let enum_adt_def_id = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
        Some(enum_adt_def.did())
    } else {
        None
    };
    DINodeCreationResult {
        di_node: build_enumeration_type_di_node(
            cx,
            &compute_debuginfo_type_name(cx.tcx, enum_type_and_layout.ty, false),
            tag_base_type(cx.tcx, enum_type_and_layout),
            enum_adt_def.discriminants(cx.tcx).map(|(variant_index, discr)| {
                let name = Cow::from(enum_adt_def.variant(variant_index).name.as_str());
                (name, discr.val)
            }),
            enum_adt_def_id,
            containing_scope,
        ),
        already_stored_in_typemap: false,
    }
}

/// Build a DW_TAG_enumeration_type debuginfo node, with the given base type and variants.
/// This is a helper function and does not register anything in the type map by itself.
///
/// `variants` is an iterator of (discr-value, variant-name).
fn build_enumeration_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    type_name: &str,
    base_type: Ty<'tcx>,
    enumerators: impl Iterator<Item = (Cow<'tcx, str>, u128)>,
    def_id: Option<rustc_span::def_id::DefId>,
    containing_scope: &'ll DIType,
) -> &'ll DIType {
    let is_unsigned = match base_type.kind() {
        ty::Int(_) => false,
        ty::Uint(_) => true,
        _ => bug!("build_enumeration_type_di_node() called with non-integer tag type."),
    };
    let (size, align) = cx.size_and_align_of(base_type);

    let enumerator_di_nodes: SmallVec<Option<&'ll DIType>> = enumerators
        .map(|(name, value)| unsafe {
            let value = [value as u64, (value >> 64) as u64];
            Some(llvm::LLVMRustDIBuilderCreateEnumerator(
                DIB(cx),
                name.as_c_char_ptr(),
                name.len(),
                value.as_ptr(),
                size.bits() as libc::c_uint,
                is_unsigned,
            ))
        })
        .collect();

    let (file_metadata, line_number) = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers
    {
        file_metadata_from_def_id(cx, def_id)
    } else {
        (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER)
    };

    unsafe {
        llvm::LLVMRustDIBuilderCreateEnumerationType(
            DIB(cx),
            containing_scope,
            type_name.as_c_char_ptr(),
            type_name.len(),
            file_metadata,
            line_number,
            size.bits(),
            align.bits() as u32,
            create_DIArray(DIB(cx), &enumerator_di_nodes[..]),
            type_di_node(cx, base_type),
            true,
        )
    }
}

/// Build the debuginfo node for the struct type describing a single variant of an enum.
///
/// ```txt
///       DW_TAG_structure_type              (top-level type for enum)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///           DW_TAG_variant                 (variant 1)
///           DW_TAG_variant                 (variant 2)
///           DW_TAG_variant                 (variant 3)
///  --->   DW_TAG_structure_type            (type of variant 1)
///  --->   DW_TAG_structure_type            (type of variant 2)
///  --->   DW_TAG_structure_type            (type of variant 3)
/// ```
///
/// In CPP-like mode, we have the exact same descriptions for each variant too:
///
/// ```txt
///       DW_TAG_union_type              (top-level type for enum)
///         DW_TAG_member                    (member for variant 1)
///         DW_TAG_member                    (member for variant 2)
///         DW_TAG_member                    (member for variant 3)
///  --->   DW_TAG_structure_type            (type of variant 1)
///  --->   DW_TAG_structure_type            (type of variant 2)
///  --->   DW_TAG_structure_type            (type of variant 3)
///         DW_TAG_enumeration_type          (type of tag)
/// ```
///
/// The node looks like:
///
/// ```txt
/// DW_TAG_structure_type
///   DW_AT_name                  <name-of-variant>
///   DW_AT_byte_size             0x00000010
///   DW_AT_alignment             0x00000008
///   DW_TAG_member
///     DW_AT_name                  <name-of-field-0>
///     DW_AT_type                  <0x0000018e>
///     DW_AT_alignment             0x00000004
///     DW_AT_data_member_location  4
///   DW_TAG_member
///     DW_AT_name                  <name-of-field-1>
///     DW_AT_type                  <0x00000195>
///     DW_AT_alignment             0x00000008
///     DW_AT_data_member_location  8
///   ...
/// ```
///
/// The type of a variant is always a struct type with the name of the variant
/// and a DW_TAG_member for each field (but not the discriminant).
fn build_enum_variant_struct_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    enum_type_di_node: &'ll DIType,
    variant_index: VariantIdx,
    variant_def: &VariantDef,
    variant_layout: TyAndLayout<'tcx>,
    di_flags: DIFlags,
) -> &'ll DIType {
    assert_eq!(variant_layout.ty, enum_type_and_layout.ty);

    let def_location = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
        Some(file_metadata_from_def_id(cx, Some(variant_def.def_id)))
    } else {
        None
    };

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            UniqueTypeId::for_enum_variant_struct_type(
                cx.tcx,
                enum_type_and_layout.ty,
                variant_index,
            ),
            variant_def.name.as_str(),
            def_location,
            // NOTE: We use size and align of enum_type, not from variant_layout:
            size_and_align_of(enum_type_and_layout),
            Some(enum_type_di_node),
            di_flags,
        ),
        |cx, struct_type_di_node| {
            (0..variant_layout.fields.count())
                .map(|field_index| {
                    let field_name = if variant_def.ctor_kind() != Some(CtorKind::Fn) {
                        // Fields have names
                        let field = &variant_def.fields[FieldIdx::from_usize(field_index)];
                        Cow::from(field.name.as_str())
                    } else {
                        // Tuple-like
                        super::tuple_field_name(field_index)
                    };

                    let field_layout = variant_layout.field(cx, field_index);

                    build_field_di_node(
                        cx,
                        struct_type_di_node,
                        &field_name,
                        field_layout,
                        variant_layout.fields.offset(field_index),
                        di_flags,
                        type_di_node(cx, field_layout.ty),
                        None,
                    )
                })
                .collect::<SmallVec<_>>()
        },
        |cx| build_generic_type_param_di_nodes(cx, enum_type_and_layout.ty),
    )
    .di_node
}

/// Build the struct type for describing a single coroutine state.
/// See [build_coroutine_variant_struct_type_di_node].
///
/// ```txt
///
///       DW_TAG_structure_type              (top-level type for enum)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///           DW_TAG_variant                 (variant 1)
///           DW_TAG_variant                 (variant 2)
///           DW_TAG_variant                 (variant 3)
///  --->   DW_TAG_structure_type            (type of variant 1)
///  --->   DW_TAG_structure_type            (type of variant 2)
///  --->   DW_TAG_structure_type            (type of variant 3)
///
/// ```
fn build_coroutine_variant_struct_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    variant_index: VariantIdx,
    coroutine_type_and_layout: TyAndLayout<'tcx>,
    coroutine_type_di_node: &'ll DIType,
    coroutine_layout: &CoroutineLayout<'tcx>,
    common_upvar_names: &IndexSlice<FieldIdx, Symbol>,
) -> &'ll DIType {
    let variant_name = CoroutineArgs::variant_name(variant_index);
    let unique_type_id = UniqueTypeId::for_enum_variant_struct_type(
        cx.tcx,
        coroutine_type_and_layout.ty,
        variant_index,
    );

    let variant_layout = coroutine_type_and_layout.for_variant(cx, variant_index);

    let coroutine_args = match coroutine_type_and_layout.ty.kind() {
        ty::Coroutine(_, args) => args.as_coroutine(),
        _ => unreachable!(),
    };

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            unique_type_id,
            &variant_name,
            None,
            size_and_align_of(coroutine_type_and_layout),
            Some(coroutine_type_di_node),
            DIFlags::FlagZero,
        ),
        |cx, variant_struct_type_di_node| {
            // Fields that just belong to this variant/state
            let state_specific_fields: SmallVec<_> = (0..variant_layout.fields.count())
                .map(|field_index| {
                    let coroutine_saved_local = coroutine_layout.variant_fields[variant_index]
                        [FieldIdx::from_usize(field_index)];
                    let field_name_maybe = coroutine_layout.field_names[coroutine_saved_local];
                    let field_name = field_name_maybe
                        .as_ref()
                        .map(|s| Cow::from(s.as_str()))
                        .unwrap_or_else(|| super::tuple_field_name(field_index));

                    let field_type = variant_layout.field(cx, field_index).ty;

                    build_field_di_node(
                        cx,
                        variant_struct_type_di_node,
                        &field_name,
                        cx.layout_of(field_type),
                        variant_layout.fields.offset(field_index),
                        DIFlags::FlagZero,
                        type_di_node(cx, field_type),
                        None,
                    )
                })
                .collect();

            // Fields that are common to all states
            let common_fields: SmallVec<_> = coroutine_args
                .prefix_tys()
                .iter()
                .zip(common_upvar_names)
                .enumerate()
                .map(|(index, (upvar_ty, upvar_name))| {
                    build_field_di_node(
                        cx,
                        variant_struct_type_di_node,
                        upvar_name.as_str(),
                        cx.layout_of(upvar_ty),
                        coroutine_type_and_layout.fields.offset(index),
                        DIFlags::FlagZero,
                        type_di_node(cx, upvar_ty),
                        None,
                    )
                })
                .collect();

            state_specific_fields.into_iter().chain(common_fields).collect()
        },
        |cx| build_generic_type_param_di_nodes(cx, coroutine_type_and_layout.ty),
    )
    .di_node
}

#[derive(Copy, Clone)]
enum DiscrResult {
    NoDiscriminant,
    Value(u128),
    Range(u128, u128),
}

impl DiscrResult {
    fn opt_single_val(&self) -> Option<u128> {
        if let Self::Value(d) = *self { Some(d) } else { None }
    }
}

/// Returns the discriminant value corresponding to the variant index.
///
/// Will return `None` if there is less than two variants (because then the enum won't have)
/// a tag, and if this is the untagged variant of a niche-layout enum (because then there is no
/// single discriminant value).
fn compute_discriminant_value<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    variant_index: VariantIdx,
) -> DiscrResult {
    match enum_type_and_layout.layout.variants() {
        &Variants::Single { .. } | &Variants::Empty => DiscrResult::NoDiscriminant,
        &Variants::Multiple { tag_encoding: TagEncoding::Direct, .. } => DiscrResult::Value(
            enum_type_and_layout.ty.discriminant_for_variant(cx.tcx, variant_index).unwrap().val,
        ),
        &Variants::Multiple {
            tag_encoding: TagEncoding::Niche { ref niche_variants, niche_start, untagged_variant },
            tag,
            ..
        } => {
            if variant_index == untagged_variant {
                let valid_range = enum_type_and_layout
                    .for_variant(cx, variant_index)
                    .largest_niche
                    .as_ref()
                    .unwrap()
                    .valid_range;

                let min = valid_range.start.min(valid_range.end);
                let min = tag.size(cx).truncate(min);

                let max = valid_range.start.max(valid_range.end);
                let max = tag.size(cx).truncate(max);

                DiscrResult::Range(min, max)
            } else {
                let value = (variant_index.as_u32() as u128)
                    .wrapping_sub(niche_variants.start().as_u32() as u128)
                    .wrapping_add(niche_start);
                let value = tag.size(cx).truncate(value);
                DiscrResult::Value(value)
            }
        }
    }
}
