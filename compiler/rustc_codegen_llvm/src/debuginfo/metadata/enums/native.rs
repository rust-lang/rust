use std::borrow::Cow;

use libc::c_uint;
use rustc_abi::{Size, TagEncoding, VariantIdx, Variants};
use rustc_codegen_ssa::debuginfo::type_names::compute_debuginfo_type_name;
use rustc_codegen_ssa::debuginfo::{tag_base_type, wants_c_like_enum_debuginfo};
use rustc_codegen_ssa::traits::{ConstCodegenMethods, MiscCodegenMethods};
use rustc_middle::bug;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::{self};
use smallvec::smallvec;

use crate::common::{AsCCharPtr, CodegenCx};
use crate::debuginfo::metadata::type_map::{self, Stub, StubInfo, UniqueTypeId};
use crate::debuginfo::metadata::{
    DINodeCreationResult, NO_GENERICS, SmallVec, UNKNOWN_LINE_NUMBER, create_member_type,
    file_metadata, file_metadata_from_def_id, size_and_align_of, type_di_node,
    unknown_file_metadata, visibility_di_flags,
};
use crate::debuginfo::utils::{DIB, create_DIArray, get_namespace_for_item};
use crate::llvm::debuginfo::{DIFile, DIFlags, DIType};
use crate::llvm::{self};

/// Build the debuginfo node for an enum type. The listing below shows how such a
/// type looks like at the LLVM IR/DWARF level. It is a `DW_TAG_structure_type`
/// with a single `DW_TAG_variant_part` that in turn contains a `DW_TAG_variant`
/// for each variant of the enum. The variant-part also contains a single member
/// describing the discriminant, and a nested struct type for each of the variants.
///
/// ```txt
///  ---> DW_TAG_structure_type              (top-level type for enum)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///           DW_TAG_variant                 (variant 1)
///           DW_TAG_variant                 (variant 2)
///           DW_TAG_variant                 (variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
/// ```
pub(super) fn build_enum_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    let enum_type = unique_type_id.expect_ty();
    let &ty::Adt(enum_adt_def, _) = enum_type.kind() else {
        bug!("build_enum_type_di_node() called with non-enum type: `{:?}`", enum_type)
    };

    let containing_scope = get_namespace_for_item(cx, enum_adt_def.did());
    let enum_type_and_layout = cx.layout_of(enum_type);
    let enum_type_name = compute_debuginfo_type_name(cx.tcx, enum_type, false);

    let visibility_flags = visibility_di_flags(cx, enum_adt_def.did(), enum_adt_def.did());

    assert!(!wants_c_like_enum_debuginfo(cx.tcx, enum_type_and_layout));

    let def_location = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
        Some(file_metadata_from_def_id(cx, Some(enum_adt_def.did())))
    } else {
        None
    };

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            unique_type_id,
            &enum_type_name,
            def_location,
            size_and_align_of(enum_type_and_layout),
            Some(containing_scope),
            visibility_flags,
        ),
        |cx, enum_type_di_node| {
            // Build the struct type for each variant. These will be referenced by the
            // DW_TAG_variant DIEs inside of the DW_TAG_variant_part DIE.
            // We also called the names for the corresponding DW_TAG_variant DIEs here.
            let variant_member_infos: SmallVec<_> = enum_adt_def
                .variant_range()
                .map(|variant_index| VariantMemberInfo {
                    variant_index,
                    variant_name: Cow::from(enum_adt_def.variant(variant_index).name.as_str()),
                    variant_struct_type_di_node: super::build_enum_variant_struct_type_di_node(
                        cx,
                        enum_type_and_layout,
                        enum_type_di_node,
                        variant_index,
                        enum_adt_def.variant(variant_index),
                        enum_type_and_layout.for_variant(cx, variant_index),
                        visibility_flags,
                    ),
                    source_info: if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
                        Some(file_metadata_from_def_id(
                            cx,
                            Some(enum_adt_def.variant(variant_index).def_id),
                        ))
                    } else {
                        None
                    },
                })
                .collect();

            let enum_adt_def_id = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
                Some(enum_adt_def.did())
            } else {
                None
            };
            smallvec![build_enum_variant_part_di_node(
                cx,
                enum_type_and_layout,
                enum_type_di_node,
                enum_adt_def_id,
                &variant_member_infos[..],
            )]
        },
        // We don't seem to be emitting generic args on the enum type, it seems. Rather
        // they get attached to the struct type of each variant.
        NO_GENERICS,
    )
}

/// Build the debuginfo node for a coroutine environment. It looks the same as the debuginfo for
/// an enum. See [build_enum_type_di_node] for more information.
///
/// ```txt
///
///  ---> DW_TAG_structure_type              (top-level type for the coroutine)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///           DW_TAG_variant                 (variant 1)
///           DW_TAG_variant                 (variant 2)
///           DW_TAG_variant                 (variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
///
/// ```
pub(super) fn build_coroutine_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    let coroutine_type = unique_type_id.expect_ty();
    let &ty::Coroutine(coroutine_def_id, coroutine_args) = coroutine_type.kind() else {
        bug!("build_coroutine_di_node() called with non-coroutine type: `{:?}`", coroutine_type)
    };

    let containing_scope = get_namespace_for_item(cx, coroutine_def_id);
    let coroutine_type_and_layout = cx.layout_of(coroutine_type);

    assert!(!wants_c_like_enum_debuginfo(cx.tcx, coroutine_type_and_layout));

    let coroutine_type_name = compute_debuginfo_type_name(cx.tcx, coroutine_type, false);

    let def_location = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
        Some(file_metadata_from_def_id(cx, Some(coroutine_def_id)))
    } else {
        None
    };

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            unique_type_id,
            &coroutine_type_name,
            def_location,
            size_and_align_of(coroutine_type_and_layout),
            Some(containing_scope),
            DIFlags::FlagZero,
        ),
        |cx, coroutine_type_di_node| {
            let coroutine_layout =
                cx.tcx.coroutine_layout(coroutine_def_id, coroutine_args).unwrap();

            let Variants::Multiple { tag_encoding: TagEncoding::Direct, ref variants, .. } =
                coroutine_type_and_layout.variants
            else {
                bug!(
                    "Encountered coroutine with non-direct-tag layout: {:?}",
                    coroutine_type_and_layout
                )
            };

            let common_upvar_names =
                cx.tcx.closure_saved_names_of_captured_variables(coroutine_def_id);

            // Build variant struct types
            let variant_struct_type_di_nodes: SmallVec<_> = variants
                .indices()
                .map(|variant_index| {
                    // FIXME: This is problematic because just a number is not a valid identifier.
                    //        CoroutineArgs::variant_name(variant_index), would be consistent
                    //        with enums?
                    let variant_name = format!("{}", variant_index.as_usize()).into();

                    let span = coroutine_layout.variant_source_info[variant_index].span;
                    let source_info = if !span.is_dummy() {
                        let loc = cx.lookup_debug_loc(span.lo());
                        Some((file_metadata(cx, &loc.file), loc.line))
                    } else {
                        None
                    };

                    VariantMemberInfo {
                        variant_index,
                        variant_name,
                        variant_struct_type_di_node:
                            super::build_coroutine_variant_struct_type_di_node(
                                cx,
                                variant_index,
                                coroutine_type_and_layout,
                                coroutine_type_di_node,
                                coroutine_layout,
                                common_upvar_names,
                            ),
                        source_info,
                    }
                })
                .collect();

            let coroutine_def_id = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers {
                Some(coroutine_def_id)
            } else {
                None
            };
            smallvec![build_enum_variant_part_di_node(
                cx,
                coroutine_type_and_layout,
                coroutine_type_di_node,
                coroutine_def_id,
                &variant_struct_type_di_nodes[..],
            )]
        },
        // We don't seem to be emitting generic args on the coroutine type, it seems. Rather
        // they get attached to the struct type of each variant.
        NO_GENERICS,
    )
}

/// Builds the DW_TAG_variant_part of an enum or coroutine debuginfo node:
///
/// ```txt
///       DW_TAG_structure_type              (top-level type for enum)
/// --->    DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///           DW_TAG_variant                 (variant 1)
///           DW_TAG_variant                 (variant 2)
///           DW_TAG_variant                 (variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
/// ```
fn build_enum_variant_part_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    enum_type_di_node: &'ll DIType,
    enum_type_def_id: Option<rustc_span::def_id::DefId>,
    variant_member_infos: &[VariantMemberInfo<'_, 'll>],
) -> &'ll DIType {
    let tag_member_di_node =
        build_discr_member_di_node(cx, enum_type_and_layout, enum_type_di_node);

    let variant_part_unique_type_id =
        UniqueTypeId::for_enum_variant_part(cx.tcx, enum_type_and_layout.ty);

    let (file_metadata, line_number) = if cx.sess().opts.unstable_opts.debug_info_type_line_numbers
    {
        file_metadata_from_def_id(cx, enum_type_def_id)
    } else {
        (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER)
    };

    let stub = StubInfo::new(
        cx,
        variant_part_unique_type_id,
        |cx, variant_part_unique_type_id_str| unsafe {
            let variant_part_name = "";
            llvm::LLVMRustDIBuilderCreateVariantPart(
                DIB(cx),
                enum_type_di_node,
                variant_part_name.as_c_char_ptr(),
                variant_part_name.len(),
                file_metadata,
                line_number,
                enum_type_and_layout.size.bits(),
                enum_type_and_layout.align.abi.bits() as u32,
                DIFlags::FlagZero,
                tag_member_di_node,
                create_DIArray(DIB(cx), &[]),
                variant_part_unique_type_id_str.as_c_char_ptr(),
                variant_part_unique_type_id_str.len(),
            )
        },
    );

    type_map::build_type_with_children(
        cx,
        stub,
        |cx, variant_part_di_node| {
            variant_member_infos
                .iter()
                .map(|variant_member_info| {
                    build_enum_variant_member_di_node(
                        cx,
                        enum_type_and_layout,
                        variant_part_di_node,
                        variant_member_info,
                    )
                })
                .collect()
        },
        NO_GENERICS,
    )
    .di_node
}

/// Builds the DW_TAG_member describing where we can find the tag of an enum.
/// Returns `None` if the enum does not have a tag.
///
/// ```txt
///
///       DW_TAG_structure_type              (top-level type for enum)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
/// --->      DW_TAG_member                  (discriminant member)
///           DW_TAG_variant                 (variant 1)
///           DW_TAG_variant                 (variant 2)
///           DW_TAG_variant                 (variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
///
/// ```
fn build_discr_member_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_or_coroutine_type_and_layout: TyAndLayout<'tcx>,
    enum_or_coroutine_type_di_node: &'ll DIType,
) -> Option<&'ll DIType> {
    let tag_name = match enum_or_coroutine_type_and_layout.ty.kind() {
        ty::Coroutine(..) => "__state",
        _ => "",
    };

    // NOTE: This is actually wrong. This will become a member of
    //       of the DW_TAG_variant_part. But, due to LLVM's API, that
    //       can only be constructed with this DW_TAG_member already in created.
    //       In LLVM IR the wrong scope will be listed but when DWARF is
    //       generated from it, the DW_TAG_member will be a child the
    //       DW_TAG_variant_part.
    let containing_scope = enum_or_coroutine_type_di_node;

    match enum_or_coroutine_type_and_layout.layout.variants() {
        // A single-variant or no-variant enum has no discriminant.
        &Variants::Single { .. } | &Variants::Empty => None,

        &Variants::Multiple { tag_field, .. } => {
            let tag_base_type = tag_base_type(cx.tcx, enum_or_coroutine_type_and_layout);
            let ty = type_di_node(cx, tag_base_type);
            let file = unknown_file_metadata(cx);

            let layout = cx.layout_of(tag_base_type);

            Some(create_member_type(
                cx,
                containing_scope,
                &tag_name,
                file,
                UNKNOWN_LINE_NUMBER,
                layout,
                enum_or_coroutine_type_and_layout.fields.offset(tag_field.as_usize()),
                DIFlags::FlagArtificial,
                ty,
            ))
        }
    }
}

/// Build the debuginfo node for `DW_TAG_variant`:
///
/// ```txt
///       DW_TAG_structure_type              (top-level type for enum)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///  --->     DW_TAG_variant                 (variant 1)
///  --->     DW_TAG_variant                 (variant 2)
///  --->     DW_TAG_variant                 (variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
/// ```
///
/// This node looks like:
///
/// ```txt
/// DW_TAG_variant
///   DW_AT_discr_value           0
///   DW_TAG_member
///     DW_AT_name                  None
///     DW_AT_type                  <0x000002a1>
///     DW_AT_alignment             0x00000002
///     DW_AT_data_member_location  0
/// ```
///
/// The DW_AT_discr_value is optional, and is omitted if
///   - This is the only variant of a univariant enum (i.e. their is no discriminant)
///   - This is the "untagged" variant of a niche-layout enum
///     (where only the other variants are identified by a single value)
///
/// There is only ever a single member, the type of which is a struct that describes the
/// fields of the variant (excluding the discriminant). The name of the member is the name
/// of the variant as given in the source code. The DW_AT_data_member_location is always
/// zero.
///
/// Note that the LLVM DIBuilder API is a bit unintuitive here. The DW_TAG_variant subtree
/// (including the DW_TAG_member) is built by a single call to
/// `LLVMRustDIBuilderCreateVariantMemberType()`.
fn build_enum_variant_member_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    variant_part_di_node: &'ll DIType,
    variant_member_info: &VariantMemberInfo<'_, 'll>,
) -> &'ll DIType {
    let variant_index = variant_member_info.variant_index;
    let discr_value = super::compute_discriminant_value(cx, enum_type_and_layout, variant_index);

    let (file_di_node, line_number) = variant_member_info
        .source_info
        .unwrap_or_else(|| (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER));

    let discr = discr_value.opt_single_val().map(|value| {
        let tag_base_type = tag_base_type(cx.tcx, enum_type_and_layout);
        let size = cx.size_of(tag_base_type);
        cx.const_uint_big(cx.type_ix(size.bits()), value)
    });

    unsafe {
        llvm::LLVMRustDIBuilderCreateVariantMemberType(
            DIB(cx),
            variant_part_di_node,
            variant_member_info.variant_name.as_c_char_ptr(),
            variant_member_info.variant_name.len(),
            file_di_node,
            line_number,
            enum_type_and_layout.size.bits(),
            enum_type_and_layout.align.abi.bits() as u32,
            Size::ZERO.bits(),
            discr,
            DIFlags::FlagZero,
            variant_member_info.variant_struct_type_di_node,
        )
    }
}

/// Information needed for building a `DW_TAG_variant`:
///
/// ```txt
///       DW_TAG_structure_type              (top-level type for enum)
///         DW_TAG_variant_part              (variant part)
///           DW_AT_discr                    (reference to discriminant DW_TAG_member)
///           DW_TAG_member                  (discriminant member)
///  --->     DW_TAG_variant                 (variant 1)
///  --->     DW_TAG_variant                 (variant 2)
///  --->     DW_TAG_variant                 (variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
/// ```
struct VariantMemberInfo<'a, 'll> {
    variant_index: VariantIdx,
    variant_name: Cow<'a, str>,
    variant_struct_type_di_node: &'ll DIType,
    source_info: Option<(&'ll DIFile, c_uint)>,
}
