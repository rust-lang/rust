use std::borrow::Cow;

use libc::c_uint;
use rustc_codegen_ssa::debuginfo::{
    type_names::compute_debuginfo_type_name, wants_c_like_enum_debuginfo,
};
use rustc_middle::{
    bug,
    ty::{
        self,
        layout::{LayoutOf, TyAndLayout},
        util::Discr,
        AdtDef, GeneratorSubsts,
    },
};
use rustc_target::abi::{Size, TagEncoding, VariantIdx, Variants};
use smallvec::smallvec;

use crate::{
    common::CodegenCx,
    debuginfo::{
        metadata::{
            build_field_di_node, closure_saved_names_of_captured_variables,
            enums::tag_base_type,
            file_metadata, generator_layout_and_saved_local_names, size_and_align_of,
            type_map::{self, UniqueTypeId},
            unknown_file_metadata, DINodeCreationResult, SmallVec, NO_GENERICS, NO_SCOPE_METADATA,
            UNKNOWN_LINE_NUMBER,
        },
        utils::DIB,
    },
    llvm::{
        self,
        debuginfo::{DIFile, DIFlags, DIType},
    },
};

/// In CPP-like mode, we generate a union of structs for each variant and an
/// explicit discriminant field roughly equivalent to the following C/C++ code:
///
/// ```c
/// union enum$<{fully-qualified-name}> {
///   struct {variant 0 name} {
///     <variant 0 fields>
///   } variant0;
///   <other variant structs>
///   {name} discriminant;
/// }
/// ```
///
/// As you can see, the type name is wrapped `enum$`. This way we can have a
/// single NatVis rule for handling all enums.
///
/// At the LLVM IR level this looks like
///
/// ```txt
///       DW_TAG_union_type              (top-level type for enum)
///         DW_TAG_member                    (member for variant 1)
///         DW_TAG_member                    (member for variant 2)
///         DW_TAG_member                    (member for variant 3)
///         DW_TAG_structure_type            (type of variant 1)
///         DW_TAG_structure_type            (type of variant 2)
///         DW_TAG_structure_type            (type of variant 3)
///         DW_TAG_enumeration_type          (type of tag)
/// ```
///
/// The above encoding applies for enums with a direct tag. For niche-tag we have to do things
/// differently in order to allow a NatVis visualizer to extract all the information needed:
/// We generate a union of two fields, one for the dataful variant
/// and one that just points to the discriminant (which is some field within the dataful variant).
/// We also create a DW_TAG_enumeration_type DIE that contains tag values for the non-dataful
/// variants and make the discriminant field that type. We then use NatVis to render the enum type
/// correctly in Windbg/VS. This will generate debuginfo roughly equivalent to the following C:
///
/// ```c
/// union enum$<{name}, {min niche}, {max niche}, {dataful variant name}> {
///   struct <dataful variant name> {
///     <fields in dataful variant>
///   } dataful_variant;
///   enum Discriminant$ {
///     <non-dataful variants>
///   } discriminant;
/// }
/// ```
///
/// The NatVis in `intrinsic.natvis` matches on the type name `enum$<*, *, *, *>`
/// and evaluates `this.discriminant`. If the value is between the min niche and max
/// niche, then the enum is in the dataful variant and `this.dataful_variant` is
/// rendered. Otherwise, the enum is in one of the non-dataful variants. In that
/// case, we just need to render the name of the `this.discriminant` enum.
pub(super) fn build_enum_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    let enum_type = unique_type_id.expect_ty();
    let &ty::Adt(enum_adt_def, _) = enum_type.kind() else {
        bug!("build_enum_type_di_node() called with non-enum type: `{:?}`", enum_type)
        };

    let enum_type_and_layout = cx.layout_of(enum_type);
    let enum_type_name = compute_debuginfo_type_name(cx.tcx, enum_type, false);

    debug_assert!(!wants_c_like_enum_debuginfo(enum_type_and_layout));

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            type_map::Stub::Union,
            unique_type_id,
            &enum_type_name,
            cx.size_and_align_of(enum_type),
            NO_SCOPE_METADATA,
            DIFlags::FlagZero,
        ),
        |cx, enum_type_di_node| {
            match enum_type_and_layout.variants {
                Variants::Single { index: variant_index } => {
                    if enum_adt_def.variants().is_empty() {
                        // Uninhabited enums have Variants::Single. We don't generate
                        // any members for them.
                        return smallvec![];
                    }

                    build_single_variant_union_fields(
                        cx,
                        enum_adt_def,
                        enum_type_and_layout,
                        enum_type_di_node,
                        variant_index,
                    )
                }
                Variants::Multiple {
                    tag_encoding: TagEncoding::Direct,
                    ref variants,
                    tag_field,
                    ..
                } => build_union_fields_for_direct_tag_enum(
                    cx,
                    enum_adt_def,
                    enum_type_and_layout,
                    enum_type_di_node,
                    &mut variants.indices(),
                    tag_field,
                ),
                Variants::Multiple {
                    tag_encoding: TagEncoding::Niche { dataful_variant, .. },
                    ref variants,
                    tag_field,
                    ..
                } => build_union_fields_for_niche_tag_enum(
                    cx,
                    enum_adt_def,
                    enum_type_and_layout,
                    enum_type_di_node,
                    dataful_variant,
                    &mut variants.indices(),
                    tag_field,
                ),
            }
        },
        NO_GENERICS,
    )
}

/// A generator debuginfo node looks the same as a that of an enum type.
///
/// See [build_enum_type_di_node] for more information.
pub(super) fn build_generator_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> DINodeCreationResult<'ll> {
    let generator_type = unique_type_id.expect_ty();
    let generator_type_and_layout = cx.layout_of(generator_type);
    let generator_type_name = compute_debuginfo_type_name(cx.tcx, generator_type, false);

    debug_assert!(!wants_c_like_enum_debuginfo(generator_type_and_layout));

    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            type_map::Stub::Union,
            unique_type_id,
            &generator_type_name,
            size_and_align_of(generator_type_and_layout),
            NO_SCOPE_METADATA,
            DIFlags::FlagZero,
        ),
        |cx, generator_type_di_node| match generator_type_and_layout.variants {
            Variants::Multiple { tag_encoding: TagEncoding::Direct, .. } => {
                build_union_fields_for_direct_tag_generator(
                    cx,
                    generator_type_and_layout,
                    generator_type_di_node,
                )
            }
            Variants::Single { .. }
            | Variants::Multiple { tag_encoding: TagEncoding::Niche { .. }, .. } => {
                bug!(
                    "Encountered generator with non-direct-tag layout: {:?}",
                    generator_type_and_layout
                )
            }
        },
        NO_GENERICS,
    )
}

fn build_single_variant_union_fields<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_adt_def: AdtDef<'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    enum_type_di_node: &'ll DIType,
    variant_index: VariantIdx,
) -> SmallVec<&'ll DIType> {
    let variant_layout = enum_type_and_layout.for_variant(cx, variant_index);
    let variant_struct_type_di_node = super::build_enum_variant_struct_type_di_node(
        cx,
        enum_type_and_layout.ty,
        enum_type_di_node,
        variant_index,
        enum_adt_def.variant(variant_index),
        variant_layout,
    );

    // NOTE: The field name of the union is the same as the variant name, not "variant0".
    let variant_name = enum_adt_def.variant(variant_index).name.as_str();

    smallvec![build_field_di_node(
        cx,
        enum_type_di_node,
        variant_name,
        // NOTE: We use the size and align of the entire type, not from variant_layout
        //       since the later is sometimes smaller (if it has fewer fields).
        size_and_align_of(enum_type_and_layout),
        Size::ZERO,
        DIFlags::FlagZero,
        variant_struct_type_di_node,
    )]
}

fn build_union_fields_for_direct_tag_enum<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_adt_def: AdtDef<'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    enum_type_di_node: &'ll DIType,
    variant_indices: &mut dyn Iterator<Item = VariantIdx>,
    tag_field: usize,
) -> SmallVec<&'ll DIType> {
    let variant_field_infos: SmallVec<VariantFieldInfo<'ll>> = variant_indices
        .map(|variant_index| {
            let variant_layout = enum_type_and_layout.for_variant(cx, variant_index);

            VariantFieldInfo {
                variant_index,
                variant_struct_type_di_node: super::build_enum_variant_struct_type_di_node(
                    cx,
                    enum_type_and_layout.ty,
                    enum_type_di_node,
                    variant_index,
                    enum_adt_def.variant(variant_index),
                    variant_layout,
                ),
                source_info: None,
            }
        })
        .collect();

    let discr_type_name = cx.tcx.item_name(enum_adt_def.did());
    let tag_base_type = super::tag_base_type(cx, enum_type_and_layout);
    let discr_type_di_node = super::build_enumeration_type_di_node(
        cx,
        discr_type_name.as_str(),
        tag_base_type,
        &mut enum_adt_def.discriminants(cx.tcx).map(|(variant_index, discr)| {
            (discr, Cow::from(enum_adt_def.variant(variant_index).name.as_str()))
        }),
        enum_type_di_node,
    );

    build_union_fields_for_direct_tag_enum_or_generator(
        cx,
        enum_type_and_layout,
        enum_type_di_node,
        &variant_field_infos,
        discr_type_di_node,
        tag_field,
    )
}

fn build_union_fields_for_niche_tag_enum<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_adt_def: AdtDef<'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    enum_type_di_node: &'ll DIType,
    dataful_variant_index: VariantIdx,
    variant_indices: &mut dyn Iterator<Item = VariantIdx>,
    tag_field: usize,
) -> SmallVec<&'ll DIType> {
    let dataful_variant_struct_type_di_node = super::build_enum_variant_struct_type_di_node(
        cx,
        enum_type_and_layout.ty,
        enum_type_di_node,
        dataful_variant_index,
        &enum_adt_def.variant(dataful_variant_index),
        enum_type_and_layout.for_variant(cx, dataful_variant_index),
    );

    let tag_base_type = super::tag_base_type(cx, enum_type_and_layout);
    // Create an DW_TAG_enumerator for each variant except the dataful one.
    let discr_type_di_node = super::build_enumeration_type_di_node(
        cx,
        "Discriminant$",
        tag_base_type,
        &mut variant_indices.filter_map(|variant_index| {
            if let Some(discr_val) =
                super::compute_discriminant_value(cx, enum_type_and_layout, variant_index)
            {
                let discr = Discr { val: discr_val as u128, ty: tag_base_type };
                let variant_name = Cow::from(enum_adt_def.variant(variant_index).name.as_str());
                Some((discr, variant_name))
            } else {
                debug_assert_eq!(variant_index, dataful_variant_index);
                None
            }
        }),
        enum_type_di_node,
    );

    smallvec![
        build_field_di_node(
            cx,
            enum_type_di_node,
            "dataful_variant",
            size_and_align_of(enum_type_and_layout),
            Size::ZERO,
            DIFlags::FlagZero,
            dataful_variant_struct_type_di_node,
        ),
        build_field_di_node(
            cx,
            enum_type_di_node,
            "discriminant",
            cx.size_and_align_of(tag_base_type),
            enum_type_and_layout.fields.offset(tag_field),
            DIFlags::FlagZero,
            discr_type_di_node,
        ),
    ]
}

fn build_union_fields_for_direct_tag_generator<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    generator_type_and_layout: TyAndLayout<'tcx>,
    generator_type_di_node: &'ll DIType,
) -> SmallVec<&'ll DIType> {
    let Variants::Multiple { tag_encoding: TagEncoding::Direct, tag_field, .. } = generator_type_and_layout.variants else {
        bug!("This function only supports layouts with direcly encoded tags.")
    };

    let (generator_def_id, generator_substs) = match generator_type_and_layout.ty.kind() {
        &ty::Generator(def_id, substs, _) => (def_id, substs.as_generator()),
        _ => unreachable!(),
    };

    let (generator_layout, state_specific_upvar_names) =
        generator_layout_and_saved_local_names(cx.tcx, generator_def_id);

    let common_upvar_names = closure_saved_names_of_captured_variables(cx.tcx, generator_def_id);
    let variant_range = generator_substs.variant_range(generator_def_id, cx.tcx);

    // Build the type node for each field.
    let variant_field_infos: SmallVec<VariantFieldInfo<'ll>> = variant_range
        .clone()
        .map(|variant_index| {
            let variant_struct_type_di_node = super::build_generator_variant_struct_type_di_node(
                cx,
                variant_index,
                generator_type_and_layout,
                generator_type_di_node,
                generator_layout,
                &state_specific_upvar_names,
                &common_upvar_names,
            );

            let span = generator_layout.variant_source_info[variant_index].span;
            let source_info = if !span.is_dummy() {
                let loc = cx.lookup_debug_loc(span.lo());
                Some((file_metadata(cx, &loc.file), loc.line as c_uint))
            } else {
                None
            };

            VariantFieldInfo { variant_index, variant_struct_type_di_node, source_info }
        })
        .collect();

    let tag_base_type = tag_base_type(cx, generator_type_and_layout);
    let discr_type_name = "Discriminant$";
    let discr_type_di_node = super::build_enumeration_type_di_node(
        cx,
        discr_type_name,
        tag_base_type,
        &mut generator_substs
            .discriminants(generator_def_id, cx.tcx)
            .map(|(variant_index, discr)| (discr, GeneratorSubsts::variant_name(variant_index))),
        generator_type_di_node,
    );

    build_union_fields_for_direct_tag_enum_or_generator(
        cx,
        generator_type_and_layout,
        generator_type_di_node,
        &variant_field_infos[..],
        discr_type_di_node,
        tag_field,
    )
}

/// This is a helper function shared between enums and generators that makes sure fields have the
/// expect names.
fn build_union_fields_for_direct_tag_enum_or_generator<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    enum_type_di_node: &'ll DIType,
    variant_field_infos: &[VariantFieldInfo<'ll>],
    discr_type_di_node: &'ll DIType,
    tag_field: usize,
) -> SmallVec<&'ll DIType> {
    let mut unions_fields = SmallVec::with_capacity(variant_field_infos.len() + 1);

    // We create a field in the union for each variant ...
    unions_fields.extend(variant_field_infos.into_iter().map(|variant_member_info| {
        let (file_di_node, line_number) = variant_member_info
            .source_info
            .unwrap_or_else(|| (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER));

        let field_name = variant_union_field_name(variant_member_info.variant_index);
        let (size, align) = size_and_align_of(enum_type_and_layout);

        // We use LLVMRustDIBuilderCreateMemberType() member type directly because
        // the build_field_di_node() function does not support specifying a source location,
        // which is something that we don't do anywhere else.
        unsafe {
            llvm::LLVMRustDIBuilderCreateMemberType(
                DIB(cx),
                enum_type_di_node,
                field_name.as_ptr().cast(),
                field_name.len(),
                file_di_node,
                line_number,
                // NOTE: We use the size and align of the entire type, not from variant_layout
                //       since the later is sometimes smaller (if it has fewer fields).
                size.bits(),
                align.bits() as u32,
                // Union fields are always at offset zero
                Size::ZERO.bits(),
                DIFlags::FlagZero,
                variant_member_info.variant_struct_type_di_node,
            )
        }
    }));

    debug_assert_eq!(
        cx.size_and_align_of(enum_type_and_layout.field(cx, tag_field).ty),
        cx.size_and_align_of(super::tag_base_type(cx, enum_type_and_layout))
    );

    // ... and a field for the discriminant.
    unions_fields.push(build_field_di_node(
        cx,
        enum_type_di_node,
        "discriminant",
        cx.size_and_align_of(enum_type_and_layout.field(cx, tag_field).ty),
        enum_type_and_layout.fields.offset(tag_field),
        DIFlags::FlagZero,
        discr_type_di_node,
    ));

    unions_fields
}

/// Information about a single field of the top-level DW_TAG_union_type.
struct VariantFieldInfo<'ll> {
    variant_index: VariantIdx,
    variant_struct_type_di_node: &'ll DIType,
    source_info: Option<(&'ll DIFile, c_uint)>,
}

fn variant_union_field_name(variant_index: VariantIdx) -> Cow<'static, str> {
    const PRE_ALLOCATED: [&str; 16] = [
        "variant0",
        "variant1",
        "variant2",
        "variant3",
        "variant4",
        "variant5",
        "variant6",
        "variant7",
        "variant8",
        "variant9",
        "variant10",
        "variant11",
        "variant12",
        "variant13",
        "variant14",
        "variant15",
    ];

    PRE_ALLOCATED
        .get(variant_index.as_usize())
        .map(|&s| Cow::from(s))
        .unwrap_or_else(|| format!("variant{}", variant_index.as_usize()).into())
}
