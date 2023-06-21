use std::borrow::Cow;

use libc::c_uint;
use rustc_codegen_ssa::{
    debuginfo::{type_names::compute_debuginfo_type_name, wants_c_like_enum_debuginfo},
    traits::ConstMethods,
};

use rustc_index::IndexVec;
use rustc_middle::{
    bug,
    ty::{
        self,
        layout::{LayoutOf, TyAndLayout},
        AdtDef, GeneratorSubsts, Ty,
    },
};
use rustc_target::abi::{Align, Endian, Size, TagEncoding, VariantIdx, Variants};
use smallvec::smallvec;

use crate::{
    common::CodegenCx,
    debuginfo::{
        metadata::{
            build_field_di_node,
            enums::{tag_base_type, DiscrResult},
            file_metadata, size_and_align_of, type_di_node,
            type_map::{self, Stub, UniqueTypeId},
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

// The names of the associated constants in each variant wrapper struct.
// These have to match up with the names being used in `intrinsic.natvis`.
const ASSOC_CONST_DISCR_NAME: &str = "NAME";
const ASSOC_CONST_DISCR_EXACT: &str = "DISCR_EXACT";
const ASSOC_CONST_DISCR_BEGIN: &str = "DISCR_BEGIN";
const ASSOC_CONST_DISCR_END: &str = "DISCR_END";

const ASSOC_CONST_DISCR128_EXACT_LO: &str = "DISCR128_EXACT_LO";
const ASSOC_CONST_DISCR128_EXACT_HI: &str = "DISCR128_EXACT_HI";
const ASSOC_CONST_DISCR128_BEGIN_LO: &str = "DISCR128_BEGIN_LO";
const ASSOC_CONST_DISCR128_BEGIN_HI: &str = "DISCR128_BEGIN_HI";
const ASSOC_CONST_DISCR128_END_LO: &str = "DISCR128_END_LO";
const ASSOC_CONST_DISCR128_END_HI: &str = "DISCR128_END_HI";

// The name of the tag field in the top-level union
const TAG_FIELD_NAME: &str = "tag";
const TAG_FIELD_NAME_128_LO: &str = "tag128_lo";
const TAG_FIELD_NAME_128_HI: &str = "tag128_hi";

// We assign a "virtual" discriminant value to the sole variant of
// a single-variant enum.
const SINGLE_VARIANT_VIRTUAL_DISR: u64 = 0;

/// In CPP-like mode, we generate a union with a field for each variant and an
/// explicit tag field. The field of each variant has a struct type
/// that encodes the discriminant of the variant and it's data layout.
/// The union also has a nested enumeration type that is only used for encoding
/// variant names in an efficient way. Its enumerator values do _not_ correspond
/// to the enum's discriminant values.
/// It's roughly equivalent to the following C/C++ code:
///
/// ```c
/// union enum2$<{fully-qualified-name}> {
///   struct Variant0 {
///     struct {name-of-variant-0} {
///        <variant 0 fields>
///     } value;
///
///     static VariantNames NAME = {name-of-variant-0};
///     static int_type DISCR_EXACT = {discriminant-of-variant-0};
///   } variant0;
///
///   <other variant structs>
///
///   int_type tag;
///
///   enum VariantNames {
///      <name-of-variant-0> = 0, // The numeric values are variant index,
///      <name-of-variant-1> = 1, // not discriminant values.
///      <name-of-variant-2> = 2,
///      ...
///   }
/// }
/// ```
///
/// As you can see, the type name is wrapped in `enum2$<_>`. This way we can
/// have a single NatVis rule for handling all enums. The `2` in `enum2$<_>`
/// is an encoding version tag, so that debuggers can decide to decode this
/// differently than the previous `enum$<_>` encoding emitted by earlier
/// compiler versions.
///
/// Niche-tag enums have one special variant, usually called the
/// "untagged variant". This variant has a field that
/// doubles as the tag of the enum. The variant is active when the value of
/// that field is within a pre-defined range. Therefore the variant struct
/// has a `DISCR_BEGIN` and `DISCR_END` field instead of `DISCR_EXACT` in
/// that case. Both `DISCR_BEGIN` and `DISCR_END` are inclusive bounds.
/// Note that these ranges can wrap around, so that `DISCR_END < DISCR_BEGIN`.
///
/// Single-variant enums don't actually have a tag field. In this case we
/// emit a static tag field (that always has the value 0) so we can use the
/// same representation (and NatVis).
///
/// For niche-layout enums it's possible to have a 128-bit tag. NatVis, VS, and
/// WinDbg (the main targets for CPP-like debuginfo at the moment) don't support
/// 128-bit integers, so all values involved get split into two 64-bit fields.
/// Instead of the `tag` field, we generate two fields `tag128_lo` and `tag128_hi`,
/// Instead of `DISCR_EXACT`, we generate `DISCR128_EXACT_LO` and `DISCR128_EXACT_HI`,
/// and so on.
///
///
/// The following pseudocode shows how to decode an enum value in a debugger:
///
/// ```text
///
/// fn find_active_variant(enum_value) -> (VariantName, VariantValue) {
///     let is_128_bit = enum_value.has_field("tag128_lo");
///
///     if !is_128_bit {
///         // Note: `tag` can be a static field for enums with only one
///         //       inhabited variant.
///         let tag = enum_value.field("tag").value;
///
///         // For each variant, check if it is a match. Only one of them will match,
///         // so if we find it we can return it immediately.
///         for variant_field in enum_value.fields().filter(|f| f.name.starts_with("variant")) {
///             if variant_field.has_field("DISCR_EXACT") {
///                 // This variant corresponds to a single tag value
///                 if variant_field.field("DISCR_EXACT").value == tag {
///                     return (variant_field.field("NAME"), variant_field.value);
///                 }
///             } else {
///                 // This is a range variant
///                 let begin = variant_field.field("DISCR_BEGIN");
///                 let end = variant_field.field("DISCR_END");
///
///                 if is_in_range(tag, begin, end) {
///                     return (variant_field.field("NAME"), variant_field.value);
///                 }
///             }
///         }
///     } else {
///         // Basically the same as with smaller tags, we just have to
///         // stitch the values together.
///         let tag: u128 = (enum_value.field("tag128_lo").value as u128) |
///                         (enum_value.field("tag128_hi").value as u128 << 64);
///
///         for variant_field in enum_value.fields().filter(|f| f.name.starts_with("variant")) {
///             if variant_field.has_field("DISCR128_EXACT_LO") {
///                 let discr_exact = (variant_field.field("DISCR128_EXACT_LO" as u128) |
///                                   (variant_field.field("DISCR128_EXACT_HI") as u128 << 64);
///
///                 // This variant corresponds to a single tag value
///                 if discr_exact.value == tag {
///                     return (variant_field.field("NAME"), variant_field.value);
///                 }
///             } else {
///                 // This is a range variant
///                 let begin = (variant_field.field("DISCR128_BEGIN_LO").value as u128) |
///                             (variant_field.field("DISCR128_BEGIN_HI").value as u128 << 64);
///                 let end = (variant_field.field("DISCR128_END_LO").value as u128) |
///                           (variant_field.field("DISCR128_END_HI").value as u128 << 64);
///
///                 if is_in_range(tag, begin, end) {
///                     return (variant_field.field("NAME"), variant_field.value);
///                 }
///             }
///         }
///     }
///
///     // We should have found an active variant at this point.
///     unreachable!();
/// }
///
/// // Check if a value is within the given range
/// // (where the range might wrap around the value space)
/// fn is_in_range(value, start, end) -> bool {
///     if start < end {
///         value >= start && value <= end
///     } else {
///         value >= start || value <= end
///     }
/// }
///
/// ```
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
                } => build_union_fields_for_enum(
                    cx,
                    enum_adt_def,
                    enum_type_and_layout,
                    enum_type_di_node,
                    variants.indices(),
                    tag_field,
                    None,
                ),
                Variants::Multiple {
                    tag_encoding: TagEncoding::Niche { untagged_variant, .. },
                    ref variants,
                    tag_field,
                    ..
                } => build_union_fields_for_enum(
                    cx,
                    enum_adt_def,
                    enum_type_and_layout,
                    enum_type_di_node,
                    variants.indices(),
                    tag_field,
                    Some(untagged_variant),
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
        enum_type_and_layout,
        enum_type_di_node,
        variant_index,
        enum_adt_def.variant(variant_index),
        variant_layout,
    );

    let tag_base_type = cx.tcx.types.u32;
    let tag_base_type_di_node = type_di_node(cx, tag_base_type);
    let tag_base_type_align = cx.align_of(tag_base_type);

    let variant_names_type_di_node = build_variant_names_type_di_node(
        cx,
        enum_type_di_node,
        std::iter::once((
            variant_index,
            Cow::from(enum_adt_def.variant(variant_index).name.as_str()),
        )),
    );

    let variant_struct_type_wrapper_di_node = build_variant_struct_wrapper_type_di_node(
        cx,
        enum_type_and_layout,
        enum_type_di_node,
        variant_index,
        None,
        variant_struct_type_di_node,
        variant_names_type_di_node,
        tag_base_type_di_node,
        tag_base_type,
        DiscrResult::NoDiscriminant,
    );

    smallvec![
        build_field_di_node(
            cx,
            enum_type_di_node,
            &variant_union_field_name(variant_index),
            // NOTE: We use the size and align of the entire type, not from variant_layout
            //       since the later is sometimes smaller (if it has fewer fields).
            size_and_align_of(enum_type_and_layout),
            Size::ZERO,
            DIFlags::FlagZero,
            variant_struct_type_wrapper_di_node,
        ),
        unsafe {
            llvm::LLVMRustDIBuilderCreateStaticMemberType(
                DIB(cx),
                enum_type_di_node,
                TAG_FIELD_NAME.as_ptr().cast(),
                TAG_FIELD_NAME.len(),
                unknown_file_metadata(cx),
                UNKNOWN_LINE_NUMBER,
                variant_names_type_di_node,
                DIFlags::FlagZero,
                Some(cx.const_u64(SINGLE_VARIANT_VIRTUAL_DISR)),
                tag_base_type_align.bits() as u32,
            )
        }
    ]
}

fn build_union_fields_for_enum<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_adt_def: AdtDef<'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
    enum_type_di_node: &'ll DIType,
    variant_indices: impl Iterator<Item = VariantIdx> + Clone,
    tag_field: usize,
    untagged_variant_index: Option<VariantIdx>,
) -> SmallVec<&'ll DIType> {
    let tag_base_type = super::tag_base_type(cx, enum_type_and_layout);

    let variant_names_type_di_node = build_variant_names_type_di_node(
        cx,
        enum_type_di_node,
        variant_indices.clone().map(|variant_index| {
            let variant_name = Cow::from(enum_adt_def.variant(variant_index).name.as_str());
            (variant_index, variant_name)
        }),
    );

    let variant_field_infos: SmallVec<VariantFieldInfo<'ll>> = variant_indices
        .map(|variant_index| {
            let variant_layout = enum_type_and_layout.for_variant(cx, variant_index);

            let variant_def = enum_adt_def.variant(variant_index);

            let variant_struct_type_di_node = super::build_enum_variant_struct_type_di_node(
                cx,
                enum_type_and_layout,
                enum_type_di_node,
                variant_index,
                variant_def,
                variant_layout,
            );

            VariantFieldInfo {
                variant_index,
                variant_struct_type_di_node,
                source_info: None,
                discr: super::compute_discriminant_value(cx, enum_type_and_layout, variant_index),
            }
        })
        .collect();

    build_union_fields_for_direct_tag_enum_or_generator(
        cx,
        enum_type_and_layout,
        enum_type_di_node,
        &variant_field_infos,
        variant_names_type_di_node,
        tag_base_type,
        tag_field,
        untagged_variant_index,
    )
}

// The base type of the VariantNames DW_AT_enumeration_type is always the same.
// It has nothing to do with the tag of the enum and just has to be big enough
// to hold all variant names.
fn variant_names_enum_base_type<'ll, 'tcx>(cx: &CodegenCx<'ll, 'tcx>) -> Ty<'tcx> {
    cx.tcx.types.u32
}

/// This function builds a DW_AT_enumeration_type that contains an entry for
/// each variant. Note that this has nothing to do with the discriminant. The
/// numeric value of each enumerator corresponds to the variant index. The
/// type is only used for efficiently encoding the name of each variant in
/// debuginfo.
fn build_variant_names_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    containing_scope: &'ll DIType,
    variants: impl Iterator<Item = (VariantIdx, Cow<'tcx, str>)>,
) -> &'ll DIType {
    // Create an enumerator for each variant.
    super::build_enumeration_type_di_node(
        cx,
        "VariantNames",
        variant_names_enum_base_type(cx),
        variants.map(|(variant_index, variant_name)| (variant_name, variant_index.as_u32().into())),
        containing_scope,
    )
}

fn build_variant_struct_wrapper_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_or_generator_type_and_layout: TyAndLayout<'tcx>,
    enum_or_generator_type_di_node: &'ll DIType,
    variant_index: VariantIdx,
    untagged_variant_index: Option<VariantIdx>,
    variant_struct_type_di_node: &'ll DIType,
    variant_names_type_di_node: &'ll DIType,
    tag_base_type_di_node: &'ll DIType,
    tag_base_type: Ty<'tcx>,
    discr: DiscrResult,
) -> &'ll DIType {
    type_map::build_type_with_children(
        cx,
        type_map::stub(
            cx,
            Stub::Struct,
            UniqueTypeId::for_enum_variant_struct_type_wrapper(
                cx.tcx,
                enum_or_generator_type_and_layout.ty,
                variant_index,
            ),
            &variant_struct_wrapper_type_name(variant_index),
            // NOTE: We use size and align of enum_type, not from variant_layout:
            size_and_align_of(enum_or_generator_type_and_layout),
            Some(enum_or_generator_type_di_node),
            DIFlags::FlagZero,
        ),
        |cx, wrapper_struct_type_di_node| {
            enum DiscrKind {
                Exact(u64),
                Exact128(u128),
                Range(u64, u64),
                Range128(u128, u128),
            }

            let (tag_base_type_size, tag_base_type_align) = cx.size_and_align_of(tag_base_type);
            let is_128_bits = tag_base_type_size.bits() > 64;

            let discr = match discr {
                DiscrResult::NoDiscriminant => DiscrKind::Exact(SINGLE_VARIANT_VIRTUAL_DISR),
                DiscrResult::Value(discr_val) => {
                    if is_128_bits {
                        DiscrKind::Exact128(discr_val)
                    } else {
                        debug_assert_eq!(discr_val, discr_val as u64 as u128);
                        DiscrKind::Exact(discr_val as u64)
                    }
                }
                DiscrResult::Range(min, max) => {
                    assert_eq!(Some(variant_index), untagged_variant_index);
                    if is_128_bits {
                        DiscrKind::Range128(min, max)
                    } else {
                        debug_assert_eq!(min, min as u64 as u128);
                        debug_assert_eq!(max, max as u64 as u128);
                        DiscrKind::Range(min as u64, max as u64)
                    }
                }
            };

            let mut fields = SmallVec::new();

            // We always have a field for the value
            fields.push(build_field_di_node(
                cx,
                wrapper_struct_type_di_node,
                "value",
                size_and_align_of(enum_or_generator_type_and_layout),
                Size::ZERO,
                DIFlags::FlagZero,
                variant_struct_type_di_node,
            ));

            let build_assoc_const =
                |name: &str, type_di_node: &'ll DIType, value: u64, align: Align| unsafe {
                    llvm::LLVMRustDIBuilderCreateStaticMemberType(
                        DIB(cx),
                        wrapper_struct_type_di_node,
                        name.as_ptr().cast(),
                        name.len(),
                        unknown_file_metadata(cx),
                        UNKNOWN_LINE_NUMBER,
                        type_di_node,
                        DIFlags::FlagZero,
                        Some(cx.const_u64(value)),
                        align.bits() as u32,
                    )
                };

            // We also always have an associated constant for the discriminant value
            // of the variant.
            fields.push(build_assoc_const(
                ASSOC_CONST_DISCR_NAME,
                variant_names_type_di_node,
                variant_index.as_u32() as u64,
                cx.align_of(variant_names_enum_base_type(cx)),
            ));

            // Emit the discriminant value (or range) corresponding to the variant.
            match discr {
                DiscrKind::Exact(discr_val) => {
                    fields.push(build_assoc_const(
                        ASSOC_CONST_DISCR_EXACT,
                        tag_base_type_di_node,
                        discr_val,
                        tag_base_type_align,
                    ));
                }
                DiscrKind::Exact128(discr_val) => {
                    let align = cx.align_of(cx.tcx.types.u64);
                    let type_di_node = type_di_node(cx, cx.tcx.types.u64);
                    let Split128 { hi, lo } = split_128(discr_val);

                    fields.push(build_assoc_const(
                        ASSOC_CONST_DISCR128_EXACT_LO,
                        type_di_node,
                        lo,
                        align,
                    ));

                    fields.push(build_assoc_const(
                        ASSOC_CONST_DISCR128_EXACT_HI,
                        type_di_node,
                        hi,
                        align,
                    ));
                }
                DiscrKind::Range(begin, end) => {
                    fields.push(build_assoc_const(
                        ASSOC_CONST_DISCR_BEGIN,
                        tag_base_type_di_node,
                        begin,
                        tag_base_type_align,
                    ));

                    fields.push(build_assoc_const(
                        ASSOC_CONST_DISCR_END,
                        tag_base_type_di_node,
                        end,
                        tag_base_type_align,
                    ));
                }
                DiscrKind::Range128(begin, end) => {
                    let align = cx.align_of(cx.tcx.types.u64);
                    let type_di_node = type_di_node(cx, cx.tcx.types.u64);
                    let Split128 { hi: begin_hi, lo: begin_lo } = split_128(begin);
                    let Split128 { hi: end_hi, lo: end_lo } = split_128(end);

                    fields.push(build_assoc_const(
                        ASSOC_CONST_DISCR128_BEGIN_HI,
                        type_di_node,
                        begin_hi,
                        align,
                    ));

                    fields.push(build_assoc_const(
                        ASSOC_CONST_DISCR128_BEGIN_LO,
                        type_di_node,
                        begin_lo,
                        align,
                    ));

                    fields.push(build_assoc_const(
                        ASSOC_CONST_DISCR128_END_HI,
                        type_di_node,
                        end_hi,
                        align,
                    ));

                    fields.push(build_assoc_const(
                        ASSOC_CONST_DISCR128_END_LO,
                        type_di_node,
                        end_lo,
                        align,
                    ));
                }
            }

            fields
        },
        NO_GENERICS,
    )
    .di_node
}

struct Split128 {
    hi: u64,
    lo: u64,
}

fn split_128(value: u128) -> Split128 {
    Split128 { hi: (value >> 64) as u64, lo: value as u64 }
}

fn build_union_fields_for_direct_tag_generator<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    generator_type_and_layout: TyAndLayout<'tcx>,
    generator_type_di_node: &'ll DIType,
) -> SmallVec<&'ll DIType> {
    let Variants::Multiple { tag_encoding: TagEncoding::Direct, tag_field, .. } = generator_type_and_layout.variants else {
        bug!("This function only supports layouts with directly encoded tags.")
    };

    let (generator_def_id, generator_substs) = match generator_type_and_layout.ty.kind() {
        &ty::Generator(def_id, substs, _) => (def_id, substs.as_generator()),
        _ => unreachable!(),
    };

    let generator_layout = cx.tcx.optimized_mir(generator_def_id).generator_layout().unwrap();

    let common_upvar_names = cx.tcx.closure_saved_names_of_captured_variables(generator_def_id);
    let variant_range = generator_substs.variant_range(generator_def_id, cx.tcx);
    let variant_count = (variant_range.start.as_u32()..variant_range.end.as_u32()).len();

    let tag_base_type = tag_base_type(cx, generator_type_and_layout);

    let variant_names_type_di_node = build_variant_names_type_di_node(
        cx,
        generator_type_di_node,
        variant_range
            .clone()
            .map(|variant_index| (variant_index, GeneratorSubsts::variant_name(variant_index))),
    );

    let discriminants: IndexVec<VariantIdx, DiscrResult> = {
        let discriminants_iter = generator_substs.discriminants(generator_def_id, cx.tcx);
        let mut discriminants: IndexVec<VariantIdx, DiscrResult> =
            IndexVec::with_capacity(variant_count);
        for (variant_index, discr) in discriminants_iter {
            // Assert that the index in the IndexMap matches up with the given VariantIdx.
            assert_eq!(variant_index, discriminants.next_index());
            discriminants.push(DiscrResult::Value(discr.val));
        }
        discriminants
    };

    // Build the type node for each field.
    let variant_field_infos: SmallVec<VariantFieldInfo<'ll>> = variant_range
        .map(|variant_index| {
            let variant_struct_type_di_node = super::build_generator_variant_struct_type_di_node(
                cx,
                variant_index,
                generator_type_and_layout,
                generator_type_di_node,
                generator_layout,
                &common_upvar_names,
            );

            let span = generator_layout.variant_source_info[variant_index].span;
            let source_info = if !span.is_dummy() {
                let loc = cx.lookup_debug_loc(span.lo());
                Some((file_metadata(cx, &loc.file), loc.line as c_uint))
            } else {
                None
            };

            VariantFieldInfo {
                variant_index,
                variant_struct_type_di_node,
                source_info,
                discr: discriminants[variant_index],
            }
        })
        .collect();

    build_union_fields_for_direct_tag_enum_or_generator(
        cx,
        generator_type_and_layout,
        generator_type_di_node,
        &variant_field_infos[..],
        variant_names_type_di_node,
        tag_base_type,
        tag_field,
        None,
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
    tag_base_type: Ty<'tcx>,
    tag_field: usize,
    untagged_variant_index: Option<VariantIdx>,
) -> SmallVec<&'ll DIType> {
    let tag_base_type_di_node = type_di_node(cx, tag_base_type);
    let mut unions_fields = SmallVec::with_capacity(variant_field_infos.len() + 1);

    // We create a field in the union for each variant ...
    unions_fields.extend(variant_field_infos.into_iter().map(|variant_member_info| {
        let (file_di_node, line_number) = variant_member_info
            .source_info
            .unwrap_or_else(|| (unknown_file_metadata(cx), UNKNOWN_LINE_NUMBER));

        let field_name = variant_union_field_name(variant_member_info.variant_index);
        let (size, align) = size_and_align_of(enum_type_and_layout);

        let variant_struct_type_wrapper = build_variant_struct_wrapper_type_di_node(
            cx,
            enum_type_and_layout,
            enum_type_di_node,
            variant_member_info.variant_index,
            untagged_variant_index,
            variant_member_info.variant_struct_type_di_node,
            discr_type_di_node,
            tag_base_type_di_node,
            tag_base_type,
            variant_member_info.discr,
        );

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
                variant_struct_type_wrapper,
            )
        }
    }));

    debug_assert_eq!(
        cx.size_and_align_of(enum_type_and_layout.field(cx, tag_field).ty),
        cx.size_and_align_of(super::tag_base_type(cx, enum_type_and_layout))
    );

    // ... and a field for the tag. If the tag is 128 bits wide, this will actually
    // be two 64-bit fields.
    let is_128_bits = cx.size_of(tag_base_type).bits() > 64;

    if is_128_bits {
        let type_di_node = type_di_node(cx, cx.tcx.types.u64);
        let size_and_align = cx.size_and_align_of(cx.tcx.types.u64);

        let (lo_offset, hi_offset) = match cx.tcx.data_layout.endian {
            Endian::Little => (0, 8),
            Endian::Big => (8, 0),
        };

        let tag_field_offset = enum_type_and_layout.fields.offset(tag_field).bytes();
        let lo_offset = Size::from_bytes(tag_field_offset + lo_offset);
        let hi_offset = Size::from_bytes(tag_field_offset + hi_offset);

        unions_fields.push(build_field_di_node(
            cx,
            enum_type_di_node,
            TAG_FIELD_NAME_128_LO,
            size_and_align,
            lo_offset,
            DIFlags::FlagZero,
            type_di_node,
        ));

        unions_fields.push(build_field_di_node(
            cx,
            enum_type_di_node,
            TAG_FIELD_NAME_128_HI,
            size_and_align,
            hi_offset,
            DIFlags::FlagZero,
            type_di_node,
        ));
    } else {
        unions_fields.push(build_field_di_node(
            cx,
            enum_type_di_node,
            TAG_FIELD_NAME,
            cx.size_and_align_of(enum_type_and_layout.field(cx, tag_field).ty),
            enum_type_and_layout.fields.offset(tag_field),
            DIFlags::FlagZero,
            tag_base_type_di_node,
        ));
    }

    unions_fields
}

/// Information about a single field of the top-level DW_TAG_union_type.
struct VariantFieldInfo<'ll> {
    variant_index: VariantIdx,
    variant_struct_type_di_node: &'ll DIType,
    source_info: Option<(&'ll DIFile, c_uint)>,
    discr: DiscrResult,
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

fn variant_struct_wrapper_type_name(variant_index: VariantIdx) -> Cow<'static, str> {
    const PRE_ALLOCATED: [&str; 16] = [
        "Variant0",
        "Variant1",
        "Variant2",
        "Variant3",
        "Variant4",
        "Variant5",
        "Variant6",
        "Variant7",
        "Variant8",
        "Variant9",
        "Variant10",
        "Variant11",
        "Variant12",
        "Variant13",
        "Variant14",
        "Variant15",
    ];

    PRE_ALLOCATED
        .get(variant_index.as_usize())
        .map(|&s| Cow::from(s))
        .unwrap_or_else(|| format!("Variant{}", variant_index.as_usize()).into())
}
