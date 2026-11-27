//! Builds type debuginfo from descriptions shared by all codegen units.
//!
//! The descriptions are computed once per type (see `rustc_middle::ty::debuginfo_type_desc`),
//! and each codegen unit only turns them into LLVM nodes. The translation below mirrors the
//! order of LLVM calls of the per-CGU builders in the parent module, so that the produced
//! metadata is identical.

use std::borrow::Cow;
use std::iter;
use std::sync::LazyLock;

use rustc_abi::{Align, FieldIdx, Size, VariantIdx, Variants};
use rustc_codegen_ssa::debuginfo::type_names::{compute_debuginfo_type_name, cpp_like_debuginfo};
use rustc_codegen_ssa::debuginfo::{tag_base_type, wants_c_like_enum_debuginfo};
use rustc_codegen_ssa::traits::*;
use rustc_hir::def::CtorKind;
use rustc_middle::ty::debuginfo_type_desc::{
    DebuginfoFieldName, DebuginfoGenericParamDesc, DebuginfoMemberDesc, DebuginfoStubDesc,
    DebuginfoTagDesc, DebuginfoTypeDesc, DebuginfoVariantDesc,
};
use rustc_middle::ty::layout::{
    HasTypingEnv, LayoutOf, TyAndLayout, WIDE_PTR_ADDR, WIDE_PTR_EXTRA,
};
use rustc_middle::ty::{self, AdtKind, Ty, Unnormalized};
use rustc_span::{Span, Symbol};
use smallvec::smallvec;

use super::type_map::{self, DINodeCreationResult, Stub, StubInfo, UniqueTypeId};
use super::{
    NO_GENERICS, SmallVec, UNKNOWN_LINE_NUMBER, create_pointer_type, enums, size_and_align_of,
    spanned_type_di_node, tuple_field_name, type_di_node, unknown_file_metadata,
    visibility_di_flags,
};
use crate::common::CodegenCx;
use crate::debuginfo::utils::{
    DIB, WidePtrKind, create_DIArray, get_namespace_for_item, wide_pointer_kind,
};
use crate::llvm;
use crate::llvm::debuginfo::{DIFlags, DIScope, DIType};

fn enabled(cx: &CodegenCx<'_, '_>) -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("RUSTC_DEBUGINFO_NO_SHARE").is_none());
    *ENABLED && !cx.sess().opts.unstable_opts.debug_info_type_line_numbers
}

/// Returns `None` if the type is not handled here, in which case the caller falls back to the
/// per-CGU builders.
pub(super) fn try_build_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    t: Ty<'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
    span: Span,
) -> Option<DINodeCreationResult<'ll>> {
    if !enabled(cx) {
        return None;
    }
    let supported = match *t.kind() {
        ty::RawPtr(..) | ty::Ref(..) | ty::Closure(..) | ty::CoroutineClosure(..) => true,
        ty::Tuple(elements) => !elements.is_empty(),
        ty::Adt(def, _) => match def.adt_kind() {
            AdtKind::Struct => !t.is_scalable_vector(),
            AdtKind::Enum => !cpp_like_debuginfo(cx.tcx),
            AdtKind::Union => false,
        },
        _ => false,
    };
    if !supported {
        return None;
    }

    let desc = cx.tcx.debuginfo_type_descs.get_or_compute(t, || compute_desc(cx, t, span));
    let result = match &*desc {
        DebuginfoTypeDesc::Unsupported => return None,
        DebuginfoTypeDesc::Struct { stub, members, generics } => {
            build_struct(cx, unique_type_id, stub, members, generics)
        }
        DebuginfoTypeDesc::ThinPointer { pointee, name, size, align } => {
            let pointee_di_node = spanned_type_di_node(cx, *pointee, span);
            if let Some(existing) = di_node_created_in_meantime(cx, unique_type_id) {
                return Some(existing);
            }
            let di_node = create_pointer_type(cx, pointee_di_node, *size, *align, name);
            DINodeCreationResult::new(di_node, false)
        }
        DebuginfoTypeDesc::WidePointer { pointee, stub, addr, extra } => {
            let pointee_di_node = spanned_type_di_node(cx, *pointee, span);
            if let Some(existing) = di_node_created_in_meantime(cx, unique_type_id) {
                return Some(existing);
            }
            type_map::build_type_with_children(
                cx,
                create_stub(cx, Stub::Struct, unique_type_id, stub, None),
                |cx, owner| {
                    let data_ptr_di_node =
                        create_pointer_type(cx, pointee_di_node, addr.size, addr.align, "");
                    let addr_di_node = create_member(cx, owner, addr, data_ptr_di_node);
                    let extra_ty_di_node = type_di_node(cx, extra.ty);
                    smallvec![addr_di_node, create_member(cx, owner, extra, extra_ty_di_node)]
                },
                NO_GENERICS,
            )
        }
        DebuginfoTypeDesc::Enum { stub, variants, tag, variant_part_unique_id, generics } => {
            build_enum(cx, t, unique_type_id, stub, variants, tag, variant_part_unique_id, generics)
        }
    };
    Some(result)
}

fn di_node_created_in_meantime<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
) -> Option<DINodeCreationResult<'ll>> {
    super::debug_context(cx)
        .type_map
        .di_node_for_unique_id(unique_type_id)
        .map(|di_node| DINodeCreationResult::new(di_node, true))
}

fn flags(bits: u32) -> DIFlags {
    DIFlags::from_bits_retain(bits)
}

fn create_stub<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    kind: Stub<'ll>,
    unique_type_id: UniqueTypeId<'tcx>,
    stub: &DebuginfoStubDesc,
    containing_scope: Option<&'ll DIScope>,
) -> StubInfo<'ll, 'tcx> {
    type_map::stub_with_id_str(
        cx,
        kind,
        unique_type_id,
        &stub.unique_id,
        &stub.name,
        None,
        (stub.size, stub.align),
        containing_scope,
        flags(stub.flags),
    )
}

fn create_member<'ll>(
    cx: &CodegenCx<'ll, '_>,
    owner: &'ll DIScope,
    member: &DebuginfoMemberDesc<'_>,
    type_di_node: &'ll DIType,
) -> &'ll DIType {
    create_member_raw(
        cx,
        owner,
        member.name.as_str(),
        (member.size, member.align),
        member.offset,
        flags(member.flags),
        type_di_node,
    )
}

fn create_member_raw<'ll>(
    cx: &CodegenCx<'ll, '_>,
    owner: &'ll DIScope,
    name: &str,
    (size, align): (Size, Align),
    offset: Size,
    flags: DIFlags,
    type_di_node: &'ll DIType,
) -> &'ll DIType {
    let file = unknown_file_metadata(cx);
    unsafe {
        llvm::LLVMDIBuilderCreateMemberType(
            DIB(cx),
            owner,
            name.as_ptr(),
            name.len(),
            file,
            UNKNOWN_LINE_NUMBER,
            size.bits(),
            align.bits() as u32,
            offset.bits(),
            flags,
            type_di_node,
        )
    }
}

fn build_members<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    owner: &'ll DIScope,
    members: &[DebuginfoMemberDesc<'tcx>],
) -> SmallVec<&'ll DIType> {
    members
        .iter()
        .map(|member| {
            let type_di_node = type_di_node(cx, member.ty);
            create_member(cx, owner, member, type_di_node)
        })
        .collect()
}

fn build_generics<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    generics: &[DebuginfoGenericParamDesc<'tcx>],
) -> SmallVec<Option<&'ll DIType>> {
    generics
        .iter()
        .map(|param| {
            let actual_type_di_node = type_di_node(cx, param.ty);
            Some(cx.create_template_type_parameter(param.name.as_str(), actual_type_di_node))
        })
        .collect()
}

fn build_struct<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
    stub: &DebuginfoStubDesc,
    members: &[DebuginfoMemberDesc<'tcx>],
    generics: &[DebuginfoGenericParamDesc<'tcx>],
) -> DINodeCreationResult<'ll> {
    let containing_scope = stub.scope_item.map(|def_id| get_namespace_for_item(cx, def_id));
    type_map::build_type_with_children(
        cx,
        create_stub(cx, Stub::Struct, unique_type_id, stub, containing_scope),
        |cx, owner| build_members(cx, owner, members),
        |cx| build_generics(cx, generics),
    )
}

fn build_enum<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    enum_type: Ty<'tcx>,
    unique_type_id: UniqueTypeId<'tcx>,
    stub: &DebuginfoStubDesc,
    variants: &[DebuginfoVariantDesc<'tcx>],
    tag: &Option<DebuginfoTagDesc<'tcx>>,
    variant_part_unique_id: &str,
    generics: &[DebuginfoGenericParamDesc<'tcx>],
) -> DINodeCreationResult<'ll> {
    let containing_scope = stub.scope_item.map(|def_id| get_namespace_for_item(cx, def_id));
    let enum_size_and_align = (stub.size, stub.align);
    type_map::build_type_with_children(
        cx,
        create_stub(cx, Stub::Struct, unique_type_id, stub, containing_scope),
        |cx, enum_type_di_node| {
            let variant_struct_type_di_nodes: SmallVec<_> = variants
                .iter()
                .enumerate()
                .map(|(variant_index, variant)| {
                    let variant_unique_type_id = UniqueTypeId::for_enum_variant_struct_type(
                        cx.tcx,
                        enum_type,
                        VariantIdx::from_usize(variant_index),
                    );
                    type_map::build_type_with_children(
                        cx,
                        type_map::stub_with_id_str(
                            cx,
                            Stub::Struct,
                            variant_unique_type_id,
                            &variant.struct_unique_id,
                            variant.name.as_str(),
                            None,
                            enum_size_and_align,
                            Some(enum_type_di_node),
                            flags(stub.flags),
                        ),
                        |cx, owner| build_members(cx, owner, &variant.members),
                        |cx| build_generics(cx, generics),
                    )
                    .di_node
                })
                .collect();

            let tag_member_di_node = tag.as_ref().map(|tag| {
                let tag_type_di_node = type_di_node(cx, tag.ty);
                create_member_raw(
                    cx,
                    enum_type_di_node,
                    "",
                    (tag.size, tag.align),
                    tag.offset,
                    DIFlags::FlagArtificial,
                    tag_type_di_node,
                )
            });

            let variant_part_unique_type_id =
                UniqueTypeId::for_enum_variant_part(cx.tcx, enum_type);
            let file_metadata = unknown_file_metadata(cx);
            let variant_part_stub = StubInfo::new_with_id_str(
                cx,
                variant_part_unique_type_id,
                variant_part_unique_id,
                |cx, variant_part_unique_type_id_str| unsafe {
                    let variant_part_name = "";
                    llvm::LLVMRustDIBuilderCreateVariantPart(
                        DIB(cx),
                        enum_type_di_node,
                        variant_part_name.as_ptr(),
                        variant_part_name.len(),
                        file_metadata,
                        UNKNOWN_LINE_NUMBER,
                        enum_size_and_align.0.bits(),
                        enum_size_and_align.1.bits() as u32,
                        DIFlags::FlagZero,
                        tag_member_di_node,
                        create_DIArray(DIB(cx), &[]),
                        variant_part_unique_type_id_str.as_ptr(),
                        variant_part_unique_type_id_str.len(),
                    )
                },
            );

            smallvec![
                type_map::build_type_with_children(
                    cx,
                    variant_part_stub,
                    |cx, variant_part_di_node| {
                        iter::zip(variants, variant_struct_type_di_nodes)
                            .map(|(variant, variant_struct_type_di_node)| {
                                let file_di_node = unknown_file_metadata(cx);
                                let discr = variant.discr.map(|value| {
                                    let tag_size = tag.as_ref().unwrap().size;
                                    cx.const_uint_big(cx.type_ix(tag_size.bits()), value)
                                });
                                let variant_name = variant.name.as_str();
                                unsafe {
                                    llvm::LLVMRustDIBuilderCreateVariantMemberType(
                                        DIB(cx),
                                        variant_part_di_node,
                                        variant_name.as_ptr(),
                                        variant_name.len(),
                                        file_di_node,
                                        UNKNOWN_LINE_NUMBER,
                                        enum_size_and_align.0.bits(),
                                        enum_size_and_align.1.bits() as u32,
                                        Size::ZERO.bits(),
                                        discr,
                                        DIFlags::FlagZero,
                                        variant_struct_type_di_node,
                                    )
                                }
                            })
                            .collect()
                    },
                    NO_GENERICS,
                )
                .di_node
            ]
        },
        NO_GENERICS,
    )
}

//=-----------------------------------------------------------------------------
// Description computation. Nothing below may create LLVM nodes.
//=-----------------------------------------------------------------------------

fn compute_desc<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    t: Ty<'tcx>,
    span: Span,
) -> DebuginfoTypeDesc<'tcx> {
    match *t.kind() {
        ty::RawPtr(pointee_type, _) | ty::Ref(_, pointee_type, _) => {
            compute_pointer_desc(cx, t, pointee_type)
        }
        // Same condition as in `spanned_type_di_node`.
        ty::Adt(def, args)
            if def.is_box()
                && args.get(1).is_none_or(|arg| cx.layout_of(arg.expect_ty()).is_1zst()) =>
        {
            compute_pointer_desc(cx, t, t.expect_boxed_ty())
        }
        ty::Closure(def_id, _) | ty::CoroutineClosure(def_id, _) => {
            compute_closure_env_desc(cx, t, def_id)
        }
        ty::Tuple(_) => compute_tuple_desc(cx, t),
        ty::Adt(def, _) => match def.adt_kind() {
            AdtKind::Struct => compute_struct_desc(cx, t, span),
            AdtKind::Enum => compute_enum_desc(cx, t, span),
            AdtKind::Union => DebuginfoTypeDesc::Unsupported,
        },
        _ => DebuginfoTypeDesc::Unsupported,
    }
}

fn unique_id_str<'tcx>(cx: &CodegenCx<'_, 'tcx>, t: Ty<'tcx>) -> String {
    UniqueTypeId::for_ty(cx.tcx, t).generate_unique_id_string(cx.tcx)
}

fn field_name(name: Cow<'static, str>) -> DebuginfoFieldName {
    match name {
        Cow::Borrowed(s) => DebuginfoFieldName::Static(s),
        Cow::Owned(s) => DebuginfoFieldName::Owned(s),
    }
}

fn member_desc<'tcx>(
    name: DebuginfoFieldName,
    layout: TyAndLayout<'tcx>,
    offset: Size,
    flags: DIFlags,
) -> DebuginfoMemberDesc<'tcx> {
    let (size, align) = size_and_align_of(layout);
    DebuginfoMemberDesc { name, size, align, offset, flags: flags.bits(), ty: layout.ty }
}

fn compute_pointer_desc<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    ptr_type: Ty<'tcx>,
    pointee_type: Ty<'tcx>,
) -> DebuginfoTypeDesc<'tcx> {
    assert_eq!(
        cx.size_and_align_of(ptr_type),
        cx.size_and_align_of(Ty::new_mut_ptr(cx.tcx, pointee_type))
    );

    let pointee = match pointee_type.kind() {
        ty::Slice(element_type) => *element_type,
        ty::Str => cx.tcx.types.u8,
        _ => pointee_type,
    };

    let data_layout = &cx.tcx.data_layout;
    let pointer_size = data_layout.pointer_size();
    let pointer_align = data_layout.pointer_align();
    let name = compute_debuginfo_type_name(cx.tcx, ptr_type, true);

    match wide_pointer_kind(cx, pointee_type) {
        None => {
            assert_eq!(
                (pointer_size, pointer_align),
                cx.size_and_align_of(ptr_type),
                "ptr_type={ptr_type}, pointee_type={pointee_type}",
            );
            DebuginfoTypeDesc::ThinPointer {
                pointee,
                name,
                size: pointer_size,
                align: pointer_align,
            }
        }
        Some(wide_pointer_kind) => {
            let (size, align) = cx.size_and_align_of(ptr_type);
            let stub = DebuginfoStubDesc {
                name,
                unique_id: unique_id_str(cx, ptr_type),
                size,
                align,
                scope_item: None,
                flags: DIFlags::FlagZero.bits(),
            };

            let layout_type =
                if ptr_type.is_box() { Ty::new_mut_ptr(cx.tcx, pointee_type) } else { ptr_type };
            let layout = cx.layout_of(layout_type);
            let addr_field = layout.field(cx, WIDE_PTR_ADDR);
            let extra_field = layout.field(cx, WIDE_PTR_EXTRA);
            let (addr_field_name, extra_field_name) = match wide_pointer_kind {
                WidePtrKind::Dyn => ("pointer", "vtable"),
                WidePtrKind::Slice => ("data_ptr", "length"),
            };

            DebuginfoTypeDesc::WidePointer {
                pointee,
                stub,
                addr: member_desc(
                    DebuginfoFieldName::Static(addr_field_name),
                    addr_field,
                    layout.fields.offset(WIDE_PTR_ADDR),
                    DIFlags::FlagZero,
                ),
                extra: member_desc(
                    DebuginfoFieldName::Static(extra_field_name),
                    extra_field,
                    layout.fields.offset(WIDE_PTR_EXTRA),
                    DIFlags::FlagZero,
                ),
            }
        }
    }
}

fn compute_struct_desc<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    struct_type: Ty<'tcx>,
    span: Span,
) -> DebuginfoTypeDesc<'tcx> {
    let ty::Adt(adt_def, _) = struct_type.kind() else { unreachable!() };
    let layout = cx.spanned_layout_of(struct_type, span);
    let variant_def = adt_def.non_enum_variant();
    let (size, align) = size_and_align_of(layout);
    let stub = DebuginfoStubDesc {
        name: compute_debuginfo_type_name(cx.tcx, struct_type, false),
        unique_id: unique_id_str(cx, struct_type),
        size,
        align,
        scope_item: Some(adt_def.did()),
        flags: visibility_di_flags(cx, adt_def.did(), adt_def.did()).bits(),
    };
    let members = variant_def
        .fields
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let name = if variant_def.ctor_kind() == Some(CtorKind::Fn) {
                field_name(tuple_field_name(i))
            } else {
                DebuginfoFieldName::Symbol(f.name)
            };
            member_desc(
                name,
                layout.field(cx, i),
                layout.fields.offset(i),
                visibility_di_flags(cx, f.did, adt_def.did()),
            )
        })
        .collect();
    DebuginfoTypeDesc::Struct { stub, members, generics: compute_generics(cx, struct_type) }
}

fn compute_tuple_desc<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    tuple_type: Ty<'tcx>,
) -> DebuginfoTypeDesc<'tcx> {
    let &ty::Tuple(component_types) = tuple_type.kind() else { unreachable!() };
    let layout = cx.layout_of(tuple_type);
    let (size, align) = size_and_align_of(layout);
    let stub = DebuginfoStubDesc {
        name: compute_debuginfo_type_name(cx.tcx, tuple_type, false),
        unique_id: unique_id_str(cx, tuple_type),
        size,
        align,
        scope_item: None,
        flags: DIFlags::FlagZero.bits(),
    };
    let members = component_types
        .into_iter()
        .enumerate()
        .map(|(index, component_type)| {
            member_desc(
                field_name(tuple_field_name(index)),
                cx.layout_of(component_type),
                layout.fields.offset(index),
                DIFlags::FlagZero,
            )
        })
        .collect();
    DebuginfoTypeDesc::Struct { stub, members, generics: vec![] }
}

fn compute_closure_env_desc<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    closure_env_type: Ty<'tcx>,
    def_id: rustc_hir::def_id::DefId,
) -> DebuginfoTypeDesc<'tcx> {
    let up_var_tys = match closure_env_type.kind() {
        ty::Closure(_, args) => args.as_closure().upvar_tys(),
        ty::CoroutineClosure(_, args) => args.as_coroutine_closure().upvar_tys(),
        _ => unreachable!(),
    };
    let (size, align) = cx.size_and_align_of(closure_env_type);
    let stub = DebuginfoStubDesc {
        name: compute_debuginfo_type_name(cx.tcx, closure_env_type, false),
        unique_id: unique_id_str(cx, closure_env_type),
        size,
        align,
        scope_item: Some(def_id),
        flags: DIFlags::FlagZero.bits(),
    };
    let capture_names = cx.tcx.closure_saved_names_of_captured_variables(def_id);
    let layout = cx.layout_of(closure_env_type);
    let members = up_var_tys
        .into_iter()
        .zip(capture_names.iter())
        .enumerate()
        .map(|(index, (up_var_ty, capture_name))| {
            cx.tcx.assert_fully_normalized(cx.typing_env(), up_var_ty);
            member_desc(
                DebuginfoFieldName::Symbol(*capture_name),
                cx.layout_of(up_var_ty),
                layout.fields.offset(index),
                DIFlags::FlagZero,
            )
        })
        .collect();
    DebuginfoTypeDesc::Struct { stub, members, generics: vec![] }
}

fn compute_enum_desc<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    enum_type: Ty<'tcx>,
    span: Span,
) -> DebuginfoTypeDesc<'tcx> {
    let &ty::Adt(enum_adt_def, _) = enum_type.kind() else { unreachable!() };
    let layout = cx.spanned_layout_of(enum_type, span);
    if wants_c_like_enum_debuginfo(cx.tcx, layout) {
        return DebuginfoTypeDesc::Unsupported;
    }

    let visibility_flags = visibility_di_flags(cx, enum_adt_def.did(), enum_adt_def.did());
    let (size, align) = size_and_align_of(layout);
    let stub = DebuginfoStubDesc {
        name: compute_debuginfo_type_name(cx.tcx, enum_type, false),
        unique_id: unique_id_str(cx, enum_type),
        size,
        align,
        scope_item: Some(enum_adt_def.did()),
        flags: visibility_flags.bits(),
    };

    let variants = enum_adt_def
        .variant_range()
        .map(|variant_index| {
            let variant_def = enum_adt_def.variant(variant_index);
            let variant_layout = layout.for_variant(cx, variant_index);
            let members = (0..variant_layout.fields.count())
                .map(|field_index| {
                    let name = if variant_def.ctor_kind() != Some(CtorKind::Fn) {
                        DebuginfoFieldName::Symbol(
                            variant_def.fields[FieldIdx::from_usize(field_index)].name,
                        )
                    } else {
                        field_name(tuple_field_name(field_index))
                    };
                    member_desc(
                        name,
                        variant_layout.field(cx, field_index),
                        variant_layout.fields.offset(field_index),
                        visibility_flags,
                    )
                })
                .collect();
            DebuginfoVariantDesc {
                name: variant_def.name,
                struct_unique_id: UniqueTypeId::for_enum_variant_struct_type(
                    cx.tcx,
                    enum_type,
                    variant_index,
                )
                .generate_unique_id_string(cx.tcx),
                members,
                discr: enums::compute_discriminant_value(cx, layout, variant_index)
                    .opt_single_val(),
            }
        })
        .collect();

    let tag = match layout.layout.variants() {
        &Variants::Single { .. } | &Variants::Empty => None,
        &Variants::Multiple { tag_field, .. } => {
            let tag_ty = tag_base_type(cx.tcx, layout);
            let (size, align) = size_and_align_of(cx.layout_of(tag_ty));
            Some(DebuginfoTagDesc {
                ty: tag_ty,
                size,
                align,
                offset: layout.fields.offset(tag_field.as_usize()),
            })
        }
    };

    DebuginfoTypeDesc::Enum {
        stub,
        variants,
        tag,
        variant_part_unique_id: UniqueTypeId::for_enum_variant_part(cx.tcx, enum_type)
            .generate_unique_id_string(cx.tcx),
        generics: compute_generics(cx, enum_type),
    }
}

/// Mirrors `build_generic_type_param_di_nodes`.
fn compute_generics<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    ty: Ty<'tcx>,
) -> Vec<DebuginfoGenericParamDesc<'tcx>> {
    let ty::Adt(def, args) = *ty.kind() else { return vec![] };
    if args.types().next().is_none() {
        return vec![];
    }
    let generics = cx.tcx.generics_of(def.did());
    let names = get_parameter_names(cx, generics);
    return iter::zip(args, names)
        .filter_map(|(kind, name)| {
            kind.as_type().map(|ty| DebuginfoGenericParamDesc {
                name,
                ty: cx.tcx.normalize_erasing_regions(cx.typing_env(), Unnormalized::new_wip(ty)),
            })
        })
        .collect();

    fn get_parameter_names(cx: &CodegenCx<'_, '_>, generics: &ty::Generics) -> Vec<Symbol> {
        let mut names = generics
            .parent
            .map_or_else(Vec::new, |def_id| get_parameter_names(cx, cx.tcx.generics_of(def_id)));
        names.extend(generics.own_params.iter().map(|param| param.name));
        names
    }
}
