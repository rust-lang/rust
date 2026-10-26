//! Debuginfo `annotations:` for `#[btf_decl_tag("...")]` and `#[btf_type_tag("...")]`.
//!
//! The encoding matches what Clang emits for the C attributes of the same name, so that the BPF
//! backend (and tools like pahole that read DWARF) turn them into BTF `DECL_TAG`/`TYPE_TAG`
//! records without needing to know the producer was rustc.

use rustc_abi::ExternAbi;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::middle::codegen_fn_attrs::BtfTags;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_span::{Span, Symbol};

use super::metadata::spanned_type_di_node;
use super::type_names::compute_debuginfo_type_name;
use super::utils::{DIB, create_DIArray, wide_pointer_kind};
use crate::common::CodegenCx;
use crate::llvm;
use crate::llvm::debuginfo::{DIArray, DIType};

fn annotations<'ll>(cx: &CodegenCx<'ll, '_>, key: &str, tags: &[Symbol]) -> Option<&'ll DIArray> {
    if tags.is_empty() {
        return None;
    }
    let key = cx.create_metadata(key.as_bytes());
    let nodes: Vec<_> = tags
        .iter()
        .map(|tag| Some(cx.md_node_in_context(&[key, cx.create_metadata(tag.as_str().as_bytes())])))
        .collect();
    Some(create_DIArray(DIB(cx), &nodes))
}

/// The `#[btf_decl_tag]`s of the item or field `def_id`.
///
/// This goes through the `btf_tags` query rather than reading the attributes, so that the
/// codegen unit doesn't depend on the HIR of local items.
pub(crate) fn decl_tags<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> &'tcx [Symbol] {
    tcx.btf_tags(def_id).decl_tags
}

/// The `#[btf_type_tag]`s of the item or field `def_id`. See [`decl_tags`].
pub(crate) fn type_tags<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> &'tcx [Symbol] {
    tcx.btf_tags(def_id).type_tags
}

/// The `btf_decl_tag` annotations for the declaration tags `tags`, if any.
pub(crate) fn decl_tag_annotations<'ll>(
    cx: &CodegenCx<'ll, '_>,
    tags: &[Symbol],
) -> Option<&'ll DIArray> {
    annotations(cx, "btf_decl_tag", tags)
}

/// Returns the debuginfo type for a declaration of type `ty`, applying the declaration's
/// `#[btf_type_tag]`s, `tags`, to its outermost pointer.
///
/// `#[btf_type_tag("user")] p: *mut T` corresponds to C's `T __attribute__((btf_type_tag("user")))
/// *p`. Like Clang, the annotation is put on the pointer type rather than on the pointee. Since
/// the tag belongs to the declaration and not to `ty`, the pointer node is created afresh instead
/// of going through the type map. Tags on anything but a thin pointer are ignored.
pub(crate) fn tagged_type_di_node<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    tags: &[Symbol],
    ty: Ty<'tcx>,
    span: Span,
) -> &'ll DIType {
    let pointee = match *ty.kind() {
        ty::RawPtr(pointee, _) | ty::Ref(_, pointee, _) => Some(pointee),
        _ => None,
    };
    let (Some(pointee), Some(annotations)) = (pointee, annotations(cx, "btf_type_tag", tags))
    else {
        return spanned_type_di_node(cx, ty, span);
    };
    if wide_pointer_kind(cx, pointee).is_some() {
        return spanned_type_di_node(cx, ty, span);
    }

    let pointee_di_node = spanned_type_di_node(cx, pointee, span);
    let data_layout = &cx.tcx.data_layout;
    let name = compute_debuginfo_type_name(cx.tcx, ty, true);
    unsafe {
        llvm::LLVMRustDIBuilderCreatePointerType(
            DIB(cx),
            pointee_di_node,
            data_layout.pointer_size().bits(),
            data_layout.pointer_align().bits() as u32,
            name.as_ptr(),
            name.len(),
            Some(annotations),
        )
    }
}

/// The BTF tags of the parameters of `instance`, if it is a local function some of whose
/// parameters are tagged. Index `i` corresponds to MIR argument `i + 1` and to `fn_abi.args[i]`.
pub(crate) fn tagged_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> Option<&'tcx [BtfTags<'tcx>]> {
    // Shims, closures and "rust-call" functions (whose last argument is untupled) don't have a
    // one-to-one mapping between HIR params and arguments.
    let ty::InstanceKind::Item(def_id) = instance.def else { return None };
    let def_id = def_id.as_local()?;
    if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn)
        || tcx.fn_sig(def_id).skip_binder().abi() == ExternAbi::RustCall
    {
        return None;
    }
    let params = tcx.fn_param_btf_tags(def_id);
    (!params.is_empty()).then_some(params)
}
