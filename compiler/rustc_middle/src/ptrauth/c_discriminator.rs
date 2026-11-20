//! `#[ptrauth_c_discriminator(N)]`: code addresses that a static holds in C's encoding.
//!
//! Rust signs a function pointer with the discriminator of the function it points to. C instead
//! encodes a code address according to the type of the slot holding it: `0` for `void *` or an
//! integer, or the discriminator of the slot's type. At runtime, `ptrauth_c_cast` can be used to
//! convert between discriminators. A static's initializer cannot call it, so the slots of a static
//! that hold a code address in C's encoding (requiring adjustment) are marked with the
//! discriminator C uses for them:
//!
//! - on a field: the field, in every static whose type contains it;
//! - on a static: the static itself, when it is a single slot or an array of slots.
//!
//! A slot is a pointer, or an integer of pointer size, or an array of them. Unmarked code
//! addresses keep the discriminator of their function.

use rustc_abi::{BackendRepr, FieldsShape, HasDataLayout, Primitive, Size};
use rustc_attr_ir::find_attr;
use rustc_hir::def_id::DefId;

use crate::ty::layout::{LayoutCx, TyAndLayout};
use crate::ty::{self, Instance, Ty, TyCtxt};

/// A misplaced `#[ptrauth_c_discriminator]`.
#[derive(Debug)]
pub enum PtrauthCDiscriminatorError<'tcx> {
    /// On a static whose type is not a slot or an array of slots.
    StaticNotSlot { ty: Ty<'tcx> },
    /// On a field whose type is not a slot or an array of slots.
    FieldNotSlot { field: DefId, ty: Ty<'tcx> },
    /// Two marked fields of a union overlap with different discriminators.
    Conflict { offset: Size },
}

/// Returns the offsets (sorted), in the allocation of the static `static_def_id`, of the slots
/// marked `#[ptrauth_c_discriminator]`, with the discriminator of each.
///
/// Only the static's own allocation is covered: code addresses behind a reference live in
/// another allocation, which may be shared with other uses. Fields inside enums are not covered,
/// as their offsets depend on the variant.
pub fn ptrauth_c_discriminator_slots<'tcx>(
    tcx: TyCtxt<'tcx>,
    static_def_id: DefId,
) -> Result<Vec<(Size, u16)>, PtrauthCDiscriminatorError<'tcx>> {
    let typing_env = ty::TypingEnv::fully_monomorphized();
    let ty = Instance::mono(tcx, static_def_id).ty(tcx, typing_env);
    let Ok(layout) = tcx.layout_of(typing_env.as_query_input(ty)) else {
        return Ok(Vec::new());
    };
    let cx = LayoutCx::new(tcx, typing_env);

    let mut slots = Vec::new();
    if let Some(discriminator) = marked_discriminator(tcx, static_def_id) {
        if !collect_marked(tcx, &cx, layout, Size::ZERO, discriminator, &mut slots) {
            return Err(PtrauthCDiscriminatorError::StaticNotSlot { ty });
        }
        return Ok(slots);
    }

    collect_fields(tcx, &cx, layout, Size::ZERO, &mut slots)?;
    slots.sort_by_key(|&(offset, _)| offset);
    slots.dedup();
    if let Some(w) = slots.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(PtrauthCDiscriminatorError::Conflict { offset: w[0].0 });
    }
    Ok(slots)
}

/// The discriminator for a field or a static marked with the attribute.
fn marked_discriminator(tcx: TyCtxt<'_>, def_id: DefId) -> Option<u16> {
    find_attr!(tcx, def_id, PtrauthCDiscriminator { discriminator } => *discriminator)
}

/// True if `layout` is a slot: a pointer, or an integer of pointer size.
fn is_slot<'tcx>(tcx: TyCtxt<'tcx>, layout: TyAndLayout<'tcx>) -> bool {
    match layout.backend_repr {
        BackendRepr::Scalar(scalar) => {
            matches!(scalar.primitive(), Primitive::Pointer(_) | Primitive::Int(..))
                && scalar.size(&tcx) == tcx.data_layout().pointer_size()
        }
        _ => false,
    }
}

/// Records `layout` - a marked slot or array of slots at `base`, with `discriminator`. Returns
/// `false` if non found.
fn collect_marked<'tcx>(
    tcx: TyCtxt<'tcx>,
    cx: &LayoutCx<'tcx>,
    layout: TyAndLayout<'tcx>,
    base: Size,
    discriminator: u16,
    slots: &mut Vec<(Size, u16)>,
) -> bool {
    if is_slot(tcx, layout) {
        slots.push((base, discriminator));
        return true;
    }
    if let ty::Array(..) = layout.ty.kind()
        && let FieldsShape::Array { stride, count } = layout.fields
    {
        let mut element = Vec::new();
        if !collect_marked(tcx, cx, layout.field(cx, 0), Size::ZERO, discriminator, &mut element) {
            return false;
        }
        repeat(&element, base, stride, count, slots);
        return true;
    }
    false
}

/// Records the marked fields reachable from `layout` at `base`, without following references.
fn collect_fields<'tcx>(
    tcx: TyCtxt<'tcx>,
    cx: &LayoutCx<'tcx>,
    layout: TyAndLayout<'tcx>,
    base: Size,
    slots: &mut Vec<(Size, u16)>,
) -> Result<(), PtrauthCDiscriminatorError<'tcx>> {
    match layout.ty.kind() {
        ty::Adt(def, _) if def.is_struct() || def.is_union() => {
            for (idx, field) in def.non_enum_variant().fields.iter_enumerated() {
                let field_layout = layout.field(cx, idx.as_usize());
                let offset = base + layout.fields.offset(idx.as_usize());
                if let Some(discriminator) = marked_discriminator(tcx, field.did) {
                    if !collect_marked(tcx, cx, field_layout, offset, discriminator, slots) {
                        return Err(PtrauthCDiscriminatorError::FieldNotSlot {
                            field: field.did,
                            ty: field_layout.ty,
                        });
                    }
                } else {
                    collect_fields(tcx, cx, field_layout, offset, slots)?;
                }
            }
        }
        ty::Array(..) => {
            if let FieldsShape::Array { stride, count } = layout.fields {
                let mut element = Vec::new();
                collect_fields(tcx, cx, layout.field(cx, 0), Size::ZERO, &mut element)?;
                repeat(&element, base, stride, count, slots);
            }
        }
        ty::Tuple(..) => {
            for idx in 0..layout.fields.count() {
                let offset = base + layout.fields.offset(idx);
                collect_fields(tcx, cx, layout.field(cx, idx), offset, slots)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Helper for arrays, records teh slots of one array element `element` and repeats it `count`
/// times.
fn repeat(
    element: &[(Size, u16)],
    base: Size,
    stride: Size,
    count: u64,
    slots: &mut Vec<(Size, u16)>,
) {
    if element.is_empty() {
        return;
    }
    for i in 0..count {
        for &(offset, discriminator) in element {
            slots.push((base + stride * i + offset, discriminator));
        }
    }
}
