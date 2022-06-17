use rustc_data_structures::fx::FxHashMap;
use rustc_hir::lang_items::LangItem;
use rustc_middle::mir::interpret::{AllocRange, Allocation, ConstAllocation, Scalar as MirScalar};
use rustc_middle::mir::Mutability;
use rustc_middle::ty::layout::LayoutCx;
use rustc_middle::ty::{ParamEnv, ParamEnvAnd};
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_target::abi::{
    Abi, FieldsShape, HasDataLayout, Integer, Primitive, Scalar, Size, TyAndLayout, Variants,
    WrappingRange,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum InvariantSize {
    U8,
    U16,
    U32,
    U64,
    U128,
    Pointer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct InvariantKey {
    offset: Size,
    size: InvariantSize,
}

// FIXME: Don't add duplicate invariants (maybe use a HashMap?)
fn add_invariants<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    invs: &mut FxHashMap<InvariantKey, WrappingRange>,
    offset: Size,
    strictness: InvariantStrictness,
) {
    if strictness == InvariantStrictness::Disable {
        return;
    }

    let x = tcx.layout_of(ParamEnvAnd { param_env: ParamEnv::reveal_all(), value: ty });

    if let Ok(layout) = x {
        if let Abi::Scalar(Scalar::Initialized { value, valid_range }) = layout.layout.abi() {
            let size = match value {
                Primitive::Int(Integer::I8, _) => InvariantSize::U8,
                Primitive::Int(Integer::I16, _) => InvariantSize::U16,
                Primitive::Int(Integer::I32, _) => InvariantSize::U32,
                Primitive::Int(Integer::I64, _) => InvariantSize::U64,
                Primitive::Int(Integer::I128, _) => InvariantSize::U128,
                Primitive::F32 => InvariantSize::U32,
                Primitive::F64 => InvariantSize::U64,
                Primitive::Pointer => InvariantSize::Pointer,
            };

            if !valid_range.is_full_for(value.size(&tcx)) || strictness == InvariantStrictness::All
            {
                // Pick the first scalar we see, this means NonZeroU8(u8) ends up with only one
                // invariant, the stricter one.
                let _: Result<_, _> = invs.try_insert(InvariantKey { offset, size }, valid_range);
            }
        }

        //dbg!(&ty, &layout);
        if !matches!(layout.layout.variants(), Variants::Single { .. }) {
            // We *don't* want to look for fields inside enums.
            return;
        }

        let param_env = ParamEnv::reveal_all();
        let layout_cx = LayoutCx { tcx, param_env };

        match layout.layout.fields() {
            FieldsShape::Primitive => {}
            FieldsShape::Union(_) => {}
            FieldsShape::Array { stride, count } => {
                // We may wish to bail out if we're generating too many invariants.
                // That would lead to false negatives, though.
                for idx in 0..*count {
                    let off = offset + *stride * idx;
                    let f = layout.field(&layout_cx, idx as usize);
                    add_invariants(tcx, f.ty, invs, off, strictness);
                }
            }
            FieldsShape::Arbitrary { offsets, .. } => {
                for (idx, &field_offset) in offsets.iter().enumerate() {
                    let f = layout.field(&layout_cx, idx);
                    if f.ty == ty {
                        // Some types contain themselves as fields, such as
                        // &mut [T]
                        // Easy solution is to just not recurse then.
                    } else {
                        add_invariants(tcx, f.ty, invs, offset + field_offset, strictness);
                    }
                }
            }
        }
    }
}

fn get_layout_of_invariant<'tcx>(tcx: TyCtxt<'tcx>) -> TyAndLayout<'tcx, Ty<'tcx>> {
    let item = tcx.require_lang_item(LangItem::ValidityInvariant, None);
    let ty = tcx.type_of(item);
    let layout = tcx
        .layout_of(ParamEnv::reveal_all().and(ty))
        .expect("invalid layout for ValidityInvariant lang item");
    layout
}

#[derive(PartialEq, Clone, Copy, Eq)]
pub(crate) enum InvariantStrictness {
    Disable,
    Normal,
    All,
}

/// Directly returns a `ConstAllocation` containing a list of validity invariants of the given type.
pub(crate) fn alloc_validity_invariants_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    strictness: InvariantStrictness,
) -> (ConstAllocation<'tcx>, usize) {
    let mut invs = FxHashMap::default();

    let layout = tcx.data_layout();
    let validity_invariant = get_layout_of_invariant(tcx);

    add_invariants(tcx, ty, &mut invs, Size::ZERO, strictness);

    let allocation_size = validity_invariant.layout.size() * invs.len() as u64;
    let mut alloc =
        Allocation::uninit(allocation_size, validity_invariant.layout.align().abi, true).unwrap();

    let offset_off = validity_invariant.layout.fields().offset(0);
    let size_off = validity_invariant.layout.fields().offset(1);
    let start_off = validity_invariant.layout.fields().offset(2);
    let end_off = validity_invariant.layout.fields().offset(3);

    for (idx, invariant) in invs.iter().enumerate() {
        let offset = idx as u64 * validity_invariant.layout.size();

        let offset_range = AllocRange { start: offset + offset_off, size: layout.pointer_size };
        alloc
            .write_scalar(
                &tcx,
                offset_range,
                MirScalar::from_machine_usize(invariant.0.offset.bytes(), &tcx).into(),
            )
            .unwrap();

        let size_range = AllocRange { start: offset + size_off, size: Size::from_bytes(1) };
        alloc
            .write_scalar(&tcx, size_range, MirScalar::from_u8(invariant.0.size as u8).into())
            .unwrap();

        let offset_range = AllocRange { start: offset + start_off, size: Size::from_bytes(16) };
        alloc
            .write_scalar(&tcx, offset_range, MirScalar::from_u128(invariant.1.start).into())
            .unwrap();

        let offset_range = AllocRange { start: offset + end_off, size: Size::from_bytes(16) };
        alloc
            .write_scalar(&tcx, offset_range, MirScalar::from_u128(invariant.1.end).into())
            .unwrap();
    }

    // The allocation is not mutable, we just needed write_scalar.
    alloc.mutability = Mutability::Not;

    (tcx.intern_const_alloc(alloc), invs.len())
}
