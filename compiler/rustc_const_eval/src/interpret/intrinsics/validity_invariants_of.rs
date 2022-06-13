use rustc_middle::mir::interpret::{Allocation, ConstAllocation};
use rustc_middle::mir::Mutability;
use rustc_middle::ty::layout::LayoutCx;
use rustc_middle::ty::{ParamEnv, ParamEnvAnd};
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_target::abi::{
    Abi, Align, Endian, FieldsShape, HasDataLayout, Scalar, Size, WrappingRange,
};

#[derive(Debug, Clone, Copy)]
struct Invariant {
    offset: Size,
    size: Size,
    start: u128,
    end: u128,
}

// TODO: Don't add duplicate invariants (maybe use a HashMap?)
fn add_invariants<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, invs: &mut Vec<Invariant>, offset: Size) {
    let x = tcx.layout_of(ParamEnvAnd { param_env: ParamEnv::reveal_all(), value: ty });

    if let Ok(layout) = x {
        if let Abi::Scalar(Scalar::Initialized { value, valid_range }) = layout.layout.abi() {
            let size = value.size(&tcx);
            let WrappingRange { start, end } = valid_range;
            invs.push(Invariant { offset, size, start, end })
        }

        let param_env = ParamEnv::reveal_all();
        let unwrap = LayoutCx { tcx, param_env };

        match layout.layout.fields() {
            FieldsShape::Primitive => {}
            FieldsShape::Union(_) => {}
            FieldsShape::Array { stride, count } => {
                // TODO: should we just bail if we're making a Too Large type?
                // (Like [bool; 1_000_000])
                for idx in 0..*count {
                    let off = offset + *stride * idx;
                    let f = layout.field(&unwrap, idx as usize);
                    add_invariants(tcx, f.ty, invs, off);
                }
            }
            FieldsShape::Arbitrary { offsets, .. } => {
                for (idx, &field_offset) in offsets.iter().enumerate() {
                    let f = layout.field(&unwrap, idx);
                    if f.ty == ty {
                        // Some types contain themselves as fields, such as
                        // &mut [T]
                        // Easy solution is to just not recurse then.
                    } else {
                        add_invariants(tcx, f.ty, invs, offset + field_offset);
                    }
                }
            }
        }
    }
}

fn extend_encoded_int(to: &mut Vec<u8>, endian: Endian, ptr_size: PointerSize, value: Size) {
    match (endian, ptr_size) {
        (Endian::Little, PointerSize::Bits16) => to.extend((value.bytes() as u16).to_le_bytes()),
        (Endian::Little, PointerSize::Bits32) => to.extend((value.bytes() as u32).to_le_bytes()),
        (Endian::Little, PointerSize::Bits64) => to.extend((value.bytes()).to_le_bytes()),
        (Endian::Big, PointerSize::Bits16) => to.extend((value.bytes() as u16).to_be_bytes()),
        (Endian::Big, PointerSize::Bits32) => to.extend((value.bytes() as u32).to_be_bytes()),
        (Endian::Big, PointerSize::Bits64) => to.extend((value.bytes()).to_be_bytes()),
    }
}

#[derive(Clone, Copy)]
enum PointerSize {
    Bits16,
    Bits32,
    Bits64,
}

/// Directly returns a `ConstAllocation` containing a list of validity invariants of the given type.
pub(crate) fn alloc_validity_invariants_of<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> ConstAllocation<'tcx> {
    let mut invs: Vec<Invariant> = Vec::new();

    let layout = tcx.data_layout();

    let ptr_size = match layout.pointer_size.bits() {
        16 => PointerSize::Bits16,
        32 => PointerSize::Bits32,
        64 => PointerSize::Bits64,
        _ => {
            // Not sure if this can happen, but just return an empty slice?
            let alloc =
                Allocation::from_bytes(Vec::new(), Align::from_bytes(8).unwrap(), Mutability::Not);
            return tcx.intern_const_alloc(alloc);
        }
    };

    add_invariants(tcx, ty, &mut invs, Size::ZERO);

    let encode_range = match layout.endian {
        Endian::Little => |r: u128| r.to_le_bytes(),
        Endian::Big => |r: u128| r.to_be_bytes(),
    };

    let mut encoded = Vec::new();

    // TODO: this needs to match the layout of `Invariant` in core/src/intrinsics.rs
    // how do we ensure that?
    for inv in invs {
        extend_encoded_int(&mut encoded, layout.endian, ptr_size, inv.offset);
        extend_encoded_int(&mut encoded, layout.endian, ptr_size, inv.size);
        encoded.extend(encode_range(inv.start));
        encoded.extend(encode_range(inv.end));
    }

    // TODO: The alignment here should be calculated from the struct definition, I guess?
    let alloc = Allocation::from_bytes(encoded, Align::from_bytes(8).unwrap(), Mutability::Not);
    tcx.intern_const_alloc(alloc)
}
