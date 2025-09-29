// Adapted from https://github.com/rust-lang/rust/blob/10a7aa14fed9b528b74b0f098c4899c37c09a9c7/compiler/rustc_codegen_llvm/src/debuginfo/metadata.rs

use gimli::write::{AttributeValue, UnitEntryId};
use rustc_codegen_ssa::debuginfo::type_names;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, Ty, TyCtxt};

use crate::{DebugContext, FullyMonomorphizedLayoutCx};

#[derive(Default)]
pub(crate) struct TypeDebugContext<'tcx> {
    type_map: FxHashMap<Ty<'tcx>, UnitEntryId>,
}

/// Returns from the enclosing function if the type debuginfo node with the given
/// unique ID can be found in the type map.
macro_rules! return_if_type_created_in_meantime {
    ($type_dbg:expr, $ty:expr) => {
        if let Some(&type_id) = $type_dbg.type_map.get(&$ty) {
            return type_id;
        }
    };
}

impl DebugContext {
    pub(crate) fn debug_type<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        type_dbg: &mut TypeDebugContext<'tcx>,
        ty: Ty<'tcx>,
    ) -> UnitEntryId {
        if let Some(&type_id) = type_dbg.type_map.get(&ty) {
            return type_id;
        }

        let type_id = match ty.kind() {
            ty::Never | ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) => {
                self.basic_type(tcx, ty)
            }
            ty::Tuple(elems) if elems.is_empty() => self.basic_type(tcx, ty),
            ty::Array(elem_ty, len) => self.array_type(
                tcx,
                type_dbg,
                ty,
                *elem_ty,
                len.try_to_target_usize(tcx).expect("expected monomorphic const in codegen"),
            ),
            // ty::Slice(_) | ty::Str
            // ty::Dynamic
            // ty::Foreign
            ty::RawPtr(pointee_type, _) | ty::Ref(_, pointee_type, _) => {
                self.pointer_type(tcx, type_dbg, ty, *pointee_type)
            }
            // ty::Adt(def, args) if def.is_box() && args.get(1).map_or(true, |arg| cx.layout_of(arg.expect_ty()).is_1zst())
            // ty::FnDef(..) | ty::FnPtr(..)
            // ty::Closure(..)
            // ty::Adt(def, ..)
            ty::Tuple(components) => self.tuple_type(tcx, type_dbg, ty, *components),
            // ty::Param(_)
            // FIXME implement remaining types and add unreachable!() to the fallback branch
            _ => self.placeholder_for_type(tcx, type_dbg, ty),
        };

        type_dbg.type_map.insert(ty, type_id);

        type_id
    }

    fn basic_type<'tcx>(&mut self, tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> UnitEntryId {
        let (name, encoding) = match ty.kind() {
            ty::Never => ("!", gimli::DW_ATE_unsigned),
            ty::Tuple(elems) if elems.is_empty() => ("()", gimli::DW_ATE_unsigned),
            ty::Bool => ("bool", gimli::DW_ATE_boolean),
            ty::Char => ("char", gimli::DW_ATE_UTF),
            ty::Int(int_ty) => (int_ty.name_str(), gimli::DW_ATE_signed),
            ty::Uint(uint_ty) => (uint_ty.name_str(), gimli::DW_ATE_unsigned),
            ty::Float(float_ty) => (float_ty.name_str(), gimli::DW_ATE_float),
            _ => unreachable!(),
        };

        let type_id = self.dwarf.unit.add(self.dwarf.unit.root(), gimli::DW_TAG_base_type);
        let type_entry = self.dwarf.unit.get_mut(type_id);
        type_entry.set(gimli::DW_AT_name, AttributeValue::StringRef(self.dwarf.strings.add(name)));
        type_entry.set(gimli::DW_AT_encoding, AttributeValue::Encoding(encoding));
        type_entry.set(
            gimli::DW_AT_byte_size,
            AttributeValue::Udata(FullyMonomorphizedLayoutCx(tcx).layout_of(ty).size.bytes()),
        );

        type_id
    }

    fn array_type<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        type_dbg: &mut TypeDebugContext<'tcx>,
        array_ty: Ty<'tcx>,
        elem_ty: Ty<'tcx>,
        len: u64,
    ) -> UnitEntryId {
        let elem_dw_ty = self.debug_type(tcx, type_dbg, elem_ty);

        return_if_type_created_in_meantime!(type_dbg, array_ty);

        let array_type_id = self.dwarf.unit.add(self.dwarf.unit.root(), gimli::DW_TAG_array_type);
        let array_type_entry = self.dwarf.unit.get_mut(array_type_id);
        array_type_entry.set(gimli::DW_AT_type, AttributeValue::UnitRef(elem_dw_ty));

        let subrange_id = self.dwarf.unit.add(array_type_id, gimli::DW_TAG_subrange_type);
        let subrange_entry = self.dwarf.unit.get_mut(subrange_id);
        subrange_entry.set(gimli::DW_AT_type, AttributeValue::UnitRef(self.array_size_type));
        subrange_entry.set(gimli::DW_AT_lower_bound, AttributeValue::Udata(0));
        subrange_entry.set(gimli::DW_AT_count, AttributeValue::Udata(len));

        array_type_id
    }

    fn pointer_type<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        type_dbg: &mut TypeDebugContext<'tcx>,
        ptr_type: Ty<'tcx>,
        pointee_type: Ty<'tcx>,
    ) -> UnitEntryId {
        let pointee_dw_ty = self.debug_type(tcx, type_dbg, pointee_type);

        return_if_type_created_in_meantime!(type_dbg, ptr_type);

        let name = type_names::compute_debuginfo_type_name(tcx, ptr_type, true);

        if !tcx.type_has_metadata(ptr_type, ty::TypingEnv::fully_monomorphized()) {
            let pointer_type_id =
                self.dwarf.unit.add(self.dwarf.unit.root(), gimli::DW_TAG_pointer_type);
            let pointer_entry = self.dwarf.unit.get_mut(pointer_type_id);
            pointer_entry.set(gimli::DW_AT_type, AttributeValue::UnitRef(pointee_dw_ty));
            pointer_entry
                .set(gimli::DW_AT_name, AttributeValue::StringRef(self.dwarf.strings.add(name)));

            pointer_type_id
        } else {
            // FIXME implement debuginfo for wide pointers
            self.placeholder_for_type(tcx, type_dbg, ptr_type)
        }
    }

    fn tuple_type<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        type_dbg: &mut TypeDebugContext<'tcx>,
        tuple_type: Ty<'tcx>,
        components: &'tcx [Ty<'tcx>],
    ) -> UnitEntryId {
        let components = components
            .into_iter()
            .map(|&ty| (ty, self.debug_type(tcx, type_dbg, ty)))
            .collect::<Vec<_>>();

        return_if_type_created_in_meantime!(type_dbg, tuple_type);

        let name = type_names::compute_debuginfo_type_name(tcx, tuple_type, false);
        let layout = FullyMonomorphizedLayoutCx(tcx).layout_of(tuple_type);

        let tuple_type_id =
            self.dwarf.unit.add(self.dwarf.unit.root(), gimli::DW_TAG_structure_type);
        let tuple_entry = self.dwarf.unit.get_mut(tuple_type_id);
        tuple_entry.set(gimli::DW_AT_name, AttributeValue::StringRef(self.dwarf.strings.add(name)));
        tuple_entry.set(gimli::DW_AT_byte_size, AttributeValue::Udata(layout.size.bytes()));
        tuple_entry.set(gimli::DW_AT_alignment, AttributeValue::Udata(layout.align.bytes()));

        for (i, (ty, dw_ty)) in components.into_iter().enumerate() {
            let member_id = self.dwarf.unit.add(tuple_type_id, gimli::DW_TAG_member);
            let member_entry = self.dwarf.unit.get_mut(member_id);
            member_entry.set(
                gimli::DW_AT_name,
                AttributeValue::StringRef(self.dwarf.strings.add(format!("__{i}"))),
            );
            member_entry.set(gimli::DW_AT_type, AttributeValue::UnitRef(dw_ty));
            member_entry.set(
                gimli::DW_AT_alignment,
                AttributeValue::Udata(FullyMonomorphizedLayoutCx(tcx).layout_of(ty).align.bytes()),
            );
            member_entry.set(
                gimli::DW_AT_data_member_location,
                AttributeValue::Udata(layout.fields.offset(i).bytes()),
            );
        }

        tuple_type_id
    }

    fn placeholder_for_type<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        type_dbg: &mut TypeDebugContext<'tcx>,
        ty: Ty<'tcx>,
    ) -> UnitEntryId {
        self.debug_type(
            tcx,
            type_dbg,
            Ty::new_array(
                tcx,
                tcx.types.u8,
                FullyMonomorphizedLayoutCx(tcx).layout_of(ty).size.bytes(),
            ),
        )
    }
}
