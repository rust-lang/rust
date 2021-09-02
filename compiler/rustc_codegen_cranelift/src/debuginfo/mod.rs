//! Handling of everything related to debuginfo.

mod emit;
mod line_info;
mod unwind;

use crate::prelude::*;

use rustc_index::vec::IndexVec;

use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::{LabelValueLoc, StackSlots, ValueLabel, ValueLoc};
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::ValueLocRange;

use gimli::write::{
    Address, AttributeValue, DwarfUnit, Expression, LineProgram, LineString, Location,
    LocationList, Range, RangeList, UnitEntryId,
};
use gimli::{Encoding, Format, LineEncoding, RunTimeEndian, X86_64};

pub(crate) use emit::{DebugReloc, DebugRelocName};
pub(crate) use unwind::UnwindContext;

fn target_endian(tcx: TyCtxt<'_>) -> RunTimeEndian {
    use rustc_target::abi::Endian;

    match tcx.data_layout.endian {
        Endian::Big => RunTimeEndian::Big,
        Endian::Little => RunTimeEndian::Little,
    }
}

pub(crate) struct DebugContext<'tcx> {
    tcx: TyCtxt<'tcx>,

    endian: RunTimeEndian,

    dwarf: DwarfUnit,
    unit_range_list: RangeList,

    types: FxHashMap<Ty<'tcx>, UnitEntryId>,
}

impl<'tcx> DebugContext<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, isa: &dyn TargetIsa) -> Self {
        let encoding = Encoding {
            format: Format::Dwarf32,
            // FIXME this should be configurable
            // macOS doesn't seem to support DWARF > 3
            // 5 version is required for md5 file hash
            version: if tcx.sess.target.is_like_osx {
                3
            } else {
                // FIXME change to version 5 once the gdb and lldb shipping with the latest debian
                // support it.
                4
            },
            address_size: isa.frontend_config().pointer_bytes(),
        };

        let mut dwarf = DwarfUnit::new(encoding);

        let producer = format!(
            "cg_clif (rustc {}, cranelift {})",
            rustc_interface::util::version_str().unwrap_or("unknown version"),
            cranelift_codegen::VERSION,
        );
        let comp_dir = tcx.sess.opts.working_dir.to_string_lossy(FileNameDisplayPreference::Remapped).into_owned();
        let (name, file_info) = match tcx.sess.local_crate_source_file.clone() {
            Some(path) => {
                let name = path.to_string_lossy().into_owned();
                (name, None)
            }
            None => (tcx.crate_name(LOCAL_CRATE).to_string(), None),
        };

        let mut line_program = LineProgram::new(
            encoding,
            LineEncoding::default(),
            LineString::new(comp_dir.as_bytes(), encoding, &mut dwarf.line_strings),
            LineString::new(name.as_bytes(), encoding, &mut dwarf.line_strings),
            file_info,
        );
        line_program.file_has_md5 = file_info.is_some();

        dwarf.unit.line_program = line_program;

        {
            let name = dwarf.strings.add(name);
            let comp_dir = dwarf.strings.add(comp_dir);

            let root = dwarf.unit.root();
            let root = dwarf.unit.get_mut(root);
            root.set(gimli::DW_AT_producer, AttributeValue::StringRef(dwarf.strings.add(producer)));
            root.set(gimli::DW_AT_language, AttributeValue::Language(gimli::DW_LANG_Rust));
            root.set(gimli::DW_AT_name, AttributeValue::StringRef(name));
            root.set(gimli::DW_AT_comp_dir, AttributeValue::StringRef(comp_dir));
            root.set(gimli::DW_AT_low_pc, AttributeValue::Address(Address::Constant(0)));
        }

        DebugContext {
            tcx,

            endian: target_endian(tcx),

            dwarf,
            unit_range_list: RangeList(Vec::new()),

            types: FxHashMap::default(),
        }
    }

    fn dwarf_ty(&mut self, ty: Ty<'tcx>) -> UnitEntryId {
        if let Some(type_id) = self.types.get(ty) {
            return *type_id;
        }

        let new_entry = |dwarf: &mut DwarfUnit, tag| dwarf.unit.add(dwarf.unit.root(), tag);

        let primitive = |dwarf: &mut DwarfUnit, ate| {
            let type_id = new_entry(dwarf, gimli::DW_TAG_base_type);
            let type_entry = dwarf.unit.get_mut(type_id);
            type_entry.set(gimli::DW_AT_encoding, AttributeValue::Encoding(ate));
            type_id
        };

        let name = format!("{}", ty);
        let layout = self.tcx.layout_of(ParamEnv::reveal_all().and(ty)).unwrap();

        let type_id = match ty.kind() {
            ty::Bool => primitive(&mut self.dwarf, gimli::DW_ATE_boolean),
            ty::Char => primitive(&mut self.dwarf, gimli::DW_ATE_UTF),
            ty::Uint(_) => primitive(&mut self.dwarf, gimli::DW_ATE_unsigned),
            ty::Int(_) => primitive(&mut self.dwarf, gimli::DW_ATE_signed),
            ty::Float(_) => primitive(&mut self.dwarf, gimli::DW_ATE_float),
            ty::Ref(_, pointee_ty, _mutbl)
            | ty::RawPtr(ty::TypeAndMut { ty: pointee_ty, mutbl: _mutbl }) => {
                let type_id = new_entry(&mut self.dwarf, gimli::DW_TAG_pointer_type);

                // Ensure that type is inserted before recursing to avoid duplicates
                self.types.insert(ty, type_id);

                let pointee = self.dwarf_ty(pointee_ty);

                let type_entry = self.dwarf.unit.get_mut(type_id);

                //type_entry.set(gimli::DW_AT_mutable, AttributeValue::Flag(mutbl == rustc_hir::Mutability::Mut));
                type_entry.set(gimli::DW_AT_type, AttributeValue::UnitRef(pointee));

                type_id
            }
            ty::Adt(adt_def, _substs) if adt_def.is_struct() && !layout.is_unsized() => {
                let type_id = new_entry(&mut self.dwarf, gimli::DW_TAG_structure_type);

                // Ensure that type is inserted before recursing to avoid duplicates
                self.types.insert(ty, type_id);

                let variant = adt_def.non_enum_variant();

                for (field_idx, field_def) in variant.fields.iter().enumerate() {
                    let field_offset = layout.fields.offset(field_idx);
                    let field_layout = layout.field(
                        &layout::LayoutCx { tcx: self.tcx, param_env: ParamEnv::reveal_all() },
                        field_idx,
                    );

                    let field_type = self.dwarf_ty(field_layout.ty);

                    let field_id = self.dwarf.unit.add(type_id, gimli::DW_TAG_member);
                    let field_entry = self.dwarf.unit.get_mut(field_id);

                    field_entry.set(
                        gimli::DW_AT_name,
                        AttributeValue::String(field_def.ident.as_str().to_string().into_bytes()),
                    );
                    field_entry.set(
                        gimli::DW_AT_data_member_location,
                        AttributeValue::Udata(field_offset.bytes()),
                    );
                    field_entry.set(gimli::DW_AT_type, AttributeValue::UnitRef(field_type));
                }

                type_id
            }
            _ => new_entry(&mut self.dwarf, gimli::DW_TAG_structure_type),
        };

        let type_entry = self.dwarf.unit.get_mut(type_id);

        type_entry.set(gimli::DW_AT_name, AttributeValue::String(name.into_bytes()));
        type_entry.set(gimli::DW_AT_byte_size, AttributeValue::Udata(layout.size.bytes()));

        self.types.insert(ty, type_id);

        type_id
    }

    fn define_local(&mut self, scope: UnitEntryId, name: String, ty: Ty<'tcx>) -> UnitEntryId {
        let dw_ty = self.dwarf_ty(ty);

        let var_id = self.dwarf.unit.add(scope, gimli::DW_TAG_variable);
        let var_entry = self.dwarf.unit.get_mut(var_id);

        var_entry.set(gimli::DW_AT_name, AttributeValue::String(name.into_bytes()));
        var_entry.set(gimli::DW_AT_type, AttributeValue::UnitRef(dw_ty));

        var_id
    }

    pub(crate) fn define_function(
        &mut self,
        instance: Instance<'tcx>,
        func_id: FuncId,
        name: &str,
        isa: &dyn TargetIsa,
        context: &Context,
        source_info_set: &indexmap::IndexSet<SourceInfo>,
        local_map: IndexVec<mir::Local, CPlace<'tcx>>,
    ) {
        let symbol = func_id.as_u32() as usize;
        let mir = self.tcx.instance_mir(instance.def);

        // FIXME: add to appropriate scope instead of root
        let scope = self.dwarf.unit.root();

        let entry_id = self.dwarf.unit.add(scope, gimli::DW_TAG_subprogram);
        let entry = self.dwarf.unit.get_mut(entry_id);
        let name_id = self.dwarf.strings.add(name);
        // Gdb requires DW_AT_name. Otherwise the DW_TAG_subprogram is skipped.
        entry.set(gimli::DW_AT_name, AttributeValue::StringRef(name_id));
        entry.set(gimli::DW_AT_linkage_name, AttributeValue::StringRef(name_id));

        let end = self.create_debug_lines(symbol, entry_id, context, mir.span, source_info_set);

        self.unit_range_list.0.push(Range::StartLength {
            begin: Address::Symbol { symbol, addend: 0 },
            length: u64::from(end),
        });

        let func_entry = self.dwarf.unit.get_mut(entry_id);
        // Gdb requires both DW_AT_low_pc and DW_AT_high_pc. Otherwise the DW_TAG_subprogram is skipped.
        func_entry.set(
            gimli::DW_AT_low_pc,
            AttributeValue::Address(Address::Symbol { symbol, addend: 0 }),
        );
        // Using Udata for DW_AT_high_pc requires at least DWARF4
        func_entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(u64::from(end)));

        // FIXME make it more reliable and implement scopes before re-enabling this.
        if false {
            let value_labels_ranges = context.build_value_labels_ranges(isa).unwrap();

            for (local, _local_decl) in mir.local_decls.iter_enumerated() {
                let ty = self.tcx.subst_and_normalize_erasing_regions(
                    instance.substs,
                    ty::ParamEnv::reveal_all(),
                    mir.local_decls[local].ty,
                );
                let var_id = self.define_local(entry_id, format!("{:?}", local), ty);

                let location = place_location(
                    self,
                    isa,
                    symbol,
                    context,
                    &local_map,
                    &value_labels_ranges,
                    Place { local, projection: ty::List::empty() },
                );

                let var_entry = self.dwarf.unit.get_mut(var_id);
                var_entry.set(gimli::DW_AT_location, location);
            }
        }

        // FIXME create locals for all entries in mir.var_debug_info
    }
}

fn place_location<'tcx>(
    debug_context: &mut DebugContext<'tcx>,
    isa: &dyn TargetIsa,
    symbol: usize,
    context: &Context,
    local_map: &IndexVec<mir::Local, CPlace<'tcx>>,
    #[allow(rustc::default_hash_types)] value_labels_ranges: &std::collections::HashMap<
        ValueLabel,
        Vec<ValueLocRange>,
    >,
    place: Place<'tcx>,
) -> AttributeValue {
    assert!(place.projection.is_empty()); // FIXME implement them

    match local_map[place.local].inner() {
        CPlaceInner::Var(_local, var) => {
            let value_label = cranelift_codegen::ir::ValueLabel::new(var.index());
            if let Some(value_loc_ranges) = value_labels_ranges.get(&value_label) {
                let loc_list = LocationList(
                    value_loc_ranges
                        .iter()
                        .map(|value_loc_range| Location::StartEnd {
                            begin: Address::Symbol {
                                symbol,
                                addend: i64::from(value_loc_range.start),
                            },
                            end: Address::Symbol { symbol, addend: i64::from(value_loc_range.end) },
                            data: translate_loc(
                                isa,
                                value_loc_range.loc,
                                &context.func.stack_slots,
                            )
                            .unwrap(),
                        })
                        .collect(),
                );
                let loc_list_id = debug_context.dwarf.unit.locations.add(loc_list);

                AttributeValue::LocationListRef(loc_list_id)
            } else {
                // FIXME set value labels for unused locals

                AttributeValue::Exprloc(Expression::new())
            }
        }
        CPlaceInner::VarPair(_, _, _) => {
            // FIXME implement this

            AttributeValue::Exprloc(Expression::new())
        }
        CPlaceInner::VarLane(_, _, _) => {
            // FIXME implement this

            AttributeValue::Exprloc(Expression::new())
        }
        CPlaceInner::Addr(_, _) => {
            // FIXME implement this (used by arguments and returns)

            AttributeValue::Exprloc(Expression::new())

            // For PointerBase::Stack:
            //AttributeValue::Exprloc(translate_loc(ValueLoc::Stack(*stack_slot), &context.func.stack_slots).unwrap())
        }
    }
}

// Adapted from https://github.com/CraneStation/wasmtime/blob/5a1845b4caf7a5dba8eda1fef05213a532ed4259/crates/debug/src/transform/expression.rs#L59-L137
fn translate_loc(
    isa: &dyn TargetIsa,
    loc: LabelValueLoc,
    stack_slots: &StackSlots,
) -> Option<Expression> {
    match loc {
        LabelValueLoc::ValueLoc(ValueLoc::Reg(reg)) => {
            let machine_reg = isa.map_dwarf_register(reg).unwrap();
            let mut expr = Expression::new();
            expr.op_reg(gimli::Register(machine_reg));
            Some(expr)
        }
        LabelValueLoc::ValueLoc(ValueLoc::Stack(ss)) => {
            if let Some(ss_offset) = stack_slots[ss].offset {
                let mut expr = Expression::new();
                expr.op_breg(X86_64::RBP, i64::from(ss_offset) + 16);
                Some(expr)
            } else {
                None
            }
        }
        LabelValueLoc::ValueLoc(ValueLoc::Unassigned) => unreachable!(),
        LabelValueLoc::Reg(reg) => {
            let machine_reg = isa.map_regalloc_reg_to_dwarf(reg).unwrap();
            let mut expr = Expression::new();
            expr.op_reg(gimli::Register(machine_reg));
            Some(expr)
        }
        LabelValueLoc::SPOffset(offset) => {
            let mut expr = Expression::new();
            expr.op_breg(X86_64::RSP, offset);
            Some(expr)
        }
    }
}
