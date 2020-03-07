mod emit;
mod line_info;

use crate::prelude::*;

use cranelift_codegen::ir::{StackSlots, ValueLabel, ValueLoc};
use cranelift_codegen::isa::RegUnit;
use cranelift_codegen::ValueLocRange;

use gimli::write::{
    self, Address, AttributeValue, DwarfUnit, Expression, LineProgram, LineString, Location,
    LocationList, Range, RangeList, UnitEntryId, Writer,
};
use gimli::{Encoding, Format, LineEncoding, Register, RunTimeEndian, X86_64};

pub use emit::{DebugReloc, DebugRelocName};

fn target_endian(tcx: TyCtxt) -> RunTimeEndian {
    use rustc::ty::layout::Endian;

    match tcx.data_layout.endian {
        Endian::Big => RunTimeEndian::Big,
        Endian::Little => RunTimeEndian::Little,
    }
}

pub struct DebugContext<'tcx> {
    tcx: TyCtxt<'tcx>,

    endian: RunTimeEndian,
    symbols: indexmap::IndexMap<FuncId, String>,

    dwarf: DwarfUnit,
    unit_range_list: RangeList,

    types: HashMap<Ty<'tcx>, UnitEntryId>,
}

impl<'tcx> DebugContext<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, address_size: u8) -> Self {
        let encoding = Encoding {
            format: Format::Dwarf32,
            // TODO: this should be configurable
            // macOS doesn't seem to support DWARF > 3
            version: 3,
            address_size,
        };

        let mut dwarf = DwarfUnit::new(encoding);

        // FIXME: how to get version when building out of tree?
        // Normally this would use option_env!("CFG_VERSION").
        let producer = format!("cranelift fn (rustc version {})", "unknown version");
        let comp_dir = tcx.sess.working_dir.0.to_string_lossy().into_owned();
        let name = match tcx.sess.local_crate_source_file {
            Some(ref path) => path.to_string_lossy().into_owned(),
            None => tcx.crate_name(LOCAL_CRATE).to_string(),
        };

        let line_program = LineProgram::new(
            encoding,
            LineEncoding::default(),
            LineString::new(comp_dir.as_bytes(), encoding, &mut dwarf.line_strings),
            LineString::new(name.as_bytes(), encoding, &mut dwarf.line_strings),
            None,
        );
        dwarf.unit.line_program = line_program;

        {
            let name = dwarf.strings.add(name);
            let comp_dir = dwarf.strings.add(comp_dir);

            let root = dwarf.unit.root();
            let root = dwarf.unit.get_mut(root);
            root.set(
                gimli::DW_AT_producer,
                AttributeValue::StringRef(dwarf.strings.add(producer)),
            );
            root.set(
                gimli::DW_AT_language,
                AttributeValue::Language(gimli::DW_LANG_Rust),
            );
            root.set(gimli::DW_AT_name, AttributeValue::StringRef(name));
            root.set(gimli::DW_AT_comp_dir, AttributeValue::StringRef(comp_dir));
            root.set(
                gimli::DW_AT_low_pc,
                AttributeValue::Address(Address::Constant(0)),
            );
        }

        DebugContext {
            tcx,

            endian: target_endian(tcx),
            symbols: indexmap::IndexMap::new(),

            dwarf,
            unit_range_list: RangeList(Vec::new()),

            types: HashMap::new(),
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

        let type_id = match ty.kind {
            ty::Bool => primitive(&mut self.dwarf, gimli::DW_ATE_boolean),
            ty::Char => primitive(&mut self.dwarf, gimli::DW_ATE_UTF),
            ty::Uint(_) => primitive(&mut self.dwarf, gimli::DW_ATE_unsigned),
            ty::Int(_) => primitive(&mut self.dwarf, gimli::DW_ATE_signed),
            ty::Float(_) => primitive(&mut self.dwarf, gimli::DW_ATE_float),
            ty::Ref(_, pointee_ty, _mutbl)
            | ty::RawPtr(ty::TypeAndMut {
                ty: pointee_ty,
                mutbl: _mutbl,
            }) => {
                let type_id = new_entry(&mut self.dwarf, gimli::DW_TAG_pointer_type);

                // Ensure that type is inserted before recursing to avoid duplicates
                self.types.insert(ty, type_id);

                let pointee = self.dwarf_ty(pointee_ty);

                let type_entry = self.dwarf.unit.get_mut(type_id);

                //type_entry.set(gimli::DW_AT_mutable, AttributeValue::Flag(mutbl == rustc_hir::Mutability::Mut));
                type_entry.set(gimli::DW_AT_type, AttributeValue::ThisUnitEntryRef(pointee));

                type_id
            }
            ty::Adt(adt_def, _substs) if adt_def.is_struct() && !layout.is_unsized() => {
                let type_id = new_entry(&mut self.dwarf, gimli::DW_TAG_structure_type);

                // Ensure that type is inserted before recursing to avoid duplicates
                self.types.insert(ty, type_id);

                let variant = adt_def.non_enum_variant();

                for (field_idx, field_def) in variant.fields.iter().enumerate() {
                    let field_offset = layout.fields.offset(field_idx);
                    let field_layout = layout.field(&layout::LayoutCx {
                        tcx: self.tcx,
                        param_env: ParamEnv::reveal_all(),
                    }, field_idx).unwrap();

                    let field_type = self.dwarf_ty(field_layout.ty);

                    let field_id = self.dwarf.unit.add(type_id, gimli::DW_TAG_member);
                    let field_entry = self.dwarf.unit.get_mut(field_id);

                    field_entry.set(gimli::DW_AT_name, AttributeValue::String(field_def.ident.as_str().to_string().into_bytes()));
                    field_entry.set(gimli::DW_AT_data_member_location, AttributeValue::Udata(field_offset.bytes()));
                    field_entry.set(gimli::DW_AT_type, AttributeValue::ThisUnitEntryRef(field_type));
                }

                type_id
            }
            _ => new_entry(&mut self.dwarf, gimli::DW_TAG_structure_type),
        };

        let type_entry = self.dwarf.unit.get_mut(type_id);

        type_entry.set(gimli::DW_AT_name, AttributeValue::String(name.into_bytes()));
        type_entry.set(
            gimli::DW_AT_byte_size,
            AttributeValue::Udata(layout.size.bytes()),
        );

        self.types.insert(ty, type_id);

        type_id
    }
}

pub struct FunctionDebugContext<'a, 'tcx> {
    debug_context: &'a mut DebugContext<'tcx>,
    entry_id: UnitEntryId,
    symbol: usize,
    instance: Instance<'tcx>,
    mir: &'tcx mir::Body<'tcx>,
}

impl<'a, 'tcx> FunctionDebugContext<'a, 'tcx> {
    pub fn new(
        debug_context: &'a mut DebugContext<'tcx>,
        instance: Instance<'tcx>,
        func_id: FuncId,
        name: &str,
    ) -> Self {
        let mir = *debug_context.tcx.instance_mir(instance.def);

        let (symbol, _) = debug_context.symbols.insert_full(func_id, name.to_string());

        // FIXME: add to appropriate scope intead of root
        let scope = debug_context.dwarf.unit.root();

        let entry_id = debug_context
            .dwarf
            .unit
            .add(scope, gimli::DW_TAG_subprogram);
        let entry = debug_context.dwarf.unit.get_mut(entry_id);
        let name_id = debug_context.dwarf.strings.add(name);
        entry.set(
            gimli::DW_AT_linkage_name,
            AttributeValue::StringRef(name_id),
        );

        FunctionDebugContext {
            debug_context,
            entry_id,
            symbol,
            instance,
            mir,
        }
    }

    fn define_local(&mut self, name: String, ty: Ty<'tcx>) -> UnitEntryId {
        let ty = self.debug_context.tcx.subst_and_normalize_erasing_regions(
            self.instance.substs,
            ty::ParamEnv::reveal_all(),
            &ty,
        );
        let dw_ty = self.debug_context.dwarf_ty(ty);

        let var_id = self
            .debug_context
            .dwarf
            .unit
            .add(self.entry_id, gimli::DW_TAG_variable);
        let var_entry = self.debug_context.dwarf.unit.get_mut(var_id);

        var_entry.set(gimli::DW_AT_name, AttributeValue::String(name.into_bytes()));
        var_entry.set(gimli::DW_AT_type, AttributeValue::ThisUnitEntryRef(dw_ty));

        var_id
    }

    pub fn define(
        &mut self,
        context: &Context,
        isa: &dyn cranelift_codegen::isa::TargetIsa,
        source_info_set: &indexmap::IndexSet<SourceInfo>,
        local_map: HashMap<mir::Local, CPlace<'tcx>>,
    ) {
        let end = self.create_debug_lines(context, isa, source_info_set);

        self.debug_context
            .unit_range_list
            .0
            .push(Range::StartLength {
                begin: Address::Symbol {
                    symbol: self.symbol,
                    addend: 0,
                },
                length: end as u64,
            });

        // FIXME make it more reliable and implement scopes before re-enabling this.
        if false {
            let value_labels_ranges = context.build_value_labels_ranges(isa).unwrap();

            for (local, _local_decl) in self.mir.local_decls.iter_enumerated() {
                let var_id = self.define_local(format!("{:?}", local), &self.mir.local_decls[local].ty);

                let location = place_location(
                    self,
                    context,
                    &local_map,
                    &value_labels_ranges,
                    Place {
                        local,
                        projection: ty::List::empty(),
                    },
                );

                let var_entry = self.debug_context.dwarf.unit.get_mut(var_id);
                var_entry.set(gimli::DW_AT_location, location);
            }
        }

        // FIXME create locals for all entries in mir.var_debug_info
    }
}

fn place_location<'a, 'tcx>(
    func_debug_ctx: &mut FunctionDebugContext<'a, 'tcx>,
    context: &Context,
    local_map: &HashMap<mir::Local, CPlace<'tcx>>,
    value_labels_ranges: &HashMap<ValueLabel, Vec<ValueLocRange>>,
    place: Place<'tcx>,
) -> AttributeValue {
    assert!(place.projection.is_empty()); // FIXME implement them

    match local_map[&place.local].inner() {
        CPlaceInner::Var(local) => {
            let value_label = cranelift_codegen::ir::ValueLabel::from_u32(local.as_u32());
            if let Some(value_loc_ranges) = value_labels_ranges.get(&value_label) {
                let loc_list = LocationList(
                    value_loc_ranges
                        .iter()
                        .map(|value_loc_range| Location::StartEnd {
                            begin: Address::Symbol {
                                symbol: func_debug_ctx.symbol,
                                addend: i64::from(value_loc_range.start),
                            },
                            end: Address::Symbol {
                                symbol: func_debug_ctx.symbol,
                                addend: i64::from(value_loc_range.end),
                            },
                            data: Expression(
                                translate_loc(value_loc_range.loc, &context.func.stack_slots).unwrap(),
                            ),
                        })
                        .collect(),
                );
                let loc_list_id = func_debug_ctx.debug_context.dwarf.unit.locations.add(loc_list);

                AttributeValue::LocationListRef(loc_list_id)
            } else {
                // FIXME set value labels for unused locals

                AttributeValue::Exprloc(Expression(vec![]))
            }
        }
        CPlaceInner::Addr(_, _) => {
            // FIXME implement this (used by arguments and returns)

            AttributeValue::Exprloc(Expression(vec![]))

            // For PointerBase::Stack:
            //AttributeValue::Exprloc(Expression(translate_loc(ValueLoc::Stack(*stack_slot), &context.func.stack_slots).unwrap()))
        }
        CPlaceInner::NoPlace => AttributeValue::Exprloc(Expression(vec![])),
    }
}





// Adapted from https://github.com/CraneStation/wasmtime/blob/5a1845b4caf7a5dba8eda1fef05213a532ed4259/crates/debug/src/transform/expression.rs#L59-L137

fn map_reg(reg: RegUnit) -> Register {
    static mut REG_X86_MAP: Option<HashMap<RegUnit, Register>> = None;
    // FIXME lazy initialization?
    unsafe {
        if REG_X86_MAP.is_none() {
            REG_X86_MAP = Some(HashMap::new());
        }
        if let Some(val) = REG_X86_MAP.as_mut().unwrap().get(&reg) {
            return *val;
        }
        let result = match reg {
            0 => X86_64::RAX,
            1 => X86_64::RCX,
            2 => X86_64::RDX,
            3 => X86_64::RBX,
            4 => X86_64::RSP,
            5 => X86_64::RBP,
            6 => X86_64::RSI,
            7 => X86_64::RDI,
            8 => X86_64::R8,
            9 => X86_64::R9,
            10 => X86_64::R10,
            11 => X86_64::R11,
            12 => X86_64::R12,
            13 => X86_64::R13,
            14 => X86_64::R14,
            15 => X86_64::R15,
            16 => X86_64::XMM0,
            17 => X86_64::XMM1,
            18 => X86_64::XMM2,
            19 => X86_64::XMM3,
            20 => X86_64::XMM4,
            21 => X86_64::XMM5,
            22 => X86_64::XMM6,
            23 => X86_64::XMM7,
            24 => X86_64::XMM8,
            25 => X86_64::XMM9,
            26 => X86_64::XMM10,
            27 => X86_64::XMM11,
            28 => X86_64::XMM12,
            29 => X86_64::XMM13,
            30 => X86_64::XMM14,
            31 => X86_64::XMM15,
            _ => panic!("unknown x86_64 register {}", reg),
        };
        REG_X86_MAP.as_mut().unwrap().insert(reg, result);
        result
    }
}

fn translate_loc(loc: ValueLoc, stack_slots: &StackSlots) -> Option<Vec<u8>> {
    match loc {
        ValueLoc::Reg(reg) => {
            let machine_reg = map_reg(reg).0 as u8;
            assert!(machine_reg <= 32); // FIXME
            Some(vec![gimli::constants::DW_OP_reg0.0 + machine_reg])
        }
        ValueLoc::Stack(ss) => {
            if let Some(ss_offset) = stack_slots[ss].offset {
                let endian = gimli::RunTimeEndian::Little;
                let mut writer = write::EndianVec::new(endian);
                writer
                    .write_u8(gimli::constants::DW_OP_breg0.0 + X86_64::RBP.0 as u8)
                    .expect("bp wr");
                writer.write_sleb128(ss_offset as i64 + 16).expect("ss wr");
                let buf = writer.into_vec();
                return Some(buf);
            }
            None
        }
        _ => None,
    }
}
