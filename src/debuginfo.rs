use crate::prelude::*;

use crate::backend::WriteDebugInfo;

use std::marker::PhantomData;

use syntax::source_map::FileName;

use cranelift::codegen::ir::{StackSlots, ValueLoc};
use cranelift::codegen::isa::RegUnit;

use gimli::write::{
    self, Address, AttributeValue, DwarfUnit, EndianVec, Expression, FileId, LineProgram,
    LineString, LineStringTable, Location, LocationList, Range, RangeList, Result, Sections,
    UnitEntryId, Writer,
};
use gimli::{Encoding, Format, LineEncoding, Register, RunTimeEndian, SectionId, X86_64};

fn target_endian(tcx: TyCtxt) -> RunTimeEndian {
    use rustc::ty::layout::Endian;

    match tcx.data_layout.endian {
        Endian::Big => RunTimeEndian::Big,
        Endian::Little => RunTimeEndian::Little,
    }
}

fn line_program_add_file(
    line_program: &mut LineProgram,
    line_strings: &mut LineStringTable,
    file: &FileName,
) -> FileId {
    match file {
        FileName::Real(path) => {
            let dir_name = path.parent().unwrap().to_str().unwrap().as_bytes();
            let dir_id = if !dir_name.is_empty() {
                let dir_name = LineString::new(dir_name, line_program.encoding(), line_strings);
                line_program.add_directory(dir_name)
            } else {
                line_program.default_directory()
            };
            let file_name = LineString::new(
                path.file_name().unwrap().to_str().unwrap().as_bytes(),
                line_program.encoding(),
                line_strings,
            );
            line_program.add_file(file_name, dir_id, None)
        }
        // FIXME give more appropriate file names
        _ => {
            let dir_id = line_program.default_directory();
            let dummy_file_name = LineString::new(
                file.to_string().into_bytes(),
                line_program.encoding(),
                line_strings,
            );
            line_program.add_file(dummy_file_name, dir_id, None)
        }
    }
}

#[derive(Clone)]
pub struct DebugReloc {
    pub offset: u32,
    pub size: u8,
    pub name: DebugRelocName,
    pub addend: i64,
}

#[derive(Clone)]
pub enum DebugRelocName {
    Section(SectionId),
    Symbol(usize),
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
            let name = dwarf.strings.add(&*name);
            let comp_dir = dwarf.strings.add(&*comp_dir);

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

    fn emit_location(&mut self, entry_id: UnitEntryId, span: Span) {
        let loc = self.tcx.sess.source_map().lookup_char_pos(span.lo());

        let file_id = line_program_add_file(
            &mut self.dwarf.unit.line_program,
            &mut self.dwarf.line_strings,
            &loc.file.name,
        );

        let entry = self.dwarf.unit.get_mut(entry_id);

        entry.set(
            gimli::DW_AT_decl_file,
            AttributeValue::FileIndex(Some(file_id)),
        );
        entry.set(
            gimli::DW_AT_decl_line,
            AttributeValue::Udata(loc.line as u64),
        );
        // FIXME: probably omit this
        entry.set(
            gimli::DW_AT_decl_column,
            AttributeValue::Udata(loc.col.to_usize() as u64),
        );
    }

    fn dwarf_ty(&mut self, ty: Ty<'tcx>) -> UnitEntryId {
        if let Some(type_id) = self.types.get(ty) {
            return *type_id;
        }

        let new_entry = |dwarf: &mut DwarfUnit, tag| {
            dwarf.unit.add(dwarf.unit.root(), tag)
        };

        let primtive = |dwarf: &mut DwarfUnit, ate| {
            let type_id = new_entry(dwarf, gimli::DW_TAG_base_type);
            let type_entry = dwarf.unit.get_mut(type_id);
            type_entry.set(gimli::DW_AT_encoding, AttributeValue::Encoding(ate));
            type_id
        };

        let type_id = match ty.kind {
            ty::Bool => primtive(&mut self.dwarf, gimli::DW_ATE_boolean),
            ty::Char => primtive(&mut self.dwarf, gimli::DW_ATE_UTF),
            ty::Uint(_) => primtive(&mut self.dwarf, gimli::DW_ATE_unsigned),
            ty::Int(_) => primtive(&mut self.dwarf, gimli::DW_ATE_signed),
            ty::Float(_) => primtive(&mut self.dwarf, gimli::DW_ATE_float),
            ty::Ref(_, pointee_ty, mutbl) | ty::RawPtr(ty::TypeAndMut { ty: pointee_ty, mutbl }) => {
                let type_id = new_entry(&mut self.dwarf, gimli::DW_TAG_pointer_type);

                // Ensure that type is inserted before recursing to avoid duplicates
                self.types.insert(ty, type_id);

                let pointee = self.dwarf_ty(pointee_ty);

                let type_entry = self.dwarf.unit.get_mut(type_id);

                //type_entry.set(gimli::DW_AT_mutable, AttributeValue::Flag(mutbl == rustc::hir::Mutability::MutMutable));
                type_entry.set(gimli::DW_AT_type, AttributeValue::ThisUnitEntryRef(pointee));

                type_id
            }
            _ => new_entry(&mut self.dwarf, gimli::DW_TAG_structure_type),
        };
        let name = format!("{}", ty);
        let layout = self.tcx.layout_of(ParamEnv::reveal_all().and(ty)).unwrap();

        let type_entry = self.dwarf.unit.get_mut(type_id);

        type_entry.set(gimli::DW_AT_name, AttributeValue::String(name.into_bytes()));
        type_entry.set(gimli::DW_AT_byte_size, AttributeValue::Udata(layout.size.bytes()));

        self.types.insert(ty, type_id);

        type_id
    }

    pub fn emit<P: WriteDebugInfo>(&mut self, product: &mut P) {
        let unit_range_list_id = self.dwarf.unit.ranges.add(self.unit_range_list.clone());
        let root = self.dwarf.unit.root();
        let root = self.dwarf.unit.get_mut(root);
        root.set(
            gimli::DW_AT_ranges,
            AttributeValue::RangeListRef(unit_range_list_id),
        );

        let mut sections = Sections::new(WriterRelocate::new(self));
        self.dwarf.write(&mut sections).unwrap();

        let mut section_map = HashMap::new();
        let _: Result<()> = sections.for_each_mut(|id, section| {
            if !section.writer.slice().is_empty() {
                let section_id = product.add_debug_section(id, section.writer.take());
                section_map.insert(id, section_id);
            }
            Ok(())
        });

        let _: Result<()> = sections.for_each(|id, section| {
            if let Some(section_id) = section_map.get(&id) {
                for reloc in &section.relocs {
                    product.add_debug_reloc(&section_map, &self.symbols, section_id, reloc);
                }
            }
            Ok(())
        });
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
        _sig: &Signature,
    ) -> Self {
        let mir = debug_context.tcx.instance_mir(instance.def);

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

        entry.set(
            gimli::DW_AT_low_pc,
            AttributeValue::Address(Address::Symbol { symbol, addend: 0 }),
        );

        debug_context.emit_location(entry_id, mir.span);

        FunctionDebugContext {
            debug_context,
            entry_id,
            symbol,
            instance,
            mir,
        }
    }

    pub fn define(
        &mut self,
        context: &Context,
        isa: &dyn cranelift::codegen::isa::TargetIsa,
        source_info_set: &indexmap::IndexSet<(Span, mir::SourceScope)>,
    ) {
        let tcx = self.debug_context.tcx;

        let line_program = &mut self.debug_context.dwarf.unit.line_program;

        line_program.begin_sequence(Some(Address::Symbol {
            symbol: self.symbol,
            addend: 0,
        }));

        let encinfo = isa.encoding_info();
        let func = &context.func;
        let mut ebbs = func.layout.ebbs().collect::<Vec<_>>();
        ebbs.sort_by_key(|ebb| func.offsets[*ebb]); // Ensure inst offsets always increase

        let line_strings = &mut self.debug_context.dwarf.line_strings;
        let mut create_row_for_span = |line_program: &mut LineProgram, span: Span| {
            let loc = tcx.sess.source_map().lookup_char_pos(span.lo());
            let file_id = line_program_add_file(line_program, line_strings, &loc.file.name);

            /*println!(
                "srcloc {:>04X} {}:{}:{}",
                line_program.row().address_offset,
                file.display(),
                loc.line,
                loc.col.to_u32()
            );*/

            line_program.row().file = file_id;
            line_program.row().line = loc.line as u64;
            line_program.row().column = loc.col.to_u32() as u64 + 1;
            line_program.generate_row();
        };

        let mut end = 0;
        for ebb in ebbs {
            for (offset, inst, size) in func.inst_offsets(ebb, &encinfo) {
                let srcloc = func.srclocs[inst];
                line_program.row().address_offset = offset as u64;
                if !srcloc.is_default() {
                    let source_info = *source_info_set.get_index(srcloc.bits() as usize).unwrap();
                    create_row_for_span(line_program, source_info.0);
                } else {
                    create_row_for_span(line_program, self.mir.span);
                }
                end = offset + size;
            }
        }

        line_program.end_sequence(end as u64);

        let entry = self.debug_context.dwarf.unit.get_mut(self.entry_id);
        entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(end as u64));

        {
            let value_labels_ranges = context.build_value_labels_ranges(isa).unwrap();

            for (value_label, value_loc_ranges) in value_labels_ranges.iter() {
                let live_ranges = RangeList(
                    Some(Range::BaseAddress {
                        address: Address::Symbol {
                            symbol: self.symbol,
                            addend: 0,
                        },
                    })
                    .into_iter()
                    .chain(
                        value_loc_ranges
                            .iter()
                            .map(|value_loc_range| Range::OffsetPair {
                                begin: u64::from(value_loc_range.start),
                                end: u64::from(value_loc_range.end),
                            }),
                    )
                    .collect(),
                );
                let live_ranges_id = self.debug_context.dwarf.unit.ranges.add(live_ranges);

                let local_ty = tcx.subst_and_normalize_erasing_regions(
                    self.instance.substs,
                    ty::ParamEnv::reveal_all(),
                    &self.mir.local_decls[mir::Local::from_u32(value_label.as_u32())].ty,
                );
                let local_type = self.debug_context.dwarf_ty(local_ty);

                let var_id = self
                    .debug_context
                    .dwarf
                    .unit
                    .add(self.entry_id, gimli::DW_TAG_variable);
                let var_entry = self.debug_context.dwarf.unit.get_mut(var_id);

                var_entry.set(
                    gimli::DW_AT_ranges,
                    AttributeValue::RangeListRef(live_ranges_id),
                );
                var_entry.set(
                    gimli::DW_AT_name,
                    AttributeValue::String(format!("{:?}", value_label).into_bytes()),
                );
                var_entry.set(
                    gimli::DW_AT_type,
                    AttributeValue::ThisUnitEntryRef(local_type),
                );


                let loc_list = LocationList(
                    Some(Location::BaseAddress {
                        address: Address::Symbol {
                            symbol: self.symbol,
                            addend: 0,
                        },
                    })
                    .into_iter()
                    .chain(
                        value_loc_ranges
                            .iter()
                            .map(|value_loc_range| Location::OffsetPair {
                                begin: u64::from(value_loc_range.start),
                                end: u64::from(value_loc_range.end),
                                data: Expression(translate_loc(value_loc_range.loc, &context.func.stack_slots).unwrap()),
                            }),
                    )
                    .collect(),
                );
                let loc_list_id = self.debug_context.dwarf.unit.locations.add(loc_list);

                let var_entry = self.debug_context.dwarf.unit.get_mut(var_id);
                var_entry.set(
                    gimli::DW_AT_location,
                    AttributeValue::LocationListRef(loc_list_id),
                );
            }
        }

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
    }
}

#[derive(Clone)]
struct WriterRelocate {
    relocs: Vec<DebugReloc>,
    writer: EndianVec<RunTimeEndian>,
}

impl WriterRelocate {
    fn new(ctx: &DebugContext) -> Self {
        WriterRelocate {
            relocs: Vec::new(),
            writer: EndianVec::new(ctx.endian),
        }
    }
}

impl Writer for WriterRelocate {
    type Endian = RunTimeEndian;

    fn endian(&self) -> Self::Endian {
        self.writer.endian()
    }

    fn len(&self) -> usize {
        self.writer.len()
    }

    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write(bytes)
    }

    fn write_at(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
        self.writer.write_at(offset, bytes)
    }

    fn write_address(&mut self, address: Address, size: u8) -> Result<()> {
        match address {
            Address::Constant(val) => self.write_udata(val, size),
            Address::Symbol { symbol, addend } => {
                let offset = self.len() as u64;
                self.relocs.push(DebugReloc {
                    offset: offset as u32,
                    size,
                    name: DebugRelocName::Symbol(symbol),
                    addend: addend as i64,
                });
                self.write_udata(0, size)
            }
        }
    }

    // TODO: implement write_eh_pointer

    fn write_offset(&mut self, val: usize, section: SectionId, size: u8) -> Result<()> {
        let offset = self.len() as u32;
        self.relocs.push(DebugReloc {
            offset,
            size,
            name: DebugRelocName::Section(section),
            addend: val as i64,
        });
        self.write_udata(0, size)
    }

    fn write_offset_at(
        &mut self,
        offset: usize,
        val: usize,
        section: SectionId,
        size: u8,
    ) -> Result<()> {
        self.relocs.push(DebugReloc {
            offset: offset as u32,
            size,
            name: DebugRelocName::Section(section),
            addend: val as i64,
        });
        self.write_udata_at(offset, 0, size)
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
                writer
                    .write_u8(gimli::constants::DW_OP_deref.0 as u8)
                    .expect("bp wr");
                let buf = writer.into_vec();
                return Some(buf);
            }
            None
        }
        _ => None,
    }
}
