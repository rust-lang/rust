use crate::prelude::*;

use std::marker::PhantomData;

use syntax::source_map::FileName;

use gimli::write::{
    Address, AttributeValue, DwarfUnit, EndianVec, FileId, LineProgram, LineString,
    LineStringTable, Range, RangeList, Result, Sections, UnitEntryId, Writer,
};
use gimli::{Encoding, Format, LineEncoding, RunTimeEndian, SectionId};

use faerie::*;

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
            let dir_name = LineString::new(
                path.parent().unwrap().to_str().unwrap().as_bytes(),
                line_program.encoding(),
                line_strings,
            );
            let dir_id = line_program.add_directory(dir_name);
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
struct DebugReloc {
    offset: u32,
    size: u8,
    name: DebugRelocName,
    addend: i64,
}

#[derive(Clone)]
enum DebugRelocName {
    Section(SectionId),
    Symbol(usize),
}

impl DebugReloc {
    fn name<'a>(&self, ctx: &'a DebugContext) -> &'a str {
        match self.name {
            DebugRelocName::Section(id) => id.name(),
            DebugRelocName::Symbol(index) => ctx.symbols.get_index(index).unwrap(),
        }
    }
}

pub struct DebugContext<'tcx> {
    endian: RunTimeEndian,
    symbols: indexmap::IndexSet<String>,

    dwarf: DwarfUnit,
    unit_range_list: RangeList,

    _dummy: PhantomData<&'tcx ()>,
}

impl<'a, 'tcx: 'a> DebugContext<'tcx> {
    pub fn new(tcx: TyCtxt, address_size: u8) -> Self {
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
            endian: target_endian(tcx),
            symbols: indexmap::IndexSet::new(),

            dwarf,
            unit_range_list: RangeList(Vec::new()),

            _dummy: PhantomData,
        }
    }

    fn emit_location(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, entry_id: UnitEntryId, span: Span) {
        let loc = tcx.sess.source_map().lookup_char_pos(span.lo());

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

    pub fn emit(&mut self, artifact: &mut Artifact) {
        let unit_range_list_id = self.dwarf.unit.ranges.add(self.unit_range_list.clone());
        let root = self.dwarf.unit.root();
        let root = self.dwarf.unit.get_mut(root);
        root.set(
            gimli::DW_AT_ranges,
            AttributeValue::RangeListRef(unit_range_list_id),
        );

        let mut sections = Sections::new(WriterRelocate::new(self));
        self.dwarf.write(&mut sections).unwrap();

        let _: Result<()> = sections.for_each_mut(|id, section| {
            if !section.writer.slice().is_empty() {
                artifact
                    .declare_with(id.name(), Decl::section(SectionKind::Debug), section.writer.take())
                    .unwrap();
            }
            Ok(())
        });

        let _: Result<()> = sections.for_each(|id, section| {
            for reloc in &section.relocs {
                artifact
                    .link_with(
                        faerie::Link {
                            from: id.name(),
                            to: reloc.name(self),
                            at: u64::from(reloc.offset),
                        },
                        faerie::Reloc::Debug {
                            size: reloc.size,
                            addend: reloc.addend as i32,
                        },
                    )
                    .expect("faerie relocation error");
            }
            Ok(())
        });
    }
}

pub struct FunctionDebugContext<'a, 'tcx> {
    debug_context: &'a mut DebugContext<'tcx>,
    entry_id: UnitEntryId,
    symbol: usize,
    mir_span: Span,
}

impl<'a, 'b, 'tcx: 'b> FunctionDebugContext<'a, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'b, 'tcx, 'tcx>,
        debug_context: &'a mut DebugContext<'tcx>,
        mir: &Body,
        name: &str,
        _sig: &Signature,
    ) -> Self {
        let (symbol, _) = debug_context.symbols.insert_full(name.to_string());

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

        debug_context.emit_location(tcx, entry_id, mir.span);

        FunctionDebugContext {
            debug_context,
            entry_id,
            symbol,
            mir_span: mir.span,
        }
    }

    pub fn define(
        &mut self,
        tcx: TyCtxt,
        context: &Context,
        isa: &dyn cranelift::codegen::isa::TargetIsa,
        source_info_set: &indexmap::IndexSet<SourceInfo>,
    ) {
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
                    create_row_for_span(line_program, source_info.span);
                } else {
                    create_row_for_span(line_program, self.mir_span);
                }
                end = offset + size;
            }
        }

        line_program.end_sequence(end as u64);

        let entry = self.debug_context.dwarf.unit.get_mut(self.entry_id);
        entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(end as u64));

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
