extern crate gimli;

use crate::prelude::*;

use std::marker::PhantomData;

use syntax::source_map::FileName;

use gimli::write::{
    Address, AttributeValue, CompilationUnit, DebugAbbrev, DebugInfo, DebugLine, DebugRanges,
    DebugRngLists, DebugStr, EndianVec, LineProgram, LineProgramId, LineProgramTable, Range,
    RangeList, RangeListTable, Result, SectionId, StringTable, UnitEntryId, UnitId, UnitTable,
    Writer, FileId,
};
use gimli::{Encoding, Format, RunTimeEndian};

use faerie::*;

fn target_endian(tcx: TyCtxt) -> RunTimeEndian {
    use rustc::ty::layout::Endian;

    match tcx.data_layout.endian {
        Endian::Big => RunTimeEndian::Big,
        Endian::Little => RunTimeEndian::Little,
    }
}

fn line_program_add_file(line_program: &mut LineProgram, file: &FileName) -> FileId {
    match file {
        FileName::Real(path) => {
            let dir_id =
                line_program.add_directory(path.parent().unwrap().to_str().unwrap().as_bytes());
            line_program.add_file(
                path.file_name().unwrap().to_str().unwrap().as_bytes(),
                dir_id,
                None,
            )
        }
        // FIXME give more appropriate file names
        _ => {
            let dir_id = line_program.default_directory();
            line_program.add_file(
                file.to_string().as_bytes(),
                dir_id,
                None,
            )
        }
    }
}

struct DebugReloc {
    offset: u32,
    size: u8,
    name: String,
    addend: i64,
}

pub struct DebugContext<'tcx> {
    // Encoding info
    encoding: Encoding,
    endian: RunTimeEndian,
    symbols: indexmap::IndexSet<String>,

    // Main data
    units: UnitTable,
    line_programs: LineProgramTable,

    // Side tables
    strings: StringTable,
    range_lists: RangeListTable,

    // Global ids
    unit_id: UnitId,
    global_line_program: LineProgramId,
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

        // FIXME: how to get version when building out of tree?
        // Normally this would use option_env!("CFG_VERSION").
        let producer = format!("cranelift fn (rustc version {})", "unknown version");
        let comp_dir = tcx.sess.working_dir.0.to_string_lossy().into_owned();
        let name = match tcx.sess.local_crate_source_file {
            Some(ref path) => path.to_string_lossy().into_owned(),
            None => tcx.crate_name(LOCAL_CRATE).to_string(),
        };

        let mut strings = StringTable::default();
        let range_lists = RangeListTable::default();

        let mut units = UnitTable::default();
        let mut line_programs = LineProgramTable::default();

        let global_line_program = line_programs.add(LineProgram::new(
            encoding,
            1,
            1,
            -5,
            14,
            comp_dir.as_bytes(),
            name.as_bytes(),
            None,
        ));

        let unit_id = units.add(CompilationUnit::new(encoding));
        {
            let name = strings.add(&*name);
            let comp_dir = strings.add(&*comp_dir);

            let unit = units.get_mut(unit_id);
            let root = unit.root();
            let root = unit.get_mut(root);
            root.set(
                gimli::DW_AT_producer,
                AttributeValue::StringRef(strings.add(producer)),
            );
            root.set(
                gimli::DW_AT_language,
                AttributeValue::Language(gimli::DW_LANG_Rust),
            );
            root.set(gimli::DW_AT_name, AttributeValue::StringRef(name));
            root.set(gimli::DW_AT_comp_dir, AttributeValue::StringRef(comp_dir));
            root.set(
                gimli::DW_AT_stmt_list,
                AttributeValue::LineProgramRef(global_line_program),
            );
            root.set(
                gimli::DW_AT_low_pc,
                AttributeValue::Address(Address::Absolute(0)),
            );
        }

        DebugContext {
            encoding,
            endian: target_endian(tcx),
            symbols: indexmap::IndexSet::new(),

            strings,
            range_lists,

            units,
            line_programs,

            unit_id,
            global_line_program,
            unit_range_list: RangeList(Vec::new()),

            _dummy: PhantomData,
        }
    }

    fn emit_location(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, entry_id: UnitEntryId, span: Span) {
        let loc = tcx.sess.source_map().lookup_char_pos(span.lo());

        let line_program = self.line_programs.get_mut(self.global_line_program);
        let file_id = line_program_add_file(line_program, &loc.file.name);

        let unit = self.units.get_mut(self.unit_id);
        let entry = unit.get_mut(entry_id);

        entry.set(gimli::DW_AT_decl_file, AttributeValue::FileIndex(file_id));
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
        let unit_range_list_id = self.range_lists.add(self.unit_range_list.clone());
        let unit = self.units.get_mut(self.unit_id);
        let root = unit.root();
        let root = unit.get_mut(root);
        root.set(
            gimli::DW_AT_ranges,
            AttributeValue::RangeListRef(unit_range_list_id),
        );

        let mut debug_abbrev = DebugAbbrev::from(WriterRelocate::new(self));
        let mut debug_info = DebugInfo::from(WriterRelocate::new(self));
        let mut debug_str = DebugStr::from(WriterRelocate::new(self));
        let mut debug_line = DebugLine::from(WriterRelocate::new(self));
        let mut debug_ranges = DebugRanges::from(WriterRelocate::new(self));
        let mut debug_rnglists = DebugRngLists::from(WriterRelocate::new(self));

        let debug_str_offsets = self.strings.write(&mut debug_str).unwrap();

        let debug_line_offsets = self.line_programs.write(&mut debug_line).unwrap();

        let range_list_offsets = self
            .range_lists
            .write(
                &mut debug_ranges,
                &mut debug_rnglists,
                self.encoding
            )
            .unwrap();
        self.units
            .write(
                &mut debug_abbrev,
                &mut debug_info,
                &debug_line_offsets,
                &range_list_offsets,
                &debug_str_offsets,
            )
            .unwrap();

        macro decl_section($section:ident = $name:ident) {
            artifact
                .declare_with(
                    SectionId::$section.name(),
                    Decl::DebugSection,
                    $name.0.writer.into_vec(),
                )
                .unwrap();
        }

        decl_section!(DebugAbbrev = debug_abbrev);
        decl_section!(DebugInfo = debug_info);
        decl_section!(DebugStr = debug_str);
        decl_section!(DebugLine = debug_line);

        let debug_ranges_not_empty = !debug_ranges.0.writer.slice().is_empty();
        if debug_ranges_not_empty {
            decl_section!(DebugRanges = debug_ranges);
        }

        let debug_rnglists_not_empty = !debug_rnglists.0.writer.slice().is_empty();
        if debug_rnglists_not_empty {
            decl_section!(DebugRngLists = debug_rnglists);
        }

        macro sect_relocs($section:ident = $name:ident) {
            for reloc in $name.0.relocs {
                artifact
                    .link_with(
                        faerie::Link {
                            from: SectionId::$section.name(),
                            to: &reloc.name,
                            at: u64::from(reloc.offset),
                        },
                        faerie::Reloc::Debug {
                            size: reloc.size,
                            addend: reloc.addend as i32,
                        },
                    )
                    .expect("faerie relocation error");
            }
        }

        sect_relocs!(DebugAbbrev = debug_abbrev);
        sect_relocs!(DebugInfo = debug_info);
        sect_relocs!(DebugStr = debug_str);
        sect_relocs!(DebugLine = debug_line);

        if debug_ranges_not_empty {
            sect_relocs!(DebugRanges = debug_ranges);
        }

        if debug_rnglists_not_empty {
            sect_relocs!(DebugRngLists = debug_rnglists);
        }
    }

    fn section_name(&self, id: SectionId) -> String {
        id.name().to_string()
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
        mir: &Mir,
        name: &str,
        _sig: &Signature,
    ) -> Self {
        let (symbol, _) = debug_context.symbols.insert_full(name.to_string());

        let unit = debug_context.units.get_mut(debug_context.unit_id);
        // FIXME: add to appropriate scope intead of root
        let scope = unit.root();

        let entry_id = unit.add(scope, gimli::DW_TAG_subprogram);
        let entry = unit.get_mut(entry_id);
        let name_id = debug_context.strings.add(name);
        entry.set(
            gimli::DW_AT_linkage_name,
            AttributeValue::StringRef(name_id),
        );

        entry.set(
            gimli::DW_AT_low_pc,
            AttributeValue::Address(Address::Relative { symbol, addend: 0 }),
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
        //module: &mut Module<impl Backend>,
        code_size: u32,
        context: &Context,
        isa: &cranelift::codegen::isa::TargetIsa,
        source_info_set: &indexmap::IndexSet<SourceInfo>,
    ) {
        let unit = self.debug_context.units.get_mut(self.debug_context.unit_id);
        let entry = unit.get_mut(self.entry_id);
        entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(code_size as u64));

        self.debug_context.unit_range_list.0.push(Range {
            begin: Address::Relative {
                symbol: self.symbol,
                addend: 0,
            },
            end: Address::Relative {
                symbol: self.symbol,
                addend: code_size as i64,
            },
        });

        let line_program = self
            .debug_context
            .line_programs
            .get_mut(self.debug_context.global_line_program);

        line_program.begin_sequence(Some(Address::Relative {
            symbol: self.symbol,
            addend: 0,
        }));

        let encinfo = isa.encoding_info();
        let func = &context.func;
        let mut ebbs = func.layout.ebbs().collect::<Vec<_>>();
        ebbs.sort_by_key(|ebb| func.offsets[*ebb]); // Ensure inst offsets always increase

        let create_row_for_span = |line_program: &mut LineProgram, span: Span| {
            let loc = tcx.sess.source_map().lookup_char_pos(span.lo());
            let file_id = line_program_add_file(line_program, &loc.file.name);

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

        if code_size != end {
            line_program.row().address_offset = end as u64;
            create_row_for_span(line_program, self.mir_span);
        }

        line_program.end_sequence(code_size as u64);
    }
}

struct WriterRelocate<'a, 'tcx> {
    ctx: &'a DebugContext<'tcx>,
    relocs: Vec<DebugReloc>,
    writer: EndianVec<RunTimeEndian>,
}

impl<'a, 'tcx> WriterRelocate<'a, 'tcx> {
    fn new(ctx: &'a DebugContext<'tcx>) -> Self {
        WriterRelocate {
            ctx,
            relocs: Vec::new(),
            writer: EndianVec::new(ctx.endian),
        }
    }
}

impl<'a, 'tcx> Writer for WriterRelocate<'a, 'tcx> {
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
            Address::Absolute(val) => self.write_word(val, size),
            Address::Relative { symbol, addend } => {
                let offset = self.len() as u64;
                self.relocs.push(DebugReloc {
                    offset: offset as u32,
                    size,
                    name: self.ctx.symbols.get_index(symbol).unwrap().clone(),
                    addend: addend as i64,
                });
                self.write_word(0, size)
            }
        }
    }

    fn write_offset(&mut self, val: usize, section: SectionId, size: u8) -> Result<()> {
        let offset = self.len() as u32;
        let name = self.ctx.section_name(section);
        self.relocs.push(DebugReloc {
            offset,
            size,
            name,
            addend: val as i64,
        });
        self.write_word(0, size)
    }

    fn write_offset_at(
        &mut self,
        offset: usize,
        val: usize,
        section: SectionId,
        size: u8,
    ) -> Result<()> {
        let name = self.ctx.section_name(section);
        self.relocs.push(DebugReloc {
            offset: offset as u32,
            size,
            name,
            addend: val as i64,
        });
        self.write_word_at(offset, 0, size)
    }
}
