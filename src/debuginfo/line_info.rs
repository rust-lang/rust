use crate::prelude::*;

use syntax::source_map::FileName;

use cranelift::codegen::binemit::CodeOffset;

use gimli::write::{
    Address, AttributeValue, FileId, LineProgram, LineString, LineStringTable, Range, UnitEntryId,
};

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

impl<'tcx> DebugContext<'tcx> {
    pub(super) fn emit_location(&mut self, entry_id: UnitEntryId, span: Span) {
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
}

impl<'a, 'tcx> FunctionDebugContext<'a, 'tcx> {
    pub(crate) fn create_debug_lines(
        &mut self,
        context: &Context,
        isa: &dyn cranelift::codegen::isa::TargetIsa,
        source_info_set: &indexmap::IndexSet<(Span, mir::SourceScope)>,
    ) -> CodeOffset {
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
        entry.set(
            gimli::DW_AT_low_pc,
            AttributeValue::Address(Address::Symbol {
                symbol: self.symbol,
                addend: 0,
            }),
        );
        entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(end as u64));

        self.debug_context
            .emit_location(self.entry_id, self.mir.span);

        end
    }
}
