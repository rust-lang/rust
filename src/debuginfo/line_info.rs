use std::ffi::OsStr;
use std::path::{Component, Path};

use crate::prelude::*;

use rustc_span::FileName;

use cranelift_codegen::binemit::CodeOffset;

use gimli::write::{
    Address, AttributeValue, FileId, LineProgram, LineString, LineStringTable, UnitEntryId,
};

// OPTIMIZATION: It is cheaper to do this in one pass than using `.parent()` and `.file_name()`.
fn split_path_dir_and_file(path: &Path) -> (&Path, &OsStr) {
    let mut iter = path.components();
    let file_name = match iter.next_back() {
        Some(Component::Normal(p)) => p,
        component => {
            panic!("Path component {:?} of path {} is an invalid filename", component, path.display());
        }
    };
    let parent = iter.as_path();
    (parent, file_name)
}

// OPTIMIZATION: Avoid UTF-8 validation on UNIX.
fn osstr_as_utf8_bytes(path: &OsStr) -> &[u8] {
    #[cfg(unix)] {
        use std::os::unix::ffi::OsStrExt;
        return path.as_bytes();
    }
    #[cfg(not(unix))] {
        return path.to_str().unwrap().as_bytes();
    }
}

fn line_program_add_file(
    line_program: &mut LineProgram,
    line_strings: &mut LineStringTable,
    file: &FileName,
) -> FileId {
    match file {
        FileName::Real(path) => {
            let (dir_path, file_name) = split_path_dir_and_file(path);
            let dir_name = osstr_as_utf8_bytes(dir_path.as_os_str());
            let file_name = osstr_as_utf8_bytes(file_name);

            let dir_id = if !dir_name.is_empty() {
                let dir_name = LineString::new(dir_name, line_program.encoding(), line_strings);
                line_program.add_directory(dir_name)
            } else {
                line_program.default_directory()
            };
            let file_name = LineString::new(
                file_name,
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
    pub(super) fn create_debug_lines(
        &mut self,
        context: &Context,
        isa: &dyn cranelift_codegen::isa::TargetIsa,
        source_info_set: &indexmap::IndexSet<SourceInfo>,
    ) -> CodeOffset {
        let tcx = self.debug_context.tcx;

        let line_program = &mut self.debug_context.dwarf.unit.line_program;

        line_program.begin_sequence(Some(Address::Symbol {
            symbol: self.symbol,
            addend: 0,
        }));

        let encinfo = isa.encoding_info();
        let func = &context.func;
        let mut blocks = func.layout.blocks().collect::<Vec<_>>();
        blocks.sort_by_key(|block| func.offsets[*block]); // Ensure inst offsets always increase

        let line_strings = &mut self.debug_context.dwarf.line_strings;
        let function_span = self.mir.span;
        let mut last_file = None;
        let mut create_row_for_span = |line_program: &mut LineProgram, span: Span| {
            // Based on https://github.com/rust-lang/rust/blob/e369d87b015a84653343032833d65d0545fd3f26/src/librustc_codegen_ssa/mir/mod.rs#L116-L131
            // In order to have a good line stepping behavior in debugger, we overwrite debug
            // locations of macro expansions with that of the outermost expansion site
            // (unless the crate is being compiled with `-Z debug-macros`).
            let span = if !span.from_expansion() ||
                tcx.sess.opts.debugging_opts.debug_macros {
                span
            } else {
                // Walk up the macro expansion chain until we reach a non-expanded span.
                // We also stop at the function body level because no line stepping can occur
                // at the level above that.
                rustc_span::hygiene::walk_chain(span, function_span.ctxt())
            };

            let loc = tcx.sess.source_map().lookup_char_pos(span.lo());

            // line_program_add_file is very slow.
            // Optimize for the common case of the current file not being changed.
            let current_file_changed = if let Some(last_file) = &mut last_file {
                // If the allocations are not equal, then the files may still be equal, but that
                // is not a problem, as this is just an optimization.
                !Lrc::ptr_eq(last_file, &loc.file)
            } else {
                true
            };
            if current_file_changed {
                let file_id = line_program_add_file(line_program, line_strings, &loc.file.name);
                line_program.row().file = file_id;
                last_file = Some(loc.file.clone());
            }

            line_program.row().line = loc.line as u64;
            line_program.row().column = loc.col.to_u32() as u64 + 1;
            line_program.generate_row();
        };

        let mut end = 0;
        for block in blocks {
            for (offset, inst, size) in func.inst_offsets(block, &encinfo) {
                let srcloc = func.srclocs[inst];
                line_program.row().address_offset = offset as u64;
                if !srcloc.is_default() {
                    let source_info = *source_info_set.get_index(srcloc.bits() as usize).unwrap();
                    create_row_for_span(line_program, source_info.span);
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
