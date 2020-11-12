//! Line info generation (`.debug_line`)

use std::ffi::OsStr;
use std::path::{Component, Path};

use crate::prelude::*;

use rustc_span::{
    FileName, Pos, SourceFile, SourceFileAndLine, SourceFileHash, SourceFileHashAlgorithm,
};

use cranelift_codegen::binemit::CodeOffset;
use cranelift_codegen::machinst::MachSrcLoc;

use gimli::write::{
    Address, AttributeValue, FileId, FileInfo, LineProgram, LineString, LineStringTable,
    UnitEntryId,
};

// OPTIMIZATION: It is cheaper to do this in one pass than using `.parent()` and `.file_name()`.
fn split_path_dir_and_file(path: &Path) -> (&Path, &OsStr) {
    let mut iter = path.components();
    let file_name = match iter.next_back() {
        Some(Component::Normal(p)) => p,
        component => {
            panic!(
                "Path component {:?} of path {} is an invalid filename",
                component,
                path.display()
            );
        }
    };
    let parent = iter.as_path();
    (parent, file_name)
}

// OPTIMIZATION: Avoid UTF-8 validation on UNIX.
fn osstr_as_utf8_bytes(path: &OsStr) -> &[u8] {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        return path.as_bytes();
    }
    #[cfg(not(unix))]
    {
        return path.to_str().unwrap().as_bytes();
    }
}

pub(crate) const MD5_LEN: usize = 16;

pub(crate) fn make_file_info(hash: SourceFileHash) -> Option<FileInfo> {
    if hash.kind == SourceFileHashAlgorithm::Md5 {
        let mut buf = [0u8; MD5_LEN];
        buf.copy_from_slice(hash.hash_bytes());
        Some(FileInfo {
            timestamp: 0,
            size: 0,
            md5: buf,
        })
    } else {
        None
    }
}

fn line_program_add_file(
    line_program: &mut LineProgram,
    line_strings: &mut LineStringTable,
    file: &SourceFile,
) -> FileId {
    match &file.name {
        FileName::Real(path) => {
            let (dir_path, file_name) = split_path_dir_and_file(path.stable_name());
            let dir_name = osstr_as_utf8_bytes(dir_path.as_os_str());
            let file_name = osstr_as_utf8_bytes(file_name);

            let dir_id = if !dir_name.is_empty() {
                let dir_name = LineString::new(dir_name, line_program.encoding(), line_strings);
                line_program.add_directory(dir_name)
            } else {
                line_program.default_directory()
            };
            let file_name = LineString::new(file_name, line_program.encoding(), line_strings);

            let info = make_file_info(file.src_hash);

            line_program.file_has_md5 &= info.is_some();
            line_program.add_file(file_name, dir_id, info)
        }
        // FIXME give more appropriate file names
        filename => {
            let dir_id = line_program.default_directory();
            let dummy_file_name = LineString::new(
                filename.to_string().into_bytes(),
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
            &loc.file,
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

    pub(super) fn create_debug_lines(
        &mut self,
        isa: &dyn cranelift_codegen::isa::TargetIsa,
        symbol: usize,
        entry_id: UnitEntryId,
        context: &Context,
        function_span: Span,
        source_info_set: &indexmap::IndexSet<SourceInfo>,
    ) -> CodeOffset {
        let tcx = self.tcx;
        let line_program = &mut self.dwarf.unit.line_program;
        let func = &context.func;

        let line_strings = &mut self.dwarf.line_strings;
        let mut last_span = None;
        let mut last_file = None;
        let mut create_row_for_span = |line_program: &mut LineProgram, span: Span| {
            if let Some(last_span) = last_span {
                if span == last_span {
                    line_program.generate_row();
                    return;
                }
            }
            last_span = Some(span);

            // Based on https://github.com/rust-lang/rust/blob/e369d87b015a84653343032833d65d0545fd3f26/src/librustc_codegen_ssa/mir/mod.rs#L116-L131
            // In order to have a good line stepping behavior in debugger, we overwrite debug
            // locations of macro expansions with that of the outermost expansion site
            // (unless the crate is being compiled with `-Z debug-macros`).
            let span = if !span.from_expansion() || tcx.sess.opts.debugging_opts.debug_macros {
                span
            } else {
                // Walk up the macro expansion chain until we reach a non-expanded span.
                // We also stop at the function body level because no line stepping can occur
                // at the level above that.
                rustc_span::hygiene::walk_chain(span, function_span.ctxt())
            };

            let (file, line, col) = match tcx.sess.source_map().lookup_line(span.lo()) {
                Ok(SourceFileAndLine { sf: file, line }) => {
                    let line_pos = file.line_begin_pos(span.lo());

                    (
                        file,
                        u64::try_from(line).unwrap() + 1,
                        u64::from((span.lo() - line_pos).to_u32()) + 1,
                    )
                }
                Err(file) => (file, 0, 0),
            };

            // line_program_add_file is very slow.
            // Optimize for the common case of the current file not being changed.
            let current_file_changed = if let Some(last_file) = &last_file {
                // If the allocations are not equal, then the files may still be equal, but that
                // is not a problem, as this is just an optimization.
                !rustc_data_structures::sync::Lrc::ptr_eq(last_file, &file)
            } else {
                true
            };
            if current_file_changed {
                let file_id = line_program_add_file(line_program, line_strings, &file);
                line_program.row().file = file_id;
                last_file = Some(file);
            }

            line_program.row().line = line;
            line_program.row().column = col;
            line_program.generate_row();
        };

        line_program.begin_sequence(Some(Address::Symbol { symbol, addend: 0 }));

        let mut func_end = 0;

        if let Some(ref mcr) = &context.mach_compile_result {
            for &MachSrcLoc { start, end, loc } in mcr.buffer.get_srclocs_sorted() {
                line_program.row().address_offset = u64::from(start);
                if !loc.is_default() {
                    let source_info = *source_info_set.get_index(loc.bits() as usize).unwrap();
                    create_row_for_span(line_program, source_info.span);
                } else {
                    create_row_for_span(line_program, function_span);
                }
                func_end = end;
            }

            line_program.end_sequence(u64::from(func_end));

            func_end = mcr.buffer.total_size();
        } else {
            let encinfo = isa.encoding_info();
            let mut blocks = func.layout.blocks().collect::<Vec<_>>();
            blocks.sort_by_key(|block| func.offsets[*block]); // Ensure inst offsets always increase

            for block in blocks {
                for (offset, inst, size) in func.inst_offsets(block, &encinfo) {
                    let srcloc = func.srclocs[inst];
                    line_program.row().address_offset = u64::from(offset);
                    if !srcloc.is_default() {
                        let source_info =
                            *source_info_set.get_index(srcloc.bits() as usize).unwrap();
                        create_row_for_span(line_program, source_info.span);
                    } else {
                        create_row_for_span(line_program, function_span);
                    }
                    func_end = offset + size;
                }
            }
            line_program.end_sequence(u64::from(func_end));
        }

        assert_ne!(func_end, 0);

        let entry = self.dwarf.unit.get_mut(entry_id);
        entry.set(
            gimli::DW_AT_low_pc,
            AttributeValue::Address(Address::Symbol { symbol, addend: 0 }),
        );
        entry.set(
            gimli::DW_AT_high_pc,
            AttributeValue::Udata(u64::from(func_end)),
        );

        self.emit_location(entry_id, function_span);

        func_end
    }
}
