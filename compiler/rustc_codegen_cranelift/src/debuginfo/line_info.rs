//! Line info generation (`.debug_line`)

use std::ffi::OsStr;
use std::path::{Component, Path};

use crate::debuginfo::FunctionDebugContext;
use crate::prelude::*;

use rustc_data_structures::sync::Lrc;
use rustc_span::{
    FileName, Pos, SourceFile, SourceFileAndLine, SourceFileHash, SourceFileHashAlgorithm,
};

use cranelift_codegen::binemit::CodeOffset;
use cranelift_codegen::MachSrcLoc;

use gimli::write::{
    Address, AttributeValue, FileId, FileInfo, LineProgram, LineString, LineStringTable,
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
        path.as_bytes()
    }
    #[cfg(not(unix))]
    {
        path.to_str().unwrap().as_bytes()
    }
}

const MD5_LEN: usize = 16;

fn make_file_info(hash: SourceFileHash) -> Option<FileInfo> {
    if hash.kind == SourceFileHashAlgorithm::Md5 {
        let mut buf = [0u8; MD5_LEN];
        buf.copy_from_slice(hash.hash_bytes());
        Some(FileInfo { timestamp: 0, size: 0, md5: buf })
    } else {
        None
    }
}

impl DebugContext {
    pub(crate) fn get_span_loc(
        tcx: TyCtxt<'_>,
        function_span: Span,
        span: Span,
    ) -> (Lrc<SourceFile>, u64, u64) {
        // Based on https://github.com/rust-lang/rust/blob/e369d87b015a84653343032833d65d0545fd3f26/src/librustc_codegen_ssa/mir/mod.rs#L116-L131
        // In order to have a good line stepping behavior in debugger, we overwrite debug
        // locations of macro expansions with that of the outermost expansion site
        // (unless the crate is being compiled with `-Z debug-macros`).
        let span = if !span.from_expansion() || tcx.sess.opts.unstable_opts.debug_macros {
            span
        } else {
            // Walk up the macro expansion chain until we reach a non-expanded span.
            // We also stop at the function body level because no line stepping can occur
            // at the level above that.
            rustc_span::hygiene::walk_chain(span, function_span.ctxt())
        };

        match tcx.sess.source_map().lookup_line(span.lo()) {
            Ok(SourceFileAndLine { sf: file, line }) => {
                let line_pos = file.line_begin_pos(span.lo());

                (
                    file,
                    u64::try_from(line).unwrap() + 1,
                    u64::from((span.lo() - line_pos).to_u32()) + 1,
                )
            }
            Err(file) => (file, 0, 0),
        }
    }

    pub(crate) fn add_source_file(&mut self, source_file: &SourceFile) -> FileId {
        let line_program: &mut LineProgram = &mut self.dwarf.unit.line_program;
        let line_strings: &mut LineStringTable = &mut self.dwarf.line_strings;

        match &source_file.name {
            FileName::Real(path) => {
                let (dir_path, file_name) =
                    split_path_dir_and_file(path.remapped_path_if_available());
                let dir_name = osstr_as_utf8_bytes(dir_path.as_os_str());
                let file_name = osstr_as_utf8_bytes(file_name);

                let dir_id = if !dir_name.is_empty() {
                    let dir_name = LineString::new(dir_name, line_program.encoding(), line_strings);
                    line_program.add_directory(dir_name)
                } else {
                    line_program.default_directory()
                };
                let file_name = LineString::new(file_name, line_program.encoding(), line_strings);

                let info = make_file_info(source_file.src_hash);

                line_program.file_has_md5 &= info.is_some();
                line_program.add_file(file_name, dir_id, info)
            }
            // FIXME give more appropriate file names
            filename => {
                let dir_id = line_program.default_directory();
                let dummy_file_name = LineString::new(
                    filename.prefer_remapped().to_string().into_bytes(),
                    line_program.encoding(),
                    line_strings,
                );
                line_program.add_file(dummy_file_name, dir_id, None)
            }
        }
    }
}

impl FunctionDebugContext {
    pub(crate) fn add_dbg_loc(&mut self, file_id: FileId, line: u64, column: u64) -> SourceLoc {
        let (index, _) = self.source_loc_set.insert_full((file_id, line, column));
        SourceLoc::new(u32::try_from(index).unwrap())
    }

    pub(super) fn create_debug_lines(
        &mut self,
        debug_context: &mut DebugContext,
        symbol: usize,
        context: &Context,
    ) -> CodeOffset {
        let create_row_for_span =
            |debug_context: &mut DebugContext, source_loc: (FileId, u64, u64)| {
                let (file_id, line, col) = source_loc;

                debug_context.dwarf.unit.line_program.row().file = file_id;
                debug_context.dwarf.unit.line_program.row().line = line;
                debug_context.dwarf.unit.line_program.row().column = col;
                debug_context.dwarf.unit.line_program.generate_row();
            };

        debug_context
            .dwarf
            .unit
            .line_program
            .begin_sequence(Some(Address::Symbol { symbol, addend: 0 }));

        let mut func_end = 0;

        let mcr = context.compiled_code().unwrap();
        for &MachSrcLoc { start, end, loc } in mcr.buffer.get_srclocs_sorted() {
            debug_context.dwarf.unit.line_program.row().address_offset = u64::from(start);
            if !loc.is_default() {
                let source_loc = *self.source_loc_set.get_index(loc.bits() as usize).unwrap();
                create_row_for_span(debug_context, source_loc);
            } else {
                create_row_for_span(debug_context, self.function_source_loc);
            }
            func_end = end;
        }

        debug_context.dwarf.unit.line_program.end_sequence(u64::from(func_end));

        let func_end = mcr.buffer.total_size();

        assert_ne!(func_end, 0);

        let entry = debug_context.dwarf.unit.get_mut(self.entry_id);
        entry.set(
            gimli::DW_AT_low_pc,
            AttributeValue::Address(Address::Symbol { symbol, addend: 0 }),
        );
        entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(u64::from(func_end)));

        func_end
    }
}
