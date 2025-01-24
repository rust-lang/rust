//! Line info generation (`.debug_line`)

use std::ffi::OsStr;
use std::path::{Component, Path};

use cranelift_codegen::MachSrcLoc;
use cranelift_codegen::binemit::CodeOffset;
use gimli::write::{AttributeValue, FileId, FileInfo, LineProgram, LineString, LineStringTable};
use rustc_span::{
    FileName, Pos, SourceFile, SourceFileAndLine, SourceFileHash, SourceFileHashAlgorithm, hygiene,
};

use crate::debuginfo::FunctionDebugContext;
use crate::debuginfo::emit::address_for_func;
use crate::prelude::*;

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
        Some(FileInfo {
            timestamp: 0,
            size: 0,
            md5: buf,
            source: None, // FIXME implement -Zembed-source
        })
    } else {
        None
    }
}

impl DebugContext {
    pub(crate) fn get_span_loc(
        &mut self,
        tcx: TyCtxt<'_>,
        function_span: Span,
        span: Span,
    ) -> (FileId, u64, u64) {
        // Match behavior of `FunctionCx::adjusted_span_and_dbg_scope`.
        let span = hygiene::walk_chain_collapsed(span, function_span);
        match tcx.sess.source_map().lookup_line(span.lo()) {
            Ok(SourceFileAndLine { sf: file, line }) => {
                let file_id = self.add_source_file(&file);
                let line_pos = file.lines()[line];
                let col = file.relative_position(span.lo()) - line_pos;

                (file_id, u64::try_from(line).unwrap() + 1, u64::from(col.to_u32()) + 1)
            }
            Err(file) => (self.add_source_file(&file), 0, 0),
        }
    }

    pub(crate) fn add_source_file(&mut self, source_file: &SourceFile) -> FileId {
        let cache_key = (source_file.stable_id, source_file.src_hash);
        *self.created_files.entry(cache_key).or_insert_with(|| {
            let line_program: &mut LineProgram = &mut self.dwarf.unit.line_program;
            let line_strings: &mut LineStringTable = &mut self.dwarf.line_strings;

            match &source_file.name {
                FileName::Real(path) => {
                    let (dir_path, file_name) =
                        split_path_dir_and_file(path.to_path(self.filename_display_preference));
                    let dir_name = osstr_as_utf8_bytes(dir_path.as_os_str());
                    let file_name = osstr_as_utf8_bytes(file_name);

                    let dir_id = if !dir_name.is_empty() {
                        let dir_name =
                            LineString::new(dir_name, line_program.encoding(), line_strings);
                        line_program.add_directory(dir_name)
                    } else {
                        line_program.default_directory()
                    };
                    let file_name =
                        LineString::new(file_name, line_program.encoding(), line_strings);

                    let info = make_file_info(source_file.src_hash);

                    line_program.file_has_md5 &= info.is_some();
                    line_program.add_file(file_name, dir_id, info)
                }
                filename => {
                    let dir_id = line_program.default_directory();
                    let dummy_file_name = LineString::new(
                        filename.display(self.filename_display_preference).to_string().into_bytes(),
                        line_program.encoding(),
                        line_strings,
                    );
                    line_program.add_file(dummy_file_name, dir_id, None)
                }
            }
        })
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
        func_id: FuncId,
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

        debug_context.dwarf.unit.line_program.begin_sequence(Some(address_for_func(func_id)));

        let mut func_end = 0;

        let mcr = context.compiled_code().unwrap();
        for &MachSrcLoc { start, end, loc } in mcr.buffer.get_srclocs_sorted() {
            debug_context.dwarf.unit.line_program.row().address_offset = u64::from(start);
            if !loc.is_default() {
                let source_loc = self.source_loc_set[loc.bits() as usize];
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
        entry.set(gimli::DW_AT_low_pc, AttributeValue::Address(address_for_func(func_id)));
        entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(u64::from(func_end)));

        func_end
    }
}
