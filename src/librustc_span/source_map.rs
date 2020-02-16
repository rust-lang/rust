//! The `SourceMap` tracks all the source code used within a single crate, mapping
//! from integer byte positions to the original source code location. Each bit
//! of source parsed during crate parsing (typically files, in-memory strings,
//! or various bits of macro expansion) cover a continuous range of bytes in the
//! `SourceMap` and are represented by `SourceFile`s. Byte positions are stored in
//! `Span` and used pervasively in the compiler. They are absolute positions
//! within the `SourceMap`, which upon request can be converted to line and column
//! information, source code snippets, etc.

pub use crate::hygiene::{ExpnData, ExpnKind};
pub use crate::*;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::sync::{Lock, LockGuard, Lrc, MappedLockGuard};
use std::cmp;
use std::hash::Hash;
use std::path::{Path, PathBuf};

use log::debug;
use std::env;
use std::fs;
use std::io;

#[cfg(test)]
mod tests;

/// Returns the span itself if it doesn't come from a macro expansion,
/// otherwise return the call site span up to the `enclosing_sp` by
/// following the `expn_data` chain.
pub fn original_sp(sp: Span, enclosing_sp: Span) -> Span {
    let expn_data1 = sp.ctxt().outer_expn_data();
    let expn_data2 = enclosing_sp.ctxt().outer_expn_data();
    if expn_data1.is_root() || !expn_data2.is_root() && expn_data1.call_site == expn_data2.call_site
    {
        sp
    } else {
        original_sp(expn_data1.call_site, enclosing_sp)
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Copy, HashStable_Generic)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

pub fn respan<T>(sp: Span, t: T) -> Spanned<T> {
    Spanned { node: t, span: sp }
}

pub fn dummy_spanned<T>(t: T) -> Spanned<T> {
    respan(DUMMY_SP, t)
}

// _____________________________________________________________________________
// SourceFile, MultiByteChar, FileName, FileLines
//

/// An abstraction over the fs operations used by the Parser.
pub trait FileLoader {
    /// Query the existence of a file.
    fn file_exists(&self, path: &Path) -> bool;

    /// Returns an absolute path to a file, if possible.
    fn abs_path(&self, path: &Path) -> Option<PathBuf>;

    /// Read the contents of an UTF-8 file into memory.
    fn read_file(&self, path: &Path) -> io::Result<String>;
}

/// A FileLoader that uses std::fs to load real files.
pub struct RealFileLoader;

impl FileLoader for RealFileLoader {
    fn file_exists(&self, path: &Path) -> bool {
        fs::metadata(path).is_ok()
    }

    fn abs_path(&self, path: &Path) -> Option<PathBuf> {
        if path.is_absolute() {
            Some(path.to_path_buf())
        } else {
            env::current_dir().ok().map(|cwd| cwd.join(path))
        }
    }

    fn read_file(&self, path: &Path) -> io::Result<String> {
        fs::read_to_string(path)
    }
}

// This is a `SourceFile` identifier that is used to correlate `SourceFile`s between
// subsequent compilation sessions (which is something we need to do during
// incremental compilation).
#[derive(Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug)]
pub struct StableSourceFileId(u128);

impl StableSourceFileId {
    pub fn new(source_file: &SourceFile) -> StableSourceFileId {
        StableSourceFileId::new_from_pieces(
            &source_file.name,
            source_file.name_was_remapped,
            source_file.unmapped_path.as_ref(),
        )
    }

    pub fn new_from_pieces(
        name: &FileName,
        name_was_remapped: bool,
        unmapped_path: Option<&FileName>,
    ) -> StableSourceFileId {
        let mut hasher = StableHasher::new();

        name.hash(&mut hasher);
        name_was_remapped.hash(&mut hasher);
        unmapped_path.hash(&mut hasher);

        StableSourceFileId(hasher.finish())
    }
}

// _____________________________________________________________________________
// SourceMap
//

#[derive(Default)]
pub(super) struct SourceMapFiles {
    source_files: Vec<Lrc<SourceFile>>,
    stable_id_to_source_file: FxHashMap<StableSourceFileId, Lrc<SourceFile>>,
}

pub struct SourceMap {
    files: Lock<SourceMapFiles>,
    file_loader: Box<dyn FileLoader + Sync + Send>,
    // This is used to apply the file path remapping as specified via
    // `--remap-path-prefix` to all `SourceFile`s allocated within this `SourceMap`.
    path_mapping: FilePathMapping,
}

impl SourceMap {
    pub fn new(path_mapping: FilePathMapping) -> SourceMap {
        SourceMap { files: Default::default(), file_loader: Box::new(RealFileLoader), path_mapping }
    }

    pub fn with_file_loader(
        file_loader: Box<dyn FileLoader + Sync + Send>,
        path_mapping: FilePathMapping,
    ) -> SourceMap {
        SourceMap { files: Default::default(), file_loader, path_mapping }
    }

    pub fn path_mapping(&self) -> &FilePathMapping {
        &self.path_mapping
    }

    pub fn file_exists(&self, path: &Path) -> bool {
        self.file_loader.file_exists(path)
    }

    pub fn load_file(&self, path: &Path) -> io::Result<Lrc<SourceFile>> {
        let src = self.file_loader.read_file(path)?;
        let filename = path.to_owned().into();
        Ok(self.new_source_file(filename, src))
    }

    /// Loads source file as a binary blob.
    ///
    /// Unlike `load_file`, guarantees that no normalization like BOM-removal
    /// takes place.
    pub fn load_binary_file(&self, path: &Path) -> io::Result<Vec<u8>> {
        // Ideally, this should use `self.file_loader`, but it can't
        // deal with binary files yet.
        let bytes = fs::read(path)?;

        // We need to add file to the `SourceMap`, so that it is present
        // in dep-info. There's also an edge case that file might be both
        // loaded as a binary via `include_bytes!` and as proper `SourceFile`
        // via `mod`, so we try to use real file contents and not just an
        // empty string.
        let text = std::str::from_utf8(&bytes).unwrap_or("").to_string();
        self.new_source_file(path.to_owned().into(), text);
        Ok(bytes)
    }

    pub fn files(&self) -> MappedLockGuard<'_, Vec<Lrc<SourceFile>>> {
        LockGuard::map(self.files.borrow(), |files| &mut files.source_files)
    }

    pub fn source_file_by_stable_id(
        &self,
        stable_id: StableSourceFileId,
    ) -> Option<Lrc<SourceFile>> {
        self.files.borrow().stable_id_to_source_file.get(&stable_id).map(|sf| sf.clone())
    }

    fn next_start_pos(&self) -> usize {
        match self.files.borrow().source_files.last() {
            None => 0,
            // Add one so there is some space between files. This lets us distinguish
            // positions in the `SourceMap`, even in the presence of zero-length files.
            Some(last) => last.end_pos.to_usize() + 1,
        }
    }

    /// Creates a new `SourceFile`.
    /// If a file already exists in the `SourceMap` with the same ID, that file is returned
    /// unmodified.
    pub fn new_source_file(&self, filename: FileName, src: String) -> Lrc<SourceFile> {
        self.try_new_source_file(filename, src).unwrap_or_else(|OffsetOverflowError| {
            eprintln!("fatal error: rustc does not support files larger than 4GB");
            crate::fatal_error::FatalError.raise()
        })
    }

    fn try_new_source_file(
        &self,
        filename: FileName,
        src: String,
    ) -> Result<Lrc<SourceFile>, OffsetOverflowError> {
        let start_pos = self.next_start_pos();

        // The path is used to determine the directory for loading submodules and
        // include files, so it must be before remapping.
        // Note that filename may not be a valid path, eg it may be `<anon>` etc,
        // but this is okay because the directory determined by `path.pop()` will
        // be empty, so the working directory will be used.
        let unmapped_path = filename.clone();

        let (filename, was_remapped) = match filename {
            FileName::Real(filename) => {
                let (filename, was_remapped) = self.path_mapping.map_prefix(filename);
                (FileName::Real(filename), was_remapped)
            }
            other => (other, false),
        };

        let file_id =
            StableSourceFileId::new_from_pieces(&filename, was_remapped, Some(&unmapped_path));

        let lrc_sf = match self.source_file_by_stable_id(file_id) {
            Some(lrc_sf) => lrc_sf,
            None => {
                let source_file = Lrc::new(SourceFile::new(
                    filename,
                    was_remapped,
                    unmapped_path,
                    src,
                    Pos::from_usize(start_pos),
                )?);

                let mut files = self.files.borrow_mut();

                files.source_files.push(source_file.clone());
                files.stable_id_to_source_file.insert(file_id, source_file.clone());

                source_file
            }
        };
        Ok(lrc_sf)
    }

    /// Allocates a new `SourceFile` representing a source file from an external
    /// crate. The source code of such an "imported `SourceFile`" is not available,
    /// but we still know enough to generate accurate debuginfo location
    /// information for things inlined from other crates.
    pub fn new_imported_source_file(
        &self,
        filename: FileName,
        name_was_remapped: bool,
        crate_of_origin: u32,
        src_hash: u128,
        name_hash: u128,
        source_len: usize,
        mut file_local_lines: Vec<BytePos>,
        mut file_local_multibyte_chars: Vec<MultiByteChar>,
        mut file_local_non_narrow_chars: Vec<NonNarrowChar>,
        mut file_local_normalized_pos: Vec<NormalizedPos>,
    ) -> Lrc<SourceFile> {
        let start_pos = self.next_start_pos();

        let end_pos = Pos::from_usize(start_pos + source_len);
        let start_pos = Pos::from_usize(start_pos);

        for pos in &mut file_local_lines {
            *pos = *pos + start_pos;
        }

        for mbc in &mut file_local_multibyte_chars {
            mbc.pos = mbc.pos + start_pos;
        }

        for swc in &mut file_local_non_narrow_chars {
            *swc = *swc + start_pos;
        }

        for nc in &mut file_local_normalized_pos {
            nc.pos = nc.pos + start_pos;
        }

        let source_file = Lrc::new(SourceFile {
            name: filename,
            name_was_remapped,
            unmapped_path: None,
            crate_of_origin,
            src: None,
            src_hash,
            external_src: Lock::new(ExternalSource::AbsentOk),
            start_pos,
            end_pos,
            lines: file_local_lines,
            multibyte_chars: file_local_multibyte_chars,
            non_narrow_chars: file_local_non_narrow_chars,
            normalized_pos: file_local_normalized_pos,
            name_hash,
        });

        let mut files = self.files.borrow_mut();

        files.source_files.push(source_file.clone());
        files
            .stable_id_to_source_file
            .insert(StableSourceFileId::new(&source_file), source_file.clone());

        source_file
    }

    pub fn mk_substr_filename(&self, sp: Span) -> String {
        let pos = self.lookup_char_pos(sp.lo());
        format!("<{}:{}:{}>", pos.file.name, pos.line, pos.col.to_usize() + 1)
    }

    // If there is a doctest offset, applies it to the line.
    pub fn doctest_offset_line(&self, file: &FileName, orig: usize) -> usize {
        return match file {
            FileName::DocTest(_, offset) => {
                return if *offset >= 0 {
                    orig + *offset as usize
                } else {
                    orig - (-(*offset)) as usize
                };
            }
            _ => orig,
        };
    }

    /// Looks up source information about a `BytePos`.
    pub fn lookup_char_pos(&self, pos: BytePos) -> Loc {
        let chpos = self.bytepos_to_file_charpos(pos);
        match self.lookup_line(pos) {
            Ok(SourceFileAndLine { sf: f, line: a }) => {
                let line = a + 1; // Line numbers start at 1
                let linebpos = f.lines[a];
                let linechpos = self.bytepos_to_file_charpos(linebpos);
                let col = chpos - linechpos;

                let col_display = {
                    let start_width_idx = f
                        .non_narrow_chars
                        .binary_search_by_key(&linebpos, |x| x.pos())
                        .unwrap_or_else(|x| x);
                    let end_width_idx = f
                        .non_narrow_chars
                        .binary_search_by_key(&pos, |x| x.pos())
                        .unwrap_or_else(|x| x);
                    let special_chars = end_width_idx - start_width_idx;
                    let non_narrow: usize = f.non_narrow_chars[start_width_idx..end_width_idx]
                        .into_iter()
                        .map(|x| x.width())
                        .sum();
                    col.0 - special_chars + non_narrow
                };
                debug!("byte pos {:?} is on the line at byte pos {:?}", pos, linebpos);
                debug!("char pos {:?} is on the line at char pos {:?}", chpos, linechpos);
                debug!("byte is on line: {}", line);
                assert!(chpos >= linechpos);
                Loc { file: f, line, col, col_display }
            }
            Err(f) => {
                let col_display = {
                    let end_width_idx = f
                        .non_narrow_chars
                        .binary_search_by_key(&pos, |x| x.pos())
                        .unwrap_or_else(|x| x);
                    let non_narrow: usize =
                        f.non_narrow_chars[0..end_width_idx].into_iter().map(|x| x.width()).sum();
                    chpos.0 - end_width_idx + non_narrow
                };
                Loc { file: f, line: 0, col: chpos, col_display }
            }
        }
    }

    // If the corresponding `SourceFile` is empty, does not return a line number.
    pub fn lookup_line(&self, pos: BytePos) -> Result<SourceFileAndLine, Lrc<SourceFile>> {
        let idx = self.lookup_source_file_idx(pos);

        let f = (*self.files.borrow().source_files)[idx].clone();

        match f.lookup_line(pos) {
            Some(line) => Ok(SourceFileAndLine { sf: f, line }),
            None => Err(f),
        }
    }

    /// Returns `Some(span)`, a union of the LHS and RHS span. The LHS must precede the RHS. If
    /// there are gaps between LHS and RHS, the resulting union will cross these gaps.
    /// For this to work,
    ///
    ///    * the syntax contexts of both spans much match,
    ///    * the LHS span needs to end on the same line the RHS span begins,
    ///    * the LHS span must start at or before the RHS span.
    pub fn merge_spans(&self, sp_lhs: Span, sp_rhs: Span) -> Option<Span> {
        // Ensure we're at the same expansion ID.
        if sp_lhs.ctxt() != sp_rhs.ctxt() {
            return None;
        }

        let lhs_end = match self.lookup_line(sp_lhs.hi()) {
            Ok(x) => x,
            Err(_) => return None,
        };
        let rhs_begin = match self.lookup_line(sp_rhs.lo()) {
            Ok(x) => x,
            Err(_) => return None,
        };

        // If we must cross lines to merge, don't merge.
        if lhs_end.line != rhs_begin.line {
            return None;
        }

        // Ensure these follow the expected order and that we don't overlap.
        if (sp_lhs.lo() <= sp_rhs.lo()) && (sp_lhs.hi() <= sp_rhs.lo()) {
            Some(sp_lhs.to(sp_rhs))
        } else {
            None
        }
    }

    pub fn span_to_string(&self, sp: Span) -> String {
        if self.files.borrow().source_files.is_empty() && sp.is_dummy() {
            return "no-location".to_string();
        }

        let lo = self.lookup_char_pos(sp.lo());
        let hi = self.lookup_char_pos(sp.hi());
        format!(
            "{}:{}:{}: {}:{}",
            lo.file.name,
            lo.line,
            lo.col.to_usize() + 1,
            hi.line,
            hi.col.to_usize() + 1,
        )
    }

    pub fn span_to_filename(&self, sp: Span) -> FileName {
        self.lookup_char_pos(sp.lo()).file.name.clone()
    }

    pub fn span_to_unmapped_path(&self, sp: Span) -> FileName {
        self.lookup_char_pos(sp.lo())
            .file
            .unmapped_path
            .clone()
            .expect("`SourceMap::span_to_unmapped_path` called for imported `SourceFile`?")
    }

    pub fn is_multiline(&self, sp: Span) -> bool {
        let lo = self.lookup_char_pos(sp.lo());
        let hi = self.lookup_char_pos(sp.hi());
        lo.line != hi.line
    }

    pub fn is_valid_span(&self, sp: Span) -> Result<(Loc, Loc), SpanLinesError> {
        let lo = self.lookup_char_pos(sp.lo());
        debug!("span_to_lines: lo={:?}", lo);
        let hi = self.lookup_char_pos(sp.hi());
        debug!("span_to_lines: hi={:?}", hi);
        if lo.file.start_pos != hi.file.start_pos {
            return Err(SpanLinesError::DistinctSources(DistinctSources {
                begin: (lo.file.name.clone(), lo.file.start_pos),
                end: (hi.file.name.clone(), hi.file.start_pos),
            }));
        }
        Ok((lo, hi))
    }

    pub fn span_to_lines(&self, sp: Span) -> FileLinesResult {
        debug!("span_to_lines(sp={:?})", sp);
        let (lo, hi) = self.is_valid_span(sp)?;
        assert!(hi.line >= lo.line);

        let mut lines = Vec::with_capacity(hi.line - lo.line + 1);

        // The span starts partway through the first line,
        // but after that it starts from offset 0.
        let mut start_col = lo.col;

        // For every line but the last, it extends from `start_col`
        // and to the end of the line. Be careful because the line
        // numbers in Loc are 1-based, so we subtract 1 to get 0-based
        // lines.
        let hi_line = hi.line.saturating_sub(1);
        for line_index in lo.line.saturating_sub(1)..hi_line {
            let line_len = lo.file.get_line(line_index).map(|s| s.chars().count()).unwrap_or(0);
            lines.push(LineInfo { line_index, start_col, end_col: CharPos::from_usize(line_len) });
            start_col = CharPos::from_usize(0);
        }

        // For the last line, it extends from `start_col` to `hi.col`:
        lines.push(LineInfo { line_index: hi_line, start_col, end_col: hi.col });

        Ok(FileLines { file: lo.file, lines })
    }

    /// Extracts the source surrounding the given `Span` using the `extract_source` function. The
    /// extract function takes three arguments: a string slice containing the source, an index in
    /// the slice for the beginning of the span and an index in the slice for the end of the span.
    fn span_to_source<F>(&self, sp: Span, extract_source: F) -> Result<String, SpanSnippetError>
    where
        F: Fn(&str, usize, usize) -> Result<String, SpanSnippetError>,
    {
        let local_begin = self.lookup_byte_offset(sp.lo());
        let local_end = self.lookup_byte_offset(sp.hi());

        if local_begin.sf.start_pos != local_end.sf.start_pos {
            return Err(SpanSnippetError::DistinctSources(DistinctSources {
                begin: (local_begin.sf.name.clone(), local_begin.sf.start_pos),
                end: (local_end.sf.name.clone(), local_end.sf.start_pos),
            }));
        } else {
            self.ensure_source_file_source_present(local_begin.sf.clone());

            let start_index = local_begin.pos.to_usize();
            let end_index = local_end.pos.to_usize();
            let source_len = (local_begin.sf.end_pos - local_begin.sf.start_pos).to_usize();

            if start_index > end_index || end_index > source_len {
                return Err(SpanSnippetError::MalformedForSourcemap(MalformedSourceMapPositions {
                    name: local_begin.sf.name.clone(),
                    source_len,
                    begin_pos: local_begin.pos,
                    end_pos: local_end.pos,
                }));
            }

            if let Some(ref src) = local_begin.sf.src {
                return extract_source(src, start_index, end_index);
            } else if let Some(src) = local_begin.sf.external_src.borrow().get_source() {
                return extract_source(src, start_index, end_index);
            } else {
                return Err(SpanSnippetError::SourceNotAvailable {
                    filename: local_begin.sf.name.clone(),
                });
            }
        }
    }

    /// Returns the source snippet as `String` corresponding to the given `Span`.
    pub fn span_to_snippet(&self, sp: Span) -> Result<String, SpanSnippetError> {
        self.span_to_source(sp, |src, start_index, end_index| {
            src.get(start_index..end_index)
                .map(|s| s.to_string())
                .ok_or_else(|| SpanSnippetError::IllFormedSpan(sp))
        })
    }

    pub fn span_to_margin(&self, sp: Span) -> Option<usize> {
        match self.span_to_prev_source(sp) {
            Err(_) => None,
            Ok(source) => source
                .split('\n')
                .last()
                .map(|last_line| last_line.len() - last_line.trim_start().len()),
        }
    }

    /// Returns the source snippet as `String` before the given `Span`.
    pub fn span_to_prev_source(&self, sp: Span) -> Result<String, SpanSnippetError> {
        self.span_to_source(sp, |src, start_index, _| {
            src.get(..start_index)
                .map(|s| s.to_string())
                .ok_or_else(|| SpanSnippetError::IllFormedSpan(sp))
        })
    }

    /// Extends the given `Span` to just after the previous occurrence of `c`. Return the same span
    /// if no character could be found or if an error occurred while retrieving the code snippet.
    pub fn span_extend_to_prev_char(&self, sp: Span, c: char) -> Span {
        if let Ok(prev_source) = self.span_to_prev_source(sp) {
            let prev_source = prev_source.rsplit(c).nth(0).unwrap_or("").trim_start();
            if !prev_source.is_empty() && !prev_source.contains('\n') {
                return sp.with_lo(BytePos(sp.lo().0 - prev_source.len() as u32));
            }
        }

        sp
    }

    /// Extends the given `Span` to just after the previous occurrence of `pat` when surrounded by
    /// whitespace. Returns the same span if no character could be found or if an error occurred
    /// while retrieving the code snippet.
    pub fn span_extend_to_prev_str(&self, sp: Span, pat: &str, accept_newlines: bool) -> Span {
        // assure that the pattern is delimited, to avoid the following
        //     fn my_fn()
        //           ^^^^ returned span without the check
        //     ---------- correct span
        for ws in &[" ", "\t", "\n"] {
            let pat = pat.to_owned() + ws;
            if let Ok(prev_source) = self.span_to_prev_source(sp) {
                let prev_source = prev_source.rsplit(&pat).nth(0).unwrap_or("").trim_start();
                if !prev_source.is_empty() && (!prev_source.contains('\n') || accept_newlines) {
                    return sp.with_lo(BytePos(sp.lo().0 - prev_source.len() as u32));
                }
            }
        }

        sp
    }

    /// Given a `Span`, tries to get a shorter span ending before the first occurrence of `char`
    /// `c`.
    pub fn span_until_char(&self, sp: Span, c: char) -> Span {
        match self.span_to_snippet(sp) {
            Ok(snippet) => {
                let snippet = snippet.split(c).nth(0).unwrap_or("").trim_end();
                if !snippet.is_empty() && !snippet.contains('\n') {
                    sp.with_hi(BytePos(sp.lo().0 + snippet.len() as u32))
                } else {
                    sp
                }
            }
            _ => sp,
        }
    }

    /// Given a `Span`, tries to get a shorter span ending just after the first occurrence of `char`
    /// `c`.
    pub fn span_through_char(&self, sp: Span, c: char) -> Span {
        if let Ok(snippet) = self.span_to_snippet(sp) {
            if let Some(offset) = snippet.find(c) {
                return sp.with_hi(BytePos(sp.lo().0 + (offset + c.len_utf8()) as u32));
            }
        }
        sp
    }

    /// Given a `Span`, gets a new `Span` covering the first token and all its trailing whitespace
    /// or the original `Span`.
    ///
    /// If `sp` points to `"let mut x"`, then a span pointing at `"let "` will be returned.
    pub fn span_until_non_whitespace(&self, sp: Span) -> Span {
        let mut whitespace_found = false;

        self.span_take_while(sp, |c| {
            if !whitespace_found && c.is_whitespace() {
                whitespace_found = true;
            }

            if whitespace_found && !c.is_whitespace() { false } else { true }
        })
    }

    /// Given a `Span`, gets a new `Span` covering the first token without its trailing whitespace
    /// or the original `Span` in case of error.
    ///
    /// If `sp` points to `"let mut x"`, then a span pointing at `"let"` will be returned.
    pub fn span_until_whitespace(&self, sp: Span) -> Span {
        self.span_take_while(sp, |c| !c.is_whitespace())
    }

    /// Given a `Span`, gets a shorter one until `predicate` yields `false`.
    pub fn span_take_while<P>(&self, sp: Span, predicate: P) -> Span
    where
        P: for<'r> FnMut(&'r char) -> bool,
    {
        if let Ok(snippet) = self.span_to_snippet(sp) {
            let offset = snippet.chars().take_while(predicate).map(|c| c.len_utf8()).sum::<usize>();

            sp.with_hi(BytePos(sp.lo().0 + (offset as u32)))
        } else {
            sp
        }
    }

    pub fn def_span(&self, sp: Span) -> Span {
        self.span_until_char(sp, '{')
    }

    /// Returns a new span representing just the start point of this span.
    pub fn start_point(&self, sp: Span) -> Span {
        let pos = sp.lo().0;
        let width = self.find_width_of_character_at_span(sp, false);
        let corrected_start_position = pos.checked_add(width).unwrap_or(pos);
        let end_point = BytePos(cmp::max(corrected_start_position, sp.lo().0));
        sp.with_hi(end_point)
    }

    /// Returns a new span representing just the end point of this span.
    pub fn end_point(&self, sp: Span) -> Span {
        let pos = sp.hi().0;

        let width = self.find_width_of_character_at_span(sp, false);
        let corrected_end_position = pos.checked_sub(width).unwrap_or(pos);

        let end_point = BytePos(cmp::max(corrected_end_position, sp.lo().0));
        sp.with_lo(end_point)
    }

    /// Returns a new span representing the next character after the end-point of this span.
    pub fn next_point(&self, sp: Span) -> Span {
        let start_of_next_point = sp.hi().0;

        let width = self.find_width_of_character_at_span(sp.shrink_to_hi(), true);
        // If the width is 1, then the next span should point to the same `lo` and `hi`. However,
        // in the case of a multibyte character, where the width != 1, the next span should
        // span multiple bytes to include the whole character.
        let end_of_next_point =
            start_of_next_point.checked_add(width - 1).unwrap_or(start_of_next_point);

        let end_of_next_point = BytePos(cmp::max(sp.lo().0 + 1, end_of_next_point));
        Span::new(BytePos(start_of_next_point), end_of_next_point, sp.ctxt())
    }

    /// Finds the width of a character, either before or after the provided span.
    fn find_width_of_character_at_span(&self, sp: Span, forwards: bool) -> u32 {
        let sp = sp.data();
        if sp.lo == sp.hi {
            debug!("find_width_of_character_at_span: early return empty span");
            return 1;
        }

        let local_begin = self.lookup_byte_offset(sp.lo);
        let local_end = self.lookup_byte_offset(sp.hi);
        debug!(
            "find_width_of_character_at_span: local_begin=`{:?}`, local_end=`{:?}`",
            local_begin, local_end
        );

        if local_begin.sf.start_pos != local_end.sf.start_pos {
            debug!("find_width_of_character_at_span: begin and end are in different files");
            return 1;
        }

        let start_index = local_begin.pos.to_usize();
        let end_index = local_end.pos.to_usize();
        debug!(
            "find_width_of_character_at_span: start_index=`{:?}`, end_index=`{:?}`",
            start_index, end_index
        );

        // Disregard indexes that are at the start or end of their spans, they can't fit bigger
        // characters.
        if (!forwards && end_index == usize::min_value())
            || (forwards && start_index == usize::max_value())
        {
            debug!("find_width_of_character_at_span: start or end of span, cannot be multibyte");
            return 1;
        }

        let source_len = (local_begin.sf.end_pos - local_begin.sf.start_pos).to_usize();
        debug!("find_width_of_character_at_span: source_len=`{:?}`", source_len);
        // Ensure indexes are also not malformed.
        if start_index > end_index || end_index > source_len {
            debug!("find_width_of_character_at_span: source indexes are malformed");
            return 1;
        }

        let src = local_begin.sf.external_src.borrow();

        // We need to extend the snippet to the end of the src rather than to end_index so when
        // searching forwards for boundaries we've got somewhere to search.
        let snippet = if let Some(ref src) = local_begin.sf.src {
            let len = src.len();
            &src[start_index..len]
        } else if let Some(src) = src.get_source() {
            let len = src.len();
            &src[start_index..len]
        } else {
            return 1;
        };
        debug!("find_width_of_character_at_span: snippet=`{:?}`", snippet);

        let mut target = if forwards { end_index + 1 } else { end_index - 1 };
        debug!("find_width_of_character_at_span: initial target=`{:?}`", target);

        while !snippet.is_char_boundary(target - start_index) && target < source_len {
            target = if forwards {
                target + 1
            } else {
                match target.checked_sub(1) {
                    Some(target) => target,
                    None => {
                        break;
                    }
                }
            };
            debug!("find_width_of_character_at_span: target=`{:?}`", target);
        }
        debug!("find_width_of_character_at_span: final target=`{:?}`", target);

        if forwards { (target - end_index) as u32 } else { (end_index - target) as u32 }
    }

    pub fn get_source_file(&self, filename: &FileName) -> Option<Lrc<SourceFile>> {
        for sf in self.files.borrow().source_files.iter() {
            if *filename == sf.name {
                return Some(sf.clone());
            }
        }
        None
    }

    /// For a global `BytePos`, computes the local offset within the containing `SourceFile`.
    pub fn lookup_byte_offset(&self, bpos: BytePos) -> SourceFileAndBytePos {
        let idx = self.lookup_source_file_idx(bpos);
        let sf = (*self.files.borrow().source_files)[idx].clone();
        let offset = bpos - sf.start_pos;
        SourceFileAndBytePos { sf, pos: offset }
    }

    /// Converts an absolute `BytePos` to a `CharPos` relative to the `SourceFile`.
    pub fn bytepos_to_file_charpos(&self, bpos: BytePos) -> CharPos {
        let idx = self.lookup_source_file_idx(bpos);
        let map = &(*self.files.borrow().source_files)[idx];

        // The number of extra bytes due to multibyte chars in the `SourceFile`.
        let mut total_extra_bytes = 0;

        for mbc in map.multibyte_chars.iter() {
            debug!("{}-byte char at {:?}", mbc.bytes, mbc.pos);
            if mbc.pos < bpos {
                // Every character is at least one byte, so we only
                // count the actual extra bytes.
                total_extra_bytes += mbc.bytes as u32 - 1;
                // We should never see a byte position in the middle of a
                // character.
                assert!(bpos.to_u32() >= mbc.pos.to_u32() + mbc.bytes as u32);
            } else {
                break;
            }
        }

        assert!(map.start_pos.to_u32() + total_extra_bytes <= bpos.to_u32());
        CharPos(bpos.to_usize() - map.start_pos.to_usize() - total_extra_bytes as usize)
    }

    // Returns the index of the `SourceFile` (in `self.files`) that contains `pos`.
    pub fn lookup_source_file_idx(&self, pos: BytePos) -> usize {
        self.files
            .borrow()
            .source_files
            .binary_search_by_key(&pos, |key| key.start_pos)
            .unwrap_or_else(|p| p - 1)
    }

    pub fn count_lines(&self) -> usize {
        self.files().iter().fold(0, |a, f| a + f.count_lines())
    }

    pub fn generate_fn_name_span(&self, span: Span) -> Option<Span> {
        let prev_span = self.span_extend_to_prev_str(span, "fn", true);
        self.span_to_snippet(prev_span)
            .map(|snippet| {
                let len = snippet
                    .find(|c: char| !c.is_alphanumeric() && c != '_')
                    .expect("no label after fn");
                prev_span.with_hi(BytePos(prev_span.lo().0 + len as u32))
            })
            .ok()
    }

    /// Takes the span of a type parameter in a function signature and try to generate a span for
    /// the function name (with generics) and a new snippet for this span with the pointed type
    /// parameter as a new local type parameter.
    ///
    /// For instance:
    /// ```rust,ignore (pseudo-Rust)
    /// // Given span
    /// fn my_function(param: T)
    /// //                    ^ Original span
    ///
    /// // Result
    /// fn my_function(param: T)
    /// // ^^^^^^^^^^^ Generated span with snippet `my_function<T>`
    /// ```
    ///
    /// Attention: The method used is very fragile since it essentially duplicates the work of the
    /// parser. If you need to use this function or something similar, please consider updating the
    /// `SourceMap` functions and this function to something more robust.
    pub fn generate_local_type_param_snippet(&self, span: Span) -> Option<(Span, String)> {
        // Try to extend the span to the previous "fn" keyword to retrieve the function
        // signature.
        let sugg_span = self.span_extend_to_prev_str(span, "fn", false);
        if sugg_span != span {
            if let Ok(snippet) = self.span_to_snippet(sugg_span) {
                // Consume the function name.
                let mut offset = snippet
                    .find(|c: char| !c.is_alphanumeric() && c != '_')
                    .expect("no label after fn");

                // Consume the generics part of the function signature.
                let mut bracket_counter = 0;
                let mut last_char = None;
                for c in snippet[offset..].chars() {
                    match c {
                        '<' => bracket_counter += 1,
                        '>' => bracket_counter -= 1,
                        '(' => {
                            if bracket_counter == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    offset += c.len_utf8();
                    last_char = Some(c);
                }

                // Adjust the suggestion span to encompass the function name with its generics.
                let sugg_span = sugg_span.with_hi(BytePos(sugg_span.lo().0 + offset as u32));

                // Prepare the new suggested snippet to append the type parameter that triggered
                // the error in the generics of the function signature.
                let mut new_snippet = if last_char == Some('>') {
                    format!("{}, ", &snippet[..(offset - '>'.len_utf8())])
                } else {
                    format!("{}<", &snippet[..offset])
                };
                new_snippet
                    .push_str(&self.span_to_snippet(span).unwrap_or_else(|_| "T".to_string()));
                new_snippet.push('>');

                return Some((sugg_span, new_snippet));
            }
        }

        None
    }
    pub fn ensure_source_file_source_present(&self, source_file: Lrc<SourceFile>) -> bool {
        source_file.add_external_src(|| match source_file.name {
            FileName::Real(ref name) => self.file_loader.read_file(name).ok(),
            _ => None,
        })
    }
}

#[derive(Clone)]
pub struct FilePathMapping {
    mapping: Vec<(PathBuf, PathBuf)>,
}

impl FilePathMapping {
    pub fn empty() -> FilePathMapping {
        FilePathMapping { mapping: vec![] }
    }

    pub fn new(mapping: Vec<(PathBuf, PathBuf)>) -> FilePathMapping {
        FilePathMapping { mapping }
    }

    /// Applies any path prefix substitution as defined by the mapping.
    /// The return value is the remapped path and a boolean indicating whether
    /// the path was affected by the mapping.
    pub fn map_prefix(&self, path: PathBuf) -> (PathBuf, bool) {
        // NOTE: We are iterating over the mapping entries from last to first
        //       because entries specified later on the command line should
        //       take precedence.
        for &(ref from, ref to) in self.mapping.iter().rev() {
            if let Ok(rest) = path.strip_prefix(from) {
                return (to.join(rest), true);
            }
        }

        (path, false)
    }
}
