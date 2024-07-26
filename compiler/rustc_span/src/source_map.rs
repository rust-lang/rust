//! Types for tracking pieces of source code within a crate.
//!
//! The [`SourceMap`] tracks all the source code used within a single crate, mapping
//! from integer byte positions to the original source code location. Each bit
//! of source parsed during crate parsing (typically files, in-memory strings,
//! or various bits of macro expansion) cover a continuous range of bytes in the
//! `SourceMap` and are represented by [`SourceFile`]s. Byte positions are stored in
//! [`Span`] and used pervasively in the compiler. They are absolute positions
//! within the `SourceMap`, which upon request can be converted to line and column
//! information, source code snippets, etc.

use crate::*;
use rustc_data_structures::sync::{IntoDynSyncSend, MappedReadGuard, ReadGuard, RwLock};
use rustc_data_structures::unhash::UnhashMap;
use rustc_macros::{Decodable, Encodable};
use std::fs;
use std::io::{self, BorrowedBuf, Read};
use std::path;
use tracing::{debug, instrument, trace};

#[cfg(test)]
mod tests;

/// Returns the span itself if it doesn't come from a macro expansion,
/// otherwise return the call site span up to the `enclosing_sp` by
/// following the `expn_data` chain.
pub fn original_sp(sp: Span, enclosing_sp: Span) -> Span {
    let ctxt = sp.ctxt();
    if ctxt.is_root() {
        return sp;
    }

    let enclosing_ctxt = enclosing_sp.ctxt();
    let expn_data1 = ctxt.outer_expn_data();
    if !enclosing_ctxt.is_root()
        && expn_data1.call_site == enclosing_ctxt.outer_expn_data().call_site
    {
        sp
    } else {
        original_sp(expn_data1.call_site, enclosing_sp)
    }
}

mod monotonic {
    use std::ops::{Deref, DerefMut};

    /// A `MonotonicVec` is a `Vec` which can only be grown.
    /// Once inserted, an element can never be removed or swapped,
    /// guaranteeing that any indices into a `MonotonicVec` are stable
    // This is declared in its own module to ensure that the private
    // field is inaccessible
    pub struct MonotonicVec<T>(Vec<T>);
    impl<T> MonotonicVec<T> {
        pub(super) fn push(&mut self, val: T) {
            self.0.push(val);
        }
    }

    impl<T> Default for MonotonicVec<T> {
        fn default() -> Self {
            MonotonicVec(vec![])
        }
    }

    impl<T> Deref for MonotonicVec<T> {
        type Target = Vec<T>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<T> !DerefMut for MonotonicVec<T> {}
}

#[derive(Clone, Encodable, Decodable, Debug, Copy, PartialEq, Hash, HashStable_Generic)]
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

    /// Read the contents of a UTF-8 file into memory.
    /// This function must return a String because we normalize
    /// source files, which may require resizing.
    fn read_file(&self, path: &Path) -> io::Result<String>;

    /// Read the contents of a potentially non-UTF-8 file into memory.
    /// We don't normalize binary files, so we can start in an Lrc.
    fn read_binary_file(&self, path: &Path) -> io::Result<Lrc<[u8]>>;
}

/// A FileLoader that uses std::fs to load real files.
pub struct RealFileLoader;

impl FileLoader for RealFileLoader {
    fn file_exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn read_file(&self, path: &Path) -> io::Result<String> {
        fs::read_to_string(path)
    }

    fn read_binary_file(&self, path: &Path) -> io::Result<Lrc<[u8]>> {
        let mut file = fs::File::open(path)?;
        let len = file.metadata()?.len();

        let mut bytes = Lrc::new_uninit_slice(len as usize);
        let mut buf = BorrowedBuf::from(Lrc::get_mut(&mut bytes).unwrap());
        match file.read_buf_exact(buf.unfilled()) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                drop(bytes);
                return fs::read(path).map(Vec::into);
            }
            Err(e) => return Err(e),
        }
        // SAFETY: If the read_buf_exact call returns Ok(()), then we have
        // read len bytes and initialized the buffer.
        let bytes = unsafe { bytes.assume_init() };

        // At this point, we've read all the bytes that filesystem metadata reported exist.
        // But we are not guaranteed to be at the end of the file, because we did not attempt to do
        // a read with a non-zero-sized buffer and get Ok(0).
        // So we do small read to a fixed-size buffer. If the read returns no bytes then we're
        // already done, and we just return the Lrc we built above.
        // If the read returns bytes however, we just fall back to reading into a Vec then turning
        // that into an Lrc, losing our nice peak memory behavior. This fallback code path should
        // be rarely exercised.

        let mut probe = [0u8; 32];
        let n = loop {
            match file.read(&mut probe) {
                Ok(0) => return Ok(bytes),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
                Ok(n) => break n,
            }
        };
        let mut bytes: Vec<u8> = bytes.iter().copied().chain(probe[..n].iter().copied()).collect();
        file.read_to_end(&mut bytes)?;
        Ok(bytes.into())
    }
}

// _____________________________________________________________________________
// SourceMap
//

#[derive(Default)]
struct SourceMapFiles {
    source_files: monotonic::MonotonicVec<Lrc<SourceFile>>,
    stable_id_to_source_file: UnhashMap<StableSourceFileId, Lrc<SourceFile>>,
}

/// Used to construct a `SourceMap` with `SourceMap::with_inputs`.
pub struct SourceMapInputs {
    pub file_loader: Box<dyn FileLoader + Send + Sync>,
    pub path_mapping: FilePathMapping,
    pub hash_kind: SourceFileHashAlgorithm,
}

pub struct SourceMap {
    files: RwLock<SourceMapFiles>,
    file_loader: IntoDynSyncSend<Box<dyn FileLoader + Sync + Send>>,

    // This is used to apply the file path remapping as specified via
    // `--remap-path-prefix` to all `SourceFile`s allocated within this `SourceMap`.
    path_mapping: FilePathMapping,

    /// The algorithm used for hashing the contents of each source file.
    hash_kind: SourceFileHashAlgorithm,
}

impl SourceMap {
    pub fn new(path_mapping: FilePathMapping) -> SourceMap {
        Self::with_inputs(SourceMapInputs {
            file_loader: Box::new(RealFileLoader),
            path_mapping,
            hash_kind: SourceFileHashAlgorithm::Md5,
        })
    }

    pub fn with_inputs(
        SourceMapInputs { file_loader, path_mapping, hash_kind }: SourceMapInputs,
    ) -> SourceMap {
        SourceMap {
            files: Default::default(),
            file_loader: IntoDynSyncSend(file_loader),
            path_mapping,
            hash_kind,
        }
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
    pub fn load_binary_file(&self, path: &Path) -> io::Result<(Lrc<[u8]>, Span)> {
        let bytes = self.file_loader.read_binary_file(path)?;

        // We need to add file to the `SourceMap`, so that it is present
        // in dep-info. There's also an edge case that file might be both
        // loaded as a binary via `include_bytes!` and as proper `SourceFile`
        // via `mod`, so we try to use real file contents and not just an
        // empty string.
        let text = std::str::from_utf8(&bytes).unwrap_or("").to_string();
        let file = self.new_source_file(path.to_owned().into(), text);
        Ok((
            bytes,
            Span::new(
                file.start_pos,
                BytePos(file.start_pos.0 + file.source_len.0),
                SyntaxContext::root(),
                None,
            ),
        ))
    }

    // By returning a `MonotonicVec`, we ensure that consumers cannot invalidate
    // any existing indices pointing into `files`.
    pub fn files(&self) -> MappedReadGuard<'_, monotonic::MonotonicVec<Lrc<SourceFile>>> {
        ReadGuard::map(self.files.borrow(), |files| &files.source_files)
    }

    pub fn source_file_by_stable_id(
        &self,
        stable_id: StableSourceFileId,
    ) -> Option<Lrc<SourceFile>> {
        self.files.borrow().stable_id_to_source_file.get(&stable_id).cloned()
    }

    fn register_source_file(
        &self,
        file_id: StableSourceFileId,
        mut file: SourceFile,
    ) -> Result<Lrc<SourceFile>, OffsetOverflowError> {
        let mut files = self.files.borrow_mut();

        file.start_pos = BytePos(if let Some(last_file) = files.source_files.last() {
            // Add one so there is some space between files. This lets us distinguish
            // positions in the `SourceMap`, even in the presence of zero-length files.
            last_file.end_position().0.checked_add(1).ok_or(OffsetOverflowError)?
        } else {
            0
        });

        let file = Lrc::new(file);
        files.source_files.push(file.clone());
        files.stable_id_to_source_file.insert(file_id, file.clone());

        Ok(file)
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
        // Note that filename may not be a valid path, eg it may be `<anon>` etc,
        // but this is okay because the directory determined by `path.pop()` will
        // be empty, so the working directory will be used.
        let (filename, _) = self.path_mapping.map_filename_prefix(&filename);

        let stable_id = StableSourceFileId::from_filename_in_current_crate(&filename);
        match self.source_file_by_stable_id(stable_id) {
            Some(lrc_sf) => Ok(lrc_sf),
            None => {
                let source_file = SourceFile::new(filename, src, self.hash_kind)?;

                // Let's make sure the file_id we generated above actually matches
                // the ID we generate for the SourceFile we just created.
                debug_assert_eq!(source_file.stable_id, stable_id);

                self.register_source_file(stable_id, source_file)
            }
        }
    }

    /// Allocates a new `SourceFile` representing a source file from an external
    /// crate. The source code of such an "imported `SourceFile`" is not available,
    /// but we still know enough to generate accurate debuginfo location
    /// information for things inlined from other crates.
    pub fn new_imported_source_file(
        &self,
        filename: FileName,
        src_hash: SourceFileHash,
        stable_id: StableSourceFileId,
        source_len: u32,
        cnum: CrateNum,
        file_local_lines: FreezeLock<SourceFileLines>,
        multibyte_chars: Vec<MultiByteChar>,
        normalized_pos: Vec<NormalizedPos>,
        metadata_index: u32,
    ) -> Lrc<SourceFile> {
        let source_len = RelativeBytePos::from_u32(source_len);

        let source_file = SourceFile {
            name: filename,
            src: None,
            src_hash,
            external_src: FreezeLock::new(ExternalSource::Foreign {
                kind: ExternalSourceKind::AbsentOk,
                metadata_index,
            }),
            start_pos: BytePos(0),
            source_len,
            lines: file_local_lines,
            multibyte_chars,
            normalized_pos,
            stable_id,
            cnum,
        };

        self.register_source_file(stable_id, source_file)
            .expect("not enough address space for imported source file")
    }

    /// If there is a doctest offset, applies it to the line.
    pub fn doctest_offset_line(&self, file: &FileName, orig: usize) -> usize {
        match file {
            FileName::DocTest(_, offset) => {
                if *offset < 0 {
                    orig - (-(*offset)) as usize
                } else {
                    orig + *offset as usize
                }
            }
            _ => orig,
        }
    }

    /// Return the SourceFile that contains the given `BytePos`
    pub fn lookup_source_file(&self, pos: BytePos) -> Lrc<SourceFile> {
        let idx = self.lookup_source_file_idx(pos);
        (*self.files.borrow().source_files)[idx].clone()
    }

    /// Looks up source information about a `BytePos`.
    pub fn lookup_char_pos(&self, pos: BytePos) -> Loc {
        let sf = self.lookup_source_file(pos);
        let (line, col, col_display) = sf.lookup_file_pos_with_col_display(pos);
        Loc { file: sf, line, col, col_display }
    }

    /// If the corresponding `SourceFile` is empty, does not return a line number.
    pub fn lookup_line(&self, pos: BytePos) -> Result<SourceFileAndLine, Lrc<SourceFile>> {
        let f = self.lookup_source_file(pos);

        let pos = f.relative_position(pos);
        match f.lookup_line(pos) {
            Some(line) => Ok(SourceFileAndLine { sf: f, line }),
            None => Err(f),
        }
    }

    pub fn span_to_string(
        &self,
        sp: Span,
        filename_display_pref: FileNameDisplayPreference,
    ) -> String {
        let (source_file, lo_line, lo_col, hi_line, hi_col) = self.span_to_location_info(sp);

        let file_name = match source_file {
            Some(sf) => sf.name.display(filename_display_pref).to_string(),
            None => return "no-location".to_string(),
        };

        format!(
            "{file_name}:{lo_line}:{lo_col}{}",
            if let FileNameDisplayPreference::Short = filename_display_pref {
                String::new()
            } else {
                format!(": {hi_line}:{hi_col}")
            }
        )
    }

    pub fn span_to_location_info(
        &self,
        sp: Span,
    ) -> (Option<Lrc<SourceFile>>, usize, usize, usize, usize) {
        if self.files.borrow().source_files.is_empty() || sp.is_dummy() {
            return (None, 0, 0, 0, 0);
        }

        let lo = self.lookup_char_pos(sp.lo());
        let hi = self.lookup_char_pos(sp.hi());
        (Some(lo.file), lo.line, lo.col.to_usize() + 1, hi.line, hi.col.to_usize() + 1)
    }

    /// Format the span location suitable for embedding in build artifacts
    pub fn span_to_embeddable_string(&self, sp: Span) -> String {
        self.span_to_string(sp, FileNameDisplayPreference::Remapped)
    }

    /// Format the span location to be printed in diagnostics. Must not be emitted
    /// to build artifacts as this may leak local file paths. Use span_to_embeddable_string
    /// for string suitable for embedding.
    pub fn span_to_diagnostic_string(&self, sp: Span) -> String {
        self.span_to_string(sp, self.path_mapping.filename_display_for_diagnostics)
    }

    pub fn span_to_filename(&self, sp: Span) -> FileName {
        self.lookup_char_pos(sp.lo()).file.name.clone()
    }

    pub fn filename_for_diagnostics<'a>(&self, filename: &'a FileName) -> FileNameDisplay<'a> {
        filename.display(self.path_mapping.filename_display_for_diagnostics)
    }

    pub fn is_multiline(&self, sp: Span) -> bool {
        let lo = self.lookup_source_file_idx(sp.lo());
        let hi = self.lookup_source_file_idx(sp.hi());
        if lo != hi {
            return true;
        }
        let f = (*self.files.borrow().source_files)[lo].clone();
        let lo = f.relative_position(sp.lo());
        let hi = f.relative_position(sp.hi());
        f.lookup_line(lo) != f.lookup_line(hi)
    }

    #[instrument(skip(self), level = "trace")]
    pub fn is_valid_span(&self, sp: Span) -> Result<(Loc, Loc), SpanLinesError> {
        let lo = self.lookup_char_pos(sp.lo());
        trace!(?lo);
        let hi = self.lookup_char_pos(sp.hi());
        trace!(?hi);
        if lo.file.start_pos != hi.file.start_pos {
            return Err(SpanLinesError::DistinctSources(Box::new(DistinctSources {
                begin: (lo.file.name.clone(), lo.file.start_pos),
                end: (hi.file.name.clone(), hi.file.start_pos),
            })));
        }
        Ok((lo, hi))
    }

    pub fn is_line_before_span_empty(&self, sp: Span) -> bool {
        match self.span_to_prev_source(sp) {
            Ok(s) => s.rsplit_once('\n').unwrap_or(("", &s)).1.trim_start().is_empty(),
            Err(_) => false,
        }
    }

    pub fn span_to_lines(&self, sp: Span) -> FileLinesResult {
        debug!("span_to_lines(sp={:?})", sp);
        let (lo, hi) = self.is_valid_span(sp)?;
        assert!(hi.line >= lo.line);

        if sp.is_dummy() {
            return Ok(FileLines { file: lo.file, lines: Vec::new() });
        }

        let mut lines = Vec::with_capacity(hi.line - lo.line + 1);

        // The span starts partway through the first line,
        // but after that it starts from offset 0.
        let mut start_col = lo.col;

        // For every line but the last, it extends from `start_col`
        // and to the end of the line. Be careful because the line
        // numbers in Loc are 1-based, so we subtract 1 to get 0-based
        // lines.
        //
        // FIXME: now that we handle DUMMY_SP up above, we should consider
        // asserting that the line numbers here are all indeed 1-based.
        let hi_line = hi.line.saturating_sub(1);
        for line_index in lo.line.saturating_sub(1)..hi_line {
            let line_len = lo.file.get_line(line_index).map_or(0, |s| s.chars().count());
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
    fn span_to_source<F, T>(&self, sp: Span, extract_source: F) -> Result<T, SpanSnippetError>
    where
        F: Fn(&str, usize, usize) -> Result<T, SpanSnippetError>,
    {
        let local_begin = self.lookup_byte_offset(sp.lo());
        let local_end = self.lookup_byte_offset(sp.hi());

        if local_begin.sf.start_pos != local_end.sf.start_pos {
            Err(SpanSnippetError::DistinctSources(Box::new(DistinctSources {
                begin: (local_begin.sf.name.clone(), local_begin.sf.start_pos),
                end: (local_end.sf.name.clone(), local_end.sf.start_pos),
            })))
        } else {
            self.ensure_source_file_source_present(&local_begin.sf);

            let start_index = local_begin.pos.to_usize();
            let end_index = local_end.pos.to_usize();
            let source_len = local_begin.sf.source_len.to_usize();

            if start_index > end_index || end_index > source_len {
                return Err(SpanSnippetError::MalformedForSourcemap(MalformedSourceMapPositions {
                    name: local_begin.sf.name.clone(),
                    source_len,
                    begin_pos: local_begin.pos,
                    end_pos: local_end.pos,
                }));
            }

            if let Some(ref src) = local_begin.sf.src {
                extract_source(src, start_index, end_index)
            } else if let Some(src) = local_begin.sf.external_src.read().get_source() {
                extract_source(src, start_index, end_index)
            } else {
                Err(SpanSnippetError::SourceNotAvailable { filename: local_begin.sf.name.clone() })
            }
        }
    }

    pub fn is_span_accessible(&self, sp: Span) -> bool {
        self.span_to_source(sp, |src, start_index, end_index| {
            Ok(src.get(start_index..end_index).is_some())
        })
        .is_ok_and(|is_accessible| is_accessible)
    }

    /// Returns the source snippet as `String` corresponding to the given `Span`.
    pub fn span_to_snippet(&self, sp: Span) -> Result<String, SpanSnippetError> {
        self.span_to_source(sp, |src, start_index, end_index| {
            src.get(start_index..end_index)
                .map(|s| s.to_string())
                .ok_or(SpanSnippetError::IllFormedSpan(sp))
        })
    }

    pub fn span_to_margin(&self, sp: Span) -> Option<usize> {
        Some(self.indentation_before(sp)?.len())
    }

    pub fn indentation_before(&self, sp: Span) -> Option<String> {
        self.span_to_source(sp, |src, start_index, _| {
            let before = &src[..start_index];
            let last_line = before.rsplit_once('\n').map_or(before, |(_, last)| last);
            Ok(last_line
                .split_once(|c: char| !c.is_whitespace())
                .map_or(last_line, |(indent, _)| indent)
                .to_string())
        })
        .ok()
    }

    /// Returns the source snippet as `String` before the given `Span`.
    pub fn span_to_prev_source(&self, sp: Span) -> Result<String, SpanSnippetError> {
        self.span_to_source(sp, |src, start_index, _| {
            src.get(..start_index).map(|s| s.to_string()).ok_or(SpanSnippetError::IllFormedSpan(sp))
        })
    }

    /// Extends the given `Span` to just after the previous occurrence of `c`. Return the same span
    /// if no character could be found or if an error occurred while retrieving the code snippet.
    pub fn span_extend_to_prev_char(&self, sp: Span, c: char, accept_newlines: bool) -> Span {
        if let Ok(prev_source) = self.span_to_prev_source(sp) {
            let prev_source = prev_source.rsplit(c).next().unwrap_or("");
            if !prev_source.is_empty() && (accept_newlines || !prev_source.contains('\n')) {
                return sp.with_lo(BytePos(sp.lo().0 - prev_source.len() as u32));
            }
        }

        sp
    }

    /// Extends the given `Span` to just after the previous occurrence of `pat` when surrounded by
    /// whitespace. Returns None if the pattern could not be found or if an error occurred while
    /// retrieving the code snippet.
    pub fn span_extend_to_prev_str(
        &self,
        sp: Span,
        pat: &str,
        accept_newlines: bool,
        include_whitespace: bool,
    ) -> Option<Span> {
        // assure that the pattern is delimited, to avoid the following
        //     fn my_fn()
        //           ^^^^ returned span without the check
        //     ---------- correct span
        let prev_source = self.span_to_prev_source(sp).ok()?;
        for ws in &[" ", "\t", "\n"] {
            let pat = pat.to_owned() + ws;
            if let Some(pat_pos) = prev_source.rfind(&pat) {
                let just_after_pat_pos = pat_pos + pat.len() - 1;
                let just_after_pat_plus_ws = if include_whitespace {
                    just_after_pat_pos
                        + prev_source[just_after_pat_pos..]
                            .find(|c: char| !c.is_whitespace())
                            .unwrap_or(0)
                } else {
                    just_after_pat_pos
                };
                let len = prev_source.len() - just_after_pat_plus_ws;
                let prev_source = &prev_source[just_after_pat_plus_ws..];
                if accept_newlines || !prev_source.trim_start().contains('\n') {
                    return Some(sp.with_lo(BytePos(sp.lo().0 - len as u32)));
                }
            }
        }

        None
    }

    /// Returns the source snippet as `String` after the given `Span`.
    pub fn span_to_next_source(&self, sp: Span) -> Result<String, SpanSnippetError> {
        self.span_to_source(sp, |src, _, end_index| {
            src.get(end_index..).map(|s| s.to_string()).ok_or(SpanSnippetError::IllFormedSpan(sp))
        })
    }

    /// Extends the given `Span` while the next character matches the predicate
    pub fn span_extend_while(
        &self,
        span: Span,
        f: impl Fn(char) -> bool,
    ) -> Result<Span, SpanSnippetError> {
        self.span_to_source(span, |s, _start, end| {
            let n = s[end..].char_indices().find(|&(_, c)| !f(c)).map_or(s.len() - end, |(i, _)| i);
            Ok(span.with_hi(span.hi() + BytePos(n as u32)))
        })
    }

    /// Extends the span to include any trailing whitespace, or returns the original
    /// span if a `SpanSnippetError` was encountered.
    pub fn span_extend_while_whitespace(&self, span: Span) -> Span {
        self.span_extend_while(span, char::is_whitespace).unwrap_or(span)
    }

    /// Extends the given `Span` to previous character while the previous character matches the predicate
    pub fn span_extend_prev_while(
        &self,
        span: Span,
        f: impl Fn(char) -> bool,
    ) -> Result<Span, SpanSnippetError> {
        self.span_to_source(span, |s, start, _end| {
            let n = s[..start]
                .char_indices()
                .rfind(|&(_, c)| !f(c))
                .map_or(start, |(i, _)| start - i - 1);
            Ok(span.with_lo(span.lo() - BytePos(n as u32)))
        })
    }

    /// Extends the given `Span` to just before the next occurrence of `c`.
    pub fn span_extend_to_next_char(&self, sp: Span, c: char, accept_newlines: bool) -> Span {
        if let Ok(next_source) = self.span_to_next_source(sp) {
            let next_source = next_source.split(c).next().unwrap_or("");
            if !next_source.is_empty() && (accept_newlines || !next_source.contains('\n')) {
                return sp.with_hi(BytePos(sp.hi().0 + next_source.len() as u32));
            }
        }

        sp
    }

    /// Extends the given `Span` to contain the entire line it is on.
    pub fn span_extend_to_line(&self, sp: Span) -> Span {
        self.span_extend_to_prev_char(self.span_extend_to_next_char(sp, '\n', true), '\n', true)
    }

    /// Given a `Span`, tries to get a shorter span ending before the first occurrence of `char`
    /// `c`.
    pub fn span_until_char(&self, sp: Span, c: char) -> Span {
        match self.span_to_snippet(sp) {
            Ok(snippet) => {
                let snippet = snippet.split(c).next().unwrap_or("").trim_end();
                if !snippet.is_empty() && !snippet.contains('\n') {
                    sp.with_hi(BytePos(sp.lo().0 + snippet.len() as u32))
                } else {
                    sp
                }
            }
            _ => sp,
        }
    }

    /// Given a 'Span', tries to tell if it's wrapped by "<>" or "()"
    /// the algorithm searches if the next character is '>' or ')' after skipping white space
    /// then searches the previous character to match '<' or '(' after skipping white space
    /// return true if wrapped by '<>' or '()'
    pub fn span_wrapped_by_angle_or_parentheses(&self, span: Span) -> bool {
        self.span_to_source(span, |src, start_index, end_index| {
            if src.get(start_index..end_index).is_none() {
                return Ok(false);
            }
            // test the right side to match '>' after skipping white space
            let end_src = &src[end_index..];
            let mut i = 0;
            let mut found_right_parentheses = false;
            let mut found_right_angle = false;
            while let Some(cc) = end_src.chars().nth(i) {
                if cc == ' ' {
                    i = i + 1;
                } else if cc == '>' {
                    // found > in the right;
                    found_right_angle = true;
                    break;
                } else if cc == ')' {
                    found_right_parentheses = true;
                    break;
                } else {
                    // failed to find '>' return false immediately
                    return Ok(false);
                }
            }
            // test the left side to match '<' after skipping white space
            i = start_index;
            let start_src = &src[0..start_index];
            while let Some(cc) = start_src.chars().nth(i) {
                if cc == ' ' {
                    if i == 0 {
                        return Ok(false);
                    }
                    i = i - 1;
                } else if cc == '<' {
                    // found < in the left
                    if !found_right_angle {
                        // skip something like "(< )>"
                        return Ok(false);
                    }
                    break;
                } else if cc == '(' {
                    if !found_right_parentheses {
                        // skip something like "<(>)"
                        return Ok(false);
                    }
                    break;
                } else {
                    // failed to find '<' return false immediately
                    return Ok(false);
                }
            }
            return Ok(true);
        })
        .is_ok_and(|is_accessible| is_accessible)
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

            !whitespace_found || c.is_whitespace()
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

    /// Given a `Span`, return a span ending in the closest `{`. This is useful when you have a
    /// `Span` enclosing a whole item but we need to point at only the head (usually the first
    /// line) of that item.
    ///
    /// *Only suitable for diagnostics.*
    pub fn guess_head_span(&self, sp: Span) -> Span {
        // FIXME: extend the AST items to have a head span, or replace callers with pointing at
        // the item's ident when appropriate.
        self.span_until_char(sp, '{')
    }

    /// Returns a new span representing just the first character of the given span.
    pub fn start_point(&self, sp: Span) -> Span {
        let width = {
            let sp = sp.data();
            let local_begin = self.lookup_byte_offset(sp.lo);
            let start_index = local_begin.pos.to_usize();
            let src = local_begin.sf.external_src.read();

            let snippet = if let Some(ref src) = local_begin.sf.src {
                Some(&src[start_index..])
            } else {
                src.get_source().map(|src| &src[start_index..])
            };

            match snippet {
                None => 1,
                Some(snippet) => match snippet.chars().next() {
                    None => 1,
                    Some(c) => c.len_utf8(),
                },
            }
        };

        sp.with_hi(BytePos(sp.lo().0 + width as u32))
    }

    /// Returns a new span representing just the last character of this span.
    pub fn end_point(&self, sp: Span) -> Span {
        let pos = sp.hi().0;

        let width = self.find_width_of_character_at_span(sp, false);
        let corrected_end_position = pos.checked_sub(width).unwrap_or(pos);

        let end_point = BytePos(cmp::max(corrected_end_position, sp.lo().0));
        sp.with_lo(end_point)
    }

    /// Returns a new span representing the next character after the end-point of this span.
    /// Special cases:
    /// - if span is a dummy one, returns the same span
    /// - if next_point reached the end of source, return a span exceeding the end of source,
    ///   which means sm.span_to_snippet(next_point) will get `Err`
    /// - respect multi-byte characters
    pub fn next_point(&self, sp: Span) -> Span {
        if sp.is_dummy() {
            return sp;
        }
        let start_of_next_point = sp.hi().0;

        let width = self.find_width_of_character_at_span(sp, true);
        // If the width is 1, then the next span should only contain the next char besides current ending.
        // However, in the case of a multibyte character, where the width != 1, the next span should
        // span multiple bytes to include the whole character.
        let end_of_next_point =
            start_of_next_point.checked_add(width).unwrap_or(start_of_next_point);

        let end_of_next_point = BytePos(cmp::max(start_of_next_point + 1, end_of_next_point));
        Span::new(BytePos(start_of_next_point), end_of_next_point, sp.ctxt(), None)
    }

    /// Check whether span is followed by some specified expected string in limit scope
    pub fn span_look_ahead(&self, span: Span, expect: &str, limit: Option<usize>) -> Option<Span> {
        let mut sp = span;
        for _ in 0..limit.unwrap_or(100_usize) {
            sp = self.next_point(sp);
            if let Ok(ref snippet) = self.span_to_snippet(sp) {
                if snippet == expect {
                    return Some(sp);
                }
                if snippet.chars().any(|c| !c.is_whitespace()) {
                    break;
                }
            }
        }
        None
    }

    /// Finds the width of the character, either before or after the end of provided span,
    /// depending on the `forwards` parameter.
    #[instrument(skip(self, sp))]
    fn find_width_of_character_at_span(&self, sp: Span, forwards: bool) -> u32 {
        let sp = sp.data();

        if sp.lo == sp.hi && !forwards {
            debug!("early return empty span");
            return 1;
        }

        let local_begin = self.lookup_byte_offset(sp.lo);
        let local_end = self.lookup_byte_offset(sp.hi);
        debug!("local_begin=`{:?}`, local_end=`{:?}`", local_begin, local_end);

        if local_begin.sf.start_pos != local_end.sf.start_pos {
            debug!("begin and end are in different files");
            return 1;
        }

        let start_index = local_begin.pos.to_usize();
        let end_index = local_end.pos.to_usize();
        debug!("start_index=`{:?}`, end_index=`{:?}`", start_index, end_index);

        // Disregard indexes that are at the start or end of their spans, they can't fit bigger
        // characters.
        if (!forwards && end_index == usize::MIN) || (forwards && start_index == usize::MAX) {
            debug!("start or end of span, cannot be multibyte");
            return 1;
        }

        let source_len = local_begin.sf.source_len.to_usize();
        debug!("source_len=`{:?}`", source_len);
        // Ensure indexes are also not malformed.
        if start_index > end_index || end_index > source_len - 1 {
            debug!("source indexes are malformed");
            return 1;
        }

        let src = local_begin.sf.external_src.read();

        let snippet = if let Some(src) = &local_begin.sf.src {
            src
        } else if let Some(src) = src.get_source() {
            src
        } else {
            return 1;
        };

        if forwards {
            (snippet.ceil_char_boundary(end_index + 1) - end_index) as u32
        } else {
            (end_index - snippet.floor_char_boundary(end_index - 1)) as u32
        }
    }

    pub fn get_source_file(&self, filename: &FileName) -> Option<Lrc<SourceFile>> {
        // Remap filename before lookup
        let filename = self.path_mapping().map_filename_prefix(filename).0;
        for sf in self.files.borrow().source_files.iter() {
            if filename == sf.name {
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

    /// Returns the index of the [`SourceFile`] (in `self.files`) that contains `pos`.
    /// This index is guaranteed to be valid for the lifetime of this `SourceMap`,
    /// since `source_files` is a `MonotonicVec`
    pub fn lookup_source_file_idx(&self, pos: BytePos) -> usize {
        self.files.borrow().source_files.partition_point(|x| x.start_pos <= pos) - 1
    }

    pub fn count_lines(&self) -> usize {
        self.files().iter().fold(0, |a, f| a + f.count_lines())
    }

    pub fn ensure_source_file_source_present(&self, source_file: &SourceFile) -> bool {
        source_file.add_external_src(|| {
            let FileName::Real(ref name) = source_file.name else {
                return None;
            };

            let local_path: Cow<'_, Path> = match name {
                RealFileName::LocalPath(local_path) => local_path.into(),
                RealFileName::Remapped { local_path: Some(local_path), .. } => local_path.into(),
                RealFileName::Remapped { local_path: None, virtual_name } => {
                    // The compiler produces better error messages if the sources of dependencies
                    // are available. Attempt to undo any path mapping so we can find remapped
                    // dependencies.
                    // We can only use the heuristic because `add_external_src` checks the file
                    // content hash.
                    self.path_mapping.reverse_map_prefix_heuristically(virtual_name)?.into()
                }
            };

            self.file_loader.read_file(&local_path).ok()
        })
    }

    pub fn is_imported(&self, sp: Span) -> bool {
        let source_file_index = self.lookup_source_file_idx(sp.lo());
        let source_file = &self.files()[source_file_index];
        source_file.is_imported()
    }

    /// Gets the span of a statement. If the statement is a macro expansion, the
    /// span in the context of the block span is found. The trailing semicolon is included
    /// on a best-effort basis.
    pub fn stmt_span(&self, stmt_span: Span, block_span: Span) -> Span {
        if !stmt_span.from_expansion() {
            return stmt_span;
        }
        let mac_call = original_sp(stmt_span, block_span);
        self.mac_call_stmt_semi_span(mac_call).map_or(mac_call, |s| mac_call.with_hi(s.hi()))
    }

    /// Tries to find the span of the semicolon of a macro call statement.
    /// The input must be the *call site* span of a statement from macro expansion.
    /// ```ignore (illustrative)
    /// //       v output
    ///    mac!();
    /// // ^^^^^^ input
    /// ```
    pub fn mac_call_stmt_semi_span(&self, mac_call: Span) -> Option<Span> {
        let span = self.span_extend_while_whitespace(mac_call);
        let span = self.next_point(span);
        if self.span_to_snippet(span).as_deref() == Ok(";") { Some(span) } else { None }
    }
}

pub fn get_source_map() -> Option<Lrc<SourceMap>> {
    with_session_globals(|session_globals| session_globals.source_map.clone())
}

#[derive(Clone)]
pub struct FilePathMapping {
    mapping: Vec<(PathBuf, PathBuf)>,
    filename_display_for_diagnostics: FileNameDisplayPreference,
}

impl FilePathMapping {
    pub fn empty() -> FilePathMapping {
        FilePathMapping::new(Vec::new(), FileNameDisplayPreference::Local)
    }

    pub fn new(
        mapping: Vec<(PathBuf, PathBuf)>,
        filename_display_for_diagnostics: FileNameDisplayPreference,
    ) -> FilePathMapping {
        FilePathMapping { mapping, filename_display_for_diagnostics }
    }

    /// Applies any path prefix substitution as defined by the mapping.
    /// The return value is the remapped path and a boolean indicating whether
    /// the path was affected by the mapping.
    pub fn map_prefix<'a>(&'a self, path: impl Into<Cow<'a, Path>>) -> (Cow<'a, Path>, bool) {
        let path = path.into();
        if path.as_os_str().is_empty() {
            // Exit early if the path is empty and therefore there's nothing to remap.
            // This is mostly to reduce spam for `RUSTC_LOG=[remap_path_prefix]`.
            return (path, false);
        }

        return remap_path_prefix(&self.mapping, path);

        #[instrument(level = "debug", skip(mapping), ret)]
        fn remap_path_prefix<'a>(
            mapping: &'a [(PathBuf, PathBuf)],
            path: Cow<'a, Path>,
        ) -> (Cow<'a, Path>, bool) {
            // NOTE: We are iterating over the mapping entries from last to first
            //       because entries specified later on the command line should
            //       take precedence.
            for (from, to) in mapping.iter().rev() {
                debug!("Trying to apply {from:?} => {to:?}");

                if let Ok(rest) = path.strip_prefix(from) {
                    let remapped = if rest.as_os_str().is_empty() {
                        // This is subtle, joining an empty path onto e.g. `foo/bar` will
                        // result in `foo/bar/`, that is, there'll be an additional directory
                        // separator at the end. This can lead to duplicated directory separators
                        // in remapped paths down the line.
                        // So, if we have an exact match, we just return that without a call
                        // to `Path::join()`.
                        to.into()
                    } else {
                        to.join(rest).into()
                    };
                    debug!("Match - remapped");

                    return (remapped, true);
                } else {
                    debug!("No match - prefix {from:?} does not match");
                }
            }

            debug!("not remapped");
            (path, false)
        }
    }

    fn map_filename_prefix(&self, file: &FileName) -> (FileName, bool) {
        match file {
            FileName::Real(realfile) if let RealFileName::LocalPath(local_path) = realfile => {
                let (mapped_path, mapped) = self.map_prefix(local_path);
                let realfile = if mapped {
                    RealFileName::Remapped {
                        local_path: Some(local_path.clone()),
                        virtual_name: mapped_path.into_owned(),
                    }
                } else {
                    realfile.clone()
                };
                (FileName::Real(realfile), mapped)
            }
            FileName::Real(_) => unreachable!("attempted to remap an already remapped filename"),
            other => (other.clone(), false),
        }
    }

    /// Applies any path prefix substitution as defined by the mapping.
    /// The return value is the local path with a "virtual path" representing the remapped
    /// part if any remapping was performed.
    pub fn to_real_filename<'a>(&self, local_path: impl Into<Cow<'a, Path>>) -> RealFileName {
        let local_path = local_path.into();
        if let (remapped_path, true) = self.map_prefix(&*local_path) {
            RealFileName::Remapped {
                virtual_name: remapped_path.into_owned(),
                local_path: Some(local_path.into_owned()),
            }
        } else {
            RealFileName::LocalPath(local_path.into_owned())
        }
    }

    /// Expand a relative path to an absolute path with remapping taken into account.
    /// Use this when absolute paths are required (e.g. debuginfo or crate metadata).
    ///
    /// The resulting `RealFileName` will have its `local_path` portion erased if
    /// possible (i.e. if there's also a remapped path).
    pub fn to_embeddable_absolute_path(
        &self,
        file_path: RealFileName,
        working_directory: &RealFileName,
    ) -> RealFileName {
        match file_path {
            // Anything that's already remapped we don't modify, except for erasing
            // the `local_path` portion.
            RealFileName::Remapped { local_path: _, virtual_name } => {
                RealFileName::Remapped {
                    // We do not want any local path to be exported into metadata
                    local_path: None,
                    // We use the remapped name verbatim, even if it looks like a relative
                    // path. The assumption is that the user doesn't want us to further
                    // process paths that have gone through remapping.
                    virtual_name,
                }
            }

            RealFileName::LocalPath(unmapped_file_path) => {
                // If no remapping has been applied yet, try to do so
                let (new_path, was_remapped) = self.map_prefix(unmapped_file_path);
                if was_remapped {
                    // It was remapped, so don't modify further
                    return RealFileName::Remapped {
                        local_path: None,
                        virtual_name: new_path.into_owned(),
                    };
                }

                if new_path.is_absolute() {
                    // No remapping has applied to this path and it is absolute,
                    // so the working directory cannot influence it either, so
                    // we are done.
                    return RealFileName::LocalPath(new_path.into_owned());
                }

                debug_assert!(new_path.is_relative());
                let unmapped_file_path_rel = new_path;

                match working_directory {
                    RealFileName::LocalPath(unmapped_working_dir_abs) => {
                        let file_path_abs = unmapped_working_dir_abs.join(unmapped_file_path_rel);

                        // Although neither `working_directory` nor the file name were subject
                        // to path remapping, the concatenation between the two may be. Hence
                        // we need to do a remapping here.
                        let (file_path_abs, was_remapped) = self.map_prefix(file_path_abs);
                        if was_remapped {
                            RealFileName::Remapped {
                                // Erase the actual path
                                local_path: None,
                                virtual_name: file_path_abs.into_owned(),
                            }
                        } else {
                            // No kind of remapping applied to this path, so
                            // we leave it as it is.
                            RealFileName::LocalPath(file_path_abs.into_owned())
                        }
                    }
                    RealFileName::Remapped {
                        local_path: _,
                        virtual_name: remapped_working_dir_abs,
                    } => {
                        // If working_directory has been remapped, then we emit
                        // Remapped variant as the expanded path won't be valid
                        RealFileName::Remapped {
                            local_path: None,
                            virtual_name: Path::new(remapped_working_dir_abs)
                                .join(unmapped_file_path_rel),
                        }
                    }
                }
            }
        }
    }

    /// Expand a relative path to an absolute path **without** remapping taken into account.
    ///
    /// The resulting `RealFileName` will have its `virtual_path` portion erased if
    /// possible (i.e. if there's also a remapped path).
    pub fn to_local_embeddable_absolute_path(
        &self,
        file_path: RealFileName,
        working_directory: &RealFileName,
    ) -> RealFileName {
        let file_path = file_path.local_path_if_available();
        if file_path.is_absolute() {
            // No remapping has applied to this path and it is absolute,
            // so the working directory cannot influence it either, so
            // we are done.
            return RealFileName::LocalPath(file_path.to_path_buf());
        }
        debug_assert!(file_path.is_relative());
        let working_directory = working_directory.local_path_if_available();
        RealFileName::LocalPath(Path::new(working_directory).join(file_path))
    }

    /// Attempts to (heuristically) reverse a prefix mapping.
    ///
    /// Returns [`Some`] if there is exactly one mapping where the "to" part is
    /// a prefix of `path` and has at least one non-empty
    /// [`Normal`](path::Component::Normal) component. The component
    /// restriction exists to avoid reverse mapping overly generic paths like
    /// `/` or `.`).
    ///
    /// This is a heuristic and not guaranteed to return the actual original
    /// path! Do not rely on the result unless you have other means to verify
    /// that the mapping is correct (e.g. by checking the file content hash).
    #[instrument(level = "debug", skip(self), ret)]
    fn reverse_map_prefix_heuristically(&self, path: &Path) -> Option<PathBuf> {
        let mut found = None;

        for (from, to) in self.mapping.iter() {
            let has_normal_component = to.components().any(|c| match c {
                path::Component::Normal(s) => !s.is_empty(),
                _ => false,
            });

            if !has_normal_component {
                continue;
            }

            let Ok(rest) = path.strip_prefix(to) else {
                continue;
            };

            if found.is_some() {
                return None;
            }

            found = Some(from.join(rest));
        }

        found
    }
}
