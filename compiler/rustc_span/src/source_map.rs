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

pub use crate::hygiene::{ExpnData, ExpnKind};
pub use crate::*;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{Hash128, Hash64, StableHasher};
use rustc_data_structures::sync::{
    AtomicU32, IntoDynSyncSend, Lrc, MappedReadGuard, ReadGuard, RwLock,
};
use std::cmp;
use std::hash::Hash;
use std::path::{self, Path, PathBuf};
use std::sync::atomic::Ordering;

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

pub mod monotonic {
    use std::ops::{Deref, DerefMut};

    /// A `MonotonicVec` is a `Vec` which can only be grown.
    /// Once inserted, an element can never be removed or swapped,
    /// guaranteeing that any indices into a `MonotonicVec` are stable
    // This is declared in its own module to ensure that the private
    // field is inaccessible
    pub struct MonotonicVec<T>(Vec<T>);
    impl<T> MonotonicVec<T> {
        pub fn new(val: Vec<T>) -> MonotonicVec<T> {
            MonotonicVec(val)
        }

        pub fn push(&mut self, val: T) {
            self.0.push(val);
        }
    }

    impl<T> Default for MonotonicVec<T> {
        fn default() -> Self {
            MonotonicVec::new(vec![])
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

#[derive(Clone, Encodable, Decodable, Debug, Copy, HashStable_Generic)]
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
    fn read_file(&self, path: &Path) -> io::Result<String>;

    /// Read the contents of a potentially non-UTF-8 file into memory.
    fn read_binary_file(&self, path: &Path) -> io::Result<Vec<u8>>;
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

    fn read_binary_file(&self, path: &Path) -> io::Result<Vec<u8>> {
        fs::read(path)
    }
}

/// This is a [SourceFile] identifier that is used to correlate source files between
/// subsequent compilation sessions (which is something we need to do during
/// incremental compilation).
///
/// The [StableSourceFileId] also contains the CrateNum of the crate the source
/// file was originally parsed for. This way we get two separate entries in
/// the [SourceMap] if the same file is part of both the local and an upstream
/// crate. Trying to only have one entry for both cases is problematic because
/// at the point where we discover that there's a local use of the file in
/// addition to the upstream one, we might already have made decisions based on
/// the assumption that it's an upstream file. Treating the two files as
/// different has no real downsides.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Encodable, Decodable, Debug)]
pub struct StableSourceFileId {
    /// A hash of the source file's [`FileName`]. This is hash so that it's size
    /// is more predictable than if we included the actual [`FileName`] value.
    pub file_name_hash: Hash64,

    /// The [`CrateNum`] of the crate this source file was originally parsed for.
    /// We cannot include this information in the hash because at the time
    /// of hashing we don't have the context to map from the [`CrateNum`]'s numeric
    /// value to a `StableCrateId`.
    pub cnum: CrateNum,
}

// FIXME: we need a more globally consistent approach to the problem solved by
// StableSourceFileId, perhaps built atop source_file.name_hash.
impl StableSourceFileId {
    pub fn new(source_file: &SourceFile) -> StableSourceFileId {
        StableSourceFileId::new_from_name(&source_file.name, source_file.cnum)
    }

    fn new_from_name(name: &FileName, cnum: CrateNum) -> StableSourceFileId {
        let mut hasher = StableHasher::new();
        name.hash(&mut hasher);
        StableSourceFileId { file_name_hash: hasher.finish(), cnum }
    }
}

// _____________________________________________________________________________
// SourceMap
//

#[derive(Default)]
pub(super) struct SourceMapFiles {
    source_files: monotonic::MonotonicVec<Lrc<SourceFile>>,
    stable_id_to_source_file: FxHashMap<StableSourceFileId, Lrc<SourceFile>>,
}

pub struct SourceMap {
    /// The address space below this value is currently used by the files in the source map.
    used_address_space: AtomicU32,

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
        Self::with_file_loader_and_hash_kind(
            Box::new(RealFileLoader),
            path_mapping,
            SourceFileHashAlgorithm::Md5,
        )
    }

    pub fn with_file_loader_and_hash_kind(
        file_loader: Box<dyn FileLoader + Sync + Send>,
        path_mapping: FilePathMapping,
        hash_kind: SourceFileHashAlgorithm,
    ) -> SourceMap {
        SourceMap {
            used_address_space: AtomicU32::new(0),
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
    pub fn load_binary_file(&self, path: &Path) -> io::Result<Vec<u8>> {
        let bytes = self.file_loader.read_binary_file(path)?;

        // We need to add file to the `SourceMap`, so that it is present
        // in dep-info. There's also an edge case that file might be both
        // loaded as a binary via `include_bytes!` and as proper `SourceFile`
        // via `mod`, so we try to use real file contents and not just an
        // empty string.
        let text = std::str::from_utf8(&bytes).unwrap_or("").to_string();
        self.new_source_file(path.to_owned().into(), text);
        Ok(bytes)
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

    fn allocate_address_space(&self, size: usize) -> Result<usize, OffsetOverflowError> {
        let size = u32::try_from(size).map_err(|_| OffsetOverflowError)?;

        loop {
            let current = self.used_address_space.load(Ordering::Relaxed);
            let next = current
                .checked_add(size)
                // Add one so there is some space between files. This lets us distinguish
                // positions in the `SourceMap`, even in the presence of zero-length files.
                .and_then(|next| next.checked_add(1))
                .ok_or(OffsetOverflowError)?;

            if self
                .used_address_space
                .compare_exchange(current, next, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(usize::try_from(current).unwrap());
            }
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
        // Note that filename may not be a valid path, eg it may be `<anon>` etc,
        // but this is okay because the directory determined by `path.pop()` will
        // be empty, so the working directory will be used.
        let (filename, _) = self.path_mapping.map_filename_prefix(&filename);

        let file_id = StableSourceFileId::new_from_name(&filename, LOCAL_CRATE);

        let lrc_sf = match self.source_file_by_stable_id(file_id) {
            Some(lrc_sf) => lrc_sf,
            None => {
                let start_pos = self.allocate_address_space(src.len())?;

                let source_file = Lrc::new(SourceFile::new(
                    filename,
                    src,
                    Pos::from_usize(start_pos),
                    self.hash_kind,
                ));

                // Let's make sure the file_id we generated above actually matches
                // the ID we generate for the SourceFile we just created.
                debug_assert_eq!(StableSourceFileId::new(&source_file), file_id);

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
        src_hash: SourceFileHash,
        name_hash: Hash128,
        source_len: usize,
        cnum: CrateNum,
        file_local_lines: Lock<SourceFileLines>,
        mut file_local_multibyte_chars: Vec<MultiByteChar>,
        mut file_local_non_narrow_chars: Vec<NonNarrowChar>,
        mut file_local_normalized_pos: Vec<NormalizedPos>,
        original_start_pos: BytePos,
        metadata_index: u32,
    ) -> Lrc<SourceFile> {
        let start_pos = self
            .allocate_address_space(source_len)
            .expect("not enough address space for imported source file");

        let end_pos = Pos::from_usize(start_pos + source_len);
        let start_pos = Pos::from_usize(start_pos);

        // Translate these positions into the new global frame of reference,
        // now that the offset of the SourceFile is known.
        //
        // These are all unsigned values. `original_start_pos` may be larger or
        // smaller than `start_pos`, but `pos` is always larger than both.
        // Therefore, `(pos - original_start_pos) + start_pos` won't overflow
        // but `start_pos - original_start_pos` might. So we use the former
        // form rather than pre-computing the offset into a local variable. The
        // compiler backend can optimize away the repeated computations in a
        // way that won't trigger overflow checks.
        match &mut *file_local_lines.borrow_mut() {
            SourceFileLines::Lines(lines) => {
                for pos in lines {
                    *pos = (*pos - original_start_pos) + start_pos;
                }
            }
            SourceFileLines::Diffs(SourceFileDiffs { line_start, .. }) => {
                *line_start = (*line_start - original_start_pos) + start_pos;
            }
        }
        for mbc in &mut file_local_multibyte_chars {
            mbc.pos = (mbc.pos - original_start_pos) + start_pos;
        }
        for swc in &mut file_local_non_narrow_chars {
            *swc = (*swc - original_start_pos) + start_pos;
        }
        for nc in &mut file_local_normalized_pos {
            nc.pos = (nc.pos - original_start_pos) + start_pos;
        }

        let source_file = Lrc::new(SourceFile {
            name: filename,
            src: None,
            src_hash,
            external_src: Lock::new(ExternalSource::Foreign {
                kind: ExternalSourceKind::AbsentOk,
                metadata_index,
            }),
            start_pos,
            end_pos,
            lines: file_local_lines,
            multibyte_chars: file_local_multibyte_chars,
            non_narrow_chars: file_local_non_narrow_chars,
            normalized_pos: file_local_normalized_pos,
            name_hash,
            cnum,
        });

        let mut files = self.files.borrow_mut();

        files.source_files.push(source_file.clone());
        files
            .stable_id_to_source_file
            .insert(StableSourceFileId::new(&source_file), source_file.clone());

        source_file
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

    /// Format the span location suitable for pretty printing annotations with relative line numbers
    pub fn span_to_relative_line_string(&self, sp: Span, relative_to: Span) -> String {
        if self.files.borrow().source_files.is_empty() || sp.is_dummy() || relative_to.is_dummy() {
            return "no-location".to_string();
        }

        let lo = self.lookup_char_pos(sp.lo());
        let hi = self.lookup_char_pos(sp.hi());
        let offset = self.lookup_char_pos(relative_to.lo());

        if lo.file.name != offset.file.name || !relative_to.contains(sp) {
            return self.span_to_embeddable_string(sp);
        }

        let lo_line = lo.line.saturating_sub(offset.line);
        let hi_line = hi.line.saturating_sub(offset.line);

        format!(
            "{}:+{}:{}: +{}:{}",
            lo.file.name.display(FileNameDisplayPreference::Remapped),
            lo_line,
            lo.col.to_usize() + 1,
            hi_line,
            hi.col.to_usize() + 1,
        )
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
        f.lookup_line(sp.lo()) != f.lookup_line(sp.hi())
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
                extract_source(src, start_index, end_index)
            } else if let Some(src) = local_begin.sf.external_src.borrow().get_source() {
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
            let src = local_begin.sf.external_src.borrow();

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

    /// Returns a new span to check next none-whitespace character or some specified expected character
    /// If `expect` is none, the first span of non-whitespace character is returned.
    /// If `expect` presented, the first span of the character `expect` is returned
    /// Otherwise, the span reached to limit is returned.
    pub fn span_look_ahead(&self, span: Span, expect: Option<&str>, limit: Option<usize>) -> Span {
        let mut sp = span;
        for _ in 0..limit.unwrap_or(100_usize) {
            sp = self.next_point(sp);
            if let Ok(ref snippet) = self.span_to_snippet(sp) {
                if expect.is_some_and(|es| snippet == es) {
                    break;
                }
                if expect.is_none() && snippet.chars().any(|c| !c.is_whitespace()) {
                    break;
                }
            }
        }
        sp
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

        let source_len = (local_begin.sf.end_pos - local_begin.sf.start_pos).to_usize();
        debug!("source_len=`{:?}`", source_len);
        // Ensure indexes are also not malformed.
        if start_index > end_index || end_index > source_len - 1 {
            debug!("source indexes are malformed");
            return 1;
        }

        let src = local_begin.sf.external_src.borrow();

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
        self.files
            .borrow()
            .source_files
            .binary_search_by_key(&pos, |key| key.start_pos)
            .unwrap_or_else(|p| p - 1)
    }

    pub fn count_lines(&self) -> usize {
        self.files().iter().fold(0, |a, f| a + f.count_lines())
    }

    pub fn ensure_source_file_source_present(&self, source_file: Lrc<SourceFile>) -> bool {
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
        let span = self.span_extend_while(mac_call, char::is_whitespace).ok()?;
        let span = span.shrink_to_hi().with_hi(BytePos(span.hi().0.checked_add(1)?));
        if self.span_to_snippet(span).as_deref() != Ok(";") {
            return None;
        }
        Some(span)
    }
}

#[derive(Clone)]
pub struct FilePathMapping {
    mapping: Vec<(PathBuf, PathBuf)>,
    filename_display_for_diagnostics: FileNameDisplayPreference,
}

impl FilePathMapping {
    pub fn empty() -> FilePathMapping {
        FilePathMapping::new(Vec::new())
    }

    pub fn new(mapping: Vec<(PathBuf, PathBuf)>) -> FilePathMapping {
        let filename_display_for_diagnostics = if mapping.is_empty() {
            FileNameDisplayPreference::Local
        } else {
            FileNameDisplayPreference::Remapped
        };

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
