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
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::sync::{AtomicU32, Lrc, MappedReadGuard, ReadGuard, RwLock};
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::{clone::Clone, cmp};
use std::{convert::TryFrom, unreachable};

use std::fs;
use std::io;
use tracing::debug;

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
    // A hash of the source file's FileName. This is hash so that it's size
    // is more predictable than if we included the actual FileName value.
    pub file_name_hash: u64,

    // The CrateNum of the crate this source file was originally parsed for.
    // We cannot include this information in the hash because at the time
    // of hashing we don't have the context to map from the CrateNum's numeric
    // value to a StableCrateId.
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
    file_loader: Box<dyn FileLoader + Sync + Send>,
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
            file_loader,
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
        name_hash: u128,
        source_len: usize,
        cnum: CrateNum,
        mut file_local_lines: Vec<BytePos>,
        mut file_local_multibyte_chars: Vec<MultiByteChar>,
        mut file_local_non_narrow_chars: Vec<NonNarrowChar>,
        mut file_local_normalized_pos: Vec<NormalizedPos>,
        original_start_pos: BytePos,
        original_end_pos: BytePos,
    ) -> Lrc<SourceFile> {
        let start_pos = self
            .allocate_address_space(source_len)
            .expect("not enough address space for imported source file");

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
            src: None,
            src_hash,
            external_src: Lock::new(ExternalSource::Foreign {
                kind: ExternalSourceKind::AbsentOk,
                original_start_pos,
                original_end_pos,
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

    // If there is a doctest offset, applies it to the line.
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

    // If the corresponding `SourceFile` is empty, does not return a line number.
    pub fn lookup_line(&self, pos: BytePos) -> Result<SourceFileAndLine, Lrc<SourceFile>> {
        let f = self.lookup_source_file(pos);

        match f.lookup_line(pos) {
            Some(line) => Ok(SourceFileAndLine { sf: f, line }),
            None => Err(f),
        }
    }

    fn span_to_string(&self, sp: Span, filename_display_pref: FileNameDisplayPreference) -> String {
        if self.files.borrow().source_files.is_empty() || sp.is_dummy() {
            return "no-location".to_string();
        }

        let lo = self.lookup_char_pos(sp.lo());
        let hi = self.lookup_char_pos(sp.hi());
        format!(
            "{}:{}:{}: {}:{}",
            lo.file.name.display(filename_display_pref),
            lo.line,
            lo.col.to_usize() + 1,
            hi.line,
            hi.col.to_usize() + 1,
        )
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
        f.lookup_line(sp.lo()) != f.lookup_line(sp.hi())
    }

    #[instrument(skip(self), level = "trace")]
    pub fn is_valid_span(&self, sp: Span) -> Result<(Loc, Loc), SpanLinesError> {
        let lo = self.lookup_char_pos(sp.lo());
        trace!(?lo);
        let hi = self.lookup_char_pos(sp.hi());
        trace!(?hi);
        if lo.file.start_pos != hi.file.start_pos {
            return Err(SpanLinesError::DistinctSources(DistinctSources {
                begin: (lo.file.name.clone(), lo.file.start_pos),
                end: (hi.file.name.clone(), hi.file.start_pos),
            }));
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
            Err(SpanSnippetError::DistinctSources(DistinctSources {
                begin: (local_begin.sf.name.clone(), local_begin.sf.start_pos),
                end: (local_end.sf.name.clone(), local_end.sf.start_pos),
            }))
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

    /// Returns whether or not this span points into a file
    /// in the current crate. This may be `false` for spans
    /// produced by a macro expansion, or for spans associated
    /// with the definition of an item in a foreign crate
    pub fn is_local_span(&self, sp: Span) -> bool {
        let local_begin = self.lookup_byte_offset(sp.lo());
        let local_end = self.lookup_byte_offset(sp.hi());
        // This might be a weird span that covers multiple files
        local_begin.sf.src.is_some() && local_end.sf.src.is_some()
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
        match self.span_to_prev_source(sp) {
            Err(_) => None,
            Ok(source) => {
                let last_line = source.rsplit_once('\n').unwrap_or(("", &source)).1;

                Some(last_line.len() - last_line.trim_start().len())
            }
        }
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
                let prev_source = prev_source.rsplit(&pat).next().unwrap_or("").trim_start();
                if prev_source.is_empty() && sp.lo().0 != 0 {
                    return sp.with_lo(BytePos(sp.lo().0 - 1));
                } else if accept_newlines || !prev_source.contains('\n') {
                    return sp.with_lo(BytePos(sp.lo().0 - prev_source.len() as u32));
                }
            }
        }

        sp
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

    /// Extends the given `Span` to just after the next occurrence of `c`.
    pub fn span_extend_to_next_char(&self, sp: Span, c: char, accept_newlines: bool) -> Span {
        if let Ok(next_source) = self.span_to_next_source(sp) {
            let next_source = next_source.split(c).next().unwrap_or("");
            if !next_source.is_empty() && (accept_newlines || !next_source.contains('\n')) {
                return sp.with_hi(BytePos(sp.hi().0 + next_source.len() as u32));
            }
        }

        sp
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
            } else if let Some(src) = src.get_source() {
                Some(&src[start_index..])
            } else {
                None
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
    pub fn next_point(&self, sp: Span) -> Span {
        if sp.is_dummy() {
            return sp;
        }
        let start_of_next_point = sp.hi().0;

        let width = self.find_width_of_character_at_span(sp.shrink_to_hi(), true);
        // If the width is 1, then the next span should point to the same `lo` and `hi`. However,
        // in the case of a multibyte character, where the width != 1, the next span should
        // span multiple bytes to include the whole character.
        let end_of_next_point =
            start_of_next_point.checked_add(width - 1).unwrap_or(start_of_next_point);

        let end_of_next_point = BytePos(cmp::max(sp.lo().0 + 1, end_of_next_point));
        Span::new(BytePos(start_of_next_point), end_of_next_point, sp.ctxt(), None)
    }

    /// Finds the width of the character, either before or after the end of provided span,
    /// depending on the `forwards` parameter.
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
        if (!forwards && end_index == usize::MIN) || (forwards && start_index == usize::MAX) {
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
            &src[start_index..]
        } else if let Some(src) = src.get_source() {
            &src[start_index..]
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

    // Returns the index of the `SourceFile` (in `self.files`) that contains `pos`.
    // This index is guaranteed to be valid for the lifetime of this `SourceMap`,
    // since `source_files` is a `MonotonicVec`
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
        if let Ok(snippet) = self.span_to_snippet(prev_span) {
            debug!(
                "generate_fn_name_span: span={:?}, prev_span={:?}, snippet={:?}",
                span, prev_span, snippet
            );

            if snippet.is_empty() {
                return None;
            };

            let len = snippet
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .expect("no label after fn");
            Some(prev_span.with_hi(BytePos(prev_span.lo().0 + len as u32)))
        } else {
            None
        }
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
        source_file.add_external_src(|| {
            match source_file.name {
                FileName::Real(ref name) if let Some(local_path) = name.local_path() => {
                    self.file_loader.read_file(local_path).ok()
                }
                _ => None,
            }
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
    ///
    ///           v output
    ///     mac!();
    ///     ^^^^^^ input
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

    fn map_filename_prefix(&self, file: &FileName) -> (FileName, bool) {
        match file {
            FileName::Real(realfile) if let RealFileName::LocalPath(local_path) = realfile => {
                let (mapped_path, mapped) = self.map_prefix(local_path.to_path_buf());
                let realfile = if mapped {
                    RealFileName::Remapped {
                        local_path: Some(local_path.clone()),
                        virtual_name: mapped_path,
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
}
