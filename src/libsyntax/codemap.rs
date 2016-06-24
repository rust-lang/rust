// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The CodeMap tracks all the source code used within a single crate, mapping
//! from integer byte positions to the original source code location. Each bit
//! of source parsed during crate parsing (typically files, in-memory strings,
//! or various bits of macro expansion) cover a continuous range of bytes in the
//! CodeMap and are represented by FileMaps. Byte positions are stored in
//! `spans` and used pervasively in the compiler. They are absolute positions
//! within the CodeMap, which upon request can be converted to line and column
//! information, source code snippets, etc.

pub use self::ExpnFormat::*;

use std::cell::RefCell;
use std::path::{Path,PathBuf};
use std::rc::Rc;

use std::env;
use std::fs;
use std::io::{self, Read};
pub use syntax_pos::*;
use errors::CodeMapper;

use ast::Name;

/// Return the span itself if it doesn't come from a macro expansion,
/// otherwise return the call site span up to the `enclosing_sp` by
/// following the `expn_info` chain.
pub fn original_sp(cm: &CodeMap, sp: Span, enclosing_sp: Span) -> Span {
    let call_site1 = cm.with_expn_info(sp.expn_id, |ei| ei.map(|ei| ei.call_site));
    let call_site2 = cm.with_expn_info(enclosing_sp.expn_id, |ei| ei.map(|ei| ei.call_site));
    match (call_site1, call_site2) {
        (None, _) => sp,
        (Some(call_site1), Some(call_site2)) if call_site1 == call_site2 => sp,
        (Some(call_site1), _) => original_sp(cm, call_site1, enclosing_sp),
    }
}

/// The source of expansion.
#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub enum ExpnFormat {
    /// e.g. #[derive(...)] <item>
    MacroAttribute(Name),
    /// e.g. `format!()`
    MacroBang(Name),
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

pub fn spanned<T>(lo: BytePos, hi: BytePos, t: T) -> Spanned<T> {
    respan(mk_sp(lo, hi), t)
}

pub fn respan<T>(sp: Span, t: T) -> Spanned<T> {
    Spanned {node: t, span: sp}
}

pub fn dummy_spanned<T>(t: T) -> Spanned<T> {
    respan(DUMMY_SP, t)
}

#[derive(Clone, Hash, Debug)]
pub struct NameAndSpan {
    /// The format with which the macro was invoked.
    pub format: ExpnFormat,
    /// Whether the macro is allowed to use #[unstable]/feature-gated
    /// features internally without forcing the whole crate to opt-in
    /// to them.
    pub allow_internal_unstable: bool,
    /// The span of the macro definition itself. The macro may not
    /// have a sensible definition span (e.g. something defined
    /// completely inside libsyntax) in which case this is None.
    pub span: Option<Span>
}

impl NameAndSpan {
    pub fn name(&self) -> Name {
        match self.format {
            ExpnFormat::MacroAttribute(s) => s,
            ExpnFormat::MacroBang(s) => s,
        }
    }
}

/// Extra information for tracking spans of macro and syntax sugar expansion
#[derive(Hash, Debug)]
pub struct ExpnInfo {
    /// The location of the actual macro invocation or syntax sugar , e.g.
    /// `let x = foo!();` or `if let Some(y) = x {}`
    ///
    /// This may recursively refer to other macro invocations, e.g. if
    /// `foo!()` invoked `bar!()` internally, and there was an
    /// expression inside `bar!`; the call_site of the expression in
    /// the expansion would point to the `bar!` invocation; that
    /// call_site span would have its own ExpnInfo, with the call_site
    /// pointing to the `foo!` invocation.
    pub call_site: Span,
    /// Information about the expansion.
    pub callee: NameAndSpan
}

// _____________________________________________________________________________
// FileMap, MultiByteChar, FileName, FileLines
//

/// An abstraction over the fs operations used by the Parser.
pub trait FileLoader {
    /// Query the existence of a file.
    fn file_exists(&self, path: &Path) -> bool;

    /// Return an absolute path to a file, if possible.
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
            env::current_dir()
                .ok()
                .map(|cwd| cwd.join(path))
        }
    }

    fn read_file(&self, path: &Path) -> io::Result<String> {
        let mut src = String::new();
        fs::File::open(path)?.read_to_string(&mut src)?;
        Ok(src)
    }
}

// _____________________________________________________________________________
// CodeMap
//

pub struct CodeMap {
    pub files: RefCell<Vec<Rc<FileMap>>>,
    expansions: RefCell<Vec<ExpnInfo>>,
    file_loader: Box<FileLoader>
}

impl CodeMap {
    pub fn new() -> CodeMap {
        CodeMap {
            files: RefCell::new(Vec::new()),
            expansions: RefCell::new(Vec::new()),
            file_loader: Box::new(RealFileLoader)
        }
    }

    pub fn with_file_loader(file_loader: Box<FileLoader>) -> CodeMap {
        CodeMap {
            files: RefCell::new(Vec::new()),
            expansions: RefCell::new(Vec::new()),
            file_loader: file_loader
        }
    }

    pub fn file_exists(&self, path: &Path) -> bool {
        self.file_loader.file_exists(path)
    }

    pub fn load_file(&self, path: &Path) -> io::Result<Rc<FileMap>> {
        let src = self.file_loader.read_file(path)?;
        let abs_path = self.file_loader.abs_path(path).map(|p| p.to_str().unwrap().to_string());
        Ok(self.new_filemap(path.to_str().unwrap().to_string(), abs_path, src))
    }

    fn next_start_pos(&self) -> usize {
        let files = self.files.borrow();
        match files.last() {
            None => 0,
            // Add one so there is some space between files. This lets us distinguish
            // positions in the codemap, even in the presence of zero-length files.
            Some(last) => last.end_pos.to_usize() + 1,
        }
    }

    /// Creates a new filemap without setting its line information. If you don't
    /// intend to set the line information yourself, you should use new_filemap_and_lines.
    pub fn new_filemap(&self, filename: FileName, abs_path: Option<FileName>,
                       mut src: String) -> Rc<FileMap> {
        let start_pos = self.next_start_pos();
        let mut files = self.files.borrow_mut();

        // Remove utf-8 BOM if any.
        if src.starts_with("\u{feff}") {
            src.drain(..3);
        }

        let end_pos = start_pos + src.len();

        let filemap = Rc::new(FileMap {
            name: filename,
            abs_path: abs_path,
            src: Some(Rc::new(src)),
            start_pos: Pos::from_usize(start_pos),
            end_pos: Pos::from_usize(end_pos),
            lines: RefCell::new(Vec::new()),
            multibyte_chars: RefCell::new(Vec::new()),
        });

        files.push(filemap.clone());

        filemap
    }

    /// Creates a new filemap and sets its line information.
    pub fn new_filemap_and_lines(&self, filename: &str, abs_path: Option<&str>,
                                 src: &str) -> Rc<FileMap> {
        let fm = self.new_filemap(filename.to_string(),
                                  abs_path.map(|s| s.to_owned()),
                                  src.to_owned());
        let mut byte_pos: u32 = fm.start_pos.0;
        for line in src.lines() {
            // register the start of this line
            fm.next_line(BytePos(byte_pos));

            // update byte_pos to include this line and the \n at the end
            byte_pos += line.len() as u32 + 1;
        }
        fm
    }


    /// Allocates a new FileMap representing a source file from an external
    /// crate. The source code of such an "imported filemap" is not available,
    /// but we still know enough to generate accurate debuginfo location
    /// information for things inlined from other crates.
    pub fn new_imported_filemap(&self,
                                filename: FileName,
                                abs_path: Option<FileName>,
                                source_len: usize,
                                mut file_local_lines: Vec<BytePos>,
                                mut file_local_multibyte_chars: Vec<MultiByteChar>)
                                -> Rc<FileMap> {
        let start_pos = self.next_start_pos();
        let mut files = self.files.borrow_mut();

        let end_pos = Pos::from_usize(start_pos + source_len);
        let start_pos = Pos::from_usize(start_pos);

        for pos in &mut file_local_lines {
            *pos = *pos + start_pos;
        }

        for mbc in &mut file_local_multibyte_chars {
            mbc.pos = mbc.pos + start_pos;
        }

        let filemap = Rc::new(FileMap {
            name: filename,
            abs_path: abs_path,
            src: None,
            start_pos: start_pos,
            end_pos: end_pos,
            lines: RefCell::new(file_local_lines),
            multibyte_chars: RefCell::new(file_local_multibyte_chars),
        });

        files.push(filemap.clone());

        filemap
    }

    pub fn mk_substr_filename(&self, sp: Span) -> String {
        let pos = self.lookup_char_pos(sp.lo);
        (format!("<{}:{}:{}>",
                 pos.file.name,
                 pos.line,
                 pos.col.to_usize() + 1)).to_string()
    }

    /// Lookup source information about a BytePos
    pub fn lookup_char_pos(&self, pos: BytePos) -> Loc {
        let chpos = self.bytepos_to_file_charpos(pos);
        match self.lookup_line(pos) {
            Ok(FileMapAndLine { fm: f, line: a }) => {
                let line = a + 1; // Line numbers start at 1
                let linebpos = (*f.lines.borrow())[a];
                let linechpos = self.bytepos_to_file_charpos(linebpos);
                debug!("byte pos {:?} is on the line at byte pos {:?}",
                       pos, linebpos);
                debug!("char pos {:?} is on the line at char pos {:?}",
                       chpos, linechpos);
                debug!("byte is on line: {}", line);
                assert!(chpos >= linechpos);
                Loc {
                    file: f,
                    line: line,
                    col: chpos - linechpos,
                }
            }
            Err(f) => {
                Loc {
                    file: f,
                    line: 0,
                    col: chpos,
                }
            }
        }
    }

    // If the relevant filemap is empty, we don't return a line number.
    fn lookup_line(&self, pos: BytePos) -> Result<FileMapAndLine, Rc<FileMap>> {
        let idx = self.lookup_filemap_idx(pos);

        let files = self.files.borrow();
        let f = (*files)[idx].clone();

        let len = f.lines.borrow().len();
        if len == 0 {
            return Err(f);
        }

        let mut a = 0;
        {
            let lines = f.lines.borrow();
            let mut b = lines.len();
            while b - a > 1 {
                let m = (a + b) / 2;
                if (*lines)[m] > pos {
                    b = m;
                } else {
                    a = m;
                }
            }
            assert!(a <= lines.len());
        }
        Ok(FileMapAndLine { fm: f, line: a })
    }

    pub fn lookup_char_pos_adj(&self, pos: BytePos) -> LocWithOpt {
        let loc = self.lookup_char_pos(pos);
        LocWithOpt {
            filename: loc.file.name.to_string(),
            line: loc.line,
            col: loc.col,
            file: Some(loc.file)
        }
    }

    pub fn span_to_string(&self, sp: Span) -> String {
        if sp == COMMAND_LINE_SP {
            return "<command line option>".to_string();
        }

        if self.files.borrow().is_empty() && sp.source_equal(&DUMMY_SP) {
            return "no-location".to_string();
        }

        let lo = self.lookup_char_pos_adj(sp.lo);
        let hi = self.lookup_char_pos_adj(sp.hi);
        return (format!("{}:{}:{}: {}:{}",
                        lo.filename,
                        lo.line,
                        lo.col.to_usize() + 1,
                        hi.line,
                        hi.col.to_usize() + 1)).to_string()
    }

    // Returns true if two spans have the same callee
    // (Assumes the same ExpnFormat implies same callee)
    fn match_callees(&self, sp_a: &Span, sp_b: &Span) -> bool {
        let fmt_a = self
            .with_expn_info(sp_a.expn_id,
                            |ei| ei.map(|ei| ei.callee.format.clone()));

        let fmt_b = self
            .with_expn_info(sp_b.expn_id,
                            |ei| ei.map(|ei| ei.callee.format.clone()));
        fmt_a == fmt_b
    }

    /// Returns a formatted string showing the expansion chain of a span
    ///
    /// Spans are printed in the following format:
    ///
    /// filename:start_line:col: end_line:col
    /// snippet
    ///   Callee:
    ///   Callee span
    ///   Callsite:
    ///   Callsite span
    ///
    /// Callees and callsites are printed recursively (if available, otherwise header
    /// and span is omitted), expanding into their own callee/callsite spans.
    /// Each layer of recursion has an increased indent, and snippets are truncated
    /// to at most 50 characters. Finally, recursive calls to the same macro are squashed,
    /// with '...' used to represent any number of recursive calls.
    pub fn span_to_expanded_string(&self, sp: Span) -> String {
        self.span_to_expanded_string_internal(sp, "")
    }

    fn span_to_expanded_string_internal(&self, sp:Span, indent: &str) -> String {
        let mut indent = indent.to_owned();
        let mut output = "".to_owned();
        let span_str = self.span_to_string(sp);
        let mut span_snip = self.span_to_snippet(sp)
            .unwrap_or("Snippet unavailable".to_owned());

        // Truncate by code points - in worst case this will be more than 50 characters,
        // but ensures at least 50 characters and respects byte boundaries.
        let char_vec: Vec<(usize, char)> = span_snip.char_indices().collect();
        if char_vec.len() > 50 {
            span_snip.truncate(char_vec[49].0);
            span_snip.push_str("...");
        }

        output.push_str(&format!("{}{}\n{}`{}`\n", indent, span_str, indent, span_snip));

        if sp.expn_id == NO_EXPANSION || sp.expn_id == COMMAND_LINE_EXPN {
            return output;
        }

        let mut callee = self.with_expn_info(sp.expn_id,
                                             |ei| ei.and_then(|ei| ei.callee.span.clone()));
        let mut callsite = self.with_expn_info(sp.expn_id,
                                               |ei| ei.map(|ei| ei.call_site.clone()));

        indent.push_str("  ");
        let mut is_recursive = false;

        while callee.is_some() && self.match_callees(&sp, &callee.unwrap()) {
            callee = self.with_expn_info(callee.unwrap().expn_id,
                                         |ei| ei.and_then(|ei| ei.callee.span.clone()));
            is_recursive = true;
        }
        if let Some(span) = callee {
            output.push_str(&indent);
            output.push_str("Callee:\n");
            if is_recursive {
                output.push_str(&indent);
                output.push_str("...\n");
            }
            output.push_str(&(self.span_to_expanded_string_internal(span, &indent)));
        }

        is_recursive = false;
        while callsite.is_some() && self.match_callees(&sp, &callsite.unwrap()) {
            callsite = self.with_expn_info(callsite.unwrap().expn_id,
                                           |ei| ei.map(|ei| ei.call_site.clone()));
            is_recursive = true;
        }
        if let Some(span) = callsite {
            output.push_str(&indent);
            output.push_str("Callsite:\n");
            if is_recursive {
                output.push_str(&indent);
                output.push_str("...\n");
            }
            output.push_str(&(self.span_to_expanded_string_internal(span, &indent)));
        }
        output
    }

    /// Return the source span - this is either the supplied span, or the span for
    /// the macro callsite that expanded to it.
    pub fn source_callsite(&self, sp: Span) -> Span {
        let mut span = sp;
        // Special case - if a macro is parsed as an argument to another macro, the source
        // callsite is the first callsite, which is also source-equivalent to the span.
        let mut first = true;
        while span.expn_id != NO_EXPANSION && span.expn_id != COMMAND_LINE_EXPN {
            if let Some(callsite) = self.with_expn_info(span.expn_id,
                                               |ei| ei.map(|ei| ei.call_site.clone())) {
                if first && span.source_equal(&callsite) {
                    if self.lookup_char_pos(span.lo).file.is_real_file() {
                        return Span { expn_id: NO_EXPANSION, .. span };
                    }
                }
                first = false;
                span = callsite;
            }
            else {
                break;
            }
        }
        span
    }

    /// Return the source callee.
    ///
    /// Returns None if the supplied span has no expansion trace,
    /// else returns the NameAndSpan for the macro definition
    /// corresponding to the source callsite.
    pub fn source_callee(&self, sp: Span) -> Option<NameAndSpan> {
        let mut span = sp;
        // Special case - if a macro is parsed as an argument to another macro, the source
        // callsite is source-equivalent to the span, and the source callee is the first callee.
        let mut first = true;
        while let Some(callsite) = self.with_expn_info(span.expn_id,
                                            |ei| ei.map(|ei| ei.call_site.clone())) {
            if first && span.source_equal(&callsite) {
                if self.lookup_char_pos(span.lo).file.is_real_file() {
                    return self.with_expn_info(span.expn_id,
                                               |ei| ei.map(|ei| ei.callee.clone()));
                }
            }
            first = false;
            if let Some(_) = self.with_expn_info(callsite.expn_id,
                                                 |ei| ei.map(|ei| ei.call_site.clone())) {
                span = callsite;
            }
            else {
                return self.with_expn_info(span.expn_id,
                                           |ei| ei.map(|ei| ei.callee.clone()));
            }
        }
        None
    }

    pub fn span_to_filename(&self, sp: Span) -> FileName {
        self.lookup_char_pos(sp.lo).file.name.to_string()
    }

    pub fn span_to_lines(&self, sp: Span) -> FileLinesResult {
        debug!("span_to_lines(sp={:?})", sp);

        if sp.lo > sp.hi {
            return Err(SpanLinesError::IllFormedSpan(sp));
        }

        let lo = self.lookup_char_pos(sp.lo);
        debug!("span_to_lines: lo={:?}", lo);
        let hi = self.lookup_char_pos(sp.hi);
        debug!("span_to_lines: hi={:?}", hi);

        if lo.file.start_pos != hi.file.start_pos {
            return Err(SpanLinesError::DistinctSources(DistinctSources {
                begin: (lo.file.name.clone(), lo.file.start_pos),
                end: (hi.file.name.clone(), hi.file.start_pos),
            }));
        }
        assert!(hi.line >= lo.line);

        let mut lines = Vec::with_capacity(hi.line - lo.line + 1);

        // The span starts partway through the first line,
        // but after that it starts from offset 0.
        let mut start_col = lo.col;

        // For every line but the last, it extends from `start_col`
        // and to the end of the line. Be careful because the line
        // numbers in Loc are 1-based, so we subtract 1 to get 0-based
        // lines.
        for line_index in lo.line-1 .. hi.line-1 {
            let line_len = lo.file.get_line(line_index)
                                  .map(|s| s.chars().count())
                                  .unwrap_or(0);
            lines.push(LineInfo { line_index: line_index,
                                  start_col: start_col,
                                  end_col: CharPos::from_usize(line_len) });
            start_col = CharPos::from_usize(0);
        }

        // For the last line, it extends from `start_col` to `hi.col`:
        lines.push(LineInfo { line_index: hi.line - 1,
                              start_col: start_col,
                              end_col: hi.col });

        Ok(FileLines {file: lo.file, lines: lines})
    }

    pub fn span_to_snippet(&self, sp: Span) -> Result<String, SpanSnippetError> {
        if sp.lo > sp.hi {
            return Err(SpanSnippetError::IllFormedSpan(sp));
        }

        let local_begin = self.lookup_byte_offset(sp.lo);
        let local_end = self.lookup_byte_offset(sp.hi);

        if local_begin.fm.start_pos != local_end.fm.start_pos {
            return Err(SpanSnippetError::DistinctSources(DistinctSources {
                begin: (local_begin.fm.name.clone(),
                        local_begin.fm.start_pos),
                end: (local_end.fm.name.clone(),
                      local_end.fm.start_pos)
            }));
        } else {
            match local_begin.fm.src {
                Some(ref src) => {
                    let start_index = local_begin.pos.to_usize();
                    let end_index = local_end.pos.to_usize();
                    let source_len = (local_begin.fm.end_pos -
                                      local_begin.fm.start_pos).to_usize();

                    if start_index > end_index || end_index > source_len {
                        return Err(SpanSnippetError::MalformedForCodemap(
                            MalformedCodemapPositions {
                                name: local_begin.fm.name.clone(),
                                source_len: source_len,
                                begin_pos: local_begin.pos,
                                end_pos: local_end.pos,
                            }));
                    }

                    return Ok((&src[start_index..end_index]).to_string())
                }
                None => {
                    return Err(SpanSnippetError::SourceNotAvailable {
                        filename: local_begin.fm.name.clone()
                    });
                }
            }
        }
    }

    pub fn get_filemap(&self, filename: &str) -> Option<Rc<FileMap>> {
        for fm in self.files.borrow().iter() {
            if filename == fm.name {
                return Some(fm.clone());
            }
        }
        None
    }

    /// For a global BytePos compute the local offset within the containing FileMap
    pub fn lookup_byte_offset(&self, bpos: BytePos) -> FileMapAndBytePos {
        let idx = self.lookup_filemap_idx(bpos);
        let fm = (*self.files.borrow())[idx].clone();
        let offset = bpos - fm.start_pos;
        FileMapAndBytePos {fm: fm, pos: offset}
    }

    /// Converts an absolute BytePos to a CharPos relative to the filemap.
    pub fn bytepos_to_file_charpos(&self, bpos: BytePos) -> CharPos {
        let idx = self.lookup_filemap_idx(bpos);
        let files = self.files.borrow();
        let map = &(*files)[idx];

        // The number of extra bytes due to multibyte chars in the FileMap
        let mut total_extra_bytes = 0;

        for mbc in map.multibyte_chars.borrow().iter() {
            debug!("{}-byte char at {:?}", mbc.bytes, mbc.pos);
            if mbc.pos < bpos {
                // every character is at least one byte, so we only
                // count the actual extra bytes.
                total_extra_bytes += mbc.bytes - 1;
                // We should never see a byte position in the middle of a
                // character
                assert!(bpos.to_usize() >= mbc.pos.to_usize() + mbc.bytes);
            } else {
                break;
            }
        }

        assert!(map.start_pos.to_usize() + total_extra_bytes <= bpos.to_usize());
        CharPos(bpos.to_usize() - map.start_pos.to_usize() - total_extra_bytes)
    }

    // Return the index of the filemap (in self.files) which contains pos.
    fn lookup_filemap_idx(&self, pos: BytePos) -> usize {
        let files = self.files.borrow();
        let files = &*files;
        let count = files.len();

        // Binary search for the filemap.
        let mut a = 0;
        let mut b = count;
        while b - a > 1 {
            let m = (a + b) / 2;
            if files[m].start_pos > pos {
                b = m;
            } else {
                a = m;
            }
        }

        assert!(a < count, "position {} does not resolve to a source location", pos.to_usize());

        return a;
    }

    pub fn record_expansion(&self, expn_info: ExpnInfo) -> ExpnId {
        let mut expansions = self.expansions.borrow_mut();
        expansions.push(expn_info);
        let len = expansions.len();
        if len > u32::max_value() as usize {
            panic!("too many ExpnInfo's!");
        }
        ExpnId(len as u32 - 1)
    }

    pub fn with_expn_info<T, F>(&self, id: ExpnId, f: F) -> T where
        F: FnOnce(Option<&ExpnInfo>) -> T,
    {
        match id {
            NO_EXPANSION | COMMAND_LINE_EXPN => f(None),
            ExpnId(i) => f(Some(&(*self.expansions.borrow())[i as usize]))
        }
    }

    /// Check if a span is "internal" to a macro in which #[unstable]
    /// items can be used (that is, a macro marked with
    /// `#[allow_internal_unstable]`).
    pub fn span_allows_unstable(&self, span: Span) -> bool {
        debug!("span_allows_unstable(span = {:?})", span);
        let mut allows_unstable = false;
        let mut expn_id = span.expn_id;
        loop {
            let quit = self.with_expn_info(expn_id, |expninfo| {
                debug!("span_allows_unstable: expninfo = {:?}", expninfo);
                expninfo.map_or(/* hit the top level */ true, |info| {

                    let span_comes_from_this_expansion =
                        info.callee.span.map_or(span.source_equal(&info.call_site), |mac_span| {
                            mac_span.contains(span)
                        });

                    debug!("span_allows_unstable: span: {:?} call_site: {:?} callee: {:?}",
                           (span.lo, span.hi),
                           (info.call_site.lo, info.call_site.hi),
                           info.callee.span.map(|x| (x.lo, x.hi)));
                    debug!("span_allows_unstable: from this expansion? {}, allows unstable? {}",
                           span_comes_from_this_expansion,
                           info.callee.allow_internal_unstable);
                    if span_comes_from_this_expansion {
                        allows_unstable = info.callee.allow_internal_unstable;
                        // we've found the right place, stop looking
                        true
                    } else {
                        // not the right place, keep looking
                        expn_id = info.call_site.expn_id;
                        false
                    }
                })
            });
            if quit {
                break
            }
        }
        debug!("span_allows_unstable? {}", allows_unstable);
        allows_unstable
    }

    pub fn count_lines(&self) -> usize {
        self.files.borrow().iter().fold(0, |a, f| a + f.count_lines())
    }

    pub fn macro_backtrace(&self, span: Span) -> Vec<MacroBacktrace> {
        let mut last_span = DUMMY_SP;
        let mut span = span;
        let mut result = vec![];
        loop {
            let span_name_span = self.with_expn_info(span.expn_id, |expn_info| {
                expn_info.map(|ei| {
                    let (pre, post) = match ei.callee.format {
                        MacroAttribute(..) => ("#[", "]"),
                        MacroBang(..) => ("", "!"),
                    };
                    let macro_decl_name = format!("{}{}{}",
                                                  pre,
                                                  ei.callee.name(),
                                                  post);
                    let def_site_span = ei.callee.span;
                    (ei.call_site, macro_decl_name, def_site_span)
                })
            });

            match span_name_span {
                None => break,
                Some((call_site, macro_decl_name, def_site_span)) => {
                    // Don't print recursive invocations
                    if !call_site.source_equal(&last_span) {
                        result.push(MacroBacktrace {
                            call_site: call_site,
                            macro_decl_name: macro_decl_name,
                            def_site_span: def_site_span,
                        });
                    }
                    last_span = span;
                    span = call_site;
                }
            }
        }
        result
    }
}

impl CodeMapper for CodeMap {
    fn lookup_char_pos(&self, pos: BytePos) -> Loc {
        self.lookup_char_pos(pos)
    }
    fn span_to_lines(&self, sp: Span) -> FileLinesResult {
        self.span_to_lines(sp)
    }
    fn span_to_string(&self, sp: Span) -> String {
        self.span_to_string(sp)
    }
    fn span_to_filename(&self, sp: Span) -> FileName {
        self.span_to_filename(sp)
    }
    fn macro_backtrace(&self, span: Span) -> Vec<MacroBacktrace> {
        self.macro_backtrace(span)
    }
}

// _____________________________________________________________________________
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;
    use errors::{Level, CodeSuggestion};
    use errors::emitter::EmitterWriter;
    use errors::snippet::{SnippetData, RenderedLine, FormatMode};
    use std::sync::{Arc, Mutex};
    use std::io::{self, Write};
    use std::str::from_utf8;
    use std::rc::Rc;

    #[test]
    fn t1 () {
        let cm = CodeMap::new();
        let fm = cm.new_filemap("blork.rs".to_string(),
                                None,
                                "first line.\nsecond line".to_string());
        fm.next_line(BytePos(0));
        // Test we can get lines with partial line info.
        assert_eq!(fm.get_line(0), Some("first line."));
        // TESTING BROKEN BEHAVIOR: line break declared before actual line break.
        fm.next_line(BytePos(10));
        assert_eq!(fm.get_line(1), Some("."));
        fm.next_line(BytePos(12));
        assert_eq!(fm.get_line(2), Some("second line"));
    }

    #[test]
    #[should_panic]
    fn t2 () {
        let cm = CodeMap::new();
        let fm = cm.new_filemap("blork.rs".to_string(),
                                None,
                                "first line.\nsecond line".to_string());
        // TESTING *REALLY* BROKEN BEHAVIOR:
        fm.next_line(BytePos(0));
        fm.next_line(BytePos(10));
        fm.next_line(BytePos(2));
    }

    fn init_code_map() -> CodeMap {
        let cm = CodeMap::new();
        let fm1 = cm.new_filemap("blork.rs".to_string(),
                                 None,
                                 "first line.\nsecond line".to_string());
        let fm2 = cm.new_filemap("empty.rs".to_string(),
                                 None,
                                 "".to_string());
        let fm3 = cm.new_filemap("blork2.rs".to_string(),
                                 None,
                                 "first line.\nsecond line".to_string());

        fm1.next_line(BytePos(0));
        fm1.next_line(BytePos(12));
        fm2.next_line(fm2.start_pos);
        fm3.next_line(fm3.start_pos);
        fm3.next_line(fm3.start_pos + BytePos(12));

        cm
    }

    #[test]
    fn t3() {
        // Test lookup_byte_offset
        let cm = init_code_map();

        let fmabp1 = cm.lookup_byte_offset(BytePos(23));
        assert_eq!(fmabp1.fm.name, "blork.rs");
        assert_eq!(fmabp1.pos, BytePos(23));

        let fmabp1 = cm.lookup_byte_offset(BytePos(24));
        assert_eq!(fmabp1.fm.name, "empty.rs");
        assert_eq!(fmabp1.pos, BytePos(0));

        let fmabp2 = cm.lookup_byte_offset(BytePos(25));
        assert_eq!(fmabp2.fm.name, "blork2.rs");
        assert_eq!(fmabp2.pos, BytePos(0));
    }

    #[test]
    fn t4() {
        // Test bytepos_to_file_charpos
        let cm = init_code_map();

        let cp1 = cm.bytepos_to_file_charpos(BytePos(22));
        assert_eq!(cp1, CharPos(22));

        let cp2 = cm.bytepos_to_file_charpos(BytePos(25));
        assert_eq!(cp2, CharPos(0));
    }

    #[test]
    fn t5() {
        // Test zero-length filemaps.
        let cm = init_code_map();

        let loc1 = cm.lookup_char_pos(BytePos(22));
        assert_eq!(loc1.file.name, "blork.rs");
        assert_eq!(loc1.line, 2);
        assert_eq!(loc1.col, CharPos(10));

        let loc2 = cm.lookup_char_pos(BytePos(25));
        assert_eq!(loc2.file.name, "blork2.rs");
        assert_eq!(loc2.line, 1);
        assert_eq!(loc2.col, CharPos(0));
    }

    fn init_code_map_mbc() -> CodeMap {
        let cm = CodeMap::new();
        // € is a three byte utf8 char.
        let fm1 =
            cm.new_filemap("blork.rs".to_string(),
                           None,
                           "fir€st €€€€ line.\nsecond line".to_string());
        let fm2 = cm.new_filemap("blork2.rs".to_string(),
                                 None,
                                 "first line€€.\n€ second line".to_string());

        fm1.next_line(BytePos(0));
        fm1.next_line(BytePos(28));
        fm2.next_line(fm2.start_pos);
        fm2.next_line(fm2.start_pos + BytePos(20));

        fm1.record_multibyte_char(BytePos(3), 3);
        fm1.record_multibyte_char(BytePos(9), 3);
        fm1.record_multibyte_char(BytePos(12), 3);
        fm1.record_multibyte_char(BytePos(15), 3);
        fm1.record_multibyte_char(BytePos(18), 3);
        fm2.record_multibyte_char(fm2.start_pos + BytePos(10), 3);
        fm2.record_multibyte_char(fm2.start_pos + BytePos(13), 3);
        fm2.record_multibyte_char(fm2.start_pos + BytePos(18), 3);

        cm
    }

    #[test]
    fn t6() {
        // Test bytepos_to_file_charpos in the presence of multi-byte chars
        let cm = init_code_map_mbc();

        let cp1 = cm.bytepos_to_file_charpos(BytePos(3));
        assert_eq!(cp1, CharPos(3));

        let cp2 = cm.bytepos_to_file_charpos(BytePos(6));
        assert_eq!(cp2, CharPos(4));

        let cp3 = cm.bytepos_to_file_charpos(BytePos(56));
        assert_eq!(cp3, CharPos(12));

        let cp4 = cm.bytepos_to_file_charpos(BytePos(61));
        assert_eq!(cp4, CharPos(15));
    }

    #[test]
    fn t7() {
        // Test span_to_lines for a span ending at the end of filemap
        let cm = init_code_map();
        let span = Span {lo: BytePos(12), hi: BytePos(23), expn_id: NO_EXPANSION};
        let file_lines = cm.span_to_lines(span).unwrap();

        assert_eq!(file_lines.file.name, "blork.rs");
        assert_eq!(file_lines.lines.len(), 1);
        assert_eq!(file_lines.lines[0].line_index, 1);
    }

    /// Given a string like " ~~~~~~~~~~~~ ", produces a span
    /// coverting that range. The idea is that the string has the same
    /// length as the input, and we uncover the byte positions.  Note
    /// that this can span lines and so on.
    fn span_from_selection(input: &str, selection: &str) -> Span {
        assert_eq!(input.len(), selection.len());
        let left_index = selection.find('~').unwrap() as u32;
        let right_index = selection.rfind('~').map(|x|x as u32).unwrap_or(left_index);
        Span { lo: BytePos(left_index), hi: BytePos(right_index + 1), expn_id: NO_EXPANSION }
    }

    /// Test span_to_snippet and span_to_lines for a span coverting 3
    /// lines in the middle of a file.
    #[test]
    fn span_to_snippet_and_lines_spanning_multiple_lines() {
        let cm = CodeMap::new();
        let inputtext = "aaaaa\nbbbbBB\nCCC\nDDDDDddddd\neee\n";
        let selection = "     \n    ~~\n~~~\n~~~~~     \n   \n";
        cm.new_filemap_and_lines("blork.rs", None, inputtext);
        let span = span_from_selection(inputtext, selection);

        // check that we are extracting the text we thought we were extracting
        assert_eq!(&cm.span_to_snippet(span).unwrap(), "BB\nCCC\nDDDDD");

        // check that span_to_lines gives us the complete result with the lines/cols we expected
        let lines = cm.span_to_lines(span).unwrap();
        let expected = vec![
            LineInfo { line_index: 1, start_col: CharPos(4), end_col: CharPos(6) },
            LineInfo { line_index: 2, start_col: CharPos(0), end_col: CharPos(3) },
            LineInfo { line_index: 3, start_col: CharPos(0), end_col: CharPos(5) }
            ];
        assert_eq!(lines.lines, expected);
    }

    #[test]
    fn t8() {
        // Test span_to_snippet for a span ending at the end of filemap
        let cm = init_code_map();
        let span = Span {lo: BytePos(12), hi: BytePos(23), expn_id: NO_EXPANSION};
        let snippet = cm.span_to_snippet(span);

        assert_eq!(snippet, Ok("second line".to_string()));
    }

    #[test]
    fn t9() {
        // Test span_to_str for a span ending at the end of filemap
        let cm = init_code_map();
        let span = Span {lo: BytePos(12), hi: BytePos(23), expn_id: NO_EXPANSION};
        let sstr =  cm.span_to_string(span);

        assert_eq!(sstr, "blork.rs:2:1: 2:12");
    }

    #[test]
    fn t10() {
        // Test span_to_expanded_string works in base case (no expansion)
        let cm = init_code_map();
        let span = Span { lo: BytePos(0), hi: BytePos(11), expn_id: NO_EXPANSION };
        let sstr = cm.span_to_expanded_string(span);
        assert_eq!(sstr, "blork.rs:1:1: 1:12\n`first line.`\n");

        let span = Span { lo: BytePos(12), hi: BytePos(23), expn_id: NO_EXPANSION };
        let sstr =  cm.span_to_expanded_string(span);
        assert_eq!(sstr, "blork.rs:2:1: 2:12\n`second line`\n");
    }

    #[test]
    fn t11() {
        // Test span_to_expanded_string works with expansion
        use ast::Name;
        let cm = init_code_map();
        let root = Span { lo: BytePos(0), hi: BytePos(11), expn_id: NO_EXPANSION };
        let format = ExpnFormat::MacroBang(Name(0u32));
        let callee = NameAndSpan { format: format,
                                   allow_internal_unstable: false,
                                   span: None };

        let info = ExpnInfo { call_site: root, callee: callee };
        let id = cm.record_expansion(info);
        let sp = Span { lo: BytePos(12), hi: BytePos(23), expn_id: id };

        let sstr = cm.span_to_expanded_string(sp);
        assert_eq!(sstr,
                   "blork.rs:2:1: 2:12\n`second line`\n  Callsite:\n  \
                    blork.rs:1:1: 1:12\n  `first line.`\n");
    }

    /// Returns the span corresponding to the `n`th occurrence of
    /// `substring` in `source_text`.
    trait CodeMapExtension {
        fn span_substr(&self,
                    file: &Rc<FileMap>,
                    source_text: &str,
                    substring: &str,
                    n: usize)
                    -> Span;
    }

    impl CodeMapExtension for CodeMap {
        fn span_substr(&self,
                    file: &Rc<FileMap>,
                    source_text: &str,
                    substring: &str,
                    n: usize)
                    -> Span
        {
            println!("span_substr(file={:?}/{:?}, substring={:?}, n={})",
                    file.name, file.start_pos, substring, n);
            let mut i = 0;
            let mut hi = 0;
            loop {
                let offset = source_text[hi..].find(substring).unwrap_or_else(|| {
                    panic!("source_text `{}` does not have {} occurrences of `{}`, only {}",
                        source_text, n, substring, i);
                });
                let lo = hi + offset;
                hi = lo + substring.len();
                if i == n {
                    let span = Span {
                        lo: BytePos(lo as u32 + file.start_pos.0),
                        hi: BytePos(hi as u32 + file.start_pos.0),
                        expn_id: NO_EXPANSION,
                    };
                    assert_eq!(&self.span_to_snippet(span).unwrap()[..],
                            substring);
                    return span;
                }
                i += 1;
            }
        }
    }

    fn splice(start: Span, end: Span) -> Span {
        Span {
            lo: start.lo,
            hi: end.hi,
            expn_id: NO_EXPANSION,
        }
    }

    fn make_string(lines: &[RenderedLine]) -> String {
        lines.iter()
            .flat_map(|rl| {
                rl.text.iter()
                        .map(|s| &s.text[..])
                        .chain(Some("\n"))
            })
            .collect()
    }

    fn init_expansion_chain(cm: &CodeMap) -> Span {
        // Creates an expansion chain containing two recursive calls
        // root -> expA -> expA -> expB -> expB -> end
        use ast::Name;

        let root = Span { lo: BytePos(0), hi: BytePos(11), expn_id: NO_EXPANSION };

        let format_root = ExpnFormat::MacroBang(Name(0u32));
        let callee_root = NameAndSpan { format: format_root,
                                        allow_internal_unstable: false,
                                        span: Some(root) };

        let info_a1 = ExpnInfo { call_site: root, callee: callee_root };
        let id_a1 = cm.record_expansion(info_a1);
        let span_a1 = Span { lo: BytePos(12), hi: BytePos(23), expn_id: id_a1 };

        let format_a = ExpnFormat::MacroBang(Name(1u32));
        let callee_a = NameAndSpan { format: format_a,
                                      allow_internal_unstable: false,
                                      span: Some(span_a1) };

        let info_a2 = ExpnInfo { call_site: span_a1, callee: callee_a.clone() };
        let id_a2 = cm.record_expansion(info_a2);
        let span_a2 = Span { lo: BytePos(12), hi: BytePos(23), expn_id: id_a2 };

        let info_b1 = ExpnInfo { call_site: span_a2, callee: callee_a };
        let id_b1 = cm.record_expansion(info_b1);
        let span_b1 = Span { lo: BytePos(25), hi: BytePos(36), expn_id: id_b1 };

        let format_b = ExpnFormat::MacroBang(Name(2u32));
        let callee_b = NameAndSpan { format: format_b,
                                     allow_internal_unstable: false,
                                     span: None };

        let info_b2 = ExpnInfo { call_site: span_b1, callee: callee_b.clone() };
        let id_b2 = cm.record_expansion(info_b2);
        let span_b2 = Span { lo: BytePos(25), hi: BytePos(36), expn_id: id_b2 };

        let info_end = ExpnInfo { call_site: span_b2, callee: callee_b };
        let id_end = cm.record_expansion(info_end);
        Span { lo: BytePos(37), hi: BytePos(48), expn_id: id_end }
    }

    #[test]
    fn t12() {
        // Test span_to_expanded_string collapses recursive macros and handles
        // recursive callsite and callee expansions
        let cm = init_code_map();
        let end = init_expansion_chain(&cm);
        let sstr = cm.span_to_expanded_string(end);
        let res_str =
r"blork2.rs:2:1: 2:12
`second line`
  Callsite:
  ...
  blork2.rs:1:1: 1:12
  `first line.`
    Callee:
    blork.rs:2:1: 2:12
    `second line`
      Callee:
      blork.rs:1:1: 1:12
      `first line.`
      Callsite:
      blork.rs:1:1: 1:12
      `first line.`
    Callsite:
    ...
    blork.rs:2:1: 2:12
    `second line`
      Callee:
      blork.rs:1:1: 1:12
      `first line.`
      Callsite:
      blork.rs:1:1: 1:12
      `first line.`
";
        assert_eq!(sstr, res_str);
    }

    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl Write for Sink {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            Write::write(&mut *self.0.lock().unwrap(), data)
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }

    // Diagnostic doesn't align properly in span where line number increases by one digit
    #[test]
    fn test_hilight_suggestion_issue_11715() {
        let data = Arc::new(Mutex::new(Vec::new()));
        let cm = Rc::new(CodeMap::new());
        let mut ew = EmitterWriter::new(Box::new(Sink(data.clone())),
                                        None,
                                        cm.clone(),
                                        FormatMode::NewErrorFormat);
        let content = "abcdefg
        koksi
        line3
        line4
        cinq
        line6
        line7
        line8
        line9
        line10
        e-lä-vän
        tolv
        dreizehn
        ";
        let file = cm.new_filemap_and_lines("dummy.txt", None, content);
        let start = file.lines.borrow()[10];
        let end = file.lines.borrow()[11];
        let sp = mk_sp(start, end);
        let lvl = Level::Error;
        println!("highlight_lines");
        ew.highlight_lines(&sp.into(), lvl).unwrap();
        println!("done");
        let vec = data.lock().unwrap().clone();
        let vec: &[u8] = &vec;
        let str = from_utf8(vec).unwrap();
        println!("r#\"\n{}\"#", str);
        assert_eq!(str, &r#"
  --> dummy.txt:11:1
   |>
11 |>         e-lä-vän
   |> ^
"#[1..]);
    }

    #[test]
    fn test_single_span_splice() {
        // Test that a `MultiSpan` containing a single span splices a substition correctly
        let cm = CodeMap::new();
        let inputtext = "aaaaa\nbbbbBB\nCCC\nDDDDDddddd\neee\n";
        let selection = "     \n    ~~\n~~~\n~~~~~     \n   \n";
        cm.new_filemap_and_lines("blork.rs", None, inputtext);
        let sp = span_from_selection(inputtext, selection);
        let msp: MultiSpan = sp.into();

        // check that we are extracting the text we thought we were extracting
        assert_eq!(&cm.span_to_snippet(sp).unwrap(), "BB\nCCC\nDDDDD");

        let substitute = "ZZZZZZ".to_owned();
        let expected = "bbbbZZZZZZddddd";
        let suggest = CodeSuggestion {
            msp: msp,
            substitutes: vec![substitute],
        };
        assert_eq!(suggest.splice_lines(&cm), expected);
    }

    #[test]
    fn test_multi_span_splice() {
        // Test that a `MultiSpan` containing multiple spans splices a substition correctly
        let cm = CodeMap::new();
        let inputtext  = "aaaaa\nbbbbBB\nCCC\nDDDDDddddd\neee\n";
        let selection1 = "     \n      \n   \n          \n ~ \n"; // intentionally out of order
        let selection2 = "     \n    ~~\n~~~\n~~~~~     \n   \n";
        cm.new_filemap_and_lines("blork.rs", None, inputtext);
        let sp1 = span_from_selection(inputtext, selection1);
        let sp2 = span_from_selection(inputtext, selection2);
        let msp: MultiSpan = MultiSpan::from_spans(vec![sp1, sp2]);

        let expected = "bbbbZZZZZZddddd\neXYZe";
        let suggest = CodeSuggestion {
            msp: msp,
            substitutes: vec!["ZZZZZZ".to_owned(),
                              "XYZ".to_owned()]
        };

        assert_eq!(suggest.splice_lines(&cm), expected);
    }

    #[test]
    fn test_multispan_highlight() {
        let data = Arc::new(Mutex::new(Vec::new()));
        let cm = Rc::new(CodeMap::new());
        let mut diag = EmitterWriter::new(Box::new(Sink(data.clone())),
                                          None,
                                          cm.clone(),
                                          FormatMode::NewErrorFormat);

        let inp =       "_____aaaaaa____bbbbbb__cccccdd_";
        let sp1 =       "     ~~~~~~                    ";
        let sp2 =       "               ~~~~~~          ";
        let sp3 =       "                       ~~~~~   ";
        let sp4 =       "                          ~~~~ ";
        let sp34 =      "                       ~~~~~~~ ";

        let expect_start = &r#"
 --> dummy.txt:1:6
  |>
1 |> _____aaaaaa____bbbbbb__cccccdd_
  |>      ^^^^^^    ^^^^^^  ^^^^^^^
"#[1..];

        let span = |sp, expected| {
            let sp = span_from_selection(inp, sp);
            assert_eq!(&cm.span_to_snippet(sp).unwrap(), expected);
            sp
        };
        cm.new_filemap_and_lines("dummy.txt", None, inp);
        let sp1 = span(sp1, "aaaaaa");
        let sp2 = span(sp2, "bbbbbb");
        let sp3 = span(sp3, "ccccc");
        let sp4 = span(sp4, "ccdd");
        let sp34 = span(sp34, "cccccdd");

        let spans = vec![sp1, sp2, sp3, sp4];

        let test = |expected, highlight: &mut FnMut()| {
            data.lock().unwrap().clear();
            highlight();
            let vec = data.lock().unwrap().clone();
            let actual = from_utf8(&vec[..]).unwrap();
            println!("actual=\n{}", actual);
            assert_eq!(actual, expected);
        };

        let msp = MultiSpan::from_spans(vec![sp1, sp2, sp34]);
        test(expect_start, &mut || {
            diag.highlight_lines(&msp, Level::Error).unwrap();
        });
        test(expect_start, &mut || {
            let msp = MultiSpan::from_spans(spans.clone());
            diag.highlight_lines(&msp, Level::Error).unwrap();
        });
    }

    #[test]
    fn test_huge_multispan_highlight() {
        let data = Arc::new(Mutex::new(Vec::new()));
        let cm = Rc::new(CodeMap::new());
        let mut diag = EmitterWriter::new(Box::new(Sink(data.clone())),
                                          None,
                                          cm.clone(),
                                          FormatMode::NewErrorFormat);

        let inp = "aaaaa\n\
                   aaaaa\n\
                   aaaaa\n\
                   bbbbb\n\
                   ccccc\n\
                   xxxxx\n\
                   yyyyy\n\
                   _____\n\
                   ddd__eee_\n\
                   elided\n\
                   __f_gg";
        let file = cm.new_filemap_and_lines("dummy.txt", None, inp);

        let span = |lo, hi, (off_lo, off_hi)| {
            let lines = file.lines.borrow();
            let (mut lo, mut hi): (BytePos, BytePos) = (lines[lo], lines[hi]);
            lo.0 += off_lo;
            hi.0 += off_hi;
            mk_sp(lo, hi)
        };
        let sp0 = span(4, 6, (0, 5));
        let sp1 = span(0, 6, (0, 5));
        let sp2 = span(8, 8, (0, 3));
        let sp3 = span(8, 8, (5, 8));
        let sp4 = span(10, 10, (2, 3));
        let sp5 = span(10, 10, (4, 6));

        let expect0 = &r#"
   --> dummy.txt:5:1
    |>
5   |> ccccc
    |> ^
...
9   |> ddd__eee_
    |> ^^^  ^^^
10  |> elided
11  |> __f_gg
    |>   ^ ^^
"#[1..];

        let expect = &r#"
   --> dummy.txt:1:1
    |>
1   |> aaaaa
    |> ^
...
9   |> ddd__eee_
    |> ^^^  ^^^
10  |> elided
11  |> __f_gg
    |>   ^ ^^
"#[1..];

        macro_rules! test {
            ($expected: expr, $highlight: expr) => ({
                data.lock().unwrap().clear();
                $highlight();
                let vec = data.lock().unwrap().clone();
                let actual = from_utf8(&vec[..]).unwrap();
                println!("actual:");
                println!("{}", actual);
                println!("expected:");
                println!("{}", $expected);
                assert_eq!(&actual[..], &$expected[..]);
            });
        }

        let msp0 = MultiSpan::from_spans(vec![sp0, sp2, sp3, sp4, sp5]);
        let msp = MultiSpan::from_spans(vec![sp1, sp2, sp3, sp4, sp5]);

        test!(expect0, || {
            diag.highlight_lines(&msp0, Level::Error).unwrap();
        });
        test!(expect, || {
            diag.highlight_lines(&msp, Level::Error).unwrap();
        });
    }

    #[test]
    fn tab() {
        let file_text = "
fn foo() {
\tbar;
}
";

        let cm = Rc::new(CodeMap::new());
        let foo = cm.new_filemap_and_lines("foo.rs", None, file_text);
        let span_bar = cm.span_substr(&foo, file_text, "bar", 0);

        let mut snippet = SnippetData::new(cm, Some(span_bar), FormatMode::NewErrorFormat);
        snippet.push(span_bar, true, None);

        let lines = snippet.render_lines();
        let text = make_string(&lines);
        assert_eq!(&text[..], &"
 --> foo.rs:3:2
  |>
3 |> \tbar;
  |> \t^^^
"[1..]);
    }

    #[test]
    fn one_line() {
        let file_text = r#"
fn foo() {
    vec.push(vec.pop().unwrap());
}
"#;

        let cm = Rc::new(CodeMap::new());
        let foo = cm.new_filemap_and_lines("foo.rs", None, file_text);
        let span_vec0 = cm.span_substr(&foo, file_text, "vec", 0);
        let span_vec1 = cm.span_substr(&foo, file_text, "vec", 1);
        let span_semi = cm.span_substr(&foo, file_text, ";", 0);

        let mut snippet = SnippetData::new(cm, None, FormatMode::NewErrorFormat);
        snippet.push(span_vec0, false, Some(format!("previous borrow of `vec` occurs here")));
        snippet.push(span_vec1, false, Some(format!("error occurs here")));
        snippet.push(span_semi, false, Some(format!("previous borrow ends here")));

        let lines = snippet.render_lines();
        println!("{:#?}", lines);

        let text: String = make_string(&lines);

        println!("text=\n{}", text);
        assert_eq!(&text[..], &r#"
 ::: foo.rs
  |>
3 |>     vec.push(vec.pop().unwrap());
  |>     ---      ---                - previous borrow ends here
  |>     |        |
  |>     |        error occurs here
  |>     previous borrow of `vec` occurs here
"#[1..]);
    }

    #[test]
    fn two_files() {
        let file_text_foo = r#"
fn foo() {
    vec.push(vec.pop().unwrap());
}
"#;

        let file_text_bar = r#"
fn bar() {
    // these blank links here
    // serve to ensure that the line numbers
    // from bar.rs
    // require more digits










    vec.push();

    // this line will get elided

    vec.pop().unwrap());
}
"#;

        let cm = Rc::new(CodeMap::new());
        let foo_map = cm.new_filemap_and_lines("foo.rs", None, file_text_foo);
        let span_foo_vec0 = cm.span_substr(&foo_map, file_text_foo, "vec", 0);
        let span_foo_vec1 = cm.span_substr(&foo_map, file_text_foo, "vec", 1);
        let span_foo_semi = cm.span_substr(&foo_map, file_text_foo, ";", 0);

        let bar_map = cm.new_filemap_and_lines("bar.rs", None, file_text_bar);
        let span_bar_vec0 = cm.span_substr(&bar_map, file_text_bar, "vec", 0);
        let span_bar_vec1 = cm.span_substr(&bar_map, file_text_bar, "vec", 1);
        let span_bar_semi = cm.span_substr(&bar_map, file_text_bar, ";", 0);

        let mut snippet = SnippetData::new(cm, Some(span_foo_vec1), FormatMode::NewErrorFormat);
        snippet.push(span_foo_vec0, false, Some(format!("a")));
        snippet.push(span_foo_vec1, true, Some(format!("b")));
        snippet.push(span_foo_semi, false, Some(format!("c")));
        snippet.push(span_bar_vec0, false, Some(format!("d")));
        snippet.push(span_bar_vec1, false, Some(format!("e")));
        snippet.push(span_bar_semi, false, Some(format!("f")));

        let lines = snippet.render_lines();
        println!("{:#?}", lines);

        let text: String = make_string(&lines);

        println!("text=\n{}", text);

        // Note that the `|>` remain aligned across both files:
        assert_eq!(&text[..], &r#"
   --> foo.rs:3:14
    |>
3   |>     vec.push(vec.pop().unwrap());
    |>     ---      ^^^                - c
    |>     |        |
    |>     |        b
    |>     a
   ::: bar.rs
    |>
17  |>     vec.push();
    |>     ---       - f
    |>     |
    |>     d
...
21  |>     vec.pop().unwrap());
    |>     --- e
"#[1..]);
    }

    #[test]
    fn multi_line() {
        let file_text = r#"
fn foo() {
    let name = find_id(&data, 22).unwrap();

    // Add one more item we forgot to the vector. Silly us.
    data.push(Data { name: format!("Hera"), id: 66 });

    // Print everything out.
    println!("Name: {:?}", name);
    println!("Data: {:?}", data);
}
"#;

        let cm = Rc::new(CodeMap::new());
        let foo = cm.new_filemap_and_lines("foo.rs", None, file_text);
        let span_data0 = cm.span_substr(&foo, file_text, "data", 0);
        let span_data1 = cm.span_substr(&foo, file_text, "data", 1);
        let span_rbrace = cm.span_substr(&foo, file_text, "}", 3);

        let mut snippet = SnippetData::new(cm, None, FormatMode::NewErrorFormat);
        snippet.push(span_data0, false, Some(format!("immutable borrow begins here")));
        snippet.push(span_data1, false, Some(format!("mutable borrow occurs here")));
        snippet.push(span_rbrace, false, Some(format!("immutable borrow ends here")));

        let lines = snippet.render_lines();
        println!("{:#?}", lines);

        let text: String = make_string(&lines);

        println!("text=\n{}", text);
        assert_eq!(&text[..], &r#"
   ::: foo.rs
    |>
3   |>     let name = find_id(&data, 22).unwrap();
    |>                         ---- immutable borrow begins here
...
6   |>     data.push(Data { name: format!("Hera"), id: 66 });
    |>     ---- mutable borrow occurs here
...
11  |> }
    |> - immutable borrow ends here
"#[1..]);
    }

    #[test]
    fn overlapping() {
        let file_text = r#"
fn foo() {
    vec.push(vec.pop().unwrap());
}
"#;

        let cm = Rc::new(CodeMap::new());
        let foo = cm.new_filemap_and_lines("foo.rs", None, file_text);
        let span0 = cm.span_substr(&foo, file_text, "vec.push", 0);
        let span1 = cm.span_substr(&foo, file_text, "vec", 0);
        let span2 = cm.span_substr(&foo, file_text, "ec.push", 0);
        let span3 = cm.span_substr(&foo, file_text, "unwrap", 0);

        let mut snippet = SnippetData::new(cm, None, FormatMode::NewErrorFormat);
        snippet.push(span0, false, Some(format!("A")));
        snippet.push(span1, false, Some(format!("B")));
        snippet.push(span2, false, Some(format!("C")));
        snippet.push(span3, false, Some(format!("D")));

        let lines = snippet.render_lines();
        println!("{:#?}", lines);
        let text: String = make_string(&lines);

        println!("text=r#\"\n{}\".trim_left()", text);
        assert_eq!(&text[..], &r#"
 ::: foo.rs
  |>
3 |>     vec.push(vec.pop().unwrap());
  |>     --------           ------ D
  |>     ||
  |>     |C
  |>     A
  |>     B
"#[1..]);
    }

    #[test]
    fn one_line_out_of_order() {
        let file_text = r#"
fn foo() {
    vec.push(vec.pop().unwrap());
}
"#;

        let cm = Rc::new(CodeMap::new());
        let foo = cm.new_filemap_and_lines("foo.rs", None, file_text);
        let span_vec0 = cm.span_substr(&foo, file_text, "vec", 0);
        let span_vec1 = cm.span_substr(&foo, file_text, "vec", 1);
        let span_semi = cm.span_substr(&foo, file_text, ";", 0);

        // intentionally don't push the snippets left to right
        let mut snippet = SnippetData::new(cm, None, FormatMode::NewErrorFormat);
        snippet.push(span_vec1, false, Some(format!("error occurs here")));
        snippet.push(span_vec0, false, Some(format!("previous borrow of `vec` occurs here")));
        snippet.push(span_semi, false, Some(format!("previous borrow ends here")));

        let lines = snippet.render_lines();
        println!("{:#?}", lines);
        let text: String = make_string(&lines);

        println!("text=r#\"\n{}\".trim_left()", text);
        assert_eq!(&text[..], &r#"
 ::: foo.rs
  |>
3 |>     vec.push(vec.pop().unwrap());
  |>     ---      ---                - previous borrow ends here
  |>     |        |
  |>     |        error occurs here
  |>     previous borrow of `vec` occurs here
"#[1..]);
    }

    #[test]
    fn elide_unnecessary_lines() {
        let file_text = r#"
fn foo() {
    let mut vec = vec![0, 1, 2];
    let mut vec2 = vec;
    vec2.push(3);
    vec2.push(4);
    vec2.push(5);
    vec2.push(6);
    vec.push(7);
}
"#;

        let cm = Rc::new(CodeMap::new());
        let foo = cm.new_filemap_and_lines("foo.rs", None, file_text);
        let span_vec0 = cm.span_substr(&foo, file_text, "vec", 3);
        let span_vec1 = cm.span_substr(&foo, file_text, "vec", 8);

        let mut snippet = SnippetData::new(cm, None, FormatMode::NewErrorFormat);
        snippet.push(span_vec0, false, Some(format!("`vec` moved here because it \
            has type `collections::vec::Vec<i32>`")));
        snippet.push(span_vec1, false, Some(format!("use of moved value: `vec`")));

        let lines = snippet.render_lines();
        println!("{:#?}", lines);
        let text: String = make_string(&lines);
        println!("text=r#\"\n{}\".trim_left()", text);
        assert_eq!(&text[..], &r#"
   ::: foo.rs
    |>
4   |>     let mut vec2 = vec;
    |>                    --- `vec` moved here because it has type `collections::vec::Vec<i32>`
...
9   |>     vec.push(7);
    |>     --- use of moved value: `vec`
"#[1..]);
    }

    #[test]
    fn spans_without_labels() {
        let file_text = r#"
fn foo() {
    let mut vec = vec![0, 1, 2];
    let mut vec2 = vec;
    vec2.push(3);
    vec2.push(4);
    vec2.push(5);
    vec2.push(6);
    vec.push(7);
}
"#;

        let cm = Rc::new(CodeMap::new());
        let foo = cm.new_filemap_and_lines("foo.rs", None, file_text);

        let mut snippet = SnippetData::new(cm.clone(), None, FormatMode::NewErrorFormat);
        for i in 0..4 {
            let span_veci = cm.span_substr(&foo, file_text, "vec", i);
            snippet.push(span_veci, false, None);
        }

        let lines = snippet.render_lines();
        let text: String = make_string(&lines);
        println!("text=&r#\"\n{}\n\"#[1..]", text);
        assert_eq!(text, &r#"
 ::: foo.rs
  |>
3 |>     let mut vec = vec![0, 1, 2];
  |>             ---   ---
4 |>     let mut vec2 = vec;
  |>             ---    ---
"#[1..]);
    }

    #[test]
    fn span_long_selection() {
        let file_text = r#"
impl SomeTrait for () {
    fn foo(x: u32) {
        // impl 1
        // impl 2
        // impl 3
    }
}
"#;

        let cm = Rc::new(CodeMap::new());
        let foo = cm.new_filemap_and_lines("foo.rs", None, file_text);

        let mut snippet = SnippetData::new(cm.clone(), None, FormatMode::NewErrorFormat);
        let fn_span = cm.span_substr(&foo, file_text, "fn", 0);
        let rbrace_span = cm.span_substr(&foo, file_text, "}", 0);
        snippet.push(splice(fn_span, rbrace_span), false, None);
        let lines = snippet.render_lines();
        let text: String = make_string(&lines);
        println!("r#\"\n{}\"", text);
        assert_eq!(text, &r#"
 ::: foo.rs
  |>
3 |>     fn foo(x: u32) {
  |>     -
"#[1..]);
    }

    #[test]
    fn span_overlap_label() {
        // Test that we don't put `x_span` to the right of its highlight,
        // since there is another highlight that overlaps it.

        let file_text = r#"
    fn foo(x: u32) {
    }
}
"#;

        let cm = Rc::new(CodeMap::new());
        let foo = cm.new_filemap_and_lines("foo.rs", None, file_text);

        let mut snippet = SnippetData::new(cm.clone(), None, FormatMode::NewErrorFormat);
        let fn_span = cm.span_substr(&foo, file_text, "fn foo(x: u32)", 0);
        let x_span = cm.span_substr(&foo, file_text, "x", 0);
        snippet.push(fn_span, false, Some(format!("fn_span")));
        snippet.push(x_span, false, Some(format!("x_span")));
        let lines = snippet.render_lines();
        let text: String = make_string(&lines);
        println!("r#\"\n{}\"", text);
        assert_eq!(text, &r#"
 ::: foo.rs
  |>
2 |>     fn foo(x: u32) {
  |>     --------------
  |>     |      |
  |>     |      x_span
  |>     fn_span
"#[1..]);
    }

    #[test]
    fn span_overlap_label2() {
        // Test that we don't put `x_span` to the right of its highlight,
        // since there is another highlight that overlaps it. In this
        // case, the overlap is only at the beginning, but it's still
        // better to show the beginning more clearly.

        let file_text = r#"
    fn foo(x: u32) {
    }
}
"#;

        let cm = Rc::new(CodeMap::new());
        let foo = cm.new_filemap_and_lines("foo.rs", None, file_text);

        let mut snippet = SnippetData::new(cm.clone(), None, FormatMode::NewErrorFormat);
        let fn_span = cm.span_substr(&foo, file_text, "fn foo(x", 0);
        let x_span = cm.span_substr(&foo, file_text, "x: u32)", 0);
        snippet.push(fn_span, false, Some(format!("fn_span")));
        snippet.push(x_span, false, Some(format!("x_span")));
        let lines = snippet.render_lines();
        let text: String = make_string(&lines);
        println!("r#\"\n{}\"", text);
        assert_eq!(text, &r#"
 ::: foo.rs
  |>
2 |>     fn foo(x: u32) {
  |>     --------------
  |>     |      |
  |>     |      x_span
  |>     fn_span
"#[1..]);
    }

    #[test]
    fn span_overlap_label3() {
        // Test that we don't put `x_span` to the right of its highlight,
        // since there is another highlight that overlaps it. In this
        // case, the overlap is only at the beginning, but it's still
        // better to show the beginning more clearly.

        let file_text = r#"
    fn foo() {
       let closure = || {
           inner
       };
    }
}
"#;

        let cm = Rc::new(CodeMap::new());
        let foo = cm.new_filemap_and_lines("foo.rs", None, file_text);

        let mut snippet = SnippetData::new(cm.clone(), None, FormatMode::NewErrorFormat);

        let closure_span = {
            let closure_start_span = cm.span_substr(&foo, file_text, "||", 0);
            let closure_end_span = cm.span_substr(&foo, file_text, "}", 0);
            splice(closure_start_span, closure_end_span)
        };

        let inner_span = cm.span_substr(&foo, file_text, "inner", 0);

        snippet.push(closure_span, false, Some(format!("foo")));
        snippet.push(inner_span, false, Some(format!("bar")));

        let lines = snippet.render_lines();
        let text: String = make_string(&lines);
        println!("r#\"\n{}\"", text);
        assert_eq!(text, &r#"
 ::: foo.rs
  |>
3 |>        let closure = || {
  |>                      - foo
4 |>            inner
  |>            ----- bar
"#[1..]);
    }

    #[test]
    fn span_empty() {
        // In one of the unit tests, we found that the parser sometimes
        // gives empty spans, and in particular it supplied an EOF span
        // like this one, which points at the very end. We want to
        // fallback gracefully in this case.

        let file_text = r#"
fn main() {
    struct Foo;

    impl !Sync for Foo {}

    unsafe impl Send for &'static Foo {
    // error: cross-crate traits with a default impl, like `core::marker::Send`,
    //        can only be implemented for a struct/enum type, not
    //        `&'static Foo`
}"#;


        let cm = Rc::new(CodeMap::new());
        let foo = cm.new_filemap_and_lines("foo.rs", None, file_text);

        let mut rbrace_span = cm.span_substr(&foo, file_text, "}", 1);
        rbrace_span.lo = rbrace_span.hi;

        let mut snippet = SnippetData::new(cm.clone(),
                                           Some(rbrace_span),
                                           FormatMode::NewErrorFormat);
        snippet.push(rbrace_span, false, None);
        let lines = snippet.render_lines();
        let text: String = make_string(&lines);
        println!("r#\"\n{}\"", text);
        assert_eq!(text, &r#"
  --> foo.rs:11:2
   |>
11 |> }
   |>  -
"#[1..]);
    }
}
