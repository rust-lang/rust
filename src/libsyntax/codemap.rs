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
    /// Desugaring done by the compiler during HIR lowering.
    CompilerDesugaring(Name)
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

/// Build a span that covers the two provided spans.
pub fn combine_spans(sp1: Span, sp2: Span) -> Span {
    if sp1 == DUMMY_SP && sp2 == DUMMY_SP {
        DUMMY_SP
    } else if sp1 == DUMMY_SP {
        sp2
    } else if sp2 == DUMMY_SP {
        sp1
    } else {
        Span {
            lo: if sp1.lo < sp2.lo { sp1.lo } else { sp2.lo },
            hi: if sp1.hi > sp2.hi { sp1.hi } else { sp2.hi },
            expn_id: if sp1.expn_id == sp2.expn_id { sp1.expn_id } else { NO_EXPANSION },
        }
    }
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
            ExpnFormat::MacroAttribute(s) |
            ExpnFormat::MacroBang(s) |
            ExpnFormat::CompilerDesugaring(s) => s,
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

        match f.lookup_line(pos) {
            Some(line) => Ok(FileMapAndLine { fm: f, line: line }),
            None => Err(f)
        }
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

    /// Returns `Some(span)`, a union of the lhs and rhs span.  The lhs must precede the rhs. If
    /// there are gaps between lhs and rhs, the resulting union will cross these gaps.
    /// For this to work, the spans have to be:
    ///    * the expn_id of both spans much match
    ///    * the lhs span needs to end on the same line the rhs span begins
    ///    * the lhs span must start at or before the rhs span
    pub fn merge_spans(&self, sp_lhs: Span, sp_rhs: Span) -> Option<Span> {
        use std::cmp;

        // make sure we're at the same expansion id
        if sp_lhs.expn_id != sp_rhs.expn_id {
            return None;
        }

        let lhs_end = match self.lookup_line(sp_lhs.hi) {
            Ok(x) => x,
            Err(_) => return None
        };
        let rhs_begin = match self.lookup_line(sp_rhs.lo) {
            Ok(x) => x,
            Err(_) => return None
        };

        // if we must cross lines to merge, don't merge
        if lhs_end.line != rhs_begin.line {
            return None;
        }

        // ensure these follow the expected order and we don't overlap
        if (sp_lhs.lo <= sp_rhs.lo) && (sp_lhs.hi <= sp_rhs.lo) {
            Some(Span {
                lo: cmp::min(sp_lhs.lo, sp_rhs.lo),
                hi: cmp::max(sp_lhs.hi, sp_rhs.hi),
                expn_id: sp_lhs.expn_id,
            })
        } else {
            None
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
    pub fn lookup_filemap_idx(&self, pos: BytePos) -> usize {
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
        let mut prev_span = DUMMY_SP;
        let mut span = span;
        let mut result = vec![];
        loop {
            let span_name_span = self.with_expn_info(span.expn_id, |expn_info| {
                expn_info.map(|ei| {
                    let (pre, post) = match ei.callee.format {
                        MacroAttribute(..) => ("#[", "]"),
                        MacroBang(..) => ("", "!"),
                        CompilerDesugaring(..) => ("desugaring of `", "`"),
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
                    if !call_site.source_equal(&prev_span) {
                        result.push(MacroBacktrace {
                            call_site: call_site,
                            macro_decl_name: macro_decl_name,
                            def_site_span: def_site_span,
                        });
                    }
                    prev_span = span;
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
    fn merge_spans(&self, sp_lhs: Span, sp_rhs: Span) -> Option<Span> {
        self.merge_spans(sp_lhs, sp_rhs)
    }
}

// _____________________________________________________________________________
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;
    use symbol::keywords;
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
        let cm = init_code_map();
        let root = Span { lo: BytePos(0), hi: BytePos(11), expn_id: NO_EXPANSION };
        let format = ExpnFormat::MacroBang(keywords::Invalid.name());
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

    /// Test merging two spans on the same line
    #[test]
    fn span_merging() {
        let cm = CodeMap::new();
        let inputtext  = "bbbb BB bb CCC\n";
        let selection1 = "     ~~       \n";
        let selection2 = "           ~~~\n";
        cm.new_filemap_and_lines("blork.rs", None, inputtext);
        let span1 = span_from_selection(inputtext, selection1);
        let span2 = span_from_selection(inputtext, selection2);

        if let Some(sp) = cm.merge_spans(span1, span2) {
            let sstr = cm.span_to_expanded_string(sp);
            assert_eq!(sstr, "blork.rs:1:6: 1:15\n`BB bb CCC`\n");
        }
        else {
            assert!(false);
        }
    }

    /// Test failing to merge two spans on different lines
    #[test]
    fn span_merging_fail() {
        let cm = CodeMap::new();
        let inputtext  = "bbbb BB\ncc CCC\n";
        let selection1 = "     ~~\n      \n";
        let selection2 = "       \n   ~~~\n";
        cm.new_filemap_and_lines("blork.rs", None, inputtext);
        let span1 = span_from_selection(inputtext, selection1);
        let span2 = span_from_selection(inputtext, selection2);

        assert!(cm.merge_spans(span1, span2).is_none());
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

    fn init_expansion_chain(cm: &CodeMap) -> Span {
        // Creates an expansion chain containing two recursive calls
        // root -> expA -> expA -> expB -> expB -> end
        let root = Span { lo: BytePos(0), hi: BytePos(11), expn_id: NO_EXPANSION };

        let format_root = ExpnFormat::MacroBang(keywords::Invalid.name());
        let callee_root = NameAndSpan { format: format_root,
                                        allow_internal_unstable: false,
                                        span: Some(root) };

        let info_a1 = ExpnInfo { call_site: root, callee: callee_root };
        let id_a1 = cm.record_expansion(info_a1);
        let span_a1 = Span { lo: BytePos(12), hi: BytePos(23), expn_id: id_a1 };

        let format_a = ExpnFormat::MacroBang(keywords::As.name());
        let callee_a = NameAndSpan { format: format_a,
                                      allow_internal_unstable: false,
                                      span: Some(span_a1) };

        let info_a2 = ExpnInfo { call_site: span_a1, callee: callee_a.clone() };
        let id_a2 = cm.record_expansion(info_a2);
        let span_a2 = Span { lo: BytePos(12), hi: BytePos(23), expn_id: id_a2 };

        let info_b1 = ExpnInfo { call_site: span_a2, callee: callee_a };
        let id_b1 = cm.record_expansion(info_b1);
        let span_b1 = Span { lo: BytePos(25), hi: BytePos(36), expn_id: id_b1 };

        let format_b = ExpnFormat::MacroBang(keywords::Box.name());
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
}
