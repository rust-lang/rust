//! This module contains utilities for verifying the token state
//! and reporting errors on verification failures.
//!
//! The purpose of this module is to encapsulate the diagnostics
//! from the actual token parsing.

use super::StringReader;

use crate::symbol::Symbol;

use errors::FatalError;
use rustc_lexer::Base;
use syntax_pos::{BytePos, Span};


// Extensions for the `StringReader` providing verification helper
// methods.
impl<'a> StringReader<'a> {
    pub(super) fn verify_doc_comment_terminated(
        &self,
        start: BytePos,
        terminated: bool,
        is_doc_comment: bool
    ) {
        if !terminated {
            let msg = if is_doc_comment {
                "unterminated block doc-comment"
            } else {
                "unterminated block comment"
            };
            let last_bpos = self.pos;
            self.fatal_span_(start, last_bpos, msg).raise();
        }
    }

    pub(super) fn verify_doc_comment_contents(
        &self,
        start: BytePos,
        string: &str,
        is_block_comment: bool
    ) {
        let message = if is_block_comment {
            "bare CR not allowed in block doc-comment"
        } else {
            "bare CR not allowed in doc-comment"
        };
        self.verify_no_bare_cr(start, string, message);
    }

    pub(super) fn verify_raw_symbol(&self, sym: &Symbol, span: Span) {
        if !sym.can_be_raw() {
            self.err_span(span, &format!("`{}` cannot be a raw identifier", sym));
        }
    }

    pub(super) fn verify_no_underscore_literal_suffix(
        &self,
        suffix_start: BytePos,
        string: &str
    ) -> Result<(), ()>{
        if string == "_" {
            self.sess
                .span_diagnostic
                .struct_span_warn(
                    self.mk_sp(suffix_start, self.pos),
                    "underscore literal suffix is not allowed",
                )
                .warn(
                    "this was previously accepted by the compiler but is \
                     being phased out; it will become a hard error in \
                     a future release!",
                )
                .note(
                    "for more information, see issue #42326 \
                     <https://github.com/rust-lang/rust/issues/42326>",
                )
                .emit();
            Err(())
        } else {
            Ok(())
        }
    }

    pub(super) fn verify_lifetime(&self, start: BytePos, starts_with_number: bool) {
        if starts_with_number {
            self.err_span_(start, self.pos, "lifetimes cannot start with a number");
        }
    }

    pub(super) fn verify_literal_enclosed(
        &self,
        start: BytePos,
        suffix_start: BytePos,
        kind: rustc_lexer::LiteralKind
    ) {
        match kind {
            rustc_lexer::LiteralKind::Char { terminated } => {
                if !terminated {
                    self.fatal_span_(start, suffix_start, "unterminated character literal".into())
                        .raise()
                }
            }
            rustc_lexer::LiteralKind::Byte { terminated } => {
                if !terminated {
                    self.fatal_span_(
                        start + BytePos(1),
                        suffix_start,
                        "unterminated byte constant".into(),
                    )
                    .raise()
                }
            }
            rustc_lexer::LiteralKind::Str { terminated } => {
                if !terminated {
                    self.fatal_span_(start, suffix_start, "unterminated double quote string".into())
                        .raise()
                }
            }
            rustc_lexer::LiteralKind::ByteStr { terminated } => {
                if !terminated {
                    self.fatal_span_(
                        start + BytePos(1),
                        suffix_start,
                        "unterminated double quote byte string".into(),
                    )
                    .raise()
                }
            }
            rustc_lexer::LiteralKind::RawStr { n_hashes, started, terminated } => {
                if !started {
                    self.report_non_started_raw_string(start);
                }
                if !terminated {
                    self.report_unterminated_raw_string(start, n_hashes)
                }
            }
            rustc_lexer::LiteralKind::RawByteStr { n_hashes, started, terminated } => {
                if !started {
                    self.report_non_started_raw_string(start);
                }
                if !terminated {
                    self.report_unterminated_raw_string(start, n_hashes)
                }
            }
            token => panic!("Literal type {:?} cannot be 'enclosed'", token),
        }
    }

    pub(super) fn verify_int_not_empty(
        &self,
        start: BytePos,
        suffix_start: BytePos,
        empty_int: bool
    ) -> Result<(), ()> {
        if empty_int {
            self.err_span_(start, suffix_start, "no valid digits found for number");
            Err(())
        } else {
            Ok(())
        }
    }

    pub(super) fn verify_float_exponent_not_empty(&self, start: BytePos, empty_exponent: bool) {
        if empty_exponent {
            let mut err = self.struct_span_fatal(
                start,
                self.pos,
                "expected at least one digit in exponent",
            );
            err.emit();
        }
    }

    pub(super) fn verify_float_base(&self, start: BytePos, suffix_start: BytePos, base: Base) {
        match base {
            Base::Hexadecimal => self.err_span_(
                start,
                suffix_start,
                "hexadecimal float literal is not supported",
            ),
            Base::Octal => {
                self.err_span_(start, suffix_start, "octal float literal is not supported")
            }
            Base::Binary => {
                self.err_span_(start, suffix_start, "binary float literal is not supported")
            }
            _ => (),
        }
    }

    /// Checks that there is no bare CR in the provided string.
    fn verify_no_bare_cr(&self, start: BytePos, s: &str, errmsg: &str) {
        let mut idx = 0;
        loop {
            idx = match s[idx..].find('\r') {
                None => break,
                Some(it) => idx + it + 1,
            };
            self.err_span_(start + BytePos(idx as u32 - 1), start + BytePos(idx as u32), errmsg);
        }
    }

    fn report_non_started_raw_string(&self, start: BytePos) -> ! {
        let bad_char = self.str_from(start).chars().last().unwrap();
        self.struct_fatal_span_char(
            start,
            self.pos,
            "found invalid character; only `#` is allowed \
             in raw string delimitation",
            bad_char,
        )
        .emit();
        FatalError.raise()
    }

    fn report_unterminated_raw_string(&self, start: BytePos, n_hashes: usize) -> ! {
        let mut err = self.struct_span_fatal(start, start, "unterminated raw string");
        err.span_label(self.mk_sp(start, start), "unterminated raw string");

        if n_hashes > 0 {
            err.note(&format!(
                "this raw string should be terminated with `\"{}`",
                "#".repeat(n_hashes as usize)
            ));
        }

        err.emit();
        FatalError.raise()
    }
}
