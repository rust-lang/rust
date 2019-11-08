//! This module contains utilities for literal tokens validation.

use super::unescape_error_reporting::emit_unescape_error;
use super::StringReader;

use rustc_lexer::unescape;
use rustc_lexer::Base;
use syntax_pos::BytePos;

// Extensions for the `StringReader` providing verification helper
// methods.
impl<'a> StringReader<'a> {
    pub(super) fn validate_raw_str_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        unescape::unescape_raw_str(lit, &mut |range, c| {
            if let Err(err) = c {
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit,
                    self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                    unescape::Mode::Str,
                    range,
                    err,
                )
            }
        })
    }

    pub(super) fn validate_char_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        if let Err((off, err)) = unescape::unescape_char(lit) {
            emit_unescape_error(
                &self.sess.span_diagnostic,
                lit,
                self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                unescape::Mode::Char,
                0..off,
                err,
            )
        }
    }

    pub(super) fn validate_byte_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        if let Err((off, err)) = unescape::unescape_byte(lit) {
            emit_unescape_error(
                &self.sess.span_diagnostic,
                lit,
                self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                unescape::Mode::Byte,
                0..off,
                err,
            )
        }
    }

    pub(super) fn validate_str_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        unescape::unescape_str(lit, &mut |range, c| {
            if let Err(err) = c {
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit,
                    self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                    unescape::Mode::Str,
                    range,
                    err,
                )
            }
        })
    }

    pub(super) fn validate_raw_byte_str_escape(
        &self,
        content_start: BytePos,
        content_end: BytePos,
    ) {
        let lit = self.str_from_to(content_start, content_end);
        unescape::unescape_raw_byte_str(lit, &mut |range, c| {
            if let Err(err) = c {
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit,
                    self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                    unescape::Mode::ByteStr,
                    range,
                    err,
                )
            }
        })
    }

    pub(super) fn validate_byte_str_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        unescape::unescape_byte_str(lit, &mut |range, c| {
            if let Err(err) = c {
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit,
                    self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                    unescape::Mode::ByteStr,
                    range,
                    err,
                )
            }
        })
    }

    pub(super) fn validate_int_literal(
        &self,
        base: Base,
        content_start: BytePos,
        content_end: BytePos,
    ) {
        let base = match base {
            Base::Binary => 2,
            Base::Octal => 8,
            _ => return,
        };
        let s = self.str_from_to(content_start + BytePos(2), content_end);
        for (idx, c) in s.char_indices() {
            let idx = idx as u32;
            if c != '_' && c.to_digit(base).is_none() {
                let lo = content_start + BytePos(2 + idx);
                let hi = content_start + BytePos(2 + idx + c.len_utf8() as u32);
                self.err_span_(lo, hi, &format!("invalid digit for a base {} literal", base));
            }
        }
    }
}
