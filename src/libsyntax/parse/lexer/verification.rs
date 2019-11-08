//! This module contains utilities for verifying the token state
//! and reporting errors on verification failures.
//!
//! The purpose of this module is to encapsulate the diagnostics
//! from the actual token parsing.

use super::unescape_error_reporting::{emit_unescape_error, push_escaped_char};
use super::StringReader;

// TODO remove unneeded imports.
use crate::sess::ParseSess;
use crate::symbol::{sym, Symbol};
use crate::token::{self, Token, TokenKind};
use crate::util::comments;

use errors::{DiagnosticBuilder, FatalError};
use rustc_lexer::unescape;
use rustc_lexer::Base;
use syntax_pos::{BytePos, Pos, Span};

use log::debug;
use rustc_data_structures::sync::Lrc;
use std::char;
use std::convert::TryInto;

// Extensions for the `StringReader` providing verification helper
// methods.
impl<'a> StringReader<'a> {



    /// Checks that there is no bare CR in the provided string.
    pub fn verify_no_bare_cr(&self, start: BytePos, s: &str, errmsg: &str) {
        let mut idx = 0;
        loop {
            idx = match s[idx..].find('\r') {
                None => break,
                Some(it) => idx + it + 1,
            };
            self.err_span_(start + BytePos(idx as u32 - 1), start + BytePos(idx as u32), errmsg);
        }
    }
}
