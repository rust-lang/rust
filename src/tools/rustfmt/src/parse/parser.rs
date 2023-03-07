use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use rustc_ast::token::TokenKind;
use rustc_ast::{ast, ptr};
use rustc_errors::Diagnostic;
use rustc_parse::{new_parser_from_file, parser::Parser as RawParser};
use rustc_span::{sym, Span};
use thin_vec::ThinVec;

use crate::attr::first_attr_value_str_by_name;
use crate::parse::session::ParseSess;
use crate::Input;

pub(crate) type DirectoryOwnership = rustc_expand::module::DirOwnership;
pub(crate) type ModulePathSuccess = rustc_expand::module::ModulePathSuccess;
pub(crate) type ModError<'a> = rustc_expand::module::ModError<'a>;

#[derive(Clone)]
pub(crate) struct Directory {
    pub(crate) path: PathBuf,
    pub(crate) ownership: DirectoryOwnership,
}

/// A parser for Rust source code.
pub(crate) struct Parser<'a> {
    parser: RawParser<'a>,
}

/// A builder for the `Parser`.
#[derive(Default)]
pub(crate) struct ParserBuilder<'a> {
    sess: Option<&'a ParseSess>,
    input: Option<Input>,
}

impl<'a> ParserBuilder<'a> {
    pub(crate) fn input(mut self, input: Input) -> ParserBuilder<'a> {
        self.input = Some(input);
        self
    }

    pub(crate) fn sess(mut self, sess: &'a ParseSess) -> ParserBuilder<'a> {
        self.sess = Some(sess);
        self
    }

    pub(crate) fn build(self) -> Result<Parser<'a>, ParserError> {
        let sess = self.sess.ok_or(ParserError::NoParseSess)?;
        let input = self.input.ok_or(ParserError::NoInput)?;

        let parser = match Self::parser(sess.inner(), input) {
            Ok(p) => p,
            Err(db) => {
                if let Some(diagnostics) = db {
                    sess.emit_diagnostics(diagnostics);
                    return Err(ParserError::ParserCreationError);
                }
                return Err(ParserError::ParsePanicError);
            }
        };

        Ok(Parser { parser })
    }

    fn parser(
        sess: &'a rustc_session::parse::ParseSess,
        input: Input,
    ) -> Result<rustc_parse::parser::Parser<'a>, Option<Vec<Diagnostic>>> {
        match input {
            Input::File(ref file) => catch_unwind(AssertUnwindSafe(move || {
                new_parser_from_file(sess, file, None)
            }))
            .map_err(|_| None),
            Input::Text(text) => rustc_parse::maybe_new_parser_from_source_str(
                sess,
                rustc_span::FileName::Custom("stdin".to_owned()),
                text,
            )
            .map_err(Some),
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum ParserError {
    NoParseSess,
    NoInput,
    ParserCreationError,
    ParseError,
    ParsePanicError,
}

impl<'a> Parser<'a> {
    pub(crate) fn submod_path_from_attr(attrs: &[ast::Attribute], path: &Path) -> Option<PathBuf> {
        let path_sym = first_attr_value_str_by_name(attrs, sym::path)?;
        let path_str = path_sym.as_str();

        // On windows, the base path might have the form
        // `\\?\foo\bar` in which case it does not tolerate
        // mixed `/` and `\` separators, so canonicalize
        // `/` to `\`.
        #[cfg(windows)]
        let path_str = path_str.replace("/", "\\");

        Some(path.join(path_str))
    }

    pub(crate) fn parse_file_as_module(
        sess: &'a ParseSess,
        path: &Path,
        span: Span,
    ) -> Result<(ast::AttrVec, ThinVec<ptr::P<ast::Item>>, Span), ParserError> {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut parser = new_parser_from_file(sess.inner(), path, Some(span));
            match parser.parse_mod(&TokenKind::Eof) {
                Ok((a, i, spans)) => Some((a, i, spans.inner_span)),
                Err(mut e) => {
                    e.emit();
                    if sess.can_reset_errors() {
                        sess.reset_errors();
                    }
                    None
                }
            }
        }));
        match result {
            Ok(Some(m)) if !sess.has_errors() => Ok(m),
            Ok(Some(m)) if sess.can_reset_errors() => {
                sess.reset_errors();
                Ok(m)
            }
            Ok(_) => Err(ParserError::ParseError),
            Err(..) if path.exists() => Err(ParserError::ParseError),
            Err(_) => Err(ParserError::ParsePanicError),
        }
    }

    pub(crate) fn parse_crate(
        input: Input,
        sess: &'a ParseSess,
    ) -> Result<ast::Crate, ParserError> {
        let krate = Parser::parse_crate_inner(input, sess)?;
        if !sess.has_errors() {
            return Ok(krate);
        }

        if sess.can_reset_errors() {
            sess.reset_errors();
            return Ok(krate);
        }

        Err(ParserError::ParseError)
    }

    fn parse_crate_inner(input: Input, sess: &'a ParseSess) -> Result<ast::Crate, ParserError> {
        ParserBuilder::default()
            .input(input)
            .sess(sess)
            .build()?
            .parse_crate_mod()
    }

    fn parse_crate_mod(&mut self) -> Result<ast::Crate, ParserError> {
        let mut parser = AssertUnwindSafe(&mut self.parser);

        match catch_unwind(move || parser.parse_crate_mod()) {
            Ok(Ok(k)) => Ok(k),
            Ok(Err(mut db)) => {
                db.emit();
                Err(ParserError::ParseError)
            }
            Err(_) => Err(ParserError::ParsePanicError),
        }
    }
}
