use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use rustc_ast::{ast, attr};
use rustc_errors::Diag;
use rustc_parse::lexer::StripTokens;
use rustc_parse::parser::Parser as RawParser;
use rustc_parse::{exp, new_parser_from_file, new_parser_from_source_str, unwrap_or_emit_fatal};
use rustc_span::{Span, sym};
use thin_vec::ThinVec;

use crate::Input;
use crate::parse::session::ParseSess;

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
    psess: Option<&'a ParseSess>,
    input: Option<Input>,
}

impl<'a> ParserBuilder<'a> {
    pub(crate) fn input(mut self, input: Input) -> ParserBuilder<'a> {
        self.input = Some(input);
        self
    }

    pub(crate) fn psess(mut self, psess: &'a ParseSess) -> ParserBuilder<'a> {
        self.psess = Some(psess);
        self
    }

    pub(crate) fn build(self) -> Result<Parser<'a>, ParserError> {
        let psess = self.psess.ok_or(ParserError::NoParseSess)?;
        let input = self.input.ok_or(ParserError::NoInput)?;

        let parser = match Self::parser(psess.inner(), input) {
            Ok(p) => p,
            Err(diagnostics) => {
                psess.emit_diagnostics(diagnostics);
                return Err(ParserError::ParserCreationError);
            }
        };

        Ok(Parser { parser })
    }

    fn parser(
        psess: &'a rustc_session::parse::ParseSess,
        input: Input,
    ) -> Result<RawParser<'a>, Vec<Diag<'a>>> {
        match input {
            Input::File(ref file) => {
                new_parser_from_file(psess, file, StripTokens::ShebangAndFrontmatter, None)
            }
            Input::Text(text) => new_parser_from_source_str(
                psess,
                rustc_span::FileName::Custom("stdin".to_owned()),
                text,
                StripTokens::ShebangAndFrontmatter,
            ),
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
        let path_sym = attr::first_attr_value_str_by_name(attrs, sym::path)?;
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
        psess: &'a ParseSess,
        path: &Path,
        span: Span,
    ) -> Result<(ast::AttrVec, ThinVec<Box<ast::Item>>, Span), ParserError> {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut parser = unwrap_or_emit_fatal(new_parser_from_file(
                psess.inner(),
                path,
                StripTokens::ShebangAndFrontmatter,
                Some(span),
            ));
            match parser.parse_mod(exp!(Eof)) {
                Ok((a, i, spans)) => Some((a, i, spans.inner_span)),
                Err(e) => {
                    e.emit();
                    if psess.can_reset_errors() {
                        psess.reset_errors();
                    }
                    None
                }
            }
        }));
        match result {
            Ok(Some(m)) if !psess.has_errors() => Ok(m),
            Ok(Some(m)) if psess.can_reset_errors() => {
                psess.reset_errors();
                Ok(m)
            }
            Ok(_) => Err(ParserError::ParseError),
            Err(..) if path.exists() => Err(ParserError::ParseError),
            Err(_) => Err(ParserError::ParsePanicError),
        }
    }

    pub(crate) fn parse_crate(
        input: Input,
        psess: &'a ParseSess,
    ) -> Result<ast::Crate, ParserError> {
        let krate = Parser::parse_crate_inner(input, psess)?;
        if !psess.has_errors() {
            return Ok(krate);
        }

        if psess.can_reset_errors() {
            psess.reset_errors();
            return Ok(krate);
        }

        Err(ParserError::ParseError)
    }

    fn parse_crate_inner(input: Input, psess: &'a ParseSess) -> Result<ast::Crate, ParserError> {
        ParserBuilder::default()
            .input(input)
            .psess(psess)
            .build()?
            .parse_crate_mod()
    }

    fn parse_crate_mod(&mut self) -> Result<ast::Crate, ParserError> {
        let mut parser = AssertUnwindSafe(&mut self.parser);
        let err = Err(ParserError::ParsePanicError);
        match catch_unwind(move || parser.parse_crate_mod()) {
            Ok(Ok(k)) => Ok(k),
            Ok(Err(db)) => {
                db.emit();
                err
            }
            Err(_) => err,
        }
    }
}
