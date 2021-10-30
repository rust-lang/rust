use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use rustc_ast::token::{DelimToken, TokenKind};
use rustc_ast::{ast, ptr};
use rustc_errors::Diagnostic;
use rustc_parse::{
    new_parser_from_file,
    parser::{ForceCollect, Parser as RawParser},
};
use rustc_span::{sym, symbol::kw, Span};

use crate::attr::first_attr_value_str_by_name;
use crate::syntux::session::ParseSess;
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
        let path_string = first_attr_value_str_by_name(attrs, sym::path)?.as_str();
        // On windows, the base path might have the form
        // `\\?\foo\bar` in which case it does not tolerate
        // mixed `/` and `\` separators, so canonicalize
        // `/` to `\`.
        #[cfg(windows)]
        let path_string = path_string.replace("/", "\\");

        Some(path.join(&*path_string))
    }

    pub(crate) fn parse_file_as_module(
        sess: &'a ParseSess,
        path: &Path,
        span: Span,
    ) -> Result<(Vec<ast::Attribute>, Vec<ptr::P<ast::Item>>, Span), ParserError> {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut parser = new_parser_from_file(sess.inner(), path, Some(span));
            match parser.parse_mod(&TokenKind::Eof) {
                Ok(result) => Some(result),
                Err(mut e) => {
                    sess.emit_or_cancel_diagnostic(&mut e);
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

    pub(crate) fn parse_cfg_if(
        sess: &'a ParseSess,
        mac: &'a ast::MacCall,
    ) -> Result<Vec<ast::Item>, &'static str> {
        match catch_unwind(AssertUnwindSafe(|| Parser::parse_cfg_if_inner(sess, mac))) {
            Ok(Ok(items)) => Ok(items),
            Ok(err @ Err(_)) => err,
            Err(..) => Err("failed to parse cfg_if!"),
        }
    }

    fn parse_cfg_if_inner(
        sess: &'a ParseSess,
        mac: &'a ast::MacCall,
    ) -> Result<Vec<ast::Item>, &'static str> {
        let token_stream = mac.args.inner_tokens();
        let mut parser = rustc_parse::stream_to_parser(sess.inner(), token_stream, Some(""));

        let mut items = vec![];
        let mut process_if_cfg = true;

        while parser.token.kind != TokenKind::Eof {
            if process_if_cfg {
                if !parser.eat_keyword(kw::If) {
                    return Err("Expected `if`");
                }
                // Inner attributes are not actually syntactically permitted here, but we don't
                // care about inner vs outer attributes in this position. Our purpose with this
                // special case parsing of cfg_if macros is to ensure we can correctly resolve
                // imported modules that may have a custom `path` defined.
                //
                // As such, we just need to advance the parser past the attribute and up to
                // to the opening brace.
                // See also https://github.com/rust-lang/rust/pull/79433
                parser
                    .parse_attribute(rustc_parse::parser::attr::InnerAttrPolicy::Permitted)
                    .map_err(|_| "Failed to parse attributes")?;
            }

            if !parser.eat(&TokenKind::OpenDelim(DelimToken::Brace)) {
                return Err("Expected an opening brace");
            }

            while parser.token != TokenKind::CloseDelim(DelimToken::Brace)
                && parser.token.kind != TokenKind::Eof
            {
                let item = match parser.parse_item(ForceCollect::No) {
                    Ok(Some(item_ptr)) => item_ptr.into_inner(),
                    Ok(None) => continue,
                    Err(mut err) => {
                        err.cancel();
                        parser.sess.span_diagnostic.reset_err_count();
                        return Err(
                            "Expected item inside cfg_if block, but failed to parse it as an item",
                        );
                    }
                };
                if let ast::ItemKind::Mod(..) = item.kind {
                    items.push(item);
                }
            }

            if !parser.eat(&TokenKind::CloseDelim(DelimToken::Brace)) {
                return Err("Expected a closing brace");
            }

            if parser.eat(&TokenKind::Eof) {
                break;
            }

            if !parser.eat_keyword(kw::Else) {
                return Err("Expected `else`");
            }

            process_if_cfg = parser.token.is_keyword(kw::If);
        }

        Ok(items)
    }
}
