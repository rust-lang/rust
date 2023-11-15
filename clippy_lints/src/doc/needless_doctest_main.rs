use std::ops::Range;
use std::{io, thread};

use crate::doc::NEEDLESS_DOCTEST_MAIN;
use clippy_utils::diagnostics::span_lint;
use rustc_ast::{Async, Fn, FnRetTy, ItemKind};
use rustc_data_structures::sync::Lrc;
use rustc_errors::emitter::EmitterWriter;
use rustc_errors::Handler;
use rustc_lint::LateContext;
use rustc_parse::maybe_new_parser_from_source_str;
use rustc_parse::parser::ForceCollect;
use rustc_session::parse::ParseSess;
use rustc_span::edition::Edition;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{sym, FileName};

use super::Fragments;

pub fn check(cx: &LateContext<'_>, text: &str, edition: Edition, range: Range<usize>, fragments: Fragments<'_>) {
    fn has_needless_main(code: String, edition: Edition) -> bool {
        rustc_driver::catch_fatal_errors(|| {
            rustc_span::create_session_globals_then(edition, || {
                let filename = FileName::anon_source_code(&code);

                let fallback_bundle =
                    rustc_errors::fallback_fluent_bundle(rustc_driver::DEFAULT_LOCALE_RESOURCES.to_vec(), false);
                let emitter = EmitterWriter::new(Box::new(io::sink()), fallback_bundle);
                let handler = Handler::with_emitter(Box::new(emitter)).disable_warnings();
                #[expect(clippy::arc_with_non_send_sync)] // `Lrc` is expected by with_span_handler
                let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
                let sess = ParseSess::with_span_handler(handler, sm);

                let mut parser = match maybe_new_parser_from_source_str(&sess, filename, code) {
                    Ok(p) => p,
                    Err(errs) => {
                        drop(errs);
                        return false;
                    },
                };

                let mut relevant_main_found = false;
                loop {
                    match parser.parse_item(ForceCollect::No) {
                        Ok(Some(item)) => match &item.kind {
                            ItemKind::Fn(box Fn {
                                sig, body: Some(block), ..
                            }) if item.ident.name == sym::main => {
                                let is_async = matches!(sig.header.asyncness, Async::Yes { .. });
                                let returns_nothing = match &sig.decl.output {
                                    FnRetTy::Default(..) => true,
                                    FnRetTy::Ty(ty) if ty.kind.is_unit() => true,
                                    FnRetTy::Ty(_) => false,
                                };

                                if returns_nothing && !is_async && !block.stmts.is_empty() {
                                    // This main function should be linted, but only if there are no other functions
                                    relevant_main_found = true;
                                } else {
                                    // This main function should not be linted, we're done
                                    return false;
                                }
                            },
                            // Tests with one of these items are ignored
                            ItemKind::Static(..)
                            | ItemKind::Const(..)
                            | ItemKind::ExternCrate(..)
                            | ItemKind::ForeignMod(..)
                            // Another function was found; this case is ignored
                            | ItemKind::Fn(..) => return false,
                            _ => {},
                        },
                        Ok(None) => break,
                        Err(e) => {
                            e.cancel();
                            return false;
                        },
                    }
                }

                relevant_main_found
            })
        })
        .ok()
        .unwrap_or_default()
    }

    let trailing_whitespace = text.len() - text.trim_end().len();

    // Because of the global session, we need to create a new session in a different thread with
    // the edition we need.
    let text = text.to_owned();
    if thread::spawn(move || has_needless_main(text, edition))
        .join()
        .expect("thread::spawn failed")
        && let Some(span) = fragments.span(cx, range.start..range.end - trailing_whitespace)
    {
        span_lint(cx, NEEDLESS_DOCTEST_MAIN, span, "needless `fn main` in doctest");
    }
}
