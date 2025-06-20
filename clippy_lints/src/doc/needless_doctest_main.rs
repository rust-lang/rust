use std::ops::Range;
use std::sync::Arc;
use std::{io, thread};

use crate::doc::{NEEDLESS_DOCTEST_MAIN, TEST_ATTR_IN_DOCTEST};
use clippy_utils::diagnostics::span_lint;
use rustc_ast::{CoroutineKind, Fn, FnRetTy, Item, ItemKind};
use rustc_errors::emitter::HumanEmitter;
use rustc_errors::{Diag, DiagCtxt};
use rustc_lint::LateContext;
use rustc_parse::new_parser_from_source_str;
use rustc_parse::parser::ForceCollect;
use rustc_session::parse::ParseSess;
use rustc_span::edition::Edition;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{FileName, Ident, Pos, sym};

use super::Fragments;

fn get_test_spans(item: &Item, ident: Ident, test_attr_spans: &mut Vec<Range<usize>>) {
    test_attr_spans.extend(
        item.attrs
            .iter()
            .find(|attr| attr.has_name(sym::test))
            .map(|attr| attr.span.lo().to_usize()..ident.span.hi().to_usize()),
    );
}

pub fn check(
    cx: &LateContext<'_>,
    text: &str,
    edition: Edition,
    range: Range<usize>,
    fragments: Fragments<'_>,
    ignore: bool,
) {
    // return whether the code contains a needless `fn main` plus a vector of byte position ranges
    // of all `#[test]` attributes in not ignored code examples
    fn check_code_sample(code: String, edition: Edition, ignore: bool) -> (bool, Vec<Range<usize>>) {
        rustc_driver::catch_fatal_errors(|| {
            rustc_span::create_session_globals_then(edition, &[], None, || {
                let mut test_attr_spans = vec![];
                let filename = FileName::anon_source_code(&code);

                let translator = rustc_driver::default_translator();
                let emitter = HumanEmitter::new(Box::new(io::sink()), translator);
                let dcx = DiagCtxt::new(Box::new(emitter)).disable_warnings();
                #[expect(clippy::arc_with_non_send_sync)] // `Arc` is expected by with_dcx
                let sm = Arc::new(SourceMap::new(FilePathMapping::empty()));
                let psess = ParseSess::with_dcx(dcx, sm);

                let mut parser = match new_parser_from_source_str(&psess, filename, code) {
                    Ok(p) => p,
                    Err(errs) => {
                        errs.into_iter().for_each(Diag::cancel);
                        return (false, test_attr_spans);
                    },
                };

                let mut relevant_main_found = false;
                let mut eligible = true;
                loop {
                    match parser.parse_item(ForceCollect::No) {
                        Ok(Some(item)) => match &item.kind {
                            ItemKind::Fn(box Fn {
                                ident,
                                sig,
                                body: Some(block),
                                ..
                            }) if ident.name == sym::main => {
                                if !ignore {
                                    get_test_spans(&item, *ident, &mut test_attr_spans);
                                }
                                let is_async = matches!(sig.header.coroutine_kind, Some(CoroutineKind::Async { .. }));
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
                                    eligible = false;
                                }
                            },
                            // Another function was found; this case is ignored for needless_doctest_main
                            ItemKind::Fn(fn_) => {
                                eligible = false;
                                if !ignore {
                                    get_test_spans(&item, fn_.ident, &mut test_attr_spans);
                                }
                            },
                            // Tests with one of these items are ignored
                            ItemKind::Static(..)
                            | ItemKind::Const(..)
                            | ItemKind::ExternCrate(..)
                            | ItemKind::ForeignMod(..) => {
                                eligible = false;
                            },
                            _ => {},
                        },
                        Ok(None) => break,
                        Err(e) => {
                            e.cancel();
                            return (false, test_attr_spans);
                        },
                    }
                }

                (relevant_main_found & eligible, test_attr_spans)
            })
        })
        .ok()
        .unwrap_or_default()
    }

    let trailing_whitespace = text.len() - text.trim_end().len();

    // Because of the global session, we need to create a new session in a different thread with
    // the edition we need.
    let text = text.to_owned();
    let (has_main, test_attr_spans) = thread::spawn(move || check_code_sample(text, edition, ignore))
        .join()
        .expect("thread::spawn failed");
    if has_main && let Some(span) = fragments.span(cx, range.start..range.end - trailing_whitespace) {
        span_lint(cx, NEEDLESS_DOCTEST_MAIN, span, "needless `fn main` in doctest");
    }
    for span in test_attr_spans {
        let span = (range.start + span.start)..(range.start + span.end);
        if let Some(span) = fragments.span(cx, span) {
            span_lint(cx, TEST_ATTR_IN_DOCTEST, span, "unit tests in doctest are not executed");
        }
    }
}
