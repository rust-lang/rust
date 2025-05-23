//! Rustdoc's doctest extraction.
//!
//! This module contains the logic to extract doctests and output a JSON containing this
//! information.

use rustc_span::edition::Edition;
use serde::Serialize;

use super::make::DocTestWrapResult;
use super::{BuildDocTestBuilder, ScrapedDocTest};
use crate::config::Options as RustdocOptions;
use crate::html::markdown;

/// The version of JSON output that this code generates.
///
/// This integer is incremented with every breaking change to the API,
/// and is returned along with the JSON blob into the `format_version` root field.
/// Consuming code should assert that this value matches the format version(s) that it supports.
const FORMAT_VERSION: u32 = 2;

#[derive(Serialize)]
pub(crate) struct ExtractedDocTests {
    format_version: u32,
    doctests: Vec<ExtractedDocTest>,
}

impl ExtractedDocTests {
    pub(crate) fn new() -> Self {
        Self { format_version: FORMAT_VERSION, doctests: Vec::new() }
    }

    pub(crate) fn add_test(
        &mut self,
        scraped_test: ScrapedDocTest,
        opts: &super::GlobalTestOptions,
        options: &RustdocOptions,
    ) {
        let edition = scraped_test.edition(options);
        self.add_test_with_edition(scraped_test, opts, edition)
    }

    /// This method is used by unit tests to not have to provide a `RustdocOptions`.
    pub(crate) fn add_test_with_edition(
        &mut self,
        scraped_test: ScrapedDocTest,
        opts: &super::GlobalTestOptions,
        edition: Edition,
    ) {
        let ScrapedDocTest { filename, line, langstr, text, name, global_crate_attrs, .. } =
            scraped_test;

        let doctest = BuildDocTestBuilder::new(&text)
            .crate_name(&opts.crate_name)
            .global_crate_attrs(global_crate_attrs)
            .edition(edition)
            .lang_str(&langstr)
            .build(None);
        let (wrapped, _size) = doctest.generate_unique_doctest(
            &text,
            langstr.test_harness,
            opts,
            Some(&opts.crate_name),
        );
        self.doctests.push(ExtractedDocTest {
            file: filename.prefer_remapped_unconditionaly().to_string(),
            line,
            doctest_attributes: langstr.into(),
            doctest_code: match wrapped {
                DocTestWrapResult::Valid { crate_level_code, wrapper, code } => Some(DocTest {
                    crate_level: crate_level_code,
                    code,
                    wrapper: wrapper.map(
                        |super::make::WrapperInfo { before, after, returns_result, .. }| {
                            WrapperInfo { before, after, returns_result }
                        },
                    ),
                }),
                DocTestWrapResult::SyntaxError { .. } => None,
            },
            original_code: text,
            name,
        });
    }

    #[cfg(test)]
    pub(crate) fn doctests(&self) -> &[ExtractedDocTest] {
        &self.doctests
    }
}

#[derive(Serialize)]
pub(crate) struct WrapperInfo {
    before: String,
    after: String,
    returns_result: bool,
}

#[derive(Serialize)]
pub(crate) struct DocTest {
    crate_level: String,
    code: String,
    /// This field can be `None` if one of the following conditions is true:
    ///
    /// * The doctest's codeblock has the `test_harness` attribute.
    /// * The doctest has a `main` function.
    /// * The doctest has the `![no_std]` attribute.
    pub(crate) wrapper: Option<WrapperInfo>,
}

#[derive(Serialize)]
pub(crate) struct ExtractedDocTest {
    file: String,
    line: usize,
    doctest_attributes: LangString,
    original_code: String,
    /// `None` if the code syntax is invalid.
    pub(crate) doctest_code: Option<DocTest>,
    name: String,
}

#[derive(Serialize)]
pub(crate) enum Ignore {
    All,
    None,
    Some(Vec<String>),
}

impl From<markdown::Ignore> for Ignore {
    fn from(original: markdown::Ignore) -> Self {
        match original {
            markdown::Ignore::All => Self::All,
            markdown::Ignore::None => Self::None,
            markdown::Ignore::Some(values) => Self::Some(values),
        }
    }
}

#[derive(Serialize)]
struct LangString {
    pub(crate) original: String,
    pub(crate) should_panic: bool,
    pub(crate) no_run: bool,
    pub(crate) ignore: Ignore,
    pub(crate) rust: bool,
    pub(crate) test_harness: bool,
    pub(crate) compile_fail: bool,
    pub(crate) standalone_crate: bool,
    pub(crate) error_codes: Vec<String>,
    pub(crate) edition: Option<String>,
    pub(crate) added_css_classes: Vec<String>,
    pub(crate) unknown: Vec<String>,
}

impl From<markdown::LangString> for LangString {
    fn from(original: markdown::LangString) -> Self {
        let markdown::LangString {
            original,
            should_panic,
            no_run,
            ignore,
            rust,
            test_harness,
            compile_fail,
            standalone_crate,
            error_codes,
            edition,
            added_classes,
            unknown,
        } = original;

        Self {
            original,
            should_panic,
            no_run,
            ignore: ignore.into(),
            rust,
            test_harness,
            compile_fail,
            standalone_crate,
            error_codes,
            edition: edition.map(|edition| edition.to_string()),
            added_css_classes: added_classes,
            unknown,
        }
    }
}
