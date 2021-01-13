//! Contains information about "passes", used to modify crate information during the documentation
//! process.

use rustc_span::{InnerSpan, Span, DUMMY_SP};
use std::ops::Range;

use self::Condition::*;
use crate::clean::{self, DocFragmentKind};
use crate::core::DocContext;

mod stripper;
crate use stripper::*;

mod non_autolinks;
crate use self::non_autolinks::CHECK_NON_AUTOLINKS;

mod strip_hidden;
crate use self::strip_hidden::STRIP_HIDDEN;

mod strip_private;
crate use self::strip_private::STRIP_PRIVATE;

mod strip_priv_imports;
crate use self::strip_priv_imports::STRIP_PRIV_IMPORTS;

mod unindent_comments;
crate use self::unindent_comments::UNINDENT_COMMENTS;

mod propagate_doc_cfg;
crate use self::propagate_doc_cfg::PROPAGATE_DOC_CFG;

mod collect_intra_doc_links;
crate use self::collect_intra_doc_links::COLLECT_INTRA_DOC_LINKS;

mod doc_test_lints;
crate use self::doc_test_lints::CHECK_PRIVATE_ITEMS_DOC_TESTS;

mod collect_trait_impls;
crate use self::collect_trait_impls::COLLECT_TRAIT_IMPLS;

mod check_code_block_syntax;
crate use self::check_code_block_syntax::CHECK_CODE_BLOCK_SYNTAX;

mod calculate_doc_coverage;
crate use self::calculate_doc_coverage::CALCULATE_DOC_COVERAGE;

mod html_tags;
crate use self::html_tags::CHECK_INVALID_HTML_TAGS;

/// A single pass over the cleaned documentation.
///
/// Runs in the compiler context, so it has access to types and traits and the like.
#[derive(Copy, Clone)]
crate struct Pass {
    crate name: &'static str,
    crate run: fn(clean::Crate, &DocContext<'_>) -> clean::Crate,
    crate description: &'static str,
}

/// In a list of passes, a pass that may or may not need to be run depending on options.
#[derive(Copy, Clone)]
crate struct ConditionalPass {
    crate pass: Pass,
    crate condition: Condition,
}

/// How to decide whether to run a conditional pass.
#[derive(Copy, Clone)]
crate enum Condition {
    Always,
    /// When `--document-private-items` is passed.
    WhenDocumentPrivate,
    /// When `--document-private-items` is not passed.
    WhenNotDocumentPrivate,
    /// When `--document-hidden-items` is not passed.
    WhenNotDocumentHidden,
}

/// The full list of passes.
crate const PASSES: &[Pass] = &[
    CHECK_PRIVATE_ITEMS_DOC_TESTS,
    STRIP_HIDDEN,
    UNINDENT_COMMENTS,
    STRIP_PRIVATE,
    STRIP_PRIV_IMPORTS,
    PROPAGATE_DOC_CFG,
    COLLECT_INTRA_DOC_LINKS,
    CHECK_CODE_BLOCK_SYNTAX,
    COLLECT_TRAIT_IMPLS,
    CALCULATE_DOC_COVERAGE,
    CHECK_INVALID_HTML_TAGS,
    CHECK_NON_AUTOLINKS,
];

/// The list of passes run by default.
crate const DEFAULT_PASSES: &[ConditionalPass] = &[
    ConditionalPass::always(COLLECT_TRAIT_IMPLS),
    ConditionalPass::always(UNINDENT_COMMENTS),
    ConditionalPass::always(CHECK_PRIVATE_ITEMS_DOC_TESTS),
    ConditionalPass::new(STRIP_HIDDEN, WhenNotDocumentHidden),
    ConditionalPass::new(STRIP_PRIVATE, WhenNotDocumentPrivate),
    ConditionalPass::new(STRIP_PRIV_IMPORTS, WhenDocumentPrivate),
    ConditionalPass::always(COLLECT_INTRA_DOC_LINKS),
    ConditionalPass::always(CHECK_CODE_BLOCK_SYNTAX),
    ConditionalPass::always(CHECK_INVALID_HTML_TAGS),
    ConditionalPass::always(PROPAGATE_DOC_CFG),
    ConditionalPass::always(CHECK_NON_AUTOLINKS),
];

/// The list of default passes run when `--doc-coverage` is passed to rustdoc.
crate const COVERAGE_PASSES: &[ConditionalPass] = &[
    ConditionalPass::always(COLLECT_TRAIT_IMPLS),
    ConditionalPass::new(STRIP_HIDDEN, WhenNotDocumentHidden),
    ConditionalPass::new(STRIP_PRIVATE, WhenNotDocumentPrivate),
    ConditionalPass::always(CALCULATE_DOC_COVERAGE),
];

impl ConditionalPass {
    crate const fn always(pass: Pass) -> Self {
        Self::new(pass, Always)
    }

    crate const fn new(pass: Pass, condition: Condition) -> Self {
        ConditionalPass { pass, condition }
    }
}

/// A shorthand way to refer to which set of passes to use, based on the presence of
/// `--no-defaults` and `--show-coverage`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
crate enum DefaultPassOption {
    Default,
    Coverage,
    None,
}

/// Returns the given default set of passes.
crate fn defaults(default_set: DefaultPassOption) -> &'static [ConditionalPass] {
    match default_set {
        DefaultPassOption::Default => DEFAULT_PASSES,
        DefaultPassOption::Coverage => COVERAGE_PASSES,
        DefaultPassOption::None => &[],
    }
}

/// If the given name matches a known pass, returns its information.
crate fn find_pass(pass_name: &str) -> Option<Pass> {
    PASSES.iter().find(|p| p.name == pass_name).copied()
}

/// Returns a span encompassing all the given attributes.
crate fn span_of_attrs(attrs: &clean::Attributes) -> Option<Span> {
    if attrs.doc_strings.is_empty() {
        return None;
    }
    let start = attrs.doc_strings[0].span;
    if start == DUMMY_SP {
        return None;
    }
    let end = attrs.doc_strings.last().expect("no doc strings provided").span;
    Some(start.to(end))
}

/// Attempts to match a range of bytes from parsed markdown to a `Span` in the source code.
///
/// This method will return `None` if we cannot construct a span from the source map or if the
/// attributes are not all sugared doc comments. It's difficult to calculate the correct span in
/// that case due to escaping and other source features.
crate fn source_span_for_markdown_range(
    cx: &DocContext<'_>,
    markdown: &str,
    md_range: &Range<usize>,
    attrs: &clean::Attributes,
) -> Option<Span> {
    let is_all_sugared_doc =
        attrs.doc_strings.iter().all(|frag| frag.kind == DocFragmentKind::SugaredDoc);

    if !is_all_sugared_doc {
        return None;
    }

    let snippet = cx.sess().source_map().span_to_snippet(span_of_attrs(attrs)?).ok()?;

    let starting_line = markdown[..md_range.start].matches('\n').count();
    let ending_line = starting_line + markdown[md_range.start..md_range.end].matches('\n').count();

    // We use `split_terminator('\n')` instead of `lines()` when counting bytes so that we treat
    // CRLF and LF line endings the same way.
    let mut src_lines = snippet.split_terminator('\n');
    let md_lines = markdown.split_terminator('\n');

    // The number of bytes from the source span to the markdown span that are not part
    // of the markdown, like comment markers.
    let mut start_bytes = 0;
    let mut end_bytes = 0;

    'outer: for (line_no, md_line) in md_lines.enumerate() {
        loop {
            let source_line = src_lines.next().expect("could not find markdown in source");
            match source_line.find(md_line) {
                Some(offset) => {
                    if line_no == starting_line {
                        start_bytes += offset;

                        if starting_line == ending_line {
                            break 'outer;
                        }
                    } else if line_no == ending_line {
                        end_bytes += offset;
                        break 'outer;
                    } else if line_no < starting_line {
                        start_bytes += source_line.len() - md_line.len();
                    } else {
                        end_bytes += source_line.len() - md_line.len();
                    }
                    break;
                }
                None => {
                    // Since this is a source line that doesn't include a markdown line,
                    // we have to count the newline that we split from earlier.
                    if line_no <= starting_line {
                        start_bytes += source_line.len() + 1;
                    } else {
                        end_bytes += source_line.len() + 1;
                    }
                }
            }
        }
    }

    Some(span_of_attrs(attrs)?.from_inner(InnerSpan::new(
        md_range.start + start_bytes,
        md_range.end + start_bytes + end_bytes,
    )))
}
