//! Detects invalid HTML (like an unclosed `<span>`) in doc comments.

use std::ops::Range;

use rustc_hir::HirId;
use rustc_resolve::rustdoc::pulldown_cmark::{BrokenLink, Event, LinkType, Parser, Tag, TagEnd};
use rustc_resolve::rustdoc::source_span_for_markdown_range;
use rustc_span::BytePos;

use crate::clean::*;
use crate::core::DocContext;
use crate::html::markdown::main_body_opts;

pub(crate) fn visit_item(
    cx: &DocContext<'_>,
    item: &Item,
    hir_id: HirId,
    dox: &str,
) {
    let tcx = cx.tcx;
    let report_diag = |mut msg: String, range: &Range<usize>| {
        let sp = match source_span_for_markdown_range(tcx, dox, range, &item.attrs.doc_strings) {
            Some((sp, _)) => sp,
            None => item.attr_span(tcx),
        };
        tcx.emit_node_span_lint(
            crate::lint::INVALID_MATH,
            hir_id,
            sp,
            rustc_errors::DiagDecorator(|lint| {
                use rustc_lint_defs::Applicability;

                if msg.len() >= 1 {
                    // different convention between math-core and rustdoc
                    msg[0..1].make_ascii_lowercase();
                }
                if msg.as_bytes().last() == Some(&b'.') {
                    msg.pop();
                }

                lint.primary_message(msg);

                let math = &dox[range.start..range.end];

                // https://gist.github.com/notriddle/108fe255ffa9f490ed8ade262935eb53
                if let Some(open) = math.find(r"\[")
                    && let Some(close) = math[open..].find(r"\]")
                {
                    let close = close + open;
                    lint.span_suggestion(
                        sp.with_lo(sp.lo() + BytePos(open.try_into().unwrap()))
                            .with_hi(sp.lo() + BytePos((close + 1).try_into().unwrap())),
                        "consider if the math span might be double-escaped",
                        format!("{}", &math[open + 1..close]),
                        Applicability::MaybeIncorrect,
                    );
                }

                // https://www.doxygen.nl/manual/formulas.html
                if math.ends_with(r"\f$")
                    && math.starts_with("$")
                    && dox[..range.start].ends_with(r"\f")
                {
                    let math_without_markers = &math[1..math.len() - 3];
                    lint.span_suggestion(
                        sp.with_lo(sp.lo() - BytePos(2)),
                        r"formulas do not require `\f`",
                        format!("${math_without_markers}$"),
                        Applicability::MaybeIncorrect,
                    );
                }
            }),
        );
    };

    let mut latexp = None;
    let mut in_code_block = false;

    let link_names = item.link_names(&cx.cache);

    let mut replacer = |broken_link: BrokenLink<'_>| {
        if let Some(link) =
            link_names.iter().find(|link| *link.original_text == *broken_link.reference)
        {
            Some((link.href.as_str().into(), link.new_text.to_string().into()))
        } else if matches!(&broken_link.link_type, LinkType::Reference | LinkType::ReferenceUnknown)
        {
            // If the link is shaped [like][this], suppress any broken HTML in the [this] part.
            // The `broken_intra_doc_links` will report typos in there anyway.
            Some((
                broken_link.reference.to_string().into(),
                broken_link.reference.to_string().into(),
            ))
        } else {
            None
        }
    };

    let p = Parser::new_with_broken_link_callback(
        dox,
        main_body_opts(cx.tcx.doc_attribute_syntax(item.item_id.expect_def_id())),
        Some(&mut replacer),
    )
    .into_offset_iter();

    for (event, range) in p {
        match event {
            Event::Start(Tag::CodeBlock(_)) => in_code_block = true,
            Event::InlineMath(latex) | Event::DisplayMath(latex) if !in_code_block => {
                let latexp = if let Some(parser) = latexp.as_ref() {
                    parser
                } else {
                    latexp.insert(math_core::LatexToMathML::new(math_core::MathCoreConfig::default()).expect("cannot fail without passing macros"))
                };
                if let Err(e) =
                    latexp.convert_with_local_counter(&latex, math_core::MathDisplay::Block)
                {
                    report_diag(e.error_message(), &range);
                }
            }
            Event::End(TagEnd::CodeBlock) => in_code_block = false,
            _ => {}
        }
    }
}
