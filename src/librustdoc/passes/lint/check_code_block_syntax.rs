//! Validates syntax inside Rust code blocks (\`\`\`rust).

use std::borrow::Cow;
use std::sync::Arc;

use rustc_data_structures::sync::Lock;
use rustc_errors::emitter::Emitter;
use rustc_errors::registry::Registry;
use rustc_errors::translation::{Translator, to_fluent_args};
use rustc_errors::{Applicability, DiagCtxt, DiagInner};
use rustc_parse::{source_str_to_stream, unwrap_or_emit_fatal};
use rustc_resolve::rustdoc::source_span_for_markdown_range;
use rustc_session::parse::ParseSess;
use rustc_span::hygiene::{AstPass, ExpnData, ExpnKind, LocalExpnId, Transparency};
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{DUMMY_SP, FileName, InnerSpan};

use crate::clean;
use crate::core::DocContext;
use crate::html::markdown::{self, RustCodeBlock};

pub(crate) fn visit_item(cx: &DocContext<'_>, item: &clean::Item, dox: &str) {
    if let Some(def_id) = item.item_id.as_local_def_id() {
        let sp = item.attr_span(cx.tcx);
        let extra = crate::html::markdown::ExtraInfo::new(cx.tcx, def_id, sp);
        for code_block in markdown::rust_code_blocks(dox, &extra) {
            check_rust_syntax(cx, item, dox, code_block);
        }
    }
}

fn check_rust_syntax(
    cx: &DocContext<'_>,
    item: &clean::Item,
    dox: &str,
    code_block: RustCodeBlock,
) {
    let buffer = Arc::new(Lock::new(Buffer::default()));
    let translator = rustc_driver::default_translator();
    let emitter = BufferEmitter { buffer: Arc::clone(&buffer), translator };

    let sm = Arc::new(SourceMap::new(FilePathMapping::empty()));
    let dcx = DiagCtxt::new(Box::new(emitter)).disable_warnings();
    let source = dox[code_block.code]
        .lines()
        .map(|line| crate::html::markdown::map_line(line).for_code())
        .intersperse(Cow::Borrowed("\n"))
        .collect::<String>();
    let psess = ParseSess::with_dcx(dcx, sm);

    let edition = code_block.lang_string.edition.unwrap_or_else(|| cx.tcx.sess.edition());
    let expn_data =
        ExpnData::default(ExpnKind::AstPass(AstPass::TestHarness), DUMMY_SP, edition, None, None);
    let expn_id = cx.tcx.with_stable_hashing_context(|hcx| LocalExpnId::fresh(expn_data, hcx));
    let span = DUMMY_SP.apply_mark(expn_id.to_expn_id(), Transparency::Transparent);

    let is_empty = rustc_driver::catch_fatal_errors(|| {
        unwrap_or_emit_fatal(source_str_to_stream(
            &psess,
            FileName::Custom(String::from("doctest")),
            source,
            Some(span),
        ))
        .is_empty()
    })
    .unwrap_or(false);
    let buffer = buffer.borrow();

    if !buffer.has_errors && !is_empty {
        // No errors in a non-empty program.
        return;
    }

    let Some(local_id) = item.item_id.as_def_id().and_then(|x| x.as_local()) else {
        // We don't need to check the syntax for other crates so returning
        // without doing anything should not be a problem.
        return;
    };

    let empty_block = code_block.lang_string == Default::default() && code_block.is_fenced;
    let is_ignore = code_block.lang_string.ignore != markdown::Ignore::None;

    // The span and whether it is precise or not.
    let (sp, precise_span) = match source_span_for_markdown_range(
        cx.tcx,
        dox,
        &code_block.range,
        &item.attrs.doc_strings,
    ) {
        Some(sp) => (sp, true),
        None => (item.attr_span(cx.tcx), false),
    };

    let msg = if buffer.has_errors {
        "could not parse code block as Rust code"
    } else {
        "Rust code block is empty"
    };

    // Finally build and emit the completed diagnostic.
    // All points of divergence have been handled earlier so this can be
    // done the same way whether the span is precise or not.
    let hir_id = cx.tcx.local_def_id_to_hir_id(local_id);
    cx.tcx.node_span_lint(crate::lint::INVALID_RUST_CODEBLOCKS, hir_id, sp, |lint| {
        lint.primary_message(msg);

        let explanation = if is_ignore {
            "`ignore` code blocks require valid Rust code for syntax highlighting; \
                    mark blocks that do not contain Rust code as text"
        } else {
            "mark blocks that do not contain Rust code as text"
        };

        if precise_span {
            if is_ignore {
                // giving an accurate suggestion is hard because `ignore` might not have come first in the list.
                // just give a `help` instead.
                lint.span_help(
                    sp.from_inner(InnerSpan::new(0, 3)),
                    format!("{explanation}: ```text"),
                );
            } else if empty_block {
                lint.span_suggestion(
                    sp.from_inner(InnerSpan::new(0, 3)).shrink_to_hi(),
                    explanation,
                    "text",
                    Applicability::MachineApplicable,
                );
            }
        } else if empty_block || is_ignore {
            lint.help(format!("{explanation}: ```text"));
        }

        // FIXME(#67563): Provide more context for these errors by displaying the spans inline.
        for message in buffer.messages.iter() {
            lint.note(message.clone());
        }
    });
}

#[derive(Default)]
struct Buffer {
    messages: Vec<String>,
    has_errors: bool,
}

struct BufferEmitter {
    buffer: Arc<Lock<Buffer>>,
    translator: Translator,
}

impl Emitter for BufferEmitter {
    fn emit_diagnostic(&mut self, diag: DiagInner, _registry: &Registry) {
        let mut buffer = self.buffer.borrow_mut();

        let fluent_args = to_fluent_args(diag.args.iter());
        let translated_main_message = self
            .translator
            .translate_message(&diag.messages[0].0, &fluent_args)
            .unwrap_or_else(|e| panic!("{e}"));

        buffer.messages.push(format!("error from rustc: {translated_main_message}"));
        if diag.is_error() {
            buffer.has_errors = true;
        }
    }

    fn source_map(&self) -> Option<&SourceMap> {
        None
    }

    fn translator(&self) -> &Translator {
        &self.translator
    }
}
