use rustc_data_structures::sync::{Lock, Lrc};
use rustc_errors::{emitter::Emitter, Applicability, Diagnostic, Handler};
use rustc_parse::parse_stream_from_source_str;
use rustc_session::parse::ParseSess;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{FileName, InnerSpan};

use crate::clean;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::{self, RustCodeBlock};
use crate::passes::{span_of_attrs, Pass};

crate const CHECK_CODE_BLOCK_SYNTAX: Pass = Pass {
    name: "check-code-block-syntax",
    run: check_code_block_syntax,
    description: "validates syntax inside Rust code blocks",
};

crate fn check_code_block_syntax(krate: clean::Crate, cx: &DocContext<'_>) -> clean::Crate {
    SyntaxChecker { cx }.fold_crate(krate)
}

struct SyntaxChecker<'a, 'tcx> {
    cx: &'a DocContext<'tcx>,
}

impl<'a, 'tcx> SyntaxChecker<'a, 'tcx> {
    fn check_rust_syntax(&self, item: &clean::Item, dox: &str, code_block: RustCodeBlock) {
        let buffer = Lrc::new(Lock::new(Buffer::default()));
        let emitter = BufferEmitter { buffer: Lrc::clone(&buffer) };

        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let handler = Handler::with_emitter(false, None, Box::new(emitter));
        let source = dox[code_block.code].to_owned();
        let sess = ParseSess::with_span_handler(handler, sm);

        let is_empty = rustc_driver::catch_fatal_errors(|| {
            parse_stream_from_source_str(
                FileName::Custom(String::from("doctest")),
                source,
                &sess,
                None,
            )
            .is_empty()
        })
        .unwrap_or(false);
        let buffer = buffer.borrow();

        if buffer.has_errors || is_empty {
            let mut diag = if let Some(sp) =
                super::source_span_for_markdown_range(self.cx, &dox, &code_block.range, &item.attrs)
            {
                let warning_message = if buffer.has_errors {
                    "could not parse code block as Rust code"
                } else {
                    "Rust code block is empty"
                };

                let mut diag = self.cx.sess().struct_span_warn(sp, warning_message);

                if code_block.syntax.is_none() && code_block.is_fenced {
                    let sp = sp.from_inner(InnerSpan::new(0, 3));
                    diag.span_suggestion(
                        sp,
                        "mark blocks that do not contain Rust code as text",
                        String::from("```text"),
                        Applicability::MachineApplicable,
                    );
                }

                diag
            } else {
                // We couldn't calculate the span of the markdown block that had the error, so our
                // diagnostics are going to be a bit lacking.
                let mut diag = self.cx.sess().struct_span_warn(
                    super::span_of_attrs(&item.attrs).unwrap_or(item.source.span()),
                    "doc comment contains an invalid Rust code block",
                );

                if code_block.syntax.is_none() && code_block.is_fenced {
                    diag.help("mark blocks that do not contain Rust code as text: ```text");
                }

                diag
            };

            // FIXME(#67563): Provide more context for these errors by displaying the spans inline.
            for message in buffer.messages.iter() {
                diag.note(&message);
            }

            diag.emit();
        }
    }
}

impl<'a, 'tcx> DocFolder for SyntaxChecker<'a, 'tcx> {
    fn fold_item(&mut self, item: clean::Item) -> Option<clean::Item> {
        if let Some(dox) = &item.attrs.collapsed_doc_value() {
            let sp = span_of_attrs(&item.attrs).unwrap_or(item.source.span());
            let extra = crate::html::markdown::ExtraInfo::new_did(&self.cx.tcx, item.def_id, sp);
            for code_block in markdown::rust_code_blocks(&dox, &extra) {
                self.check_rust_syntax(&item, &dox, code_block);
            }
        }

        Some(self.fold_item_recur(item))
    }
}

#[derive(Default)]
struct Buffer {
    messages: Vec<String>,
    has_errors: bool,
}

struct BufferEmitter {
    buffer: Lrc<Lock<Buffer>>,
}

impl Emitter for BufferEmitter {
    fn emit_diagnostic(&mut self, diag: &Diagnostic) {
        let mut buffer = self.buffer.borrow_mut();
        buffer.messages.push(format!("error from rustc: {}", diag.message[0].0));
        if diag.is_error() {
            buffer.has_errors = true;
        }
    }

    fn source_map(&self) -> Option<&Lrc<SourceMap>> {
        None
    }
}
