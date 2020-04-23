use rustc_ast::token;
use rustc_data_structures::sync::{Lock, Lrc};
use rustc_errors::{emitter::Emitter, Applicability, Diagnostic, Handler};
use rustc_parse::lexer::StringReader as Lexer;
use rustc_session::parse::ParseSess;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{FileName, InnerSpan};

use crate::clean;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::{self, RustCodeBlock};
use crate::passes::{span_of_attrs, Pass};

pub const CHECK_CODE_BLOCK_SYNTAX: Pass = Pass {
    name: "check-code-block-syntax",
    run: check_code_block_syntax,
    description: "validates syntax inside Rust code blocks",
};

pub fn check_code_block_syntax(krate: clean::Crate, cx: &DocContext<'_>) -> clean::Crate {
    SyntaxChecker { cx }.fold_crate(krate)
}

struct SyntaxChecker<'a, 'tcx> {
    cx: &'a DocContext<'tcx>,
}

impl<'a, 'tcx> SyntaxChecker<'a, 'tcx> {
    fn check_rust_syntax(&self, item: &clean::Item, dox: &str, code_block: RustCodeBlock) {
        let buffered_messages = Lrc::new(Lock::new(vec![]));

        let emitter = BufferEmitter { messages: Lrc::clone(&buffered_messages) };

        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let handler = Handler::with_emitter(false, None, Box::new(emitter));
        let sess = ParseSess::with_span_handler(handler, sm);
        let source_file = sess.source_map().new_source_file(
            FileName::Custom(String::from("doctest")),
            dox[code_block.code].to_owned(),
        );

        let validation_status = rustc_driver::catch_fatal_errors(|| {
            let mut has_syntax_errors = false;
            let mut only_whitespace = true;
            // even if there is a syntax error, we need to run the lexer over the whole file
            let mut lexer = Lexer::new(&sess, source_file, None);
            loop {
                match lexer.next_token().kind {
                    token::Eof => break,
                    token::Whitespace => (),
                    token::Unknown(..) => has_syntax_errors = true,
                    _ => only_whitespace = false,
                }
            }

            if has_syntax_errors {
                Some(CodeBlockInvalid::SyntaxError)
            } else if only_whitespace {
                Some(CodeBlockInvalid::Empty)
            } else {
                None
            }
        })
        .unwrap_or(Some(CodeBlockInvalid::SyntaxError));

        if let Some(code_block_invalid) = validation_status {
            let mut diag = if let Some(sp) =
                super::source_span_for_markdown_range(self.cx, &dox, &code_block.range, &item.attrs)
            {
                let warning_message = match code_block_invalid {
                    CodeBlockInvalid::SyntaxError => "could not parse code block as Rust code",
                    CodeBlockInvalid::Empty => "Rust code block is empty",
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
            for message in buffered_messages.borrow().iter() {
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

        self.fold_item_recur(item)
    }
}

struct BufferEmitter {
    messages: Lrc<Lock<Vec<String>>>,
}

impl Emitter for BufferEmitter {
    fn emit_diagnostic(&mut self, diag: &Diagnostic) {
        self.messages.borrow_mut().push(format!("error from rustc: {}", diag.message[0].0));
    }

    fn source_map(&self) -> Option<&Lrc<SourceMap>> {
        None
    }
}

enum CodeBlockInvalid {
    SyntaxError,
    Empty,
}
