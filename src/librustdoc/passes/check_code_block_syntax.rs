use rustc_data_structures::sync::{Lock, Lrc};
use rustc_errors::{emitter::Emitter, Applicability, Diagnostic, Handler};
use rustc_middle::lint::LintDiagnosticBuilder;
use rustc_parse::parse_stream_from_source_str;
use rustc_session::parse::ParseSess;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{hygiene::AstPass, ExpnData, ExpnKind, FileName, InnerSpan, DUMMY_SP};

use crate::clean;
use crate::core::DocContext;
use crate::html::markdown::{self, RustCodeBlock};
use crate::passes::Pass;
use crate::visit::DocVisitor;

crate const CHECK_CODE_BLOCK_SYNTAX: Pass = Pass {
    name: "check-code-block-syntax",
    run: check_code_block_syntax,
    description: "validates syntax inside Rust code blocks",
};

crate fn check_code_block_syntax(krate: clean::Crate, cx: &mut DocContext<'_>) -> clean::Crate {
    SyntaxChecker { cx }.visit_crate(&krate);
    krate
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

        let edition = code_block.lang_string.edition.unwrap_or_else(|| self.cx.tcx.sess.edition());
        let expn_data = ExpnData::default(
            ExpnKind::AstPass(AstPass::TestHarness),
            DUMMY_SP,
            edition,
            None,
            None,
        );
        let span = DUMMY_SP.fresh_expansion(expn_data, self.cx.tcx.create_stable_hashing_context());

        let is_empty = rustc_driver::catch_fatal_errors(|| {
            parse_stream_from_source_str(
                FileName::Custom(String::from("doctest")),
                source,
                &sess,
                Some(span),
            )
            .is_empty()
        })
        .unwrap_or(false);
        let buffer = buffer.borrow();

        if !buffer.has_errors && !is_empty {
            // No errors in a non-empty program.
            return;
        }

        let local_id = match item.def_id.as_def_id().and_then(|x| x.as_local()) {
            Some(id) => id,
            // We don't need to check the syntax for other crates so returning
            // without doing anything should not be a problem.
            None => return,
        };

        let hir_id = self.cx.tcx.hir().local_def_id_to_hir_id(local_id);
        let empty_block = code_block.lang_string == Default::default() && code_block.is_fenced;
        let is_ignore = code_block.lang_string.ignore != markdown::Ignore::None;

        // The span and whether it is precise or not.
        let (sp, precise_span) = match super::source_span_for_markdown_range(
            self.cx.tcx,
            dox,
            &code_block.range,
            &item.attrs,
        ) {
            Some(sp) => (sp, true),
            None => (item.attr_span(self.cx.tcx), false),
        };

        // lambda that will use the lint to start a new diagnostic and add
        // a suggestion to it when needed.
        let diag_builder = |lint: LintDiagnosticBuilder<'_>| {
            let explanation = if is_ignore {
                "`ignore` code blocks require valid Rust code for syntax highlighting; \
                    mark blocks that do not contain Rust code as text"
            } else {
                "mark blocks that do not contain Rust code as text"
            };
            let msg = if buffer.has_errors {
                "could not parse code block as Rust code"
            } else {
                "Rust code block is empty"
            };
            let mut diag = lint.build(msg);

            if precise_span {
                if is_ignore {
                    // giving an accurate suggestion is hard because `ignore` might not have come first in the list.
                    // just give a `help` instead.
                    diag.span_help(
                        sp.from_inner(InnerSpan::new(0, 3)),
                        &format!("{}: ```text", explanation),
                    );
                } else if empty_block {
                    diag.span_suggestion(
                        sp.from_inner(InnerSpan::new(0, 3)).shrink_to_hi(),
                        explanation,
                        String::from("text"),
                        Applicability::MachineApplicable,
                    );
                }
            } else if empty_block || is_ignore {
                diag.help(&format!("{}: ```text", explanation));
            }

            // FIXME(#67563): Provide more context for these errors by displaying the spans inline.
            for message in buffer.messages.iter() {
                diag.note(message);
            }

            diag.emit();
        };

        // Finally build and emit the completed diagnostic.
        // All points of divergence have been handled earlier so this can be
        // done the same way whether the span is precise or not.
        self.cx.tcx.struct_span_lint_hir(
            crate::lint::INVALID_RUST_CODEBLOCKS,
            hir_id,
            sp,
            diag_builder,
        );
    }
}

impl<'a, 'tcx> DocVisitor for SyntaxChecker<'a, 'tcx> {
    fn visit_item(&mut self, item: &clean::Item) {
        if let Some(dox) = &item.attrs.collapsed_doc_value() {
            let sp = item.attr_span(self.cx.tcx);
            let extra = crate::html::markdown::ExtraInfo::new_did(
                self.cx.tcx,
                item.def_id.expect_def_id(),
                sp,
            );
            for code_block in markdown::rust_code_blocks(dox, &extra) {
                self.check_rust_syntax(&item, dox, code_block);
            }
        }

        self.visit_item_recur(item)
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
