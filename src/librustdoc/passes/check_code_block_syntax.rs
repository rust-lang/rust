use errors::Applicability;
use syntax::parse::lexer::{TokenAndSpan, StringReader as Lexer};
use syntax::parse::{ParseSess, token};
use syntax::source_map::FilePathMapping;
use syntax_pos::FileName;

use clean;
use core::DocContext;
use fold::DocFolder;
use html::markdown::{self, RustCodeBlock};
use passes::Pass;

pub const CHECK_CODE_BLOCK_SYNTAX: Pass =
    Pass::early("check-code-block-syntax", check_code_block_syntax,
                "validates syntax inside Rust code blocks");

pub fn check_code_block_syntax(krate: clean::Crate, cx: &DocContext) -> clean::Crate {
    SyntaxChecker { cx }.fold_crate(krate)
}

struct SyntaxChecker<'a, 'tcx: 'a, 'rcx: 'a> {
    cx: &'a DocContext<'a, 'tcx, 'rcx>,
}

impl<'a, 'tcx, 'rcx> SyntaxChecker<'a, 'tcx, 'rcx> {
    fn check_rust_syntax(&self, item: &clean::Item, dox: &str, code_block: RustCodeBlock) {
        let sess = ParseSess::new(FilePathMapping::empty());
        let source_file = sess.source_map().new_source_file(
            FileName::Custom(String::from("doctest")),
            dox[code_block.code].to_owned(),
        );

        let errors = Lexer::new_or_buffered_errs(&sess, source_file, None).and_then(|mut lexer| {
            while let Ok(TokenAndSpan { tok, .. }) = lexer.try_next_token() {
                if tok == token::Eof {
                    break;
                }
            }

            let errors = lexer.buffer_fatal_errors();

            if !errors.is_empty() {
                Err(errors)
            } else {
                Ok(())
            }
        });

        if let Err(errors) = errors {
            let mut diag = if let Some(sp) =
                super::source_span_for_markdown_range(self.cx, &dox, &code_block.range, &item.attrs)
            {
                let mut diag = self
                    .cx
                    .sess()
                    .struct_span_warn(sp, "could not parse code block as Rust code");

                for mut err in errors {
                    diag.note(&format!("error from rustc: {}", err.message()));
                    err.cancel();
                }

                if code_block.syntax.is_none() && code_block.is_fenced {
                    let sp = sp.from_inner_byte_pos(0, 3);
                    diag.span_suggestion_with_applicability(
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
                    super::span_of_attrs(&item.attrs),
                    "doc comment contains an invalid Rust code block",
                );

                for mut err in errors {
                    // Don't bother reporting the error, because we can't show where it happened.
                    err.cancel();
                }

                if code_block.syntax.is_none() && code_block.is_fenced {
                    diag.help("mark blocks that do not contain Rust code as text: ```text");
                }

                diag
            };

            diag.emit();
        }
    }
}

impl<'a, 'tcx, 'rcx> DocFolder for SyntaxChecker<'a, 'tcx, 'rcx> {
    fn fold_item(&mut self, item: clean::Item) -> Option<clean::Item> {
        if let Some(dox) = &item.attrs.collapsed_doc_value() {
            for code_block in markdown::rust_code_blocks(&dox) {
                self.check_rust_syntax(&item, &dox, code_block);
            }
        }

        self.fold_item_recur(item)
    }
}
