use rustc::session::Session;

use crate::generated_code;

use syntax::parse::lexer::{self, StringReader};
use syntax::parse::token::{self, TokenKind};
use syntax_pos::*;

#[derive(Clone)]
pub struct SpanUtils<'a> {
    pub sess: &'a Session,
}

impl<'a> SpanUtils<'a> {
    pub fn new(sess: &'a Session) -> SpanUtils<'a> {
        SpanUtils {
            sess,
        }
    }

    pub fn make_filename_string(&self, file: &SourceFile) -> String {
        match &file.name {
            FileName::Real(path) if !file.name_was_remapped => {
                if path.is_absolute() {
                    self.sess.source_map().path_mapping()
                        .map_prefix(path.clone()).0
                        .display()
                        .to_string()
                } else {
                    self.sess.working_dir.0
                        .join(&path)
                        .display()
                        .to_string()
                }
            },
            // If the file name is already remapped, we assume the user
            // configured it the way they wanted to, so use that directly
            filename => filename.to_string()
        }
    }

    pub fn snippet(&self, span: Span) -> String {
        match self.sess.source_map().span_to_snippet(span) {
            Ok(s) => s,
            Err(_) => String::new(),
        }
    }

    pub fn retokenise_span(&self, span: Span) -> StringReader<'a> {
        lexer::StringReader::retokenize(&self.sess.parse_sess, span)
    }

    pub fn sub_span_of_token(&self, span: Span, tok: TokenKind) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        loop {
            let next = toks.real_token();
            if next == token::Eof {
                return None;
            }
            if next == tok {
                return Some(next.span);
            }
        }
    }

    // // Return the name for a macro definition (identifier after first `!`)
    // pub fn span_for_macro_def_name(&self, span: Span) -> Option<Span> {
    //     let mut toks = self.retokenise_span(span);
    //     loop {
    //         let ts = toks.real_token();
    //         if ts == token::Eof {
    //             return None;
    //         }
    //         if ts == token::Not {
    //             let ts = toks.real_token();
    //             if ts.kind.is_ident() {
    //                 return Some(ts.sp);
    //             } else {
    //                 return None;
    //             }
    //         }
    //     }
    // }

    // // Return the name for a macro use (identifier before first `!`).
    // pub fn span_for_macro_use_name(&self, span:Span) -> Option<Span> {
    //     let mut toks = self.retokenise_span(span);
    //     let mut prev = toks.real_token();
    //     loop {
    //         if prev == token::Eof {
    //             return None;
    //         }
    //         let ts = toks.real_token();
    //         if ts == token::Not {
    //             if prev.kind.is_ident() {
    //                 return Some(prev.sp);
    //             } else {
    //                 return None;
    //             }
    //         }
    //         prev = ts;
    //     }
    // }

    /// Return true if the span is generated code, and
    /// it is not a subspan of the root callsite.
    ///
    /// Used to filter out spans of minimal value,
    /// such as references to macro internal variables.
    pub fn filter_generated(&self, span: Span) -> bool {
        if generated_code(span) {
            return true;
        }

        //If the span comes from a fake source_file, filter it.
        !self.sess
            .source_map()
            .lookup_char_pos(span.lo())
            .file
            .is_real_file()
    }
}

macro_rules! filter {
    ($util: expr, $parent: expr) => {
        if $util.filter_generated($parent) {
            return None;
        }
    };
}
