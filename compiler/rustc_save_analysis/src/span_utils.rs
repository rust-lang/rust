use crate::generated_code;
use rustc_data_structures::sync::Lrc;
use rustc_lexer::{tokenize, TokenKind};
use rustc_session::Session;
use rustc_span::*;

#[derive(Clone)]
pub struct SpanUtils<'a> {
    pub sess: &'a Session,
}

impl<'a> SpanUtils<'a> {
    pub fn new(sess: &'a Session) -> SpanUtils<'a> {
        SpanUtils { sess }
    }

    pub fn make_filename_string(&self, file: &SourceFile) -> String {
        match &file.name {
            FileName::Real(name) if !file.name_was_remapped => {
                let path = name.local_path();
                if path.is_absolute() {
                    self.sess
                        .source_map()
                        .path_mapping()
                        .map_prefix(path.into())
                        .0
                        .display()
                        .to_string()
                } else {
                    self.sess.working_dir.0.join(&path).display().to_string()
                }
            }
            // If the file name is already remapped, we assume the user
            // configured it the way they wanted to, so use that directly
            filename => filename.to_string(),
        }
    }

    pub fn snippet(&self, span: Span) -> String {
        match self.sess.source_map().span_to_snippet(span) {
            Ok(s) => s,
            Err(_) => String::new(),
        }
    }

    /// Finds the span of `*` token withing the larger `span`.
    pub fn sub_span_of_star(&self, mut span: Span) -> Option<Span> {
        let begin = self.sess.source_map().lookup_byte_offset(span.lo());
        let end = self.sess.source_map().lookup_byte_offset(span.hi());
        // Make the range zero-length if the span is invalid.
        if begin.sf.start_pos != end.sf.start_pos {
            span = span.shrink_to_lo();
        }

        let sf = Lrc::clone(&begin.sf);

        self.sess.source_map().ensure_source_file_source_present(Lrc::clone(&sf));
        let src =
            sf.src.clone().or_else(|| sf.external_src.borrow().get_source().map(Lrc::clone))?;
        let to_index = |pos: BytePos| -> usize { (pos - sf.start_pos).0 as usize };
        let text = &src[to_index(span.lo())..to_index(span.hi())];
        let start_pos = {
            let mut pos = 0;
            tokenize(text)
                .map(|token| {
                    let start = pos;
                    pos += token.len;
                    (start, token)
                })
                .find(|(_pos, token)| token.kind == TokenKind::Star)?
                .0
        };
        let lo = span.lo() + BytePos(start_pos as u32);
        let hi = lo + BytePos(1);
        Some(span.with_lo(lo).with_hi(hi))
    }

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
        !self.sess.source_map().lookup_char_pos(span.lo()).file.is_real_file()
    }
}

macro_rules! filter {
    ($util: expr, $parent: expr) => {
        if $util.filter_generated($parent) {
            return None;
        }
    };
}
