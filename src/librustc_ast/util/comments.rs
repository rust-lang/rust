use crate::ast::AttrStyle;
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, CharPos, FileName, Pos, Symbol};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CommentStyle {
    /// No code on either side of each line of the comment
    Isolated,
    /// Code exists to the left of the comment
    Trailing,
    /// Code before /* foo */ and after the comment
    Mixed,
    /// Just a manual blank line "\n\n", for layout
    BlankLine,
}

#[derive(Clone)]
pub struct Comment {
    pub style: CommentStyle,
    pub lines: Vec<String>,
    pub pos: BytePos,
}

/// For a full line comment string returns its doc comment style if it's a doc comment
/// and returns `None` if it's a regular comment.
pub fn line_doc_comment_style(line_comment: &str) -> Option<AttrStyle> {
    let line_comment = line_comment.as_bytes();
    assert!(line_comment.starts_with(b"//"));
    match line_comment.get(2) {
        // `//!` is an inner line doc comment.
        Some(b'!') => Some(AttrStyle::Inner),
        Some(b'/') => match line_comment.get(3) {
            // `////` (more than 3 slashes) is not considered a doc comment.
            Some(b'/') => None,
            // Otherwise `///` is an outer line doc comment.
            _ => Some(AttrStyle::Outer),
        },
        _ => None,
    }
}

/// For a full block comment string returns its doc comment style if it's a doc comment
/// and returns `None` if it's a regular comment.
pub fn block_doc_comment_style(block_comment: &str, terminated: bool) -> Option<AttrStyle> {
    let block_comment = block_comment.as_bytes();
    assert!(block_comment.starts_with(b"/*"));
    assert!(!terminated || block_comment.ends_with(b"*/"));
    match block_comment.get(2) {
        // `/*!` is an inner block doc comment.
        Some(b'!') => Some(AttrStyle::Inner),
        Some(b'*') => match block_comment.get(3) {
            // `/***` (more than 2 stars) is not considered a doc comment.
            Some(b'*') => None,
            // `/**/` is not considered a doc comment.
            Some(b'/') if block_comment.len() == 4 => None,
            // Otherwise `/**` is an outer block doc comment.
            _ => Some(AttrStyle::Outer),
        },
        _ => None,
    }
}

/// Makes a doc string more presentable to users.
/// Used by rustdoc and perhaps other tools, but not by rustc.
pub fn beautify_doc_string(data: Symbol) -> String {
    /// remove whitespace-only lines from the start/end of lines
    fn vertical_trim(lines: Vec<String>) -> Vec<String> {
        let mut i = 0;
        let mut j = lines.len();
        // first line of all-stars should be omitted
        if !lines.is_empty() && lines[0].chars().all(|c| c == '*') {
            i += 1;
        }

        while i < j && lines[i].trim().is_empty() {
            i += 1;
        }
        // like the first, a last line of all stars should be omitted
        if j > i && lines[j - 1].chars().skip(1).all(|c| c == '*') {
            j -= 1;
        }

        while j > i && lines[j - 1].trim().is_empty() {
            j -= 1;
        }

        lines[i..j].to_vec()
    }

    /// remove a "[ \t]*\*" block from each line, if possible
    fn horizontal_trim(lines: Vec<String>) -> Vec<String> {
        let mut i = usize::MAX;
        let mut can_trim = true;
        let mut first = true;

        for line in &lines {
            for (j, c) in line.chars().enumerate() {
                if j > i || !"* \t".contains(c) {
                    can_trim = false;
                    break;
                }
                if c == '*' {
                    if first {
                        i = j;
                        first = false;
                    } else if i != j {
                        can_trim = false;
                    }
                    break;
                }
            }
            if i >= line.len() {
                can_trim = false;
            }
            if !can_trim {
                break;
            }
        }

        if can_trim {
            lines.iter().map(|line| (&line[i + 1..line.len()]).to_string()).collect()
        } else {
            lines
        }
    }

    let data = data.as_str();
    if data.contains('\n') {
        let lines = data.lines().map(|s| s.to_string()).collect::<Vec<String>>();
        let lines = vertical_trim(lines);
        let lines = horizontal_trim(lines);
        lines.join("\n")
    } else {
        data.to_string()
    }
}

/// Returns `None` if the first `col` chars of `s` contain a non-whitespace char.
/// Otherwise returns `Some(k)` where `k` is first char offset after that leading
/// whitespace. Note that `k` may be outside bounds of `s`.
fn all_whitespace(s: &str, col: CharPos) -> Option<usize> {
    let mut idx = 0;
    for (i, ch) in s.char_indices().take(col.to_usize()) {
        if !ch.is_whitespace() {
            return None;
        }
        idx = i + ch.len_utf8();
    }
    Some(idx)
}

fn trim_whitespace_prefix(s: &str, col: CharPos) -> &str {
    let len = s.len();
    match all_whitespace(&s, col) {
        Some(col) => {
            if col < len {
                &s[col..]
            } else {
                ""
            }
        }
        None => s,
    }
}

fn split_block_comment_into_lines(text: &str, col: CharPos) -> Vec<String> {
    let mut res: Vec<String> = vec![];
    let mut lines = text.lines();
    // just push the first line
    res.extend(lines.next().map(|it| it.to_string()));
    // for other lines, strip common whitespace prefix
    for line in lines {
        res.push(trim_whitespace_prefix(line, col).to_string())
    }
    res
}

// it appears this function is called only from pprust... that's
// probably not a good thing.
pub fn gather_comments(sm: &SourceMap, path: FileName, src: String) -> Vec<Comment> {
    let sm = SourceMap::new(sm.path_mapping().clone());
    let source_file = sm.new_source_file(path, src);
    let text = (*source_file.src.as_ref().unwrap()).clone();

    let text: &str = text.as_str();
    let start_bpos = source_file.start_pos;
    let mut pos = 0;
    let mut comments: Vec<Comment> = Vec::new();
    let mut code_to_the_left = false;

    if let Some(shebang_len) = rustc_lexer::strip_shebang(text) {
        comments.push(Comment {
            style: CommentStyle::Isolated,
            lines: vec![text[..shebang_len].to_string()],
            pos: start_bpos,
        });
        pos += shebang_len;
    }

    for token in rustc_lexer::tokenize(&text[pos..]) {
        let token_text = &text[pos..pos + token.len];
        match token.kind {
            rustc_lexer::TokenKind::Whitespace => {
                if let Some(mut idx) = token_text.find('\n') {
                    code_to_the_left = false;
                    while let Some(next_newline) = &token_text[idx + 1..].find('\n') {
                        idx = idx + 1 + next_newline;
                        comments.push(Comment {
                            style: CommentStyle::BlankLine,
                            lines: vec![],
                            pos: start_bpos + BytePos((pos + idx) as u32),
                        });
                    }
                }
            }
            rustc_lexer::TokenKind::BlockComment { terminated } => {
                if block_doc_comment_style(token_text, terminated).is_none() {
                    let code_to_the_right = match text[pos + token.len..].chars().next() {
                        Some('\r' | '\n') => false,
                        _ => true,
                    };
                    let style = match (code_to_the_left, code_to_the_right) {
                        (_, true) => CommentStyle::Mixed,
                        (false, false) => CommentStyle::Isolated,
                        (true, false) => CommentStyle::Trailing,
                    };

                    // Count the number of chars since the start of the line by rescanning.
                    let pos_in_file = start_bpos + BytePos(pos as u32);
                    let line_begin_in_file = source_file.line_begin_pos(pos_in_file);
                    let line_begin_pos = (line_begin_in_file - start_bpos).to_usize();
                    let col = CharPos(text[line_begin_pos..pos].chars().count());

                    let lines = split_block_comment_into_lines(token_text, col);
                    comments.push(Comment { style, lines, pos: pos_in_file })
                }
            }
            rustc_lexer::TokenKind::LineComment => {
                if line_doc_comment_style(token_text).is_none() {
                    comments.push(Comment {
                        style: if code_to_the_left {
                            CommentStyle::Trailing
                        } else {
                            CommentStyle::Isolated
                        },
                        lines: vec![token_text.to_string()],
                        pos: start_bpos + BytePos(pos as u32),
                    })
                }
            }
            _ => {
                code_to_the_left = true;
            }
        }
        pos += token.len;
    }

    comments
}
