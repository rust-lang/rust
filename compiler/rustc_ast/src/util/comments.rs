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

/// Makes a doc string more presentable to users.
/// Used by rustdoc and perhaps other tools, but not by rustc.
pub fn beautify_doc_string(data: Symbol) -> Symbol {
    fn get_vertical_trim(lines: &[&str]) -> Option<(usize, usize)> {
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

        if i != 0 || j != lines.len() { Some((i, j)) } else { None }
    }

    fn get_horizontal_trim(lines: &[&str]) -> Option<usize> {
        let mut i = usize::MAX;
        let mut first = true;

        for line in lines {
            for (j, c) in line.chars().enumerate() {
                if j > i || !"* \t".contains(c) {
                    return None;
                }
                if c == '*' {
                    if first {
                        i = j;
                        first = false;
                    } else if i != j {
                        return None;
                    }
                    break;
                }
            }
            if i >= line.len() {
                return None;
            }
        }
        Some(i)
    }

    let data_s = data.as_str();
    if data_s.contains('\n') {
        let mut lines = data_s.lines().collect::<Vec<&str>>();
        let mut changes = false;
        let lines = if let Some((i, j)) = get_vertical_trim(&lines) {
            changes = true;
            // remove whitespace-only lines from the start/end of lines
            &mut lines[i..j]
        } else {
            &mut lines
        };
        if let Some(horizontal) = get_horizontal_trim(&lines) {
            changes = true;
            // remove a "[ \t]*\*" block from each line, if possible
            for line in lines.iter_mut() {
                *line = &line[horizontal + 1..];
            }
        }
        if changes {
            return Symbol::intern(&lines.join("\n"));
        }
    }
    data
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
            rustc_lexer::TokenKind::BlockComment { doc_style, .. } => {
                if doc_style.is_none() {
                    let code_to_the_right =
                        !matches!(text[pos + token.len..].chars().next(), Some('\r' | '\n'));
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
            rustc_lexer::TokenKind::LineComment { doc_style } => {
                if doc_style.is_none() {
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
