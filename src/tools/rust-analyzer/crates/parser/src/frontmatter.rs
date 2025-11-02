// Copied from https://github.com/rust-lang/cargo/blob/367fd9f213750cd40317803dd0a5a3ce3f0c676d/src/cargo/util/frontmatter.rs
#![expect(dead_code)] // avoid editing
#![expect(unreachable_pub)] // avoid editing
#![expect(clippy::useless_format)] // avoid editing

type Span = std::ops::Range<usize>;

#[derive(Debug)]
pub struct ScriptSource<'s> {
    /// The full file
    raw: &'s str,
    /// The `#!/usr/bin/env cargo` line, if present
    shebang: Option<Span>,
    /// The code fence opener (`---`)
    open: Option<Span>,
    /// Trailing text after `ScriptSource::open` that identifies the meaning of
    /// `ScriptSource::frontmatter`
    info: Option<Span>,
    /// The lines between `ScriptSource::open` and `ScriptSource::close`
    frontmatter: Option<Span>,
    /// The code fence closer (`---`)
    close: Option<Span>,
    /// All content after the frontmatter and shebang
    content: Span,
}

impl<'s> ScriptSource<'s> {
    pub fn parse(raw: &'s str) -> Result<Self, FrontmatterError> {
        use winnow::stream::FindSlice as _;
        use winnow::stream::Location as _;
        use winnow::stream::Offset as _;
        use winnow::stream::Stream as _;

        let content_end = raw.len();
        let mut source = Self {
            raw,
            shebang: None,
            open: None,
            info: None,
            frontmatter: None,
            close: None,
            content: 0..content_end,
        };

        let mut input = winnow::stream::LocatingSlice::new(raw);

        if let Some(shebang_end) = strip_shebang(input.as_ref()) {
            let shebang_start = input.current_token_start();
            let _ = input.next_slice(shebang_end);
            let shebang_end = input.current_token_start();
            source.shebang = Some(shebang_start..shebang_end);
            source.content = shebang_end..content_end;
        }

        // Whitespace may precede a frontmatter but must end with a newline
        if let Some(nl_end) = strip_ws_lines(input.as_ref()) {
            let _ = input.next_slice(nl_end);
        }

        // Opens with a line that starts with 3 or more `-` followed by an optional identifier
        const FENCE_CHAR: char = '-';
        let fence_length = input
            .as_ref()
            .char_indices()
            .find_map(|(i, c)| (c != FENCE_CHAR).then_some(i))
            .unwrap_or_else(|| input.eof_offset());
        let open_start = input.current_token_start();
        let fence_pattern = input.next_slice(fence_length);
        let open_end = input.current_token_start();
        match fence_length {
            0 => {
                return Ok(source);
            }
            1 | 2 => {
                // either not a frontmatter or invalid frontmatter opening
                return Err(FrontmatterError::new(
                    format!(
                        "found {fence_length} `{FENCE_CHAR}` in rust frontmatter, expected at least 3"
                    ),
                    raw.len()..raw.len(),
                ).push_visible_span(open_start..open_end));
            }
            _ => {}
        }
        source.open = Some(open_start..open_end);
        let Some(info_nl) = input.find_slice("\n") else {
            return Err(FrontmatterError::new(
                format!("unclosed frontmatter; expected `{fence_pattern}`"),
                raw.len()..raw.len(),
            )
            .push_visible_span(open_start..open_end));
        };
        let info = input.next_slice(info_nl.start);
        let info = info.strip_suffix('\r').unwrap_or(info); // already excludes `\n`
        let info = info.trim_matches(is_horizontal_whitespace);
        if !info.is_empty() {
            let info_start = info.offset_from(&raw);
            let info_end = info_start + info.len();
            source.info = Some(info_start..info_end);
        }

        // Ends with a line that starts with a matching number of `-` only followed by whitespace
        let nl_fence_pattern = format!("\n{fence_pattern}");
        let Some(frontmatter_nl) = input.find_slice(nl_fence_pattern.as_str()) else {
            for len in (2..(nl_fence_pattern.len() - 1)).rev() {
                let Some(frontmatter_nl) = input.find_slice(&nl_fence_pattern[0..len]) else {
                    continue;
                };
                let _ = input.next_slice(frontmatter_nl.start + 1);
                let close_start = input.current_token_start();
                let _ = input.next_slice(len);
                let close_end = input.current_token_start();
                let fewer_dashes = fence_length - len;
                return Err(FrontmatterError::new(
                    format!(
                        "closing code fence has {fewer_dashes} less `-` than the opening fence"
                    ),
                    close_start..close_end,
                )
                .push_visible_span(open_start..open_end));
            }
            return Err(FrontmatterError::new(
                format!("unclosed frontmatter; expected `{fence_pattern}`"),
                raw.len()..raw.len(),
            )
            .push_visible_span(open_start..open_end));
        };
        let frontmatter_start = input.current_token_start() + 1; // skip nl from infostring
        let _ = input.next_slice(frontmatter_nl.start + 1);
        let frontmatter_end = input.current_token_start();
        source.frontmatter = Some(frontmatter_start..frontmatter_end);
        let close_start = input.current_token_start();
        let _ = input.next_slice(fence_length);
        let close_end = input.current_token_start();
        source.close = Some(close_start..close_end);

        let nl = input.find_slice("\n");
        let after_closing_fence =
            input.next_slice(nl.map(|span| span.end).unwrap_or_else(|| input.eof_offset()));
        let content_start = input.current_token_start();
        let extra_dashes = after_closing_fence.chars().take_while(|b| *b == FENCE_CHAR).count();
        if 0 < extra_dashes {
            let extra_start = close_end;
            let extra_end = extra_start + extra_dashes;
            return Err(FrontmatterError::new(
                format!("closing code fence has {extra_dashes} more `-` than the opening fence"),
                extra_start..extra_end,
            )
            .push_visible_span(open_start..open_end));
        } else {
            let after_closing_fence = strip_newline(after_closing_fence);
            let after_closing_fence = after_closing_fence.trim_matches(is_horizontal_whitespace);
            if !after_closing_fence.is_empty() {
                // extra characters beyond the original fence pattern
                let after_start = after_closing_fence.offset_from(&raw);
                let after_end = after_start + after_closing_fence.len();
                return Err(FrontmatterError::new(
                    format!("unexpected characters after frontmatter close"),
                    after_start..after_end,
                )
                .push_visible_span(open_start..open_end));
            }
        }

        source.content = content_start..content_end;

        if let Some(nl_end) = strip_ws_lines(input.as_ref()) {
            let _ = input.next_slice(nl_end);
        }
        let fence_length = input
            .as_ref()
            .char_indices()
            .find_map(|(i, c)| (c != FENCE_CHAR).then_some(i))
            .unwrap_or_else(|| input.eof_offset());
        if 0 < fence_length {
            let fence_start = input.current_token_start();
            let fence_end = fence_start + fence_length;
            return Err(FrontmatterError::new(
                format!("only one frontmatter is supported"),
                fence_start..fence_end,
            )
            .push_visible_span(open_start..open_end)
            .push_visible_span(close_start..close_end));
        }

        Ok(source)
    }

    pub fn shebang(&self) -> Option<&'s str> {
        self.shebang.clone().map(|span| &self.raw[span])
    }

    pub fn shebang_span(&self) -> Option<Span> {
        self.shebang.clone()
    }

    pub fn open_span(&self) -> Option<Span> {
        self.open.clone()
    }

    pub fn info(&self) -> Option<&'s str> {
        self.info.clone().map(|span| &self.raw[span])
    }

    pub fn info_span(&self) -> Option<Span> {
        self.info.clone()
    }

    pub fn frontmatter(&self) -> Option<&'s str> {
        self.frontmatter.clone().map(|span| &self.raw[span])
    }

    pub fn frontmatter_span(&self) -> Option<Span> {
        self.frontmatter.clone()
    }

    pub fn close_span(&self) -> Option<Span> {
        self.close.clone()
    }

    pub fn content(&self) -> &'s str {
        &self.raw[self.content.clone()]
    }

    pub fn content_span(&self) -> Span {
        self.content.clone()
    }
}

/// Returns the index after the shebang line, if present
pub fn strip_shebang(input: &str) -> Option<usize> {
    // See rust-lang/rust's compiler/rustc_lexer/src/lib.rs's `strip_shebang`
    // Shebang must start with `#!` literally, without any preceding whitespace.
    // For simplicity we consider any line starting with `#!` a shebang,
    // regardless of restrictions put on shebangs by specific platforms.
    if let Some(rest) = input.strip_prefix("#!") {
        // Ok, this is a shebang but if the next non-whitespace token is `[`,
        // then it may be valid Rust code, so consider it Rust code.
        //
        // NOTE: rustc considers line and block comments to be whitespace but to avoid
        // any more awareness of Rust grammar, we are excluding it.
        if !rest.trim_start().starts_with('[') {
            // No other choice than to consider this a shebang.
            let newline_end = input.find('\n').map(|pos| pos + 1).unwrap_or(input.len());
            return Some(newline_end);
        }
    }
    None
}

/// Returns the index after any lines with only whitespace, if present
pub fn strip_ws_lines(input: &str) -> Option<usize> {
    let ws_end = input.find(|c| !is_whitespace(c)).unwrap_or(input.len());
    if ws_end == 0 {
        return None;
    }

    let nl_start = input[0..ws_end].rfind('\n')?;
    let nl_end = nl_start + 1;
    Some(nl_end)
}

/// True if `c` is considered a whitespace according to Rust language definition.
/// See [Rust language reference](https://doc.rust-lang.org/reference/whitespace.html)
/// for definitions of these classes.
fn is_whitespace(c: char) -> bool {
    // This is Pattern_White_Space.
    //
    // Note that this set is stable (ie, it doesn't change with different
    // Unicode versions), so it's ok to just hard-code the values.

    matches!(
        c,
        // End-of-line characters
        | '\u{000A}' // line feed (\n)
        | '\u{000B}' // vertical tab
        | '\u{000C}' // form feed
        | '\u{000D}' // carriage return (\r)
        | '\u{0085}' // next line (from latin1)
        | '\u{2028}' // LINE SEPARATOR
        | '\u{2029}' // PARAGRAPH SEPARATOR

        // `Default_Ignorable_Code_Point` characters
        | '\u{200E}' // LEFT-TO-RIGHT MARK
        | '\u{200F}' // RIGHT-TO-LEFT MARK

        // Horizontal space characters
        | '\u{0009}'   // tab (\t)
        | '\u{0020}' // space
    )
}

/// True if `c` is considered horizontal whitespace according to Rust language definition.
fn is_horizontal_whitespace(c: char) -> bool {
    // This is Pattern_White_Space.
    //
    // Note that this set is stable (ie, it doesn't change with different
    // Unicode versions), so it's ok to just hard-code the values.

    matches!(
        c,
        // Horizontal space characters
        '\u{0009}'   // tab (\t)
        | '\u{0020}' // space
    )
}

fn strip_newline(text: &str) -> &str {
    text.strip_suffix("\r\n").or_else(|| text.strip_suffix('\n')).unwrap_or(text)
}

#[derive(Debug)]
pub struct FrontmatterError {
    message: String,
    primary_span: Span,
    visible_spans: Vec<Span>,
}

impl FrontmatterError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self { message: message.into(), primary_span: span, visible_spans: Vec::new() }
    }

    pub fn push_visible_span(mut self, span: Span) -> Self {
        self.visible_spans.push(span);
        self
    }

    pub fn message(&self) -> &str {
        self.message.as_str()
    }

    pub fn primary_span(&self) -> Span {
        self.primary_span.clone()
    }

    pub fn visible_spans(&self) -> &[Span] {
        &self.visible_spans
    }
}

impl std::fmt::Display for FrontmatterError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(fmt)
    }
}

impl std::error::Error for FrontmatterError {}
