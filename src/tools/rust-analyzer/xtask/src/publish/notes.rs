use anyhow::{anyhow, bail};
use std::{
    borrow::Cow,
    io::{BufRead, Lines},
    iter::Peekable,
};

const LISTING_DELIMITER: &str = "----";
const IMAGE_BLOCK_PREFIX: &str = "image::";
const VIDEO_BLOCK_PREFIX: &str = "video::";

struct Converter<'a, 'b, R: BufRead> {
    iter: &'a mut Peekable<Lines<R>>,
    output: &'b mut String,
}

impl<'a, 'b, R: BufRead> Converter<'a, 'b, R> {
    fn new(iter: &'a mut Peekable<Lines<R>>, output: &'b mut String) -> Self {
        Self { iter, output }
    }

    fn process(&mut self) -> anyhow::Result<()> {
        self.process_document_header()?;
        self.skip_blank_lines()?;
        self.output.push('\n');

        loop {
            let line = self.iter.peek().unwrap().as_deref().map_err(|e| anyhow!("{e}"))?;
            if get_title(line).is_some() {
                let line = self.iter.next().unwrap().unwrap();
                let (level, title) = get_title(&line).unwrap();
                self.write_title(level, title);
            } else if get_list_item(line).is_some() {
                self.process_list()?;
            } else if line.starts_with('[') {
                self.process_source_code_block(0)?;
            } else if line.starts_with(LISTING_DELIMITER) {
                self.process_listing_block(None, 0)?;
            } else if line.starts_with('.') {
                self.process_block_with_title(0)?;
            } else if line.starts_with(IMAGE_BLOCK_PREFIX) {
                self.process_image_block(None, 0)?;
            } else if line.starts_with(VIDEO_BLOCK_PREFIX) {
                self.process_video_block(None, 0)?;
            } else {
                self.process_paragraph(0, |line| line.is_empty())?;
            }

            self.skip_blank_lines()?;
            if self.iter.peek().is_none() {
                break;
            }
            self.output.push('\n');
        }
        Ok(())
    }

    fn process_document_header(&mut self) -> anyhow::Result<()> {
        self.process_document_title()?;

        while let Some(line) = self.iter.next() {
            let line = line?;
            if line.is_empty() {
                break;
            }
            if !line.starts_with(':') {
                self.write_line(&line, 0)
            }
        }

        Ok(())
    }

    fn process_document_title(&mut self) -> anyhow::Result<()> {
        if let Some(Ok(line)) = self.iter.next()
            && let Some((level, title)) = get_title(&line)
        {
            let title = process_inline_macros(title)?;
            if level == 1 {
                self.write_title(level, &title);
                return Ok(());
            }
        }
        bail!("document title not found")
    }

    fn process_list(&mut self) -> anyhow::Result<()> {
        let mut nesting = ListNesting::default();
        while let Some(line) = self.iter.peek() {
            let line = line.as_deref().map_err(|e| anyhow!("{e}"))?;

            if get_list_item(line).is_some() {
                let line = self.iter.next().unwrap()?;
                let line = process_inline_macros(&line)?;
                let (marker, item) = get_list_item(&line).unwrap();
                nesting.set_current(marker);
                self.write_list_item(item, &nesting);
                self.process_paragraph(nesting.indent(), |line| {
                    line.is_empty() || get_list_item(line).is_some() || line == "+"
                })?;
            } else if line == "+" {
                let _ = self.iter.next().unwrap()?;
                let line = self
                    .iter
                    .peek()
                    .ok_or_else(|| anyhow!("list continuation unexpectedly terminated"))?;
                let line = line.as_deref().map_err(|e| anyhow!("{e}"))?;

                let indent = nesting.indent();
                if line.starts_with('[') {
                    self.write_line("", 0);
                    self.process_source_code_block(indent)?;
                } else if line.starts_with(LISTING_DELIMITER) {
                    self.write_line("", 0);
                    self.process_listing_block(None, indent)?;
                } else if line.starts_with('.') {
                    self.write_line("", 0);
                    self.process_block_with_title(indent)?;
                } else if line.starts_with(IMAGE_BLOCK_PREFIX) {
                    self.write_line("", 0);
                    self.process_image_block(None, indent)?;
                } else if line.starts_with(VIDEO_BLOCK_PREFIX) {
                    self.write_line("", 0);
                    self.process_video_block(None, indent)?;
                } else {
                    self.write_line("", 0);
                    let current = nesting.current().unwrap();
                    self.process_paragraph(indent, |line| {
                        line.is_empty()
                            || get_list_item(line).filter(|(m, _)| m == current).is_some()
                            || line == "+"
                    })?;
                }
            } else {
                break;
            }
            self.skip_blank_lines()?;
        }

        Ok(())
    }

    fn process_source_code_block(&mut self, level: usize) -> anyhow::Result<()> {
        if let Some(Ok(line)) = self.iter.next()
            && let Some(styles) = line.strip_prefix("[source").and_then(|s| s.strip_suffix(']'))
        {
            let mut styles = styles.split(',');
            if !styles.next().unwrap().is_empty() {
                bail!("not a source code block");
            }
            let language = styles.next();
            return self.process_listing_block(language, level);
        }
        bail!("not a source code block")
    }

    fn process_listing_block(&mut self, style: Option<&str>, level: usize) -> anyhow::Result<()> {
        if let Some(Ok(line)) = self.iter.next()
            && line == LISTING_DELIMITER
        {
            self.write_indent(level);
            self.output.push_str("```");
            if let Some(style) = style {
                self.output.push_str(style);
            }
            self.output.push('\n');
            while let Some(line) = self.iter.next() {
                let line = line?;
                if line == LISTING_DELIMITER {
                    self.write_line("```", level);
                    return Ok(());
                } else {
                    self.write_line(&line, level);
                }
            }
            bail!("listing block is not terminated")
        }
        bail!("not a listing block")
    }

    fn process_block_with_title(&mut self, level: usize) -> anyhow::Result<()> {
        if let Some(Ok(line)) = self.iter.next() {
            let title =
                line.strip_prefix('.').ok_or_else(|| anyhow!("extraction of the title failed"))?;

            let line = self
                .iter
                .peek()
                .ok_or_else(|| anyhow!("target block for the title is not found"))?;
            let line = line.as_deref().map_err(|e| anyhow!("{e}"))?;
            if line.starts_with(IMAGE_BLOCK_PREFIX) {
                return self.process_image_block(Some(title), level);
            } else if line.starts_with(VIDEO_BLOCK_PREFIX) {
                return self.process_video_block(Some(title), level);
            } else {
                bail!("title for that block type is not supported");
            }
        }
        bail!("not a title")
    }

    fn process_image_block(&mut self, caption: Option<&str>, level: usize) -> anyhow::Result<()> {
        if let Some(Ok(line)) = self.iter.next()
            && let Some((url, attrs)) = parse_media_block(&line, IMAGE_BLOCK_PREFIX)
        {
            let alt =
                if let Some(stripped) = attrs.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                    stripped
                } else {
                    attrs
                };
            if let Some(caption) = caption {
                self.write_caption_line(caption, level);
            }
            self.write_indent(level);
            self.output.push_str("![");
            self.output.push_str(alt);
            self.output.push_str("](");
            self.output.push_str(url);
            self.output.push_str(")\n");
            return Ok(());
        }
        bail!("not a image block")
    }

    fn process_video_block(&mut self, caption: Option<&str>, level: usize) -> anyhow::Result<()> {
        if let Some(Ok(line)) = self.iter.next()
            && let Some((url, attrs)) = parse_media_block(&line, VIDEO_BLOCK_PREFIX)
        {
            let html_attrs = match attrs {
                "options=loop" => "controls loop",
                r#"options="autoplay,loop""# => "autoplay controls loop",
                _ => bail!("unsupported video syntax"),
            };
            if let Some(caption) = caption {
                self.write_caption_line(caption, level);
            }
            self.write_indent(level);
            self.output.push_str(r#"<video src=""#);
            self.output.push_str(url);
            self.output.push_str(r#"" "#);
            self.output.push_str(html_attrs);
            self.output.push_str(">Your browser does not support the video tag.</video>\n");
            return Ok(());
        }
        bail!("not a video block")
    }

    fn process_paragraph<P>(&mut self, level: usize, predicate: P) -> anyhow::Result<()>
    where
        P: Fn(&str) -> bool,
    {
        while let Some(line) = self.iter.peek() {
            let line = line.as_deref().map_err(|e| anyhow!("{e}"))?;
            if predicate(line) {
                break;
            }

            self.write_indent(level);
            let line = self.iter.next().unwrap()?;
            let line = line.trim_start();
            let line = process_inline_macros(line)?;
            if let Some(stripped) = line.strip_suffix('+') {
                self.output.push_str(stripped);
                self.output.push('\\');
            } else {
                self.output.push_str(&line);
            }
            self.output.push('\n');
        }

        Ok(())
    }

    fn skip_blank_lines(&mut self) -> anyhow::Result<()> {
        while let Some(line) = self.iter.peek() {
            if !line.as_deref().unwrap().is_empty() {
                break;
            }
            self.iter.next().unwrap()?;
        }
        Ok(())
    }

    fn write_title(&mut self, indent: usize, title: &str) {
        for _ in 0..indent {
            self.output.push('#');
        }
        self.output.push(' ');
        self.output.push_str(title);
        self.output.push('\n');
    }

    fn write_list_item(&mut self, item: &str, nesting: &ListNesting) {
        let (marker, indent) = nesting.marker();
        self.write_indent(indent);
        self.output.push_str(marker);
        self.output.push_str(item);
        self.output.push('\n');
    }

    fn write_caption_line(&mut self, caption: &str, indent: usize) {
        self.write_indent(indent);
        self.output.push('_');
        self.output.push_str(caption);
        self.output.push_str("_\\\n");
    }

    fn write_indent(&mut self, indent: usize) {
        for _ in 0..indent {
            self.output.push(' ');
        }
    }

    fn write_line(&mut self, line: &str, indent: usize) {
        self.write_indent(indent);
        self.output.push_str(line);
        self.output.push('\n');
    }
}

pub(crate) fn convert_asciidoc_to_markdown<R>(input: R) -> anyhow::Result<String>
where
    R: BufRead,
{
    let mut output = String::new();
    let mut iter = input.lines().peekable();

    let mut converter = Converter::new(&mut iter, &mut output);
    converter.process()?;

    Ok(output)
}

fn get_title(line: &str) -> Option<(usize, &str)> {
    strip_prefix_symbol(line, '=')
}

fn get_list_item(line: &str) -> Option<(ListMarker, &str)> {
    const HYPHEN_MARKER: &str = "- ";
    if let Some(text) = line.strip_prefix(HYPHEN_MARKER) {
        Some((ListMarker::Hyphen, text))
    } else if let Some((count, text)) = strip_prefix_symbol(line, '*') {
        Some((ListMarker::Asterisk(count), text))
    } else if let Some((count, text)) = strip_prefix_symbol(line, '.') {
        Some((ListMarker::Dot(count), text))
    } else {
        None
    }
}

fn strip_prefix_symbol(line: &str, symbol: char) -> Option<(usize, &str)> {
    let mut iter = line.chars();
    if iter.next()? != symbol {
        return None;
    }
    let mut count = 1;
    loop {
        match iter.next() {
            Some(ch) if ch == symbol => {
                count += 1;
            }
            Some(' ') => {
                break;
            }
            _ => return None,
        }
    }
    Some((count, iter.as_str()))
}

fn parse_media_block<'a>(line: &'a str, prefix: &str) -> Option<(&'a str, &'a str)> {
    if let Some(line) = line.strip_prefix(prefix)
        && let Some((url, rest)) = line.split_once('[')
        && let Some(attrs) = rest.strip_suffix(']')
    {
        return Some((url, attrs));
    }
    None
}

#[derive(Debug)]
struct ListNesting(Vec<ListMarker>);

impl ListNesting {
    fn current(&mut self) -> Option<&ListMarker> {
        self.0.last()
    }

    fn set_current(&mut self, marker: ListMarker) {
        let Self(markers) = self;
        if let Some(index) = markers.iter().position(|m| *m == marker) {
            markers.truncate(index + 1);
        } else {
            markers.push(marker);
        }
    }

    fn indent(&self) -> usize {
        self.0.iter().map(|m| m.in_markdown().len()).sum()
    }

    fn marker(&self) -> (&str, usize) {
        let Self(markers) = self;
        let indent = markers.iter().take(markers.len() - 1).map(|m| m.in_markdown().len()).sum();
        let marker = match markers.last() {
            None => "",
            Some(marker) => marker.in_markdown(),
        };
        (marker, indent)
    }
}

impl Default for ListNesting {
    fn default() -> Self {
        Self(Vec::<ListMarker>::with_capacity(6))
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ListMarker {
    Asterisk(usize),
    Hyphen,
    Dot(usize),
}

impl ListMarker {
    fn in_markdown(&self) -> &str {
        match self {
            ListMarker::Asterisk(_) => "- ",
            ListMarker::Hyphen => "- ",
            ListMarker::Dot(_) => "1. ",
        }
    }
}

fn process_inline_macros(line: &str) -> anyhow::Result<Cow<'_, str>> {
    let mut chars = line.char_indices();
    loop {
        let (start, end, a_macro) = match get_next_line_component(&mut chars) {
            Component::None => break,
            Component::Text => continue,
            Component::Macro(s, e, m) => (s, e, m),
        };
        let mut src = line.chars();
        let mut processed = String::new();
        for _ in 0..start {
            processed.push(src.next().unwrap());
        }
        processed.push_str(a_macro.process()?.as_str());
        for _ in start..end {
            let _ = src.next().unwrap();
        }
        let mut pos = end;

        loop {
            let (start, end, a_macro) = match get_next_line_component(&mut chars) {
                Component::None => break,
                Component::Text => continue,
                Component::Macro(s, e, m) => (s, e, m),
            };
            for _ in pos..start {
                processed.push(src.next().unwrap());
            }
            processed.push_str(a_macro.process()?.as_str());
            for _ in start..end {
                let _ = src.next().unwrap();
            }
            pos = end;
        }
        for ch in src {
            processed.push(ch);
        }
        return Ok(Cow::Owned(processed));
    }
    Ok(Cow::Borrowed(line))
}

fn get_next_line_component(chars: &mut std::str::CharIndices<'_>) -> Component {
    let (start, mut macro_name) = match chars.next() {
        None => return Component::None,
        Some((_, ch)) if ch == ' ' || !ch.is_ascii() => return Component::Text,
        Some((pos, ch)) => (pos, String::from(ch)),
    };
    loop {
        match chars.next() {
            None => return Component::None,
            Some((_, ch)) if ch == ' ' || !ch.is_ascii() => return Component::Text,
            Some((_, ':')) => break,
            Some((_, ch)) => macro_name.push(ch),
        }
    }

    let mut macro_target = String::new();
    loop {
        match chars.next() {
            None => return Component::None,
            Some((_, ' ')) => return Component::Text,
            Some((_, '[')) => break,
            Some((_, ch)) => macro_target.push(ch),
        }
    }

    let mut attr_value = String::new();
    let end = loop {
        match chars.next() {
            None => return Component::None,
            Some((pos, ']')) => break pos + 1,
            Some((_, ch)) => attr_value.push(ch),
        }
    };

    Component::Macro(start, end, Macro::new(macro_name, macro_target, attr_value))
}

enum Component {
    None,
    Text,
    Macro(usize, usize, Macro),
}

struct Macro {
    name: String,
    target: String,
    attrs: String,
}

impl Macro {
    fn new(name: String, target: String, attrs: String) -> Self {
        Self { name, target, attrs }
    }

    fn process(&self) -> anyhow::Result<String> {
        let name = &self.name;
        let text = match name.as_str() {
            "https" => {
                let url = &self.target;
                let anchor_text = &self.attrs;
                format!("[{anchor_text}](https:{url})")
            }
            "image" => {
                let url = &self.target;
                let alt = &self.attrs;
                format!("![{alt}]({url})")
            }
            "kbd" => {
                let keys = self.attrs.split('+').map(|k| Cow::Owned(format!("<kbd>{k}</kbd>")));
                keys.collect::<Vec<_>>().join("+")
            }
            "pr" => {
                let pr = &self.target;
                let url = format!("https://github.com/rust-lang/rust-analyzer/pull/{pr}");
                format!("[`#{pr}`]({url})")
            }
            "commit" => {
                let hash = &self.target;
                let short = &hash[0..7];
                let url = format!("https://github.com/rust-lang/rust-analyzer/commit/{hash}");
                format!("[`{short}`]({url})")
            }
            "release" => {
                let date = &self.target;
                let url = format!("https://github.com/rust-lang/rust-analyzer/releases/{date}");
                format!("[`{date}`]({url})")
            }
            _ => bail!("macro not supported: {name}"),
        };
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::read_to_string;

    #[test]
    fn test_asciidoc_to_markdown_conversion() {
        let input = read_to_string("test_data/input.adoc").unwrap();
        let expected = read_to_string("test_data/expected.md").unwrap();
        let actual = convert_asciidoc_to_markdown(std::io::Cursor::new(&input)).unwrap();

        assert_eq!(actual, expected);
    }

    macro_rules! test_inline_macro_processing {
        ($((
            $name:ident,
            $input:expr,
            $expected:expr
        ),)*) => ($(
            #[test]
            fn $name() {
                let input = $input;
                let actual = process_inline_macros(&input).unwrap();
                let expected = $expected;
                assert_eq!(actual, expected)
            }
        )*);
    }

    test_inline_macro_processing! {
        (inline_macro_processing_for_empty_line, "", ""),
        (inline_macro_processing_for_line_with_no_macro, "foo bar", "foo bar"),
        (
            inline_macro_processing_for_macro_in_line_start,
            "kbd::[Ctrl+T] foo",
            "<kbd>Ctrl</kbd>+<kbd>T</kbd> foo"
        ),
        (
            inline_macro_processing_for_macro_in_line_end,
            "foo kbd::[Ctrl+T]",
            "foo <kbd>Ctrl</kbd>+<kbd>T</kbd>"
        ),
        (
            inline_macro_processing_for_macro_in_the_middle_of_line,
            "foo kbd::[Ctrl+T] foo",
            "foo <kbd>Ctrl</kbd>+<kbd>T</kbd> foo"
        ),
        (
            inline_macro_processing_for_several_macros,
            "foo kbd::[Ctrl+T] foo kbd::[Enter] foo",
            "foo <kbd>Ctrl</kbd>+<kbd>T</kbd> foo <kbd>Enter</kbd> foo"
        ),
        (
            inline_macro_processing_for_several_macros_without_text_in_between,
            "foo kbd::[Ctrl+T]kbd::[Enter] foo",
            "foo <kbd>Ctrl</kbd>+<kbd>T</kbd><kbd>Enter</kbd> foo"
        ),
    }
}
