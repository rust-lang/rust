use anyhow::{anyhow, bail};
use std::{
    io::{BufRead, Lines},
    iter::Peekable,
};

const LISTING_DELIMITER: &'static str = "----";
const IMAGE_BLOCK_PREFIX: &'static str = "image::";
const VIDEO_BLOCK_PREFIX: &'static str = "video::";

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
                self.process_paragraph(0)?;
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
        if let Some(Ok(line)) = self.iter.next() {
            if let Some((level, title)) = get_title(&line) {
                if level == 1 {
                    self.write_title(level, title);
                    return Ok(());
                }
            }
        }
        bail!("document title not found")
    }

    fn process_list(&mut self) -> anyhow::Result<()> {
        while let Some(line) = self.iter.next() {
            let line = line?;
            if line.is_empty() {
                break;
            }

            if let Some(item) = get_list_item(&line) {
                self.write_list_item(item);
            } else if line == "+" {
                let line = self
                    .iter
                    .peek()
                    .ok_or_else(|| anyhow!("list continuation unexpectedly terminated"))?;
                let line = line.as_deref().map_err(|e| anyhow!("{e}"))?;
                if line.starts_with('[') {
                    self.write_line("", 0);
                    self.process_source_code_block(1)?;
                } else if line.starts_with(LISTING_DELIMITER) {
                    self.write_line("", 0);
                    self.process_listing_block(None, 1)?;
                } else if line.starts_with('.') {
                    self.write_line("", 0);
                    self.process_block_with_title(1)?;
                } else if line.starts_with(IMAGE_BLOCK_PREFIX) {
                    self.write_line("", 0);
                    self.process_image_block(None, 1)?;
                } else if line.starts_with(VIDEO_BLOCK_PREFIX) {
                    self.write_line("", 0);
                    self.process_video_block(None, 1)?;
                } else {
                    self.write_line("", 0);
                    self.process_paragraph(1)?;
                }
            } else {
                bail!("not a list block")
            }
        }

        Ok(())
    }

    fn process_source_code_block(&mut self, level: usize) -> anyhow::Result<()> {
        if let Some(Ok(line)) = self.iter.next() {
            if let Some(styles) = line.strip_prefix("[source").and_then(|s| s.strip_suffix(']')) {
                let mut styles = styles.split(',');
                if !styles.next().unwrap().is_empty() {
                    bail!("not a source code block");
                }
                let language = styles.next();
                return self.process_listing_block(language, level);
            }
        }
        bail!("not a source code block")
    }

    fn process_listing_block(&mut self, style: Option<&str>, level: usize) -> anyhow::Result<()> {
        if let Some(Ok(line)) = self.iter.next() {
            if line == LISTING_DELIMITER {
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
        if let Some(Ok(line)) = self.iter.next() {
            if let Some((url, attrs)) = parse_media_block(&line, IMAGE_BLOCK_PREFIX) {
                let alt = if let Some(stripped) =
                    attrs.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
                {
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
        }
        bail!("not a image block")
    }

    fn process_video_block(&mut self, caption: Option<&str>, level: usize) -> anyhow::Result<()> {
        if let Some(Ok(line)) = self.iter.next() {
            if let Some((url, attrs)) = parse_media_block(&line, VIDEO_BLOCK_PREFIX) {
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
        }
        bail!("not a video block")
    }

    fn process_paragraph(&mut self, level: usize) -> anyhow::Result<()> {
        while let Some(line) = self.iter.peek() {
            let line = line.as_deref().map_err(|e| anyhow!("{e}"))?;
            if line.is_empty() || (level > 0 && line == "+") {
                break;
            }

            self.write_indent(level);
            let line = self.iter.next().unwrap()?;
            if line.ends_with('+') {
                let line = &line[..(line.len() - 1)];
                self.output.push_str(line);
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

    fn write_title(&mut self, level: usize, title: &str) {
        for _ in 0..level {
            self.output.push('#');
        }
        self.output.push(' ');
        self.output.push_str(title);
        self.output.push('\n');
    }

    fn write_list_item(&mut self, item: &str) {
        self.output.push_str("- ");
        self.output.push_str(item);
        self.output.push('\n');
    }

    fn write_caption_line(&mut self, caption: &str, level: usize) {
        self.write_indent(level);
        self.output.push('_');
        self.output.push_str(caption);
        self.output.push_str("_\\\n");
    }

    fn write_indent(&mut self, level: usize) {
        for _ in 0..level {
            self.output.push_str("  ");
        }
    }

    fn write_line(&mut self, line: &str, level: usize) {
        self.write_indent(level);
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
    const MARKER: char = '=';
    let mut iter = line.chars();
    if iter.next()? != MARKER {
        return None;
    }
    let mut count = 1;
    loop {
        match iter.next() {
            Some(MARKER) => {
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

fn get_list_item(line: &str) -> Option<&str> {
    const MARKER: &'static str = "* ";
    if line.starts_with(MARKER) {
        let item = &line[MARKER.len()..];
        Some(item)
    } else {
        None
    }
}

fn parse_media_block<'a>(line: &'a str, prefix: &str) -> Option<(&'a str, &'a str)> {
    if let Some(line) = line.strip_prefix(prefix) {
        if let Some((url, rest)) = line.split_once('[') {
            if let Some(attrs) = rest.strip_suffix(']') {
                return Some((url, attrs));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asciidoc_to_markdown_conversion() {
        let input = "\
= Changelog #256
:sectanchors:
:page-layout: post

Hello!

Commit: commit:0123456789abcdef0123456789abcdef01234567[] +
Release: release:2022-01-01[]

== New Features

* pr:1111[] foo bar baz
* pr:2222[] foo bar baz
+
image::https://example.com/animation.gif[]
+
image::https://example.com/animation.gif[\"alt text\"]
+
video::https://example.com/movie.mp4[options=loop]
+
video::https://example.com/movie.mp4[options=\"autoplay,loop\"]
+
.Image
image::https://example.com/animation.gif[]
+
.Video
video::https://example.com/movie.mp4[options=loop]
+
[source,bash]
----
rustup update nightly
----
+
----
This is a plain listing.
----
+
paragraph
paragraph

== Fixes

* pr:3333[] foo bar baz
* pr:4444[] foo bar baz

== Internal Improvements

* pr:5555[] foo bar baz
* pr:6666[] foo bar baz

The highlight of the month is probably pr:1111[].

[source,bash]
----
rustup update nightly
----

[source]
----
rustup update nightly
----

----
This is a plain listing.
----
";
        let expected = "\
# Changelog #256

Hello!

Commit: commit:0123456789abcdef0123456789abcdef01234567[] \\
Release: release:2022-01-01[]

## New Features

- pr:1111[] foo bar baz
- pr:2222[] foo bar baz

  ![](https://example.com/animation.gif)

  ![alt text](https://example.com/animation.gif)

  <video src=\"https://example.com/movie.mp4\" controls loop>Your browser does not support the video tag.</video>

  <video src=\"https://example.com/movie.mp4\" autoplay controls loop>Your browser does not support the video tag.</video>

  _Image_\\
  ![](https://example.com/animation.gif)

  _Video_\\
  <video src=\"https://example.com/movie.mp4\" controls loop>Your browser does not support the video tag.</video>

  ```bash
  rustup update nightly
  ```

  ```
  This is a plain listing.
  ```

  paragraph
  paragraph

## Fixes

- pr:3333[] foo bar baz
- pr:4444[] foo bar baz

## Internal Improvements

- pr:5555[] foo bar baz
- pr:6666[] foo bar baz

The highlight of the month is probably pr:1111[].

```bash
rustup update nightly
```

```
rustup update nightly
```

```
This is a plain listing.
```
";
        let actual = convert_asciidoc_to_markdown(std::io::Cursor::new(input)).unwrap();

        assert_eq!(actual, expected);
    }
}
