//! Standalone markdown rendering.
//!
//! For the (much more common) case of rendering markdown in doc-comments, see
//! [crate::html::markdown].
//!
//! This is used when [rendering a markdown file to an html file][docs], without processing
//! rust source code.
//!
//! [docs]: https://doc.rust-lang.org/stable/rustdoc/#using-standalone-markdown-files

use std::fmt::{self, Write as _};
use std::fs::{File, create_dir_all, read_to_string};
use std::io::prelude::*;
use std::path::Path;

use rustc_span::edition::Edition;

use crate::config::RenderOptions;
use crate::html::escape::Escape;
use crate::html::markdown;
use crate::html::markdown::{ErrorCodes, HeadingOffset, IdMap, Markdown, MarkdownWithToc};

/// Separate any lines at the start of the file that begin with `# ` or `%`.
fn extract_leading_metadata(s: &str) -> (Vec<&str>, &str) {
    let mut metadata = Vec::new();
    let mut count = 0;

    for line in s.lines() {
        if line.starts_with("# ") || line.starts_with('%') {
            // trim the whitespace after the symbol
            metadata.push(line[1..].trim_start());
            count += line.len() + 1;
        } else {
            return (metadata, &s[count..]);
        }
    }

    // if we're here, then all lines were metadata `# ` or `%` lines.
    (metadata, "")
}

/// Render `input` (e.g., "foo.md") into an HTML file in `output`
/// (e.g., output = "bar" => "bar/foo.html").
///
/// Requires session globals to be available, for symbol interning.
pub(crate) fn render_and_write<P: AsRef<Path>>(
    input: P,
    options: RenderOptions,
    edition: Edition,
) -> Result<(), String> {
    if let Err(e) = create_dir_all(&options.output) {
        return Err(format!("{output}: {e}", output = options.output.display()));
    }

    let input = input.as_ref();
    let mut output = options.output;
    output.push(input.file_name().unwrap());
    output.set_extension("html");

    let mut css = String::new();
    for name in &options.markdown_css {
        write!(css, r#"<link rel="stylesheet" href="{name}">"#)
            .expect("Writing to a String can't fail");
    }

    let input_str =
        read_to_string(input).map_err(|err| format!("{input}: {err}", input = input.display()))?;
    let playground_url = options.markdown_playground_url.or(options.playground_url);
    let playground = playground_url.map(|url| markdown::Playground { crate_name: None, url });

    let mut out =
        File::create(&output).map_err(|e| format!("{output}: {e}", output = output.display()))?;

    let (metadata, text) = extract_leading_metadata(&input_str);
    if metadata.is_empty() {
        return Err("invalid markdown file: no initial lines starting with `# ` or `%`".to_owned());
    }
    let title = metadata[0];

    let error_codes = ErrorCodes::from(options.unstable_features.is_nightly_build());
    let text = fmt::from_fn(|f| {
        if !options.markdown_no_toc {
            MarkdownWithToc {
                content: text,
                links: &[],
                ids: &mut IdMap::new(),
                error_codes,
                edition,
                playground: &playground,
            }
            .write_into(f)
        } else {
            Markdown {
                content: text,
                links: &[],
                ids: &mut IdMap::new(),
                error_codes,
                edition,
                playground: &playground,
                heading_offset: HeadingOffset::H1,
            }
            .write_into(f)
        }
    });

    let res = write!(
        &mut out,
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="rustdoc">
    <title>{title}</title>

    {css}
    {in_header}
</head>
<body class="rustdoc">
    <!--[if lte IE 8]>
    <div class="warning">
        This old browser is unsupported and will most likely display funky
        things.
    </div>
    <![endif]-->

    {before_content}
    <h1 class="title">{title}</h1>
    {text}
    {after_content}
</body>
</html>"#,
        title = Escape(title),
        in_header = options.external_html.in_header,
        before_content = options.external_html.before_content,
        after_content = options.external_html.after_content,
    );

    res.map_err(|e| format!("cannot write to `{output}`: {e}", output = output.display()))
}
