// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;
use std::string::String;

use core;
use getopts;
use testing;

use externalfiles::ExternalHtml;

use html::escape::Escape;
use html::markdown;
use html::markdown::{Markdown, MarkdownWithToc, find_testable_code, reset_headers};
use test::Collector;

/// Separate any lines at the start of the file that begin with `%`.
fn extract_leading_metadata<'a>(s: &'a str) -> (Vec<&'a str>, &'a str) {
    let mut metadata = Vec::new();
    for line in s.lines() {
        if line.starts_with("%") {
            // remove %<whitespace>
            metadata.push(line.slice_from(1).trim_left())
        } else {
            let line_start_byte = s.subslice_offset(line);
            return (metadata, s.slice_from(line_start_byte));
        }
    }
    // if we're here, then all lines were metadata % lines.
    (metadata, "")
}

/// Render `input` (e.g. "foo.md") into an HTML file in `output`
/// (e.g. output = "bar" => "bar/foo.html").
pub fn render(input: &str, mut output: Path, matches: &getopts::Matches,
              external_html: &ExternalHtml, include_toc: bool) -> int {
    let input_p = Path::new(input);
    output.push(input_p.filestem().unwrap());
    output.set_extension("html");

    let mut css = String::new();
    for name in matches.opt_strs("markdown-css").iter() {
        let s = format!("<link rel=\"stylesheet\" type=\"text/css\" href=\"{}\">\n", name);
        css.push_str(s.as_slice())
    }

    let input_str = load_or_return!(input, 1, 2);
    let playground = matches.opt_str("markdown-playground-url");
    if playground.is_some() {
        markdown::playground_krate.replace(Some(None));
    }
    let playground = playground.unwrap_or("".to_string());

    let mut out = match io::File::create(&output) {
        Err(e) => {
            let _ = writeln!(&mut io::stderr(),
                             "error opening `{}` for writing: {}",
                             output.display(), e);
            return 4;
        }
        Ok(f) => f
    };

    let (metadata, text) = extract_leading_metadata(input_str.as_slice());
    if metadata.len() == 0 {
        let _ = writeln!(&mut io::stderr(),
                         "invalid markdown file: expecting initial line with `% ...TITLE...`");
        return 5;
    }
    let title = metadata[0].as_slice();

    reset_headers();

    let rendered = if include_toc {
        format!("{}", MarkdownWithToc(text))
    } else {
        format!("{}", Markdown(text))
    };

    let err = write!(
        &mut out,
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
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
    <script type="text/javascript">
        window.playgroundUrl = "{playground}";
    </script>
    {after_content}
</body>
</html>"#,
        title = Escape(title),
        css = css,
        in_header = external_html.in_header,
        before_content = external_html.before_content,
        text = rendered,
        after_content = external_html.after_content,
        playground = playground,
        );

    match err {
        Err(e) => {
            let _ = writeln!(&mut io::stderr(),
                             "error writing to `{}`: {}",
                             output.display(), e);
            6
        }
        Ok(_) => 0
    }
}

/// Run any tests/code examples in the markdown file `input`.
pub fn test(input: &str, libs: Vec<Path>, externs: core::Externs,
            mut test_args: Vec<String>) -> int {
    let input_str = load_or_return!(input, 1, 2);

    let mut collector = Collector::new(input.to_string(), libs, externs, true);
    find_testable_code(input_str.as_slice(), &mut collector);
    test_args.insert(0, "rustdoctest".to_string());
    testing::test_main(test_args.as_slice(), collector.tests);
    0
}
