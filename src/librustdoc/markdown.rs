// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::default::Default;
use std::fs::File;
use std::io::prelude::*;
use std::io;
use std::path::{PathBuf, Path};

use core;
use getopts;
use testing;
use rustc::session::search_paths::SearchPaths;

use externalfiles::ExternalHtml;

use html::escape::Escape;
use html::markdown;
use html::markdown::{Markdown, MarkdownWithToc, find_testable_code, reset_headers};
use test::{TestOptions, Collector};

/// Separate any lines at the start of the file that begin with `%`.
fn extract_leading_metadata<'a>(s: &'a str) -> (Vec<&'a str>, &'a str) {
    let mut metadata = Vec::new();
    let mut count = 0;
    for line in s.lines() {
        if line.starts_with("%") {
            // remove %<whitespace>
            metadata.push(line[1..].trim_left());
            count += line.len() + 1;
        } else {
            return (metadata, &s[count..]);
        }
    }
    // if we're here, then all lines were metadata % lines.
    (metadata, "")
}

/// Render `input` (e.g. "foo.md") into an HTML file in `output`
/// (e.g. output = "bar" => "bar/foo.html").
pub fn render(input: &str, mut output: PathBuf, matches: &getopts::Matches,
              external_html: &ExternalHtml, include_toc: bool) -> isize {
    let input_p = Path::new(input);
    output.push(input_p.file_stem().unwrap());
    output.set_extension("html");

    let mut css = String::new();
    for name in &matches.opt_strs("markdown-css") {
        let s = format!("<link rel=\"stylesheet\" type=\"text/css\" href=\"{}\">\n", name);
        css.push_str(&s)
    }

    let input_str = load_or_return!(input, 1, 2);
    let playground = matches.opt_str("markdown-playground-url");
    if playground.is_some() {
        markdown::PLAYGROUND_KRATE.with(|s| { *s.borrow_mut() = Some(None); });
    }
    let playground = playground.unwrap_or("".to_string());

    let mut out = match File::create(&output) {
        Err(e) => {
            let _ = writeln!(&mut io::stderr(),
                             "error opening `{}` for writing: {}",
                             output.display(), e);
            return 4;
        }
        Ok(f) => f
    };

    let (metadata, text) = extract_leading_metadata(&input_str);
    if metadata.is_empty() {
        let _ = writeln!(&mut io::stderr(),
                         "invalid markdown file: expecting initial line with `% ...TITLE...`");
        return 5;
    }
    let title = metadata[0];

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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="rustdoc">
    <title>{title}</title>

    {css}
    {in_header}
</head>
<body class="rustdoc">
    <div id="style-changer">
        <div class="light" onclick="change_style(this);"></div>
        <div class="dark" onclick="change_style(this);"></div>
    </div>
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
pub fn test(input: &str, libs: SearchPaths, externs: core::Externs,
            mut test_args: Vec<String>) -> isize {
    let input_str = load_or_return!(input, 1, 2);

    let mut opts = TestOptions::default();
    opts.no_crate_inject = true;
    let mut collector = Collector::new(input.to_string(), libs, externs,
                                       true, opts);
    find_testable_code(&input_str, &mut collector);
    test_args.insert(0, "rustdoctest".to_string());
    testing::test_main(&test_args, collector.tests);
    0
}
