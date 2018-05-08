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
use std::path::{PathBuf, Path};

use errors;
use getopts;
use testing;
use rustc::session::search_paths::SearchPaths;
use rustc::session::config::{Externs, CodegenOptions};
use syntax::codemap::DUMMY_SP;
use syntax::edition::Edition;

use externalfiles::{ExternalHtml, LoadStringError, load_string};

use html::render::reset_ids;
use html::escape::Escape;
use html::markdown;
use html::markdown::{Markdown, MarkdownWithToc, find_testable_code};
use test::{TestOptions, Collector};

/// Separate any lines at the start of the file that begin with `# ` or `%`.
fn extract_leading_metadata<'a>(s: &'a str) -> (Vec<&'a str>, &'a str) {
    let mut metadata = Vec::new();
    let mut count = 0;

    for line in s.lines() {
        if line.starts_with("# ") || line.starts_with("%") {
            // trim the whitespace after the symbol
            metadata.push(line[1..].trim_left());
            count += line.len() + 1;
        } else {
            return (metadata, &s[count..]);
        }
    }

    // if we're here, then all lines were metadata `# ` or `%` lines.
    (metadata, "")
}

/// Render `input` (e.g. "foo.md") into an HTML file in `output`
/// (e.g. output = "bar" => "bar/foo.html").
pub fn render(input: &Path, mut output: PathBuf, matches: &getopts::Matches,
              external_html: &ExternalHtml, include_toc: bool, diag: &errors::Handler) -> isize {
    output.push(input.file_stem().unwrap());
    output.set_extension("html");

    let mut css = String::new();
    for name in &matches.opt_strs("markdown-css") {
        let s = format!("<link rel=\"stylesheet\" type=\"text/css\" href=\"{}\">\n", name);
        css.push_str(&s)
    }

    let input_str = match load_string(input, diag) {
        Ok(s) => s,
        Err(LoadStringError::ReadFail) => return 1,
        Err(LoadStringError::BadUtf8) => return 2,
    };
    if let Some(playground) = matches.opt_str("markdown-playground-url").or(
                              matches.opt_str("playground-url")) {
        markdown::PLAYGROUND.with(|s| { *s.borrow_mut() = Some((None, playground)); });
    }

    let mut out = match File::create(&output) {
        Err(e) => {
            diag.struct_err(&format!("{}: {}", output.display(), e)).emit();
            return 4;
        }
        Ok(f) => f
    };

    let (metadata, text) = extract_leading_metadata(&input_str);
    if metadata.is_empty() {
        diag.struct_err("invalid markdown file: no initial lines starting with `# ` or `%`").emit();
        return 5;
    }
    let title = metadata[0];

    reset_ids(false);

    let text = if include_toc {
        format!("{}", MarkdownWithToc(text))
    } else {
        format!("{}", Markdown(text, &[]))
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
        css = css,
        in_header = external_html.in_header,
        before_content = external_html.before_content,
        text = text,
        after_content = external_html.after_content,
    );

    match err {
        Err(e) => {
            diag.struct_err(&format!("cannot write to `{}`: {}", output.display(), e)).emit();
            6
        }
        Ok(_) => 0,
    }
}

/// Run any tests/code examples in the markdown file `input`.
pub fn test(input: &str, cfgs: Vec<String>, libs: SearchPaths, externs: Externs,
            mut test_args: Vec<String>, maybe_sysroot: Option<PathBuf>,
            display_warnings: bool, linker: Option<PathBuf>, edition: Edition,
            cg: CodegenOptions, diag: &errors::Handler) -> isize {
    let input_str = match load_string(input, diag) {
        Ok(s) => s,
        Err(LoadStringError::ReadFail) => return 1,
        Err(LoadStringError::BadUtf8) => return 2,
    };

    let mut opts = TestOptions::default();
    opts.no_crate_inject = true;
    opts.display_warnings = display_warnings;
    let mut collector = Collector::new(input.to_owned(), cfgs, libs, cg, externs,
                                       true, opts, maybe_sysroot, None,
                                       Some(PathBuf::from(input)),
                                       linker, edition);
    find_testable_code(&input_str, &mut collector, DUMMY_SP, None);
    test_args.insert(0, "rustdoctest".to_string());
    testing::test_main(&test_args, collector.tests,
                       testing::Options::new().display_output(display_warnings));
    0
}
