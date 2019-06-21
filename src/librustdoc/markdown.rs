use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;
use std::cell::RefCell;

use errors;
use testing;
use syntax::edition::Edition;
use syntax::source_map::DUMMY_SP;
use syntax::feature_gate::UnstableFeatures;

use crate::externalfiles::{LoadStringError, load_string};
use crate::config::{Options, RenderOptions};
use crate::html::escape::Escape;
use crate::html::markdown;
use crate::html::markdown::{ErrorCodes, IdMap, Markdown, MarkdownWithToc, find_testable_code};
use crate::test::{TestOptions, Collector};

/// Separate any lines at the start of the file that begin with `# ` or `%`.
fn extract_leading_metadata(s: &str) -> (Vec<&str>, &str) {
    let mut metadata = Vec::new();
    let mut count = 0;

    for line in s.lines() {
        if line.starts_with("# ") || line.starts_with("%") {
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
pub fn render(
    input: PathBuf,
    options: RenderOptions,
    diag: &errors::Handler,
    edition: Edition
) -> i32 {
    let mut output = options.output;
    output.push(input.file_stem().unwrap());
    output.set_extension("html");

    let mut css = String::new();
    for name in &options.markdown_css {
        let s = format!("<link rel=\"stylesheet\" type=\"text/css\" href=\"{}\">\n", name);
        css.push_str(&s)
    }

    let input_str = match load_string(&input, diag) {
        Ok(s) => s,
        Err(LoadStringError::ReadFail) => return 1,
        Err(LoadStringError::BadUtf8) => return 2,
    };
    let playground_url = options.markdown_playground_url
                            .or(options.playground_url);
    if let Some(playground) = playground_url {
        markdown::PLAYGROUND.with(|s| { *s.borrow_mut() = Some((None, playground)); });
    }

    let mut out = match File::create(&output) {
        Err(e) => {
            diag.struct_err(&format!("{}: {}", output.display(), e)).emit();
            return 4;
        }
        Ok(f) => f,
    };

    let (metadata, text) = extract_leading_metadata(&input_str);
    if metadata.is_empty() {
        diag.struct_err("invalid markdown file: no initial lines starting with `# ` or `%`").emit();
        return 5;
    }
    let title = metadata[0];

    let mut ids = IdMap::new();
    let error_codes = ErrorCodes::from(UnstableFeatures::from_environment().is_nightly_build());
    let text = if !options.markdown_no_toc {
        MarkdownWithToc(text, RefCell::new(&mut ids), error_codes, edition).to_string()
    } else {
        Markdown(text, &[], RefCell::new(&mut ids), error_codes, edition).to_string()
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
        in_header = options.external_html.in_header,
        before_content = options.external_html.before_content,
        text = text,
        after_content = options.external_html.after_content,
    );

    match err {
        Err(e) => {
            diag.struct_err(&format!("cannot write to `{}`: {}", output.display(), e)).emit();
            6
        }
        Ok(_) => 0,
    }
}

/// Runs any tests/code examples in the markdown file `input`.
pub fn test(mut options: Options, diag: &errors::Handler) -> i32 {
    let input_str = match load_string(&options.input, diag) {
        Ok(s) => s,
        Err(LoadStringError::ReadFail) => return 1,
        Err(LoadStringError::BadUtf8) => return 2,
    };

    let mut opts = TestOptions::default();
    opts.no_crate_inject = true;
    opts.display_warnings = options.display_warnings;
    let mut collector = Collector::new(options.input.display().to_string(), options.cfgs,
                                       options.libs, options.codegen_options, options.externs,
                                       true, opts, options.maybe_sysroot, None,
                                       Some(options.input),
                                       options.linker, options.edition, options.persist_doctests);
    collector.set_position(DUMMY_SP);
    let codes = ErrorCodes::from(UnstableFeatures::from_environment().is_nightly_build());

    find_testable_code(&input_str, &mut collector, codes);

    options.test_args.insert(0, "rustdoctest".to_string());
    testing::test_main(&options.test_args, collector.tests,
                       testing::Options::new().display_output(options.display_warnings));
    0
}
