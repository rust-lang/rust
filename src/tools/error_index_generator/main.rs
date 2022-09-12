#![feature(rustc_private)]

extern crate rustc_driver;

// We use the function we generate from `register_diagnostics!`.
use crate::error_codes::error_codes;

use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use std::str::FromStr;

use mdbook::book::{parse_summary, BookItem, Chapter};
use mdbook::{Config, MDBook};

macro_rules! register_diagnostics {
    ($($error_code:ident: $message:expr,)+ ; $($undocumented:ident,)* ) => {
        pub fn error_codes() -> Vec<(&'static str, Option<&'static str>)> {
            let mut errors: Vec<(&str, Option<&str>)> = vec![
                $((stringify!($error_code), Some($message)),)+
                $((stringify!($undocumented), None),)+
            ];
            errors.sort();
            errors
        }
    }
}

#[path = "../../../compiler/rustc_error_codes/src/error_codes.rs"]
mod error_codes;

enum OutputFormat {
    HTML,
    Markdown,
    Unknown(String),
}

impl OutputFormat {
    fn from(format: &str) -> OutputFormat {
        match &*format.to_lowercase() {
            "html" => OutputFormat::HTML,
            "markdown" => OutputFormat::Markdown,
            s => OutputFormat::Unknown(s.to_owned()),
        }
    }
}

/// Output an HTML page for the errors in `err_map` to `output_path`.
fn render_markdown(output_path: &Path) -> Result<(), Box<dyn Error>> {
    let mut output_file = File::create(output_path)?;

    write!(output_file, "# Rust Compiler Error Index\n")?;

    for (err_code, description) in error_codes().iter() {
        match description {
            Some(ref desc) => write!(output_file, "## {}\n{}\n", err_code, desc)?,
            None => {}
        }
    }

    Ok(())
}

// By default, mdbook doesn't consider code blocks as Rust ones contrary to rustdoc so we have
// to manually add `rust` attribute whenever needed.
fn add_rust_attribute_on_codeblock(explanation: &str) -> String {
    // Very hacky way to add the rust attribute on all code blocks.
    let mut skip = true;
    explanation.split("\n```").fold(String::new(), |mut acc, part| {
        if !acc.is_empty() {
            acc.push_str("\n```");
        }
        if !skip {
            if let Some(attrs) = part.split('\n').next() {
                if !attrs.contains("rust")
                    && (attrs.is_empty()
                        || attrs.contains("compile_fail")
                        || attrs.contains("ignore")
                        || attrs.contains("edition"))
                {
                    if !attrs.is_empty() {
                        acc.push_str("rust,");
                    } else {
                        acc.push_str("rust");
                    }
                }
            }
        }
        skip = !skip;
        acc.push_str(part);
        acc
    })
}

fn render_html(output_path: &Path) -> Result<(), Box<dyn Error>> {
    let mut introduction = format!(
        "<script src='redirect.js'></script>
# Rust error codes index

This page lists all the error codes emitted by the Rust compiler.

"
    );

    let err_codes = error_codes();
    let mut chapters = Vec::with_capacity(err_codes.len());

    for (err_code, explanation) in err_codes.iter() {
        if let Some(explanation) = explanation {
            introduction.push_str(&format!(" * [{0}](./{0}.html)\n", err_code));

            let content = add_rust_attribute_on_codeblock(explanation);
            chapters.push(BookItem::Chapter(Chapter {
                name: err_code.to_string(),
                content: format!("# Error code {}\n\n{}\n", err_code, content),
                number: None,
                sub_items: Vec::new(),
                // We generate it into the `error_codes` folder.
                path: Some(PathBuf::from(&format!("{}.html", err_code))),
                source_path: None,
                parent_names: Vec::new(),
            }));
        } else {
            introduction.push_str(&format!(" * {}\n", err_code));
        }
    }

    let mut config = Config::from_str(include_str!("book_config.toml"))?;
    config.build.build_dir = output_path.join("error_codes").to_path_buf();
    let mut book = MDBook::load_with_config_and_summary(
        env!("CARGO_MANIFEST_DIR"),
        config,
        parse_summary("")?,
    )?;
    let chapter = Chapter {
        name: "Rust error codes index".to_owned(),
        content: introduction,
        number: None,
        sub_items: chapters,
        // Very important: this file is named as `error-index.html` and not `index.html`!
        path: Some(PathBuf::from("error-index.html")),
        source_path: None,
        parent_names: Vec::new(),
    };
    book.book.sections.push(BookItem::Chapter(chapter));
    book.build()?;

    // We can't put this content into another file, otherwise `mbdbook` will also put it into the
    // output directory, making a duplicate.
    fs::write(
        output_path.join("error-index.html"),
        r#"<!DOCTYPE html>
<html>
    <head>
        <title>Rust error codes index - Error codes index</title>
        <meta content="text/html; charset=utf-8" http-equiv="Content-Type">
        <meta name="description" content="Book listing all Rust error codes">
        <script src="error_codes/redirect.js"></script>
    </head>
    <body>
        <div>If you are not automatically redirected to the error code index, please <a id="index-link" href="./error_codes/error-index.html">here</a>.
        <script>document.getElementById("index-link").click()</script>
    </body>
</html>"#,
    )?;

    // No need for a 404 file, it's already handled by the server.
    fs::remove_file(output_path.join("error_codes/404.html"))?;

    Ok(())
}

fn main_with_result(format: OutputFormat, dst: &Path) -> Result<(), Box<dyn Error>> {
    match format {
        OutputFormat::Unknown(s) => panic!("Unknown output format: {}", s),
        OutputFormat::HTML => render_html(dst),
        OutputFormat::Markdown => render_markdown(dst),
    }
}

fn parse_args() -> (OutputFormat, PathBuf) {
    let mut args = env::args().skip(1);
    let format = args.next();
    let dst = args.next();
    let format = format.map(|a| OutputFormat::from(&a)).unwrap_or(OutputFormat::from("html"));
    let dst = dst.map(PathBuf::from).unwrap_or_else(|| match format {
        OutputFormat::HTML => PathBuf::from("doc"),
        OutputFormat::Markdown => PathBuf::from("doc/error-index.md"),
        OutputFormat::Unknown(..) => PathBuf::from("<nul>"),
    });
    (format, dst)
}

fn main() {
    rustc_driver::init_env_logger("RUST_LOG");
    let (format, dst) = parse_args();
    let result = main_with_result(format, &dst);
    if let Err(e) = result {
        panic!("{:?}", e);
    }
}
