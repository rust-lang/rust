#![feature(rustc_private)]

extern crate rustc_driver;

// We use the function we generate from `register_diagnostics!`.
use crate::error_codes::error_codes;

use std::env;
use std::error::Error;
use std::fs::{self, create_dir_all, File};
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

fn move_folder(source: &Path, target: &Path) -> Result<(), Box<dyn Error>> {
    let entries =
        fs::read_dir(source)?.map(|res| res.map(|e| e.path())).collect::<Result<Vec<_>, _>>()?;

    for entry in entries {
        let file_name = entry.file_name().expect("file_name() failed").to_os_string();
        let output = target.join(file_name);
        if entry.is_file() {
            fs::rename(entry, output)?;
        } else {
            if !output.exists() {
                create_dir_all(&output)?;
            }
            move_folder(&entry, &output)?;
        }
    }

    fs::remove_dir(&source)?;

    Ok(())
}

fn render_html(output_path: &Path) -> Result<(), Box<dyn Error>> {
    // We need to render into a temporary folder to prevent `mdbook` from removing everything
    // in the output folder (including other completely unrelated things).
    let tmp_output = output_path.join("tmp");

    if !tmp_output.exists() {
        create_dir_all(&tmp_output)?;
    }

    render_html_inner(&tmp_output)?;

    move_folder(&tmp_output, output_path)?;

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

fn render_html_inner(output_path: &Path) -> Result<(), Box<dyn Error>> {
    // We need to have a little difference between `summary` and `introduction` because the "draft"
    // chapters (the ones looking like `[a]()`) are not handled correctly when being put into a
    // `Chapter` directly: they generate a link whereas they shouldn't.
    let mut introduction = format!(
        "<script>{}</script>
# Rust error codes index

This page lists all the error codes emitted by the Rust compiler.

",
        include_str!("redirect.js")
    );

    let err_codes = error_codes();
    let mut chapters = Vec::with_capacity(err_codes.len());

    for (err_code, explanation) in err_codes.iter() {
        if let Some(explanation) = explanation {
            introduction.push_str(&format!(" * [{0}](./error_codes/{0}.html)\n", err_code));

            let content = add_rust_attribute_on_codeblock(explanation);
            chapters.push(BookItem::Chapter(Chapter {
                name: err_code.to_string(),
                content: format!("# Error code {}\n\n{}\n", err_code, content),
                number: None,
                sub_items: Vec::new(),
                // We generate it into the `error_codes` folder.
                path: Some(PathBuf::from(&format!("error_codes/{}.html", err_code))),
                source_path: None,
                parent_names: Vec::new(),
            }));
        } else {
            introduction.push_str(&format!(" * {}\n", err_code));
        }
    }

    let mut config = Config::from_str(include_str!("book_config.toml"))?;
    config.build.build_dir = output_path.to_path_buf();
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

    // We don't need this file since it's handled by doc.rust-lang.org directly.
    let _ = fs::remove_file(output_path.join("404.html"));
    // We don't want this file either because it would overwrite the already existing `index.html`.
    let _ = fs::remove_file(output_path.join("index.html"));

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
