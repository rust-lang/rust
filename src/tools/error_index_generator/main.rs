#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_span;

use crate::error_codes::error_codes;

use std::env;
use std::error::Error;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use rustc_span::edition::DEFAULT_EDITION;

use rustdoc::html::markdown::{ErrorCodes, HeadingOffset, IdMap, Markdown, Playground};

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
    HTML(HTMLFormatter),
    Markdown,
    Unknown(String),
}

impl OutputFormat {
    fn from(format: &str, resource_suffix: &str) -> OutputFormat {
        match &*format.to_lowercase() {
            "html" => OutputFormat::HTML(HTMLFormatter(resource_suffix.to_owned())),
            "markdown" => OutputFormat::Markdown,
            s => OutputFormat::Unknown(s.to_owned()),
        }
    }
}

struct HTMLFormatter(String);

impl HTMLFormatter {
    fn create_error_code_file(
        &self,
        err_code: &str,
        explanation: &str,
        parent_dir: &Path,
    ) -> Result<(), Box<dyn Error>> {
        let mut output_file = File::create(parent_dir.join(err_code).with_extension("html"))?;

        self.header(&mut output_file, "../", "")?;
        self.title(&mut output_file, &format!("Error code {}", err_code))?;

        let mut id_map = IdMap::new();
        let playground =
            Playground { crate_name: None, url: String::from("https://play.rust-lang.org/") };
        write!(
            output_file,
            "{}",
            Markdown {
                content: explanation,
                links: &[],
                ids: &mut id_map,
                error_codes: ErrorCodes::Yes,
                edition: DEFAULT_EDITION,
                playground: &Some(playground),
                heading_offset: HeadingOffset::H1,
            }
            .into_string()
        )?;
        write!(
            output_file,
            "<p>\
                <a style='text-align: center;display: block;width: 100%;' \
                   href='../error-index.html'>Back to list of error codes</a>\
             </p>",
        )?;

        self.footer(&mut output_file)
    }

    fn header(
        &self,
        output: &mut dyn Write,
        extra_path: &str,
        extra: &str,
    ) -> Result<(), Box<dyn Error>> {
        write!(
            output,
            r##"<!DOCTYPE html>
<html>
<head>
<title>Rust Compiler Error Index</title>
<meta charset="utf-8">
<!-- Include rust.css after light.css so its rules take priority. -->
<link rel="stylesheet" type="text/css" href="{extra_path}rustdoc{suffix}.css"/>
<link rel="stylesheet" type="text/css" href="{extra_path}light{suffix}.css"/>
<link rel="stylesheet" type="text/css" href="{extra_path}rust.css"/>
<style>
.error-undescribed {{
    display: none;
}}
</style>{extra}
</head>
<body>
"##,
            suffix = self.0,
        )?;
        Ok(())
    }

    fn title(&self, output: &mut dyn Write, title: &str) -> Result<(), Box<dyn Error>> {
        write!(output, "<h1>{}</h1>\n", title)?;
        Ok(())
    }

    fn footer(&self, output: &mut dyn Write) -> Result<(), Box<dyn Error>> {
        write!(output, "</body></html>")?;
        Ok(())
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

fn render_html(output_path: &Path, formatter: HTMLFormatter) -> Result<(), Box<dyn Error>> {
    let mut output_file = File::create(output_path)?;

    let error_codes_dir = "error_codes";

    let parent = output_path.parent().expect("There should have a parent").join(error_codes_dir);

    if !parent.exists() {
        create_dir_all(&parent)?;
    }

    formatter.header(
        &mut output_file,
        "",
        &format!(
            r#"<script>(function() {{
    if (window.location.hash) {{
        let code = window.location.hash.replace(/^#/, '');
        // We have to make sure this pattern matches to avoid inadvertently creating an
        // open redirect.
        if (/^E[0-9]+$/.test(code)) {{
            window.location = './{error_codes_dir}/' + code + '.html';
        }}
    }}
}})()</script>"#
        ),
    )?;
    formatter.title(&mut output_file, "Rust Compiler Error Index")?;

    write!(
        output_file,
        "<p>This page lists all the error codes emitted by the Rust compiler. If you want a full \
            explanation on an error code, click on it.</p>\
         <ul>",
    )?;
    for (err_code, explanation) in error_codes().iter() {
        if let Some(explanation) = explanation {
            write!(
                output_file,
                "<li><a href='./{0}/{1}.html'>{1}</a></li>",
                error_codes_dir, err_code
            )?;
            formatter.create_error_code_file(err_code, explanation, &parent)?;
        } else {
            write!(output_file, "<li>{}</li>", err_code)?;
        }
    }
    write!(output_file, "</ul>")?;
    formatter.footer(&mut output_file)
}

fn main_with_result(format: OutputFormat, dst: &Path) -> Result<(), Box<dyn Error>> {
    match format {
        OutputFormat::Unknown(s) => panic!("Unknown output format: {}", s),
        OutputFormat::HTML(h) => render_html(dst, h),
        OutputFormat::Markdown => render_markdown(dst),
    }
}

fn parse_args() -> (OutputFormat, PathBuf) {
    let mut args = env::args().skip(1);
    let format = args.next();
    let dst = args.next();
    let resource_suffix = args.next().unwrap_or_else(String::new);
    let format = format
        .map(|a| OutputFormat::from(&a, &resource_suffix))
        .unwrap_or(OutputFormat::from("html", &resource_suffix));
    let dst = dst.map(PathBuf::from).unwrap_or_else(|| match format {
        OutputFormat::HTML(..) => PathBuf::from("doc/error-index.html"),
        OutputFormat::Markdown => PathBuf::from("doc/error-index.md"),
        OutputFormat::Unknown(..) => PathBuf::from("<nul>"),
    });
    (format, dst)
}

fn main() {
    rustc_driver::init_env_logger("RUST_LOG");
    let (format, dst) = parse_args();
    let result =
        rustc_span::create_default_session_globals_then(move || main_with_result(format, &dst));
    if let Err(e) = result {
        panic!("{}", e.to_string());
    }
}
