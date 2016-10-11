// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the `build` subcommand, used to compile a book.

use std::env;
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};
use rustc_back::tempdir::TempDir;

use subcommand::Subcommand;
use term::Term;
use error::{err, CliResult, CommandResult};
use book;
use book::{Book, BookItem};

use rustdoc;

struct Build;

pub fn parse_cmd(name: &str) -> Option<Box<Subcommand>> {
    if name == "build" {
        Some(Box::new(Build))
    } else {
        None
    }
}

fn write_toc(book: &Book, current_page: &BookItem, out: &mut Write) -> io::Result<()> {
    fn walk_items(items: &[BookItem],
                  section: &str,
                  current_page: &BookItem,
                  out: &mut Write) -> io::Result<()> {
        for (i, item) in items.iter().enumerate() {
            walk_item(item, &format!("{}{}.", section, i + 1)[..], current_page, out)?;
        }
        Ok(())
    }
    fn walk_item(item: &BookItem,
                 section: &str,
                 current_page: &BookItem,
                 out: &mut Write) -> io::Result<()> {
        let class_string = if item.path == current_page.path {
            "class='active'"
        } else {
            ""
        };

        writeln!(out, "<li><a {} href='{}'><b>{}</b> {}</a>",
                 class_string,
                 current_page.path_to_root.join(&item.path).with_extension("html").display(),
                 section,
                 item.title)?;
        if !item.children.is_empty() {
            writeln!(out, "<ol class='section'>")?;
            let _ = walk_items(&item.children[..], section, current_page, out);
            writeln!(out, "</ol>")?;
        }
        writeln!(out, "</li>")?;

        Ok(())
    }

    writeln!(out, "<div id='toc' class='mobile-hidden'>")?;
    writeln!(out, "<ol class='chapter'>")?;
    walk_items(&book.chapters[..], "", &current_page, out)?;
    writeln!(out, "</ol>")?;
    writeln!(out, "</div>")?;

    Ok(())
}

fn render(book: &Book, tgt: &Path) -> CliResult<()> {
    let tmp = TempDir::new("rustbook")?;

    for (_section, item) in book.iter() {
        let out_path = match item.path.parent() {
            Some(p) => tgt.join(p),
            None => tgt.to_path_buf(),
        };

        let src;
        if env::args().len() < 3 {
            src = env::current_dir().unwrap().clone();
        } else {
            src = PathBuf::from(&env::args().nth(2).unwrap());
        }
        // preprocess the markdown, rerouting markdown references to html
        // references
        let mut markdown_data = String::new();
        File::open(&src.join(&item.path)).and_then(|mut f| {
            f.read_to_string(&mut markdown_data)
        })?;
        let preprocessed_path = tmp.path().join(item.path.file_name().unwrap());
        {
            let urls = markdown_data.replace(".md)", ".html)");
            File::create(&preprocessed_path).and_then(|mut f| {
                f.write_all(urls.as_bytes())
            })?;
        }

        // write the prelude to a temporary HTML file for rustdoc inclusion
        let prelude = tmp.path().join("prelude.html");
        {
            let mut buffer = BufWriter::new(File::create(&prelude)?);
            writeln!(&mut buffer, r#"
                <div id="nav">
                    <button id="toggle-nav">
                        <span class="sr-only">Toggle navigation</span>
                        <span class="bar"></span>
                        <span class="bar"></span>
                        <span class="bar"></span>
                    </button>
                </div>"#)?;
            let _ = write_toc(book, &item, &mut buffer);
            writeln!(&mut buffer, "<div id='page-wrapper'>")?;
            writeln!(&mut buffer, "<div id='page'>")?;
        }

        // write the postlude to a temporary HTML file for rustdoc inclusion
        let postlude = tmp.path().join("postlude.html");
        {
            let mut buffer = BufWriter::new(File::create(&postlude)?);
            writeln!(&mut buffer, "<script src='rustbook.js'></script>")?;
            writeln!(&mut buffer, "</div></div>")?;
        }

        fs::create_dir_all(&out_path)?;

        let rustdoc_args: &[String] = &[
            "".to_string(),
            preprocessed_path.display().to_string(),
            format!("-o{}", out_path.display()),
            format!("--html-before-content={}", prelude.display()),
            format!("--html-after-content={}", postlude.display()),
            format!("--markdown-playground-url=https://play.rust-lang.org/"),
            format!("--markdown-css={}", item.path_to_root.join("rustbook.css").display()),
            "--markdown-no-toc".to_string(),
        ];
        let output_result = rustdoc::main_args(rustdoc_args);
        if output_result != 0 {
            let message = format!("Could not execute `rustdoc` with {:?}: {}",
                                  rustdoc_args, output_result);
            return Err(err(&message));
        }
    }

    // create index.html from the root README
    fs::copy(&tgt.join("README.html"), &tgt.join("index.html"))?;

    Ok(())
}

impl Subcommand for Build {
    fn parse_args(&mut self, _: &[String]) -> CliResult<()> {
        Ok(())
    }
    fn usage(&self) {}
    fn execute(&mut self, term: &mut Term) -> CommandResult<()> {
        let cwd = env::current_dir().unwrap();
        let src;
        let tgt;

        if env::args().len() < 3 {
            src = cwd.clone();
        } else {
            src = PathBuf::from(&env::args().nth(2).unwrap());
        }

        if env::args().len() < 4 {
            tgt = cwd.join("_book");
        } else {
            tgt = PathBuf::from(&env::args().nth(3).unwrap());
        }

        // `_book` directory may already exist from previous runs. Check and
        // delete it if it exists.
        for entry in fs::read_dir(&cwd)? {
            let path = entry?.path();
            if path == tgt { fs::remove_dir_all(&tgt)? }
        }
        fs::create_dir(&tgt)?;

        // Copy static files
        let css = include_bytes!("static/rustbook.css");
        let js = include_bytes!("static/rustbook.js");

        let mut css_file = File::create(tgt.join("rustbook.css"))?;
        css_file.write_all(css)?;

        let mut js_file = File::create(tgt.join("rustbook.js"))?;
        js_file.write_all(js)?;


        let mut summary = File::open(&src.join("SUMMARY.md"))?;
        match book::parse_summary(&mut summary, &src) {
            Ok(book) => {
                // execute rustdoc on the whole book
                render(&book, &tgt)
            }
            Err(errors) => {
                let n = errors.len();
                for err in errors {
                    term.err(&format!("error: {}", err)[..]);
                }

                Err(err(&format!("{} errors occurred", n)))
            }
        }
    }
}
