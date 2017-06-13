// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Documentation generation for rustbuild.
//!
//! This module implements generation for all bits and pieces of documentation
//! for the Rust project. This notably includes suites like the rust book, the
//! nomicon, standalone documentation, etc.
//!
//! Everything here is basically just a shim around calling either `rustbook` or
//! `rustdoc`.

use std::fs::{self, File};
use std::io::prelude::*;
use std::io;
use std::path::Path;
use std::process::Command;

use {Build, Compiler, Mode};
use util::{cp_r, symlink_dir};
use build_helper::up_to_date;

/// Invoke `rustbook` as compiled in `stage` for `target` for the doc book
/// `name` into the `out` path.
///
/// This will not actually generate any documentation if the documentation has
/// already been generated.
pub fn rustbook(build: &Build, target: &str, name: &str) {
    let out = build.doc_out(target);
    t!(fs::create_dir_all(&out));

    let out = out.join(name);
    let compiler = Compiler::new(0, &build.config.build);
    let src = build.src.join("src/doc").join(name);
    let index = out.join("index.html");
    let rustbook = build.tool(&compiler, "rustbook");
    if up_to_date(&src, &index) && up_to_date(&rustbook, &index) {
        return
    }
    println!("Rustbook ({}) - {}", target, name);
    let _ = fs::remove_dir_all(&out);
    build.run(build.tool_cmd(&compiler, "rustbook")
                   .arg("build")
                   .arg(&src)
                   .arg("-d")
                   .arg(out));
}

/// Build the book and associated stuff.
///
/// We need to build:
///
/// * Book (first edition)
/// * Book (second edition)
/// * Index page
/// * Redirect pages
pub fn book(build: &Build, target: &str, name: &str) {
    // build book first edition
    rustbook(build, target, &format!("{}/first-edition", name));

    // build book second edition
    rustbook(build, target, &format!("{}/second-edition", name));

    // build the index page
    let index = format!("{}/index.md", name);
    println!("Documenting book index ({})", target);
    invoke_rustdoc(build, target, &index);

    // build the redirect pages
    println!("Documenting book redirect pages ({})", target);
    for file in t!(fs::read_dir(build.src.join("src/doc/book/redirects"))) {
        let file = t!(file);
        let path = file.path();
        let path = path.to_str().unwrap();

        invoke_rustdoc(build, target, path);
    }
}

fn invoke_rustdoc(build: &Build, target: &str, markdown: &str) {
    let out = build.doc_out(target);

    let compiler = Compiler::new(0, &build.config.build);

    let path = build.src.join("src/doc").join(markdown);

    let rustdoc = build.rustdoc(&compiler);

    let favicon = build.src.join("src/doc/favicon.inc");
    let footer = build.src.join("src/doc/footer.inc");

    let version_input = build.src.join("src/doc/version_info.html.template");
    let version_info = out.join("version_info.html");

    if !up_to_date(&version_input, &version_info) {
        let mut info = String::new();
        t!(t!(File::open(&version_input)).read_to_string(&mut info));
        let info = info.replace("VERSION", &build.rust_release())
                       .replace("SHORT_HASH", build.rust_info.sha_short().unwrap_or(""))
                       .replace("STAMP", build.rust_info.sha().unwrap_or(""));
        t!(t!(File::create(&version_info)).write_all(info.as_bytes()));
    }

    let mut cmd = Command::new(&rustdoc);

    build.add_rustc_lib_path(&compiler, &mut cmd);

    let out = out.join("book");

    t!(fs::copy(build.src.join("src/doc/rust.css"), out.join("rust.css")));

    cmd.arg("--html-after-content").arg(&footer)
        .arg("--html-before-content").arg(&version_info)
        .arg("--html-in-header").arg(&favicon)
        .arg("--markdown-playground-url")
        .arg("https://play.rust-lang.org/")
        .arg("-o").arg(&out)
        .arg(&path)
        .arg("--markdown-css")
        .arg("rust.css");

    build.run(&mut cmd);
}

/// Generates all standalone documentation as compiled by the rustdoc in `stage`
/// for the `target` into `out`.
///
/// This will list all of `src/doc` looking for markdown files and appropriately
/// perform transformations like substituting `VERSION`, `SHORT_HASH`, and
/// `STAMP` alongw ith providing the various header/footer HTML we've cutomized.
///
/// In the end, this is just a glorified wrapper around rustdoc!
pub fn standalone(build: &Build, target: &str) {
    println!("Documenting standalone ({})", target);
    let out = build.doc_out(target);
    t!(fs::create_dir_all(&out));

    let compiler = Compiler::new(0, &build.config.build);

    let favicon = build.src.join("src/doc/favicon.inc");
    let footer = build.src.join("src/doc/footer.inc");
    let full_toc = build.src.join("src/doc/full-toc.inc");
    t!(fs::copy(build.src.join("src/doc/rust.css"), out.join("rust.css")));

    let version_input = build.src.join("src/doc/version_info.html.template");
    let version_info = out.join("version_info.html");

    if !up_to_date(&version_input, &version_info) {
        let mut info = String::new();
        t!(t!(File::open(&version_input)).read_to_string(&mut info));
        let info = info.replace("VERSION", &build.rust_release())
                       .replace("SHORT_HASH", build.rust_info.sha_short().unwrap_or(""))
                       .replace("STAMP", build.rust_info.sha().unwrap_or(""));
        t!(t!(File::create(&version_info)).write_all(info.as_bytes()));
    }

    for file in t!(fs::read_dir(build.src.join("src/doc"))) {
        let file = t!(file);
        let path = file.path();
        let filename = path.file_name().unwrap().to_str().unwrap();
        if !filename.ends_with(".md") || filename == "README.md" {
            continue
        }

        let html = out.join(filename).with_extension("html");
        let rustdoc = build.rustdoc(&compiler);
        if up_to_date(&path, &html) &&
           up_to_date(&footer, &html) &&
           up_to_date(&favicon, &html) &&
           up_to_date(&full_toc, &html) &&
           up_to_date(&version_info, &html) &&
           up_to_date(&rustdoc, &html) {
            continue
        }

        let mut cmd = Command::new(&rustdoc);
        build.add_rustc_lib_path(&compiler, &mut cmd);
        cmd.arg("--html-after-content").arg(&footer)
           .arg("--html-before-content").arg(&version_info)
           .arg("--html-in-header").arg(&favicon)
           .arg("--markdown-playground-url")
           .arg("https://play.rust-lang.org/")
           .arg("-o").arg(&out)
           .arg(&path);

        if filename == "not_found.md" {
            cmd.arg("--markdown-no-toc")
               .arg("--markdown-css")
               .arg("https://doc.rust-lang.org/rust.css");
        } else {
            cmd.arg("--markdown-css").arg("rust.css");
        }
        build.run(&mut cmd);
    }
}

/// Compile all standard library documentation.
///
/// This will generate all documentation for the standard library and its
/// dependencies. This is largely just a wrapper around `cargo doc`.
pub fn std(build: &Build, stage: u32, target: &str) {
    println!("Documenting stage{} std ({})", stage, target);
    let out = build.doc_out(target);
    t!(fs::create_dir_all(&out));
    let compiler = Compiler::new(stage, &build.config.build);
    let compiler = if build.force_use_stage1(&compiler, target) {
        Compiler::new(1, compiler.host)
    } else {
        compiler
    };
    let out_dir = build.stage_out(&compiler, Mode::Libstd)
                       .join(target).join("doc");
    let rustdoc = build.rustdoc(&compiler);

    // Here what we're doing is creating a *symlink* (directory junction on
    // Windows) to the final output location. This is not done as an
    // optimization but rather for correctness. We've got three trees of
    // documentation, one for std, one for test, and one for rustc. It's then
    // our job to merge them all together.
    //
    // Unfortunately rustbuild doesn't know nearly as well how to merge doc
    // trees as rustdoc does itself, so instead of actually having three
    // separate trees we just have rustdoc output to the same location across
    // all of them.
    //
    // This way rustdoc generates output directly into the output, and rustdoc
    // will also directly handle merging.
    let my_out = build.crate_doc_out(target);
    build.clear_if_dirty(&my_out, &rustdoc);
    t!(symlink_dir_force(&my_out, &out_dir));

    let mut cargo = build.cargo(&compiler, Mode::Libstd, target, "doc");
    cargo.arg("--manifest-path")
         .arg(build.src.join("src/libstd/Cargo.toml"))
         .arg("--features").arg(build.std_features());

    // We don't want to build docs for internal std dependencies unless
    // in compiler-docs mode. When not in that mode, we whitelist the crates
    // for which docs must be built.
    if !build.config.compiler_docs {
        cargo.arg("--no-deps");
        for krate in &["alloc", "core", "std", "std_unicode"] {
            cargo.arg("-p").arg(krate);
            // Create all crate output directories first to make sure rustdoc uses
            // relative links.
            // FIXME: Cargo should probably do this itself.
            t!(fs::create_dir_all(out_dir.join(krate)));
        }
    }


    build.run(&mut cargo);
    cp_r(&my_out, &out);
}

/// Compile all libtest documentation.
///
/// This will generate all documentation for libtest and its dependencies. This
/// is largely just a wrapper around `cargo doc`.
pub fn test(build: &Build, stage: u32, target: &str) {
    println!("Documenting stage{} test ({})", stage, target);
    let out = build.doc_out(target);
    t!(fs::create_dir_all(&out));
    let compiler = Compiler::new(stage, &build.config.build);
    let compiler = if build.force_use_stage1(&compiler, target) {
        Compiler::new(1, compiler.host)
    } else {
        compiler
    };
    let out_dir = build.stage_out(&compiler, Mode::Libtest)
                       .join(target).join("doc");
    let rustdoc = build.rustdoc(&compiler);

    // See docs in std above for why we symlink
    let my_out = build.crate_doc_out(target);
    build.clear_if_dirty(&my_out, &rustdoc);
    t!(symlink_dir_force(&my_out, &out_dir));

    let mut cargo = build.cargo(&compiler, Mode::Libtest, target, "doc");
    cargo.arg("--manifest-path")
         .arg(build.src.join("src/libtest/Cargo.toml"));
    build.run(&mut cargo);
    cp_r(&my_out, &out);
}

/// Generate all compiler documentation.
///
/// This will generate all documentation for the compiler libraries and their
/// dependencies. This is largely just a wrapper around `cargo doc`.
pub fn rustc(build: &Build, stage: u32, target: &str) {
    println!("Documenting stage{} compiler ({})", stage, target);
    let out = build.doc_out(target);
    t!(fs::create_dir_all(&out));
    let compiler = Compiler::new(stage, &build.config.build);
    let compiler = if build.force_use_stage1(&compiler, target) {
        Compiler::new(1, compiler.host)
    } else {
        compiler
    };
    let out_dir = build.stage_out(&compiler, Mode::Librustc)
                       .join(target).join("doc");
    let rustdoc = build.rustdoc(&compiler);

    // See docs in std above for why we symlink
    let my_out = build.crate_doc_out(target);
    build.clear_if_dirty(&my_out, &rustdoc);
    t!(symlink_dir_force(&my_out, &out_dir));

    let mut cargo = build.cargo(&compiler, Mode::Librustc, target, "doc");
    cargo.arg("--manifest-path")
         .arg(build.src.join("src/rustc/Cargo.toml"))
         .arg("--features").arg(build.rustc_features());

    if build.config.compiler_docs {
        // src/rustc/Cargo.toml contains bin crates called rustc and rustdoc
        // which would otherwise overwrite the docs for the real rustc and
        // rustdoc lib crates.
        cargo.arg("-p").arg("rustc_driver")
             .arg("-p").arg("rustdoc");
    } else {
        // Like with libstd above if compiler docs aren't enabled then we're not
        // documenting internal dependencies, so we have a whitelist.
        cargo.arg("--no-deps");
        for krate in &["proc_macro"] {
            cargo.arg("-p").arg(krate);
        }
    }

    build.run(&mut cargo);
    cp_r(&my_out, &out);
}

/// Generates the HTML rendered error-index by running the
/// `error_index_generator` tool.
pub fn error_index(build: &Build, target: &str) {
    println!("Documenting error index ({})", target);
    let out = build.doc_out(target);
    t!(fs::create_dir_all(&out));
    let compiler = Compiler::new(0, &build.config.build);
    let mut index = build.tool_cmd(&compiler, "error_index_generator");
    index.arg("html");
    index.arg(out.join("error-index.html"));

    // FIXME: shouldn't have to pass this env var
    index.env("CFG_BUILD", &build.config.build);

    build.run(&mut index);
}

fn symlink_dir_force(src: &Path, dst: &Path) -> io::Result<()> {
    if let Ok(m) = fs::symlink_metadata(dst) {
        if m.file_type().is_dir() {
            try!(fs::remove_dir_all(dst));
        } else {
            // handle directory junctions on windows by falling back to
            // `remove_dir`.
            try!(fs::remove_file(dst).or_else(|_| {
                fs::remove_dir(dst)
            }));
        }
    }

    symlink_dir(src, dst)
}
