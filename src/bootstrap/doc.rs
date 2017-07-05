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

// rules.doc("doc-nomicon", "src/doc/nomicon")
//      .dep(move |s| {
//          s.name("tool-rustbook")
//           .host(&build.build)
//           .target(&build.build)
//           .stage(0)
//      })
//      .default(build.config.docs)
//      .run(move |s| doc::rustbook(build, s.target, "nomicon"));
// rules.doc("doc-reference", "src/doc/reference")
//      .dep(move |s| {
//          s.name("tool-rustbook")
//           .host(&build.build)
//           .target(&build.build)
//           .stage(0)
//      })
//      .default(build.config.docs)
//      .run(move |s| doc::rustbook(build, s.target, "reference"));

#[derive(Serialize)]
pub struct Rustbook<'a> {
    target: &'a str,
    name: &'a str,
}

impl<'a> Step<'a> for Rustbook<'a> {
    type Output = ();

    /// Invoke `rustbook` for `target` for the doc book `name`.
    ///
    /// This will not actually generate any documentation if the documentation has
    /// already been generated.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let target = self.target;
        let name = self.name;
        let src = build.src.join("src/doc");
        rustbook_src(build, target, name, &src);
    }
}

//rules.doc("doc-unstable-book", "src/doc/unstable-book")
//     .dep(move |s| {
//         s.name("tool-rustbook")
//          .host(&build.build)
//          .target(&build.build)
//          .stage(0)
//     })
//     .dep(move |s| s.name("doc-unstable-book-gen"))
//     .default(build.config.docs)
//     .run(move |s| doc::rustbook_src(build,
//                                     s.target,
//                                     "unstable-book",
//                                     &build.md_doc_out(s.target)));

#[derive(Serialize)]
pub struct RustbookSrc<'a> {
    target: &'a str,
    name: &'a str,
    src: &'a Path,
}

impl<'a> Step<'a> for RustbookSrc<'a> {
    type Output = ();

    /// Invoke `rustbook` for `target` for the doc book `name` from the `src` path.
    ///
    /// This will not actually generate any documentation if the documentation has
    /// already been generated.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let target = self.target;
        let name = self.name;
        let src = self.src;
        let out = build.doc_out(target);
        t!(fs::create_dir_all(&out));

        let out = out.join(name);
        let compiler = Compiler::new(0, &build.build);
        let src = src.join(name);
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
}

// rules.doc("doc-book", "src/doc/book")
//      .dep(move |s| {
//          s.name("tool-rustbook")
//           .host(&build.build)
//           .target(&build.build)
//           .stage(0)
//      })
//      .default(build.config.docs)
//      .run(move |s| doc::book(build, s.target, "book"));

#[derive(Serialize)]
pub struct TheBook<'a> {
    target: &'a str,
    name: &'a str,
}

impl<'a> Step<'a> for TheBook<'a> {
    type Output = ();

    /// Build the book and associated stuff.
    ///
    /// We need to build:
    ///
    /// * Book (first edition)
    /// * Book (second edition)
    /// * Index page
    /// * Redirect pages
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let target = self.target;
        let name = self.name;
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
}

fn invoke_rustdoc(build: &Build, target: &str, markdown: &str) {
    let out = build.doc_out(target);

    let compiler = Compiler::new(0, &build.build);

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

// rules.doc("doc-standalone", "src/doc")
//      .dep(move |s| {
//          s.name("rustc")
//           .host(&build.build)
//           .target(&build.build)
//           .stage(0)
//      })
//      .default(build.config.docs)
//      .run(move |s| doc::standalone(build, s.target));

#[derive(Serialize)]
pub struct Standalone<'a> {
    target: &'a str,
}

impl<'a> Step<'a> for Standalone<'a> {
    type Output = ();

    /// Generates all standalone documentation as compiled by the rustdoc in `stage`
    /// for the `target` into `out`.
    ///
    /// This will list all of `src/doc` looking for markdown files and appropriately
    /// perform transformations like substituting `VERSION`, `SHORT_HASH`, and
    /// `STAMP` alongw ith providing the various header/footer HTML we've cutomized.
    ///
    /// In the end, this is just a glorified wrapper around rustdoc!
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let target = self.target;
        println!("Documenting standalone ({})", target);
        let out = build.doc_out(target);
        t!(fs::create_dir_all(&out));

        let compiler = Compiler::new(0, &build.build);

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
}

// for (krate, path, default) in krates("std") {
//     rules.doc(&krate.doc_step, path)
//          .dep(|s| s.name("libstd-link"))
//          .default(default && build.config.docs)
//          .run(move |s| doc::std(build, s.stage, s.target));
// }

#[derive(Serialize)]
pub struct Std<'a> {
    stage: u32,
    target: &'a str,
}

impl<'a> Step<'a> for Std<'a> {
    type Output = ();

    /// Compile all standard library documentation.
    ///
    /// This will generate all documentation for the standard library and its
    /// dependencies. This is largely just a wrapper around `cargo doc`.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let stage = self.stage;
        let target = self.target;
        println!("Documenting stage{} std ({})", stage, target);
        let out = build.doc_out(target);
        t!(fs::create_dir_all(&out));
        let compiler = Compiler::new(stage, &build.build);
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
            for krate in &["alloc", "collections", "core", "std", "std_unicode"] {
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
}

// for (krate, path, default) in krates("test") {
//     rules.doc(&krate.doc_step, path)
//          .dep(|s| s.name("libtest-link"))
//          // Needed so rustdoc generates relative links to std.
//          .dep(|s| s.name("doc-crate-std"))
//          .default(default && build.config.compiler_docs)
//          .run(move |s| doc::test(build, s.stage, s.target));
// }

#[derive(Serialize)]
pub struct Test<'a> {
    stage: u32,
    test: &'a str,
}

impl<'a> Step<'a> for Test<'a> {
    type Output = ();

    /// Compile all libtest documentation.
    ///
    /// This will generate all documentation for libtest and its dependencies. This
    /// is largely just a wrapper around `cargo doc`.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let stage = self.stage;
        let target = self.target;
        println!("Documenting stage{} test ({})", stage, target);
        let out = build.doc_out(target);
        t!(fs::create_dir_all(&out));
        let compiler = Compiler::new(stage, &build.build);
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
}

// for (krate, path, default) in krates("rustc-main") {
//     rules.doc(&krate.doc_step, path)
//          .dep(|s| s.name("librustc-link"))
//          // Needed so rustdoc generates relative links to std.
//          .dep(|s| s.name("doc-crate-std"))
//          .host(true)
//          .default(default && build.config.docs)
//          .run(move |s| doc::rustc(build, s.stage, s.target));
// }
//

#[derive(Serialize)]
pub struct Rustc<'a> {
    stage: u32,
    target: &'a str,
}

impl<'a> Step<'a> for Rustc<'a> {
    type Output = ();

    /// Generate all compiler documentation.
    ///
    /// This will generate all documentation for the compiler libraries and their
    /// dependencies. This is largely just a wrapper around `cargo doc`.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let stage = self.stage;
        let target = self.target;
        println!("Documenting stage{} compiler ({})", stage, target);
        let out = build.doc_out(target);
        t!(fs::create_dir_all(&out));
        let compiler = Compiler::new(stage, &build.build);
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
}

// rules.doc("doc-error-index", "src/tools/error_index_generator")
//      .dep(move |s| s.name("tool-error-index").target(&build.build).stage(0))
//      .dep(move |s| s.name("librustc-link"))
//      .default(build.config.docs)
//      .host(true)
//      .run(move |s| doc::error_index(build, s.target));

#[derive(Serialize)]
pub struct ErrorIndex<'a> {
    target: &'a str,
}

impl<'a> Step<'a> for ErrorIndex<'a> {
    type Output = ();

    /// Generates the HTML rendered error-index by running the
    /// `error_index_generator` tool.
    fn run(self, builder: &Builder) {
        let builder = builder.build;
        let target = self.target;
        println!("Documenting error index ({})", target);
        let out = build.doc_out(target);
        t!(fs::create_dir_all(&out));
        let compiler = Compiler::new(0, &build.build);
        let mut index = build.tool_cmd(&compiler, "error_index_generator");
        index.arg("html");
        index.arg(out.join("error-index.html"));

        // FIXME: shouldn't have to pass this env var
        index.env("CFG_BUILD", &build.build);

        build.run(&mut index);
    }
}

// rules.doc("doc-unstable-book-gen", "src/tools/unstable-book-gen")
//      .dep(move |s| {
//          s.name("tool-unstable-book-gen")
//           .host(&build.build)
//           .target(&build.build)
//           .stage(0)
//      })
//      .dep(move |s| s.name("libstd-link"))
//      .default(build.config.docs)
//      .host(true)
//      .run(move |s| doc::unstable_book_gen(build, s.target));

#[derive(Serialize)]
pub struct UnstableBookGen<'a> {
    target: &'a str,
}

impl<'a> Step<'a> for UnstableBookGen<'a> {
    type Output = ();

    fn run(self, builder: &Builder) {
        let build = builder.build;
        let target = self.target;
        println!("Generating unstable book md files ({})", target);
        let out = build.md_doc_out(target).join("unstable-book");
        t!(fs::create_dir_all(&out));
        t!(fs::remove_dir_all(&out));
        let compiler = Compiler::new(0, &build.build);
        let mut cmd = build.tool_cmd(&compiler, "unstable-book-gen");
        cmd.arg(build.src.join("src"));
        cmd.arg(out);

        build.run(&mut cmd);
    }
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
