//! Documentation generation for rustbuilder.
//!
//! This module implements generation for all bits and pieces of documentation
//! for the Rust project. This notably includes suites like the rust book, the
//! nomicon, rust by example, standalone documentation, etc.
//!
//! Everything here is basically just a shim around calling either `rustbook` or
//! `rustdoc`.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{PathBuf, Path};

use crate::Mode;
use build_helper::{t, up_to_date};

use crate::util::symlink_dir;
use crate::builder::{Builder, Compiler, RunConfig, ShouldRun, Step};
use crate::tool::{self, prepare_tool_cargo, Tool, SourceType};
use crate::compile;
use crate::cache::{INTERNER, Interned};
use crate::config::Config;

macro_rules! book {
    ($($name:ident, $path:expr, $book_name:expr, $book_ver:expr;)+) => {
        $(
            #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            target: Interned<String>,
        }

        impl Step for $name {
            type Output = ();
            const DEFAULT: bool = true;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                let builder = run.builder;
                run.path($path).default_condition(builder.config.docs)
            }

            fn make_run(run: RunConfig<'_>) {
                run.builder.ensure($name {
                    target: run.target,
                });
            }

            fn run(self, builder: &Builder<'_>) {
                builder.ensure(RustbookSrc {
                    target: self.target,
                    name: INTERNER.intern_str($book_name),
                    version: $book_ver,
                    src: doc_src(builder),
                })
            }
        }
        )+
    }
}

// NOTE: When adding a book here, make sure to ALSO build the book by
// adding a build step in `src/bootstrap/builder.rs`!
book!(
    EditionGuide, "src/doc/edition-guide", "edition-guide", RustbookVersion::Latest;
    EmbeddedBook, "src/doc/embedded-book", "embedded-book", RustbookVersion::Latest;
    Nomicon, "src/doc/nomicon", "nomicon", RustbookVersion::Latest;
    Reference, "src/doc/reference", "reference", RustbookVersion::MdBook1;
    RustByExample, "src/doc/rust-by-example", "rust-by-example", RustbookVersion::Latest;
    RustcBook, "src/doc/rustc", "rustc", RustbookVersion::MdBook1;
    RustdocBook, "src/doc/rustdoc", "rustdoc", RustbookVersion::Latest;
);

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
enum RustbookVersion {
    MdBook1,
    Latest,
}

fn doc_src(builder: &Builder<'_>) -> Interned<PathBuf> {
    INTERNER.intern_path(builder.src.join("src/doc"))
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct UnstableBook {
    target: Interned<String>,
}

impl Step for UnstableBook {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/doc/unstable-book").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(UnstableBook {
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(UnstableBookGen {
            target: self.target,
        });
        builder.ensure(RustbookSrc {
            target: self.target,
            name: INTERNER.intern_str("unstable-book"),
            src: builder.md_doc_out(self.target),
            version: RustbookVersion::Latest,
        })
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct CargoBook {
    target: Interned<String>,
    name: Interned<String>,
}

impl Step for CargoBook {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/tools/cargo/src/doc/book").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CargoBook {
            target: run.target,
            name: INTERNER.intern_str("cargo"),
        });
    }

    fn run(self, builder: &Builder<'_>) {
        let target = self.target;
        let name = self.name;
        let src = builder.src.join("src/tools/cargo/src/doc");

        let out = builder.doc_out(target);
        t!(fs::create_dir_all(&out));

        let out = out.join(name);

        builder.info(&format!("Cargo Book ({}) - {}", target, name));

        let _ = fs::remove_dir_all(&out);

        builder.run(builder.tool_cmd(Tool::Rustbook)
                       .arg("build")
                       .arg(&src)
                       .arg("-d")
                       .arg(out));
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
struct RustbookSrc {
    target: Interned<String>,
    name: Interned<String>,
    src: Interned<PathBuf>,
    version: RustbookVersion,
}

impl Step for RustbookSrc {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Invoke `rustbook` for `target` for the doc book `name` from the `src` path.
    ///
    /// This will not actually generate any documentation if the documentation has
    /// already been generated.
    fn run(self, builder: &Builder<'_>) {
        let target = self.target;
        let name = self.name;
        let src = self.src;
        let out = builder.doc_out(target);
        t!(fs::create_dir_all(&out));

        let out = out.join(name);
        let src = src.join(name);
        let index = out.join("index.html");
        let rustbook = builder.tool_exe(Tool::Rustbook);
        let mut rustbook_cmd = builder.tool_cmd(Tool::Rustbook);
        if up_to_date(&src, &index) && up_to_date(&rustbook, &index) {
            return
        }
        builder.info(&format!("Rustbook ({}) - {}", target, name));
        let _ = fs::remove_dir_all(&out);

        let vers = match self.version {
            RustbookVersion::MdBook1 => "1",
            RustbookVersion::Latest => "3",
        };

        builder.run(rustbook_cmd
                       .arg("build")
                       .arg(&src)
                       .arg("-d")
                       .arg(out)
                       .arg("-m")
                       .arg(vers));
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct TheBook {
    compiler: Compiler,
    target: Interned<String>,
    name: &'static str,
}

impl Step for TheBook {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/doc/book").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(TheBook {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
            name: "book",
        });
    }

    /// Builds the book and associated stuff.
    ///
    /// We need to build:
    ///
    /// * Book (first edition)
    /// * Book (second edition)
    /// * Version info and CSS
    /// * Index page
    /// * Redirect pages
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target = self.target;
        let name = self.name;

        // build book
        builder.ensure(RustbookSrc {
            target,
            name: INTERNER.intern_string(name.to_string()),
            version: RustbookVersion::Latest,
            src: doc_src(builder),
        });

        // building older edition redirects

        let source_name = format!("{}/first-edition", name);
        builder.ensure(RustbookSrc {
            target,
            name: INTERNER.intern_string(source_name),
            version: RustbookVersion::Latest,
            src: doc_src(builder),
        });

        let source_name = format!("{}/second-edition", name);
        builder.ensure(RustbookSrc {
            target,
            name: INTERNER.intern_string(source_name),
            version: RustbookVersion::Latest,
            src: doc_src(builder),
        });

        let source_name = format!("{}/2018-edition", name);
        builder.ensure(RustbookSrc {
            target,
            name: INTERNER.intern_string(source_name),
            version: RustbookVersion::Latest,
            src: doc_src(builder),
        });

        // build the version info page and CSS
        builder.ensure(Standalone {
            compiler,
            target,
        });

        // build the redirect pages
        builder.info(&format!("Documenting book redirect pages ({})", target));
        for file in t!(fs::read_dir(builder.src.join("src/doc/book/redirects"))) {
            let file = t!(file);
            let path = file.path();
            let path = path.to_str().unwrap();

            invoke_rustdoc(builder, compiler, target, path);
        }
    }
}

fn invoke_rustdoc(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: Interned<String>,
    markdown: &str,
) {
    let out = builder.doc_out(target);

    let path = builder.src.join("src/doc").join(markdown);

    let header = builder.src.join("src/doc/redirect.inc");
    let footer = builder.src.join("src/doc/footer.inc");
    let version_info = out.join("version_info.html");

    let mut cmd = builder.rustdoc_cmd(compiler);

    let out = out.join("book");

    cmd.arg("--html-after-content").arg(&footer)
        .arg("--html-before-content").arg(&version_info)
        .arg("--html-in-header").arg(&header)
        .arg("--markdown-no-toc")
        .arg("--markdown-playground-url").arg("https://play.rust-lang.org/")
        .arg("-o").arg(&out).arg(&path)
        .arg("--markdown-css").arg("../rust.css");

    builder.run(&mut cmd);
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Standalone {
    compiler: Compiler,
    target: Interned<String>,
}

impl Step for Standalone {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/doc").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Standalone {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
        });
    }

    /// Generates all standalone documentation as compiled by the rustdoc in `stage`
    /// for the `target` into `out`.
    ///
    /// This will list all of `src/doc` looking for markdown files and appropriately
    /// perform transformations like substituting `VERSION`, `SHORT_HASH`, and
    /// `STAMP` along with providing the various header/footer HTML we've customized.
    ///
    /// In the end, this is just a glorified wrapper around rustdoc!
    fn run(self, builder: &Builder<'_>) {
        let target = self.target;
        let compiler = self.compiler;
        builder.info(&format!("Documenting standalone ({})", target));
        let out = builder.doc_out(target);
        t!(fs::create_dir_all(&out));

        let favicon = builder.src.join("src/doc/favicon.inc");
        let footer = builder.src.join("src/doc/footer.inc");
        let full_toc = builder.src.join("src/doc/full-toc.inc");
        t!(fs::copy(builder.src.join("src/doc/rust.css"), out.join("rust.css")));

        let version_input = builder.src.join("src/doc/version_info.html.template");
        let version_info = out.join("version_info.html");

        if !builder.config.dry_run && !up_to_date(&version_input, &version_info) {
            let info = t!(fs::read_to_string(&version_input))
                .replace("VERSION", &builder.rust_release())
                .replace("SHORT_HASH", builder.rust_info.sha_short().unwrap_or(""))
                .replace("STAMP", builder.rust_info.sha().unwrap_or(""));
            t!(fs::write(&version_info, &info));
        }

        for file in t!(fs::read_dir(builder.src.join("src/doc"))) {
            let file = t!(file);
            let path = file.path();
            let filename = path.file_name().unwrap().to_str().unwrap();
            if !filename.ends_with(".md") || filename == "README.md" {
                continue
            }

            let html = out.join(filename).with_extension("html");
            let rustdoc = builder.rustdoc(compiler);
            if up_to_date(&path, &html) &&
               up_to_date(&footer, &html) &&
               up_to_date(&favicon, &html) &&
               up_to_date(&full_toc, &html) &&
               up_to_date(&version_info, &html) &&
               (builder.config.dry_run || up_to_date(&rustdoc, &html)) {
                continue
            }

            let mut cmd = builder.rustdoc_cmd(compiler);
            cmd.arg("--html-after-content").arg(&footer)
               .arg("--html-before-content").arg(&version_info)
               .arg("--html-in-header").arg(&favicon)
               .arg("--markdown-no-toc")
               .arg("--index-page").arg(&builder.src.join("src/doc/index.md"))
               .arg("--markdown-playground-url").arg("https://play.rust-lang.org/")
               .arg("-o").arg(&out)
               .arg(&path);

            if filename == "not_found.md" {
                cmd.arg("--markdown-css")
                   .arg("https://doc.rust-lang.org/rust.css");
            } else {
                cmd.arg("--markdown-css").arg("rust.css");
            }
            builder.run(&mut cmd);
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Std {
    pub stage: u32,
    pub target: Interned<String>,
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.all_krates("std").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Std {
            stage: run.builder.top_stage,
            target: run.target
        });
    }

    /// Compile all standard library documentation.
    ///
    /// This will generate all documentation for the standard library and its
    /// dependencies. This is largely just a wrapper around `cargo doc`.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let target = self.target;
        builder.info(&format!("Documenting stage{} std ({})", stage, target));
        let out = builder.doc_out(target);
        t!(fs::create_dir_all(&out));
        let compiler = builder.compiler_for(stage, builder.config.build, target);

        builder.ensure(compile::Std { compiler, target });
        let out_dir = builder.stage_out(compiler, Mode::Std)
                           .join(target).join("doc");

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
        let my_out = builder.crate_doc_out(target);
        t!(symlink_dir_force(&builder.config, &my_out, &out_dir));
        t!(fs::copy(builder.src.join("src/doc/rust.css"), out.join("rust.css")));

        let run_cargo_rustdoc_for = |package: &str| {
            let mut cargo = builder.cargo(compiler, Mode::Std, target, "rustdoc");
            compile::std_cargo(builder, &compiler, target, &mut cargo);

            // Keep a whitelist so we do not build internal stdlib crates, these will be
            // build by the rustc step later if enabled.
            cargo.arg("-Z").arg("unstable-options")
                 .arg("-p").arg(package);
            // Create all crate output directories first to make sure rustdoc uses
            // relative links.
            // FIXME: Cargo should probably do this itself.
            t!(fs::create_dir_all(out_dir.join(package)));
            cargo.arg("--")
                 .arg("--markdown-css").arg("rust.css")
                 .arg("--markdown-no-toc")
                 .arg("--generate-redirect-pages")
                 .arg("--resource-suffix").arg(crate::channel::CFG_RELEASE_NUM)
                 .arg("--index-page").arg(&builder.src.join("src/doc/index.md"));

            builder.run(&mut cargo);
            builder.cp_r(&my_out, &out);
        };
        for krate in &["alloc", "core", "std"] {
            run_cargo_rustdoc_for(krate);
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Test {
    stage: u32,
    target: Interned<String>,
}

impl Step for Test {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.krate("test").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Test {
            stage: run.builder.top_stage,
            target: run.target,
        });
    }

    /// Compile all libtest documentation.
    ///
    /// This will generate all documentation for libtest and its dependencies. This
    /// is largely just a wrapper around `cargo doc`.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let target = self.target;
        builder.info(&format!("Documenting stage{} test ({})", stage, target));
        let out = builder.doc_out(target);
        t!(fs::create_dir_all(&out));
        let compiler = builder.compiler_for(stage, builder.config.build, target);

        // Build libstd docs so that we generate relative links
        builder.ensure(Std { stage, target });

        builder.ensure(compile::Test { compiler, target });
        let out_dir = builder.stage_out(compiler, Mode::Test)
                           .join(target).join("doc");

        // See docs in std above for why we symlink
        let my_out = builder.crate_doc_out(target);
        t!(symlink_dir_force(&builder.config, &my_out, &out_dir));

        let mut cargo = builder.cargo(compiler, Mode::Test, target, "doc");
        compile::test_cargo(builder, &compiler, target, &mut cargo);

        cargo.arg("--no-deps")
             .arg("-p").arg("test")
             .env("RUSTDOC_RESOURCE_SUFFIX", crate::channel::CFG_RELEASE_NUM)
             .env("RUSTDOC_GENERATE_REDIRECT_PAGES", "1");

        builder.run(&mut cargo);
        builder.cp_r(&my_out, &out);
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct WhitelistedRustc {
    stage: u32,
    target: Interned<String>,
}

impl Step for WhitelistedRustc {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.krate("rustc-main").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(WhitelistedRustc {
            stage: run.builder.top_stage,
            target: run.target,
        });
    }

    /// Generates whitelisted compiler crate documentation.
    ///
    /// This will generate all documentation for crates that are whitelisted
    /// to be included in the standard documentation. This documentation is
    /// included in the standard Rust documentation, so we should always
    /// document it and symlink to merge with the rest of the std and test
    /// documentation. We don't build other compiler documentation
    /// here as we want to be able to keep it separate from the standard
    /// documentation. This is largely just a wrapper around `cargo doc`.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let target = self.target;
        builder.info(&format!("Documenting stage{} whitelisted compiler ({})", stage, target));
        let out = builder.doc_out(target);
        t!(fs::create_dir_all(&out));
        let compiler = builder.compiler_for(stage, builder.config.build, target);

        // Build libstd docs so that we generate relative links
        builder.ensure(Std { stage, target });

        builder.ensure(compile::Rustc { compiler, target });
        let out_dir = builder.stage_out(compiler, Mode::Rustc)
                           .join(target).join("doc");

        // See docs in std above for why we symlink
        let my_out = builder.crate_doc_out(target);
        t!(symlink_dir_force(&builder.config, &my_out, &out_dir));

        let mut cargo = builder.cargo(compiler, Mode::Rustc, target, "doc");
        compile::rustc_cargo(builder, &mut cargo);

        // We don't want to build docs for internal compiler dependencies in this
        // step (there is another step for that). Therefore, we whitelist the crates
        // for which docs must be built.
        for krate in &["proc_macro"] {
            cargo.arg("-p").arg(krate)
                 .env("RUSTDOC_RESOURCE_SUFFIX", crate::channel::CFG_RELEASE_NUM)
                 .env("RUSTDOC_GENERATE_REDIRECT_PAGES", "1");
        }

        builder.run(&mut cargo);
        builder.cp_r(&my_out, &out);
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Rustc {
    stage: u32,
    target: Interned<String>,
}

impl Step for Rustc {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.krate("rustc-main").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rustc {
            stage: run.builder.top_stage,
            target: run.target,
        });
    }

    /// Generates compiler documentation.
    ///
    /// This will generate all documentation for compiler and dependencies.
    /// Compiler documentation is distributed separately, so we make sure
    /// we do not merge it with the other documentation from std, test and
    /// proc_macros. This is largely just a wrapper around `cargo doc`.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let target = self.target;
        builder.info(&format!("Documenting stage{} compiler ({})", stage, target));

        // This is the intended out directory for compiler documentation.
        let out = builder.compiler_doc_out(target);
        t!(fs::create_dir_all(&out));

        // Get the correct compiler for this stage.
        let compiler = builder.compiler_for(stage, builder.config.build, target);

        if !builder.config.compiler_docs {
            builder.info("\tskipping - compiler/librustdoc docs disabled");
            return;
        }

        // Build rustc.
        builder.ensure(compile::Rustc { compiler, target });

        // We do not symlink to the same shared folder that already contains std library
        // documentation from previous steps as we do not want to include that.
        let out_dir = builder.stage_out(compiler, Mode::Rustc).join(target).join("doc");
        t!(symlink_dir_force(&builder.config, &out, &out_dir));

        // Build cargo command.
        let mut cargo = builder.cargo(compiler, Mode::Rustc, target, "doc");
        cargo.env("RUSTDOCFLAGS", "--document-private-items --passes strip-hidden");
        compile::rustc_cargo(builder, &mut cargo);

        // Only include compiler crates, no dependencies of those, such as `libc`.
        cargo.arg("--no-deps");

        // Find dependencies for top level crates.
        let mut compiler_crates = HashSet::new();
        for root_crate in &["rustc_driver", "rustc_codegen_llvm", "rustc_codegen_ssa"] {
            let interned_root_crate = INTERNER.intern_str(root_crate);
            find_compiler_crates(builder, &interned_root_crate, &mut compiler_crates);
        }

        for krate in &compiler_crates {
            // Create all crate output directories first to make sure rustdoc uses
            // relative links.
            // FIXME: Cargo should probably do this itself.
            t!(fs::create_dir_all(out_dir.join(krate)));
            cargo.arg("-p").arg(krate);
        }

        builder.run(&mut cargo);
    }
}

fn find_compiler_crates(
    builder: &Builder<'_>,
    name: &Interned<String>,
    crates: &mut HashSet<Interned<String>>
) {
    // Add current crate.
    crates.insert(*name);

    // Look for dependencies.
    for dep in builder.crates.get(name).unwrap().deps.iter() {
        if builder.crates.get(dep).unwrap().is_local(builder) {
            find_compiler_crates(builder, dep, crates);
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Rustdoc {
    stage: u32,
    target: Interned<String>,
}

impl Step for Rustdoc {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.krate("rustdoc-tool")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rustdoc {
            stage: run.builder.top_stage,
            target: run.target,
        });
    }

    /// Generates compiler documentation.
    ///
    /// This will generate all documentation for compiler and dependencies.
    /// Compiler documentation is distributed separately, so we make sure
    /// we do not merge it with the other documentation from std, test and
    /// proc_macros. This is largely just a wrapper around `cargo doc`.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let target = self.target;
        builder.info(&format!("Documenting stage{} rustdoc ({})", stage, target));

        // This is the intended out directory for compiler documentation.
        let out = builder.compiler_doc_out(target);
        t!(fs::create_dir_all(&out));

        // Get the correct compiler for this stage.
        let compiler = builder.compiler_for(stage, builder.config.build, target);

        if !builder.config.compiler_docs {
            builder.info("\tskipping - compiler/librustdoc docs disabled");
            return;
        }

        // Build rustc docs so that we generate relative links.
        builder.ensure(Rustc { stage, target });

        // Build rustdoc.
        builder.ensure(tool::Rustdoc { compiler: compiler });

        // Symlink compiler docs to the output directory of rustdoc documentation.
        let out_dir = builder.stage_out(compiler, Mode::ToolRustc)
            .join(target)
            .join("doc");
        t!(fs::create_dir_all(&out_dir));
        t!(symlink_dir_force(&builder.config, &out, &out_dir));

        // Build cargo command.
        let mut cargo = prepare_tool_cargo(
            builder,
            compiler,
            Mode::ToolRustc,
            target,
            "doc",
            "src/tools/rustdoc",
            SourceType::InTree,
            &[]
        );

        // Only include compiler crates, no dependencies of those, such as `libc`.
        cargo.arg("--no-deps");
        cargo.arg("-p").arg("rustdoc");

        cargo.env("RUSTDOCFLAGS", "--document-private-items");
        builder.run(&mut cargo);
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct ErrorIndex {
    target: Interned<String>,
}

impl Step for ErrorIndex {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/tools/error_index_generator").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(ErrorIndex {
            target: run.target,
        });
    }

    /// Generates the HTML rendered error-index by running the
    /// `error_index_generator` tool.
    fn run(self, builder: &Builder<'_>) {
        let target = self.target;

        builder.info(&format!("Documenting error index ({})", target));
        let out = builder.doc_out(target);
        t!(fs::create_dir_all(&out));
        let compiler = builder.compiler(2, builder.config.build);
        let mut index = tool::ErrorIndex::command(
            builder,
            compiler,
        );
        index.arg("html");
        index.arg(out.join("error-index.html"));
        index.arg(crate::channel::CFG_RELEASE_NUM);

        // FIXME: shouldn't have to pass this env var
        index.env("CFG_BUILD", &builder.config.build)
             .env("RUSTC_ERROR_METADATA_DST", builder.extended_error_dir());

        builder.run(&mut index);
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct UnstableBookGen {
    target: Interned<String>,
}

impl Step for UnstableBookGen {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/tools/unstable-book-gen").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(UnstableBookGen {
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        let target = self.target;

        builder.info(&format!("Generating unstable book md files ({})", target));
        let out = builder.md_doc_out(target).join("unstable-book");
        builder.create_dir(&out);
        builder.remove_dir(&out);
        let mut cmd = builder.tool_cmd(Tool::UnstableBookGen);
        cmd.arg(builder.src.join("src"));
        cmd.arg(out);

        builder.run(&mut cmd);
    }
}

fn symlink_dir_force(config: &Config, src: &Path, dst: &Path) -> io::Result<()> {
    if config.dry_run {
        return Ok(());
    }
    if let Ok(m) = fs::symlink_metadata(dst) {
        if m.file_type().is_dir() {
            fs::remove_dir_all(dst)?;
        } else {
            // handle directory junctions on windows by falling back to
            // `remove_dir`.
            fs::remove_file(dst).or_else(|_| {
                fs::remove_dir(dst)
            })?;
        }
    }

    symlink_dir(config, src, dst)
}
