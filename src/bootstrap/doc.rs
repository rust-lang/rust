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
use std::path::{Path, PathBuf};

use crate::Mode;
use build_helper::{t, up_to_date};

use crate::builder::{Builder, Compiler, RunConfig, ShouldRun, Step};
use crate::cache::{Interned, INTERNER};
use crate::compile;
use crate::config::{Config, TargetSelection};
use crate::tool::{self, prepare_tool_cargo, SourceType, Tool};
use crate::util::symlink_dir;

macro_rules! submodule_helper {
    ($path:expr, submodule) => {
        $path
    };
    ($path:expr, submodule = $submodule:literal) => {
        $submodule
    };
}

macro_rules! book {
    ($($name:ident, $path:expr, $book_name:expr $(, submodule $(= $submodule:literal)? )? ;)+) => {
        $(
            #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            target: TargetSelection,
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
                $(
                    let path = Path::new(submodule_helper!( $path, submodule $( = $submodule )? ));
                    builder.update_submodule(&path);
                )?
                builder.ensure(RustbookSrc {
                    target: self.target,
                    name: INTERNER.intern_str($book_name),
                    src: INTERNER.intern_path(builder.src.join($path)),
                })
            }
        }
        )+
    }
}

// NOTE: When adding a book here, make sure to ALSO build the book by
// adding a build step in `src/bootstrap/builder.rs`!
// NOTE: Make sure to add the corresponding submodule when adding a new book.
// FIXME: Make checking for a submodule automatic somehow (maybe by having a list of all submodules
// and checking against it?).
book!(
    CargoBook, "src/tools/cargo/src/doc", "cargo", submodule = "src/tools/cargo";
    EditionGuide, "src/doc/edition-guide", "edition-guide", submodule;
    EmbeddedBook, "src/doc/embedded-book", "embedded-book", submodule;
    Nomicon, "src/doc/nomicon", "nomicon", submodule;
    Reference, "src/doc/reference", "reference", submodule;
    RustByExample, "src/doc/rust-by-example", "rust-by-example", submodule;
    RustdocBook, "src/doc/rustdoc", "rustdoc";
);

fn open(builder: &Builder<'_>, path: impl AsRef<Path>) {
    if builder.config.dry_run || !builder.config.cmd.open() {
        return;
    }

    let path = path.as_ref();
    builder.info(&format!("Opening doc {}", path.display()));
    if let Err(err) = opener::open(path) {
        builder.info(&format!("{}\n", err));
    }
}

// "library/std" -> ["library", "std"]
//
// Used for deciding whether a particular step is one requested by the user on
// the `x.py doc` command line, which determines whether `--open` will open that
// page.
fn components_simplified(path: &PathBuf) -> Vec<&str> {
    path.iter().map(|component| component.to_str().unwrap_or("???")).collect()
}

fn is_explicit_request(builder: &Builder<'_>, path: &str) -> bool {
    builder
        .paths
        .iter()
        .map(components_simplified)
        .any(|requested| requested.iter().copied().eq(path.split('/')))
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct UnstableBook {
    target: TargetSelection,
}

impl Step for UnstableBook {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/doc/unstable-book").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(UnstableBook { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        builder.ensure(UnstableBookGen { target: self.target });
        builder.ensure(RustbookSrc {
            target: self.target,
            name: INTERNER.intern_str("unstable-book"),
            src: INTERNER.intern_path(builder.md_doc_out(self.target).join("unstable-book")),
        })
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
struct RustbookSrc {
    target: TargetSelection,
    name: Interned<String>,
    src: Interned<PathBuf>,
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
        let index = out.join("index.html");
        let rustbook = builder.tool_exe(Tool::Rustbook);
        let mut rustbook_cmd = builder.tool_cmd(Tool::Rustbook);
        if builder.config.dry_run || up_to_date(&src, &index) && up_to_date(&rustbook, &index) {
            return;
        }
        builder.info(&format!("Rustbook ({}) - {}", target, name));
        let _ = fs::remove_dir_all(&out);

        builder.run(rustbook_cmd.arg("build").arg(&src).arg("-d").arg(out));
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct TheBook {
    compiler: Compiler,
    target: TargetSelection,
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
        });
    }

    /// Builds the book and associated stuff.
    ///
    /// We need to build:
    ///
    /// * Book
    /// * Older edition redirects
    /// * Version info and CSS
    /// * Index page
    /// * Redirect pages
    fn run(self, builder: &Builder<'_>) {
        let relative_path = Path::new("src").join("doc").join("book");
        builder.update_submodule(&relative_path);

        let compiler = self.compiler;
        let target = self.target;

        // build book
        builder.ensure(RustbookSrc {
            target,
            name: INTERNER.intern_str("book"),
            src: INTERNER.intern_path(builder.src.join(&relative_path)),
        });

        // building older edition redirects
        for edition in &["first-edition", "second-edition", "2018-edition"] {
            builder.ensure(RustbookSrc {
                target,
                name: INTERNER.intern_string(format!("book/{}", edition)),
                src: INTERNER.intern_path(builder.src.join(&relative_path).join(edition)),
            });
        }

        // build the version info page and CSS
        builder.ensure(Standalone { compiler, target });

        // build the redirect pages
        builder.info(&format!("Documenting book redirect pages ({})", target));
        for file in t!(fs::read_dir(builder.src.join(&relative_path).join("redirects"))) {
            let file = t!(file);
            let path = file.path();
            let path = path.to_str().unwrap();

            invoke_rustdoc(builder, compiler, target, path);
        }

        if is_explicit_request(builder, "src/doc/book") {
            let out = builder.doc_out(target);
            let index = out.join("book").join("index.html");
            open(builder, &index);
        }
    }
}

fn invoke_rustdoc(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: TargetSelection,
    markdown: &str,
) {
    let out = builder.doc_out(target);

    let path = builder.src.join("src/doc").join(markdown);

    let header = builder.src.join("src/doc/redirect.inc");
    let footer = builder.src.join("src/doc/footer.inc");
    let version_info = out.join("version_info.html");

    let mut cmd = builder.rustdoc_cmd(compiler);

    let out = out.join("book");

    cmd.arg("--html-after-content")
        .arg(&footer)
        .arg("--html-before-content")
        .arg(&version_info)
        .arg("--html-in-header")
        .arg(&header)
        .arg("--markdown-no-toc")
        .arg("--markdown-playground-url")
        .arg("https://play.rust-lang.org/")
        .arg("-o")
        .arg(&out)
        .arg(&path)
        .arg("--markdown-css")
        .arg("../rust.css");

    if !builder.config.docs_minification {
        cmd.arg("-Z").arg("unstable-options").arg("--disable-minification");
    }

    builder.run(&mut cmd);
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Standalone {
    compiler: Compiler,
    target: TargetSelection,
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
                continue;
            }

            let html = out.join(filename).with_extension("html");
            let rustdoc = builder.rustdoc(compiler);
            if up_to_date(&path, &html)
                && up_to_date(&footer, &html)
                && up_to_date(&favicon, &html)
                && up_to_date(&full_toc, &html)
                && (builder.config.dry_run || up_to_date(&version_info, &html))
                && (builder.config.dry_run || up_to_date(&rustdoc, &html))
            {
                continue;
            }

            let mut cmd = builder.rustdoc_cmd(compiler);
            // Needed for --index-page flag
            cmd.arg("-Z").arg("unstable-options");

            cmd.arg("--html-after-content")
                .arg(&footer)
                .arg("--html-before-content")
                .arg(&version_info)
                .arg("--html-in-header")
                .arg(&favicon)
                .arg("--markdown-no-toc")
                .arg("--index-page")
                .arg(&builder.src.join("src/doc/index.md"))
                .arg("--markdown-playground-url")
                .arg("https://play.rust-lang.org/")
                .arg("-o")
                .arg(&out)
                .arg(&path);

            if !builder.config.docs_minification {
                cmd.arg("--disable-minification");
            }

            if filename == "not_found.md" {
                cmd.arg("--markdown-css")
                    .arg(format!("https://doc.rust-lang.org/rustdoc{}.css", &builder.version))
                    .arg("--markdown-css")
                    .arg("https://doc.rust-lang.org/rust.css");
            } else {
                cmd.arg("--markdown-css")
                    .arg(format!("rustdoc{}.css", &builder.version))
                    .arg("--markdown-css")
                    .arg("rust.css");
            }
            builder.run(&mut cmd);
        }

        // We open doc/index.html as the default if invoked as `x.py doc --open`
        // with no particular explicit doc requested (e.g. library/core).
        if builder.paths.is_empty() || is_explicit_request(builder, "src/doc") {
            let index = out.join("index.html");
            open(builder, &index);
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Std {
    pub stage: u32,
    pub target: TargetSelection,
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.all_krates("test").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Std { stage: run.builder.top_stage, target: run.target });
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
        let compiler = builder.compiler(stage, builder.config.build);

        builder.ensure(compile::Std { compiler, target });
        let out_dir = builder.stage_out(compiler, Mode::Std).join(target.triple).join("doc");

        t!(fs::copy(builder.src.join("src/doc/rust.css"), out.join("rust.css")));

        let run_cargo_rustdoc_for = |package: &str| {
            let mut cargo =
                builder.cargo(compiler, Mode::Std, SourceType::InTree, target, "rustdoc");
            compile::std_cargo(builder, target, compiler.stage, &mut cargo);

            cargo
                .arg("-p")
                .arg(package)
                .arg("-Zskip-rustdoc-fingerprint")
                .arg("--")
                .arg("--markdown-css")
                .arg("rust.css")
                .arg("--markdown-no-toc")
                .arg("-Z")
                .arg("unstable-options")
                .arg("--resource-suffix")
                .arg(&builder.version)
                .arg("--index-page")
                .arg(&builder.src.join("src/doc/index.md"));

            if !builder.config.docs_minification {
                cargo.arg("--disable-minification");
            }

            builder.run(&mut cargo.into());
        };

        let paths = builder
            .paths
            .iter()
            .map(components_simplified)
            .filter_map(|path| {
                if path.get(0) == Some(&"library") {
                    Some(path[1].to_owned())
                } else if !path.is_empty() {
                    Some(path[0].to_owned())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        // Only build the following crates. While we could just iterate over the
        // folder structure, that would also build internal crates that we do
        // not want to show in documentation. These crates will later be visited
        // by the rustc step, so internal documentation will show them.
        //
        // Note that the order here is important! The crates need to be
        // processed starting from the leaves, otherwise rustdoc will not
        // create correct links between crates because rustdoc depends on the
        // existence of the output directories to know if it should be a local
        // or remote link.
        let krates = ["core", "alloc", "std", "proc_macro", "test"];
        for krate in &krates {
            run_cargo_rustdoc_for(krate);
            if paths.iter().any(|p| p == krate) {
                // No need to document more of the libraries if we have the one we want.
                break;
            }
        }
        builder.cp_r(&out_dir, &out);

        // Look for library/std, library/core etc in the `x.py doc` arguments and
        // open the corresponding rendered docs.
        for requested_crate in paths {
            if krates.iter().any(|k| *k == requested_crate.as_str()) {
                let index = out.join(requested_crate).join("index.html");
                open(builder, &index);
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Rustc {
    pub stage: u32,
    pub target: TargetSelection,
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
        run.builder.ensure(Rustc { stage: run.builder.top_stage, target: run.target });
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

        if !builder.config.compiler_docs {
            builder.info("\tskipping - compiler/librustdoc docs disabled");
            return;
        }

        // This is the intended out directory for compiler documentation.
        let out = builder.compiler_doc_out(target);
        t!(fs::create_dir_all(&out));

        // Build rustc.
        let compiler = builder.compiler(stage, builder.config.build);
        builder.ensure(compile::Rustc { compiler, target });

        // This uses a shared directory so that librustdoc documentation gets
        // correctly built and merged with the rustc documentation. This is
        // needed because rustdoc is built in a different directory from
        // rustc. rustdoc needs to be able to see everything, for example when
        // merging the search index, or generating local (relative) links.
        let out_dir = builder.stage_out(compiler, Mode::Rustc).join(target.triple).join("doc");
        t!(symlink_dir_force(&builder.config, &out, &out_dir));
        // Cargo puts proc macros in `target/doc` even if you pass `--target`
        // explicitly (https://github.com/rust-lang/cargo/issues/7677).
        let proc_macro_out_dir = builder.stage_out(compiler, Mode::Rustc).join("doc");
        t!(symlink_dir_force(&builder.config, &out, &proc_macro_out_dir));

        // Build cargo command.
        let mut cargo = builder.cargo(compiler, Mode::Rustc, SourceType::InTree, target, "doc");
        cargo.rustdocflag("--document-private-items");
        // Since we always pass --document-private-items, there's no need to warn about linking to private items.
        cargo.rustdocflag("-Arustdoc::private-intra-doc-links");
        cargo.rustdocflag("--enable-index-page");
        cargo.rustdocflag("-Zunstable-options");
        cargo.rustdocflag("-Znormalize-docs");
        cargo.rustdocflag("--show-type-layout");
        compile::rustc_cargo(builder, &mut cargo, target);
        cargo.arg("-Zunstable-options");
        cargo.arg("-Zskip-rustdoc-fingerprint");

        // Only include compiler crates, no dependencies of those, such as `libc`.
        // Do link to dependencies on `docs.rs` however using `rustdoc-map`.
        cargo.arg("--no-deps");
        cargo.arg("-Zrustdoc-map");

        // FIXME: `-Zrustdoc-map` does not yet correctly work for transitive dependencies,
        // once this is no longer an issue the special case for `ena` can be removed.
        cargo.rustdocflag("--extern-html-root-url");
        cargo.rustdocflag("ena=https://docs.rs/ena/latest/");

        // Find dependencies for top level crates.
        let mut compiler_crates = HashSet::new();
        for root_crate in &["rustc_driver", "rustc_codegen_llvm", "rustc_codegen_ssa"] {
            compiler_crates.extend(
                builder
                    .in_tree_crates(root_crate, Some(target))
                    .into_iter()
                    .map(|krate| krate.name),
            );
        }

        for krate in &compiler_crates {
            // Create all crate output directories first to make sure rustdoc uses
            // relative links.
            // FIXME: Cargo should probably do this itself.
            t!(fs::create_dir_all(out_dir.join(krate)));
            cargo.arg("-p").arg(krate);
        }

        builder.run(&mut cargo.into());
    }
}

macro_rules! tool_doc {
    ($tool: ident, $should_run: literal, $path: literal, [$($krate: literal),+ $(,)?] $(,)?) => {
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        pub struct $tool {
            stage: u32,
            target: TargetSelection,
        }

        impl Step for $tool {
            type Output = ();
            const DEFAULT: bool = true;
            const ONLY_HOSTS: bool = true;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.krate($should_run)
            }

            fn make_run(run: RunConfig<'_>) {
                run.builder.ensure($tool { stage: run.builder.top_stage, target: run.target });
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
                builder.info(&format!("Documenting stage{} {} ({})", stage, stringify!($tool).to_lowercase(), target));

                // This is the intended out directory for compiler documentation.
                let out = builder.compiler_doc_out(target);
                t!(fs::create_dir_all(&out));

                let compiler = builder.compiler(stage, builder.config.build);

                if !builder.config.compiler_docs {
                    builder.info("\tskipping - compiler/tool docs disabled");
                    return;
                }

                // Build rustc docs so that we generate relative links.
                builder.ensure(Rustc { stage, target });

                // Symlink compiler docs to the output directory of rustdoc documentation.
                let out_dir = builder.stage_out(compiler, Mode::ToolRustc).join(target.triple).join("doc");
                t!(fs::create_dir_all(&out_dir));
                t!(symlink_dir_force(&builder.config, &out, &out_dir));

                // Build cargo command.
                let mut cargo = prepare_tool_cargo(
                    builder,
                    compiler,
                    Mode::ToolRustc,
                    target,
                    "doc",
                    $path,
                    SourceType::InTree,
                    &[],
                );

                cargo.arg("-Zskip-rustdoc-fingerprint");
                // Only include compiler crates, no dependencies of those, such as `libc`.
                cargo.arg("--no-deps");
                $(
                    cargo.arg("-p").arg($krate);
                )+

                cargo.rustdocflag("--document-private-items");
                cargo.rustdocflag("--enable-index-page");
                cargo.rustdocflag("--show-type-layout");
                cargo.rustdocflag("-Zunstable-options");
                builder.run(&mut cargo.into());
            }
        }
    }
}

tool_doc!(Rustdoc, "rustdoc-tool", "src/tools/rustdoc", ["rustdoc", "rustdoc-json-types"]);
tool_doc!(
    Rustfmt,
    "rustfmt-nightly",
    "src/tools/rustfmt",
    ["rustfmt-nightly", "rustfmt-config_proc_macro"],
);

#[derive(Ord, PartialOrd, Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct ErrorIndex {
    pub target: TargetSelection,
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
        let target = run.target;
        run.builder.ensure(ErrorIndex { target });
    }

    /// Generates the HTML rendered error-index by running the
    /// `error_index_generator` tool.
    fn run(self, builder: &Builder<'_>) {
        builder.info(&format!("Documenting error index ({})", self.target));
        let out = builder.doc_out(self.target);
        t!(fs::create_dir_all(&out));
        let mut index = tool::ErrorIndex::command(builder);
        index.arg("html");
        index.arg(out.join("error-index.html"));
        index.arg(&builder.version);

        builder.run(&mut index);
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct UnstableBookGen {
    target: TargetSelection,
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
        run.builder.ensure(UnstableBookGen { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) {
        let target = self.target;

        builder.info(&format!("Generating unstable book md files ({})", target));
        let out = builder.md_doc_out(target).join("unstable-book");
        builder.create_dir(&out);
        builder.remove_dir(&out);
        let mut cmd = builder.tool_cmd(Tool::UnstableBookGen);
        cmd.arg(builder.src.join("library"));
        cmd.arg(builder.src.join("compiler"));
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
            fs::remove_file(dst).or_else(|_| fs::remove_dir(dst))?;
        }
    }

    symlink_dir(config, src, dst)
}

#[derive(Ord, PartialOrd, Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct RustcBook {
    pub compiler: Compiler,
    pub target: TargetSelection,
    pub validate: bool,
}

impl Step for RustcBook {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/doc/rustc").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustcBook {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
            validate: false,
        });
    }

    /// Builds the rustc book.
    ///
    /// The lints are auto-generated by a tool, and then merged into the book
    /// in the "md-doc" directory in the build output directory. Then
    /// "rustbook" is used to convert it to HTML.
    fn run(self, builder: &Builder<'_>) {
        let out_base = builder.md_doc_out(self.target).join("rustc");
        t!(fs::create_dir_all(&out_base));
        let out_listing = out_base.join("src/lints");
        builder.cp_r(&builder.src.join("src/doc/rustc"), &out_base);
        builder.info(&format!("Generating lint docs ({})", self.target));

        let rustc = builder.rustc(self.compiler);
        // The tool runs `rustc` for extracting output examples, so it needs a
        // functional sysroot.
        builder.ensure(compile::Std { compiler: self.compiler, target: self.target });
        let mut cmd = builder.tool_cmd(Tool::LintDocs);
        cmd.arg("--src");
        cmd.arg(builder.src.join("compiler"));
        cmd.arg("--out");
        cmd.arg(&out_listing);
        cmd.arg("--rustc");
        cmd.arg(&rustc);
        cmd.arg("--rustc-target").arg(&self.target.rustc_target_arg());
        if builder.config.verbose() {
            cmd.arg("--verbose");
        }
        if self.validate {
            cmd.arg("--validate");
        }
        // If the lib directories are in an unusual location (changed in
        // config.toml), then this needs to explicitly update the dylib search
        // path.
        builder.add_rustc_lib_path(self.compiler, &mut cmd);
        builder.run(&mut cmd);
        // Run rustbook/mdbook to generate the HTML pages.
        builder.ensure(RustbookSrc {
            target: self.target,
            name: INTERNER.intern_str("rustc"),
            src: INTERNER.intern_path(out_base),
        });
        if is_explicit_request(builder, "src/doc/rustc") {
            let out = builder.doc_out(self.target);
            let index = out.join("rustc").join("index.html");
            open(builder, &index);
        }
    }
}
