//! Documentation generation for rustbuilder.
//!
//! This module implements generation for all bits and pieces of documentation
//! for the Rust project. This notably includes suites like the rust book, the
//! nomicon, rust by example, standalone documentation, etc.
//!
//! Everything here is basically just a shim around calling either `rustbook` or
//! `rustdoc`.

use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::builder::crate_description;
use crate::builder::{Builder, Compiler, Kind, RunConfig, ShouldRun, Step};
use crate::cache::{Interned, INTERNER};
use crate::compile;
use crate::config::{Config, TargetSelection};
use crate::tool::{self, prepare_tool_cargo, SourceType, Tool};
use crate::util::{symlink_dir, t, up_to_date};
use crate::Mode;

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
                    parent: Some(self),
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
    ClippyBook, "src/tools/clippy/book", "clippy";
    EditionGuide, "src/doc/edition-guide", "edition-guide", submodule;
    EmbeddedBook, "src/doc/embedded-book", "embedded-book", submodule;
    Nomicon, "src/doc/nomicon", "nomicon", submodule;
    Reference, "src/doc/reference", "reference", submodule;
    RustByExample, "src/doc/rust-by-example", "rust-by-example", submodule;
    RustdocBook, "src/doc/rustdoc", "rustdoc";
    StyleGuide, "src/doc/style-guide", "style-guide";
);

// "library/std" -> ["library", "std"]
//
// Used for deciding whether a particular step is one requested by the user on
// the `x.py doc` command line, which determines whether `--open` will open that
// page.
pub(crate) fn components_simplified(path: &PathBuf) -> Vec<&str> {
    path.iter().map(|component| component.to_str().unwrap_or("???")).collect()
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
            parent: Some(self),
        })
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
struct RustbookSrc<P: Step> {
    target: TargetSelection,
    name: Interned<String>,
    src: Interned<PathBuf>,
    parent: Option<P>,
}

impl<P: Step> Step for RustbookSrc<P> {
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

        if !builder.config.dry_run() && !(up_to_date(&src, &index) || up_to_date(&rustbook, &index))
        {
            builder.info(&format!("Rustbook ({}) - {}", target, name));
            let _ = fs::remove_dir_all(&out);

            builder.run(rustbook_cmd.arg("build").arg(&src).arg("-d").arg(out));
        }

        if self.parent.is_some() {
            builder.maybe_open_in_browser::<P>(index)
        }
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
            parent: Some(self),
        });

        // building older edition redirects
        for edition in &["first-edition", "second-edition", "2018-edition"] {
            builder.ensure(RustbookSrc {
                target,
                name: INTERNER.intern_string(format!("book/{}", edition)),
                src: INTERNER.intern_path(builder.src.join(&relative_path).join(edition)),
                // There should only be one book that is marked as the parent for each target, so
                // treat the other editions as not having a parent.
                parent: Option::<Self>::None,
            });
        }

        // build the version info page and CSS
        let shared_assets = builder.ensure(SharedAssets { target });

        // build the redirect pages
        builder.info(&format!("Documenting book redirect pages ({})", target));
        for file in t!(fs::read_dir(builder.src.join(&relative_path).join("redirects"))) {
            let file = t!(file);
            let path = file.path();
            let path = path.to_str().unwrap();

            invoke_rustdoc(builder, compiler, &shared_assets, target, path);
        }
    }
}

fn invoke_rustdoc(
    builder: &Builder<'_>,
    compiler: Compiler,
    shared_assets: &SharedAssetsPaths,
    target: TargetSelection,
    markdown: &str,
) {
    let out = builder.doc_out(target);

    let path = builder.src.join("src/doc").join(markdown);

    let header = builder.src.join("src/doc/redirect.inc");
    let footer = builder.src.join("src/doc/footer.inc");

    let mut cmd = builder.rustdoc_cmd(compiler);

    let out = out.join("book");

    cmd.arg("--html-after-content")
        .arg(&footer)
        .arg("--html-before-content")
        .arg(&shared_assets.version_info)
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
        run.path("src/doc").alias("standalone").default_condition(builder.config.docs)
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

        let version_info = builder.ensure(SharedAssets { target: self.target }).version_info;

        let favicon = builder.src.join("src/doc/favicon.inc");
        let footer = builder.src.join("src/doc/footer.inc");
        let full_toc = builder.src.join("src/doc/full-toc.inc");

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
                && (builder.config.dry_run() || up_to_date(&version_info, &html))
                && (builder.config.dry_run() || up_to_date(&rustdoc, &html))
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
                cmd.arg("--markdown-css").arg("https://doc.rust-lang.org/rust.css");
            } else {
                cmd.arg("--markdown-css").arg("rust.css");
            }
            builder.run(&mut cmd);
        }

        // We open doc/index.html as the default if invoked as `x.py doc --open`
        // with no particular explicit doc requested (e.g. library/core).
        if builder.paths.is_empty() || builder.was_invoked_explicitly::<Self>(Kind::Doc) {
            let index = out.join("index.html");
            builder.open_in_browser(&index);
        }
    }
}

#[derive(Debug, Clone)]
pub struct SharedAssetsPaths {
    pub version_info: PathBuf,
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct SharedAssets {
    target: TargetSelection,
}

impl Step for SharedAssets {
    type Output = SharedAssetsPaths;
    const DEFAULT: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        // Other tasks depend on this, no need to execute it on its own
        run.never()
    }

    // Generate shared resources used by other pieces of documentation.
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let out = builder.doc_out(self.target);

        let version_input = builder.src.join("src").join("doc").join("version_info.html.template");
        let version_info = out.join("version_info.html");
        if !builder.config.dry_run() && !up_to_date(&version_input, &version_info) {
            let info = t!(fs::read_to_string(&version_input))
                .replace("VERSION", &builder.rust_release())
                .replace("SHORT_HASH", builder.rust_info().sha_short().unwrap_or(""))
                .replace("STAMP", builder.rust_info().sha().unwrap_or(""));
            t!(fs::write(&version_info, &info));
        }

        builder.copy(&builder.src.join("src").join("doc").join("rust.css"), &out.join("rust.css"));

        SharedAssetsPaths { version_info }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Std {
    pub stage: u32,
    pub target: TargetSelection,
    pub format: DocumentationFormat,
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.all_krates("test").path("library").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Std {
            stage: run.builder.top_stage,
            target: run.target,
            format: if run.builder.config.cmd.json() {
                DocumentationFormat::JSON
            } else {
                DocumentationFormat::HTML
            },
        });
    }

    /// Compile all standard library documentation.
    ///
    /// This will generate all documentation for the standard library and its
    /// dependencies. This is largely just a wrapper around `cargo doc`.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let target = self.target;
        let out = match self.format {
            DocumentationFormat::HTML => builder.doc_out(target),
            DocumentationFormat::JSON => builder.json_doc_out(target),
        };

        t!(fs::create_dir_all(&out));

        if self.format == DocumentationFormat::HTML {
            builder.ensure(SharedAssets { target: self.target });
        }

        let index_page = builder.src.join("src/doc/index.md").into_os_string();
        let mut extra_args = match self.format {
            DocumentationFormat::HTML => vec![
                OsStr::new("--markdown-css"),
                OsStr::new("rust.css"),
                OsStr::new("--markdown-no-toc"),
                OsStr::new("--index-page"),
                &index_page,
            ],
            DocumentationFormat::JSON => vec![OsStr::new("--output-format"), OsStr::new("json")],
        };

        if !builder.config.docs_minification {
            extra_args.push(OsStr::new("--disable-minification"));
        }

        let requested_crates = builder
            .paths
            .iter()
            .map(components_simplified)
            .filter_map(|path| {
                if path.len() >= 2 && path.get(0) == Some(&"library") {
                    // single crate
                    Some(path[1].to_owned())
                } else if !path.is_empty() {
                    // ??
                    Some(path[0].to_owned())
                } else {
                    // all library crates
                    None
                }
            })
            .collect::<Vec<_>>();

        doc_std(builder, self.format, stage, target, &out, &extra_args, &requested_crates);

        // Don't open if the format is json
        if let DocumentationFormat::JSON = self.format {
            return;
        }

        // Look for library/std, library/core etc in the `x.py doc` arguments and
        // open the corresponding rendered docs.
        for requested_crate in requested_crates {
            if requested_crate == "library" {
                // For `x.py doc library --open`, open `std` by default.
                let index = out.join("std").join("index.html");
                builder.open_in_browser(index);
            } else if STD_PUBLIC_CRATES.iter().any(|&k| k == requested_crate) {
                let index = out.join(requested_crate).join("index.html");
                builder.open_in_browser(index);
            }
        }
    }
}

/// Name of the crates that are visible to consumers of the standard library.
/// Documentation for internal crates is handled by the rustc step, so internal crates will show
/// up there.
///
/// Order here is important!
/// Crates need to be processed starting from the leaves, otherwise rustdoc will not
/// create correct links between crates because rustdoc depends on the
/// existence of the output directories to know if it should be a local
/// or remote link.
const STD_PUBLIC_CRATES: [&str; 5] = ["core", "alloc", "std", "proc_macro", "test"];

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum DocumentationFormat {
    HTML,
    JSON,
}

impl DocumentationFormat {
    fn as_str(&self) -> &str {
        match self {
            DocumentationFormat::HTML => "HTML",
            DocumentationFormat::JSON => "JSON",
        }
    }
}

/// Build the documentation for public standard library crates.
///
/// `requested_crates` can be used to build only a subset of the crates. If empty, all crates will
/// be built.
fn doc_std(
    builder: &Builder<'_>,
    format: DocumentationFormat,
    stage: u32,
    target: TargetSelection,
    out: &Path,
    extra_args: &[&OsStr],
    requested_crates: &[String],
) {
    builder.info(&format!(
        "Documenting{} stage{} library ({}) in {} format",
        crate_description(requested_crates),
        stage,
        target,
        format.as_str()
    ));
    if builder.no_std(target) == Some(true) {
        panic!(
            "building std documentation for no_std target {target} is not supported\n\
             Set `docs = false` in the config to disable documentation."
        );
    }
    let compiler = builder.compiler(stage, builder.config.build);

    let target_doc_dir_name = if format == DocumentationFormat::JSON { "json-doc" } else { "doc" };
    let target_dir =
        builder.stage_out(compiler, Mode::Std).join(target.triple).join(target_doc_dir_name);

    // This is directory where the compiler will place the output of the command.
    // We will then copy the files from this directory into the final `out` directory, the specified
    // as a function parameter.
    let out_dir = target_dir.join(target.triple).join("doc");

    let run_cargo_rustdoc_for = |package: &str| {
        let mut cargo = builder.cargo(compiler, Mode::Std, SourceType::InTree, target, "rustdoc");
        compile::std_cargo(builder, target, compiler.stage, &mut cargo);
        cargo
            .arg("--target-dir")
            .arg(&*target_dir.to_string_lossy())
            .arg("-p")
            .arg(package)
            .arg("-Zskip-rustdoc-fingerprint")
            .arg("--")
            .arg("-Z")
            .arg("unstable-options")
            .arg("--resource-suffix")
            .arg(&builder.version)
            .args(extra_args);
        if builder.config.library_docs_private_items {
            cargo.arg("--document-private-items").arg("--document-hidden-items");
        }
        builder.run(&mut cargo.into());
    };

    for krate in STD_PUBLIC_CRATES {
        run_cargo_rustdoc_for(krate);
        if requested_crates.iter().any(|p| p == krate) {
            // No need to document more of the libraries if we have the one we want.
            break;
        }
    }

    builder.cp_r(&out_dir, &out);
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
        run.crate_or_deps("rustc-main")
            .path("compiler")
            .default_condition(builder.config.compiler_docs)
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

        let paths = builder
            .paths
            .iter()
            .filter(|path| {
                let components = components_simplified(path);
                components.len() >= 2 && components[0] == "compiler"
            })
            .collect::<Vec<_>>();

        // This is the intended out directory for compiler documentation.
        let out = builder.compiler_doc_out(target);
        t!(fs::create_dir_all(&out));

        // Build the standard library, so that proc-macros can use it.
        // (Normally, only the metadata would be necessary, but proc-macros are special since they run at compile-time.)
        let compiler = builder.compiler(stage, builder.config.build);
        builder.ensure(compile::Std::new(compiler, builder.config.build));

        builder.info(&format!("Documenting stage{} compiler ({})", stage, target));

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
        cargo.rustdocflag("--generate-link-to-definition");
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

        let root_crates = if paths.is_empty() {
            vec![
                INTERNER.intern_str("rustc_driver"),
                INTERNER.intern_str("rustc_codegen_llvm"),
                INTERNER.intern_str("rustc_codegen_ssa"),
            ]
        } else {
            paths.into_iter().map(|p| builder.crate_paths[p]).collect()
        };
        // Find dependencies for top level crates.
        let compiler_crates = root_crates.iter().flat_map(|krate| {
            builder.in_tree_crates(krate, Some(target)).into_iter().map(|krate| krate.name)
        });

        let mut to_open = None;
        for krate in compiler_crates {
            // Create all crate output directories first to make sure rustdoc uses
            // relative links.
            // FIXME: Cargo should probably do this itself.
            t!(fs::create_dir_all(out_dir.join(krate)));
            cargo.arg("-p").arg(krate);
            if to_open.is_none() {
                to_open = Some(krate);
            }
        }

        builder.run(&mut cargo.into());
        // Let's open the first crate documentation page:
        if let Some(krate) = to_open {
            let index = out.join(krate).join("index.html");
            builder.open_in_browser(index);
        }
    }
}

macro_rules! tool_doc {
    ($tool: ident, $should_run: literal, $path: literal, $(rustc_tool = $rustc_tool:literal, )? $(in_tree = $in_tree:literal, )? [$($krate: literal),+ $(,)?] $(,)?) => {
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        pub struct $tool {
            target: TargetSelection,
        }

        impl Step for $tool {
            type Output = ();
            const DEFAULT: bool = true;
            const ONLY_HOSTS: bool = true;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                let builder = run.builder;
                run.crate_or_deps($should_run).default_condition(builder.config.compiler_docs)
            }

            fn make_run(run: RunConfig<'_>) {
                run.builder.ensure($tool { target: run.target });
            }

            /// Generates compiler documentation.
            ///
            /// This will generate all documentation for compiler and dependencies.
            /// Compiler documentation is distributed separately, so we make sure
            /// we do not merge it with the other documentation from std, test and
            /// proc_macros. This is largely just a wrapper around `cargo doc`.
            fn run(self, builder: &Builder<'_>) {
                let stage = builder.top_stage;
                let target = self.target;

                // This is the intended out directory for compiler documentation.
                let out = builder.compiler_doc_out(target);
                t!(fs::create_dir_all(&out));

                let compiler = builder.compiler(stage, builder.config.build);
                builder.ensure(compile::Std::new(compiler, target));

                if true $(&& $rustc_tool)? {
                    // Build rustc docs so that we generate relative links.
                    builder.ensure(Rustc { stage, target });

                    // Rustdoc needs the rustc sysroot available to build.
                    // FIXME: is there a way to only ensure `check::Rustc` here? Last time I tried it failed
                    // with strange errors, but only on a full bors test ...
                    builder.ensure(compile::Rustc::new(compiler, target));
                }

                let source_type = if true $(&& $in_tree)? {
                    SourceType::InTree
                } else {
                    SourceType::Submodule
                };

                builder.info(
                    &format!(
                        "Documenting stage{} {} ({})",
                        stage,
                        stringify!($tool).to_lowercase(),
                        target,
                    ),
                );

                // Symlink compiler docs to the output directory of rustdoc documentation.
                let out_dirs = [
                    builder.stage_out(compiler, Mode::ToolRustc).join(target.triple).join("doc"),
                    // Cargo uses a different directory for proc macros.
                    builder.stage_out(compiler, Mode::ToolRustc).join("doc"),
                ];
                for out_dir in out_dirs {
                    t!(fs::create_dir_all(&out_dir));
                    t!(symlink_dir_force(&builder.config, &out, &out_dir));
                }

                // Build cargo command.
                let mut cargo = prepare_tool_cargo(
                    builder,
                    compiler,
                    Mode::ToolRustc,
                    target,
                    "doc",
                    $path,
                    source_type,
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
                cargo.rustdocflag("--generate-link-to-definition");
                cargo.rustdocflag("-Zunstable-options");
                builder.run(&mut cargo.into());
            }
        }
    }
}

tool_doc!(Rustdoc, "rustdoc-tool", "src/tools/rustdoc", ["rustdoc", "rustdoc-json-types"],);
tool_doc!(
    Rustfmt,
    "rustfmt-nightly",
    "src/tools/rustfmt",
    ["rustfmt-nightly", "rustfmt-config_proc_macro"],
);
tool_doc!(Clippy, "clippy", "src/tools/clippy", ["clippy_utils"]);
tool_doc!(Miri, "miri", "src/tools/miri", ["miri"]);
tool_doc!(
    Cargo,
    "cargo",
    "src/tools/cargo",
    rustc_tool = false,
    in_tree = false,
    [
        "cargo",
        "cargo-platform",
        "cargo-util",
        "crates-io",
        "cargo-test-macro",
        "cargo-test-support",
        "cargo-credential",
        "cargo-credential-1password",
        "mdman",
        // FIXME: this trips a license check in tidy.
        // "resolver-tests",
        // FIXME: we should probably document these, but they're different per-platform so we can't use `tool_doc`.
        // "cargo-credential-gnome-secret",
        // "cargo-credential-macos-keychain",
        // "cargo-credential-wincred",
    ]
);
tool_doc!(Tidy, "tidy", "src/tools/tidy", ["tidy"]);

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
        index.arg(out);
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
    if config.dry_run() {
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
        builder.ensure(compile::Std::new(self.compiler, self.target));
        let mut cmd = builder.tool_cmd(Tool::LintDocs);
        cmd.arg("--src");
        cmd.arg(builder.src.join("compiler"));
        cmd.arg("--out");
        cmd.arg(&out_listing);
        cmd.arg("--rustc");
        cmd.arg(&rustc);
        cmd.arg("--rustc-target").arg(&self.target.rustc_target_arg());
        if builder.is_verbose() {
            cmd.arg("--verbose");
        }
        if self.validate {
            cmd.arg("--validate");
        }
        // We need to validate nightly features, even on the stable channel.
        // Set this unconditionally as the stage0 compiler may be being used to
        // document.
        cmd.env("RUSTC_BOOTSTRAP", "1");

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
            parent: Some(self),
        });
    }
}
