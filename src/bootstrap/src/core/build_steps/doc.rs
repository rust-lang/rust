//! Documentation generation for bootstrap.
//!
//! This module implements generation for all bits and pieces of documentation
//! for the Rust project. This notably includes suites like the rust book, the
//! nomicon, rust by example, standalone documentation, etc.
//!
//! Everything here is basically just a shim around calling either `rustbook` or
//! `rustdoc`.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::{env, fs, mem};

use crate::core::build_steps::compile;
use crate::core::build_steps::tool::{self, SourceType, Tool, prepare_tool_cargo};
use crate::core::builder::{
    self, Alias, Builder, Compiler, Kind, RunConfig, ShouldRun, Step, crate_description,
};
use crate::core::config::{Config, TargetSelection};
use crate::helpers::{submodule_path_of, symlink_dir, t, up_to_date};
use crate::{FileType, Mode};

macro_rules! book {
    ($($name:ident, $path:expr, $book_name:expr, $lang:expr ;)+) => {
        $(
            #[derive(Debug, Clone, Hash, PartialEq, Eq)]
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
                if let Some(submodule_path) = submodule_path_of(&builder, $path) {
                    builder.require_submodule(&submodule_path, None)
                }

                builder.ensure(RustbookSrc {
                    target: self.target,
                    name: $book_name.to_owned(),
                    src: builder.src.join($path),
                    parent: Some(self),
                    languages: $lang.into(),
                    rustdoc_compiler: None,
                })
            }
        }
        )+
    }
}

// NOTE: When adding a book here, make sure to ALSO build the book by
// adding a build step in `src/bootstrap/code/builder/mod.rs`!
// NOTE: Make sure to add the corresponding submodule when adding a new book.
book!(
    CargoBook, "src/tools/cargo/src/doc", "cargo", &[];
    ClippyBook, "src/tools/clippy/book", "clippy", &[];
    EditionGuide, "src/doc/edition-guide", "edition-guide", &[];
    EmbeddedBook, "src/doc/embedded-book", "embedded-book", &[];
    Nomicon, "src/doc/nomicon", "nomicon", &[];
    RustByExample, "src/doc/rust-by-example", "rust-by-example", &["ja", "zh"];
    RustdocBook, "src/doc/rustdoc", "rustdoc", &[];
    StyleGuide, "src/doc/style-guide", "style-guide", &[];
);

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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
            name: "unstable-book".to_owned(),
            src: builder.md_doc_out(self.target).join("unstable-book"),
            parent: Some(self),
            languages: vec![],
            rustdoc_compiler: None,
        })
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct RustbookSrc<P: Step> {
    target: TargetSelection,
    name: String,
    src: PathBuf,
    parent: Option<P>,
    languages: Vec<&'static str>,
    rustdoc_compiler: Option<Compiler>,
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

        let out = out.join(&name);
        let index = out.join("index.html");
        let rustbook = builder.tool_exe(Tool::Rustbook);

        if !builder.config.dry_run()
            && (!up_to_date(&src, &index) || !up_to_date(&rustbook, &index))
        {
            builder.info(&format!("Rustbook ({target}) - {name}"));
            let _ = fs::remove_dir_all(&out);

            let mut rustbook_cmd = builder.tool_cmd(Tool::Rustbook);

            if let Some(compiler) = self.rustdoc_compiler {
                let mut rustdoc = builder.rustdoc(compiler);
                rustdoc.pop();
                let old_path = env::var_os("PATH").unwrap_or_default();
                let new_path =
                    env::join_paths(std::iter::once(rustdoc).chain(env::split_paths(&old_path)))
                        .expect("could not add rustdoc to PATH");

                rustbook_cmd.env("PATH", new_path);
                builder.add_rustc_lib_path(compiler, &mut rustbook_cmd);
            }

            rustbook_cmd
                .arg("build")
                .arg(&src)
                .arg("-d")
                .arg(&out)
                .arg("--rust-root")
                .arg(&builder.src)
                .run(builder);

            for lang in &self.languages {
                let out = out.join(lang);

                builder.info(&format!("Rustbook ({target}) - {name} - {lang}"));
                let _ = fs::remove_dir_all(&out);

                builder
                    .tool_cmd(Tool::Rustbook)
                    .arg("build")
                    .arg(&src)
                    .arg("-d")
                    .arg(&out)
                    .arg("-l")
                    .arg(lang)
                    .run(builder);
            }
        }

        if self.parent.is_some() {
            builder.maybe_open_in_browser::<P>(index)
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.host_target),
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
        builder.require_submodule("src/doc/book", None);

        let compiler = self.compiler;
        let target = self.target;

        let absolute_path = builder.src.join("src/doc/book");
        let redirect_path = absolute_path.join("redirects");

        // build book
        builder.ensure(RustbookSrc {
            target,
            name: "book".to_owned(),
            src: absolute_path.clone(),
            parent: Some(self),
            languages: vec![],
            rustdoc_compiler: None,
        });

        // building older edition redirects
        for edition in &["first-edition", "second-edition", "2018-edition"] {
            builder.ensure(RustbookSrc {
                target,
                name: format!("book/{edition}"),
                src: absolute_path.join(edition),
                // There should only be one book that is marked as the parent for each target, so
                // treat the other editions as not having a parent.
                parent: Option::<Self>::None,
                languages: vec![],
                rustdoc_compiler: None,
            });
        }

        // build the version info page and CSS
        let shared_assets = builder.ensure(SharedAssets { target });

        // build the redirect pages
        let _guard = builder.msg_doc(compiler, "book redirect pages", target);
        for file in t!(fs::read_dir(redirect_path)) {
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
        .arg("../rust.css")
        .arg("-Zunstable-options");

    if !builder.config.docs_minification {
        cmd.arg("--disable-minification");
    }

    cmd.run(builder);
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.host_target),
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
        let _guard = builder.msg_doc(compiler, "standalone", target);
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

            cmd.arg("--html-after-content")
                .arg(&footer)
                .arg("--html-before-content")
                .arg(&version_info)
                .arg("--html-in-header")
                .arg(&favicon)
                .arg("--markdown-no-toc")
                .arg("-Zunstable-options")
                .arg("--index-page")
                .arg(builder.src.join("src/doc/index.md"))
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
            cmd.run(builder);
        }

        // We open doc/index.html as the default if invoked as `x.py doc --open`
        // with no particular explicit doc requested (e.g. library/core).
        if builder.paths.is_empty() || builder.was_invoked_explicitly::<Self>(Kind::Doc) {
            let index = out.join("index.html");
            builder.open_in_browser(index);
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Releases {
    compiler: Compiler,
    target: TargetSelection,
}

impl Step for Releases {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("RELEASES.md").alias("releases").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Releases {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.host_target),
            target: run.target,
        });
    }

    /// Generates HTML release notes to include in the final docs bundle.
    ///
    /// This uses the same stylesheet and other tools as Standalone, but the
    /// RELEASES.md file is included at the root of the repository and gets
    /// the headline added. In the end, the conversion is done by Rustdoc.
    fn run(self, builder: &Builder<'_>) {
        let target = self.target;
        let compiler = self.compiler;
        let _guard = builder.msg_doc(compiler, "releases", target);
        let out = builder.doc_out(target);
        t!(fs::create_dir_all(&out));

        builder.ensure(Standalone {
            compiler: builder.compiler(builder.top_stage, builder.config.host_target),
            target,
        });

        let version_info = builder.ensure(SharedAssets { target: self.target }).version_info;

        let favicon = builder.src.join("src/doc/favicon.inc");
        let footer = builder.src.join("src/doc/footer.inc");
        let full_toc = builder.src.join("src/doc/full-toc.inc");

        let html = out.join("releases.html");
        let tmppath = out.join("releases.md");
        let inpath = builder.src.join("RELEASES.md");
        let rustdoc = builder.rustdoc(compiler);
        if !up_to_date(&inpath, &html)
            || !up_to_date(&footer, &html)
            || !up_to_date(&favicon, &html)
            || !up_to_date(&full_toc, &html)
            || !(builder.config.dry_run()
                || up_to_date(&version_info, &html)
                || up_to_date(&rustdoc, &html))
        {
            let mut tmpfile = t!(fs::File::create(&tmppath));
            t!(tmpfile.write_all(b"% Rust Release Notes\n\n"));
            t!(io::copy(&mut t!(fs::File::open(&inpath)), &mut tmpfile));
            mem::drop(tmpfile);
            let mut cmd = builder.rustdoc_cmd(compiler);

            cmd.arg("--html-after-content")
                .arg(&footer)
                .arg("--html-before-content")
                .arg(&version_info)
                .arg("--html-in-header")
                .arg(&favicon)
                .arg("--markdown-no-toc")
                .arg("--markdown-css")
                .arg("rust.css")
                .arg("-Zunstable-options")
                .arg("--index-page")
                .arg(builder.src.join("src/doc/index.md"))
                .arg("--markdown-playground-url")
                .arg("https://play.rust-lang.org/")
                .arg("-o")
                .arg(&out)
                .arg(&tmppath);

            if !builder.config.docs_minification {
                cmd.arg("--disable-minification");
            }

            cmd.run(builder);
        }

        // We open doc/RELEASES.html as the default if invoked as `x.py doc --open RELEASES.md`
        // with no particular explicit doc requested (e.g. library/core).
        if builder.was_invoked_explicitly::<Self>(Kind::Doc) {
            builder.open_in_browser(&html);
        }
    }
}

#[derive(Debug, Clone)]
pub struct SharedAssetsPaths {
    pub version_info: PathBuf,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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

    /// Generate shared resources used by other pieces of documentation.
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let out = builder.doc_out(self.target);

        let version_input = builder.src.join("src").join("doc").join("version_info.html.template");
        let version_info = out.join("version_info.html");
        if !builder.config.dry_run() && !up_to_date(&version_input, &version_info) {
            let info = t!(fs::read_to_string(&version_input))
                .replace("VERSION", &builder.rust_release())
                .replace("SHORT_HASH", builder.rust_info().sha_short().unwrap_or(""))
                .replace("STAMP", builder.rust_info().sha().unwrap_or(""));
            t!(fs::write(&version_info, info));
        }

        builder.copy_link(
            &builder.src.join("src").join("doc").join("rust.css"),
            &out.join("rust.css"),
            FileType::Regular,
        );

        SharedAssetsPaths { version_info }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Std {
    pub stage: u32,
    pub target: TargetSelection,
    pub format: DocumentationFormat,
    crates: Vec<String>,
}

impl Std {
    pub(crate) fn new(stage: u32, target: TargetSelection, format: DocumentationFormat) -> Self {
        Std { stage, target, format, crates: vec![] }
    }
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.crate_or_deps("sysroot").path("library").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        let crates = compile::std_crates_for_run_make(&run);
        let target_is_no_std = run.builder.no_std(run.target).unwrap_or(false);
        if crates.is_empty() && target_is_no_std {
            return;
        }
        run.builder.ensure(Std {
            stage: run.builder.top_stage,
            target: run.target,
            format: if run.builder.config.cmd.json() {
                DocumentationFormat::Json
            } else {
                DocumentationFormat::Html
            },
            crates,
        });
    }

    /// Compile all standard library documentation.
    ///
    /// This will generate all documentation for the standard library and its
    /// dependencies. This is largely just a wrapper around `cargo doc`.
    fn run(self, builder: &Builder<'_>) {
        let stage = self.stage;
        let target = self.target;
        let crates = if self.crates.is_empty() {
            builder
                .in_tree_crates("sysroot", Some(target))
                .iter()
                .map(|c| c.name.to_string())
                .collect()
        } else {
            self.crates
        };

        let out = match self.format {
            DocumentationFormat::Html => builder.doc_out(target),
            DocumentationFormat::Json => builder.json_doc_out(target),
        };

        t!(fs::create_dir_all(&out));

        if self.format == DocumentationFormat::Html {
            builder.ensure(SharedAssets { target: self.target });
        }

        let index_page = builder
            .src
            .join("src/doc/index.md")
            .into_os_string()
            .into_string()
            .expect("non-utf8 paths are unsupported");
        let mut extra_args = match self.format {
            DocumentationFormat::Html => {
                vec!["--markdown-css", "rust.css", "--markdown-no-toc", "--index-page", &index_page]
            }
            DocumentationFormat::Json => vec!["--output-format", "json"],
        };

        if !builder.config.docs_minification {
            extra_args.push("--disable-minification");
        }
        // For `--index-page` and `--output-format=json`.
        extra_args.push("-Zunstable-options");

        doc_std(builder, self.format, stage, target, &out, &extra_args, &crates);

        // Don't open if the format is json
        if let DocumentationFormat::Json = self.format {
            return;
        }

        if builder.paths.iter().any(|path| path.ends_with("library")) {
            // For `x.py doc library --open`, open `std` by default.
            let index = out.join("std").join("index.html");
            builder.open_in_browser(index);
        } else {
            for requested_crate in crates {
                if STD_PUBLIC_CRATES.iter().any(|&k| k == requested_crate) {
                    let index = out.join(requested_crate).join("index.html");
                    builder.open_in_browser(index);
                    break;
                }
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

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum DocumentationFormat {
    Html,
    Json,
}

impl DocumentationFormat {
    fn as_str(&self) -> &str {
        match self {
            DocumentationFormat::Html => "HTML",
            DocumentationFormat::Json => "JSON",
        }
    }
}

/// Build the documentation for public standard library crates.
fn doc_std(
    builder: &Builder<'_>,
    format: DocumentationFormat,
    stage: u32,
    target: TargetSelection,
    out: &Path,
    extra_args: &[&str],
    requested_crates: &[String],
) {
    let compiler = builder.compiler(stage, builder.config.host_target);

    let target_doc_dir_name = if format == DocumentationFormat::Json { "json-doc" } else { "doc" };
    let target_dir = builder.stage_out(compiler, Mode::Std).join(target).join(target_doc_dir_name);

    // This is directory where the compiler will place the output of the command.
    // We will then copy the files from this directory into the final `out` directory, the specified
    // as a function parameter.
    let out_dir = target_dir.join(target).join("doc");

    let mut cargo =
        builder::Cargo::new(builder, compiler, Mode::Std, SourceType::InTree, target, Kind::Doc);

    compile::std_cargo(builder, target, compiler.stage, &mut cargo);
    cargo
        .arg("--no-deps")
        .arg("--target-dir")
        .arg(&*target_dir.to_string_lossy())
        .arg("-Zskip-rustdoc-fingerprint")
        .arg("-Zrustdoc-map")
        .rustdocflag("--extern-html-root-url")
        .rustdocflag("std_detect=https://docs.rs/std_detect/latest/")
        .rustdocflag("--extern-html-root-takes-precedence")
        .rustdocflag("--resource-suffix")
        .rustdocflag(&builder.version);
    for arg in extra_args {
        cargo.rustdocflag(arg);
    }

    if builder.config.library_docs_private_items {
        cargo.rustdocflag("--document-private-items").rustdocflag("--document-hidden-items");
    }

    for krate in requested_crates {
        if krate == "sysroot" {
            // The sysroot crate is an implementation detail, don't include it in public docs.
            continue;
        }
        cargo.arg("-p").arg(krate);
    }

    let description =
        format!("library{} in {} format", crate_description(requested_crates), format.as_str());
    let _guard = builder.msg_doc(compiler, description, target);

    cargo.into_cmd().run(builder);
    builder.cp_link_r(&out_dir, out);
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Rustc {
    pub stage: u32,
    pub target: TargetSelection,
    crates: Vec<String>,
}

impl Rustc {
    pub(crate) fn new(stage: u32, target: TargetSelection, builder: &Builder<'_>) -> Self {
        let crates = builder
            .in_tree_crates("rustc-main", Some(target))
            .into_iter()
            .map(|krate| krate.name.to_string())
            .collect();
        Self { stage, target, crates }
    }
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
        run.builder.ensure(Rustc {
            stage: run.builder.top_stage,
            target: run.target,
            crates: run.make_run_crates(Alias::Compiler),
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

        // This is the intended out directory for compiler documentation.
        let out = builder.compiler_doc_out(target);
        t!(fs::create_dir_all(&out));

        // Build the standard library, so that proc-macros can use it.
        // (Normally, only the metadata would be necessary, but proc-macros are special since they run at compile-time.)
        let compiler = builder.compiler(stage, builder.config.host_target);
        builder.ensure(compile::Std::new(compiler, builder.config.host_target));

        let _guard = builder.msg_sysroot_tool(
            Kind::Doc,
            stage,
            format!("compiler{}", crate_description(&self.crates)),
            compiler.host,
            target,
        );

        // Build cargo command.
        let mut cargo = builder::Cargo::new(
            builder,
            compiler,
            Mode::Rustc,
            SourceType::InTree,
            target,
            Kind::Doc,
        );

        cargo.rustdocflag("--document-private-items");
        // Since we always pass --document-private-items, there's no need to warn about linking to private items.
        cargo.rustdocflag("-Arustdoc::private-intra-doc-links");
        cargo.rustdocflag("--enable-index-page");
        cargo.rustdocflag("-Znormalize-docs");
        cargo.rustdocflag("--show-type-layout");
        // FIXME: `--generate-link-to-definition` tries to resolve cfged out code
        // see https://github.com/rust-lang/rust/pull/122066#issuecomment-1983049222
        // If there is any bug, please comment out the next line.
        cargo.rustdocflag("--generate-link-to-definition");

        compile::rustc_cargo(builder, &mut cargo, target, &compiler, &self.crates);
        cargo.arg("-Zskip-rustdoc-fingerprint");

        // Only include compiler crates, no dependencies of those, such as `libc`.
        // Do link to dependencies on `docs.rs` however using `rustdoc-map`.
        cargo.arg("--no-deps");
        cargo.arg("-Zrustdoc-map");

        // FIXME: `-Zrustdoc-map` does not yet correctly work for transitive dependencies,
        // once this is no longer an issue the special case for `ena` can be removed.
        cargo.rustdocflag("--extern-html-root-url");
        cargo.rustdocflag("ena=https://docs.rs/ena/latest/");

        let mut to_open = None;

        let out_dir = builder.stage_out(compiler, Mode::Rustc).join(target).join("doc");
        for krate in &*self.crates {
            // Create all crate output directories first to make sure rustdoc uses
            // relative links.
            // FIXME: Cargo should probably do this itself.
            let dir_name = krate.replace('-', "_");
            t!(fs::create_dir_all(out_dir.join(&*dir_name)));
            cargo.arg("-p").arg(krate);
            if to_open.is_none() {
                to_open = Some(dir_name);
            }
        }

        // This uses a shared directory so that librustdoc documentation gets
        // correctly built and merged with the rustc documentation.
        //
        // This is needed because rustdoc is built in a different directory from
        // rustc. rustdoc needs to be able to see everything, for example when
        // merging the search index, or generating local (relative) links.
        symlink_dir_force(&builder.config, &out, &out_dir);
        // Cargo puts proc macros in `target/doc` even if you pass `--target`
        // explicitly (https://github.com/rust-lang/cargo/issues/7677).
        let proc_macro_out_dir = builder.stage_out(compiler, Mode::Rustc).join("doc");
        symlink_dir_force(&builder.config, &out, &proc_macro_out_dir);

        cargo.into_cmd().run(builder);

        if !builder.config.dry_run() {
            // Sanity check on linked compiler crates
            for krate in &*self.crates {
                let dir_name = krate.replace('-', "_");
                // Making sure the directory exists and is not empty.
                assert!(out.join(&*dir_name).read_dir().unwrap().next().is_some());
            }
        }

        if builder.paths.iter().any(|path| path.ends_with("compiler")) {
            // For `x.py doc compiler --open`, open `rustc_middle` by default.
            let index = out.join("rustc_middle").join("index.html");
            builder.open_in_browser(index);
        } else if let Some(krate) = to_open {
            // Let's open the first crate documentation page:
            let index = out.join(krate).join("index.html");
            builder.open_in_browser(index);
        }
    }
}

macro_rules! tool_doc {
    (
        $tool: ident,
        $path: literal,
        $(rustc_tool = $rustc_tool:literal, )?
        $(is_library = $is_library:expr,)?
        $(crates = $crates:expr)?
       ) => {
        #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $tool {
            target: TargetSelection,
        }

        impl Step for $tool {
            type Output = ();
            const DEFAULT: bool = true;
            const ONLY_HOSTS: bool = true;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                let builder = run.builder;
                run.path($path).default_condition(builder.config.compiler_docs)
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
                let mut source_type = SourceType::InTree;

                if let Some(submodule_path) = submodule_path_of(&builder, $path) {
                    source_type = SourceType::Submodule;
                    builder.require_submodule(&submodule_path, None);
                }

                let stage = builder.top_stage;
                let target = self.target;

                // This is the intended out directory for compiler documentation.
                let out = builder.compiler_doc_out(target);
                t!(fs::create_dir_all(&out));

                let compiler = builder.compiler(stage, builder.config.host_target);
                builder.ensure(compile::Std::new(compiler, target));

                if true $(&& $rustc_tool)? {
                    // Build rustc docs so that we generate relative links.
                    builder.ensure(Rustc::new(stage, target, builder));

                    // Rustdoc needs the rustc sysroot available to build.
                    // FIXME: is there a way to only ensure `check::Rustc` here? Last time I tried it failed
                    // with strange errors, but only on a full bors test ...
                    builder.ensure(compile::Rustc::new(compiler, target));
                }

                // Build cargo command.
                let mut cargo = prepare_tool_cargo(
                    builder,
                    compiler,
                    Mode::ToolRustc,
                    target,
                    Kind::Doc,
                    $path,
                    source_type,
                    &[],
                );

                cargo.arg("-Zskip-rustdoc-fingerprint");
                // Only include compiler crates, no dependencies of those, such as `libc`.
                cargo.arg("--no-deps");

                if false $(|| $is_library)? {
                    cargo.arg("--lib");
                }

                $(for krate in $crates {
                    cargo.arg("-p").arg(krate);
                })?

                cargo.rustdocflag("--document-private-items");
                // Since we always pass --document-private-items, there's no need to warn about linking to private items.
                cargo.rustdocflag("-Arustdoc::private-intra-doc-links");
                cargo.rustdocflag("--enable-index-page");
                cargo.rustdocflag("--show-type-layout");
                cargo.rustdocflag("--generate-link-to-definition");

                let out_dir = builder.stage_out(compiler, Mode::ToolRustc).join(target).join("doc");
                $(for krate in $crates {
                    let dir_name = krate.replace("-", "_");
                    t!(fs::create_dir_all(out_dir.join(&*dir_name)));
                })?

                // Symlink compiler docs to the output directory of rustdoc documentation.
                symlink_dir_force(&builder.config, &out, &out_dir);
                let proc_macro_out_dir = builder.stage_out(compiler, Mode::ToolRustc).join("doc");
                symlink_dir_force(&builder.config, &out, &proc_macro_out_dir);

                let _guard = builder.msg_doc(compiler, stringify!($tool).to_lowercase(), target);
                cargo.into_cmd().run(builder);

                if !builder.config.dry_run() {
                    // Sanity check on linked doc directories
                    $(for krate in $crates {
                        let dir_name = krate.replace("-", "_");
                        // Making sure the directory exists and is not empty.
                        assert!(out.join(&*dir_name).read_dir().unwrap().next().is_some());
                    })?
                }
            }
        }
    }
}

// NOTE: make sure to register these in `Builder::get_step_description`.
tool_doc!(
    BuildHelper,
    "src/build_helper",
    rustc_tool = false,
    is_library = true,
    crates = ["build_helper"]
);
tool_doc!(Rustdoc, "src/tools/rustdoc", crates = ["rustdoc", "rustdoc-json-types"]);
tool_doc!(Rustfmt, "src/tools/rustfmt", crates = ["rustfmt-nightly", "rustfmt-config_proc_macro"]);
tool_doc!(Clippy, "src/tools/clippy", crates = ["clippy_config", "clippy_utils"]);
tool_doc!(Miri, "src/tools/miri", crates = ["miri"]);
tool_doc!(
    Cargo,
    "src/tools/cargo",
    rustc_tool = false,
    crates = [
        "cargo",
        "cargo-credential",
        "cargo-platform",
        "cargo-test-macro",
        "cargo-test-support",
        "cargo-util",
        "cargo-util-schemas",
        "crates-io",
        "mdman",
        "rustfix",
    ]
);
tool_doc!(Tidy, "src/tools/tidy", rustc_tool = false, crates = ["tidy"]);
tool_doc!(
    Bootstrap,
    "src/bootstrap",
    rustc_tool = false,
    is_library = true,
    crates = ["bootstrap"]
);
tool_doc!(
    RunMakeSupport,
    "src/tools/run-make-support",
    rustc_tool = false,
    is_library = true,
    crates = ["run_make_support"]
);
tool_doc!(
    Compiletest,
    "src/tools/compiletest",
    rustc_tool = false,
    is_library = true,
    crates = ["compiletest"]
);

#[derive(Ord, PartialOrd, Debug, Clone, Hash, PartialEq, Eq)]
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
        tool::ErrorIndex::command(builder).arg("html").arg(out).arg(&builder.version).run(builder);
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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

        builder.info(&format!("Generating unstable book md files ({target})"));
        let out = builder.md_doc_out(target).join("unstable-book");
        builder.create_dir(&out);
        builder.remove_dir(&out);
        let mut cmd = builder.tool_cmd(Tool::UnstableBookGen);
        cmd.arg(builder.src.join("library"));
        cmd.arg(builder.src.join("compiler"));
        cmd.arg(builder.src.join("src"));
        cmd.arg(out);

        cmd.run(builder);
    }
}

fn symlink_dir_force(config: &Config, original: &Path, link: &Path) {
    if config.dry_run() {
        return;
    }
    if let Ok(m) = fs::symlink_metadata(link) {
        if m.file_type().is_dir() {
            t!(fs::remove_dir_all(link));
        } else {
            // handle directory junctions on windows by falling back to
            // `remove_dir`.
            t!(fs::remove_file(link).or_else(|_| fs::remove_dir(link)));
        }
    }

    t!(
        symlink_dir(config, original, link),
        format!("failed to create link from {} -> {}", link.display(), original.display())
    );
}

#[derive(Ord, PartialOrd, Debug, Clone, Hash, PartialEq, Eq)]
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
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.host_target),
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
        builder.cp_link_r(&builder.src.join("src/doc/rustc"), &out_base);
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
        cmd.arg("--rustc-target").arg(self.target.rustc_target_arg());
        if let Some(target_linker) = builder.linker(self.target) {
            cmd.arg("--rustc-linker").arg(target_linker);
        }
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
        // bootstrap.toml), then this needs to explicitly update the dylib search
        // path.
        builder.add_rustc_lib_path(self.compiler, &mut cmd);
        let doc_generator_guard = builder.msg(
            Kind::Run,
            self.compiler.stage,
            "lint-docs",
            self.compiler.host,
            self.target,
        );
        cmd.run(builder);
        drop(doc_generator_guard);

        // Run rustbook/mdbook to generate the HTML pages.
        builder.ensure(RustbookSrc {
            target: self.target,
            name: "rustc".to_owned(),
            src: out_base,
            parent: Some(self),
            languages: vec![],
            rustdoc_compiler: None,
        });
    }
}

#[derive(Ord, PartialOrd, Debug, Clone, Hash, PartialEq, Eq)]
pub struct Reference {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Reference {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/doc/reference").default_condition(builder.config.docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Reference {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.host_target),
            target: run.target,
        });
    }

    /// Builds the reference book.
    fn run(self, builder: &Builder<'_>) {
        builder.require_submodule("src/doc/reference", None);

        // This is needed for generating links to the standard library using
        // the mdbook-spec plugin.
        builder.ensure(compile::Std::new(self.compiler, builder.config.host_target));

        // Run rustbook/mdbook to generate the HTML pages.
        builder.ensure(RustbookSrc {
            target: self.target,
            name: "reference".to_owned(),
            src: builder.src.join("src/doc/reference"),
            rustdoc_compiler: Some(self.compiler),
            parent: Some(self),
            languages: vec![],
        });
    }
}
