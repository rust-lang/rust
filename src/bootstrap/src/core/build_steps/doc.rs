//! Documentation generation for bootstrap.
//!
//! This module implements generation for all bits and pieces of documentation
//! for the Rust project. This notably includes suites like the rust book, the
//! nomicon, rust by example, standalone documentation, etc.
//!
//! Everything here is basically just a shim around calling either `rustbook` or
//! `rustdoc`.

use std::collections::HashSet;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::{env, fs, mem};

use crate::core::build_steps::compile;
use crate::core::build_steps::compile::{CargoMessage, stream_cargo};
use crate::core::build_steps::tool::{
    self, RustcPrivateCompilers, SourceType, Tool, prepare_tool_cargo,
};
use crate::core::builder::{
    self, Builder, CommandLineStep, Kind, RunConfig, ShouldRun, Step, StepMetadata,
    crate_description,
};
use crate::core::compiler::Compiler;
use crate::core::config::TargetSelection;
use crate::core::session::{FileType, Mode};
use crate::utils::helpers::{exit_process, submodule_path_of, symlink_dir, t, up_to_date};

macro_rules! book {
    ($($name:ident, $path:expr, $book_name:expr, $lang:expr ;)+) => {
        $(
        #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            target: TargetSelection,
        }

        impl CommandLineStep for $name {
            type Output = ();

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.path($path)
            }

            fn is_default_step(builder: &Builder<'_>) -> bool {
                builder.config.docs
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
                    build_compiler: None,
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
    CargoBook, "src/tools/cargo/doc/book", "cargo", &[];
    ClippyBook, "src/tools/clippy/book", "clippy", &[];
    EditionGuide, "src/doc/edition-guide", "edition-guide", &[];
    EmbeddedBook, "src/doc/embedded-book", "embedded-book", &[];
    Nomicon, "src/doc/nomicon", "nomicon", &[];
    RustByExample, "src/doc/rust-by-example", "rust-by-example", &["es", "ja", "zh", "ko"];
    RustdocBook, "src/doc/rustdoc", "rustdoc", &[];
    StyleGuide, "src/doc/style-guide", "style-guide", &[];
);

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct UnstableBook {
    build_compiler: Compiler,
    target: TargetSelection,
}

impl CommandLineStep for UnstableBook {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/doc/unstable-book")
    }

    fn is_default_step(builder: &Builder<'_>) -> bool {
        builder.config.docs
    }

    fn make_run(run: RunConfig<'_>) {
        // Bump the stage to 2, because the unstable book requires an in-tree compiler.
        // At the same time, since this step is enabled by default, we don't want `x doc` to fail
        // in stage 1.
        let stage = if run.builder.config.is_explicit_stage() || run.builder.top_stage >= 2 {
            run.builder.top_stage
        } else {
            2
        };

        run.builder.ensure(UnstableBook {
            build_compiler: prepare_doc_compiler(run.builder, run.target, stage),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        let unstable_book_md_dir = builder
            .ensure(UnstableBookGen { build_compiler: self.build_compiler, target: self.target });
        builder.ensure(RustbookSrc {
            target: self.target,
            name: "unstable-book".to_owned(),
            src: unstable_book_md_dir,
            parent: Some(self),
            languages: vec![],
            build_compiler: None,
        })
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct RustbookSrc<P: CommandLineStep> {
    target: TargetSelection,
    name: String,
    src: PathBuf,
    parent: Option<P>,
    languages: Vec<&'static str>,
    /// Compiler whose rustdoc should be used to document things using `mdbook-spec`.
    build_compiler: Option<Compiler>,
}

impl<P: CommandLineStep> Step for RustbookSrc<P> {
    type Output = ();

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

            if let Some(compiler) = self.build_compiler {
                let mut rustdoc = builder.rustdoc_for_compiler(compiler);
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

    fn metadata(&self) -> Option<StepMetadata> {
        let mut metadata = StepMetadata::doc(&format!("{} (book)", self.name), self.target);
        if let Some(compiler) = self.build_compiler {
            metadata = metadata.built_by(compiler);
        }

        Some(metadata)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TheBook {
    /// Compiler whose rustdoc will be used to generated documentation.
    build_compiler: Compiler,
    target: TargetSelection,
}

impl CommandLineStep for TheBook {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/doc/book")
    }

    fn is_default_step(builder: &Builder<'_>) -> bool {
        builder.config.docs
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(TheBook {
            build_compiler: prepare_doc_compiler(run.builder, run.target, run.builder.top_stage),
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

        let build_compiler = self.build_compiler;
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
            build_compiler: None,
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
                build_compiler: None,
            });
        }

        // build the version info page and CSS
        let shared_assets = builder.ensure(SharedAssets { target });

        // build the redirect pages
        let _guard = builder.msg(Kind::Doc, "book redirect pages", None, build_compiler, target);
        if builder.config.dry_run() {
            return;
        }

        for file in t!(fs::read_dir(redirect_path)) {
            let file = t!(file);
            let path = file.path();
            let path = path.to_str().unwrap();

            invoke_rustdoc(builder, build_compiler, &shared_assets, target, path);
        }
    }
}

fn invoke_rustdoc(
    builder: &Builder<'_>,
    build_compiler: Compiler,
    shared_assets: &SharedAssetsPaths,
    target: TargetSelection,
    markdown: &str,
) {
    let out = builder.doc_out(target);

    let path = builder.src.join("src/doc").join(markdown);

    let header = builder.src.join("src/doc/redirect.inc");
    let footer = builder.src.join("src/doc/footer.inc");

    let mut cmd = builder.rustdoc_cmd(build_compiler);

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
    build_compiler: Compiler,
    target: TargetSelection,
}

impl CommandLineStep for Standalone {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/doc").alias("standalone")
    }

    fn is_default_step(builder: &Builder<'_>) -> bool {
        builder.config.docs
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Standalone {
            build_compiler: prepare_doc_compiler(
                run.builder,
                run.builder.host_target,
                run.builder.top_stage,
            ),
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
        let build_compiler = self.build_compiler;
        let _guard = builder.msg(Kind::Doc, "standalone", None, build_compiler, target);
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
            let rustdoc = builder.rustdoc_for_compiler(build_compiler);
            if up_to_date(&path, &html)
                && up_to_date(&footer, &html)
                && up_to_date(&favicon, &html)
                && up_to_date(&full_toc, &html)
                && (builder.config.dry_run() || up_to_date(&version_info, &html))
                && (builder.config.dry_run() || up_to_date(&rustdoc, &html))
            {
                continue;
            }

            let mut cmd = builder.rustdoc_cmd(build_compiler);

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

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::doc("standalone", self.target).built_by(self.build_compiler))
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Releases {
    build_compiler: Compiler,
    target: TargetSelection,
}

impl CommandLineStep for Releases {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("RELEASES.md").alias("releases")
    }

    fn is_default_step(builder: &Builder<'_>) -> bool {
        builder.config.docs
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Releases {
            build_compiler: prepare_doc_compiler(
                run.builder,
                run.builder.host_target,
                run.builder.top_stage,
            ),
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
        let build_compiler = self.build_compiler;
        let _guard = builder.msg(Kind::Doc, "releases", None, build_compiler, target);
        let out = builder.doc_out(target);
        t!(fs::create_dir_all(&out));

        builder.ensure(Standalone { build_compiler, target });

        let version_info = builder.ensure(SharedAssets { target: self.target }).version_info;

        let favicon = builder.src.join("src/doc/favicon.inc");
        let footer = builder.src.join("src/doc/footer.inc");
        let full_toc = builder.src.join("src/doc/full-toc.inc");

        let html = out.join("releases.html");
        let tmppath = out.join("releases.md");
        let inpath = builder.src.join("RELEASES.md");
        let rustdoc = builder.rustdoc_for_compiler(build_compiler);
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
            let mut cmd = builder.rustdoc_cmd(build_compiler);

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

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::doc("releases", self.target).built_by(self.build_compiler))
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

        builder.copy_link(
            &builder
                .src
                .join("src")
                .join("librustdoc")
                .join("html")
                .join("static")
                .join("images")
                .join("favicon.svg"),
            &out.join("favicon.svg"),
            FileType::Regular,
        );
        builder.copy_link(
            &builder
                .src
                .join("src")
                .join("librustdoc")
                .join("html")
                .join("static")
                .join("images")
                .join("favicon-32x32.png"),
            &out.join("favicon-32x32.png"),
            FileType::Regular,
        );

        SharedAssetsPaths { version_info }
    }
}

/// Document the standard library using `build_compiler`.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Std {
    build_compiler: Compiler,
    target: TargetSelection,
    format: DocumentationFormat,
    crates: Vec<String>,
}

impl Std {
    pub(crate) fn from_build_compiler(
        build_compiler: Compiler,
        target: TargetSelection,
        format: DocumentationFormat,
    ) -> Self {
        Std { build_compiler, target, format, crates: vec![] }
    }
}

impl CommandLineStep for Std {
    /// Path to a directory with the built documentation.
    type Output = PathBuf;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("sysroot").path("library")
    }

    fn is_default_step(builder: &Builder<'_>) -> bool {
        builder.config.docs
    }

    fn make_run(run: RunConfig<'_>) {
        let crates = compile::std_crates_for_make_run(&run);
        let target_is_no_std = run.builder.no_std(run.target).unwrap_or(false);
        if crates.is_empty() && target_is_no_std {
            return;
        }
        run.builder.ensure(Std {
            build_compiler: run.builder.compiler_for_std(run.builder.top_stage),
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
    fn run(self, builder: &Builder<'_>) -> Self::Output {
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
            DocumentationFormat::Json => vec![],
        };

        if !builder.config.docs_minification {
            extra_args.push("--disable-minification");
        }
        // For `--index-page` and `--output-format=json`.
        extra_args.push("-Zunstable-options");

        let target_doc_dir_name =
            if self.format == DocumentationFormat::Json { "json-doc" } else { "doc" };
        let target_dir = builder
            .stage_out(self.build_compiler, Mode::Std)
            .join(target)
            .join(target_doc_dir_name);

        // This is directory where the compiler will place the output of the command.
        // We will then copy the files from this directory into the final `out` directory, the specified
        // as a function parameter.
        let out_dir = target_dir.join(target).join("doc");

        let mut cargo = doc_std(
            builder,
            self.format,
            self.build_compiler,
            target,
            &target_dir,
            &extra_args,
            &crates,
        );
        match self.format {
            DocumentationFormat::Html => {}
            DocumentationFormat::Json => {
                // We have to pass these directly to cargo, rather than through RUSTDOCFLAGS,
                // otherwise Cargo will not detect freshness of the output correctly, and keep
                // rebuilding the docs on every invocation.
                cargo.args(["-Zunstable-options", "--output-format", "json"]);
            }
        }

        let description =
            format!("library{} in {} format", crate_description(&crates), self.format.as_str());

        {
            let _guard =
                builder.msg(Kind::Doc, description, Mode::Std, self.build_compiler, target);

            cargo.into_cmd().run(builder);
            builder.cp_link_r(&out_dir, &out);
        }

        // Open if the format is HTML
        if let DocumentationFormat::Html = self.format {
            if builder.paths.iter().any(|path| path.ends_with("library")) {
                // For `x.py doc library --open`, open `std` by default.
                let index = out.join("std").join("index.html");
                builder.maybe_open_in_browser::<Self>(index);
            } else {
                for requested_crate in crates {
                    if STD_PUBLIC_CRATES.iter().any(|&k| k == requested_crate) {
                        let index = out.join(requested_crate).join("index.html");
                        builder.maybe_open_in_browser::<Self>(index);
                        break;
                    }
                }
            }
        }

        out
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(
            StepMetadata::doc("std", self.target)
                .built_by(self.build_compiler)
                .with_metadata(format!("crates=[{}]", self.crates.join(","))),
        )
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

/// Prepare a Cargo command for building the documentation for public standard library crates.
fn doc_std(
    builder: &Builder<'_>,
    format: DocumentationFormat,
    build_compiler: Compiler,
    target: TargetSelection,
    target_dir: &Path,
    extra_args: &[&str],
    requested_crates: &[String],
) -> builder::Cargo {
    let mut cargo = builder::Cargo::new(
        builder,
        build_compiler,
        Mode::Std,
        SourceType::InTree,
        target,
        Kind::Doc,
    );

    compile::std_cargo(builder, target, &mut cargo, requested_crates);
    cargo
        .arg("--no-deps")
        .arg("--target-dir")
        .arg(&*target_dir.to_string_lossy())
        .arg("-Zrustdoc-map")
        .rustdocflag("--extern-html-root-url")
        .rustdocflag("std_detect=https://docs.rs/std_detect/latest/")
        .rustdocflag("--extern-html-root-takes-precedence")
        .rustdocflag("--resource-suffix")
        .rustdocflag(&builder.version);
    for arg in extra_args {
        cargo.rustdocflag(arg);
    }

    // This is needed for cargo-semver-checks and potentially other downstream tools that consume
    // the JSON data.
    if format == DocumentationFormat::Json || builder.config.library_docs_private_items {
        cargo.rustdocflag("--document-private-items").rustdocflag("--document-hidden-items");
    }
    cargo
}

/// Prepare a compiler that will be able to document something for `target` at `stage`.
pub fn prepare_doc_compiler(
    builder: &Builder<'_>,
    target: TargetSelection,
    stage: u32,
) -> Compiler {
    assert!(stage > 0, "Cannot document anything in stage 0");
    let build_compiler = builder.compiler(stage - 1, builder.host_target);
    builder.std(build_compiler, target);
    build_compiler
}

/// Run rustdoc to merge cross-crate info metadata (like the search index) from individual
/// executions of rustdoc into `out_dir`.
/// The `json_files` parameter should contain paths to JSON file artifacts generated by previous
/// executions of `cargo doc`.
fn merge_rustdoc_cci(
    builder: &Builder<'_>,
    build_compiler: Compiler,
    json_files: &[PathBuf],
    out_dir: &Path,
) {
    let mut cmd = builder.rustdoc_cmd(build_compiler);

    cmd.arg("--enable-index-page").arg("-Zunstable-options").arg("-o").arg(out_dir);

    if !builder.config.docs_minification {
        cmd.arg("--disable-minification");
    }

    for json_file in json_files {
        cmd.arg("--read-doc-meta-dir").arg(json_file.parent().unwrap());
    }

    cmd.run(builder);
}

/// Generate the combined compiler + tools docs for a given toolchain.
/// This contains both the compiler docs, docs of rustc_private tools (miri, clippy, etc.), cargo
/// and also some bootstrap related tools (bootstrap itself, compiletest, tidy, etc.).
///
/// It gets hosted at https://doc.rust-lang.org/nightly/nightly-rustc/index.html.
///
/// Compiler documentation is distributed separately, so we make sure
/// we do not merge it with the other documentation from std, test and
/// proc_macros. This is largely just a wrapper around `cargo doc`.
///
/// Returns a path to a directory with the generated documentation.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CompilerWithTools {
    build_compiler: Compiler,
    target: TargetSelection,
    stage: u32,
}

impl CompilerWithTools {
    /// Document `stage` compiler for the given `target`.
    pub(crate) fn for_stage(builder: &Builder<'_>, stage: u32, target: TargetSelection) -> Self {
        let build_compiler = prepare_doc_compiler(builder, target, stage);
        Self { build_compiler, target, stage }
    }
}

impl CommandLineStep for CompilerWithTools {
    type Output = PathBuf;
    const IS_HOST: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("compiler-with-tools")
    }

    fn is_default_step(builder: &Builder<'_>) -> bool {
        builder.config.compiler_docs
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CompilerWithTools::for_stage(
            run.builder,
            run.builder.top_stage,
            run.target,
        ));
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let CompilerWithTools { target, build_compiler, stage } = self;

        // This is the intended out directory for combined compiler documentation.
        let out = builder.compiler_doc_out(target);
        let _ = fs::remove_dir_all(&out);

        let _guard =
            builder.msg(Kind::Doc, "compiler-with-tools", Mode::Rustc, build_compiler, target);

        let combined_docs = vec![
            builder.ensure(Rustc::for_stage(builder, stage, target)),
            builder.ensure(Rustdoc::new(builder, target)),
            builder.ensure(Rustfmt::new(builder, target)),
            builder.ensure(Clippy::new(builder, target)),
            builder.ensure(Miri::new(builder, target)),
            builder.ensure(Cargo::new(builder, target)),
            builder.ensure(Tidy::new(builder, target)),
            builder.ensure(Bootstrap::new(builder, target)),
            builder.ensure(BuildHelper::new(builder, target)),
            builder.ensure(Compiletest::new(builder, target)),
            builder.ensure(RunMakeSupport::new(builder, target)),
        ];

        if !builder.config.dry_run() {
            // Now copy all the individual docs into a single directory
            let mut json_files = vec![];
            for docs in combined_docs {
                json_files.extend(docs.artifacts.json_files);

                // Doc directories to link to the shared output directory
                // We add the host doc dirs, which should already be symlinked in the target
                // docs dir at this point (see `merge_host_and_target_docs`).
                let dirs_to_copy: Vec<_> = docs
                    .artifacts
                    .target_dirs
                    .iter()
                    .chain(docs.artifacts.host_dirs.iter())
                    .map(|d| d.file_name().unwrap().to_str().unwrap())
                    .collect();
                for dir in dirs_to_copy {
                    // Link the docs dir
                    let docs_dir = docs.out_dir.join(dir);
                    assert!(docs_dir.exists(), "Docs directory {docs_dir:?} does not exist.");
                    let out_docs_dir = out.join(dir);
                    builder.create_dir(&out_docs_dir);
                    builder.cp_link_r(&docs_dir, &out_docs_dir);

                    // And the src dir
                    let src_dir = docs.out_dir.join("src").join(dir);
                    assert!(src_dir.exists(), "Docs source directory {src_dir:?} does not exist.");
                    let out_src_dir = out.join("src").join(dir);
                    builder.create_dir(&out_src_dir);
                    builder.cp_link_r(&src_dir, &out_src_dir);
                }
            }
            // And finally merge all the CCI metadata
            merge_rustdoc_cci(builder, build_compiler, &json_files, &out);
        }

        // Handle `--open`.
        builder.open_in_browser(out.join("index.html"));
        out
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::doc("CompilerWithTools", self.target).built_by(self.build_compiler))
    }
}

/// Output of a Doc step.
#[derive(Clone)]
pub struct BuiltDocs {
    /// Target doc directory with the generated documentation.
    out_dir: PathBuf,
    /// Doc artifacts gathered from Cargo during the doc build.
    artifacts: DocArtifacts,
}

/// Document the compiler for the given `target` using rustdoc from `build_compiler`.
///
/// Return the path to the generated rustc documentation directory.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Rustc {
    build_compiler: Compiler,
    target: TargetSelection,
    crates: Vec<String>,
}

impl Rustc {
    /// Document `stage` compiler for the given `target`.
    pub(crate) fn for_stage(builder: &Builder<'_>, stage: u32, target: TargetSelection) -> Self {
        let build_compiler = prepare_doc_compiler(builder, target, stage);
        Self::from_build_compiler(builder, build_compiler, target)
    }

    fn from_build_compiler(
        builder: &Builder<'_>,
        build_compiler: Compiler,
        target: TargetSelection,
    ) -> Self {
        let crates = builder
            .in_tree_crates("rustc-main", Some(target))
            .into_iter()
            .map(|krate| krate.name.to_string())
            .collect();
        Self { build_compiler, target, crates }
    }
}

impl CommandLineStep for Rustc {
    type Output = BuiltDocs;
    const IS_HOST: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("rustc-main").path("compiler")
    }

    fn is_default_step(builder: &Builder<'_>) -> bool {
        builder.config.compiler_docs
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rustc::for_stage(run.builder, run.builder.top_stage, run.target));
    }

    /// Generates compiler documentation.
    ///
    /// This will generate all documentation for compiler and dependencies.
    /// Compiler documentation is distributed separately, so we make sure
    /// we do not merge it with the other documentation from std, test and
    /// proc_macros. This is largely just a wrapper around `cargo doc`.
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let target = self.target;

        // Build the standard library, so that proc-macros can use it.
        // (Normally, only the metadata would be necessary, but proc-macros are special since they run at compile-time.)
        let build_compiler = self.build_compiler;
        builder.std(build_compiler, builder.config.host_target);

        let _guard = builder.msg(
            Kind::Doc,
            format!("compiler{}", crate_description(&self.crates)),
            Mode::Rustc,
            build_compiler,
            target,
        );

        // Build cargo command.
        let mut cargo = builder::Cargo::new(
            builder,
            build_compiler,
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
        cargo.rustdocflag("--generate-macro-expansion");

        compile::rustc_cargo(builder, &mut cargo, target, &build_compiler, &self.crates);

        // Only include compiler crates, no dependencies of those, such as `libc`.
        // Do link to dependencies on `docs.rs` however using `rustdoc-map`.
        cargo.arg("--no-deps");
        cargo.arg("-Zrustdoc-map");

        // FIXME: `-Zrustdoc-map` does not yet correctly work for transitive dependencies,
        // once this is no longer an issue the special case for `ena` can be removed.
        cargo.rustdocflag("--extern-html-root-url");
        cargo.rustdocflag("ena=https://docs.rs/ena/latest/");

        let cargo_target_dir = builder.stage_out(build_compiler, Mode::Rustc);
        let target_doc_dir = cargo_target_dir.join(target).join("doc");
        let host_doc_dir = cargo_target_dir.join("doc");
        for krate in &*self.crates {
            // Create all crate output directories first to make sure rustdoc uses
            // relative links.
            // FIXME: Cargo should probably do this itself.
            let dir_name = normalize_doc_crate_name(krate);
            t!(fs::create_dir_all(target_doc_dir.join(&*dir_name)));
            cargo.arg("-p").arg(krate);
        }

        let artifacts = create_docs_and_gather_artifacts(builder, cargo);
        artifacts.sanity_check_crates(builder, self.crates.iter());

        if !builder.config.dry_run() {
            merge_host_and_target_docs(builder, &artifacts, &host_doc_dir, &target_doc_dir);
            merge_rustdoc_cci(builder, build_compiler, &artifacts.json_files, &target_doc_dir);
        }

        // We open rustc_middle as the default if invoked as `x.py doc --open RELEASES.md`
        // with no particular explicit doc requested (e.g. library/core).
        if builder.was_invoked_explicitly::<Self>(Kind::Doc) {
            let index = if builder.paths.iter().any(|path| path.ends_with("compiler")) {
                // For `x.py doc compiler --open`, open `rustc_middle` by default.
                target_doc_dir.join("rustc_middle").join("index.html")
            } else if let Some(krate) = self.crates.first() {
                // Let's open the first crate documentation page:
                target_doc_dir.join(krate).join("index.html")
            } else {
                target_doc_dir.clone()
            };
            builder.open_in_browser(index);
        }

        BuiltDocs { out_dir: target_doc_dir, artifacts }
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::doc("rustc", self.target).built_by(self.build_compiler))
    }
}

/// Stores generated documentation artifacts.
#[derive(Clone, Debug)]
struct DocArtifacts {
    /// Directories with HTML docs for host (proc-macro) crates.
    host_dirs: Vec<PathBuf>,
    /// Directories with HTML docs for target crates.
    target_dirs: Vec<PathBuf>,
    /// JSON files used to create the final CCI index
    json_files: Vec<PathBuf>,
}

impl DocArtifacts {
    /// Ensure that all passed crates were documented.
    fn sanity_check_crates<S>(&self, builder: &Builder<'_>, crates: impl Iterator<Item = S>)
    where
        S: AsRef<str>,
    {
        if builder.config.dry_run() {
            return;
        }
        let crate_names: HashSet<&str> = self
            .host_dirs
            .iter()
            .chain(self.target_dirs.iter())
            .filter_map(|d| d.file_name().and_then(|d| d.to_str()))
            .collect();
        for krate in crates {
            let krate = krate.as_ref();
            let krate = normalize_doc_crate_name(krate);
            if !crate_names.contains(krate.as_str()) {
                eprintln!("ERROR: crate {krate} was not documented!");
                exit_process(1);
            }
        }
    }
}

/// Run `cargo doc` and gather generated documentation artifacts.
fn create_docs_and_gather_artifacts(builder: &Builder<'_>, cargo: builder::Cargo) -> DocArtifacts {
    let mut json_files = vec![];
    let mut host_dirs = vec![];
    let mut target_dirs = vec![];
    stream_cargo(builder, cargo, vec![], &mut |msg| {
        let CargoMessage::CompilerArtifact { filenames, target } = msg else {
            return;
        };
        if !target.doc {
            return;
        }
        // Note: An alternative way to check host docs would be to check whether the generated
        // output is a child of the host doc directory (which we would have to pass to this
        // function).
        let is_host = target.crate_types.iter().any(|t| t == "proc-macro");
        for filename in filenames {
            let path = Path::new(filename.as_ref());
            let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
                continue;
            };
            let path = path.to_path_buf();
            if extension == "json" {
                json_files.push(path.to_path_buf());
            } else if extension == "html" {
                if is_host {
                    // doc/<crate>/index.html -> doc/<crate>
                    host_dirs.push(path.parent().unwrap().to_path_buf());
                } else {
                    target_dirs.push(path.parent().unwrap().to_path_buf());
                }
            }
        }
    });
    DocArtifacts { host_dirs, target_dirs, json_files }
}

/// Merge host and target documentation for a set of crates.
/// We pass `--target` when documenting, so Cargo will put the built documentation into two places:
/// - `target/doc` - contains documentation of host code, so proc macros
/// - `target/<target>/doc` - contains documentation of "normal" code
///
/// See https://github.com/rust-lang/cargo/issues/7677.
///
/// To produce a single unified documentation, we want to merge them together.
/// We do that by creating symlinks into the target doc dir that will point to the host doc
/// directories.
/// The target doc directory will then contain the combined docs.
fn merge_host_and_target_docs(
    builder: &Builder<'_>,
    docs: &DocArtifacts,
    host_doc_dir: &Path,
    target_doc_dir: &Path,
) {
    // Sanity check that there is no host/target overlap
    for dir in &docs.host_dirs {
        let name = dir.file_name().and_then(|d| d.to_str()).unwrap();
        if let Some(target_dir) = docs.target_dirs.iter().find_map(|d| {
            let dirname = d.file_name().and_then(|d| d.to_str())?;
            if dirname == name { Some(d) } else { None }
        }) {
            eprintln!(
                "ERROR: host docs directory `{name}` ({dir:?}) is also contained in target doc directory ({target_dir:?})"
            );
            exit_process(1);
        }
    }

    let target_src_dir = target_doc_dir.join("src");
    let host_src_dir = host_doc_dir.join("src");

    // Ideally, we would remove all previous symlinks here.
    // However, some of the tools actually share the same build docs directory, so we shouldn't do
    // that, otherwise they will invalidate one another.

    for host_docs_crate in &docs.host_dirs {
        let dir_name = host_docs_crate.file_name().unwrap().to_str().unwrap();
        // Normalize crate name
        let dir_name = normalize_doc_crate_name(dir_name);

        t!(symlink_dir(&builder.config, host_docs_crate, &target_doc_dir.join(&dir_name)));

        // Also symlink its source directory
        let target_src_out = target_src_dir.join(&dir_name);
        let host_src_out = host_src_dir.join(&dir_name);
        t!(symlink_dir(&builder.config, &host_src_out, &target_src_out));
    }

    // Sanity check that all directories contain some documentation
    for dir in docs.target_dirs.iter().chain(docs.host_dirs.iter()) {
        // Making sure the directory exists and is not empty.
        assert!(dir.exists(), "Doc directory {dir:?} does not exist");
        assert!(t!(dir.read_dir()).next().is_some(), "Doc directory {dir:?} is empty");
    }
}

/// Normalizes crate name to get a name that is used to generate documentation on disk.
/// Turns `rustc-main` into `rustc_main`.
fn normalize_doc_crate_name(name: &str) -> String {
    name.replace("-", "_")
}

macro_rules! tool_doc {
    (
        $tool: ident,
        $path: literal,
        mode = $mode:expr
        $(, is_library = $is_library:expr )?
        $(, crates = $crates:expr )?
        // Subset of nightly features that are allowed to be used when documenting
        $(, allow_features: $allow_features:expr )?
       ) => {
        #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $tool {
            build_compiler: Compiler,
            mode: Mode,
            target: TargetSelection,
        }

        impl $tool {
            fn new(builder: &Builder<'_>, target: TargetSelection) -> $tool {
                let build_compiler = match $mode {
                    Mode::ToolRustcPrivate => {
                        // Rustdoc needs the rustc sysroot available to build.
                        let compilers = RustcPrivateCompilers::new(builder, builder.top_stage, target);

                        // Build rustc docs so that we generate relative links.
                        builder.ensure(Rustc::from_build_compiler(builder, compilers.build_compiler(), target));
                        compilers.build_compiler()
                    }
                    Mode::ToolTarget => {
                        // when shipping multiple docs together in one folder,
                        // they all need to use the same rustdoc version
                        prepare_doc_compiler(builder, builder.host_target, builder.top_stage)
                    }
                    _ => {
                        panic!("Unexpected tool mode for documenting: {:?}", $mode);
                    }
                };
                $tool { build_compiler, mode: $mode, target }
            }
            fn crates() -> &'static [&'static str] {
                &$($crates)?[..]
            }
        }

        impl CommandLineStep for $tool {
            type Output = BuiltDocs;
            const IS_HOST: bool = true;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.path($path)
            }

            fn is_default_step(builder: &Builder<'_>) -> bool {
                builder.config.compiler_docs
            }

            fn make_run(run: RunConfig<'_>) {
                run.builder.ensure($tool::new(run.builder, run.target));
            }

            /// Generates documentation for a tool.
            ///
            /// This is largely just a wrapper around `cargo doc`.
            fn run(self, builder: &Builder<'_>) -> Self::Output {
                let mut source_type = SourceType::InTree;

                if let Some(submodule_path) = submodule_path_of(&builder, $path) {
                    source_type = SourceType::Submodule;
                    builder.require_submodule(&submodule_path, None);
                }

                let $tool { build_compiler, mode, target } = self;

                // Build cargo command.
                let mut cargo = prepare_tool_cargo(
                    builder,
                    build_compiler,
                    mode,
                    target,
                    Kind::Doc,
                    $path,
                    source_type,
                    &[],
                );
                let allow_features = {
                    let mut _value = "";
                    $( _value = $allow_features; )?
                    _value
                };

                if !allow_features.is_empty() {
                    cargo.allow_features(allow_features);
                }

                // Only include compiler crates, no dependencies of those, such as `libc`.
                cargo.arg("--no-deps");

                if false $(|| $is_library)? {
                    cargo.arg("--lib");
                }

                for krate in $tool::crates() {
                    cargo.arg("-p").arg(krate);
                }

                cargo.rustdocflag("--document-private-items");
                // Since we always pass --document-private-items, there's no need to warn about linking to private items.
                cargo.rustdocflag("-Arustdoc::private-intra-doc-links");
                cargo.rustdocflag("--enable-index-page");
                cargo.rustdocflag("--show-type-layout");
                cargo.rustdocflag("--generate-link-to-definition");

                let cargo_target_dir = builder.stage_out(build_compiler, mode);
                let target_doc_dir = cargo_target_dir.join(target).join("doc");
                let host_doc_dir = cargo_target_dir.join("doc");
                for krate in $tool::crates() {
                    let dir_name = normalize_doc_crate_name(krate);
                    t!(fs::create_dir_all(target_doc_dir.join(&*dir_name)));
                }

                let _guard = builder.msg(Kind::Doc, stringify!($tool).to_lowercase(), None, build_compiler, target);
                let artifacts = create_docs_and_gather_artifacts(builder, cargo);
                artifacts.sanity_check_crates(builder, $tool::crates().iter());

                if !builder.config.dry_run() {
                    merge_host_and_target_docs(builder, &artifacts, &host_doc_dir, &target_doc_dir);
                    merge_rustdoc_cci(builder, build_compiler, &artifacts.json_files, &target_doc_dir);
                }
                BuiltDocs { out_dir: target_doc_dir, artifacts }
            }

            fn metadata(&self) -> Option<StepMetadata> {
                Some(StepMetadata::doc(stringify!($tool), self.target).built_by(self.build_compiler))
            }
        }
    }
}

// NOTE: make sure to register these in `Builder::get_step_description`.
tool_doc!(
    BuildHelper,
    "src/build_helper",
    // ideally, this would use ToolBootstrap,
    // but we distribute these docs together in the same folder
    // as a bunch of stage1 tools, and you can't mix rustdoc versions
    // because that breaks cross-crate data (particularly search)
    mode = Mode::ToolTarget,
    is_library = true,
    crates = ["build_helper"]
);
tool_doc!(
    Rustdoc,
    "src/tools/rustdoc",
    mode = Mode::ToolRustcPrivate,
    crates = ["rustdoc", "rustdoc-json-types"]
);
tool_doc!(
    Rustfmt,
    "src/tools/rustfmt",
    mode = Mode::ToolRustcPrivate,
    crates = ["rustfmt-nightly", "rustfmt-config_proc_macro"]
);
tool_doc!(
    Clippy,
    "src/tools/clippy",
    mode = Mode::ToolRustcPrivate,
    crates = ["clippy_config", "clippy_utils"]
);
tool_doc!(Miri, "src/tools/miri", mode = Mode::ToolRustcPrivate, crates = ["miri"]);
tool_doc!(
    Cargo,
    "src/tools/cargo",
    mode = Mode::ToolTarget,
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
    ],
    // Required because of the im-rc dependency of Cargo, which automatically opts into the
    // "specialization" feature in its build script when it detects a nightly toolchain.
    allow_features: "specialization"
);
tool_doc!(Tidy, "src/tools/tidy", mode = Mode::ToolTarget, crates = ["tidy"]);
tool_doc!(
    Bootstrap,
    "src/bootstrap",
    mode = Mode::ToolTarget,
    is_library = true,
    crates = ["bootstrap"]
);
tool_doc!(
    RunMakeSupport,
    "src/tools/run-make-support",
    mode = Mode::ToolTarget,
    is_library = true,
    crates = ["run_make_support"]
);
tool_doc!(
    Compiletest,
    "src/tools/compiletest",
    mode = Mode::ToolTarget,
    is_library = true,
    crates = ["compiletest"]
);

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ErrorIndex {
    compilers: RustcPrivateCompilers,
}

impl CommandLineStep for ErrorIndex {
    type Output = ();
    const IS_HOST: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/error_index_generator")
    }

    fn is_default_step(builder: &Builder<'_>) -> bool {
        builder.config.docs
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(ErrorIndex {
            compilers: RustcPrivateCompilers::new(run.builder, run.builder.top_stage, run.target),
        });
    }

    /// Generates the HTML rendered error-index by running the
    /// `error_index_generator` tool.
    fn run(self, builder: &Builder<'_>) {
        builder.info(&format!("Documenting error index ({})", self.compilers.target()));
        let out = builder.doc_out(self.compilers.target());
        t!(fs::create_dir_all(&out));
        tool::ErrorIndex::command(builder, self.compilers)
            .arg("html")
            .arg(&out)
            .arg(&builder.version)
            .run(builder);

        let index = out.join("error-index.html");
        builder.maybe_open_in_browser::<Self>(index);
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(
            StepMetadata::doc("error-index", self.compilers.target())
                .built_by(self.compilers.build_compiler()),
        )
    }
}

/// Runs the `unstable-book-gen` tool and returns a path to the generate unstable book markdown
/// files.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct UnstableBookGen {
    build_compiler: Compiler,
    target: TargetSelection,
}

impl CommandLineStep for UnstableBookGen {
    type Output = PathBuf;
    const IS_HOST: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/unstable-book-gen")
    }

    fn is_default_step(builder: &Builder<'_>) -> bool {
        builder.config.docs
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(UnstableBookGen {
            build_compiler: prepare_doc_compiler(run.builder, run.target, run.builder.top_stage),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let target = self.target;
        let rustc_path = builder.rustc(self.build_compiler);

        builder.info(&format!("Generating unstable book md files ({target})"));
        let out = builder.out.join(target).join("md-doc").join("unstable-book");
        builder.create_dir(&out);
        builder.remove_dir(&out);
        let mut cmd = builder.tool_cmd(Tool::UnstableBookGen);
        cmd.arg(builder.src.join("library"));
        cmd.arg(builder.src.join("compiler"));
        cmd.arg(builder.src.join("src"));
        cmd.arg(rustc_path);
        cmd.arg(&out);

        // Running rustc requires the library path if rust.rpath = false
        // or any other libraries are in a custom location.
        builder.add_rustc_lib_path(self.build_compiler, &mut cmd);

        cmd.run(builder);
        out
    }
}

/// Builds the Rust compiler book.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustcBook {
    build_compiler: Compiler,
    target: TargetSelection,
    /// Test that the examples of lints in the book produce the correct lints in the expected
    /// format.
    validate: bool,
}

impl RustcBook {
    pub fn validate(build_compiler: Compiler, target: TargetSelection) -> Self {
        Self { build_compiler, target, validate: true }
    }
}

impl CommandLineStep for RustcBook {
    type Output = ();
    const IS_HOST: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/doc/rustc")
    }

    fn is_default_step(builder: &Builder<'_>) -> bool {
        builder.config.docs
    }

    fn make_run(run: RunConfig<'_>) {
        // Bump the stage to 2, because the rustc book requires an in-tree compiler.
        // At the same time, since this step is enabled by default, we don't want `x doc` to fail
        // in stage 1.
        let stage = if run.builder.config.is_explicit_stage() || run.builder.top_stage >= 2 {
            run.builder.top_stage
        } else {
            2
        };

        run.builder.ensure(RustcBook {
            build_compiler: prepare_doc_compiler(run.builder, run.target, stage),
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
        // FIXME: Temporary workaround for https://github.com/rust-lang/rust/issues/158378
        // Make sure this workaround doesn't break unit tests on the affected host.
        if cfg!(not(test)) && self.target == "i686-pc-windows-msvc" {
            eprintln!("WARNING: Skipping rustc book build to work around #158378");
            return;
        }

        let out_base = builder.out.join(self.target).join("md-doc").join("rustc");
        t!(fs::create_dir_all(&out_base));
        let out_listing = out_base.join("src/lints");
        builder.cp_link_r(&builder.src.join("src/doc/rustc"), &out_base);
        builder.info(&format!("Generating lint docs ({})", self.target));

        let rustc = builder.rustc(self.build_compiler);
        // The tool runs `rustc` for extracting output examples, so it needs a
        // functional sysroot.
        builder.std(self.build_compiler, self.target);
        let mut cmd = builder.tool_cmd(Tool::LintDocs);
        cmd.arg("--build-rustc-stage");
        cmd.arg(self.build_compiler.stage.to_string());
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
        builder.add_rustc_lib_path(self.build_compiler, &mut cmd);
        let doc_generator_guard =
            builder.msg(Kind::Run, "lint-docs", None, self.build_compiler, self.target);
        cmd.run(builder);
        drop(doc_generator_guard);

        // Run rustbook/mdbook to generate the HTML pages.
        builder.ensure(RustbookSrc {
            target: self.target,
            name: "rustc".to_owned(),
            src: out_base,
            parent: Some(self),
            languages: vec![],
            build_compiler: None,
        });
    }
}

/// Documents the reference.
/// It has to always be done using a stage 1+ compiler, because it references in-tree
/// compiler/stdlib concepts.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Reference {
    build_compiler: Compiler,
    target: TargetSelection,
}

impl CommandLineStep for Reference {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/doc/reference")
    }

    fn is_default_step(builder: &Builder<'_>) -> bool {
        builder.config.docs
    }

    fn make_run(run: RunConfig<'_>) {
        // Bump the stage to 2, because the reference requires an in-tree compiler.
        // At the same time, since this step is enabled by default, we don't want `x doc` to fail
        // in stage 1.
        // FIXME: create a shared method on builder for auto-bumping, and print some warning when
        // it happens.
        let stage = if run.builder.config.is_explicit_stage() || run.builder.top_stage >= 2 {
            run.builder.top_stage
        } else {
            2
        };

        run.builder.ensure(Reference {
            build_compiler: prepare_doc_compiler(run.builder, run.target, stage),
            target: run.target,
        });
    }

    /// Builds the reference book.
    fn run(self, builder: &Builder<'_>) {
        builder.require_submodule("src/doc/reference", None);

        // This is needed for generating links to the standard library using
        // the mdbook-spec plugin.
        builder.std(self.build_compiler, builder.config.host_target);

        // Run rustbook/mdbook to generate the HTML pages.
        builder.ensure(RustbookSrc {
            target: self.target,
            name: "reference".to_owned(),
            src: builder.src.join("src/doc/reference"),
            build_compiler: Some(self.build_compiler),
            parent: Some(self),
            languages: vec![],
        });
    }
}
