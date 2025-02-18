//! Collection of book-related tests.
//!
//! There are two general categories here:
//! 1. Tests that try to run documentation tests (through `mdbook test` via `rustdoc`, or via
//!    `rustdoc` on individual markdown files directly).
//! 2. `src/tools/error_index_generator` is special in that its test step involves building a
//!    suitably-staged sysroot (which matches what is ordinarily used to build `rustdoc`) to run
//!    tests in the error index.

use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::{env, fs, iter};

use crate::core::build_steps::compile::{self, run_cargo};
use crate::core::build_steps::tool::{self, SourceType, Tool};
use crate::core::build_steps::toolstate::ToolState;
use crate::core::builder::{Builder, Compiler, Kind, RunConfig, ShouldRun, Step};
use crate::utils::build_stamp::BuildStamp;
use crate::utils::helpers;
use crate::{Mode, t};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BookTest {
    compiler: Compiler,
    path: PathBuf,
    name: &'static str,
    is_ext_doc: bool,
    dependencies: Vec<&'static str>,
}

impl Step for BookTest {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Runs the documentation tests for a book in `src/doc`.
    ///
    /// This uses the `rustdoc` that sits next to `compiler`.
    fn run(self, builder: &Builder<'_>) {
        // External docs are different from local because:
        // - Some books need pre-processing by mdbook before being tested.
        // - They need to save their state to toolstate.
        // - They are only tested on the "checktools" builders.
        //
        // The local docs are tested by default, and we don't want to pay the cost of building
        // mdbook, so they use `rustdoc --test` directly. Also, the unstable book is special because
        // SUMMARY.md is generated, so it is easier to just run `rustdoc` on its files.
        if self.is_ext_doc {
            self.run_ext_doc(builder);
        } else {
            self.run_local_doc(builder);
        }
    }
}

impl BookTest {
    /// This runs the equivalent of `mdbook test` (via the rustbook wrapper) which in turn runs
    /// `rustdoc --test` on each file in the book.
    fn run_ext_doc(self, builder: &Builder<'_>) {
        let compiler = self.compiler;

        builder.ensure(compile::Std::new(compiler, compiler.host));

        // mdbook just executes a binary named "rustdoc", so we need to update PATH so that it
        // points to our rustdoc.
        let mut rustdoc_path = builder.rustdoc(compiler);
        rustdoc_path.pop();
        let old_path = env::var_os("PATH").unwrap_or_default();
        let new_path = env::join_paths(iter::once(rustdoc_path).chain(env::split_paths(&old_path)))
            .expect("could not add rustdoc to PATH");

        let mut rustbook_cmd = builder.tool_cmd(Tool::Rustbook);
        let path = builder.src.join(&self.path);
        // Books often have feature-gated example text.
        rustbook_cmd.env("RUSTC_BOOTSTRAP", "1");
        rustbook_cmd.env("PATH", new_path).arg("test").arg(path);

        // Books may also need to build dependencies. For example, `TheBook` has code samples which
        // use the `trpl` crate. For the `rustdoc` invocation to find them them successfully, they
        // need to be built first and their paths used to generate the
        let libs = if !self.dependencies.is_empty() {
            let mut lib_paths = vec![];
            for dep in self.dependencies {
                let mode = Mode::ToolRustc;
                let target = builder.config.build;
                let cargo = tool::prepare_tool_cargo(
                    builder,
                    compiler,
                    mode,
                    target,
                    Kind::Build,
                    dep,
                    SourceType::Submodule,
                    &[],
                );

                let stamp = BuildStamp::new(&builder.cargo_out(compiler, mode, target))
                    .with_prefix(PathBuf::from(dep).file_name().and_then(|v| v.to_str()).unwrap());

                let output_paths = run_cargo(builder, cargo, vec![], &stamp, vec![], false, false);
                let directories = output_paths
                    .into_iter()
                    .filter_map(|p| p.parent().map(ToOwned::to_owned))
                    .fold(HashSet::new(), |mut set, dir| {
                        set.insert(dir);
                        set
                    });

                lib_paths.extend(directories);
            }
            lib_paths
        } else {
            vec![]
        };

        if !libs.is_empty() {
            let paths = libs
                .into_iter()
                .map(|path| path.into_os_string())
                .collect::<Vec<OsString>>()
                .join(OsStr::new(","));
            rustbook_cmd.args([OsString::from("--library-path"), paths]);
        }

        builder.add_rust_test_threads(&mut rustbook_cmd);
        let _guard = builder.msg(
            Kind::Test,
            compiler.stage,
            format_args!("mdbook {}", self.path.display()),
            compiler.host,
            compiler.host,
        );
        let _time = helpers::timeit(builder);
        let toolstate = if rustbook_cmd.delay_failure().run(builder) {
            ToolState::TestPass
        } else {
            ToolState::TestFail
        };
        builder.save_toolstate(self.name, toolstate);
    }

    /// This runs `rustdoc --test` on all `.md` files in the path.
    fn run_local_doc(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let host = self.compiler.host;

        builder.ensure(compile::Std::new(compiler, host));

        let _guard =
            builder.msg(Kind::Test, compiler.stage, format!("book {}", self.name), host, host);

        // Do a breadth-first traversal of the `src/doc` directory and just run tests for all files
        // that end in `*.md`
        let mut stack = vec![builder.src.join(self.path)];
        let _time = helpers::timeit(builder);
        let mut files = Vec::new();
        while let Some(p) = stack.pop() {
            if p.is_dir() {
                stack.extend(t!(p.read_dir()).map(|p| t!(p).path()));
                continue;
            }

            if p.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }

            files.push(p);
        }

        files.sort();

        for file in files {
            markdown_test(builder, compiler, &file);
        }
    }
}

macro_rules! test_book {
    ($(
        $name:ident, $path:expr, $book_name:expr,
        default=$default:expr
        $(,submodules = $submodules:expr)?
        $(,dependencies=$dependencies:expr)?
        ;
    )+) => {
        $(
            #[derive(Debug, Clone, PartialEq, Eq, Hash)]
            pub struct $name {
                compiler: Compiler,
            }

            impl Step for $name {
                type Output = ();
                const DEFAULT: bool = $default;
                const ONLY_HOSTS: bool = true;

                fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                    run.path($path)
                }

                fn make_run(run: RunConfig<'_>) {
                    run.builder.ensure($name {
                        compiler: run.builder.compiler(run.builder.top_stage, run.target),
                    });
                }

                fn run(self, builder: &Builder<'_>) {
                    $(
                        for submodule in $submodules {
                            builder.require_submodule(submodule, None);
                        }
                    )*

                    let dependencies = vec![];
                    $(
                        let mut dependencies = dependencies;
                        for dep in $dependencies {
                            dependencies.push(dep);
                        }
                    )?

                    builder.ensure(BookTest {
                        compiler: self.compiler,
                        path: PathBuf::from($path),
                        name: $book_name,
                        is_ext_doc: !$default,
                        dependencies,
                    });
                }
            }
        )+
    }
}

test_book!(
    Nomicon, "src/doc/nomicon", "nomicon", default=false, submodules=["src/doc/nomicon"];
    Reference, "src/doc/reference", "reference", default=false, submodules=["src/doc/reference"];
    RustdocBook, "src/doc/rustdoc", "rustdoc", default=true;
    RustcBook, "src/doc/rustc", "rustc", default=true;
    RustByExample, "src/doc/rust-by-example", "rust-by-example", default=false, submodules=["src/doc/rust-by-example"];
    EmbeddedBook, "src/doc/embedded-book", "embedded-book", default=false, submodules=["src/doc/embedded-book"];
    TheBook, "src/doc/book", "book", default=false, submodules=["src/doc/book"], dependencies=["src/doc/book/packages/trpl"];
    UnstableBook, "src/doc/unstable-book", "unstable-book", default=true;
    EditionGuide, "src/doc/edition-guide", "edition-guide", default=false, submodules=["src/doc/edition-guide"];
);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ErrorIndex {
    compiler: Compiler,
}

impl Step for ErrorIndex {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        // Also add `error-index` here since that is what appears in the error message
        // when this fails.
        run.path("src/tools/error_index_generator").alias("error-index")
    }

    fn make_run(run: RunConfig<'_>) {
        // error_index_generator depends on librustdoc. Use the compiler that is normally used to
        // build rustdoc for other tests (like compiletest tests in tests/rustdoc) so that it shares
        // the same artifacts.
        let compiler = run.builder.compiler(run.builder.top_stage, run.builder.config.build);
        run.builder.ensure(ErrorIndex { compiler });
    }

    /// Runs the error index generator tool to execute the tests located in the error index.
    ///
    /// The `error_index_generator` tool lives in `src/tools` and is used to generate a markdown
    /// file from the error indexes of the code base which is then passed to `rustdoc --test`.
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;

        let dir = builder.test_out(compiler.host);
        t!(fs::create_dir_all(&dir));
        let output = dir.join("error-index.md");

        let mut tool = tool::ErrorIndex::command(builder);
        tool.arg("markdown").arg(&output);

        let guard =
            builder.msg(Kind::Test, compiler.stage, "error-index", compiler.host, compiler.host);
        let _time = helpers::timeit(builder);
        tool.run_capture(builder);
        drop(guard);
        // The tests themselves need to link to std, so make sure it is available.
        builder.ensure(compile::Std::new(compiler, compiler.host));
        markdown_test(builder, compiler, &output);
    }
}

fn markdown_test(builder: &Builder<'_>, compiler: Compiler, markdown: &Path) -> bool {
    if let Ok(contents) = fs::read_to_string(markdown) {
        if !contents.contains("```") {
            return true;
        }
    }

    builder.verbose(|| println!("doc tests for: {}", markdown.display()));
    let mut cmd = builder.rustdoc_cmd(compiler);
    builder.add_rust_test_threads(&mut cmd);
    // allow for unstable options such as new editions
    cmd.arg("-Z");
    cmd.arg("unstable-options");
    cmd.arg("--test");
    cmd.arg(markdown);
    cmd.env("RUSTC_BOOTSTRAP", "1");

    let test_args = builder.config.test_args().join(" ");
    cmd.arg("--test-args").arg(test_args);

    cmd = cmd.delay_failure();
    if !builder.config.verbose_tests {
        cmd.run_capture(builder).is_success()
    } else {
        cmd.run(builder)
    }
}
