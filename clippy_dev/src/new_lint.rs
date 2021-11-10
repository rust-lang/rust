use crate::clippy_project_root;
use indoc::indoc;
use std::fs::{self, OpenOptions};
use std::io::prelude::*;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

struct LintData<'a> {
    pass: &'a str,
    name: &'a str,
    category: &'a str,
    project_root: PathBuf,
}

trait Context {
    fn context<C: AsRef<str>>(self, text: C) -> Self;
}

impl<T> Context for io::Result<T> {
    fn context<C: AsRef<str>>(self, text: C) -> Self {
        match self {
            Ok(t) => Ok(t),
            Err(e) => {
                let message = format!("{}: {}", text.as_ref(), e);
                Err(io::Error::new(ErrorKind::Other, message))
            },
        }
    }
}

/// Creates the files required to implement and test a new lint and runs `update_lints`.
///
/// # Errors
///
/// This function errors out if the files couldn't be created or written to.
pub fn create(pass: Option<&str>, lint_name: Option<&str>, category: Option<&str>, msrv: bool) -> io::Result<()> {
    let lint = LintData {
        pass: pass.expect("`pass` argument is validated by clap"),
        name: lint_name.expect("`name` argument is validated by clap"),
        category: category.expect("`category` argument is validated by clap"),
        project_root: clippy_project_root(),
    };

    create_lint(&lint, msrv).context("Unable to create lint implementation")?;
    create_test(&lint).context("Unable to create a test for the new lint")?;
    add_lint(&lint, msrv).context("Unable to add lint to clippy_lints/src/lib.rs")
}

fn create_lint(lint: &LintData<'_>, enable_msrv: bool) -> io::Result<()> {
    let lint_contents = get_lint_file_contents(lint, enable_msrv);

    let lint_path = format!("clippy_lints/src/{}.rs", lint.name);
    write_file(lint.project_root.join(&lint_path), lint_contents.as_bytes())
}

fn create_test(lint: &LintData<'_>) -> io::Result<()> {
    fn create_project_layout<P: Into<PathBuf>>(lint_name: &str, location: P, case: &str, hint: &str) -> io::Result<()> {
        let mut path = location.into().join(case);
        fs::create_dir(&path)?;
        write_file(path.join("Cargo.toml"), get_manifest_contents(lint_name, hint))?;

        path.push("src");
        fs::create_dir(&path)?;
        let header = format!("// compile-flags: --crate-name={}", lint_name);
        write_file(path.join("main.rs"), get_test_file_contents(lint_name, Some(&header)))?;

        Ok(())
    }

    if lint.category == "cargo" {
        let relative_test_dir = format!("tests/ui-cargo/{}", lint.name);
        let test_dir = lint.project_root.join(relative_test_dir);
        fs::create_dir(&test_dir)?;

        create_project_layout(lint.name, &test_dir, "fail", "Content that triggers the lint goes here")?;
        create_project_layout(lint.name, &test_dir, "pass", "This file should not trigger the lint")
    } else {
        let test_path = format!("tests/ui/{}.rs", lint.name);
        let test_contents = get_test_file_contents(lint.name, None);
        write_file(lint.project_root.join(test_path), test_contents)
    }
}

fn add_lint(lint: &LintData<'_>, enable_msrv: bool) -> io::Result<()> {
    let path = "clippy_lints/src/lib.rs";
    let mut lib_rs = fs::read_to_string(path).context("reading")?;

    let comment_start = lib_rs.find("// add lints here,").expect("Couldn't find comment");

    let new_lint = if enable_msrv {
        format!(
            "store.register_{lint_pass}_pass(move || Box::new({module_name}::{camel_name}::new(msrv)));\n    ",
            lint_pass = lint.pass,
            module_name = lint.name,
            camel_name = to_camel_case(lint.name),
        )
    } else {
        format!(
            "store.register_{lint_pass}_pass(|| Box::new({module_name}::{camel_name}));\n    ",
            lint_pass = lint.pass,
            module_name = lint.name,
            camel_name = to_camel_case(lint.name),
        )
    };

    lib_rs.insert_str(comment_start, &new_lint);

    fs::write(path, lib_rs).context("writing")
}

fn write_file<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    fn inner(path: &Path, contents: &[u8]) -> io::Result<()> {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(contents)
    }

    inner(path.as_ref(), contents.as_ref()).context(format!("writing to file: {}", path.as_ref().display()))
}

fn to_camel_case(name: &str) -> String {
    name.split('_')
        .map(|s| {
            if s.is_empty() {
                String::from("")
            } else {
                [&s[0..1].to_uppercase(), &s[1..]].concat()
            }
        })
        .collect()
}

fn get_stabilisation_version() -> String {
    let mut command = cargo_metadata::MetadataCommand::new();
    command.no_deps();
    if let Ok(metadata) = command.exec() {
        if let Some(pkg) = metadata.packages.iter().find(|pkg| pkg.name == "clippy") {
            return format!("{}.{}.0", pkg.version.minor, pkg.version.patch);
        }
    }

    String::from("<TODO set version(see doc/adding_lints.md)>")
}

fn get_test_file_contents(lint_name: &str, header_commands: Option<&str>) -> String {
    let mut contents = format!(
        indoc! {"
            #![warn(clippy::{})]

            fn main() {{
                // test code goes here
            }}
        "},
        lint_name
    );

    if let Some(header) = header_commands {
        contents = format!("{}\n{}", header, contents);
    }

    contents
}

fn get_manifest_contents(lint_name: &str, hint: &str) -> String {
    format!(
        indoc! {r#"
            # {}

            [package]
            name = "{}"
            version = "0.1.0"
            publish = false

            [workspace]
        "#},
        hint, lint_name
    )
}

fn get_lint_file_contents(lint: &LintData<'_>, enable_msrv: bool) -> String {
    let mut result = String::new();

    let (pass_type, pass_lifetimes, pass_import, context_import) = match lint.pass {
        "early" => ("EarlyLintPass", "", "use rustc_ast::ast::*;", "EarlyContext"),
        "late" => ("LateLintPass", "<'_>", "use rustc_hir::*;", "LateContext"),
        _ => {
            unreachable!("`pass_type` should only ever be `early` or `late`!");
        },
    };

    let version = get_stabilisation_version();
    let lint_name = lint.name;
    let category = lint.category;
    let name_camel = to_camel_case(lint.name);
    let name_upper = lint_name.to_uppercase();

    result.push_str(&if enable_msrv {
        format!(
            indoc! {"
                use clippy_utils::msrvs;
                {pass_import}
                use rustc_lint::{{{context_import}, {pass_type}, LintContext}};
                use rustc_semver::RustcVersion;
                use rustc_session::{{declare_tool_lint, impl_lint_pass}};

            "},
            pass_type = pass_type,
            pass_import = pass_import,
            context_import = context_import,
        )
    } else {
        format!(
            indoc! {"
                {pass_import}
                use rustc_lint::{{{context_import}, {pass_type}}};
                use rustc_session::{{declare_lint_pass, declare_tool_lint}};

            "},
            pass_import = pass_import,
            pass_type = pass_type,
            context_import = context_import
        )
    });

    result.push_str(&format!(
        indoc! {r#"
            declare_clippy_lint! {{
                /// ### What it does
                ///
                /// ### Why is this bad?
                ///
                /// ### Example
                /// ```rust
                /// // example code where clippy issues a warning
                /// ```
                /// Use instead:
                /// ```rust
                /// // example code which does not raise clippy warning
                /// ```
                #[clippy::version = "{version}"]
                pub {name_upper},
                {category},
                "default lint description"
            }}
        "#},
        version = version,
        name_upper = name_upper,
        category = category,
    ));

    result.push_str(&if enable_msrv {
        format!(
            indoc! {"
                pub struct {name_camel} {{
                    msrv: Option<RustcVersion>,
                }}

                impl {name_camel} {{
                    #[must_use]
                    pub fn new(msrv: Option<RustcVersion>) -> Self {{
                        Self {{ msrv }}
                    }}
                }}

                impl_lint_pass!({name_camel} => [{name_upper}]);

                impl {pass_type}{pass_lifetimes} for {name_camel} {{
                    extract_msrv_attr!({context_import});
                }}

                // TODO: Add MSRV level to `clippy_utils/src/msrvs.rs` if needed.
                // TODO: Add MSRV test to `tests/ui/min_rust_version_attr.rs`.
                // TODO: Update msrv config comment in `clippy_lints/src/utils/conf.rs`
            "},
            pass_type = pass_type,
            pass_lifetimes = pass_lifetimes,
            name_upper = name_upper,
            name_camel = name_camel,
            context_import = context_import,
        )
    } else {
        format!(
            indoc! {"
                declare_lint_pass!({name_camel} => [{name_upper}]);

                impl {pass_type}{pass_lifetimes} for {name_camel} {{}}
            "},
            pass_type = pass_type,
            pass_lifetimes = pass_lifetimes,
            name_upper = name_upper,
            name_camel = name_camel,
        )
    });

    result
}

#[test]
fn test_camel_case() {
    let s = "a_lint";
    let s2 = to_camel_case(s);
    assert_eq!(s2, "ALint");

    let name = "a_really_long_new_lint";
    let name2 = to_camel_case(name);
    assert_eq!(name2, "AReallyLongNewLint");

    let name3 = "lint__name";
    let name4 = to_camel_case(name3);
    assert_eq!(name4, "LintName");
}
