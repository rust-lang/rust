use crate::clippy_project_root;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::prelude::*;
use std::io::ErrorKind;
use std::path::Path;

/// Creates files required to implement and test a new lint and runs `update_lints`.
///
/// # Errors
///
/// This function errors, if the files couldn't be created
pub fn create(pass: Option<&str>, lint_name: Option<&str>, category: Option<&str>) -> Result<(), io::Error> {
    let pass = pass.expect("`pass` argument is validated by clap");
    let lint_name = lint_name.expect("`name` argument is validated by clap");
    let category = category.expect("`category` argument is validated by clap");

    match open_files(lint_name) {
        Ok((mut test_file, mut lint_file)) => {
            let (pass_type, pass_lifetimes, pass_import, context_import) = match pass {
                "early" => ("EarlyLintPass", "", "use rustc_ast::ast::*;", "EarlyContext"),
                "late" => ("LateLintPass", "<'_, '_>", "use rustc_hir::*;", "LateContext"),
                _ => {
                    unreachable!("`pass_type` should only ever be `early` or `late`!");
                },
            };

            let camel_case_name = to_camel_case(lint_name);

            if let Err(e) = test_file.write_all(get_test_file_contents(lint_name).as_bytes()) {
                return Err(io::Error::new(
                    ErrorKind::Other,
                    format!("Could not write to test file: {}", e),
                ));
            };

            if let Err(e) = lint_file.write_all(
                get_lint_file_contents(
                    pass_type,
                    pass_lifetimes,
                    lint_name,
                    &camel_case_name,
                    category,
                    pass_import,
                    context_import,
                )
                .as_bytes(),
            ) {
                return Err(io::Error::new(
                    ErrorKind::Other,
                    format!("Could not write to lint file: {}", e),
                ));
            }
            Ok(())
        },
        Err(e) => Err(io::Error::new(
            ErrorKind::Other,
            format!("Unable to create lint: {}", e),
        )),
    }
}

fn open_files(lint_name: &str) -> Result<(File, File), io::Error> {
    let project_root = clippy_project_root();

    let test_file_path = project_root.join("tests").join("ui").join(format!("{}.rs", lint_name));
    let lint_file_path = project_root
        .join("clippy_lints")
        .join("src")
        .join(format!("{}.rs", lint_name));

    if Path::new(&test_file_path).exists() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            format!("test file {:?} already exists", test_file_path),
        ));
    }
    if Path::new(&lint_file_path).exists() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            format!("lint file {:?} already exists", lint_file_path),
        ));
    }

    let test_file = OpenOptions::new().write(true).create_new(true).open(test_file_path)?;
    let lint_file = OpenOptions::new().write(true).create_new(true).open(lint_file_path)?;

    Ok((test_file, lint_file))
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

fn get_test_file_contents(lint_name: &str) -> String {
    format!(
        "#![warn(clippy::{})]

fn main() {{
    // test code goes here
}}
",
        lint_name
    )
}

fn get_lint_file_contents(
    pass_type: &str,
    pass_lifetimes: &str,
    lint_name: &str,
    camel_case_name: &str,
    category: &str,
    pass_import: &str,
    context_import: &str,
) -> String {
    format!(
        "use rustc_lint::{{{type}, {context_import}}};
use rustc_session::{{declare_lint_pass, declare_tool_lint}};
{pass_import}

declare_clippy_lint! {{
    /// **What it does:**
    ///
    /// **Why is this bad?**
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// // example code where clippy issues a warning
    /// ```
    /// Use instead:
    /// ```rust
    /// // example code which does not raise clippy warning
    /// ```
    pub {name_upper},
    {category},
    \"default lint description\"
}}

declare_lint_pass!({name_camel} => [{name_upper}]);

impl {type}{lifetimes} for {name_camel} {{}}
",
        type=pass_type,
        lifetimes=pass_lifetimes,
        name_upper=lint_name.to_uppercase(),
        name_camel=camel_case_name,
        category=category,
        pass_import=pass_import,
        context_import=context_import
    )
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
