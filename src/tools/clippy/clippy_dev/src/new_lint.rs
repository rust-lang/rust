use crate::utils::{RustSearcher, Token, Version};
use clap::ValueEnum;
use indoc::{formatdoc, writedoc};
use std::fmt::{self, Write as _};
use std::fs::{self, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, ValueEnum)]
pub enum Pass {
    Early,
    Late,
}

impl fmt::Display for Pass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Pass::Early => "early",
            Pass::Late => "late",
        })
    }
}

struct LintData<'a> {
    clippy_version: Version,
    pass: Pass,
    name: &'a str,
    category: &'a str,
    ty: Option<&'a str>,
}

trait Context {
    fn context<C: AsRef<str>>(self, text: C) -> Self;
}

impl<T> Context for io::Result<T> {
    fn context<C: AsRef<str>>(self, text: C) -> Self {
        match self {
            Ok(t) => Ok(t),
            Err(e) => {
                let message = format!("{}: {e}", text.as_ref());
                Err(io::Error::other(message))
            },
        }
    }
}

/// Creates the files required to implement and test a new lint and runs `update_lints`.
///
/// # Errors
///
/// This function errors out if the files couldn't be created or written to.
pub fn create(
    clippy_version: Version,
    pass: Pass,
    name: &str,
    category: &str,
    mut ty: Option<&str>,
    msrv: bool,
) -> io::Result<()> {
    if category == "cargo" && ty.is_none() {
        // `cargo` is a special category, these lints should always be in `clippy_lints/src/cargo`
        ty = Some("cargo");
    }

    let lint = LintData {
        clippy_version,
        pass,
        name,
        category,
        ty,
    };

    create_lint(&lint, msrv).context("Unable to create lint implementation")?;
    create_test(&lint, msrv).context("Unable to create a test for the new lint")?;

    if lint.ty.is_none() {
        add_lint(&lint, msrv).context("Unable to add lint to clippy_lints/src/lib.rs")?;
    }

    if pass == Pass::Early {
        println!(
            "\n\
            NOTE: Use a late pass unless you need something specific from\n\
            an early pass, as they lack many features and utilities"
        );
    }

    Ok(())
}

fn create_lint(lint: &LintData<'_>, enable_msrv: bool) -> io::Result<()> {
    if let Some(ty) = lint.ty {
        create_lint_for_ty(lint, enable_msrv, ty)
    } else {
        let lint_contents = get_lint_file_contents(lint, enable_msrv);
        let lint_path = format!("clippy_lints/src/{}.rs", lint.name);
        write_file(&lint_path, lint_contents.as_bytes())?;
        println!("Generated lint file: `{lint_path}`");

        Ok(())
    }
}

fn create_test(lint: &LintData<'_>, msrv: bool) -> io::Result<()> {
    fn create_project_layout<P: Into<PathBuf>>(
        lint_name: &str,
        location: P,
        case: &str,
        hint: &str,
        msrv: bool,
    ) -> io::Result<()> {
        let mut path = location.into().join(case);
        fs::create_dir(&path)?;
        write_file(path.join("Cargo.toml"), get_manifest_contents(lint_name, hint))?;

        path.push("src");
        fs::create_dir(&path)?;
        write_file(path.join("main.rs"), get_test_file_contents(lint_name, msrv))?;

        Ok(())
    }

    if lint.category == "cargo" {
        let test_dir = format!("tests/ui-cargo/{}", lint.name);
        fs::create_dir(&test_dir)?;

        create_project_layout(
            lint.name,
            &test_dir,
            "fail",
            "Content that triggers the lint goes here",
            msrv,
        )?;
        create_project_layout(
            lint.name,
            &test_dir,
            "pass",
            "This file should not trigger the lint",
            false,
        )?;

        println!("Generated test directories: `{test_dir}/pass`, `{test_dir}/fail`");
    } else {
        let test_path = format!("tests/ui/{}.rs", lint.name);
        let test_contents = get_test_file_contents(lint.name, msrv);
        write_file(&test_path, test_contents)?;

        println!("Generated test file: `{test_path}`");
    }

    Ok(())
}

fn add_lint(lint: &LintData<'_>, enable_msrv: bool) -> io::Result<()> {
    let path = "clippy_lints/src/lib.rs";
    let mut lib_rs = fs::read_to_string(path).context("reading")?;

    let comment_start = lib_rs.find("// add lints here,").expect("Couldn't find comment");
    let ctor_arg = if lint.pass == Pass::Late { "_" } else { "" };
    let lint_pass = lint.pass;
    let module_name = lint.name;
    let camel_name = to_camel_case(lint.name);

    let new_lint = if enable_msrv {
        format!(
            "store.register_{lint_pass}_pass(move |{ctor_arg}| Box::new({module_name}::{camel_name}::new(conf)));\n    ",
        )
    } else {
        format!("store.register_{lint_pass}_pass(|{ctor_arg}| Box::new({module_name}::{camel_name}));\n    ",)
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
                String::new()
            } else {
                [&s[0..1].to_uppercase(), &s[1..]].concat()
            }
        })
        .collect()
}

fn get_test_file_contents(lint_name: &str, msrv: bool) -> String {
    let mut test = formatdoc!(
        r"
        #![warn(clippy::{lint_name})]

        fn main() {{
            // test code goes here
        }}
    "
    );

    if msrv {
        let _ = writedoc!(
            test,
            r#"

                // TODO: set xx to the version one below the MSRV used by the lint, and yy to
                // the version used by the lint
                #[clippy::msrv = "1.xx"]
                fn msrv_1_xx() {{
                    // a simple example that would trigger the lint if the MSRV were met
                }}

                #[clippy::msrv = "1.yy"]
                fn msrv_1_yy() {{
                    // the same example as above
                }}
            "#
        );
    }

    test
}

fn get_manifest_contents(lint_name: &str, hint: &str) -> String {
    formatdoc!(
        r#"
        # {hint}

        [package]
        name = "{lint_name}"
        version = "0.1.0"
        publish = false

        [workspace]
    "#
    )
}

fn get_lint_file_contents(lint: &LintData<'_>, enable_msrv: bool) -> String {
    let mut result = String::new();

    let (pass_type, pass_lifetimes, pass_import, context_import) = match lint.pass {
        Pass::Early => ("EarlyLintPass", "", "use rustc_ast::ast::*;", "EarlyContext"),
        Pass::Late => ("LateLintPass", "<'_>", "use rustc_hir::*;", "LateContext"),
    };
    let (msrv_ty, msrv_ctor, extract_msrv) = match lint.pass {
        Pass::Early => (
            "MsrvStack",
            "MsrvStack::new(conf.msrv)",
            "\n    extract_msrv_attr!();\n",
        ),
        Pass::Late => ("Msrv", "conf.msrv", ""),
    };

    let lint_name = lint.name;
    let category = lint.category;
    let name_camel = to_camel_case(lint.name);
    let name_upper = lint_name.to_uppercase();

    if enable_msrv {
        let _: fmt::Result = writedoc!(
            result,
            r"
            use clippy_utils::msrvs::{{self, {msrv_ty}}};
            use clippy_config::Conf;
            {pass_import}
            use rustc_lint::{{{context_import}, {pass_type}}};
            use rustc_session::impl_lint_pass;

        "
        );
    } else {
        let _: fmt::Result = writedoc!(
            result,
            r"
            {pass_import}
            use rustc_lint::{{{context_import}, {pass_type}}};
            use rustc_session::declare_lint_pass;

        "
        );
    }

    let _: fmt::Result = writeln!(
        result,
        "{}",
        get_lint_declaration(lint.clippy_version, &name_upper, category)
    );

    if enable_msrv {
        let _: fmt::Result = writedoc!(
            result,
            r"
            pub struct {name_camel} {{
                msrv: {msrv_ty},
            }}

            impl {name_camel} {{
                pub fn new(conf: &'static Conf) -> Self {{
                    Self {{ msrv: {msrv_ctor} }}
                }}
            }}

            impl_lint_pass!({name_camel} => [{name_upper}]);

            impl {pass_type}{pass_lifetimes} for {name_camel} {{{extract_msrv}}}

            // TODO: Add MSRV level to `clippy_config/src/msrvs.rs` if needed.
            // TODO: Update msrv config comment in `clippy_config/src/conf.rs`
        "
        );
    } else {
        let _: fmt::Result = writedoc!(
            result,
            r"
            declare_lint_pass!({name_camel} => [{name_upper}]);

            impl {pass_type}{pass_lifetimes} for {name_camel} {{}}
        "
        );
    }

    result
}

fn get_lint_declaration(version: Version, name_upper: &str, category: &str) -> String {
    let justification_heading = if category == "restriction" {
        "Why restrict this?"
    } else {
        "Why is this bad?"
    };
    formatdoc!(
        r#"
            declare_clippy_lint! {{
                /// ### What it does
                ///
                /// ### {justification_heading}
                ///
                /// ### Example
                /// ```no_run
                /// // example code where clippy issues a warning
                /// ```
                /// Use instead:
                /// ```no_run
                /// // example code which does not raise clippy warning
                /// ```
                #[clippy::version = "{}"]
                pub {name_upper},
                {category},
                "default lint description"
            }}"#,
        version.rust_display(),
    )
}

fn create_lint_for_ty(lint: &LintData<'_>, enable_msrv: bool, ty: &str) -> io::Result<()> {
    match ty {
        "cargo" => assert_eq!(
            lint.category, "cargo",
            "Lints of type `cargo` must have the `cargo` category"
        ),
        _ if lint.category == "cargo" => panic!("Lints of category `cargo` must have the `cargo` type"),
        _ => {},
    }

    let ty_dir = PathBuf::from(format!("clippy_lints/src/{ty}"));
    assert!(
        ty_dir.exists() && ty_dir.is_dir(),
        "Directory `{}` does not exist!",
        ty_dir.display()
    );

    let lint_file_path = ty_dir.join(format!("{}.rs", lint.name));
    assert!(
        !lint_file_path.exists(),
        "File `{}` already exists",
        lint_file_path.display()
    );

    let mod_file_path = ty_dir.join("mod.rs");
    let context_import = setup_mod_file(&mod_file_path, lint)?;
    let (pass_lifetimes, msrv_ty, msrv_ref, msrv_cx) = match context_import {
        "LateContext" => ("<'_>", "Msrv", "", "cx, "),
        _ => ("", "MsrvStack", "&", ""),
    };

    let name_upper = lint.name.to_uppercase();
    let mut lint_file_contents = String::new();

    if enable_msrv {
        let _: fmt::Result = writedoc!(
            lint_file_contents,
            r#"
                use clippy_utils::msrvs::{{self, {msrv_ty}}};
                use rustc_lint::{{{context_import}, LintContext}};

                use super::{name_upper};

                // TODO: Adjust the parameters as necessary
                pub(super) fn check(cx: &{context_import}{pass_lifetimes}, msrv: {msrv_ref}{msrv_ty}) {{
                    if !msrv.meets({msrv_cx}todo!("Add a new entry in `clippy_utils/src/msrvs`")) {{
                        return;
                    }}
                    todo!();
                }}
           "#
        );
    } else {
        let _: fmt::Result = writedoc!(
            lint_file_contents,
            r"
                use rustc_lint::{{{context_import}, LintContext}};

                use super::{name_upper};

                // TODO: Adjust the parameters as necessary
                pub(super) fn check(cx: &{context_import}{pass_lifetimes}) {{
                    todo!();
                }}
           "
        );
    }

    write_file(lint_file_path.as_path(), lint_file_contents)?;
    println!("Generated lint file: `clippy_lints/src/{ty}/{}.rs`", lint.name);
    println!(
        "Be sure to add a call to `{}::check` in `clippy_lints/src/{ty}/mod.rs`!",
        lint.name
    );

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn setup_mod_file(path: &Path, lint: &LintData<'_>) -> io::Result<&'static str> {
    let lint_name_upper = lint.name.to_uppercase();

    let mut file_contents = fs::read_to_string(path)?;
    assert!(
        !file_contents.contains(&format!("pub {lint_name_upper},")),
        "Lint `{}` already defined in `{}`",
        lint.name,
        path.display()
    );

    let (lint_context, lint_decl_end) = parse_mod_file(path, &file_contents);

    // Add the lint declaration to `mod.rs`
    file_contents.insert_str(
        lint_decl_end,
        &format!(
            "\n\n{}",
            get_lint_declaration(lint.clippy_version, &lint_name_upper, lint.category)
        ),
    );

    // Add the lint to `impl_lint_pass`/`declare_lint_pass`
    let impl_lint_pass_start = file_contents.find("impl_lint_pass!").unwrap_or_else(|| {
        file_contents
            .find("declare_lint_pass!")
            .unwrap_or_else(|| panic!("failed to find `impl_lint_pass`/`declare_lint_pass`"))
    });

    let mut arr_start = file_contents[impl_lint_pass_start..].find('[').unwrap_or_else(|| {
        panic!("malformed `impl_lint_pass`/`declare_lint_pass`");
    });

    arr_start += impl_lint_pass_start;

    let mut arr_end = file_contents[arr_start..]
        .find(']')
        .expect("failed to find `impl_lint_pass` terminator");

    arr_end += arr_start;

    let mut arr_content = file_contents[arr_start + 1..arr_end].to_string();
    arr_content.retain(|c| !c.is_whitespace());

    let mut new_arr_content = String::new();
    for ident in arr_content
        .split(',')
        .chain(std::iter::once(&*lint_name_upper))
        .filter(|s| !s.is_empty())
    {
        let _: fmt::Result = write!(new_arr_content, "\n    {ident},");
    }
    new_arr_content.push('\n');

    file_contents.replace_range(arr_start + 1..arr_end, &new_arr_content);

    // Just add the mod declaration at the top, it'll be fixed by rustfmt
    file_contents.insert_str(0, &format!("mod {};\n", &lint.name));

    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .context(format!("trying to open: `{}`", path.display()))?;
    file.write_all(file_contents.as_bytes())
        .context(format!("writing to file: `{}`", path.display()))?;

    Ok(lint_context)
}

// Find both the last lint declaration (declare_clippy_lint!) and the lint pass impl
fn parse_mod_file(path: &Path, contents: &str) -> (&'static str, usize) {
    #[allow(clippy::enum_glob_use)]
    use Token::*;

    let mut context = None;
    let mut decl_end = None;
    let mut searcher = RustSearcher::new(contents);
    while let Some(name) = searcher.find_capture_token(CaptureIdent) {
        match name {
            "declare_clippy_lint" => {
                if searcher.match_tokens(&[Bang, OpenBrace], &mut []) && searcher.find_token(CloseBrace) {
                    decl_end = Some(searcher.pos());
                }
            },
            "impl" => {
                let mut capture = "";
                if searcher.match_tokens(&[Lt, Lifetime, Gt, CaptureIdent], &mut [&mut capture]) {
                    match capture {
                        "LateLintPass" => context = Some("LateContext"),
                        "EarlyLintPass" => context = Some("EarlyContext"),
                        _ => {},
                    }
                }
            },
            _ => {},
        }
    }

    (
        context.unwrap_or_else(|| panic!("No lint pass implementation found in `{}`", path.display())),
        decl_end.unwrap_or_else(|| panic!("No lint declarations found in `{}`", path.display())) as usize,
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
