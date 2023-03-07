use crate::clippy_project_root;
use indoc::{formatdoc, writedoc};
use std::fmt;
use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::prelude::*;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

struct LintData<'a> {
    pass: &'a str,
    name: &'a str,
    category: &'a str,
    ty: Option<&'a str>,
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
                let message = format!("{}: {e}", text.as_ref());
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
pub fn create(
    pass: Option<&String>,
    lint_name: Option<&String>,
    category: Option<&str>,
    mut ty: Option<&str>,
    msrv: bool,
) -> io::Result<()> {
    if category == Some("cargo") && ty.is_none() {
        // `cargo` is a special category, these lints should always be in `clippy_lints/src/cargo`
        ty = Some("cargo");
    }

    let lint = LintData {
        pass: pass.map_or("", String::as_str),
        name: lint_name.expect("`name` argument is validated by clap"),
        category: category.expect("`category` argument is validated by clap"),
        ty,
        project_root: clippy_project_root(),
    };

    create_lint(&lint, msrv).context("Unable to create lint implementation")?;
    create_test(&lint).context("Unable to create a test for the new lint")?;

    if lint.ty.is_none() {
        add_lint(&lint, msrv).context("Unable to add lint to clippy_lints/src/lib.rs")?;
    }

    Ok(())
}

fn create_lint(lint: &LintData<'_>, enable_msrv: bool) -> io::Result<()> {
    if let Some(ty) = lint.ty {
        create_lint_for_ty(lint, enable_msrv, ty)
    } else {
        let lint_contents = get_lint_file_contents(lint, enable_msrv);
        let lint_path = format!("clippy_lints/src/{}.rs", lint.name);
        write_file(lint.project_root.join(&lint_path), lint_contents.as_bytes())?;
        println!("Generated lint file: `{lint_path}`");

        Ok(())
    }
}

fn create_test(lint: &LintData<'_>) -> io::Result<()> {
    fn create_project_layout<P: Into<PathBuf>>(lint_name: &str, location: P, case: &str, hint: &str) -> io::Result<()> {
        let mut path = location.into().join(case);
        fs::create_dir(&path)?;
        write_file(path.join("Cargo.toml"), get_manifest_contents(lint_name, hint))?;

        path.push("src");
        fs::create_dir(&path)?;
        let header = format!("// compile-flags: --crate-name={lint_name}");
        write_file(path.join("main.rs"), get_test_file_contents(lint_name, Some(&header)))?;

        Ok(())
    }

    if lint.category == "cargo" {
        let relative_test_dir = format!("tests/ui-cargo/{}", lint.name);
        let test_dir = lint.project_root.join(&relative_test_dir);
        fs::create_dir(&test_dir)?;

        create_project_layout(lint.name, &test_dir, "fail", "Content that triggers the lint goes here")?;
        create_project_layout(lint.name, &test_dir, "pass", "This file should not trigger the lint")?;

        println!("Generated test directories: `{relative_test_dir}/pass`, `{relative_test_dir}/fail`");
    } else {
        let test_path = format!("tests/ui/{}.rs", lint.name);
        let test_contents = get_test_file_contents(lint.name, None);
        write_file(lint.project_root.join(&test_path), test_contents)?;

        println!("Generated test file: `{test_path}`");
    }

    Ok(())
}

fn add_lint(lint: &LintData<'_>, enable_msrv: bool) -> io::Result<()> {
    let path = "clippy_lints/src/lib.rs";
    let mut lib_rs = fs::read_to_string(path).context("reading")?;

    let comment_start = lib_rs.find("// add lints here,").expect("Couldn't find comment");

    let new_lint = if enable_msrv {
        format!(
            "store.register_{lint_pass}_pass(move |{ctor_arg}| Box::new({module_name}::{camel_name}::new(msrv())));\n    ",
            lint_pass = lint.pass,
            ctor_arg = if lint.pass == "late" { "_" } else { "" },
            module_name = lint.name,
            camel_name = to_camel_case(lint.name),
        )
    } else {
        format!(
            "store.register_{lint_pass}_pass(|{ctor_arg}| Box::new({module_name}::{camel_name}));\n    ",
            lint_pass = lint.pass,
            ctor_arg = if lint.pass == "late" { "_" } else { "" },
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
                String::new()
            } else {
                [&s[0..1].to_uppercase(), &s[1..]].concat()
            }
        })
        .collect()
}

pub(crate) fn get_stabilization_version() -> String {
    fn parse_manifest(contents: &str) -> Option<String> {
        let version = contents
            .lines()
            .filter_map(|l| l.split_once('='))
            .find_map(|(k, v)| (k.trim() == "version").then(|| v.trim()))?;
        let Some(("0", version)) = version.get(1..version.len() - 1)?.split_once('.') else {
            return None;
        };
        let (minor, patch) = version.split_once('.')?;
        Some(format!(
            "{}.{}.0",
            minor.parse::<u32>().ok()?,
            patch.parse::<u32>().ok()?
        ))
    }
    let contents = fs::read_to_string("Cargo.toml").expect("Unable to read `Cargo.toml`");
    parse_manifest(&contents).expect("Unable to find package version in `Cargo.toml`")
}

fn get_test_file_contents(lint_name: &str, header_commands: Option<&str>) -> String {
    let mut contents = formatdoc!(
        r#"
        #![allow(unused)]
        #![warn(clippy::{lint_name})]

        fn main() {{
            // test code goes here
        }}
    "#
    );

    if let Some(header) = header_commands {
        contents = format!("{header}\n{contents}");
    }

    contents
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
        "early" => ("EarlyLintPass", "", "use rustc_ast::ast::*;", "EarlyContext"),
        "late" => ("LateLintPass", "<'_>", "use rustc_hir::*;", "LateContext"),
        _ => {
            unreachable!("`pass_type` should only ever be `early` or `late`!");
        },
    };

    let lint_name = lint.name;
    let category = lint.category;
    let name_camel = to_camel_case(lint.name);
    let name_upper = lint_name.to_uppercase();

    result.push_str(&if enable_msrv {
        formatdoc!(
            r#"
            use clippy_utils::msrvs::{{self, Msrv}};
            {pass_import}
            use rustc_lint::{{{context_import}, {pass_type}, LintContext}};
            use rustc_session::{{declare_tool_lint, impl_lint_pass}};

        "#
        )
    } else {
        formatdoc!(
            r#"
            {pass_import}
            use rustc_lint::{{{context_import}, {pass_type}}};
            use rustc_session::{{declare_lint_pass, declare_tool_lint}};

        "#
        )
    });

    let _: fmt::Result = write!(result, "{}", get_lint_declaration(&name_upper, category));

    result.push_str(&if enable_msrv {
        formatdoc!(
            r#"
            pub struct {name_camel} {{
                msrv: Msrv,
            }}

            impl {name_camel} {{
                #[must_use]
                pub fn new(msrv: Msrv) -> Self {{
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
        "#
        )
    } else {
        formatdoc!(
            r#"
            declare_lint_pass!({name_camel} => [{name_upper}]);

            impl {pass_type}{pass_lifetimes} for {name_camel} {{}}
        "#
        )
    });

    result
}

fn get_lint_declaration(name_upper: &str, category: &str) -> String {
    formatdoc!(
        r#"
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
                #[clippy::version = "{}"]
                pub {name_upper},
                {category},
                "default lint description"
            }}
        "#,
        get_stabilization_version(),
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

    let ty_dir = lint.project_root.join(format!("clippy_lints/src/{ty}"));
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

    let name_upper = lint.name.to_uppercase();
    let mut lint_file_contents = String::new();

    if enable_msrv {
        let _: fmt::Result = writedoc!(
            lint_file_contents,
            r#"
                use clippy_utils::msrvs::{{self, Msrv}};
                use rustc_lint::{{{context_import}, LintContext}};

                use super::{name_upper};

                // TODO: Adjust the parameters as necessary
                pub(super) fn check(cx: &{context_import}, msrv: &Msrv) {{
                    if !msrv.meets(todo!("Add a new entry in `clippy_utils/src/msrvs`")) {{
                        return;
                    }}
                    todo!();
                }}
           "#,
            context_import = context_import,
            name_upper = name_upper,
        );
    } else {
        let _: fmt::Result = writedoc!(
            lint_file_contents,
            r#"
                use rustc_lint::{{{context_import}, LintContext}};

                use super::{name_upper};

                // TODO: Adjust the parameters as necessary
                pub(super) fn check(cx: &{context_import}) {{
                    todo!();
                }}
           "#,
            context_import = context_import,
            name_upper = name_upper,
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
    use super::update_lints::{match_tokens, LintDeclSearchResult};
    use rustc_lexer::TokenKind;

    let lint_name_upper = lint.name.to_uppercase();

    let mut file_contents = fs::read_to_string(path)?;
    assert!(
        !file_contents.contains(&lint_name_upper),
        "Lint `{}` already defined in `{}`",
        lint.name,
        path.display()
    );

    let mut offset = 0usize;
    let mut last_decl_curly_offset = None;
    let mut lint_context = None;

    let mut iter = rustc_lexer::tokenize(&file_contents).map(|t| {
        let range = offset..offset + t.len as usize;
        offset = range.end;

        LintDeclSearchResult {
            token_kind: t.kind,
            content: &file_contents[range.clone()],
            range,
        }
    });

    // Find both the last lint declaration (declare_clippy_lint!) and the lint pass impl
    while let Some(LintDeclSearchResult { content, .. }) = iter.find(|result| result.token_kind == TokenKind::Ident) {
        let mut iter = iter
            .by_ref()
            .filter(|t| !matches!(t.token_kind, TokenKind::Whitespace | TokenKind::LineComment { .. }));

        match content {
            "declare_clippy_lint" => {
                // matches `!{`
                match_tokens!(iter, Bang OpenBrace);
                if let Some(LintDeclSearchResult { range, .. }) =
                    iter.find(|result| result.token_kind == TokenKind::CloseBrace)
                {
                    last_decl_curly_offset = Some(range.end);
                }
            },
            "impl" => {
                let mut token = iter.next();
                match token {
                    // matches <'foo>
                    Some(LintDeclSearchResult {
                        token_kind: TokenKind::Lt,
                        ..
                    }) => {
                        match_tokens!(iter, Lifetime { .. } Gt);
                        token = iter.next();
                    },
                    None => break,
                    _ => {},
                }

                if let Some(LintDeclSearchResult {
                    token_kind: TokenKind::Ident,
                    content,
                    ..
                }) = token
                {
                    // Get the appropriate lint context struct
                    lint_context = match content {
                        "LateLintPass" => Some("LateContext"),
                        "EarlyLintPass" => Some("EarlyContext"),
                        _ => continue,
                    };
                }
            },
            _ => {},
        }
    }

    drop(iter);

    let last_decl_curly_offset =
        last_decl_curly_offset.unwrap_or_else(|| panic!("No lint declarations found in `{}`", path.display()));
    let lint_context =
        lint_context.unwrap_or_else(|| panic!("No lint pass implementation found in `{}`", path.display()));

    // Add the lint declaration to `mod.rs`
    file_contents.replace_range(
        // Remove the trailing newline, which should always be present
        last_decl_curly_offset..=last_decl_curly_offset,
        &format!("\n\n{}", get_lint_declaration(&lint_name_upper, lint.category)),
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
