use crate::update_lints::{
    DeprecatedLint, DeprecatedLints, Lint, find_lint_decls, generate_lint_files, read_deprecated_lints,
};
use crate::utils::{UpdateMode, Version};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::{fs, io};

/// Runs the `deprecate` command
///
/// This does the following:
/// * Adds an entry to `deprecated_lints.rs`.
/// * Removes the lint declaration (and the entire file if applicable)
///
/// # Panics
///
/// If a file path could not read from or written to
pub fn deprecate(clippy_version: Version, name: &str, reason: &str) {
    let prefixed_name = if name.starts_with("clippy::") {
        name.to_owned()
    } else {
        format!("clippy::{name}")
    };
    let stripped_name = &prefixed_name[8..];

    let mut lints = find_lint_decls();
    let DeprecatedLints {
        renamed: renamed_lints,
        deprecated: mut deprecated_lints,
        file: mut deprecated_file,
        contents: mut deprecated_contents,
        deprecated_end,
        ..
    } = read_deprecated_lints();

    let Some(lint) = lints.iter().find(|l| l.name == stripped_name) else {
        eprintln!("error: failed to find lint `{name}`");
        return;
    };

    let mod_path = {
        let mut mod_path = PathBuf::from(format!("clippy_lints/src/{}", lint.module));
        if mod_path.is_dir() {
            mod_path = mod_path.join("mod");
        }

        mod_path.set_extension("rs");
        mod_path
    };

    if remove_lint_declaration(stripped_name, &mod_path, &mut lints).unwrap_or(false) {
        deprecated_contents.insert_str(
            deprecated_end as usize,
            &format!(
                "    #[clippy::version = \"{}\"]\n    (\"{}\", \"{}\"),\n",
                clippy_version.rust_display(),
                prefixed_name,
                reason,
            ),
        );
        deprecated_file.replace_contents(deprecated_contents.as_bytes());
        drop(deprecated_file);

        deprecated_lints.push(DeprecatedLint {
            name: prefixed_name,
            reason: reason.into(),
        });

        generate_lint_files(UpdateMode::Change, &lints, &deprecated_lints, &renamed_lints);
        println!("info: `{name}` has successfully been deprecated");
        println!("note: you must run `cargo uitest` to update the test results");
    } else {
        eprintln!("error: lint not found");
    }
}

fn remove_lint_declaration(name: &str, path: &Path, lints: &mut Vec<Lint>) -> io::Result<bool> {
    fn remove_lint(name: &str, lints: &mut Vec<Lint>) {
        lints.iter().position(|l| l.name == name).map(|pos| lints.remove(pos));
    }

    fn remove_test_assets(name: &str) {
        let test_file_stem = format!("tests/ui/{name}");
        let path = Path::new(&test_file_stem);

        // Some lints have their own directories, delete them
        if path.is_dir() {
            let _ = fs::remove_dir_all(path);
            return;
        }

        // Remove all related test files
        let _ = fs::remove_file(path.with_extension("rs"));
        let _ = fs::remove_file(path.with_extension("stderr"));
        let _ = fs::remove_file(path.with_extension("fixed"));
    }

    fn remove_impl_lint_pass(lint_name_upper: &str, content: &mut String) {
        let impl_lint_pass_start = content.find("impl_lint_pass!").unwrap_or_else(|| {
            content
                .find("declare_lint_pass!")
                .unwrap_or_else(|| panic!("failed to find `impl_lint_pass`"))
        });
        let mut impl_lint_pass_end = content[impl_lint_pass_start..]
            .find(']')
            .expect("failed to find `impl_lint_pass` terminator");

        impl_lint_pass_end += impl_lint_pass_start;
        if let Some(lint_name_pos) = content[impl_lint_pass_start..impl_lint_pass_end].find(lint_name_upper) {
            let mut lint_name_end = impl_lint_pass_start + (lint_name_pos + lint_name_upper.len());
            for c in content[lint_name_end..impl_lint_pass_end].chars() {
                // Remove trailing whitespace
                if c == ',' || c.is_whitespace() {
                    lint_name_end += 1;
                } else {
                    break;
                }
            }

            content.replace_range(impl_lint_pass_start + lint_name_pos..lint_name_end, "");
        }
    }

    if path.exists()
        && let Some(lint) = lints.iter().find(|l| l.name == name)
    {
        if lint.module == name {
            // The lint name is the same as the file, we can just delete the entire file
            fs::remove_file(path)?;
        } else {
            // We can't delete the entire file, just remove the declaration

            if let Some(Some("mod.rs")) = path.file_name().map(OsStr::to_str) {
                // Remove clippy_lints/src/some_mod/some_lint.rs
                let mut lint_mod_path = path.to_path_buf();
                lint_mod_path.set_file_name(name);
                lint_mod_path.set_extension("rs");

                let _ = fs::remove_file(lint_mod_path);
            }

            let mut content =
                fs::read_to_string(path).unwrap_or_else(|_| panic!("failed to read `{}`", path.to_string_lossy()));

            eprintln!(
                "warn: you will have to manually remove any code related to `{name}` from `{}`",
                path.display()
            );

            assert!(
                content[lint.declaration_range.clone()].contains(&name.to_uppercase()),
                "error: `{}` does not contain lint `{}`'s declaration",
                path.display(),
                lint.name
            );

            // Remove lint declaration (declare_clippy_lint!)
            content.replace_range(lint.declaration_range.clone(), "");

            // Remove the module declaration (mod xyz;)
            let mod_decl = format!("\nmod {name};");
            content = content.replacen(&mod_decl, "", 1);

            remove_impl_lint_pass(&lint.name.to_uppercase(), &mut content);
            fs::write(path, content).unwrap_or_else(|_| panic!("failed to write to `{}`", path.to_string_lossy()));
        }

        remove_test_assets(name);
        remove_lint(name, lints);
        return Ok(true);
    }

    Ok(false)
}
