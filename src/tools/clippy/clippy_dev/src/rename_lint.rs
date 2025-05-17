use crate::update_lints::{
    DeprecatedLints, RenamedLint, find_lint_decls, gen_renamed_lints_test_fn, generate_lint_files,
    read_deprecated_lints,
};
use crate::utils::{FileUpdater, StringReplacer, UpdateMode, Version, try_rename_file};
use std::ffi::OsStr;
use std::path::Path;
use walkdir::WalkDir;

/// Runs the `rename_lint` command.
///
/// This does the following:
/// * Adds an entry to `renamed_lints.rs`.
/// * Renames all lint attributes to the new name (e.g. `#[allow(clippy::lint_name)]`).
/// * Renames the lint struct to the new name.
/// * Renames the module containing the lint struct to the new name if it shares a name with the
///   lint.
///
/// # Panics
/// Panics for the following conditions:
/// * If a file path could not read from or then written to
/// * If either lint name has a prefix
/// * If `old_name` doesn't name an existing lint.
/// * If `old_name` names a deprecated or renamed lint.
#[allow(clippy::too_many_lines)]
pub fn rename(clippy_version: Version, old_name: &str, new_name: &str, uplift: bool) {
    if let Some((prefix, _)) = old_name.split_once("::") {
        panic!("`{old_name}` should not contain the `{prefix}` prefix");
    }
    if let Some((prefix, _)) = new_name.split_once("::") {
        panic!("`{new_name}` should not contain the `{prefix}` prefix");
    }

    let mut updater = FileUpdater::default();
    let mut lints = find_lint_decls();
    let DeprecatedLints {
        renamed: mut renamed_lints,
        deprecated: deprecated_lints,
        file: mut deprecated_file,
        contents: mut deprecated_contents,
        renamed_end,
        ..
    } = read_deprecated_lints();

    let mut old_lint_index = None;
    let mut found_new_name = false;
    for (i, lint) in lints.iter().enumerate() {
        if lint.name == old_name {
            old_lint_index = Some(i);
        } else if lint.name == new_name {
            found_new_name = true;
        }
    }
    let old_lint_index = old_lint_index.unwrap_or_else(|| panic!("could not find lint `{old_name}`"));

    let lint = RenamedLint {
        old_name: format!("clippy::{old_name}"),
        new_name: if uplift {
            new_name.into()
        } else {
            format!("clippy::{new_name}")
        },
    };

    // Renamed lints and deprecated lints shouldn't have been found in the lint list, but check just in
    // case.
    assert!(
        !renamed_lints.iter().any(|l| lint.old_name == l.old_name),
        "`{old_name}` has already been renamed"
    );
    assert!(
        !deprecated_lints.iter().any(|l| lint.old_name == l.name),
        "`{old_name}` has already been deprecated"
    );

    // Update all lint level attributes. (`clippy::lint_name`)
    let replacements = &[(&*lint.old_name, &*lint.new_name)];
    let replacer = StringReplacer::new(replacements);
    for file in WalkDir::new(".").into_iter().map(Result::unwrap).filter(|f| {
        let name = f.path().file_name();
        let ext = f.path().extension();
        (ext == Some(OsStr::new("rs")) || ext == Some(OsStr::new("fixed")))
            && name != Some(OsStr::new("rename.rs"))
            && name != Some(OsStr::new("deprecated_lints.rs"))
    }) {
        updater.update_file(file.path(), &mut replacer.replace_ident_fn());
    }

    deprecated_contents.insert_str(
        renamed_end as usize,
        &format!(
            "    #[clippy::version = \"{}\"]\n    (\"{}\", \"{}\"),\n",
            clippy_version.rust_display(),
            lint.old_name,
            lint.new_name,
        ),
    );
    deprecated_file.replace_contents(deprecated_contents.as_bytes());
    drop(deprecated_file);

    renamed_lints.push(lint);
    renamed_lints.sort_by(|lhs, rhs| {
        lhs.new_name
            .starts_with("clippy::")
            .cmp(&rhs.new_name.starts_with("clippy::"))
            .reverse()
            .then_with(|| lhs.old_name.cmp(&rhs.old_name))
    });

    if uplift {
        updater.update_file("tests/ui/rename.rs", &mut gen_renamed_lints_test_fn(&renamed_lints));
        println!(
            "`{old_name}` has be uplifted. All the code inside `clippy_lints` related to it needs to be removed manually."
        );
    } else if found_new_name {
        updater.update_file("tests/ui/rename.rs", &mut gen_renamed_lints_test_fn(&renamed_lints));
        println!(
            "`{new_name}` is already defined. The old linting code inside `clippy_lints` needs to be updated/removed manually."
        );
    } else {
        // Rename the lint struct and source files sharing a name with the lint.
        let lint = &mut lints[old_lint_index];
        let old_name_upper = old_name.to_uppercase();
        let new_name_upper = new_name.to_uppercase();
        lint.name = new_name.into();

        // Rename test files. only rename `.stderr` and `.fixed` files if the new test name doesn't exist.
        if try_rename_file(
            Path::new(&format!("tests/ui/{old_name}.rs")),
            Path::new(&format!("tests/ui/{new_name}.rs")),
        ) {
            try_rename_file(
                Path::new(&format!("tests/ui/{old_name}.stderr")),
                Path::new(&format!("tests/ui/{new_name}.stderr")),
            );
            try_rename_file(
                Path::new(&format!("tests/ui/{old_name}.fixed")),
                Path::new(&format!("tests/ui/{new_name}.fixed")),
            );
        }

        // Try to rename the file containing the lint if the file name matches the lint's name.
        let replacements;
        let replacements = if lint.module == old_name
            && try_rename_file(
                Path::new(&format!("clippy_lints/src/{old_name}.rs")),
                Path::new(&format!("clippy_lints/src/{new_name}.rs")),
            ) {
            // Edit the module name in the lint list. Note there could be multiple lints.
            for lint in lints.iter_mut().filter(|l| l.module == old_name) {
                lint.module = new_name.into();
            }
            replacements = [(&*old_name_upper, &*new_name_upper), (old_name, new_name)];
            replacements.as_slice()
        } else if !lint.module.contains("::")
            // Catch cases like `methods/lint_name.rs` where the lint is stored in `methods/mod.rs`
            && try_rename_file(
                Path::new(&format!("clippy_lints/src/{}/{old_name}.rs", lint.module)),
                Path::new(&format!("clippy_lints/src/{}/{new_name}.rs", lint.module)),
            )
        {
            // Edit the module name in the lint list. Note there could be multiple lints, or none.
            let renamed_mod = format!("{}::{old_name}", lint.module);
            for lint in lints.iter_mut().filter(|l| l.module == renamed_mod) {
                lint.module = format!("{}::{new_name}", lint.module);
            }
            replacements = [(&*old_name_upper, &*new_name_upper), (old_name, new_name)];
            replacements.as_slice()
        } else {
            replacements = [(&*old_name_upper, &*new_name_upper), ("", "")];
            &replacements[0..1]
        };

        // Don't change `clippy_utils/src/renamed_lints.rs` here as it would try to edit the lint being
        // renamed.
        let replacer = StringReplacer::new(replacements);
        for file in WalkDir::new("clippy_lints/src") {
            let file = file.expect("error reading `clippy_lints/src`");
            if file
                .path()
                .as_os_str()
                .to_str()
                .is_some_and(|x| x.ends_with("*.rs") && x["clippy_lints/src/".len()..] != *"deprecated_lints.rs")
            {
                updater.update_file(file.path(), &mut replacer.replace_ident_fn());
            }
        }

        generate_lint_files(UpdateMode::Change, &lints, &deprecated_lints, &renamed_lints);
        println!("{old_name} has been successfully renamed");
    }

    println!("note: `cargo uitest` still needs to be run to update the test results");
}
