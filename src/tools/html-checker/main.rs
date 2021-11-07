use std::env;
use std::path::Path;
use std::process::{Command, Output};

fn check_html_file(file: &Path) -> usize {
    let to_mute = &[
        // "disabled" on <link> or "autocomplete" on <select> emit this warning
        "PROPRIETARY_ATTRIBUTE",
        // It complains when multiple in the same page link to the same anchor for some reason...
        "ANCHOR_NOT_UNIQUE",
        // If a <span> contains only HTML elements and no text, it complains about it.
        "TRIM_EMPTY_ELEMENT",
        // FIXME: the three next warnings are about <pre> elements which are not supposed to
        //        contain HTML. The solution here would be to replace them with a <div>
        "MISSING_ENDTAG_BEFORE",
        "INSERTING_TAG",
        "DISCARDING_UNEXPECTED",
        // This error is caused by nesting the Notable Traits tooltip within an <h4> tag.
        // The solution is to avoid doing that, but we need to have the <h4> tags for accessibility
        // reasons, and we need the Notable Traits tooltip to help everyone understand the Iterator
        // combinators
        "TAG_NOT_ALLOWED_IN",
    ];
    let to_mute_s = to_mute.join(",");
    let mut command = Command::new("tidy");
    command
        .arg("-errors")
        .arg("-quiet")
        .arg("--mute-id") // this option is useful in case we want to mute more warnings
        .arg("yes")
        .arg("--mute")
        .arg(&to_mute_s)
        .arg(file);

    let Output { status, stderr, .. } = command.output().expect("failed to run tidy command");
    if status.success() {
        0
    } else {
        let stderr = String::from_utf8(stderr).expect("String::from_utf8 failed...");
        if stderr.is_empty() && status.code() != Some(2) {
            0
        } else {
            eprintln!(
                "=> Errors for `{}` (error code: {}) <=",
                file.display(),
                status.code().unwrap_or(-1)
            );
            eprintln!("{}", stderr);
            stderr.lines().count()
        }
    }
}

const DOCS_TO_CHECK: &[&str] =
    &["alloc", "core", "proc_macro", "implementors", "src", "std", "test"];

// Returns the number of files read and the number of errors.
fn find_all_html_files(dir: &Path) -> (usize, usize) {
    let mut files_read = 0;
    let mut errors = 0;

    for entry in walkdir::WalkDir::new(dir).into_iter().filter_entry(|e| {
        e.depth() != 1
            || e.file_name()
                .to_str()
                .map(|s| DOCS_TO_CHECK.into_iter().any(|d| *d == s))
                .unwrap_or(false)
    }) {
        let entry = entry.expect("failed to read file");
        if !entry.file_type().is_file() {
            continue;
        }
        let entry = entry.path();
        if entry.extension().and_then(|s| s.to_str()) == Some("html") {
            errors += check_html_file(&entry);
            files_read += 1;
        }
    }
    (files_read, errors)
}

/// Default `tidy` command for macOS is too old that it does not have `mute-id` and `mute` options.
/// `tidy` on macOS Monterey was released on 31 October 2006, and the same date can be seen seven
/// years ago at <https://stackoverflow.com/questions/22283382/overwrite-osx-tidy>. Accordingly,
/// the macOS environment using pre-installed `tidy` should immediately suspend HTML checker process
/// and show a hint to install a newer one.
#[cfg(target_os = "macos")]
fn check_tidy_version() -> Result<(), String> {
    let output = Command::new("tidy").arg("-v").output().expect("failed to run tidy command");
    let version = String::from_utf8(output.stdout).expect("failed to read version of tidy command");
    if version.contains("HTML Tidy for Mac OS X released on 31 October 2006") {
        eprintln!("The pre-installed HTML Tidy for macOS is not supported.");
        eprintln!("Consider installing a newer one and re-running.");
        eprintln!("If you're using Homebrew, you can install it by the following command:");
        eprintln!("    brew install tidy-html5");
        eprintln!();
        Err("HTML check failed: 1 error".to_string())
    } else {
        Ok(())
    }
}

fn main() -> Result<(), String> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        return Err(format!("Usage: {} <doc folder>", args[0]));
    }
    #[cfg(target_os = "macos")]
    check_tidy_version()?;

    println!("Running HTML checker...");

    let (files_read, errors) = find_all_html_files(&Path::new(&args[1]));
    println!("Done! Read {} files...", files_read);
    if errors > 0 {
        Err(format!("HTML check failed: {} errors", errors))
    } else {
        println!("No error found!");
        Ok(())
    }
}
