use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{fs, process};

const EXTENSIONS: &[&str] =
    &["rs", "py", "js", "sh", "c", "cpp", "h", "md", "css", "ftl", "toml", "yml", "yaml"];

fn has_supported_extension(path: &Path) -> bool {
    path.extension().is_some_and(|ext| EXTENSIONS.iter().any(|e| ext == OsStr::new(e)))
}

fn list_tracked_files() -> Result<Vec<PathBuf>, String> {
    let output = Command::new("git")
        .args(["ls-files", "-z"])
        .output()
        .map_err(|e| format!("Failed to run `git ls-files`: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("`git ls-files` failed: {stderr}"));
    }

    let mut files = Vec::new();
    for entry in output.stdout.split(|b| *b == 0) {
        if entry.is_empty() {
            continue;
        }
        let path = std::str::from_utf8(entry).unwrap();
        files.push(PathBuf::from(path));
    }

    Ok(files)
}

pub(crate) fn run() -> ! {
    let files = list_tracked_files().unwrap();
    let mut error_count = 0;
    // Avoid embedding the task marker in source so greps only find real occurrences.
    let todo_marker = "todo".to_ascii_uppercase();

    for file in files {
        if !has_supported_extension(&file) {
            continue;
        }

        let bytes = fs::read(&file).unwrap();
        let contents = std::str::from_utf8(&bytes).unwrap();

        for (i, line) in contents.split('\n').enumerate() {
            let trimmed = line.trim();
            if trimmed.contains(&todo_marker) {
                eprintln!(
                    "{}:{}: {} is used for tasks that should be done before merging a PR; if you want to leave a message in the codebase use FIXME",
                    file.display(),
                    i + 1,
                    todo_marker
                );
                error_count += 1;
            }
        }
    }

    if error_count == 0 {
        process::exit(0);
    }

    eprintln!("found {} {}(s)", error_count, todo_marker);
    process::exit(1);
}
