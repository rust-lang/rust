use regex::Regex;
use std::fs;
use walkdir::WalkDir;

#[test]
fn old_test_headers() {
    let old_headers = Regex::new(
        r"^//( ?\[\w+\])? ?((check|build|run|ignore|aux|only|needs|rustc|unset|no|normalize|run|compile)-|edition|incremental|revisions).*",
    )
    .unwrap();
    let mut failed = false;

    for entry in WalkDir::new("tests") {
        let entry = entry.unwrap();
        let is_hidden_file = entry
            .file_name()
            .to_str()
            .expect("non-UTF-8 file name")
            .starts_with('.');
        if is_hidden_file || !entry.file_type().is_file() {
            continue;
        }

        let file = fs::read_to_string(entry.path()).unwrap_or_else(|err| panic!("{}: {err}", entry.path().display()));

        if let Some(header) = old_headers.find(&file) {
            println!("Found header `{}` in {}", header.as_str(), entry.path().display());

            failed = true;
        }
    }

    assert!(!failed, "use `//@foo` style test headers instead");
}
