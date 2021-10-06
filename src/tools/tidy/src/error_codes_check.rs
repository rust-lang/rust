//! Checks that all error codes have at least one test to prevent having error
//! codes that are silently not thrown by the compiler anymore.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::read_to_string;
use std::path::Path;

use regex::Regex;

// A few of those error codes can't be tested but all the others can and *should* be tested!
const EXEMPTED_FROM_TEST: &[&str] = &[
    "E0227", "E0279", "E0280", "E0313", "E0377", "E0461", "E0462", "E0465", "E0476", "E0514",
    "E0519", "E0523", "E0554", "E0640", "E0717", "E0729",
];

// Some error codes don't have any tests apparently...
const IGNORE_EXPLANATION_CHECK: &[&str] = &["E0464", "E0570", "E0601", "E0602", "E0729"];

// If the file path contains any of these, we don't want to try to extract error codes from it.
//
// We need to declare each path in the windows version (with backslash).
const PATHS_TO_IGNORE_FOR_EXTRACTION: &[&str] =
    &["src/test/", "src\\test\\", "src/doc/", "src\\doc\\", "src/tools/", "src\\tools\\"];

#[derive(Default, Debug)]
struct ErrorCodeStatus {
    has_test: bool,
    has_explanation: bool,
    is_used: bool,
}

fn check_error_code_explanation(
    f: &str,
    error_codes: &mut HashMap<String, ErrorCodeStatus>,
    err_code: String,
) -> bool {
    let mut invalid_compile_fail_format = false;
    let mut found_error_code = false;

    for line in f.lines() {
        let s = line.trim();
        if s.starts_with("```") {
            if s.contains("compile_fail") && s.contains('E') {
                if !found_error_code {
                    error_codes.get_mut(&err_code).map(|x| x.has_test = true);
                    found_error_code = true;
                }
            } else if s.contains("compile-fail") {
                invalid_compile_fail_format = true;
            }
        } else if s.starts_with("#### Note: this error code is no longer emitted by the compiler") {
            if !found_error_code {
                error_codes.get_mut(&err_code).map(|x| x.has_test = true);
                found_error_code = true;
            }
        }
    }
    invalid_compile_fail_format
}

fn check_if_error_code_is_test_in_explanation(f: &str, err_code: &str) -> bool {
    let mut ignore_found = false;

    for line in f.lines() {
        let s = line.trim();
        if s.starts_with("#### Note: this error code is no longer emitted by the compiler") {
            return true;
        }
        if s.starts_with("```") {
            if s.contains("compile_fail") && s.contains(err_code) {
                return true;
            } else if s.contains("ignore") {
                // It's very likely that we can't actually make it fail compilation...
                ignore_found = true;
            }
        }
    }
    ignore_found
}

macro_rules! some_or_continue {
    ($e:expr) => {
        match $e {
            Some(e) => e,
            None => continue,
        }
    };
}

fn extract_error_codes(
    f: &str,
    error_codes: &mut HashMap<String, ErrorCodeStatus>,
    path: &Path,
    errors: &mut Vec<String>,
) {
    let mut reached_no_explanation = false;

    for line in f.lines() {
        let s = line.trim();
        if !reached_no_explanation && s.starts_with('E') && s.contains("include_str!(\"") {
            let err_code = s
                .split_once(':')
                .expect(
                    format!(
                        "Expected a line with the format `E0xxx: include_str!(\"..\")`, but got {} \
                         without a `:` delimiter",
                        s,
                    )
                    .as_str(),
                )
                .0
                .to_owned();
            error_codes.entry(err_code.clone()).or_default().has_explanation = true;

            // Now we extract the tests from the markdown file!
            let md_file_name = match s.split_once("include_str!(\"") {
                None => continue,
                Some((_, md)) => match md.split_once("\")") {
                    None => continue,
                    Some((file_name, _)) => file_name,
                },
            };
            let path = some_or_continue!(path.parent())
                .join(md_file_name)
                .canonicalize()
                .expect("failed to canonicalize error explanation file path");
            match read_to_string(&path) {
                Ok(content) => {
                    let has_test = check_if_error_code_is_test_in_explanation(&content, &err_code);
                    if !has_test && !IGNORE_EXPLANATION_CHECK.contains(&err_code.as_str()) {
                        errors.push(format!(
                            "`{}` doesn't use its own error code in compile_fail example",
                            path.display(),
                        ));
                    } else if has_test && IGNORE_EXPLANATION_CHECK.contains(&err_code.as_str()) {
                        errors.push(format!(
                            "`{}` has a compile_fail example with its own error code, it shouldn't \
                             be listed in IGNORE_EXPLANATION_CHECK!",
                            path.display(),
                        ));
                    }
                    if check_error_code_explanation(&content, error_codes, err_code) {
                        errors.push(format!(
                            "`{}` uses invalid tag `compile-fail` instead of `compile_fail`",
                            path.display(),
                        ));
                    }
                }
                Err(e) => {
                    eprintln!("Couldn't read `{}`: {}", path.display(), e);
                }
            }
        } else if reached_no_explanation && s.starts_with('E') {
            let err_code = match s.split_once(',') {
                None => s,
                Some((err_code, _)) => err_code,
            }
            .to_string();
            if !error_codes.contains_key(&err_code) {
                // this check should *never* fail!
                error_codes.insert(err_code, ErrorCodeStatus::default());
            }
        } else if s == ";" {
            reached_no_explanation = true;
        }
    }
}

fn extract_error_codes_from_tests(f: &str, error_codes: &mut HashMap<String, ErrorCodeStatus>) {
    for line in f.lines() {
        let s = line.trim();
        if s.starts_with("error[E") || s.starts_with("warning[E") {
            let err_code = match s.split_once(']') {
                None => continue,
                Some((err_code, _)) => match err_code.split_once('[') {
                    None => continue,
                    Some((_, err_code)) => err_code,
                },
            };
            error_codes.entry(err_code.to_owned()).or_default().has_test = true;
        }
    }
}

fn extract_error_codes_from_source(
    f: &str,
    error_codes: &mut HashMap<String, ErrorCodeStatus>,
    regex: &Regex,
) {
    for line in f.lines() {
        if line.trim_start().starts_with("//") {
            continue;
        }
        for cap in regex.captures_iter(line) {
            if let Some(error_code) = cap.get(1) {
                error_codes.entry(error_code.as_str().to_owned()).or_default().is_used = true;
            }
        }
    }
}

pub fn check(paths: &[&Path], bad: &mut bool) {
    let mut errors = Vec::new();
    let mut found_explanations = 0;
    let mut found_tests = 0;
    let mut error_codes: HashMap<String, ErrorCodeStatus> = HashMap::new();
    // We want error codes which match the following cases:
    //
    // * foo(a, E0111, a)
    // * foo(a, E0111)
    // * foo(E0111, a)
    // * #[error = "E0111"]
    let regex = Regex::new(r#"[(,"\s](E\d{4})[,)"]"#).unwrap();

    println!("Checking which error codes lack tests...");

    for path in paths {
        super::walk(path, &mut |path| super::filter_dirs(path), &mut |entry, contents| {
            let file_name = entry.file_name();
            if file_name == "error_codes.rs" {
                extract_error_codes(contents, &mut error_codes, entry.path(), &mut errors);
                found_explanations += 1;
            } else if entry.path().extension() == Some(OsStr::new("stderr")) {
                extract_error_codes_from_tests(contents, &mut error_codes);
                found_tests += 1;
            } else if entry.path().extension() == Some(OsStr::new("rs")) {
                let path = entry.path().to_string_lossy();
                if PATHS_TO_IGNORE_FOR_EXTRACTION.iter().all(|c| !path.contains(c)) {
                    extract_error_codes_from_source(contents, &mut error_codes, &regex);
                }
            }
        });
    }
    if found_explanations == 0 {
        eprintln!("No error code explanation was tested!");
        *bad = true;
    }
    if found_tests == 0 {
        eprintln!("No error code was found in compilation errors!");
        *bad = true;
    }
    if errors.is_empty() {
        println!("Found {} error codes", error_codes.len());

        for (err_code, error_status) in &error_codes {
            if !error_status.has_test && !EXEMPTED_FROM_TEST.contains(&err_code.as_str()) {
                errors.push(format!("Error code {} needs to have at least one UI test!", err_code));
            } else if error_status.has_test && EXEMPTED_FROM_TEST.contains(&err_code.as_str()) {
                errors.push(format!(
                    "Error code {} has a UI test, it shouldn't be listed into EXEMPTED_FROM_TEST!",
                    err_code
                ));
            }
            if !error_status.is_used && !error_status.has_explanation {
                errors.push(format!(
                    "Error code {} isn't used and doesn't have an error explanation, it should be \
                     commented in error_codes.rs file",
                    err_code
                ));
            }
        }
    }
    if errors.is_empty() {
        // Checking if local constants need to be cleaned.
        for err_code in EXEMPTED_FROM_TEST {
            match error_codes.get(err_code.to_owned()) {
                Some(status) => {
                    if status.has_test {
                        errors.push(format!(
                            "{} error code has a test and therefore should be \
                            removed from the `EXEMPTED_FROM_TEST` constant",
                            err_code
                        ));
                    }
                }
                None => errors.push(format!(
                    "{} error code isn't used anymore and therefore should be removed \
                        from `EXEMPTED_FROM_TEST` constant",
                    err_code
                )),
            }
        }
    }
    errors.sort();
    for err in &errors {
        eprintln!("{}", err);
    }
    println!("Found {} error codes with no tests", errors.len());
    if !errors.is_empty() {
        *bad = true;
    }
    println!("Done!");
}
