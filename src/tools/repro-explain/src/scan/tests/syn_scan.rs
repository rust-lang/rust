use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use tempfile::tempdir;

use super::scan_file;

#[test]
fn detects_read_dir_and_cargo_println() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("build.rs");
    fs::write(
        &path,
        r#"
            fn main() {
                for e in std::fs::read_dir(".").unwrap() {
                    println!("cargo::rustc-link-lib={:?}", e.unwrap().path());
                }
            }
            "#,
    )
    .expect("write");

    let hits = scan_file(&path, true, false);
    assert!(hits.iter().any(|h| h.rule_id == "RE001"));
    assert!(hits.iter().any(|h| h.rule_id == "RE005"));
}

#[test]
fn detects_time_and_env_calls() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("build.rs");
    fs::write(
        &path,
        r#"
            fn main() {
                let _ = std::time::SystemTime::now();
                let _ = std::env::var("HOME");
            }
            "#,
    )
    .expect("write");

    let hits = scan_file(&path, true, false);
    assert!(hits.iter().any(|h| h.rule_id == "RE003"));
    assert!(hits.iter().any(|h| h.rule_id == "RE004"));
}

#[test]
fn detects_proc_macro_side_effect() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("lib.rs");
    fs::write(
        &path,
        r#"
            #[proc_macro]
            pub fn x(_i: proc_macro::TokenStream) -> proc_macro::TokenStream {
                let _ = std::env::var("X");
                "fn a() {}".parse().unwrap()
            }
            "#,
    )
    .expect("write");

    let hits = scan_file(&path, false, true);
    assert!(hits.iter().any(|h| h.rule_id == "RE007"));
}

#[test]
fn fixture_build_rs_unsorted_read_dir_matches_golden() {
    assert_fixture_matches_golden(
        "build-rs-unsorted-read-dir/build.rs",
        true,
        false,
        "build-rs-unsorted-read-dir.rule-ids.json",
    );
}

#[test]
fn fixture_build_rs_time_matches_golden() {
    assert_fixture_matches_golden(
        "build-rs-time/build.rs",
        true,
        false,
        "build-rs-time.rule-ids.json",
    );
}

#[test]
fn fixture_build_rs_hashmap_order_matches_golden() {
    assert_fixture_matches_golden(
        "build-rs-hashmap-order/build.rs",
        true,
        false,
        "build-rs-hashmap-order.rule-ids.json",
    );
}

#[test]
fn fixture_parallel_build_script_matches_golden() {
    assert_fixture_matches_golden(
        "parallel-build-script/build.rs",
        true,
        false,
        "parallel-build-script.rule-ids.json",
    );
}

#[test]
fn fixture_proc_macro_env_matches_golden() {
    assert_fixture_matches_golden(
        "proc-macro-env/src/lib.rs",
        false,
        true,
        "proc-macro-env.rule-ids.json",
    );
}

#[test]
fn out_dir_write_does_not_trigger_re006() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("build.rs");
    fs::write(
        &path,
        r#"
            fn main() {
                let out = std::env::var("OUT_DIR").unwrap();
                std::fs::write(format!("{out}/stamp.txt"), "x").unwrap();
            }
            "#,
    )
    .expect("write");

    let hits = scan_file(&path, true, false);
    assert!(!hits.iter().any(|h| h.rule_id == "RE006"));
}

fn assert_fixture_matches_golden(
    fixture_rel: &str,
    is_build_script: bool,
    is_proc_macro: bool,
    golden_rel: &str,
) {
    let fixture = fixture_path("tests/fixtures").join(fixture_rel);
    let golden = fixture_path("tests/golden").join(golden_rel);
    let hits = scan_file(&fixture, is_build_script, is_proc_macro);
    let actual = hits.into_iter().map(|h| h.rule_id).collect::<BTreeSet<_>>();
    let expected = load_golden_rule_ids(&golden).into_iter().collect::<BTreeSet<_>>();
    assert_eq!(actual, expected, "fixture={}, golden={}", fixture.display(), golden.display());
}

fn load_golden_rule_ids(path: &PathBuf) -> Vec<String> {
    let body = fs::read_to_string(path).expect("read golden");
    serde_json::from_str(&body).expect("parse golden json")
}

fn fixture_path(root_rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(root_rel)
}
