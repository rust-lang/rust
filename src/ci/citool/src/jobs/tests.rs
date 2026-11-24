use std::collections::BTreeMap;
use std::path::Path;

use super::Job;
use crate::jobs::{JobDatabase, load_job_db};
use crate::{DOCKER_DIRECTORY, JOBS_YML_PATH, utils};

#[test]
fn lookup_job_pattern() {
    let db = load_job_db(
        r#"
envs:
  pr:
  try:
  auto:

pr:
try:
auto:
    - name: dist-a
      os: ubuntu
      env: {}
    - name: dist-a-alt
      os: ubuntu
      env: {}
    - name: dist-b
      os: ubuntu
      env: {}
    - name: dist-b-alt
      os: ubuntu
      env: {}
    - name: test-a
      os: ubuntu
      env: {}
    - name: test-a-alt
      os: ubuntu
      env: {}
    - name: test-i686
      os: ubuntu
      env: {}
    - name: dist-i686
      os: ubuntu
      env: {}
    - name: test-msvc-i686-1
      os: ubuntu
      env: {}
    - name: test-msvc-i686-2
      os: ubuntu
      env: {}
optional:
    - name: optional-job-1
      os: ubuntu
      env: {}
    - name: optional-dist-x86_64
      os: ubuntu
      env: {}
"#,
    )
    .unwrap();
    check_pattern(&db, "dist-*", &["dist-a", "dist-a-alt", "dist-b", "dist-b-alt", "dist-i686"]);
    check_pattern(&db, "*-alt", &["dist-a-alt", "dist-b-alt", "test-a-alt"]);
    check_pattern(&db, "dist*-alt", &["dist-a-alt", "dist-b-alt"]);
    check_pattern(
        &db,
        "*i686*",
        &["test-i686", "dist-i686", "test-msvc-i686-1", "test-msvc-i686-2"],
    );
    // Test that optional jobs are found
    check_pattern(&db, "optional-*", &["optional-job-1", "optional-dist-x86_64"]);
    check_pattern(&db, "*optional*", &["optional-job-1", "optional-dist-x86_64"]);
}

#[track_caller]
fn check_pattern(db: &JobDatabase, pattern: &str, expected: &[&str]) {
    let jobs = db
        .find_auto_or_optional_jobs_by_pattern(pattern)
        .into_iter()
        .map(|j| j.name)
        .collect::<Vec<_>>();

    assert_eq!(jobs, expected);
}

#[test]
fn pr_job_implies_auto_job() {
    let db = load_job_db(
        r#"
envs:
  pr:
  try:
  auto:
  optional:

pr:
    - name: test-pr-ci-a
      os: ubuntu
      env: {}
try:
auto:
optional:
"#,
    )
    .unwrap();

    assert_eq!(
        db.auto_jobs.iter().map(|j| j.name.as_str()).collect::<Vec<_>>(),
        vec!["test-pr-ci-a"]
    )
}

#[test]
fn implied_auto_job_keeps_env_and_fails_fast() {
    let db = load_job_db(
        r#"
envs:
  pr:
  try:
  auto:
  optional:

pr:
    - name: test-tidy
      env:
        DEPLOY_TOOLSTATES_JSON: toolstates-linux.json
      continue_on_error: true
      os: ubuntu
try:
auto:
optional:
"#,
    )
    .unwrap();

    assert_eq!(db.auto_jobs.iter().map(|j| j.name.as_str()).collect::<Vec<_>>(), vec!["test-tidy"]);
    assert_eq!(db.auto_jobs[0].continue_on_error, Some(false));
    assert_eq!(
        db.auto_jobs[0].env,
        BTreeMap::from([(
            "DEPLOY_TOOLSTATES_JSON".to_string(),
            serde_yaml::Value::String("toolstates-linux.json".to_string())
        )])
    );
}

#[test]
#[should_panic = "duplicate"]
fn duplicate_job_name() {
    let _ = load_job_db(
        r#"
envs:
  pr:
  try:
  auto:


pr:
    - name: test-pr-ci-a
      os: ubuntu
      env: {}
    - name: test-pr-ci-a
      os: ubuntu
      env: {}
try:
auto:
optional:
"#,
    )
    .unwrap();
}

#[test]
fn auto_job_can_override_pr_job_spec() {
    let db = load_job_db(
        r#"
envs:
  pr:
  try:
  auto:
  optional:

pr:
    - name: test-tidy
      os: ubuntu
      env: {}
try:
auto:
    - name: test-tidy
      env:
        DEPLOY_TOOLSTATES_JSON: toolstates-linux.json
      continue_on_error: false
      os: ubuntu
optional:
"#,
    )
    .unwrap();

    assert_eq!(db.auto_jobs.iter().map(|j| j.name.as_str()).collect::<Vec<_>>(), vec!["test-tidy"]);
    assert_eq!(db.auto_jobs[0].continue_on_error, Some(false));
    assert_eq!(
        db.auto_jobs[0].env,
        BTreeMap::from([(
            "DEPLOY_TOOLSTATES_JSON".to_string(),
            serde_yaml::Value::String("toolstates-linux.json".to_string())
        )])
    );
}

#[test]
fn compatible_divergence_pr_auto_job() {
    let db = load_job_db(
        r#"
envs:
  pr:
  try:
  auto:
  optional:

pr:
    - name: test-tidy
      continue_on_error: true
      env:
        ENV_ALLOWED_TO_DIFFER: "hello world"
      os: ubuntu
try:
auto:
    - name: test-tidy
      continue_on_error: false
      env:
        ENV_ALLOWED_TO_DIFFER: "goodbye world"
      os: ubuntu
optional:
"#,
    )
    .unwrap();

    // `continue_on_error` and `env` are carve-outs *allowed* to diverge between PR and Auto job of
    // the same name. Should load successfully.

    assert_eq!(db.auto_jobs.iter().map(|j| j.name.as_str()).collect::<Vec<_>>(), vec!["test-tidy"]);
    assert_eq!(db.auto_jobs[0].continue_on_error, Some(false));
    assert_eq!(
        db.auto_jobs[0].env,
        BTreeMap::from([(
            "ENV_ALLOWED_TO_DIFFER".to_string(),
            serde_yaml::Value::String("goodbye world".to_string())
        )])
    );
}

#[test]
#[should_panic = "differs"]
fn incompatible_divergence_pr_auto_job() {
    // `os` is not one of the carve-out options allowed to diverge. This should fail.
    let _ = load_job_db(
        r#"
envs:
  pr:
  try:
  auto:
  optional:

pr:
    - name: test-tidy
      continue_on_error: true
      env:
        ENV_ALLOWED_TO_DIFFER: "hello world"
      os: ubuntu
try:
auto:
    - name: test-tidy
      continue_on_error: false
      env:
        ENV_ALLOWED_TO_DIFFER: "goodbye world"
      os: windows
optional:
"#,
    )
    .unwrap();
}

#[test]
#[should_panic = "cannot have `continue_on_error: true`"]
fn auto_job_continue_on_error() {
    // Auto CI jobs must fail-fast.
    let _ = load_job_db(
        r#"
envs:
  pr:
  try:
  auto:
  optional:

pr:
try:
auto:
    - name: test-tidy
      continue_on_error: true
      os: windows
      env: {}
optional:
"#,
    )
    .unwrap();
}

#[test]
#[should_panic = "must start with test-"]
fn missing_test_job_prefix() {
    let _ = load_job_db(
        r#"
envs:
  pr:
  try:
  auto:
  optional:

pr:
try:
auto:
    - name: tidy
      os: windows
      env: {}
optional:
"#,
    )
    .unwrap();
}
