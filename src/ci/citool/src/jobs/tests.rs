use crate::jobs::{JobDatabase, load_job_db};

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
}

#[track_caller]
fn check_pattern(db: &JobDatabase, pattern: &str, expected: &[&str]) {
    let jobs =
        db.find_auto_jobs_by_pattern(pattern).into_iter().map(|j| j.name).collect::<Vec<_>>();

    assert_eq!(jobs, expected);
}
