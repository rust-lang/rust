use std::process::{Command, Stdio};

const TEST_JOBS_YML_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/test-jobs.yml");

#[test]
fn auto_jobs() {
    let stdout = get_matrix("push", "commit", "refs/heads/auto");
    insta::assert_snapshot!(stdout, @r#"
    jobs=[{"name":"aarch64-gnu","full_name":"auto - aarch64-gnu","os":"ubuntu-22.04-arm","env":{"ARTIFACTS_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZN24CBO55","AWS_REGION":"us-west-1","CACHES_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZI5DHEBFL","DEPLOY_BUCKET":"rust-lang-ci2","TOOLSTATE_PUBLISH":1},"free_disk":true},{"name":"x86_64-gnu-llvm-18-1","full_name":"auto - x86_64-gnu-llvm-18-1","os":"ubuntu-24.04","env":{"ARTIFACTS_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZN24CBO55","AWS_REGION":"us-west-1","CACHES_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZI5DHEBFL","DEPLOY_BUCKET":"rust-lang-ci2","DOCKER_SCRIPT":"stage_2_test_set1.sh","IMAGE":"x86_64-gnu-llvm-18","READ_ONLY_SRC":"0","RUST_BACKTRACE":1,"TOOLSTATE_PUBLISH":1},"free_disk":true},{"name":"aarch64-apple","full_name":"auto - aarch64-apple","os":"macos-14","env":{"ARTIFACTS_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZN24CBO55","AWS_REGION":"us-west-1","CACHES_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZI5DHEBFL","DEPLOY_BUCKET":"rust-lang-ci2","MACOSX_DEPLOYMENT_TARGET":11.0,"MACOSX_STD_DEPLOYMENT_TARGET":11.0,"NO_DEBUG_ASSERTIONS":1,"NO_LLVM_ASSERTIONS":1,"NO_OVERFLOW_CHECKS":1,"RUSTC_RETRY_LINKER_ON_SEGFAULT":1,"RUST_CONFIGURE_ARGS":"--enable-sanitizers --enable-profiler --set rust.jemalloc","SCRIPT":"./x.py --stage 2 test --host=aarch64-apple-darwin --target=aarch64-apple-darwin","SELECT_XCODE":"/Applications/Xcode_15.4.app","TOOLSTATE_PUBLISH":1,"USE_XCODE_CLANG":1}},{"name":"dist-i686-msvc","full_name":"auto - dist-i686-msvc","os":"windows-2022","env":{"ARTIFACTS_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZN24CBO55","AWS_REGION":"us-west-1","CACHES_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZI5DHEBFL","CODEGEN_BACKENDS":"llvm,cranelift","DEPLOY_BUCKET":"rust-lang-ci2","DIST_REQUIRE_ALL_TOOLS":1,"RUST_CONFIGURE_ARGS":"--build=i686-pc-windows-msvc --host=i686-pc-windows-msvc --target=i686-pc-windows-msvc,i586-pc-windows-msvc --enable-full-tools --enable-profiler","SCRIPT":"python x.py dist bootstrap --include-default-paths","TOOLSTATE_PUBLISH":1}},{"name":"pr-check-1","full_name":"auto - pr-check-1","os":"ubuntu-24.04","env":{"ARTIFACTS_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZN24CBO55","AWS_REGION":"us-west-1","CACHES_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZI5DHEBFL","DEPLOY_BUCKET":"rust-lang-ci2","TOOLSTATE_PUBLISH":1},"continue_on_error":false,"free_disk":true},{"name":"pr-check-2","full_name":"auto - pr-check-2","os":"ubuntu-24.04","env":{"ARTIFACTS_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZN24CBO55","AWS_REGION":"us-west-1","CACHES_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZI5DHEBFL","DEPLOY_BUCKET":"rust-lang-ci2","TOOLSTATE_PUBLISH":1},"continue_on_error":false,"free_disk":true},{"name":"tidy","full_name":"auto - tidy","os":"ubuntu-24.04","env":{"ARTIFACTS_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZN24CBO55","AWS_REGION":"us-west-1","CACHES_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZI5DHEBFL","DEPLOY_BUCKET":"rust-lang-ci2","TOOLSTATE_PUBLISH":1},"continue_on_error":false,"free_disk":true,"doc_url":"https://foo.bar"}]
    run_type=auto
    "#);
}

#[test]
fn try_jobs() {
    let stdout = get_matrix("push", "commit", "refs/heads/try");
    insta::assert_snapshot!(stdout, @r###"
    jobs=[{"name":"dist-x86_64-linux","full_name":"try - dist-x86_64-linux","os":"ubuntu-22.04-16core-64gb","env":{"ARTIFACTS_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZN24CBO55","AWS_REGION":"us-west-1","CACHES_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZI5DHEBFL","CODEGEN_BACKENDS":"llvm,cranelift","DEPLOY_BUCKET":"rust-lang-ci2","DIST_TRY_BUILD":1,"TOOLSTATE_PUBLISH":1}}]
    run_type=try
    "###);
}

#[test]
fn try_custom_jobs() {
    let stdout = get_matrix(
        "push",
        r#"This is a test PR

try-job: aarch64-gnu
try-job: dist-i686-msvc"#,
        "refs/heads/try",
    );
    insta::assert_snapshot!(stdout, @r###"
    jobs=[{"name":"aarch64-gnu","full_name":"try - aarch64-gnu","os":"ubuntu-22.04-arm","env":{"ARTIFACTS_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZN24CBO55","AWS_REGION":"us-west-1","CACHES_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZI5DHEBFL","DEPLOY_BUCKET":"rust-lang-ci2","TOOLSTATE_PUBLISH":1},"free_disk":true},{"name":"dist-i686-msvc","full_name":"try - dist-i686-msvc","os":"windows-2022","env":{"ARTIFACTS_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZN24CBO55","AWS_REGION":"us-west-1","CACHES_AWS_ACCESS_KEY_ID":"AKIA46X5W6CZI5DHEBFL","CODEGEN_BACKENDS":"llvm,cranelift","DEPLOY_BUCKET":"rust-lang-ci2","DIST_REQUIRE_ALL_TOOLS":1,"RUST_CONFIGURE_ARGS":"--build=i686-pc-windows-msvc --host=i686-pc-windows-msvc --target=i686-pc-windows-msvc,i586-pc-windows-msvc --enable-full-tools --enable-profiler","SCRIPT":"python x.py dist bootstrap --include-default-paths","TOOLSTATE_PUBLISH":1}}]
    run_type=try
    "###);
}

#[test]
fn pr_jobs() {
    let stdout = get_matrix("pull_request", "commit", "refs/heads/pr/1234");
    insta::assert_snapshot!(stdout, @r#"
    jobs=[{"name":"pr-check-1","full_name":"PR - pr-check-1","os":"ubuntu-24.04","env":{"PR_CI_JOB":1},"free_disk":true},{"name":"pr-check-2","full_name":"PR - pr-check-2","os":"ubuntu-24.04","env":{"PR_CI_JOB":1},"free_disk":true},{"name":"tidy","full_name":"PR - tidy","os":"ubuntu-24.04","env":{"PR_CI_JOB":1},"continue_on_error":true,"free_disk":true,"doc_url":"https://foo.bar"}]
    run_type=pr
    "#);
}

#[test]
fn master_jobs() {
    let stdout = get_matrix("push", "commit", "refs/heads/master");
    insta::assert_snapshot!(stdout, @r#"
    jobs=[]
    run_type=master
    "#);
}

fn get_matrix(event_name: &str, commit_msg: &str, branch_ref: &str) -> String {
    let path = std::env::var("PATH");
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "-q", "calculate-job-matrix", "--jobs-file", TEST_JOBS_YML_PATH])
        .env_clear()
        .env("GITHUB_EVENT_NAME", event_name)
        .env("COMMIT_MESSAGE", commit_msg)
        .env("GITHUB_REF", branch_ref)
        .env("GITHUB_RUN_ID", "123")
        .env("GITHUB_RUN_ATTEMPT", "1")
        .stdout(Stdio::piped());
    if let Ok(path) = path {
        cmd.env("PATH", path);
    }

    let output = cmd.output().expect("Failed to execute command");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    if !output.status.success() {
        panic!("cargo run failed: {}\n{}", stdout, stderr);
    }
    stdout
}
