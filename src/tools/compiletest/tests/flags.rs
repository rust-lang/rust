use std::ops::Deref;
use std::path::{Path, PathBuf};

use assert_cmd::assert::Assert;
use assert_cmd::{Command, cargo_bin_cmd};
use bstr::ByteSlice;
use build_helper::metrics::compiletest::{Message, TestMessage};
use predicates::str::is_empty;
use rand::Rng;

fn path_str(p: PathBuf) -> String {
    p.canonicalize().unwrap().to_str().unwrap().to_owned()
}

// NOTE: incomplete, you'll probably need to extend this if you add more self-tests.
fn compiletest() -> Command {
    let src_root = path_str(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../"));
    let build_host_root = path_str(Path::new(env!("CARGO_TARGET_DIR")).join("../"));
    let build_root = path_str(Path::new(&build_host_root).join("../"));
    let host = env!("CFG_COMPILER_BUILD_TRIPLE");
    let sysroot =
        path_str(Path::new(&std::env::var("TEST_RUSTC").unwrap()).parent().unwrap().join(".."));
    let mut rng = rand::rng();

    let mut compiletest = cargo_bin_cmd!("compiletest");
    compiletest.args([
        "--mode",
        "ui",
        "--suite",
        "ui",
        "--compile-lib-path=",
        "--run-lib-path",
        &std::env::var("COMPILETEST_RUNTIME_PATH").unwrap(),
        "--python=",
        "--jsondocck-path=",
        "--src-root",
        &src_root,
        "--src-test-suite-root",
        &(src_root.clone() + "/tests/ui"),
        "--build-root",
        &build_root,
        "--build-test-suite-root",
        // needs to be random so different tests don't conflict with each other
        &(build_host_root.clone()
            + "/test/compiletest-integration-test-"
            + &rng.random::<u32>().to_string()),
        "--sysroot-base",
        &sysroot,
        "--cc=c",
        "--cxx=c++",
        "--cflags=",
        "--cxxflags=",
        "--llvm-components=",
        "--android-cross-path=",
        "--stage",
        "2",
        "--stage-id",
        &format!("stage2-{host}"),
        "--channel",
        env!("CFG_RELEASE_CHANNEL"),
        "--host",
        host,
        "--target",
        host,
        "--nightly-branch=",
        "--git-merge-commit-email=",
        "--minicore-path=",
        "--jobs=0",
        "--rustc-path",
        &std::env::var("TEST_RUSTC").expect("run this test through bootstrap"),
        "compiletest-self-test",
    ]);

    compiletest.current_dir(src_root);

    compiletest
}

fn run_ok(mut compiletest: Command) -> Assert {
    compiletest.assert().success().stderr(is_empty())
}

#[derive(serde_derive::Deserialize, Debug)]
struct CompiletestMetrics(Vec<Message>);

impl CompiletestMetrics {
    fn find_test(&self, expected: &str) -> Option<&TestMessage> {
        self.0.iter().find_map(|msg| {
            let (msg, name) = match msg {
                Message::Test(
                    inner @ TestMessage::Ok(outcome)
                    | inner @ TestMessage::Ignored(outcome)
                    | inner @ TestMessage::Failed(outcome)
                    | inner @ TestMessage::FilteredOut(outcome),
                ) => (inner, &outcome.name),
                Message::Test(inner @ TestMessage::Timeout { name }) => (inner, name),
                _ => return None,
            };
            let stripped = name.strip_prefix("[ui] tests/ui/compiletest-self-test/").unwrap()
                .strip_suffix(".rs").unwrap();
            if stripped == expected {
                Some(msg)
            } else {
                None
            }
        })
    }
}

impl Deref for CompiletestMetrics {
    type Target = Vec<Message>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

trait CompiletestMetricsExt {
    fn metrics(&self) -> CompiletestMetrics;
}

impl CompiletestMetricsExt for Assert {
    fn metrics(&self) -> CompiletestMetrics {
        let lines = self.get_output().stdout.lines();
        let parsed = lines.map(serde_json::from_slice).collect::<Result<_, _>>().unwrap();
        CompiletestMetrics(parsed)
    }
}

#[test]
fn ignore_directive_tests_reported_ignored() {
    let mut ct = compiletest();
    let metrics = dbg!(run_ok(ct).metrics());
    assert!(matches!(metrics.find_test("ignore-directive").unwrap(), TestMessage::Ignored(..)));
}
