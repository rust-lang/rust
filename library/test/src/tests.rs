use rand::RngCore;
use std::fs;
use std::path::PathBuf;

use super::*;

use crate::{
    console::OutputLocation,
    formatters::PrettyFormatter,
    test::{
        parse_opts,
        MetricMap,
        // FIXME (introduced by #65251)
        // ShouldPanic, StaticTestName, TestDesc, TestDescAndFn, TestOpts, TestTimeOptions,
        // TestType, TrFailedMsg, TrIgnored, TrOk,
    },
    time::{TestTimeOptions, TimeThreshold},
};

impl TestOpts {
    fn new() -> TestOpts {
        TestOpts {
            list: false,
            filters: vec![],
            filter_exact: false,
            force_run_in_process: false,
            exclude_should_panic: false,
            run_ignored: RunIgnored::No,
            run_tests: false,
            bench_benchmarks: false,
            logfile: None,
            nocapture: false,
            color: AutoColor,
            format: OutputFormat::Pretty,
            shuffle: false,
            shuffle_seed: None,
            test_threads: None,
            skip: vec![],
            time_options: None,
            options: Options::new(),
            fail_fast: false,
            output_postprocess_executable: None,
            output_postprocess_args: vec![],
        }
    }
}

// These implementations of TempDir and tmpdir are forked from rust/library/std/src/sys_common/io.rs.
struct TempDir(PathBuf);

impl TempDir {
    fn join(&self, path: &str) -> PathBuf {
        let TempDir(ref p) = *self;
        p.join(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let TempDir(ref p) = *self;
        let result = fs::remove_dir_all(p);
        // Avoid panicking while panicking as this causes the process to
        // immediately abort, without displaying test results.
        if !thread::panicking() {
            result.unwrap();
        }
    }
}

fn tmpdir() -> TempDir {
    let p = env::temp_dir();
    let mut r = rand::thread_rng();
    let ret = p.join(&format!("rust-{}", r.next_u32()));
    fs::create_dir(&ret).unwrap();
    TempDir(ret)
}

fn one_ignored_one_unignored_test() -> Vec<TestDescAndFn> {
    vec![
        TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("1"),
                ignore: true,
                ignore_message: None,
                source_file: "",
                start_line: 0,
                start_col: 0,
                end_line: 0,
                end_col: 0,
                should_panic: ShouldPanic::No,
                compile_fail: false,
                no_run: false,
                test_type: TestType::Unknown,
            },
            testfn: DynTestFn(Box::new(move || Ok(()))),
        },
        TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("2"),
                ignore: false,
                ignore_message: None,
                source_file: "",
                start_line: 0,
                start_col: 0,
                end_line: 0,
                end_col: 0,
                should_panic: ShouldPanic::No,
                compile_fail: false,
                no_run: false,
                test_type: TestType::Unknown,
            },
            testfn: DynTestFn(Box::new(move || Ok(()))),
        },
    ]
}

#[test]
pub fn do_not_run_ignored_tests() {
    fn f() -> Result<(), String> {
        panic!();
    }
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: true,
            ignore_message: None,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            should_panic: ShouldPanic::No,
            compile_fail: false,
            no_run: false,
            test_type: TestType::Unknown,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    run_test(&TestOpts::new(), false, TestId(0), desc, RunStrategy::InProcess, tx);
    let result = rx.recv().unwrap().result;
    assert_ne!(result, TrOk);
}

#[test]
pub fn ignored_tests_result_in_ignored() {
    fn f() -> Result<(), String> {
        Ok(())
    }
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: true,
            ignore_message: None,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            should_panic: ShouldPanic::No,
            compile_fail: false,
            no_run: false,
            test_type: TestType::Unknown,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    run_test(&TestOpts::new(), false, TestId(0), desc, RunStrategy::InProcess, tx);
    let result = rx.recv().unwrap().result;
    assert_eq!(result, TrIgnored);
}

// FIXME: Re-enable emscripten once it can catch panics again (introduced by #65251)
#[test]
#[cfg(not(target_os = "emscripten"))]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_should_panic() {
    fn f() -> Result<(), String> {
        panic!();
    }
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: false,
            ignore_message: None,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            should_panic: ShouldPanic::Yes,
            compile_fail: false,
            no_run: false,
            test_type: TestType::Unknown,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    run_test(&TestOpts::new(), false, TestId(0), desc, RunStrategy::InProcess, tx);
    let result = rx.recv().unwrap().result;
    assert_eq!(result, TrOk);
}

// FIXME: Re-enable emscripten once it can catch panics again (introduced by #65251)
#[test]
#[cfg(not(target_os = "emscripten"))]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_should_panic_good_message() {
    fn f() -> Result<(), String> {
        panic!("an error message");
    }
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: false,
            ignore_message: None,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            should_panic: ShouldPanic::YesWithMessage("error message"),
            compile_fail: false,
            no_run: false,
            test_type: TestType::Unknown,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    run_test(&TestOpts::new(), false, TestId(0), desc, RunStrategy::InProcess, tx);
    let result = rx.recv().unwrap().result;
    assert_eq!(result, TrOk);
}

// FIXME: Re-enable emscripten once it can catch panics again (introduced by #65251)
#[test]
#[cfg(not(target_os = "emscripten"))]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_should_panic_bad_message() {
    use crate::tests::TrFailedMsg;
    fn f() -> Result<(), String> {
        panic!("an error message");
    }
    let expected = "foobar";
    let failed_msg = r#"panic did not contain expected string
      panic message: `"an error message"`,
 expected substring: `"foobar"`"#;
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: false,
            ignore_message: None,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            should_panic: ShouldPanic::YesWithMessage(expected),
            compile_fail: false,
            no_run: false,
            test_type: TestType::Unknown,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    run_test(&TestOpts::new(), false, TestId(0), desc, RunStrategy::InProcess, tx);
    let result = rx.recv().unwrap().result;
    assert_eq!(result, TrFailedMsg(failed_msg.to_string()));
}

// FIXME: Re-enable emscripten once it can catch panics again (introduced by #65251)
#[test]
#[cfg(not(target_os = "emscripten"))]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_should_panic_non_string_message_type() {
    use crate::tests::TrFailedMsg;
    use std::any::TypeId;
    fn f() -> Result<(), String> {
        std::panic::panic_any(1i32);
    }
    let expected = "foobar";
    let failed_msg = format!(
        r#"expected panic with string value,
 found non-string value: `{:?}`
     expected substring: `"foobar"`"#,
        TypeId::of::<i32>()
    );
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: false,
            ignore_message: None,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            should_panic: ShouldPanic::YesWithMessage(expected),
            compile_fail: false,
            no_run: false,
            test_type: TestType::Unknown,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    run_test(&TestOpts::new(), false, TestId(0), desc, RunStrategy::InProcess, tx);
    let result = rx.recv().unwrap().result;
    assert_eq!(result, TrFailedMsg(failed_msg));
}

// FIXME: Re-enable emscripten once it can catch panics again (introduced by #65251)
#[test]
#[cfg(not(target_os = "emscripten"))]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_should_panic_but_succeeds() {
    let should_panic_variants = [ShouldPanic::Yes, ShouldPanic::YesWithMessage("error message")];

    for &should_panic in should_panic_variants.iter() {
        fn f() -> Result<(), String> {
            Ok(())
        }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: false,
                ignore_message: None,
                source_file: "",
                start_line: 0,
                start_col: 0,
                end_line: 0,
                end_col: 0,
                should_panic,
                compile_fail: false,
                no_run: false,
                test_type: TestType::Unknown,
            },
            testfn: DynTestFn(Box::new(f)),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, TestId(0), desc, RunStrategy::InProcess, tx);
        let result = rx.recv().unwrap().result;
        assert_eq!(
            result,
            TrFailedMsg("test did not panic as expected".to_string()),
            "should_panic == {:?}",
            should_panic
        );
    }
}

fn report_time_test_template(report_time: bool) -> Option<TestExecTime> {
    fn f() -> Result<(), String> {
        Ok(())
    }
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: false,
            ignore_message: None,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            should_panic: ShouldPanic::No,
            compile_fail: false,
            no_run: false,
            test_type: TestType::Unknown,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let time_options = if report_time { Some(TestTimeOptions::default()) } else { None };

    let test_opts = TestOpts { time_options, ..TestOpts::new() };
    let (tx, rx) = channel();
    run_test(&test_opts, false, TestId(0), desc, RunStrategy::InProcess, tx);
    let exec_time = rx.recv().unwrap().exec_time;
    exec_time
}

#[test]
fn test_should_not_report_time() {
    let exec_time = report_time_test_template(false);
    assert!(exec_time.is_none());
}

#[test]
fn test_should_report_time() {
    let exec_time = report_time_test_template(true);
    assert!(exec_time.is_some());
}

fn time_test_failure_template(test_type: TestType) -> TestResult {
    fn f() -> Result<(), String> {
        Ok(())
    }
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: false,
            ignore_message: None,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            should_panic: ShouldPanic::No,
            compile_fail: false,
            no_run: false,
            test_type,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    // `Default` will initialize all the thresholds to 0 milliseconds.
    let mut time_options = TestTimeOptions::default();
    time_options.error_on_excess = true;

    let test_opts = TestOpts { time_options: Some(time_options), ..TestOpts::new() };
    let (tx, rx) = channel();
    run_test(&test_opts, false, TestId(0), desc, RunStrategy::InProcess, tx);
    let result = rx.recv().unwrap().result;

    result
}

#[test]
fn test_error_on_exceed() {
    let types = [TestType::UnitTest, TestType::IntegrationTest, TestType::DocTest];

    for test_type in types.iter() {
        let result = time_test_failure_template(*test_type);

        assert_eq!(result, TestResult::TrTimedFail);
    }

    // Check that for unknown tests thresholds aren't applied.
    let result = time_test_failure_template(TestType::Unknown);
    assert_eq!(result, TestResult::TrOk);
}

fn typed_test_desc(test_type: TestType) -> TestDesc {
    TestDesc {
        name: StaticTestName("whatever"),
        ignore: false,
        ignore_message: None,
        source_file: "",
        start_line: 0,
        start_col: 0,
        end_line: 0,
        end_col: 0,
        should_panic: ShouldPanic::No,
        compile_fail: false,
        no_run: false,
        test_type,
    }
}

fn test_exec_time(millis: u64) -> TestExecTime {
    TestExecTime(Duration::from_millis(millis))
}

#[test]
fn test_time_options_threshold() {
    let unit = TimeThreshold::new(Duration::from_millis(50), Duration::from_millis(100));
    let integration = TimeThreshold::new(Duration::from_millis(500), Duration::from_millis(1000));
    let doc = TimeThreshold::new(Duration::from_millis(5000), Duration::from_millis(10000));

    let options = TestTimeOptions {
        error_on_excess: false,
        unit_threshold: unit.clone(),
        integration_threshold: integration.clone(),
        doctest_threshold: doc.clone(),
    };

    let test_vector = [
        (TestType::UnitTest, unit.warn.as_millis() - 1, false, false),
        (TestType::UnitTest, unit.warn.as_millis(), true, false),
        (TestType::UnitTest, unit.critical.as_millis(), true, true),
        (TestType::IntegrationTest, integration.warn.as_millis() - 1, false, false),
        (TestType::IntegrationTest, integration.warn.as_millis(), true, false),
        (TestType::IntegrationTest, integration.critical.as_millis(), true, true),
        (TestType::DocTest, doc.warn.as_millis() - 1, false, false),
        (TestType::DocTest, doc.warn.as_millis(), true, false),
        (TestType::DocTest, doc.critical.as_millis(), true, true),
    ];

    for (test_type, time, expected_warn, expected_critical) in test_vector.iter() {
        let test_desc = typed_test_desc(*test_type);
        let exec_time = test_exec_time(*time as u64);

        assert_eq!(options.is_warn(&test_desc, &exec_time), *expected_warn);
        assert_eq!(options.is_critical(&test_desc, &exec_time), *expected_critical);
    }
}

#[test]
fn parse_ignored_flag() {
    let args = vec!["progname".to_string(), "filter".to_string(), "--ignored".to_string()];
    let opts = parse_opts(&args).unwrap().unwrap();
    assert_eq!(opts.run_ignored, RunIgnored::Only);
}

#[test]
fn parse_show_output_flag() {
    let args = vec!["progname".to_string(), "filter".to_string(), "--show-output".to_string()];
    let opts = parse_opts(&args).unwrap().unwrap();
    assert!(opts.options.display_output);
}

#[test]
fn parse_include_ignored_flag() {
    let args = vec!["progname".to_string(), "filter".to_string(), "--include-ignored".to_string()];
    let opts = parse_opts(&args).unwrap().unwrap();
    assert_eq!(opts.run_ignored, RunIgnored::Yes);
}

#[test]
fn parse_output_postprocess() {
    let args = vec![
        "progname".to_string(),
        "filter".to_string(),
        "--output_postprocess_executable".to_string(),
        "/tmp/postprocess.sh".to_string(),
        "--output_postprocess_args".to_string(),
        "--test1=a".to_string(),
        "--output_postprocess_args=--test2=b".to_string(),
    ];
    let opts = parse_opts(&args).unwrap().unwrap();
    assert_eq!(opts.output_postprocess_executable, Some(PathBuf::from("/tmp/postprocess.sh")));
    assert_eq!(
        opts.output_postprocess_args,
        vec!["--test1=a".to_string(), "--test2=b".to_string()]
    );
}

#[test]
pub fn filter_for_ignored_option() {
    // When we run ignored tests the test filter should filter out all the
    // unignored tests and flip the ignore flag on the rest to false

    let mut opts = TestOpts::new();
    opts.run_tests = true;
    opts.run_ignored = RunIgnored::Only;

    let tests = one_ignored_one_unignored_test();
    let filtered = filter_tests(&opts, tests);

    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].desc.name.to_string(), "1");
    assert!(!filtered[0].desc.ignore);
}

#[test]
pub fn run_include_ignored_option() {
    // When we "--include-ignored" tests, the ignore flag should be set to false on
    // all tests and no test filtered out

    let mut opts = TestOpts::new();
    opts.run_tests = true;
    opts.run_ignored = RunIgnored::Yes;

    let tests = one_ignored_one_unignored_test();
    let filtered = filter_tests(&opts, tests);

    assert_eq!(filtered.len(), 2);
    assert!(!filtered[0].desc.ignore);
    assert!(!filtered[1].desc.ignore);
}

#[test]
pub fn exclude_should_panic_option() {
    let mut opts = TestOpts::new();
    opts.run_tests = true;
    opts.exclude_should_panic = true;

    let mut tests = one_ignored_one_unignored_test();
    tests.push(TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("3"),
            ignore: false,
            ignore_message: None,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            should_panic: ShouldPanic::Yes,
            compile_fail: false,
            no_run: false,
            test_type: TestType::Unknown,
        },
        testfn: DynTestFn(Box::new(move || Ok(()))),
    });

    let filtered = filter_tests(&opts, tests);

    assert_eq!(filtered.len(), 2);
    assert!(filtered.iter().all(|test| test.desc.should_panic == ShouldPanic::No));
}

#[test]
pub fn exact_filter_match() {
    fn tests() -> Vec<TestDescAndFn> {
        ["base", "base::test", "base::test1", "base::test2"]
            .into_iter()
            .map(|name| TestDescAndFn {
                desc: TestDesc {
                    name: StaticTestName(name),
                    ignore: false,
                    ignore_message: None,
                    source_file: "",
                    start_line: 0,
                    start_col: 0,
                    end_line: 0,
                    end_col: 0,
                    should_panic: ShouldPanic::No,
                    compile_fail: false,
                    no_run: false,
                    test_type: TestType::Unknown,
                },
                testfn: DynTestFn(Box::new(move || Ok(()))),
            })
            .collect()
    }

    let substr =
        filter_tests(&TestOpts { filters: vec!["base".into()], ..TestOpts::new() }, tests());
    assert_eq!(substr.len(), 4);

    let substr =
        filter_tests(&TestOpts { filters: vec!["bas".into()], ..TestOpts::new() }, tests());
    assert_eq!(substr.len(), 4);

    let substr =
        filter_tests(&TestOpts { filters: vec!["::test".into()], ..TestOpts::new() }, tests());
    assert_eq!(substr.len(), 3);

    let substr =
        filter_tests(&TestOpts { filters: vec!["base::test".into()], ..TestOpts::new() }, tests());
    assert_eq!(substr.len(), 3);

    let substr = filter_tests(
        &TestOpts { filters: vec!["test1".into(), "test2".into()], ..TestOpts::new() },
        tests(),
    );
    assert_eq!(substr.len(), 2);

    let exact = filter_tests(
        &TestOpts { filters: vec!["base".into()], filter_exact: true, ..TestOpts::new() },
        tests(),
    );
    assert_eq!(exact.len(), 1);

    let exact = filter_tests(
        &TestOpts { filters: vec!["bas".into()], filter_exact: true, ..TestOpts::new() },
        tests(),
    );
    assert_eq!(exact.len(), 0);

    let exact = filter_tests(
        &TestOpts { filters: vec!["::test".into()], filter_exact: true, ..TestOpts::new() },
        tests(),
    );
    assert_eq!(exact.len(), 0);

    let exact = filter_tests(
        &TestOpts { filters: vec!["base::test".into()], filter_exact: true, ..TestOpts::new() },
        tests(),
    );
    assert_eq!(exact.len(), 1);

    let exact = filter_tests(
        &TestOpts {
            filters: vec!["base".into(), "base::test".into()],
            filter_exact: true,
            ..TestOpts::new()
        },
        tests(),
    );
    assert_eq!(exact.len(), 2);
}

fn sample_tests() -> Vec<TestDescAndFn> {
    let names = vec![
        "sha1::test".to_string(),
        "isize::test_to_str".to_string(),
        "isize::test_pow".to_string(),
        "test::do_not_run_ignored_tests".to_string(),
        "test::ignored_tests_result_in_ignored".to_string(),
        "test::first_free_arg_should_be_a_filter".to_string(),
        "test::parse_ignored_flag".to_string(),
        "test::parse_include_ignored_flag".to_string(),
        "test::filter_for_ignored_option".to_string(),
        "test::run_include_ignored_option".to_string(),
        "test::sort_tests".to_string(),
    ];
    fn testfn() -> Result<(), String> {
        Ok(())
    }
    let mut tests = Vec::new();
    for name in &names {
        let test = TestDescAndFn {
            desc: TestDesc {
                name: DynTestName((*name).clone()),
                ignore: false,
                ignore_message: None,
                source_file: "",
                start_line: 0,
                start_col: 0,
                end_line: 0,
                end_col: 0,
                should_panic: ShouldPanic::No,
                compile_fail: false,
                no_run: false,
                test_type: TestType::Unknown,
            },
            testfn: DynTestFn(Box::new(testfn)),
        };
        tests.push(test);
    }
    tests
}

#[test]
pub fn shuffle_tests() {
    let mut opts = TestOpts::new();
    opts.shuffle = true;

    let shuffle_seed = get_shuffle_seed(&opts).unwrap();

    let left =
        sample_tests().into_iter().enumerate().map(|(i, e)| (TestId(i), e)).collect::<Vec<_>>();
    let mut right =
        sample_tests().into_iter().enumerate().map(|(i, e)| (TestId(i), e)).collect::<Vec<_>>();

    assert!(left.iter().zip(&right).all(|(a, b)| a.1.desc.name == b.1.desc.name));

    helpers::shuffle::shuffle_tests(shuffle_seed, right.as_mut_slice());

    assert!(left.iter().zip(right).any(|(a, b)| a.1.desc.name != b.1.desc.name));
}

#[test]
pub fn shuffle_tests_with_seed() {
    let mut opts = TestOpts::new();
    opts.shuffle = true;

    let shuffle_seed = get_shuffle_seed(&opts).unwrap();

    let mut left =
        sample_tests().into_iter().enumerate().map(|(i, e)| (TestId(i), e)).collect::<Vec<_>>();
    let mut right =
        sample_tests().into_iter().enumerate().map(|(i, e)| (TestId(i), e)).collect::<Vec<_>>();

    helpers::shuffle::shuffle_tests(shuffle_seed, left.as_mut_slice());
    helpers::shuffle::shuffle_tests(shuffle_seed, right.as_mut_slice());

    assert!(left.iter().zip(right).all(|(a, b)| a.1.desc.name == b.1.desc.name));
}

#[test]
pub fn order_depends_on_more_than_seed() {
    let mut opts = TestOpts::new();
    opts.shuffle = true;

    let shuffle_seed = get_shuffle_seed(&opts).unwrap();

    let mut left_tests = sample_tests();
    let mut right_tests = sample_tests();

    left_tests.pop();
    right_tests.remove(0);

    let mut left =
        left_tests.into_iter().enumerate().map(|(i, e)| (TestId(i), e)).collect::<Vec<_>>();
    let mut right =
        right_tests.into_iter().enumerate().map(|(i, e)| (TestId(i), e)).collect::<Vec<_>>();

    assert_eq!(left.len(), right.len());

    assert!(left.iter().zip(&right).all(|(a, b)| a.0 == b.0));

    helpers::shuffle::shuffle_tests(shuffle_seed, left.as_mut_slice());
    helpers::shuffle::shuffle_tests(shuffle_seed, right.as_mut_slice());

    assert!(left.iter().zip(right).any(|(a, b)| a.0 != b.0));
}

#[test]
pub fn test_metricmap_compare() {
    let mut m1 = MetricMap::new();
    let mut m2 = MetricMap::new();
    m1.insert_metric("in-both-noise", 1000.0, 200.0);
    m2.insert_metric("in-both-noise", 1100.0, 200.0);

    m1.insert_metric("in-first-noise", 1000.0, 2.0);
    m2.insert_metric("in-second-noise", 1000.0, 2.0);

    m1.insert_metric("in-both-want-downwards-but-regressed", 1000.0, 10.0);
    m2.insert_metric("in-both-want-downwards-but-regressed", 2000.0, 10.0);

    m1.insert_metric("in-both-want-downwards-and-improved", 2000.0, 10.0);
    m2.insert_metric("in-both-want-downwards-and-improved", 1000.0, 10.0);

    m1.insert_metric("in-both-want-upwards-but-regressed", 2000.0, -10.0);
    m2.insert_metric("in-both-want-upwards-but-regressed", 1000.0, -10.0);

    m1.insert_metric("in-both-want-upwards-and-improved", 1000.0, -10.0);
    m2.insert_metric("in-both-want-upwards-and-improved", 2000.0, -10.0);
}

#[test]
pub fn test_bench_once_no_iter() {
    fn f(_: &mut Bencher) -> Result<(), String> {
        Ok(())
    }
    bench::run_once(f).unwrap();
}

#[test]
pub fn test_bench_once_iter() {
    fn f(b: &mut Bencher) -> Result<(), String> {
        b.iter(|| {});
        Ok(())
    }
    bench::run_once(f).unwrap();
}

#[test]
pub fn test_bench_no_iter() {
    fn f(_: &mut Bencher) -> Result<(), String> {
        Ok(())
    }

    let (tx, rx) = channel();

    let desc = TestDesc {
        name: StaticTestName("f"),
        ignore: false,
        ignore_message: None,
        source_file: "",
        start_line: 0,
        start_col: 0,
        end_line: 0,
        end_col: 0,
        should_panic: ShouldPanic::No,
        compile_fail: false,
        no_run: false,
        test_type: TestType::Unknown,
    };

    crate::bench::benchmark(TestId(0), desc, tx, true, f);
    rx.recv().unwrap();
}

#[test]
pub fn test_bench_iter() {
    fn f(b: &mut Bencher) -> Result<(), String> {
        b.iter(|| {});
        Ok(())
    }

    let (tx, rx) = channel();

    let desc = TestDesc {
        name: StaticTestName("f"),
        ignore: false,
        ignore_message: None,
        source_file: "",
        start_line: 0,
        start_col: 0,
        end_line: 0,
        end_col: 0,
        should_panic: ShouldPanic::No,
        compile_fail: false,
        no_run: false,
        test_type: TestType::Unknown,
    };

    crate::bench::benchmark(TestId(0), desc, tx, true, f);
    rx.recv().unwrap();
}

#[test]
fn should_sort_failures_before_printing_them() {
    let test_a = TestDesc {
        name: StaticTestName("a"),
        ignore: false,
        ignore_message: None,
        source_file: "",
        start_line: 0,
        start_col: 0,
        end_line: 0,
        end_col: 0,
        should_panic: ShouldPanic::No,
        compile_fail: false,
        no_run: false,
        test_type: TestType::Unknown,
    };

    let test_b = TestDesc {
        name: StaticTestName("b"),
        ignore: false,
        ignore_message: None,
        source_file: "",
        start_line: 0,
        start_col: 0,
        end_line: 0,
        end_col: 0,
        should_panic: ShouldPanic::No,
        compile_fail: false,
        no_run: false,
        test_type: TestType::Unknown,
    };

    let mut out = PrettyFormatter::new(OutputLocation::Raw(Vec::new()), false, 10, false, None);

    let st = console::ConsoleTestState {
        log_out: None,
        total: 0,
        passed: 0,
        failed: 0,
        ignored: 0,
        filtered_out: 0,
        measured: 0,
        exec_time: None,
        metrics: MetricMap::new(),
        failures: vec![(test_b, Vec::new()), (test_a, Vec::new())],
        options: Options::new(),
        not_failures: Vec::new(),
        ignores: Vec::new(),
        time_failures: Vec::new(),
    };

    out.write_failures(&st).unwrap();
    let s = match out.output_location() {
        &OutputLocation::Raw(ref m) => String::from_utf8_lossy(&m[..]),
        &OutputLocation::Pretty(_) => unreachable!(),
    };

    let apos = s.find("a").unwrap();
    let bpos = s.find("b").unwrap();
    assert!(apos < bpos);
}

#[test]
#[cfg(not(target_os = "emscripten"))]
fn test_dyn_bench_returning_err_fails_when_run_as_test() {
    fn f(_: &mut Bencher) -> Result<(), String> {
        Result::Err("An error".into())
    }
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: false,
            ignore_message: None,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            should_panic: ShouldPanic::No,
            compile_fail: false,
            no_run: false,
            test_type: TestType::Unknown,
        },
        testfn: DynBenchFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    let notify = move |event: TestEvent| {
        if let TestEvent::TeResult(result) = event {
            tx.send(result).unwrap();
        }
        Ok(())
    };
    run_tests(&TestOpts { run_tests: true, ..TestOpts::new() }, vec![desc], notify).unwrap();
    let result = rx.recv().unwrap().result;
    assert_eq!(result, TrFailed);
}

#[test]
fn test_output_postprocessing() {
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: false,
            ignore_message: None,
            source_file: "",
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
            should_panic: ShouldPanic::No,
            compile_fail: false,
            no_run: false,
            test_type: TestType::Unknown,
        },
        testfn: DynTestFn(Box::new(move || Ok(()))),
    };

    let mut test_postprocessor: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if cfg!(target_os = "windows") {
        test_postprocessor.push("src/testdata/postprocess.cmd");
    } else {
        test_postprocessor.push("src/testdata/postprocess.sh");
    }

    let tmpdir = tmpdir();
    let output_path = &tmpdir.join("output.txt");

    std::env::set_var("TEST_POSTPROCESSOR_OUTPUT_FILE", output_path);

    let opts = TestOpts {
        run_tests: true,
        output_postprocess_executable: Some(test_postprocessor),
        output_postprocess_args: vec!["--test1=a".to_string(), "--test2=b".to_string()],
        format: OutputFormat::Json,
        ..TestOpts::new()
    };
    run_tests_console(&opts, vec![desc]).unwrap();

    // Read output and replace the decimal value at `"exec_time": 0.000084974` to make the text deterministic.
    // This replacement could be done easier with a regex, but `std` has no regex.
    let mut contents =
        fs::read_to_string(output_path).expect("Test postprocessor did not create file");
    let replace_trigger = r#""exec_time": "#;
    let replace_start =
        contents.find(replace_trigger).expect("exec_time not found in the output JSON")
            + replace_trigger.len();
    let replace_end = replace_start
        + contents[replace_start..]
            .find(' ')
            .expect("No space found after the decimal value for exec_time");
    contents.replace_range(replace_start..replace_end, "AAA.BBB");

    // Split output at line breaks to make the comparison platform-agnostic regarding newline style.
    let contents_lines = contents.as_str().lines().collect::<Vec<&str>>();

    let expected_lines = vec![
        r#"{ "type": "suite", "event": "started", "test_count": 1 }"#,
        r#"{ "type": "test", "event": "started", "name": "whatever" }"#,
        r#"{ "type": "test", "name": "whatever", "event": "ok" }"#,
        r#"{ "type": "suite", "event": "ok", "passed": 1, "failed": 0, "ignored": 0, "measured": 0, "filtered_out": 0, "exec_time": AAA.BBB }"#,
        r#"--test1=a"#,
        r#"--test2=b"#,
    ];

    assert_eq!(contents_lines, expected_lines);
}
