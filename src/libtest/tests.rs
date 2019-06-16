use crate::bench;
use crate::test::{
    filter_tests, parse_opts, run_test, DynTestFn, DynTestName, MetricMap, RunIgnored,
    ShouldPanic, StaticTestName, TestDesc, TestDescAndFn, TestOpts, TrFailed, TrFailedMsg,
    TrIgnored, TrOk,
};
use crate::Bencher;
use crate::Concurrent;
use std::sync::mpsc::channel;

fn one_ignored_one_unignored_test() -> Vec<TestDescAndFn> {
    vec![
        TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("1"),
                ignore: true,
                should_panic: ShouldPanic::No,
                allow_fail: false,
            },
            testfn: DynTestFn(Box::new(move || {})),
        },
        TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("2"),
                ignore: false,
                should_panic: ShouldPanic::No,
                allow_fail: false,
            },
            testfn: DynTestFn(Box::new(move || {})),
        },
    ]
}

#[test]
pub fn do_not_run_ignored_tests() {
    fn f() {
        panic!();
    }
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: true,
            should_panic: ShouldPanic::No,
            allow_fail: false,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    run_test(&TestOpts::new(), false, desc, tx, Concurrent::No);
    let (_, res, _) = rx.recv().unwrap();
    assert!(res != TrOk);
}

#[test]
pub fn ignored_tests_result_in_ignored() {
    fn f() {}
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: true,
            should_panic: ShouldPanic::No,
            allow_fail: false,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    run_test(&TestOpts::new(), false, desc, tx, Concurrent::No);
    let (_, res, _) = rx.recv().unwrap();
    assert!(res == TrIgnored);
}

#[test]
fn test_should_panic() {
    fn f() {
        panic!();
    }
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: false,
            should_panic: ShouldPanic::Yes,
            allow_fail: false,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    run_test(&TestOpts::new(), false, desc, tx, Concurrent::No);
    let (_, res, _) = rx.recv().unwrap();
    assert!(res == TrOk);
}

#[test]
fn test_should_panic_good_message() {
    fn f() {
        panic!("an error message");
    }
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: false,
            should_panic: ShouldPanic::YesWithMessage("error message"),
            allow_fail: false,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    run_test(&TestOpts::new(), false, desc, tx, Concurrent::No);
    let (_, res, _) = rx.recv().unwrap();
    assert!(res == TrOk);
}

#[test]
fn test_should_panic_bad_message() {
    fn f() {
        panic!("an error message");
    }
    let expected = "foobar";
    let failed_msg = "panic did not include expected string";
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: false,
            should_panic: ShouldPanic::YesWithMessage(expected),
            allow_fail: false,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    run_test(&TestOpts::new(), false, desc, tx, Concurrent::No);
    let (_, res, _) = rx.recv().unwrap();
    assert!(res == TrFailedMsg(format!("{} '{}'", failed_msg, expected)));
}

#[test]
fn test_should_panic_but_succeeds() {
    fn f() {}
    let desc = TestDescAndFn {
        desc: TestDesc {
            name: StaticTestName("whatever"),
            ignore: false,
            should_panic: ShouldPanic::Yes,
            allow_fail: false,
        },
        testfn: DynTestFn(Box::new(f)),
    };
    let (tx, rx) = channel();
    run_test(&TestOpts::new(), false, desc, tx, Concurrent::No);
    let (_, res, _) = rx.recv().unwrap();
    assert!(res == TrFailed);
}

#[test]
fn parse_ignored_flag() {
    let args = vec![
        "progname".to_string(),
        "filter".to_string(),
        "--ignored".to_string(),
    ];
    let opts = parse_opts(&args).unwrap().unwrap();
    assert_eq!(opts.run_ignored, RunIgnored::Only);
}

#[test]
fn parse_include_ignored_flag() {
    let args = vec![
        "progname".to_string(),
        "filter".to_string(),
        "-Zunstable-options".to_string(),
        "--include-ignored".to_string(),
    ];
    let opts = parse_opts(&args).unwrap().unwrap();
    assert_eq!(opts.run_ignored, RunIgnored::Yes);
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
            should_panic: ShouldPanic::Yes,
            allow_fail: false,
        },
        testfn: DynTestFn(Box::new(move || {})),
    });

    let filtered = filter_tests(&opts, tests);

    assert_eq!(filtered.len(), 2);
    assert!(filtered.iter().all(|test| test.desc.should_panic == ShouldPanic::No));
}

#[test]
pub fn exact_filter_match() {
    fn tests() -> Vec<TestDescAndFn> {
        vec!["base", "base::test", "base::test1", "base::test2"]
            .into_iter()
            .map(|name| TestDescAndFn {
                desc: TestDesc {
                    name: StaticTestName(name),
                    ignore: false,
                    should_panic: ShouldPanic::No,
                    allow_fail: false,
                },
                testfn: DynTestFn(Box::new(move || {})),
            })
            .collect()
    }

    let substr = filter_tests(
        &TestOpts {
            filter: Some("base".into()),
            ..TestOpts::new()
        },
        tests(),
    );
    assert_eq!(substr.len(), 4);

    let substr = filter_tests(
        &TestOpts {
            filter: Some("bas".into()),
            ..TestOpts::new()
        },
        tests(),
    );
    assert_eq!(substr.len(), 4);

    let substr = filter_tests(
        &TestOpts {
            filter: Some("::test".into()),
            ..TestOpts::new()
        },
        tests(),
    );
    assert_eq!(substr.len(), 3);

    let substr = filter_tests(
        &TestOpts {
            filter: Some("base::test".into()),
            ..TestOpts::new()
        },
        tests(),
    );
    assert_eq!(substr.len(), 3);

    let exact = filter_tests(
        &TestOpts {
            filter: Some("base".into()),
            filter_exact: true,
            ..TestOpts::new()
        },
        tests(),
    );
    assert_eq!(exact.len(), 1);

    let exact = filter_tests(
        &TestOpts {
            filter: Some("bas".into()),
            filter_exact: true,
            ..TestOpts::new()
        },
        tests(),
    );
    assert_eq!(exact.len(), 0);

    let exact = filter_tests(
        &TestOpts {
            filter: Some("::test".into()),
            filter_exact: true,
            ..TestOpts::new()
        },
        tests(),
    );
    assert_eq!(exact.len(), 0);

    let exact = filter_tests(
        &TestOpts {
            filter: Some("base::test".into()),
            filter_exact: true,
            ..TestOpts::new()
        },
        tests(),
    );
    assert_eq!(exact.len(), 1);
}

#[test]
pub fn sort_tests() {
    let mut opts = TestOpts::new();
    opts.run_tests = true;

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
    let tests = {
        fn testfn() {}
        let mut tests = Vec::new();
        for name in &names {
            let test = TestDescAndFn {
                desc: TestDesc {
                    name: DynTestName((*name).clone()),
                    ignore: false,
                    should_panic: ShouldPanic::No,
                    allow_fail: false,
                },
                testfn: DynTestFn(Box::new(testfn)),
            };
            tests.push(test);
        }
        tests
    };
    let filtered = filter_tests(&opts, tests);

    let expected = vec![
        "isize::test_pow".to_string(),
        "isize::test_to_str".to_string(),
        "sha1::test".to_string(),
        "test::do_not_run_ignored_tests".to_string(),
        "test::filter_for_ignored_option".to_string(),
        "test::first_free_arg_should_be_a_filter".to_string(),
        "test::ignored_tests_result_in_ignored".to_string(),
        "test::parse_ignored_flag".to_string(),
        "test::parse_include_ignored_flag".to_string(),
        "test::run_include_ignored_option".to_string(),
        "test::sort_tests".to_string(),
    ];

    for (a, b) in expected.iter().zip(filtered) {
        assert!(*a == b.desc.name.to_string());
    }
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
    fn f(_: &mut Bencher) {}
    bench::run_once(f);
}

#[test]
pub fn test_bench_once_iter() {
    fn f(b: &mut Bencher) {
        b.iter(|| {})
    }
    bench::run_once(f);
}

#[test]
pub fn test_bench_no_iter() {
    fn f(_: &mut Bencher) {}

    let (tx, rx) = channel();

    let desc = TestDesc {
        name: StaticTestName("f"),
        ignore: false,
        should_panic: ShouldPanic::No,
        allow_fail: false,
    };

    crate::bench::benchmark(desc, tx, true, f);
    rx.recv().unwrap();
}

#[test]
pub fn test_bench_iter() {
    fn f(b: &mut Bencher) {
        b.iter(|| {})
    }

    let (tx, rx) = channel();

    let desc = TestDesc {
        name: StaticTestName("f"),
        ignore: false,
        should_panic: ShouldPanic::No,
        allow_fail: false,
    };

    crate::bench::benchmark(desc, tx, true, f);
    rx.recv().unwrap();
}
