//! Module converting command-line arguments into test configuration.

use std::env;
use std::path::PathBuf;

use super::helpers::isatty;
use super::options::{ColorConfig, Options, OutputFormat, RunIgnored};
use super::time::TestTimeOptions;

#[derive(Debug)]
pub struct TestOpts {
    pub list: bool,
    pub filters: Vec<String>,
    pub filter_exact: bool,
    pub force_run_in_process: bool,
    pub exclude_should_panic: bool,
    pub run_ignored: RunIgnored,
    pub run_tests: bool,
    pub bench_benchmarks: bool,
    pub logfile: Option<PathBuf>,
    pub nocapture: bool,
    pub color: ColorConfig,
    pub format: OutputFormat,
    pub shuffle: bool,
    pub shuffle_seed: Option<u64>,
    pub test_threads: Option<usize>,
    pub skip: Vec<String>,
    pub time_options: Option<TestTimeOptions>,
    pub options: Options,
}

impl TestOpts {
    pub fn use_color(&self) -> bool {
        match self.color {
            ColorConfig::AutoColor => !self.nocapture && isatty::stdout_isatty(),
            ColorConfig::AlwaysColor => true,
            ColorConfig::NeverColor => false,
        }
    }
}

/// Result of parsing the options.
pub type OptRes = Result<TestOpts, String>;
/// Result of parsing the option part.
type OptPartRes<T> = Result<T, String>;

fn optgroups() -> getopts::Options {
    let mut opts = getopts::Options::new();
    opts.optflag("", "include-ignored", "Run ignored and not ignored tests")
        .optflag("", "ignored", "Run only ignored tests")
        .optflag("", "force-run-in-process", "Forces tests to run in-process when panic=abort")
        .optflag("", "exclude-should-panic", "Excludes tests marked as should_panic")
        .optflag("", "test", "Run tests and not benchmarks")
        .optflag("", "bench", "Run benchmarks instead of tests")
        .optflag("", "list", "List all tests and benchmarks")
        .optflag("h", "help", "Display this message")
        .optopt("", "logfile", "Write logs to the specified file", "PATH")
        .optflag(
            "",
            "nocapture",
            "don't capture stdout/stderr of each \
             task, allow printing directly",
        )
        .optopt(
            "",
            "test-threads",
            "Number of threads used for running tests \
             in parallel",
            "n_threads",
        )
        .optmulti(
            "",
            "skip",
            "Skip tests whose names contain FILTER (this flag can \
             be used multiple times)",
            "FILTER",
        )
        .optflag(
            "q",
            "quiet",
            "Display one character per test instead of one line. \
             Alias to --format=terse",
        )
        .optflag("", "exact", "Exactly match filters rather than by substring")
        .optopt(
            "",
            "color",
            "Configure coloring of output:
            auto   = colorize if stdout is a tty and tests are run on serially (default);
            always = always colorize output;
            never  = never colorize output;",
            "auto|always|never",
        )
        .optopt(
            "",
            "format",
            "Configure formatting of output:
            pretty = Print verbose output;
            terse  = Display one character per test;
            json   = Output a json document;
            junit  = Output a JUnit document",
            "pretty|terse|json|junit",
        )
        .optflag("", "show-output", "Show captured stdout of successful tests")
        .optopt(
            "Z",
            "",
            "Enable nightly-only flags:
            unstable-options = Allow use of experimental features",
            "unstable-options",
        )
        .optflag(
            "",
            "report-time",
            "Show execution time of each test.

            Threshold values for colorized output can be configured via
            `RUST_TEST_TIME_UNIT`, `RUST_TEST_TIME_INTEGRATION` and
            `RUST_TEST_TIME_DOCTEST` environment variables.

            Expected format of environment variable is `VARIABLE=WARN_TIME,CRITICAL_TIME`.
            Durations must be specified in milliseconds, e.g. `500,2000` means that the warn time
            is 0.5 seconds, and the critical time is 2 seconds.

            Not available for --format=terse",
        )
        .optflag(
            "",
            "ensure-time",
            "Treat excess of the test execution time limit as error.

            Threshold values for this option can be configured via
            `RUST_TEST_TIME_UNIT`, `RUST_TEST_TIME_INTEGRATION` and
            `RUST_TEST_TIME_DOCTEST` environment variables.

            Expected format of environment variable is `VARIABLE=WARN_TIME,CRITICAL_TIME`.

            `CRITICAL_TIME` here means the limit that should not be exceeded by test.
            ",
        )
        .optflag("", "shuffle", "Run tests in random order")
        .optopt(
            "",
            "shuffle-seed",
            "Run tests in random order; seed the random number generator with SEED",
            "SEED",
        );
    opts
}

fn usage(binary: &str, options: &getopts::Options) {
    let message = format!("Usage: {} [OPTIONS] [FILTERS...]", binary);
    println!(
        r#"{usage}

The FILTER string is tested against the name of all tests, and only those
tests whose names contain the filter are run. Multiple filter strings may
be passed, which will run all tests matching any of the filters.

By default, all tests are run in parallel. This can be altered with the
--test-threads flag or the RUST_TEST_THREADS environment variable when running
tests (set it to 1).

By default, the tests are run in alphabetical order. Use --shuffle or set
RUST_TEST_SHUFFLE to run the tests in random order. Pass the generated
"shuffle seed" to --shuffle-seed (or set RUST_TEST_SHUFFLE_SEED) to run the
tests in the same order again. Note that --shuffle and --shuffle-seed do not
affect whether the tests are run in parallel.

All tests have their standard output and standard error captured by default.
This can be overridden with the --nocapture flag or setting RUST_TEST_NOCAPTURE
environment variable to a value other than "0". Logging is not captured by default.

Test Attributes:

    `#[test]`        - Indicates a function is a test to be run. This function
                       takes no arguments.
    `#[bench]`       - Indicates a function is a benchmark to be run. This
                       function takes one argument (test::Bencher).
    `#[should_panic]` - This function (also labeled with `#[test]`) will only pass if
                        the code causes a panic (an assertion failure or panic!)
                        A message may be provided, which the failure string must
                        contain: #[should_panic(expected = "foo")].
    `#[ignore]`       - When applied to a function which is already attributed as a
                        test, then the test runner will ignore these tests during
                        normal test runs. Running with --ignored or --include-ignored will run
                        these tests."#,
        usage = options.usage(&message)
    );
}

/// Parses command line arguments into test options.
/// Returns `None` if help was requested (since we only show help message and don't run tests),
/// returns `Some(Err(..))` if provided arguments are incorrect,
/// otherwise creates a `TestOpts` object and returns it.
pub fn parse_opts(args: &[String]) -> Option<OptRes> {
    // Parse matches.
    let opts = optgroups();
    let args = args.get(1..).unwrap_or(args);
    let matches = match opts.parse(args) {
        Ok(m) => m,
        Err(f) => return Some(Err(f.to_string())),
    };

    // Check if help was requested.
    if matches.opt_present("h") {
        // Show help and do nothing more.
        usage(&args[0], &opts);
        return None;
    }

    // Actually parse the opts.
    let opts_result = parse_opts_impl(matches);

    Some(opts_result)
}

// Gets the option value and checks if unstable features are enabled.
macro_rules! unstable_optflag {
    ($matches:ident, $allow_unstable:ident, $option_name:literal) => {{
        let opt = $matches.opt_present($option_name);
        if !$allow_unstable && opt {
            return Err(format!(
                "The \"{}\" flag is only accepted on the nightly compiler with -Z unstable-options",
                $option_name
            ));
        }

        opt
    }};
}

// Gets the option value and checks if unstable features are enabled.
macro_rules! unstable_optopt {
    ($matches:ident, $allow_unstable:ident, $option_name:literal) => {{
        let opt = $matches.opt_str($option_name);
        if !$allow_unstable && opt.is_some() {
            return Err(format!(
                "The \"{}\" option is only accepted on the nightly compiler with -Z unstable-options",
                $option_name
            ));
        }

        opt
    }};
}

// Implementation of `parse_opts` that doesn't care about help message
// and returns a `Result`.
fn parse_opts_impl(matches: getopts::Matches) -> OptRes {
    let allow_unstable = get_allow_unstable(&matches)?;

    // Unstable flags
    let force_run_in_process = unstable_optflag!(matches, allow_unstable, "force-run-in-process");
    let exclude_should_panic = unstable_optflag!(matches, allow_unstable, "exclude-should-panic");
    let time_options = get_time_options(&matches, allow_unstable)?;
    let shuffle = get_shuffle(&matches, allow_unstable)?;
    let shuffle_seed = get_shuffle_seed(&matches, allow_unstable)?;

    let include_ignored = matches.opt_present("include-ignored");
    let quiet = matches.opt_present("quiet");
    let exact = matches.opt_present("exact");
    let list = matches.opt_present("list");
    let skip = matches.opt_strs("skip");

    let bench_benchmarks = matches.opt_present("bench");
    let run_tests = !bench_benchmarks || matches.opt_present("test");

    let logfile = get_log_file(&matches)?;
    let run_ignored = get_run_ignored(&matches, include_ignored)?;
    let filters = matches.free.clone();
    let nocapture = get_nocapture(&matches)?;
    let test_threads = get_test_threads(&matches)?;
    let color = get_color_config(&matches)?;
    let format = get_format(&matches, quiet, allow_unstable)?;

    let options = Options::new().display_output(matches.opt_present("show-output"));

    let test_opts = TestOpts {
        list,
        filters,
        filter_exact: exact,
        force_run_in_process,
        exclude_should_panic,
        run_ignored,
        run_tests,
        bench_benchmarks,
        logfile,
        nocapture,
        color,
        format,
        shuffle,
        shuffle_seed,
        test_threads,
        skip,
        time_options,
        options,
    };

    Ok(test_opts)
}

// FIXME: Copied from librustc_ast until linkage errors are resolved. Issue #47566
fn is_nightly() -> bool {
    // Whether this is a feature-staged build, i.e., on the beta or stable channel
    let disable_unstable_features = option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_some();
    // Whether we should enable unstable features for bootstrapping
    let bootstrap = env::var("RUSTC_BOOTSTRAP").is_ok();

    bootstrap || !disable_unstable_features
}

// Gets the CLI options associated with `report-time` feature.
fn get_time_options(
    matches: &getopts::Matches,
    allow_unstable: bool,
) -> OptPartRes<Option<TestTimeOptions>> {
    let report_time = unstable_optflag!(matches, allow_unstable, "report-time");
    let ensure_test_time = unstable_optflag!(matches, allow_unstable, "ensure-time");

    // If `ensure-test-time` option is provided, time output is enforced,
    // so user won't be confused if any of tests will silently fail.
    let options = if report_time || ensure_test_time {
        Some(TestTimeOptions::new_from_env(ensure_test_time))
    } else {
        None
    };

    Ok(options)
}

fn get_shuffle(matches: &getopts::Matches, allow_unstable: bool) -> OptPartRes<bool> {
    let mut shuffle = unstable_optflag!(matches, allow_unstable, "shuffle");
    if !shuffle && allow_unstable {
        shuffle = match env::var("RUST_TEST_SHUFFLE") {
            Ok(val) => &val != "0",
            Err(_) => false,
        };
    }

    Ok(shuffle)
}

fn get_shuffle_seed(matches: &getopts::Matches, allow_unstable: bool) -> OptPartRes<Option<u64>> {
    let mut shuffle_seed = match unstable_optopt!(matches, allow_unstable, "shuffle-seed") {
        Some(n_str) => match n_str.parse::<u64>() {
            Ok(n) => Some(n),
            Err(e) => {
                return Err(format!(
                    "argument for --shuffle-seed must be a number \
                     (error: {})",
                    e
                ));
            }
        },
        None => None,
    };

    if shuffle_seed.is_none() && allow_unstable {
        shuffle_seed = match env::var("RUST_TEST_SHUFFLE_SEED") {
            Ok(val) => match val.parse::<u64>() {
                Ok(n) => Some(n),
                Err(_) => panic!("RUST_TEST_SHUFFLE_SEED is `{}`, should be a number.", val),
            },
            Err(_) => None,
        };
    }

    Ok(shuffle_seed)
}

fn get_test_threads(matches: &getopts::Matches) -> OptPartRes<Option<usize>> {
    let test_threads = match matches.opt_str("test-threads") {
        Some(n_str) => match n_str.parse::<usize>() {
            Ok(0) => return Err("argument for --test-threads must not be 0".to_string()),
            Ok(n) => Some(n),
            Err(e) => {
                return Err(format!(
                    "argument for --test-threads must be a number > 0 \
                     (error: {})",
                    e
                ));
            }
        },
        None => None,
    };

    Ok(test_threads)
}

fn get_format(
    matches: &getopts::Matches,
    quiet: bool,
    allow_unstable: bool,
) -> OptPartRes<OutputFormat> {
    let format = match matches.opt_str("format").as_deref() {
        None if quiet => OutputFormat::Terse,
        Some("pretty") | None => OutputFormat::Pretty,
        Some("terse") => OutputFormat::Terse,
        Some("json") => {
            if !allow_unstable {
                return Err("The \"json\" format is only accepted on the nightly compiler".into());
            }
            OutputFormat::Json
        }
        Some("junit") => {
            if !allow_unstable {
                return Err("The \"junit\" format is only accepted on the nightly compiler".into());
            }
            OutputFormat::Junit
        }
        Some(v) => {
            return Err(format!(
                "argument for --format must be pretty, terse, json or junit (was \
                 {})",
                v
            ));
        }
    };

    Ok(format)
}

fn get_color_config(matches: &getopts::Matches) -> OptPartRes<ColorConfig> {
    let color = match matches.opt_str("color").as_deref() {
        Some("auto") | None => ColorConfig::AutoColor,
        Some("always") => ColorConfig::AlwaysColor,
        Some("never") => ColorConfig::NeverColor,

        Some(v) => {
            return Err(format!(
                "argument for --color must be auto, always, or never (was \
                 {})",
                v
            ));
        }
    };

    Ok(color)
}

fn get_nocapture(matches: &getopts::Matches) -> OptPartRes<bool> {
    let mut nocapture = matches.opt_present("nocapture");
    if !nocapture {
        nocapture = match env::var("RUST_TEST_NOCAPTURE") {
            Ok(val) => &val != "0",
            Err(_) => false,
        };
    }

    Ok(nocapture)
}

fn get_run_ignored(matches: &getopts::Matches, include_ignored: bool) -> OptPartRes<RunIgnored> {
    let run_ignored = match (include_ignored, matches.opt_present("ignored")) {
        (true, true) => {
            return Err("the options --include-ignored and --ignored are mutually exclusive".into());
        }
        (true, false) => RunIgnored::Yes,
        (false, true) => RunIgnored::Only,
        (false, false) => RunIgnored::No,
    };

    Ok(run_ignored)
}

fn get_allow_unstable(matches: &getopts::Matches) -> OptPartRes<bool> {
    let mut allow_unstable = false;

    if let Some(opt) = matches.opt_str("Z") {
        if !is_nightly() {
            return Err("the option `Z` is only accepted on the nightly compiler".into());
        }

        match &*opt {
            "unstable-options" => {
                allow_unstable = true;
            }
            _ => {
                return Err("Unrecognized option to `Z`".into());
            }
        }
    };

    Ok(allow_unstable)
}

fn get_log_file(matches: &getopts::Matches) -> OptPartRes<Option<PathBuf>> {
    let logfile = matches.opt_str("logfile").map(|s| PathBuf::from(&s));

    Ok(logfile)
}
