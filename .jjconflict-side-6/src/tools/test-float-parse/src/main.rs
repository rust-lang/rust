use std::process::ExitCode;
use std::time::Duration;

use test_float_parse as tfp;

static HELP: &str = r#"Usage:

  ./test-float-parse [--timeout x] [--exclude x] [--max-failures x] [INCLUDE ...]
  ./test-float-parse [--fuzz-count x] [INCLUDE ...]
  ./test-float-parse [--skip-huge] [INCLUDE ...]
  ./test-float-parse --list

Args:

  INCLUDE                  Include only tests with names containing these
                           strings. If this argument is not specified, all tests
                           are run.
  --timeout N              Exit after this amount of time (in seconds).
  --exclude FILTER         Skip tests containing this string. May be specified
                           more than once.
  --list                   List available tests.
  --max-failures N         Limit to N failures per test. Defaults to 20. Pass
                           "--max-failures none" to remove this limit.
  --fuzz-count N           Run the fuzzer with N iterations. Only has an effect
                           if fuzz tests are enabled. Pass `--fuzz-count none`
                           to remove this limit.
  --skip-huge              Skip tests that run for a long time.
  --all                    Reset previous `--exclude`, `--skip-huge`, and
                           `INCLUDE` arguments (useful for running all tests
                           via `./x`).
"#;

enum ArgMode {
    Any,
    Timeout,
    Exclude,
    FuzzCount,
    MaxFailures,
}

fn main() -> ExitCode {
    if cfg!(debug_assertions) {
        println!(
            "WARNING: running in debug mode. Release mode is recommended to reduce test duration."
        );
        std::thread::sleep(Duration::from_secs(2));
    }

    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{HELP}");
        return ExitCode::SUCCESS;
    }

    if args.iter().any(|arg| arg == "--list") {
        let tests = tfp::register_tests(&tfp::Config::default());
        println!("Available tests:");
        for t in tests {
            println!("{}", t.name);
        }

        return ExitCode::SUCCESS;
    }

    let (cfg, include, exclude) = parse_args(args);

    tfp::run(cfg, &include, &exclude)
}

/// Simple command argument parser
fn parse_args(args: Vec<String>) -> (tfp::Config, Vec<String>, Vec<String>) {
    let mut cfg = tfp::Config::default();

    let mut mode = ArgMode::Any;
    let mut include = Vec::new();
    let mut exclude = Vec::new();

    for arg in args {
        mode = match mode {
            ArgMode::Any if arg == "--timeout" => ArgMode::Timeout,
            ArgMode::Any if arg == "--exclude" => ArgMode::Exclude,
            ArgMode::Any if arg == "--max-failures" => ArgMode::MaxFailures,
            ArgMode::Any if arg == "--fuzz-count" => ArgMode::FuzzCount,
            ArgMode::Any if arg == "--skip-huge" => {
                cfg.skip_huge = true;
                ArgMode::Any
            }
            ArgMode::Any if arg == "--all" => {
                cfg.skip_huge = false;
                include.clear();
                exclude.clear();
                ArgMode::Any
            }
            ArgMode::Any if arg.starts_with('-') => {
                panic!("Unknown argument {arg}. Usage:\n{HELP}")
            }
            ArgMode::Any => {
                include.push(arg);
                ArgMode::Any
            }
            ArgMode::Timeout => {
                cfg.timeout = Duration::from_secs(arg.parse().unwrap());
                ArgMode::Any
            }
            ArgMode::MaxFailures => {
                if arg == "none" {
                    cfg.disable_max_failures = true;
                } else {
                    cfg.max_failures = arg.parse().unwrap();
                }
                ArgMode::Any
            }
            ArgMode::FuzzCount => {
                if arg == "none" {
                    cfg.fuzz_count = Some(u64::MAX);
                } else {
                    cfg.fuzz_count = Some(arg.parse().unwrap());
                }
                ArgMode::Any
            }
            ArgMode::Exclude => {
                exclude.push(arg);
                ArgMode::Any
            }
        }
    }

    (cfg, include, exclude)
}
