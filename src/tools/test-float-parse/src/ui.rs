//! Progress bars and such.

use std::any::type_name;
use std::fmt;
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

use crate::{Completed, Config, EarlyExit, FinishedAll, TestInfo};

/// Templates for progress bars.
const PB_TEMPLATE: &str = "[{elapsed:3} {percent:3}%] {bar:20.cyan/blue} NAME \
        {human_pos:>8}/{human_len:8} {msg} f {per_sec:14} eta {eta:8}";
const PB_TEMPLATE_FINAL: &str = "[{elapsed:3} {percent:3}%] {bar:20.cyan/blue} NAME \
        {human_pos:>8}/{human_len:8} {msg:.COLOR} {per_sec:18} {elapsed_precise}";

/// Thin abstraction over our usage of a `ProgressBar`.
#[derive(Debug)]
pub struct Progress {
    pb: ProgressBar,
    make_final_style: NoDebug<Box<dyn Fn(&'static str) -> ProgressStyle + Sync>>,
}

impl Progress {
    /// Create a new progress bar within a multiprogress bar.
    pub fn new(test: &TestInfo, all_bars: &mut Vec<ProgressBar>) -> Self {
        let initial_template = PB_TEMPLATE.replace("NAME", &test.short_name_padded);
        let final_template = PB_TEMPLATE_FINAL.replace("NAME", &test.short_name_padded);
        let initial_style =
            ProgressStyle::with_template(&initial_template).unwrap().progress_chars("##-");
        let make_final_style = move |color| {
            ProgressStyle::with_template(&final_template.replace("COLOR", color))
                .unwrap()
                .progress_chars("##-")
        };

        let pb = ProgressBar::new(test.total_tests);
        pb.set_style(initial_style);
        pb.set_length(test.total_tests);
        pb.set_message("0");
        all_bars.push(pb.clone());

        Progress { pb, make_final_style: NoDebug(Box::new(make_final_style)) }
    }

    /// Completed a out of b tests.
    pub fn update(&self, completed: u64, failures: u64) {
        // Infrequently update the progress bar.
        if completed % 5_000 == 0 || failures > 0 {
            self.pb.set_position(completed);
        }

        if failures > 0 {
            self.pb.set_message(format! {"{failures}"});
        }
    }

    /// Finalize the progress bar.
    pub fn complete(&self, c: &Completed, real_total: u64) {
        let f = c.failures;
        let (color, msg, finish_fn): (&str, String, fn(&ProgressBar)) = match &c.result {
            Ok(FinishedAll) if f > 0 => {
                ("red", format!("{f} f (completed with errors)",), ProgressBar::finish)
            }
            Ok(FinishedAll) => {
                ("green", format!("{f} f (completed successfully)",), ProgressBar::finish)
            }
            Err(EarlyExit::Timeout) => ("red", format!("{f} f (timed out)"), ProgressBar::abandon),
            Err(EarlyExit::MaxFailures) => {
                ("red", format!("{f} f (failure limit)"), ProgressBar::abandon)
            }
        };

        self.pb.set_position(real_total);
        self.pb.set_style(self.make_final_style.0(color));
        self.pb.set_message(msg);
        finish_fn(&self.pb);
    }

    /// Print a message to stdout above the current progress bar.
    pub fn println(&self, msg: &str) {
        self.pb.suspend(|| println!("{msg}"));
    }
}

/// Print final messages after all tests are complete.
pub fn finish_all(tests: &[TestInfo], total_elapsed: Duration, cfg: &Config) -> ExitCode {
    println!("\n\nResults:");

    let mut failed_generators = 0;
    let mut stopped_generators = 0;

    for t in tests {
        let Completed { executed, failures, elapsed, warning, result } = t.completed.get().unwrap();

        let stat = if result.is_err() {
            stopped_generators += 1;
            "STOPPED"
        } else if *failures > 0 {
            failed_generators += 1;
            "FAILURE"
        } else {
            "SUCCESS"
        };

        println!(
            "    {stat} for generator '{name}'. {passed}/{executed} passed in {elapsed:?}",
            name = t.name,
            passed = executed - failures,
        );

        if let Some(warning) = warning {
            println!("      warning: {warning}");
        }

        match result {
            Ok(FinishedAll) => (),
            Err(EarlyExit::Timeout) => {
                println!("      exited early; exceeded {:?} timeout", cfg.timeout)
            }
            Err(EarlyExit::MaxFailures) => {
                println!("      exited early; exceeded {:?} max failures", cfg.max_failures)
            }
        }
    }

    println!(
        "{passed}/{} tests succeeded in {total_elapsed:?} ({passed} passed, {} failed, {} stopped)",
        tests.len(),
        failed_generators,
        stopped_generators,
        passed = tests.len() - failed_generators - stopped_generators,
    );

    if failed_generators > 0 || stopped_generators > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// indicatif likes to eat panic messages. This workaround isn't ideal, but it improves things.
/// <https://github.com/console-rs/indicatif/issues/121>.
pub fn set_panic_hook(drop_bars: &[ProgressBar]) {
    let hook = std::panic::take_hook();
    let drop_bars = drop_bars.to_owned();
    std::panic::set_hook(Box::new(move |info| {
        for bar in &drop_bars {
            bar.abandon();
            println!();
            io::stdout().flush().unwrap();
            io::stderr().flush().unwrap();
        }
        hook(info);
    }));
}

/// Allow non-Debug items in a `derive(Debug)` struct.
#[derive(Clone)]
struct NoDebug<T>(T);

impl<T> fmt::Debug for NoDebug<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(type_name::<Self>())
    }
}
