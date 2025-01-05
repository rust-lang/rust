//! Exhaustive tests for `f16` and `f32`, high-iteration for `f64` and `f128`.

use std::fmt;
use std::io::{self, IsTerminal};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};
use libm_test::gen::extensive::{self, ExtensiveInput};
use libm_test::mpfloat::MpOp;
use libm_test::{
    CheckBasis, CheckCtx, CheckOutput, MathOp, TestResult, TupleCall, skip_extensive_test,
};
use libtest_mimic::{Arguments, Trial};
use rayon::prelude::*;

/// Run the extensive test suite.
pub fn run() {
    let mut args = Arguments::from_args();
    // Prevent multiple tests from running in parallel, each test gets parallized internally.
    args.test_threads = Some(1);
    let tests = register_all_tests();

    // With default parallelism, the CPU doesn't saturate. We don't need to be nice to
    // other processes, so do 1.5x to make sure we use all available resources.
    let threads = std::thread::available_parallelism().map(Into::into).unwrap_or(0) * 3 / 2;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();

    libtest_mimic::run(&args, tests).exit();
}

macro_rules! mp_extensive_tests {
    (
        fn_name: $fn_name:ident,
        attrs: [$($attr:meta),*],
        extra: [$push_to:ident],
    ) => {
        $(#[$attr])*
        register_single_test::<libm_test::op::$fn_name::Routine>(&mut $push_to);
    };
}

/// Create a list of tests for consumption by `libtest_mimic`.
fn register_all_tests() -> Vec<Trial> {
    let mut all_tests = Vec::new();

    libm_macros::for_each_function! {
        callback: mp_extensive_tests,
        extra: [all_tests],
        skip: [
            // FIXME: test needed, see
            // https://github.com/rust-lang/libm/pull/311#discussion_r1818273392
            nextafter,
            nextafterf,
        ],
    }

    all_tests
}

/// Add a single test to the list.
fn register_single_test<Op>(all: &mut Vec<Trial>)
where
    Op: MathOp + MpOp,
    Op::RustArgs: ExtensiveInput<Op> + Send,
{
    let test_name = format!("mp_extensive_{}", Op::NAME);
    let ctx = CheckCtx::new(Op::IDENTIFIER, CheckBasis::Mpfr);
    let skip = skip_extensive_test(&ctx);

    let runner = move || {
        if !cfg!(optimizations_enabled) {
            panic!("extensive tests should be run with --release");
        }

        let res = run_single_test::<Op>();
        let e = match res {
            Ok(()) => return Ok(()),
            Err(e) => e,
        };

        // Format with the `Debug` implementation so we get the error cause chain, and print it
        // here so we see the result immediately (rather than waiting for all tests to conclude).
        let e = format!("{e:?}");
        eprintln!("failure testing {}:{e}\n", Op::IDENTIFIER);

        Err(e.into())
    };

    all.push(Trial::test(test_name, runner).with_ignored_flag(skip));
}

/// Test runner for a signle routine.
fn run_single_test<Op>() -> TestResult
where
    Op: MathOp + MpOp,
    Op::RustArgs: ExtensiveInput<Op> + Send,
{
    // Small delay before printing anything so other output from the runner has a chance to flush.
    std::thread::sleep(Duration::from_millis(500));
    eprintln!();

    let completed = AtomicU64::new(0);
    let ctx = CheckCtx::new(Op::IDENTIFIER, CheckBasis::Mpfr);
    let (ref mut cases, total) = extensive::get_test_cases::<Op>(&ctx);
    let pb = Progress::new(Op::NAME, total);

    let test_single_chunk = |mp_vals: &mut Op::MpTy, input_vec: Vec<Op::RustArgs>| -> TestResult {
        for input in input_vec {
            // Test the input.
            let mp_res = Op::run(mp_vals, input);
            let crate_res = input.call(Op::ROUTINE);
            crate_res.validate(mp_res, input, &ctx)?;

            let completed = completed.fetch_add(1, Ordering::Relaxed) + 1;
            pb.update(completed, input);
        }

        Ok(())
    };

    // Chunk the cases so Rayon doesn't switch threads between each iterator item. 50k seems near
    // a performance sweet spot. Ideally we would reuse these allocations rather than discarding,
    // but that is difficult with Rayon's API.
    let chunk_size = 50_000;
    let chunks = std::iter::from_fn(move || {
        let mut v = Vec::with_capacity(chunk_size);
        v.extend(cases.take(chunk_size));
        (!v.is_empty()).then_some(v)
    });

    // Run the actual tests
    let res = chunks.par_bridge().try_for_each_init(Op::new_mp, test_single_chunk);

    let real_total = completed.load(Ordering::Relaxed);
    pb.complete(real_total);

    if res.is_ok() && real_total != total {
        // Provide a warning if our estimate needs to be updated.
        panic!("total run {real_total} does not match expected {total}");
    }

    res
}

/// Wrapper around a `ProgressBar` that handles styles and non-TTY messages.
struct Progress {
    pb: ProgressBar,
    name_padded: String,
    final_style: ProgressStyle,
    is_tty: bool,
}

impl Progress {
    const PB_TEMPLATE: &str = "[{elapsed:3} {percent:3}%] {bar:20.cyan/blue} NAME \
        {human_pos:>13}/{human_len:13} {per_sec:18} eta {eta:8} {msg}";
    const PB_TEMPLATE_FINAL: &str = "[{elapsed:3} {percent:3}%] {bar:20.cyan/blue} NAME \
        {human_pos:>13}/{human_len:13} {per_sec:18} done in {elapsed_precise}";

    fn new(name: &str, total: u64) -> Self {
        eprintln!("starting extensive tests for `{name}`");
        let name_padded = format!("{name:9}");
        let is_tty = io::stderr().is_terminal();

        let initial_style =
            ProgressStyle::with_template(&Self::PB_TEMPLATE.replace("NAME", &name_padded))
                .unwrap()
                .progress_chars("##-");

        let final_style =
            ProgressStyle::with_template(&Self::PB_TEMPLATE_FINAL.replace("NAME", &name_padded))
                .unwrap()
                .progress_chars("##-");

        let pb = ProgressBar::new(total);
        pb.set_style(initial_style);

        Self { pb, final_style, name_padded, is_tty }
    }

    fn update(&self, completed: u64, input: impl fmt::Debug) {
        // Infrequently update the progress bar.
        if completed % 20_000 == 0 {
            self.pb.set_position(completed);
        }

        if completed % 500_000 == 0 {
            self.pb.set_message(format!("input: {input:<24?}"));
        }

        if !self.is_tty && completed % 5_000_000 == 0 {
            let len = self.pb.length().unwrap_or_default();
            eprintln!(
                "[{elapsed:3?}s {percent:3.0}%] {name} \
                {human_pos:>10}/{human_len:<10} {per_sec:14.2}/s eta {eta:4}s {input:<24?}",
                elapsed = self.pb.elapsed().as_secs(),
                percent = completed as f32 * 100.0 / len as f32,
                name = self.name_padded,
                human_pos = completed,
                human_len = len,
                per_sec = self.pb.per_sec(),
                eta = self.pb.eta().as_secs()
            );
        }
    }

    fn complete(self, real_total: u64) {
        self.pb.set_style(self.final_style);
        self.pb.set_position(real_total);
        self.pb.abandon();

        if !self.is_tty {
            let len = self.pb.length().unwrap_or_default();
            eprintln!(
                "[{elapsed:3}s {percent:3.0}%] {name} \
                {human_pos:>10}/{human_len:<10} {per_sec:14.2}/s done in {elapsed_precise}",
                elapsed = self.pb.elapsed().as_secs(),
                percent = real_total as f32 * 100.0 / len as f32,
                name = self.name_padded,
                human_pos = real_total,
                human_len = len,
                per_sec = self.pb.per_sec(),
                elapsed_precise = self.pb.elapsed().as_secs(),
            );
        }

        eprintln!();
    }
}
