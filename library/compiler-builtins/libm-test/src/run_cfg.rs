//! Configuration for how tests get run.

use std::ops::RangeInclusive;
use std::sync::LazyLock;
use std::{env, str};

use crate::generate::random::{SEED, SEED_ENV};
use crate::{BaseName, FloatTy, Identifier, test_log};

/// The environment variable indicating which extensive tests should be run.
pub const EXTENSIVE_ENV: &str = "LIBM_EXTENSIVE_TESTS";

/// Specify the number of iterations via this environment variable, rather than using the default.
pub const EXTENSIVE_ITER_ENV: &str = "LIBM_EXTENSIVE_ITERATIONS";

/// The override value, if set by the above environment.
static EXTENSIVE_ITER_OVERRIDE: LazyLock<Option<u64>> = LazyLock::new(|| {
    env::var(EXTENSIVE_ITER_ENV)
        .map(|v| v.parse().expect("failed to parse iteration count"))
        .ok()
});

/// Specific tests that need to have a reduced amount of iterations to complete in a reasonable
/// amount of time.
///
/// Contains the itentifier+generator combo to match on, plus the factor to reduce by.
const EXTEMELY_SLOW_TESTS: &[(Identifier, GeneratorKind, u64)] = &[
    (Identifier::Fmodf128, GeneratorKind::QuickSpaced, 50),
    (Identifier::Fmodf128, GeneratorKind::Extensive, 50),
];

/// Maximum number of iterations to run for a single routine.
///
/// The default value of one greater than `u32::MAX` allows testing single-argument `f32` routines
/// and single- or double-argument `f16` routines exhaustively. `f64` and `f128` can't feasibly
/// be tested exhaustively; however, [`EXTENSIVE_ITER_ENV`] can be set to run tests for multiple
/// hours.
pub fn extensive_max_iterations() -> u64 {
    let default = 1 << 32; // default value
    EXTENSIVE_ITER_OVERRIDE.unwrap_or(default)
}

/// Context passed to [`CheckOutput`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckCtx {
    /// Allowed ULP deviation
    pub ulp: u32,
    pub fn_ident: Identifier,
    pub base_name: BaseName,
    /// Function name.
    pub fn_name: &'static str,
    /// Return the unsuffixed version of the function name.
    pub base_name_str: &'static str,
    /// Source of truth for tests.
    pub basis: CheckBasis,
    pub gen_kind: GeneratorKind,
    /// If specified, this value will override the value returned by [`iteration_count`].
    pub override_iterations: Option<u64>,
}

impl CheckCtx {
    /// Create a new check context, using the default ULP for the function.
    pub fn new(fn_ident: Identifier, basis: CheckBasis, gen_kind: GeneratorKind) -> Self {
        let mut ret = Self {
            ulp: 0,
            fn_ident,
            fn_name: fn_ident.as_str(),
            base_name: fn_ident.base_name(),
            base_name_str: fn_ident.base_name().as_str(),
            basis,
            gen_kind,
            override_iterations: None,
        };
        ret.ulp = crate::default_ulp(&ret);
        ret
    }

    /// The number of input arguments for this function.
    pub fn input_count(&self) -> usize {
        self.fn_ident.math_op().rust_sig.args.len()
    }

    pub fn override_iterations(&mut self, count: u64) {
        self.override_iterations = Some(count)
    }
}

/// Possible items to test against
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckBasis {
    /// Check against Musl's math sources.
    Musl,
    /// Check against infinite precision (MPFR).
    Mpfr,
    /// Benchmarks or other times when this is not relevant.
    None,
}

/// The different kinds of generators that provide test input, which account for input pattern
/// and quantity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneratorKind {
    EdgeCases,
    Extensive,
    QuickSpaced,
    Random,
    List,
}

/// A list of all functions that should get extensive tests.
///
/// This also supports the special test name `all` to run all tests, as well as `all_f16`,
/// `all_f32`, `all_f64`, and `all_f128` to run all tests for a specific float type.
static EXTENSIVE: LazyLock<Vec<Identifier>> = LazyLock::new(|| {
    let var = env::var(EXTENSIVE_ENV).unwrap_or_default();
    let list = var.split(",").filter(|s| !s.is_empty()).collect::<Vec<_>>();
    let mut ret = Vec::new();

    let append_ty_ops = |ret: &mut Vec<_>, fty: FloatTy| {
        let iter = Identifier::ALL
            .iter()
            .filter(move |id| id.math_op().float_ty == fty)
            .copied();
        ret.extend(iter);
    };

    for item in list {
        match item {
            "all" => ret = Identifier::ALL.to_owned(),
            "all_f16" => append_ty_ops(&mut ret, FloatTy::F16),
            "all_f32" => append_ty_ops(&mut ret, FloatTy::F32),
            "all_f64" => append_ty_ops(&mut ret, FloatTy::F64),
            "all_f128" => append_ty_ops(&mut ret, FloatTy::F128),
            s => {
                let id = Identifier::from_str(s)
                    .unwrap_or_else(|| panic!("unrecognized test name `{s}`"));
                ret.push(id);
            }
        }
    }

    ret
});

/// Information about the function to be tested.
#[derive(Debug)]
struct TestEnv {
    /// Tests should be reduced because the platform is slow. E.g. 32-bit or emulated.
    slow_platform: bool,
    /// The float cannot be tested exhaustively, `f64` or `f128`.
    large_float_ty: bool,
    /// Env indicates that an extensive test should be run.
    should_run_extensive: bool,
    /// Multiprecision tests will be run.
    mp_tests_enabled: bool,
    /// The number of inputs to the function.
    input_count: usize,
}

impl TestEnv {
    fn from_env(ctx: &CheckCtx) -> Self {
        let id = ctx.fn_ident;
        let op = id.math_op();

        let will_run_mp = cfg!(feature = "build-mpfr");
        let large_float_ty = match op.float_ty {
            FloatTy::F16 | FloatTy::F32 => false,
            FloatTy::F64 | FloatTy::F128 => true,
        };

        let will_run_extensive = EXTENSIVE.contains(&id);

        let input_count = op.rust_sig.args.len();

        Self {
            slow_platform: slow_platform(),
            large_float_ty,
            should_run_extensive: will_run_extensive,
            mp_tests_enabled: will_run_mp,
            input_count,
        }
    }
}

/// Tests are pretty slow on non-64-bit targets, x86 MacOS, and targets that run in QEMU. Start
/// with a reduced number on these platforms.
fn slow_platform() -> bool {
    let slow_on_ci = crate::emulated()
        || usize::BITS < 64
        || cfg!(all(target_arch = "x86_64", target_vendor = "apple"));

    // If not running in CI, there is no need to reduce iteration count.
    slow_on_ci && crate::ci()
}

/// The number of iterations to run for a given test.
pub fn iteration_count(ctx: &CheckCtx, argnum: usize) -> u64 {
    let t_env = TestEnv::from_env(ctx);

    // Ideally run 5M tests
    let mut domain_iter_count: u64 = 4_000_000;

    // Start with a reduced number of tests on slow platforms.
    if t_env.slow_platform {
        domain_iter_count = 100_000;
    }

    // If we will be running tests against MPFR, we don't need to test as much against musl.
    // However, there are some platforms where we have to test against musl since MPFR can't be
    // built.
    if t_env.mp_tests_enabled && ctx.basis == CheckBasis::Musl {
        domain_iter_count /= 100;
    }

    // Run fewer random tests than domain tests.
    let random_iter_count = domain_iter_count / 100;

    let mut total_iterations = match ctx.gen_kind {
        GeneratorKind::QuickSpaced => domain_iter_count,
        GeneratorKind::Random => random_iter_count,
        GeneratorKind::Extensive => extensive_max_iterations(),
        GeneratorKind::EdgeCases | GeneratorKind::List => {
            unimplemented!("shoudn't need `iteration_count` for {:?}", ctx.gen_kind)
        }
    };

    // Larger float types get more iterations.
    if t_env.large_float_ty && ctx.gen_kind != GeneratorKind::Extensive {
        if ctx.gen_kind == GeneratorKind::Extensive {
            // Extensive already has a pretty high test count.
            total_iterations *= 2;
        } else {
            total_iterations *= 4;
        }
    }

    // Functions with more arguments get more iterations.
    let arg_multiplier = 1 << (t_env.input_count - 1);
    total_iterations *= arg_multiplier;

    // FMA has a huge domain but is reasonably fast to run, so increase another 1.5x.
    if ctx.base_name == BaseName::Fma {
        total_iterations = 3 * total_iterations / 2;
    }

    // Some tests are significantly slower than others and need to be further reduced.
    if let Some((_id, _gen, scale)) = EXTEMELY_SLOW_TESTS
        .iter()
        .find(|(id, generator, _scale)| *id == ctx.fn_ident && *generator == ctx.gen_kind)
    {
        // However, do not override if the extensive iteration count has been manually set.
        if !(ctx.gen_kind == GeneratorKind::Extensive && EXTENSIVE_ITER_OVERRIDE.is_some()) {
            total_iterations /= scale;
        }
    }

    if cfg!(optimizations_enabled) {
        // Always run at least 10,000 tests.
        total_iterations = total_iterations.max(10_000);
    } else {
        // Without optimizations, just run a quick check regardless of other parameters.
        total_iterations = 800;
    }

    let mut overridden = false;
    if let Some(count) = ctx.override_iterations {
        total_iterations = count;
        overridden = true;
    }

    // Adjust for the number of inputs
    let ntests = match t_env.input_count {
        1 => total_iterations,
        2 => (total_iterations as f64).sqrt().ceil() as u64,
        3 => (total_iterations as f64).cbrt().ceil() as u64,
        _ => panic!("test has more than three arguments"),
    };

    let total = ntests.pow(t_env.input_count.try_into().unwrap());

    let seed_msg = match ctx.gen_kind {
        GeneratorKind::QuickSpaced | GeneratorKind::Extensive => String::new(),
        GeneratorKind::Random => {
            format!(
                " using `{SEED_ENV}={}`",
                str::from_utf8(SEED.as_slice()).unwrap()
            )
        }
        GeneratorKind::EdgeCases | GeneratorKind::List => unimplemented!(),
    };

    test_log(&format!(
        "{gen_kind:?} {basis:?} {fn_ident} arg {arg}/{args}: {ntests} iterations \
         ({total} total){seed_msg}{omsg}",
        gen_kind = ctx.gen_kind,
        basis = ctx.basis,
        fn_ident = ctx.fn_ident,
        arg = argnum + 1,
        args = t_env.input_count,
        omsg = if overridden { " (overridden)" } else { "" }
    ));

    ntests
}

/// Some tests require that an integer be kept within reasonable limits; generate that here.
pub fn int_range(ctx: &CheckCtx, argnum: usize) -> RangeInclusive<i32> {
    let t_env = TestEnv::from_env(ctx);

    if !matches!(ctx.base_name, BaseName::Jn | BaseName::Yn) {
        return i32::MIN..=i32::MAX;
    }

    assert_eq!(
        argnum, 0,
        "For `jn`/`yn`, only the first argument takes an integer"
    );

    // The integer argument to `jn` is an iteration count. Limit this to ensure tests can be
    // completed in a reasonable amount of time.
    let non_extensive_range = if t_env.slow_platform || !cfg!(optimizations_enabled) {
        (-0xf)..=0xff
    } else {
        (-0xff)..=0xffff
    };

    let extensive_range = (-0xfff)..=0xfffff;

    match ctx.gen_kind {
        GeneratorKind::Extensive => extensive_range,
        GeneratorKind::QuickSpaced | GeneratorKind::Random => non_extensive_range,
        GeneratorKind::EdgeCases => extensive_range,
        GeneratorKind::List => unimplemented!("shoudn't need range for {:?}", ctx.gen_kind),
    }
}

/// For domain tests, limit how many asymptotes or specified check points we test.
pub fn check_point_count(ctx: &CheckCtx) -> usize {
    assert_eq!(
        ctx.gen_kind,
        GeneratorKind::EdgeCases,
        "check_point_count is intended for edge case tests"
    );
    let t_env = TestEnv::from_env(ctx);
    if t_env.slow_platform || !cfg!(optimizations_enabled) {
        4
    } else {
        10
    }
}

/// When validating points of interest (e.g. asymptotes, inflection points, extremes), also check
/// this many surrounding values.
pub fn check_near_count(ctx: &CheckCtx) -> u64 {
    assert_eq!(
        ctx.gen_kind,
        GeneratorKind::EdgeCases,
        "check_near_count is intended for edge case tests"
    );
    if cfg!(optimizations_enabled) {
        // Taper based on the number of inputs.
        match ctx.input_count() {
            1 | 2 => 100,
            3 => 50,
            x => panic!("unexpected argument count {x}"),
        }
    } else {
        8
    }
}

/// Check whether extensive actions should be run or skipped.
pub fn skip_extensive_test(ctx: &CheckCtx) -> bool {
    let t_env = TestEnv::from_env(ctx);
    !t_env.should_run_extensive
}

/// The number of iterations to run for `u256` fuzz tests.
pub fn bigint_fuzz_iteration_count() -> u64 {
    if !cfg!(optimizations_enabled) {
        return 1000;
    }

    if slow_platform() { 100_000 } else { 5_000_000 }
}
