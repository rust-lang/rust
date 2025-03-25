use camino::Utf8PathBuf;
use derive_builder::Builder;

#[derive(Builder)]
pub struct Environment {
    host_tuple: String,
    python_binary: String,
    /// The rustc checkout, where the compiler source is located.
    checkout_dir: Utf8PathBuf,
    /// The main directory where the build occurs. Stage0 rustc and cargo have to be available in
    /// this directory before `opt-dist` is started.
    build_dir: Utf8PathBuf,
    /// Directory where the optimization artifacts (PGO/BOLT profiles, etc.)
    /// will be stored.
    artifact_dir: Utf8PathBuf,
    /// Path to the host LLVM used to compile LLVM in `src/llvm-project`.
    host_llvm_dir: Utf8PathBuf,
    /// List of test paths that should be skipped when testing the optimized artifacts.
    skipped_tests: Vec<String>,
    /// Arguments passed to `rustc-perf --cargo-config <value>` when running benchmarks.
    #[builder(default)]
    benchmark_cargo_config: Vec<String>,
    /// Directory containing a pre-built rustc-perf checkout.
    #[builder(default)]
    prebuilt_rustc_perf: Option<Utf8PathBuf>,
    use_bolt: bool,
    shared_llvm: bool,
    /// Additional configuration that bootstrap needs to know only when running tests.
    #[builder(default)]
    test_config: TestConfig,
}

/// Builds have optional components, and their presence/absence can enable/disable a subset of
/// tests. When testing the optimized artifacts, bootstrap needs to know about these enabled
/// components to run the expected subset. This structure holds the known components where this
/// matters: currently only whether the build to test is using debug assertions.
///
/// FIXME: ultimately, this is a temporary band-aid, and opt-dist should be more transparent to the
/// CI config and bootstrap optional components: bootstrap has default values, combinations of flags
/// that cascade into others, etc logic that we'd have to duplicate here otherwise. It's more
/// sensible for opt-dist to never know about the config apart from the minimal set of paths
/// required to configure stage0 tests.
#[derive(Builder, Default, Clone, Debug)]
pub struct TestConfig {
    /// Whether the build under test is explicitly using `--enable-debug-assertions`.
    /// Note that this flag can be implied from others, like `rust.debug`, and we do not handle any
    /// of these subtleties and defaults here, as per the FIXME above.
    pub enable_debug_assertions: bool,
}

impl Environment {
    pub fn host_tuple(&self) -> &str {
        &self.host_tuple
    }

    pub fn python_binary(&self) -> &str {
        &self.python_binary
    }

    pub fn checkout_path(&self) -> Utf8PathBuf {
        self.checkout_dir.clone()
    }

    pub fn build_root(&self) -> Utf8PathBuf {
        self.build_dir.clone()
    }

    pub fn build_artifacts(&self) -> Utf8PathBuf {
        self.build_root().join("build").join(&self.host_tuple)
    }

    pub fn artifact_dir(&self) -> Utf8PathBuf {
        self.artifact_dir.clone()
    }

    pub fn cargo_stage_0(&self) -> Utf8PathBuf {
        self.build_artifacts()
            .join("stage0")
            .join("bin")
            .join(format!("cargo{}", executable_extension()))
    }

    pub fn rustc_stage_0(&self) -> Utf8PathBuf {
        self.build_artifacts()
            .join("stage0")
            .join("bin")
            .join(format!("rustc{}", executable_extension()))
    }

    pub fn rustc_stage_2(&self) -> Utf8PathBuf {
        self.build_artifacts()
            .join("stage2")
            .join("bin")
            .join(format!("rustc{}", executable_extension()))
    }

    pub fn prebuilt_rustc_perf(&self) -> Option<Utf8PathBuf> {
        self.prebuilt_rustc_perf.clone()
    }

    /// Path to the built rustc-perf benchmark suite.
    pub fn rustc_perf_dir(&self) -> Utf8PathBuf {
        self.artifact_dir.join("rustc-perf")
    }

    pub fn host_llvm_dir(&self) -> Utf8PathBuf {
        self.host_llvm_dir.clone()
    }

    pub fn use_bolt(&self) -> bool {
        self.use_bolt
    }

    pub fn supports_shared_llvm(&self) -> bool {
        self.shared_llvm
    }

    pub fn skipped_tests(&self) -> &[String] {
        &self.skipped_tests
    }

    pub fn benchmark_cargo_config(&self) -> &[String] {
        &self.benchmark_cargo_config
    }

    pub fn test_config(&self) -> &TestConfig {
        &self.test_config
    }
}

impl TestConfig {
    /// Returns the test config matching the given `RUST_CONFIGURE_ARGS` for the known optional
    /// components for tests. This is obviously extremely fragile and we'd rather opt-dist not
    /// handle any optional components.
    pub fn from_configure_args(configure_args: &str) -> TestConfig {
        let enable_debug_assertions =
            configure_args.split(" ").find(|part| *part == "--enable-debug-assertions").is_some();
        TestConfig { enable_debug_assertions }
    }
}

/// What is the extension of binary executables on this platform?
#[cfg(target_family = "unix")]
pub fn executable_extension() -> &'static str {
    ""
}

#[cfg(target_family = "windows")]
pub fn executable_extension() -> &'static str {
    ".exe"
}
