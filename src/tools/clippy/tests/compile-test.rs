#![feature(rustc_private, let_chains)]
#![warn(rust_2018_idioms, unused_lifetimes)]
#![allow(unused_extern_crates)]

use askama::Template;
use askama::filters::Safe;
use cargo_metadata::Message;
use cargo_metadata::diagnostic::{Applicability, Diagnostic};
use clippy_config::ClippyConfiguration;
use clippy_lints::LintInfo;
use clippy_lints::declared_lints::LINTS;
use clippy_lints::deprecated_lints::{DEPRECATED, DEPRECATED_VERSION, RENAMED};
use pulldown_cmark::{Options, Parser, html};
use serde::Deserialize;
use test_utils::IS_RUSTC_TEST_SUITE;
use ui_test::custom_flags::Flag;
use ui_test::custom_flags::edition::Edition;
use ui_test::custom_flags::rustfix::RustfixMode;
use ui_test::spanned::Spanned;
use ui_test::{Args, CommandBuilder, Config, Match, error_on_output_conflict, status_emitter};

use std::collections::{BTreeMap, HashMap};
use std::env::{self, set_var, var_os};
use std::ffi::{OsStr, OsString};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Sender, channel};
use std::{fs, iter, thread};

// Test dependencies may need an `extern crate` here to ensure that they show up
// in the depinfo file (otherwise cargo thinks they are unused)
extern crate futures;
extern crate if_chain;
extern crate itertools;
extern crate parking_lot;
extern crate quote;
extern crate syn;
extern crate tokio;

mod test_utils;

/// All crates used in UI tests are listed here
static TEST_DEPENDENCIES: &[&str] = &[
    "clippy_config",
    "clippy_lints",
    "clippy_utils",
    "futures",
    "if_chain",
    "itertools",
    "parking_lot",
    "quote",
    "regex",
    "serde_derive",
    "serde",
    "syn",
    "tokio",
];

/// Produces a string with an `--extern` flag for all UI test crate
/// dependencies.
///
/// The dependency files are located by parsing the depinfo file for this test
/// module. This assumes the `-Z binary-dep-depinfo` flag is enabled. All test
/// dependencies must be added to Cargo.toml at the project root. Test
/// dependencies that are not *directly* used by this test module require an
/// `extern crate` declaration.
fn extern_flags() -> Vec<String> {
    let current_exe_depinfo = {
        let mut path = env::current_exe().unwrap();
        path.set_extension("d");
        fs::read_to_string(path).unwrap()
    };
    let mut crates = BTreeMap::<&str, &str>::new();
    for line in current_exe_depinfo.lines() {
        // each dependency is expected to have a Makefile rule like `/path/to/crate-hash.rlib:`
        let parse_name_path = || {
            if line.starts_with(char::is_whitespace) {
                return None;
            }
            let path_str = line.strip_suffix(':')?;
            let path = Path::new(path_str);
            if !matches!(path.extension()?.to_str()?, "rlib" | "so" | "dylib" | "dll") {
                return None;
            }
            let (name, _hash) = path.file_stem()?.to_str()?.rsplit_once('-')?;
            // the "lib" prefix is not present for dll files
            let name = name.strip_prefix("lib").unwrap_or(name);
            Some((name, path_str))
        };
        if let Some((name, path)) = parse_name_path()
            && TEST_DEPENDENCIES.contains(&name)
        {
            // A dependency may be listed twice if it is available in sysroot,
            // and the sysroot dependencies are listed first. As of the writing,
            // this only seems to apply to if_chain.
            crates.insert(name, path);
        }
    }
    let not_found: Vec<&str> = TEST_DEPENDENCIES
        .iter()
        .copied()
        .filter(|n| !crates.contains_key(n))
        .collect();
    assert!(
        not_found.is_empty(),
        "dependencies not found in depinfo: {not_found:?}\n\
        help: Make sure the `-Z binary-dep-depinfo` rust flag is enabled\n\
        help: Try adding to dev-dependencies in Cargo.toml\n\
        help: Be sure to also add `extern crate ...;` to tests/compile-test.rs",
    );
    crates
        .into_iter()
        .map(|(name, path)| format!("--extern={name}={path}"))
        .collect()
}

// whether to run internal tests or not
const RUN_INTERNAL_TESTS: bool = cfg!(feature = "internal");

struct TestContext {
    args: Args,
    extern_flags: Vec<String>,
    diagnostic_collector: Option<DiagnosticCollector>,
    collector_thread: Option<thread::JoinHandle<()>>,
}

impl TestContext {
    fn new() -> Self {
        let mut args = Args::test().unwrap();
        args.bless |= var_os("RUSTC_BLESS").is_some_and(|v| v != "0");
        let (diagnostic_collector, collector_thread) = var_os("COLLECT_METADATA")
            .is_some()
            .then(DiagnosticCollector::spawn)
            .unzip();
        Self {
            args,
            extern_flags: extern_flags(),
            diagnostic_collector,
            collector_thread,
        }
    }

    fn base_config(&self, test_dir: &str, mandatory_annotations: bool) -> Config {
        let target_dir = PathBuf::from(var_os("CARGO_TARGET_DIR").unwrap_or_else(|| "target".into()));
        let mut config = Config {
            output_conflict_handling: error_on_output_conflict,
            filter_files: env::var("TESTNAME")
                .map(|filters| filters.split(',').map(str::to_string).collect())
                .unwrap_or_default(),
            target: None,
            bless_command: Some(if IS_RUSTC_TEST_SUITE {
                "./x test src/tools/clippy --bless".into()
            } else {
                "cargo uibless".into()
            }),
            out_dir: target_dir.join("ui_test"),
            ..Config::rustc(Path::new("tests").join(test_dir))
        };
        let defaults = config.comment_defaults.base();
        defaults.set_custom("edition", Edition("2024".into()));
        defaults.exit_status = None.into();
        if mandatory_annotations {
            defaults.require_annotations = Some(Spanned::dummy(true)).into();
        } else {
            defaults.require_annotations = None.into();
        }
        defaults.diagnostic_code_prefix = Some(Spanned::dummy("clippy::".into())).into();
        defaults.set_custom("rustfix", RustfixMode::Everything);
        if let Some(collector) = self.diagnostic_collector.clone() {
            defaults.set_custom("diagnostic-collector", collector);
        }
        config.with_args(&self.args);
        let current_exe_path = env::current_exe().unwrap();
        let deps_path = current_exe_path.parent().unwrap();
        let profile_path = deps_path.parent().unwrap();

        config.program.args.extend(
            [
                "--emit=metadata",
                "-Aunused",
                "-Ainternal_features",
                "-Zui-testing",
                "-Zdeduplicate-diagnostics=no",
                "-Dwarnings",
                &format!("-Ldependency={}", deps_path.display()),
            ]
            .map(OsString::from),
        );

        config.program.args.extend(self.extern_flags.iter().map(OsString::from));
        // Prevent rustc from creating `rustc-ice-*` files the console output is enough.
        config.program.envs.push(("RUSTC_ICE".into(), Some("0".into())));

        if let Some(host_libs) = option_env!("HOST_LIBS") {
            let dep = format!("-Ldependency={}", Path::new(host_libs).join("deps").display());
            config.program.args.push(dep.into());
        }

        config.program.program = profile_path.join(if cfg!(windows) {
            "clippy-driver.exe"
        } else {
            "clippy-driver"
        });

        config
    }
}

fn run_ui(cx: &TestContext) {
    let mut config = cx.base_config("ui", true);
    config
        .program
        .envs
        .push(("CLIPPY_CONF_DIR".into(), Some("tests".into())));

    ui_test::run_tests_generic(
        vec![config],
        ui_test::default_file_filter,
        ui_test::default_per_file_config,
        status_emitter::Text::from(cx.args.format),
    )
    .unwrap();
}

fn run_internal_tests(cx: &TestContext) {
    if !RUN_INTERNAL_TESTS {
        return;
    }
    let mut config = cx.base_config("ui-internal", true);
    config.bless_command = Some("cargo uitest --features internal -- -- --bless".into());

    ui_test::run_tests_generic(
        vec![config],
        ui_test::default_file_filter,
        ui_test::default_per_file_config,
        status_emitter::Text::from(cx.args.format),
    )
    .unwrap();
}

fn run_ui_toml(cx: &TestContext) {
    let mut config = cx.base_config("ui-toml", true);

    config
        .comment_defaults
        .base()
        .normalize_stderr
        .push((Match::from(env::current_dir().unwrap().as_path()), b"$DIR".into()));

    ui_test::run_tests_generic(
        vec![config],
        ui_test::default_file_filter,
        |config, file_contents| {
            let path = file_contents.span().file;
            config
                .program
                .envs
                .push(("CLIPPY_CONF_DIR".into(), Some(path.parent().unwrap().into())));
        },
        status_emitter::Text::from(cx.args.format),
    )
    .unwrap();
}

// Allow `Default::default` as `OptWithSpan` is not nameable
#[allow(clippy::default_trait_access)]
fn run_ui_cargo(cx: &TestContext) {
    if IS_RUSTC_TEST_SUITE {
        return;
    }

    let mut config = cx.base_config("ui-cargo", false);
    config.program.input_file_flag = CommandBuilder::cargo().input_file_flag;
    config.program.out_dir_flag = CommandBuilder::cargo().out_dir_flag;
    config.program.args = vec!["clippy".into(), "--color".into(), "never".into(), "--quiet".into()];
    config
        .program
        .envs
        .push(("RUSTFLAGS".into(), Some("-Dwarnings".into())));
    // We need to do this while we still have a rustc in the `program` field.
    config.fill_host_and_target().unwrap();
    config.program.program.set_file_name(if cfg!(windows) {
        "cargo-clippy.exe"
    } else {
        "cargo-clippy"
    });
    config.comment_defaults.base().custom.clear();

    config
        .comment_defaults
        .base()
        .normalize_stderr
        .push((Match::from(env::current_dir().unwrap().as_path()), b"$DIR".into()));

    let ignored_32bit = |path: &Path| {
        // FIXME: for some reason the modules are linted in a different order for this test
        cfg!(target_pointer_width = "32") && path.ends_with("tests/ui-cargo/module_style/fail_mod/Cargo.toml")
    };

    ui_test::run_tests_generic(
        vec![config],
        |path, config| {
            path.ends_with("Cargo.toml")
                .then(|| ui_test::default_any_file_filter(path, config) && !ignored_32bit(path))
        },
        |_config, _file_contents| {},
        status_emitter::Text::from(cx.args.format),
    )
    .unwrap();
}

fn main() {
    unsafe {
        set_var("CLIPPY_DISABLE_DOCS_LINKS", "true");
    }

    let cx = TestContext::new();

    // The SPEEDTEST_* env variables can be used to check Clippy's performance on your PR. It runs the
    // affected test 1000 times and gets the average.
    if let Ok(speedtest) = std::env::var("SPEEDTEST") {
        println!("----------- STARTING SPEEDTEST -----------");
        let f = match speedtest.as_str() {
            "ui" => run_ui,
            "cargo" => run_ui_cargo,
            "toml" => run_ui_toml,
            "internal" => run_internal_tests,

            _ => panic!("unknown speedtest: {speedtest} || accepted speedtests are: [ui, cargo, toml, internal]"),
        };

        let iterations;
        if let Ok(iterations_str) = std::env::var("SPEEDTEST_ITERATIONS") {
            iterations = iterations_str
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("Couldn't parse `{iterations_str}`, please use a valid u64"));
        } else {
            iterations = 1000;
        }

        let mut sum = 0;
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            f(&cx);
            sum += start.elapsed().as_millis();
        }
        println!(
            "average {} time: {} millis.",
            speedtest.to_uppercase(),
            sum / u128::from(iterations)
        );
    } else {
        run_ui(&cx);
        run_ui_toml(&cx);
        run_ui_cargo(&cx);
        run_internal_tests(&cx);
        drop(cx.diagnostic_collector);

        ui_cargo_toml_metadata();

        if let Some(thread) = cx.collector_thread {
            thread.join().unwrap();
        }
    }
}

fn ui_cargo_toml_metadata() {
    let ui_cargo_path = Path::new("tests/ui-cargo");
    let cargo_common_metadata_path = ui_cargo_path.join("cargo_common_metadata");
    let publish_exceptions =
        ["fail_publish", "fail_publish_true", "pass_publish_empty"].map(|path| cargo_common_metadata_path.join(path));

    for entry in walkdir::WalkDir::new(ui_cargo_path) {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.file_name() != Some(OsStr::new("Cargo.toml")) {
            continue;
        }

        let toml = fs::read_to_string(path).unwrap().parse::<toml::Value>().unwrap();

        let package = toml.as_table().unwrap().get("package").unwrap().as_table().unwrap();

        let name = package.get("name").unwrap().as_str().unwrap().replace('-', "_");
        assert!(
            path.parent()
                .unwrap()
                .components()
                .map(|component| component.as_os_str().to_string_lossy().replace('-', "_"))
                .any(|s| *s == name)
                || path.starts_with(&cargo_common_metadata_path),
            "`{}` has incorrect package name",
            path.display(),
        );

        let publish = package.get("publish").and_then(toml::Value::as_bool).unwrap_or(true);
        assert!(
            !publish || publish_exceptions.contains(&path.parent().unwrap().to_path_buf()),
            "`{}` lacks `publish = false`",
            path.display(),
        );
    }
}

#[derive(Template)]
#[template(path = "index_template.html")]
struct Renderer<'a> {
    lints: &'a Vec<LintMetadata>,
}

impl Renderer<'_> {
    fn markdown(input: &str) -> Safe<String> {
        let input = clippy_config::sanitize_explanation(input);
        let parser = Parser::new_ext(&input, Options::all());
        let mut html_output = String::new();
        html::push_html(&mut html_output, parser);
        // Oh deer, what a hack :O
        Safe(html_output.replace("<table", "<table class=\"table\""))
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DiagnosticOrMessage {
    Diagnostic(Diagnostic),
    Message(Message),
}

/// Collects applicabilities from the diagnostics produced for each UI test, producing the
/// `util/gh-pages/lints.json` file used by <https://rust-lang.github.io/rust-clippy/>
#[derive(Debug, Clone)]
struct DiagnosticCollector {
    sender: Sender<Vec<u8>>,
}

impl DiagnosticCollector {
    #[allow(clippy::assertions_on_constants)]
    fn spawn() -> (Self, thread::JoinHandle<()>) {
        assert!(!IS_RUSTC_TEST_SUITE && !RUN_INTERNAL_TESTS);

        let (sender, receiver) = channel::<Vec<u8>>();

        let handle = thread::spawn(|| {
            let mut applicabilities = HashMap::new();

            for stderr in receiver {
                for line in stderr.split(|&byte| byte == b'\n') {
                    let diag = match serde_json::from_slice(line) {
                        Ok(DiagnosticOrMessage::Diagnostic(diag)) => diag,
                        Ok(DiagnosticOrMessage::Message(Message::CompilerMessage(message))) => message.message,
                        _ => continue,
                    };

                    if let Some(lint) = diag.code.as_ref().and_then(|code| code.code.strip_prefix("clippy::")) {
                        let applicability = applicabilities
                            .entry(lint.to_string())
                            .or_insert(Applicability::Unspecified);
                        let diag_applicability = diag
                            .children
                            .iter()
                            .flat_map(|child| &child.spans)
                            .filter_map(|span| span.suggestion_applicability.clone())
                            .max_by_key(applicability_ord);
                        if let Some(diag_applicability) = diag_applicability
                            && applicability_ord(&diag_applicability) > applicability_ord(applicability)
                        {
                            *applicability = diag_applicability;
                        }
                    }
                }
            }

            let configs = clippy_config::get_configuration_metadata();
            let mut metadata: Vec<LintMetadata> = LINTS
                .iter()
                .map(|lint| LintMetadata::new(lint, &applicabilities, &configs))
                .chain(
                    iter::zip(DEPRECATED, DEPRECATED_VERSION)
                        .map(|((lint, reason), version)| LintMetadata::new_deprecated(lint, reason, version)),
                )
                .collect();

            metadata.sort_unstable_by(|a, b| a.id.cmp(&b.id));

            fs::write(
                "util/gh-pages/index.html",
                Renderer { lints: &metadata }.render().unwrap(),
            )
            .unwrap();
        });

        (Self { sender }, handle)
    }
}

fn applicability_ord(applicability: &Applicability) -> u8 {
    match applicability {
        Applicability::MachineApplicable => 4,
        Applicability::HasPlaceholders => 3,
        Applicability::MaybeIncorrect => 2,
        Applicability::Unspecified => 1,
        _ => unimplemented!(),
    }
}

impl Flag for DiagnosticCollector {
    fn post_test_action(
        &self,
        _config: &ui_test::per_test_config::TestConfig,
        output: &std::process::Output,
        _build_manager: &ui_test::build_manager::BuildManager,
    ) -> Result<(), ui_test::Errored> {
        if !output.stderr.is_empty() {
            self.sender.send(output.stderr.clone()).unwrap();
        }
        Ok(())
    }

    fn clone_inner(&self) -> Box<dyn Flag> {
        Box::new(self.clone())
    }

    fn must_be_unique(&self) -> bool {
        true
    }
}

#[derive(Debug)]
struct LintMetadata {
    id: String,
    id_location: Option<&'static str>,
    group: &'static str,
    level: &'static str,
    docs: String,
    version: &'static str,
    applicability: Applicability,
}

impl LintMetadata {
    fn new(lint: &LintInfo, applicabilities: &HashMap<String, Applicability>, configs: &[ClippyConfiguration]) -> Self {
        let name = lint.name_lower();
        let applicability = applicabilities
            .get(&name)
            .cloned()
            .unwrap_or(Applicability::Unspecified);
        let past_names = RENAMED
            .iter()
            .filter(|(_, new_name)| new_name.strip_prefix("clippy::") == Some(&name))
            .map(|(old_name, _)| old_name.strip_prefix("clippy::").unwrap())
            .collect::<Vec<_>>();
        let mut docs = lint.explanation.to_string();
        if !past_names.is_empty() {
            docs.push_str("\n### Past names\n\n");
            for past_name in past_names {
                writeln!(&mut docs, " * {past_name}").unwrap();
            }
        }
        let configs: Vec<_> = configs
            .iter()
            .filter(|conf| conf.lints.contains(&name.as_str()))
            .collect();
        if !configs.is_empty() {
            docs.push_str("\n### Configuration\n\n");
            for config in configs {
                writeln!(&mut docs, "{config}").unwrap();
            }
        }
        Self {
            id: name,
            id_location: Some(lint.location),
            group: lint.category_str(),
            level: lint.lint.default_level.as_str(),
            docs,
            version: lint.version.unwrap(),
            applicability,
        }
    }

    fn new_deprecated(name: &str, reason: &str, version: &'static str) -> Self {
        // The reason starts with a lowercase letter and ends without a period.
        // This needs to be fixed for the website.
        let mut reason = reason.to_owned();
        if let Some(reason) = reason.get_mut(0..1) {
            reason.make_ascii_uppercase();
        }
        Self {
            id: name.strip_prefix("clippy::").unwrap().into(),
            id_location: None,
            group: "deprecated",
            level: "none",
            docs: format!(
                "### What it does\n\n\
                Nothing. This lint has been deprecated\n\n\
                ### Deprecation reason\n\n{reason}.\n",
            ),
            version,
            applicability: Applicability::Unspecified,
        }
    }

    fn applicability_str(&self) -> &str {
        match self.applicability {
            Applicability::MachineApplicable => "MachineApplicable",
            Applicability::HasPlaceholders => "HasPlaceholders",
            Applicability::MaybeIncorrect => "MaybeIncorrect",
            Applicability::Unspecified => "Unspecified",
            _ => panic!("needs to update this code"),
        }
    }
}
