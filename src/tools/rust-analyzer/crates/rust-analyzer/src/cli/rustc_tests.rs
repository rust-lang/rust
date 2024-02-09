//! Run all tests in a project, similar to `cargo test`, but using the mir interpreter.

use std::{
    cell::RefCell, collections::HashMap, fs::read_to_string, panic::AssertUnwindSafe, path::PathBuf,
};

use hir::{Change, Crate};
use ide::{AnalysisHost, DiagnosticCode, DiagnosticsConfig};
use profile::StopWatch;
use project_model::{CargoConfig, ProjectWorkspace, RustLibSource, Sysroot};

use load_cargo::{load_workspace, LoadCargoConfig, ProcMacroServerChoice};
use triomphe::Arc;
use vfs::{AbsPathBuf, FileId};
use walkdir::WalkDir;

use crate::cli::{flags, report_metric, Result};

struct Tester {
    host: AnalysisHost,
    root_file: FileId,
    pass_count: u64,
    ignore_count: u64,
    fail_count: u64,
    stopwatch: StopWatch,
}

fn string_to_diagnostic_code_leaky(code: &str) -> DiagnosticCode {
    thread_local! {
        static LEAK_STORE: RefCell<HashMap<String, DiagnosticCode>> = RefCell::new(HashMap::new());
    }
    LEAK_STORE.with_borrow_mut(|s| match s.get(code) {
        Some(c) => *c,
        None => {
            let v = DiagnosticCode::RustcHardError(format!("E{code}").leak());
            s.insert(code.to_owned(), v);
            v
        }
    })
}

fn detect_errors_from_rustc_stderr_file(p: PathBuf) -> HashMap<DiagnosticCode, usize> {
    let text = read_to_string(p).unwrap();
    let mut result = HashMap::new();
    {
        let mut text = &*text;
        while let Some(p) = text.find("error[E") {
            text = &text[p + 7..];
            let code = string_to_diagnostic_code_leaky(&text[..4]);
            *result.entry(code).or_insert(0) += 1;
        }
    }
    result
}

impl Tester {
    fn new() -> Result<Self> {
        let mut path = std::env::temp_dir();
        path.push("ra-rustc-test.rs");
        let tmp_file = AbsPathBuf::try_from(path).unwrap();
        std::fs::write(&tmp_file, "")?;
        let cargo_config =
            CargoConfig { sysroot: Some(RustLibSource::Discover), ..Default::default() };
        let workspace = ProjectWorkspace::DetachedFiles {
            files: vec![tmp_file.clone()],
            sysroot: Ok(Sysroot::discover(
                tmp_file.parent().unwrap(),
                &cargo_config.extra_env,
                false,
            )
            .unwrap()),
            rustc_cfg: vec![],
        };
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: false,
            with_proc_macro_server: ProcMacroServerChoice::Sysroot,
            prefill_caches: false,
        };
        let (host, _vfs, _proc_macro) =
            load_workspace(workspace, &cargo_config.extra_env, &load_cargo_config)?;
        let db = host.raw_database();
        let krates = Crate::all(db);
        let root_crate = krates.iter().cloned().find(|krate| krate.origin(db).is_local()).unwrap();
        let root_file = root_crate.root_file(db);
        Ok(Self {
            host,
            root_file,
            pass_count: 0,
            ignore_count: 0,
            fail_count: 0,
            stopwatch: StopWatch::start(),
        })
    }

    fn test(&mut self, p: PathBuf) {
        if p.parent().unwrap().file_name().unwrap() == "auxiliary" {
            // These are not tests
            return;
        }
        if IGNORED_TESTS.iter().any(|ig| p.file_name().is_some_and(|x| x == *ig)) {
            println!("{p:?} IGNORE");
            self.ignore_count += 1;
            return;
        }
        let stderr_path = p.with_extension("stderr");
        let expected = if stderr_path.exists() {
            detect_errors_from_rustc_stderr_file(stderr_path)
        } else {
            HashMap::new()
        };
        let text = read_to_string(&p).unwrap();
        let mut change = Change::new();
        // Ignore unstable tests, since they move too fast and we do not intend to support all of them.
        let mut ignore_test = text.contains("#![feature");
        // Ignore test with extern crates, as this infra don't support them yet.
        ignore_test |= text.contains("// aux-build:") || text.contains("// aux-crate:");
        // Ignore test with extern modules similarly.
        ignore_test |= text.contains("mod ");
        // These should work, but they don't, and I don't know why, so ignore them.
        ignore_test |= text.contains("extern crate proc_macro");
        let should_have_no_error = text.contains("// check-pass")
            || text.contains("// build-pass")
            || text.contains("// run-pass");
        change.change_file(self.root_file, Some(Arc::from(text)));
        self.host.apply_change(change);
        let diagnostic_config = DiagnosticsConfig::test_sample();

        let mut actual = HashMap::new();
        let panicked = match std::panic::catch_unwind(|| {
            self.host
                .analysis()
                .diagnostics(&diagnostic_config, ide::AssistResolveStrategy::None, self.root_file)
                .unwrap()
        }) {
            Err(e) => Some(e),
            Ok(diags) => {
                for diag in diags {
                    if !matches!(diag.code, DiagnosticCode::RustcHardError(_)) {
                        continue;
                    }
                    if !should_have_no_error && !SUPPORTED_DIAGNOSTICS.contains(&diag.code) {
                        continue;
                    }
                    *actual.entry(diag.code).or_insert(0) += 1;
                }
                None
            }
        };
        // Ignore tests with diagnostics that we don't emit.
        ignore_test |= expected.keys().any(|k| !SUPPORTED_DIAGNOSTICS.contains(k));
        if ignore_test {
            println!("{p:?} IGNORE");
            self.ignore_count += 1;
        } else if let Some(panic) = panicked {
            if let Some(msg) = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
            {
                println!("{msg:?} ")
            }
            println!("PANIC");
            self.fail_count += 1;
        } else if actual == expected {
            println!("{p:?} PASS");
            self.pass_count += 1;
        } else {
            println!("{p:?} FAIL");
            println!("actual   (r-a)   = {:?}", actual);
            println!("expected (rustc) = {:?}", expected);
            self.fail_count += 1;
        }
    }

    fn report(&mut self) {
        println!(
            "Pass count = {}, Fail count = {}, Ignore count = {}",
            self.pass_count, self.fail_count, self.ignore_count
        );
        println!("Testing time and memory = {}", self.stopwatch.elapsed());
        report_metric("rustc failed tests", self.fail_count, "#");
        report_metric("rustc testing time", self.stopwatch.elapsed().time.as_millis() as u64, "ms");
    }
}

/// These tests break rust-analyzer (either by panicking or hanging) so we should ignore them.
const IGNORED_TESTS: &[&str] = &[
    "trait-with-missing-associated-type-restriction.rs", // #15646
    "trait-with-missing-associated-type-restriction-fixable.rs", // #15646
    "resolve-self-in-impl.rs",
    "basic.rs", // ../rust/tests/ui/associated-type-bounds/return-type-notation/basic.rs
    "issue-26056.rs",
    "float-field.rs",
    "invalid_operator_trait.rs",
    "type-alias-impl-trait-assoc-dyn.rs",
    "deeply-nested_closures.rs",    // exponential time
    "hang-on-deeply-nested-dyn.rs", // exponential time
    "dyn-rpit-and-let.rs", // unexpected free variable with depth `^1.0` with outer binder ^0
    "issue-16098.rs",      // Huge recursion limit for macros?
    "issue-83471.rs", // crates/hir-ty/src/builder.rs:78:9: assertion failed: self.remaining() > 0
];

const SUPPORTED_DIAGNOSTICS: &[DiagnosticCode] = &[
    DiagnosticCode::RustcHardError("E0023"),
    DiagnosticCode::RustcHardError("E0046"),
    DiagnosticCode::RustcHardError("E0063"),
    DiagnosticCode::RustcHardError("E0107"),
    DiagnosticCode::RustcHardError("E0117"),
    DiagnosticCode::RustcHardError("E0133"),
    DiagnosticCode::RustcHardError("E0210"),
    DiagnosticCode::RustcHardError("E0268"),
    DiagnosticCode::RustcHardError("E0308"),
    DiagnosticCode::RustcHardError("E0384"),
    DiagnosticCode::RustcHardError("E0407"),
    DiagnosticCode::RustcHardError("E0432"),
    DiagnosticCode::RustcHardError("E0451"),
    DiagnosticCode::RustcHardError("E0507"),
    DiagnosticCode::RustcHardError("E0583"),
    DiagnosticCode::RustcHardError("E0559"),
    DiagnosticCode::RustcHardError("E0616"),
    DiagnosticCode::RustcHardError("E0618"),
    DiagnosticCode::RustcHardError("E0624"),
    DiagnosticCode::RustcHardError("E0774"),
    DiagnosticCode::RustcHardError("E0767"),
    DiagnosticCode::RustcHardError("E0777"),
];

impl flags::RustcTests {
    pub fn run(self) -> Result<()> {
        let mut tester = Tester::new()?;
        let walk_dir = WalkDir::new(self.rustc_repo.join("tests/ui"));
        for i in walk_dir {
            let i = i?;
            let p = i.into_path();
            if let Some(f) = &self.filter {
                if !p.as_os_str().to_string_lossy().contains(f) {
                    continue;
                }
            }
            if p.extension().map_or(true, |x| x != "rs") {
                continue;
            }
            if let Err(e) = std::panic::catch_unwind({
                let tester = AssertUnwindSafe(&mut tester);
                let p = p.clone();
                move || {
                    let _guard = stdx::panic_context::enter(p.display().to_string());
                    { tester }.0.test(p);
                }
            }) {
                std::panic::resume_unwind(e);
            }
        }
        tester.report();
        Ok(())
    }
}
