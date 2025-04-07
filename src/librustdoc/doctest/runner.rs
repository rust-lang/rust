use std::fmt::Write;

use rustc_data_structures::fx::FxIndexSet;
use rustc_span::edition::Edition;

use crate::doctest::{
    DocTestBuilder, GlobalTestOptions, IndividualTestOptions, RunnableDocTest, RustdocOptions,
    ScrapedDocTest, TestFailure, UnusedExterns, run_test,
};
use crate::html::markdown::{Ignore, LangString};

/// Convenient type to merge compatible doctests into one.
pub(crate) struct DocTestRunner {
    crate_attrs: FxIndexSet<String>,
    ids: String,
    output: String,
    output_merged_tests: String,
    supports_color: bool,
    nb_tests: usize,
}

impl DocTestRunner {
    pub(crate) fn new() -> Self {
        Self {
            crate_attrs: FxIndexSet::default(),
            ids: String::new(),
            output: String::new(),
            output_merged_tests: String::new(),
            supports_color: true,
            nb_tests: 0,
        }
    }

    pub(crate) fn add_test(
        &mut self,
        doctest: &DocTestBuilder,
        scraped_test: &ScrapedDocTest,
        target_str: &str,
    ) {
        let ignore = match scraped_test.langstr.ignore {
            Ignore::All => true,
            Ignore::None => false,
            Ignore::Some(ref ignores) => ignores.iter().any(|s| target_str.contains(s)),
        };
        if !ignore {
            for line in doctest.crate_attrs.split('\n') {
                self.crate_attrs.insert(line.to_string());
            }
        }
        self.ids.push_str(&format!(
            "tests.push({}::TEST);\n",
            generate_mergeable_doctest(
                doctest,
                scraped_test,
                ignore,
                self.nb_tests,
                &mut self.output,
                &mut self.output_merged_tests,
            ),
        ));
        self.supports_color &= doctest.supports_color;
        self.nb_tests += 1;
    }

    pub(crate) fn run_merged_tests(
        &mut self,
        test_options: IndividualTestOptions,
        edition: Edition,
        opts: &GlobalTestOptions,
        test_args: &[String],
        rustdoc_options: &RustdocOptions,
    ) -> Result<bool, ()> {
        let mut code = "\
#![allow(unused_extern_crates)]
#![allow(internal_features)]
#![feature(test)]
#![feature(rustc_attrs)]
"
        .to_string();

        let mut code_prefix = String::new();

        for crate_attr in &self.crate_attrs {
            code_prefix.push_str(crate_attr);
            code_prefix.push('\n');
        }

        if opts.attrs.is_empty() {
            // If there aren't any attributes supplied by #![doc(test(attr(...)))], then allow some
            // lints that are commonly triggered in doctests. The crate-level test attributes are
            // commonly used to make tests fail in case they trigger warnings, so having this there in
            // that case may cause some tests to pass when they shouldn't have.
            code_prefix.push_str("#![allow(unused)]\n");
        }

        // Next, any attributes that came from the crate root via #![doc(test(attr(...)))].
        for attr in &opts.attrs {
            code_prefix.push_str(&format!("#![{attr}]\n"));
        }

        code.push_str("extern crate test;\n");
        writeln!(code, "extern crate doctest_bundle_{edition} as doctest_bundle;").unwrap();

        let test_args = test_args.iter().fold(String::new(), |mut x, arg| {
            write!(x, "{arg:?}.to_string(),").unwrap();
            x
        });
        write!(
            code,
            "\
{output}

mod __doctest_mod {{
    use std::sync::OnceLock;
    use std::path::PathBuf;
    use std::process::ExitCode;

    pub static BINARY_PATH: OnceLock<PathBuf> = OnceLock::new();
    pub const RUN_OPTION: &str = \"RUSTDOC_DOCTEST_RUN_NB_TEST\";

    #[allow(unused)]
    pub fn doctest_path() -> Option<&'static PathBuf> {{
        self::BINARY_PATH.get()
    }}

    #[allow(unused)]
    pub fn doctest_runner(bin: &std::path::Path, test_nb: usize) -> ExitCode {{
        let out = std::process::Command::new(bin)
            .env(self::RUN_OPTION, test_nb.to_string())
            .args(std::env::args().skip(1).collect::<Vec<_>>())
            .output()
            .expect(\"failed to run command\");
        if !out.status.success() {{
            eprint!(\"{{}}\", String::from_utf8_lossy(&out.stderr));
            ExitCode::FAILURE
        }} else {{
            ExitCode::SUCCESS
        }}
    }}
}}

#[rustc_main]
fn main() -> std::process::ExitCode {{
let tests = {{
    let mut tests = Vec::with_capacity({nb_tests});
    {ids}
    tests
}};
let test_args = &[{test_args}];
const ENV_BIN: &'static str = \"RUSTDOC_DOCTEST_BIN_PATH\";

if let Ok(binary) = std::env::var(ENV_BIN) {{
    let _ = crate::__doctest_mod::BINARY_PATH.set(binary.into());
    unsafe {{ std::env::remove_var(ENV_BIN); }}
    return std::process::Termination::report(test::test_main(test_args, tests, None));
}} else if let Ok(nb_test) = std::env::var(__doctest_mod::RUN_OPTION) {{
    if let Ok(nb_test) = nb_test.parse::<usize>() {{
        if let Some(test) = tests.get(nb_test) {{
            if let test::StaticTestFn(f) = &test.testfn {{
                return std::process::Termination::report(f());
            }}
        }}
    }}
    panic!(\"Unexpected value for `{{}}`\", __doctest_mod::RUN_OPTION);
}}

eprintln!(\"WARNING: No rustdoc doctest environment variable provided so doctests will be run in \
the same process\");
std::process::Termination::report(test::test_main(test_args, tests, None))
}}",
            nb_tests = self.nb_tests,
            output = self.output_merged_tests,
            ids = self.ids,
        )
        .expect("failed to generate test code");
        let runnable_test = RunnableDocTest {
            full_test_code: format!("{code_prefix}{code}", code = self.output),
            full_test_line_offset: 0,
            test_opts: test_options,
            global_opts: opts.clone(),
            langstr: LangString::default(),
            line: 0,
            edition,
            no_run: false,
            merged_test_code: Some(code),
        };
        let ret =
            run_test(runnable_test, rustdoc_options, self.supports_color, |_: UnusedExterns| {});
        if let Err(TestFailure::CompileError) = ret { Err(()) } else { Ok(ret.is_ok()) }
    }
}

/// Push new doctest content into `output`. Returns the test ID for this doctest.
fn generate_mergeable_doctest(
    doctest: &DocTestBuilder,
    scraped_test: &ScrapedDocTest,
    ignore: bool,
    id: usize,
    output: &mut String,
    output_merged_tests: &mut String,
) -> String {
    let test_id = format!("__doctest_{id}");

    if ignore {
        // We generate nothing else.
        writeln!(output, "pub mod {test_id} {{}}\n").unwrap();
    } else {
        writeln!(output, "pub mod {test_id} {{\n{}{}", doctest.crates, doctest.maybe_crate_attrs)
            .unwrap();
        if doctest.has_main_fn {
            output.push_str(&doctest.everything_else);
        } else {
            let returns_result = if doctest.everything_else.trim_end().ends_with("(())") {
                "-> Result<(), impl core::fmt::Debug>"
            } else {
                ""
            };
            write!(
                output,
                "\
fn main() {returns_result} {{
{}
}}",
                doctest.everything_else
            )
            .unwrap();
        }
        writeln!(
            output,
            "\npub fn __main_fn() -> impl std::process::Termination {{ main() }} \n}}\n"
        )
        .unwrap();
    }
    let not_running = ignore || scraped_test.langstr.no_run;
    writeln!(
        output_merged_tests,
        "
mod {test_id} {{
pub const TEST: test::TestDescAndFn = test::TestDescAndFn::new_doctest(
{test_name:?}, {ignore}, {file:?}, {line}, {no_run}, {should_panic},
test::StaticTestFn(
    || {{{runner}}},
));
}}",
        test_name = scraped_test.name,
        file = scraped_test.path(),
        line = scraped_test.line,
        no_run = scraped_test.langstr.no_run,
        should_panic = !scraped_test.langstr.no_run && scraped_test.langstr.should_panic,
        // Setting `no_run` to `true` in `TestDesc` still makes the test run, so we simply
        // don't give it the function to run.
        runner = if not_running {
            "test::assert_test_result(Ok::<(), String>(()))".to_string()
        } else {
            format!(
                "
if let Some(bin_path) = crate::__doctest_mod::doctest_path() {{
    test::assert_test_result(crate::__doctest_mod::doctest_runner(bin_path, {id}))
}} else {{
    test::assert_test_result(doctest_bundle::{test_id}::__main_fn())
}}
",
            )
        },
    )
    .unwrap();
    test_id
}
