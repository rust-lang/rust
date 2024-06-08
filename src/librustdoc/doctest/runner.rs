use rustc_data_structures::fx::FxHashSet;
use rustc_span::edition::Edition;

use std::fmt::Write;
use std::sync::{Arc, Mutex};

use crate::doctest::{
    run_test, DirState, DocTest, GlobalTestOptions, IndividualTestOptions, RunnableDoctest,
    RustdocOptions, ScrapedDoctest, TestFailure, UnusedExterns,
};
use crate::html::markdown::LangString;

/// Convenient type to merge compatible doctests into one.
pub(crate) struct DocTestRunner {
    crate_attrs: FxHashSet<String>,
    ids: String,
    output: String,
    supports_color: bool,
    nb_tests: usize,
    doctests: Vec<DocTest>,
}

impl DocTestRunner {
    pub(crate) fn new() -> Self {
        Self {
            crate_attrs: FxHashSet::default(),
            ids: String::new(),
            output: String::new(),
            supports_color: true,
            nb_tests: 0,
            doctests: Vec::with_capacity(10),
        }
    }

    pub(crate) fn add_test(&mut self, doctest: &DocTest, scraped_test: &ScrapedDoctest) {
        if !doctest.ignore {
            for line in doctest.crate_attrs.split('\n') {
                self.crate_attrs.insert(line.to_string());
            }
        }
        if !self.ids.is_empty() {
            self.ids.push(',');
        }
        self.ids.push_str(&format!(
            "{}::TEST",
            generate_mergeable_doctest(doctest, scraped_test, self.nb_tests, &mut self.output),
        ));
        self.supports_color &= doctest.supports_color;
        self.nb_tests += 1;
        self.doctests.push(doctest);
    }

    pub(crate) fn run_tests(
        &mut self,
        test_options: IndividualTestOptions,
        edition: Edition,
        opts: &GlobalTestOptions,
        test_args: &[String],
        outdir: &Arc<DirState>,
        rustdoc_options: &RustdocOptions,
        unused_externs: Arc<Mutex<Vec<UnusedExterns>>>,
    ) -> Result<bool, ()> {
        let mut code = "\
#![allow(unused_extern_crates)]
#![allow(internal_features)]
#![feature(test)]
#![feature(rustc_attrs)]
#![feature(coverage_attribute)]\n"
            .to_string();

        for crate_attr in &self.crate_attrs {
            code.push_str(crate_attr);
            code.push('\n');
        }

        DocTest::push_attrs(&mut code, opts, &mut 0);
        code.push_str("extern crate test;\n");

        let test_args =
            test_args.iter().map(|arg| format!("{arg:?}.to_string(),")).collect::<String>();
        write!(
            code,
            "\
{output}
#[rustc_main]
#[coverage(off)]
fn main() {{
test::test_main(&[{test_args}], vec![{ids}], None);
}}",
            output = self.output,
            ids = self.ids,
        )
        .expect("failed to generate test code");
        // let out_dir = build_test_dir(outdir, true, "");
        let runnable_test = RunnableDoctest {
            full_test_code: code,
            full_test_line_offset: 0,
            test_opts: test_options,
            global_opts: opts.clone(),
            langstr: LangString::default(),
            line: 0,
            edition,
            no_run: false,
        };
        let ret = run_test(runnable_test, rustdoc_options, self.supports_color, unused_externs);
        if let Err(TestFailure::CompileError) = ret { Err(()) } else { Ok(ret.is_ok()) }
    }
}

/// Push new doctest content into `output`. Returns the test ID for this doctest.
fn generate_mergeable_doctest(
    doctest: &DocTest,
    scraped_test: &ScrapedDoctest,
    id: usize,
    output: &mut String,
) -> String {
    let test_id = format!("__doctest_{id}");

    if doctest.ignore {
        // We generate nothing else.
        writeln!(output, "mod {test_id} {{\n").unwrap();
    } else {
        writeln!(output, "mod {test_id} {{\n{}", doctest.crates).unwrap();
        if doctest.main_fn_span.is_some() {
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
    }
    writeln!(
        output,
        "
#[rustc_test_marker = {test_name:?}]
pub const TEST: test::TestDescAndFn = test::TestDescAndFn {{
    desc: test::TestDesc {{
        name: test::StaticTestName({test_name:?}),
        ignore: {ignore},
        ignore_message: None,
        source_file: {file:?},
        start_line: {line},
        start_col: 0,
        end_line: 0,
        end_col: 0,
        compile_fail: false,
        no_run: {no_run},
        should_panic: test::ShouldPanic::{should_panic},
        test_type: test::TestType::UnitTest,
    }},
    testfn: test::StaticTestFn(
        #[coverage(off)]
        || test::assert_test_result({runner}),
    )
}};
}}",
        test_name = scraped_test.name,
        ignore = scraped_test.langstr.ignore,
        file = scraped_test.file,
        line = scraped_test.line,
        no_run = scraped_test.langstr.no_run,
        should_panic = if !scraped_test.langstr.no_run && scraped_test.langstr.should_panic {
            "Yes"
        } else {
            "No"
        },
        // Setting `no_run` to `true` in `TestDesc` still makes the test run, so we simply
        // don't give it the function to run.
        runner = if scraped_test.langstr.no_run || scraped_test.langstr.ignore {
            "Ok::<(), String>(())"
        } else {
            "self::main()"
        },
    )
    .unwrap();
    test_id
}
