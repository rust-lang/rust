//! Doctest functionality used only for doctests in `.md` Markdown files.

use std::fs::read_to_string;
use std::sync::{Arc, Mutex};

use rustc_session::config::Input;
use rustc_span::{DUMMY_SP, FileName};
use tempfile::tempdir;

use super::{
    CreateRunnableDocTests, DocTestVisitor, GlobalTestOptions, ScrapedDocTest, generate_args_file,
};
use crate::config::Options;
use crate::html::markdown::{ErrorCodes, LangString, MdRelLine, find_testable_code};

struct MdCollector {
    tests: Vec<ScrapedDocTest>,
    cur_path: Vec<String>,
    filename: FileName,
}

impl DocTestVisitor for MdCollector {
    fn visit_test(&mut self, test: String, config: LangString, rel_line: MdRelLine) {
        let filename = self.filename.clone();
        // First line of Markdown is line 1.
        let line = 1 + rel_line.offset();
        self.tests.push(ScrapedDocTest::new(
            filename,
            line,
            self.cur_path.clone(),
            config,
            test,
            DUMMY_SP,
            Vec::new(),
        ));
    }

    fn visit_header(&mut self, name: &str, level: u32) {
        // We use these headings as test names, so it's good if
        // they're valid identifiers.
        let name = name
            .chars()
            .enumerate()
            .map(|(i, c)| {
                if (i == 0 && rustc_lexer::is_id_start(c))
                    || (i != 0 && rustc_lexer::is_id_continue(c))
                {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();

        // Here we try to efficiently assemble the header titles into the
        // test name in the form of `h1::h2::h3::h4::h5::h6`.
        //
        // Suppose that originally `self.cur_path` contains `[h1, h2, h3]`...
        let level = level as usize;
        if level <= self.cur_path.len() {
            // ... Consider `level == 2`. All headers in the lower levels
            // are irrelevant in this new level. So we should reset
            // `self.names` to contain headers until <h2>, and replace that
            // slot with the new name: `[h1, name]`.
            self.cur_path.truncate(level);
            self.cur_path[level - 1] = name;
        } else {
            // ... On the other hand, consider `level == 5`. This means we
            // need to extend `self.names` to contain five headers. We fill
            // in the missing level (<h4>) with `_`. Thus `self.names` will
            // become `[h1, h2, h3, "_", name]`.
            if level - 1 > self.cur_path.len() {
                self.cur_path.resize(level - 1, "_".to_owned());
            }
            self.cur_path.push(name);
        }
    }
}

/// Runs any tests/code examples in the markdown file `options.input`.
pub(crate) fn test(input: &Input, options: Options) -> Result<(), String> {
    let input_str = match input {
        Input::File(path) => {
            read_to_string(path).map_err(|err| format!("{}: {err}", path.display()))?
        }
        Input::Str { name: _, input } => input.clone(),
    };

    // Obviously not a real crate name, but close enough for purposes of doctests.
    let crate_name = input.filestem().to_string();
    let temp_dir =
        tempdir().map_err(|error| format!("failed to create temporary directory: {error:?}"))?;
    let args_file = temp_dir.path().join("rustdoc-cfgs");
    generate_args_file(&args_file, &options)?;

    let opts = GlobalTestOptions {
        crate_name,
        no_crate_inject: true,
        insert_indent_space: false,
        args_file,
    };

    let mut md_collector = MdCollector {
        tests: vec![],
        cur_path: vec![],
        filename: input
            .opt_path()
            .map(ToOwned::to_owned)
            .map(FileName::from)
            .unwrap_or(FileName::Custom("input".to_owned())),
    };
    let codes = ErrorCodes::from(options.unstable_features.is_nightly_build());

    find_testable_code(&input_str, &mut md_collector, codes, None);

    let mut collector = CreateRunnableDocTests::new(options.clone(), opts);
    md_collector.tests.into_iter().for_each(|t| collector.add_test(t, None));
    let CreateRunnableDocTests { opts, rustdoc_options, standalone_tests, mergeable_tests, .. } =
        collector;
    crate::doctest::run_tests(
        opts,
        &rustdoc_options,
        &Arc::new(Mutex::new(Vec::new())),
        standalone_tests,
        mergeable_tests,
        None,
    );
    Ok(())
}
