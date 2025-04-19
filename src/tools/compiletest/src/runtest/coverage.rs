//! Code specific to the coverage test suites.

use std::ffi::OsStr;
use std::process::Command;

use camino::{Utf8Path, Utf8PathBuf};
use glob::glob;

use crate::common::{UI_COVERAGE, UI_COVERAGE_MAP};
use crate::runtest::{Emit, ProcRes, TestCx, WillExecute};
use crate::util::static_regex;

impl<'test> TestCx<'test> {
    fn coverage_dump_path(&self) -> &Utf8Path {
        self.config
            .coverage_dump_path
            .as_deref()
            .unwrap_or_else(|| self.fatal("missing --coverage-dump"))
    }

    pub(super) fn run_coverage_map_test(&self) {
        let coverage_dump_path = self.coverage_dump_path();

        let (proc_res, llvm_ir_path) = self.compile_test_and_save_ir();
        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }
        drop(proc_res);

        let mut dump_command = Command::new(coverage_dump_path);
        dump_command.arg(llvm_ir_path);
        let proc_res = self.run_command_to_procres(&mut dump_command);
        if !proc_res.status.success() {
            self.fatal_proc_rec("coverage-dump failed!", &proc_res);
        }

        let kind = UI_COVERAGE_MAP;

        let expected_coverage_dump = self.load_expected_output(kind);
        let actual_coverage_dump = self.normalize_output(&proc_res.stdout, &[]);

        let coverage_dump_compare_outcome = self.compare_output(
            kind,
            &actual_coverage_dump,
            &proc_res.stdout,
            &expected_coverage_dump,
        );

        if coverage_dump_compare_outcome.should_error() {
            self.fatal_proc_rec(
                &format!("an error occurred comparing coverage output."),
                &proc_res,
            );
        }
    }

    pub(super) fn run_coverage_run_test(&self) {
        let should_run = self.run_if_enabled();
        let proc_res = self.compile_test(should_run, Emit::None);

        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }
        drop(proc_res);

        if let WillExecute::Disabled = should_run {
            return;
        }

        let profraw_path = self.output_base_dir().join("default.profraw");
        let profdata_path = self.output_base_dir().join("default.profdata");

        // Delete any existing profraw/profdata files to rule out unintended
        // interference between repeated test runs.
        if profraw_path.exists() {
            std::fs::remove_file(&profraw_path).unwrap();
        }
        if profdata_path.exists() {
            std::fs::remove_file(&profdata_path).unwrap();
        }

        let proc_res =
            self.exec_compiled_test_general(&[("LLVM_PROFILE_FILE", profraw_path.as_str())], false);
        if self.props.failure_status.is_some() {
            self.check_correct_failure_status(&proc_res);
        } else if !proc_res.status.success() {
            self.fatal_proc_rec("test run failed!", &proc_res);
        }
        drop(proc_res);

        let mut profraw_paths = vec![profraw_path];
        let mut bin_paths = vec![self.make_exe_name()];

        if self.config.suite == "coverage-run-rustdoc" {
            self.run_doctests_for_coverage(&mut profraw_paths, &mut bin_paths);
        }

        // Run `llvm-profdata merge` to index the raw coverage output.
        let proc_res = self.run_llvm_tool("llvm-profdata", |cmd| {
            cmd.args(["merge", "--sparse", "--output"]);
            cmd.arg(&profdata_path);
            cmd.args(&profraw_paths);
        });
        if !proc_res.status.success() {
            self.fatal_proc_rec("llvm-profdata merge failed!", &proc_res);
        }
        drop(proc_res);

        // Run `llvm-cov show` to produce a coverage report in text format.
        let proc_res = self.run_llvm_tool("llvm-cov", |cmd| {
            cmd.args(["show", "--format=text", "--show-line-counts-or-regions"]);

            // Specify the demangler binary and its arguments.
            let coverage_dump_path = self.coverage_dump_path();
            cmd.arg("--Xdemangler").arg(coverage_dump_path);
            cmd.arg("--Xdemangler").arg("--demangle");

            cmd.arg("--instr-profile");
            cmd.arg(&profdata_path);

            for bin in &bin_paths {
                cmd.arg("--object");
                cmd.arg(bin);
            }

            cmd.args(&self.props.llvm_cov_flags);
        });
        if !proc_res.status.success() {
            self.fatal_proc_rec("llvm-cov show failed!", &proc_res);
        }

        let kind = UI_COVERAGE;

        let expected_coverage = self.load_expected_output(kind);
        let normalized_actual_coverage =
            self.normalize_coverage_output(&proc_res.stdout).unwrap_or_else(|err| {
                self.fatal_proc_rec(&err, &proc_res);
            });

        let coverage_dump_compare_outcome = self.compare_output(
            kind,
            &normalized_actual_coverage,
            &proc_res.stdout,
            &expected_coverage,
        );

        if coverage_dump_compare_outcome.should_error() {
            self.fatal_proc_rec(
                &format!("an error occurred comparing coverage output."),
                &proc_res,
            );
        }
    }

    /// Run any doctests embedded in this test file, and add any resulting
    /// `.profraw` files and doctest executables to the given vectors.
    fn run_doctests_for_coverage(
        &self,
        profraw_paths: &mut Vec<Utf8PathBuf>,
        bin_paths: &mut Vec<Utf8PathBuf>,
    ) {
        // Put .profraw files and doctest executables in dedicated directories,
        // to make it easier to glob them all later.
        let profraws_dir = self.output_base_dir().join("doc_profraws");
        let bins_dir = self.output_base_dir().join("doc_bins");

        // Remove existing directories to prevent cross-run interference.
        if profraws_dir.try_exists().unwrap() {
            std::fs::remove_dir_all(&profraws_dir).unwrap();
        }
        if bins_dir.try_exists().unwrap() {
            std::fs::remove_dir_all(&bins_dir).unwrap();
        }

        let mut rustdoc_cmd =
            Command::new(self.config.rustdoc_path.as_ref().expect("--rustdoc-path not passed"));

        // In general there will be multiple doctest binaries running, so we
        // tell the profiler runtime to write their coverage data into separate
        // profraw files.
        rustdoc_cmd.env("LLVM_PROFILE_FILE", profraws_dir.join("%p-%m.profraw"));

        rustdoc_cmd.args(["--test", "-Cinstrument-coverage"]);

        // Without this, the doctests complain about not being able to find
        // their enclosing file's crate for some reason.
        rustdoc_cmd.args(["--crate-name", "workaround_for_79771"]);

        // Persist the doctest binaries so that `llvm-cov show` can read their
        // embedded coverage mappings later.
        rustdoc_cmd.arg("-Zunstable-options");
        rustdoc_cmd.arg("--persist-doctests");
        rustdoc_cmd.arg(&bins_dir);

        rustdoc_cmd.arg("-L");
        rustdoc_cmd.arg(self.aux_output_dir_name());

        rustdoc_cmd.arg(&self.testpaths.file);

        let proc_res = self.compose_and_run_compiler(rustdoc_cmd, None, self.testpaths);
        if !proc_res.status.success() {
            self.fatal_proc_rec("rustdoc --test failed!", &proc_res)
        }

        fn glob_iter(path: impl AsRef<Utf8Path>) -> impl Iterator<Item = Utf8PathBuf> {
            let iter = glob(path.as_ref().as_str()).unwrap();
            iter.map(Result::unwrap).map(Utf8PathBuf::try_from).map(Result::unwrap)
        }

        // Find all profraw files in the profraw directory.
        for p in glob_iter(profraws_dir.join("*.profraw")) {
            profraw_paths.push(p);
        }
        // Find all executables in the `--persist-doctests` directory, while
        // avoiding other file types (e.g. `.pdb` on Windows). This doesn't
        // need to be perfect, as long as it can handle the files actually
        // produced by `rustdoc --test`.
        for p in glob_iter(bins_dir.join("**/*")) {
            let is_bin = p.is_file()
                && match p.extension() {
                    None => true,
                    Some(ext) => ext == OsStr::new("exe"),
                };
            if is_bin {
                bin_paths.push(p);
            }
        }
    }

    fn run_llvm_tool(&self, name: &str, configure_cmd_fn: impl FnOnce(&mut Command)) -> ProcRes {
        let tool_path = self
            .config
            .llvm_bin_dir
            .as_ref()
            .expect("this test expects the LLVM bin dir to be available")
            .join(name);

        let mut cmd = Command::new(tool_path);
        configure_cmd_fn(&mut cmd);

        self.run_command_to_procres(&mut cmd)
    }

    fn normalize_coverage_output(&self, coverage: &str) -> Result<String, String> {
        let normalized = self.normalize_output(coverage, &[]);
        let normalized = Self::anonymize_coverage_line_numbers(&normalized);

        let mut lines = normalized.lines().collect::<Vec<_>>();

        Self::sort_coverage_file_sections(&mut lines)?;
        Self::sort_coverage_subviews(&mut lines)?;

        let joined_lines = lines.iter().flat_map(|line| [line, "\n"]).collect::<String>();
        Ok(joined_lines)
    }

    /// Replace line numbers in coverage reports with the placeholder `LL`,
    /// so that the tests are less sensitive to lines being added/removed.
    fn anonymize_coverage_line_numbers(coverage: &str) -> String {
        // The coverage reporter prints line numbers at the start of a line.
        // They are truncated or left-padded to occupy exactly 5 columns.
        // (`LineNumberColumnWidth` in `SourceCoverageViewText.cpp`.)
        // A pipe character `|` appears immediately after the final digit.
        //
        // Line numbers that appear inside expansion/instantiation subviews
        // have an additional prefix of `  |` for each nesting level.
        //
        // Branch views also include the relevant line number, so we want to
        // redact those too. (These line numbers don't have padding.)
        //
        // Note: The pattern `(?m:^)` matches the start of a line.

        // `    1|` => `   LL|`
        // `   10|` => `   LL|`
        // `  100|` => `   LL|`
        // `  | 1000|`    => `  |   LL|`
        // `  |  | 1000|` => `  |  |   LL|`
        let coverage = static_regex!(r"(?m:^)(?<prefix>(?:  \|)*) *[0-9]+\|")
            .replace_all(&coverage, "${prefix}   LL|");

        // `  |  Branch (1:`     => `  |  Branch (LL:`
        // `  |  |  Branch (10:` => `  |  |  Branch (LL:`
        let coverage = static_regex!(r"(?m:^)(?<prefix>(?:  \|)+  Branch \()[0-9]+:")
            .replace_all(&coverage, "${prefix}LL:");

        // `  |---> MC/DC Decision Region (1:30) to (2:`     => `  |---> MC/DC Decision Region (LL:30) to (LL:`
        let coverage =
            static_regex!(r"(?m:^)(?<prefix>(?:  \|)+---> MC/DC Decision Region \()[0-9]+:(?<middle>[0-9]+\) to \()[0-9]+:")
            .replace_all(&coverage, "${prefix}LL:${middle}LL:");

        // `  |     Condition C1 --> (1:`     => `  |     Condition C1 --> (LL:`
        let coverage =
            static_regex!(r"(?m:^)(?<prefix>(?:  \|)+     Condition C[0-9]+ --> \()[0-9]+:")
                .replace_all(&coverage, "${prefix}LL:");

        coverage.into_owned()
    }

    /// Coverage reports can describe multiple source files, separated by
    /// blank lines. The order of these files is unpredictable (since it
    /// depends on implementation details), so we need to sort the file
    /// sections into a consistent order before comparing against a snapshot.
    fn sort_coverage_file_sections(coverage_lines: &mut Vec<&str>) -> Result<(), String> {
        // Group the lines into file sections, separated by blank lines.
        let mut sections = coverage_lines.split(|line| line.is_empty()).collect::<Vec<_>>();

        // The last section should be empty, representing an extra trailing blank line.
        if !sections.last().is_some_and(|last| last.is_empty()) {
            return Err("coverage report should end with an extra blank line".to_owned());
        }

        // Sort the file sections (not including the final empty "section").
        let except_last = sections.len() - 1;
        (&mut sections[..except_last]).sort();

        // Join the file sections back into a flat list of lines, with
        // sections separated by blank lines.
        let joined = sections.join(&[""] as &[_]);
        assert_eq!(joined.len(), coverage_lines.len());
        *coverage_lines = joined;

        Ok(())
    }

    fn sort_coverage_subviews(coverage_lines: &mut Vec<&str>) -> Result<(), String> {
        let mut output_lines = Vec::new();

        // We accumulate a list of zero or more "subviews", where each
        // subview is a list of one or more lines.
        let mut subviews: Vec<Vec<&str>> = Vec::new();

        fn flush<'a>(subviews: &mut Vec<Vec<&'a str>>, output_lines: &mut Vec<&'a str>) {
            if subviews.is_empty() {
                return;
            }

            // Take and clear the list of accumulated subviews.
            let mut subviews = std::mem::take(subviews);

            // The last "subview" should be just a boundary line on its own,
            // so exclude it when sorting the other subviews.
            let except_last = subviews.len() - 1;
            (&mut subviews[..except_last]).sort();

            for view in subviews {
                for line in view {
                    output_lines.push(line);
                }
            }
        }

        for (line, line_num) in coverage_lines.iter().zip(1..) {
            if line.starts_with("  ------------------") {
                // This is a subview boundary line, so start a new subview.
                subviews.push(vec![line]);
            } else if line.starts_with("  |") {
                // Add this line to the current subview.
                subviews
                    .last_mut()
                    .ok_or(format!(
                        "unexpected subview line outside of a subview on line {line_num}"
                    ))?
                    .push(line);
            } else {
                // This line is not part of a subview, so sort and print any
                // accumulated subviews, and then print the line as-is.
                flush(&mut subviews, &mut output_lines);
                output_lines.push(line);
            }
        }

        flush(&mut subviews, &mut output_lines);
        assert!(subviews.is_empty());

        assert_eq!(output_lines.len(), coverage_lines.len());
        *coverage_lines = output_lines;

        Ok(())
    }
}
