//! Doctest functionality used only for doctests in `.md` Markdown files.

use std::fs::read_to_string;

use rustc_span::DUMMY_SP;
use tempfile::tempdir;

use super::{generate_args_file, Collector, GlobalTestOptions};
use crate::config::Options;
use crate::html::markdown::{find_testable_code, ErrorCodes};

/// Runs any tests/code examples in the markdown file `input`.
pub(crate) fn test(options: Options) -> Result<(), String> {
    use rustc_session::config::Input;
    let input_str = match &options.input {
        Input::File(path) => {
            read_to_string(&path).map_err(|err| format!("{}: {err}", path.display()))?
        }
        Input::Str { name: _, input } => input.clone(),
    };

    let mut opts = GlobalTestOptions::default();
    opts.no_crate_inject = true;

    let temp_dir =
        tempdir().map_err(|error| format!("failed to create temporary directory: {error:?}"))?;
    let file_path = temp_dir.path().join("rustdoc-cfgs");
    generate_args_file(&file_path, &options)?;

    let mut collector = Collector::new(
        options.input.filestem().to_string(),
        options.clone(),
        true,
        opts,
        None,
        options.input.opt_path().map(ToOwned::to_owned),
        options.enable_per_target_ignores,
        file_path,
    );
    collector.set_position(DUMMY_SP);
    let codes = ErrorCodes::from(options.unstable_features.is_nightly_build());

    find_testable_code(&input_str, &mut collector, codes, options.enable_per_target_ignores, None);

    crate::doctest::run_tests(options.test_args, options.nocapture, collector.tests);
    Ok(())
}
