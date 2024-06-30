use std::{env, io::IsTerminal, sync::Arc};

use compiletest::{
    common::{Config, Mode},
    log_config, parse_config, run_tests,
};

// Warn if filters point to a non .rs file
fn warn_invalid_files(config: &Config) {
    for filter in &config.filters {
        let path = config.src_base.join(filter);
        if path.is_file() && !path.extension().is_some_and(|ext| ext == "rs") {
            eprintln!();
            eprintln!(
                "WARNING: Running {} test on an invalid source file: {}",
                config.mode,
                path.display(),
            );
        }
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    // colored checks stdout by default, but for some reason only stderr is a terminal.
    // compiletest *does* print many things to stdout, but it doesn't really matter.
    if std::io::stderr().is_terminal()
        && matches!(std::env::var("NO_COLOR").as_deref(), Err(_) | Ok("0"))
    {
        colored::control::set_override(true);
    }

    let config = Arc::new(parse_config(env::args().collect()));

    if config.valgrind_path.is_none() && config.force_valgrind {
        panic!("Can't find Valgrind to run Valgrind tests");
    }

    if !config.has_tidy && config.mode == Mode::Rustdoc {
        eprintln!("warning: `tidy` is not installed; diffs will not be generated");
    }

    if !config.profiler_support && config.mode == Mode::CoverageRun {
        let actioned = if config.bless { "blessed" } else { "checked" };
        eprintln!(
            r#"
WARNING: profiler runtime is not available, so `.coverage` files won't be {actioned}
help: try setting `profiler = true` in the `[build]` section of `config.toml`"#
        );
    }

    warn_invalid_files(&config);
    log_config(&config);
    run_tests(config);
}
