use std::env;
use std::io::IsTerminal;
use std::sync::Arc;

use compiletest::common::Mode;
use compiletest::{log_config, parse_config, run_tests};

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

    if !config.has_html_tidy && config.mode == Mode::Rustdoc {
        eprintln!("warning: `tidy` (html-tidy.org) is not installed; diffs will not be generated");
    }

    if !config.profiler_runtime && config.mode == Mode::CoverageRun {
        let actioned = if config.bless { "blessed" } else { "checked" };
        eprintln!(
            r#"
WARNING: profiler runtime is not available, so `.coverage` files won't be {actioned}
help: try setting `profiler = true` in the `[build]` section of `bootstrap.toml`"#
        );
    }

    log_config(&config);
    run_tests(config);
}
